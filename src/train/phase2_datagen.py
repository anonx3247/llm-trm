"""
Phase 2 Data Generation: Extract Hidden State Sequences (Optimized)

Generate training data for TRM iteration training by:
1. Running SmolLM3 in thinking mode on reasoning problems (BATCHED)
2. Capturing hidden states DURING generation (no second forward pass):
   - hidden_pre: Full sequence [L, D] before <think> token (context)
   - hidden_post: Single vector [D] after </think> token (target)
3. Saving dataset for variable-length sequence training

Optimizations over original:
- Batched generation (4-8x speedup depending on batch_size)
- Single forward pass (hidden states captured during generation)
- Reduced max_new_tokens default (512 vs 1024)
- Checkpoint saving for resumable generation

Usage:
    python -m src.train.phase2_datagen \\
        --dataset gsm8k \\
        --output_dir ./data/hidden_pairs \\
        --num_samples 1000 \\
        --batch_size 4
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class DataGenConfig:
    """Configuration for hidden state pair generation"""

    # Model
    model_name: str = "HuggingFaceTB/SmolLM3-3B"

    # Data source
    dataset: str = "gsm8k"  # Dataset to use for reasoning problems
    dataset_subset: str = "main"  # GSM8K has "main" and "socratic" subsets
    num_samples: int = 1000
    max_seq_length: int = 2048

    # Output
    output_dir: str = "./data/hidden_pairs"
    output_filename: str = "hidden_pairs.pt"

    # Generation (optimized defaults)
    temperature: float = 0.7
    max_new_tokens: int = 512  # Reduced from 1024 - most thinking fits in 512
    do_sample: bool = True

    # Batching (KEY OPTIMIZATION)
    batch_size: int = 4  # Increase based on GPU memory (4 for 24GB, 8 for 48GB)
    skip_failures: bool = True

    # Checkpointing
    checkpoint_every: int = 100  # Save progress every N samples
    resume_from: str | None = None  # Path to checkpoint to resume from


class ThinkingDataGenerator:
    """
    Generate hidden state sequences from SmolLM3 thinking trajectories.

    Optimized for speed:
    - Batched generation processes multiple problems in parallel
    - Hidden states captured during generation (no second forward pass)
    - Checkpoint saving allows resuming interrupted generation
    """

    def __init__(self, config: DataGenConfig):
        self.config = config
        self.device = self._get_device()

        # Initialize model and tokenizer
        self._init_model()

        # Get special token IDs
        self._init_special_tokens()

    def _get_device(self) -> torch.device:
        """Determine device to use."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _init_model(self) -> None:
        """Load SmolLM3 model and tokenizer."""
        print(f"Loading {self.config.model_name}...")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        # Use left padding for batched generation (decoder-only models)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine dtype based on device
        if self.device.type == "cuda":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        # Load model
        if self.device.type == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=dtype,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=dtype,
            )
            self.model = self.model.to(self.device)  # type: ignore[arg-type]

        self.model.eval()
        self.hidden_size = self.model.config.hidden_size
        print(f"Model loaded. Hidden size: {self.hidden_size}")
        print(f"Batch size: {self.config.batch_size}")

    def _init_special_tokens(self) -> None:
        """Initialize special token IDs for thinking boundaries."""
        self.think_start_token = "<think>"
        self.think_end_token = "</think>"

        think_start_ids = self.tokenizer.encode(self.think_start_token, add_special_tokens=False)
        think_end_ids = self.tokenizer.encode(self.think_end_token, add_special_tokens=False)

        print(f"Think start token '{self.think_start_token}' -> {think_start_ids}")
        print(f"Think end token '{self.think_end_token}' -> {think_end_ids}")

        self.think_start_ids = think_start_ids
        self.think_end_ids = think_end_ids

    def _find_token_sequence(self, input_ids: list[int], pattern: list[int]) -> int | None:
        """Find the starting position of a token sequence in input_ids."""
        pattern_len = len(pattern)
        for i in range(len(input_ids) - pattern_len + 1):
            if input_ids[i : i + pattern_len] == pattern:
                return i
        return None

    def _format_problems(self, problems: list[str]) -> list[str]:
        """Format multiple problems using chat template with thinking enabled."""
        formatted = []
        for problem in problems:
            messages = [{"role": "user", "content": problem}]
            text: str = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            formatted.append(text)
        return formatted

    @torch.no_grad()
    def _generate_batch_with_hidden_states(
        self, problems: list[str]
    ) -> list[tuple[list[int], torch.Tensor, int] | None]:
        """
        Generate responses for a batch of problems and extract hidden states.

        Returns:
            List of (generated_ids, full_hidden_states, input_length) tuples,
            or None for failed samples.
        """
        # Format all problems
        formatted = self._format_problems(problems)

        # Tokenize with padding (left-padded for decoder-only)
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Track input lengths (accounting for left padding)
        input_lengths = attention_mask.sum(dim=1).tolist()

        # Generate WITH hidden states output
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

        # outputs.sequences: [B, total_len]
        # outputs.hidden_states: tuple of (num_gen_steps + 1) elements
        #   - [0]: prefill hidden states (tuple of layers, each [B, input_len, D])
        #   - [i>0]: gen step hidden states (tuple of layers, each [B, 1, D])

        # Type assertion for return_dict_in_generate=True
        assert hasattr(outputs, "sequences"), "Expected GenerateDecoderOnlyOutput"
        assert hasattr(outputs, "hidden_states"), "Expected hidden_states in output"
        sequences = outputs.sequences
        hidden_states = outputs.hidden_states
        assert hidden_states is not None, "hidden_states should not be None"

        results: list[tuple[list[int], torch.Tensor, int] | None] = []
        batch_size = input_ids.shape[0]

        for b in range(batch_size):
            try:
                # Get generated sequence for this sample
                seq = sequences[b].tolist()

                # Reconstruct full hidden states from generation outputs
                # Prefill: last layer hidden states [input_len, D]
                prefill_hidden = hidden_states[0][-1][b]  # [padded_input_len, D]

                # Remove left padding from prefill
                pad_len = input_ids.shape[1] - input_lengths[b]
                prefill_hidden = prefill_hidden[pad_len:]  # [actual_input_len, D]

                # Generated tokens: concat all generation step hidden states
                if len(hidden_states) > 1:
                    gen_hidden_list = []
                    for step_hidden in hidden_states[1:]:
                        # step_hidden is tuple of layers, get last layer
                        gen_hidden_list.append(step_hidden[-1][b])  # [1, D]
                    gen_hidden = torch.cat(gen_hidden_list, dim=0)  # [gen_len, D]

                    # Full sequence hidden states
                    full_hidden = torch.cat([prefill_hidden, gen_hidden], dim=0)
                else:
                    full_hidden = prefill_hidden

                # Remove padding tokens from sequence
                seq = seq[pad_len:]

                results.append((seq, full_hidden, input_lengths[b]))

            except Exception as e:
                print(f"\nError extracting hidden states for sample {b}: {e}")
                results.append(None)

        return results

    def _extract_from_hidden_states(
        self,
        generated_ids: list[int],
        hidden_states: torch.Tensor,
        input_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int] | None:
        """
        Extract hidden_pre and hidden_post from full hidden states.

        Args:
            generated_ids: Full generated sequence (without padding)
            hidden_states: Full hidden states [total_len, D]
            input_length: Length of input (before generation)

        Returns:
            (hidden_pre, hidden_post, seq_length, num_thinking_tokens) or None
        """
        # Find thinking boundaries
        think_start_pos = self._find_token_sequence(generated_ids, self.think_start_ids)
        think_end_pos = self._find_token_sequence(generated_ids, self.think_end_ids)

        if think_start_pos is None or think_end_pos is None:
            return None

        if think_end_pos <= think_start_pos:
            return None

        if think_start_pos < 1:
            return None

        # hidden_pre: everything before <think>
        hidden_pre = hidden_states[:think_start_pos]  # [L, D]
        seq_length = think_start_pos

        # hidden_post: state after </think>
        post_pos = think_end_pos + len(self.think_end_ids) - 1
        if post_pos >= hidden_states.shape[0]:
            post_pos = hidden_states.shape[0] - 1
        hidden_post = hidden_states[post_pos]  # [D]

        # Number of thinking tokens
        num_thinking_tokens = think_end_pos + len(self.think_end_ids) - think_start_pos

        return hidden_pre, hidden_post, seq_length, num_thinking_tokens

    def _load_problems(self) -> list[str]:
        """Load reasoning problems from dataset."""
        print(f"Loading dataset: {self.config.dataset}...")

        # Support comma-separated datasets for mixing
        datasets_to_load = [d.strip() for d in self.config.dataset.split(",")]
        samples_per_dataset = self.config.num_samples // len(datasets_to_load)

        all_problems: list[str] = []

        for dataset_name in datasets_to_load:
            problems = self._load_single_dataset(dataset_name, samples_per_dataset)
            all_problems.extend(problems)
            print(f"  {dataset_name}: {len(problems)} problems")

        print(f"Total loaded: {len(all_problems)} problems")
        return all_problems

    def _load_single_dataset(self, dataset_name: str, num_samples: int) -> list[str]:
        """Load problems from a single dataset."""
        problems: list[str] = []

        if dataset_name == "gsm8k":
            dataset = load_dataset(
                "gsm8k", self.config.dataset_subset, split="train", streaming=True
            )
            for i, item in enumerate(
                tqdm(dataset, total=num_samples, desc=f"Loading {dataset_name}")
            ):
                if i >= num_samples:
                    break
                problems.append(str(item["question"]))

        elif dataset_name == "math":
            dataset = load_dataset("lighteval/MATH", split="train", streaming=True)
            for i, item in enumerate(
                tqdm(dataset, total=num_samples, desc=f"Loading {dataset_name}")
            ):
                if i >= num_samples:
                    break
                problems.append(str(item["problem"]))

        elif dataset_name == "aime":
            # AIME competition math - very challenging
            dataset = load_dataset("qq8933/AIME_1983_2024", split="train", streaming=True)
            for i, item in enumerate(
                tqdm(dataset, total=num_samples, desc=f"Loading {dataset_name}")
            ):
                if i >= num_samples:
                    break
                problems.append(str(item["Question"]))

        elif dataset_name == "humaneval":
            # OpenAI HumanEval coding problems
            dataset = load_dataset("openai/openai_humaneval", split="test", streaming=True)
            for i, item in enumerate(
                tqdm(dataset, total=num_samples, desc=f"Loading {dataset_name}")
            ):
                if i >= num_samples:
                    break
                # Format as a coding task
                prompt = item["prompt"]
                problems.append(f"Complete the following Python function:\n\n{prompt}")

        elif dataset_name == "mbpp":
            # MBPP coding problems
            dataset = load_dataset("mbpp", split="train", streaming=True)
            for i, item in enumerate(
                tqdm(dataset, total=num_samples, desc=f"Loading {dataset_name}")
            ):
                if i >= num_samples:
                    break
                text = item["text"]
                problems.append(f"Write a Python function for the following task:\n\n{text}")

        elif dataset_name == "apps":
            # APPS coding problems (harder)
            dataset = load_dataset(
                "codeparrot/apps", split="train", streaming=True, trust_remote_code=True
            )
            for i, item in enumerate(
                tqdm(dataset, total=num_samples, desc=f"Loading {dataset_name}")
            ):
                if i >= num_samples:
                    break
                question = item["question"]
                problems.append(str(question))

        else:
            # Generic fallback
            dataset = load_dataset(dataset_name, split="train", streaming=True)
            for i, item in enumerate(
                tqdm(dataset, total=num_samples, desc=f"Loading {dataset_name}")
            ):
                if i >= num_samples:
                    break
                if "question" in item:
                    problems.append(str(item["question"]))
                elif "problem" in item:
                    problems.append(str(item["problem"]))
                elif "Problem" in item:
                    problems.append(str(item["Problem"]))
                elif "text" in item:
                    problems.append(str(item["text"]))
                elif "prompt" in item:
                    problems.append(str(item["prompt"]))
                else:
                    problems.append(str(list(item.values())[0]))

        return problems

    def _save_checkpoint(self, data: dict[str, Any], checkpoint_idx: int, output_dir: str) -> None:
        """Save intermediate checkpoint."""
        checkpoint_path = os.path.join(output_dir, f"checkpoint_{checkpoint_idx}.pt")
        torch.save(data, checkpoint_path)
        print(f"\nCheckpoint saved: {checkpoint_path} ({len(data['hidden_pre'])} samples)")

    def _load_checkpoint(self, checkpoint_path: str) -> dict[str, Any]:
        """Load checkpoint to resume generation."""
        print(f"Resuming from checkpoint: {checkpoint_path}")
        data: dict[str, Any] = torch.load(checkpoint_path, weights_only=False)
        return data

    def generate_pairs(self, problems: list[str]) -> dict[str, Any]:
        """
        Generate hidden state sequence pairs using batched generation.

        Args:
            problems: List of reasoning problems

        Returns:
            Dictionary with hidden_pre (list), hidden_post, seq_lengths, etc.
        """
        hidden_pres: list[torch.Tensor] = []
        hidden_posts: list[torch.Tensor] = []
        seq_lengths: list[int] = []
        num_thinking_tokens_list: list[int] = []
        successful_problems: list[str] = []
        failures = 0
        start_idx = 0

        # Resume from checkpoint if specified
        if self.config.resume_from and os.path.exists(self.config.resume_from):
            checkpoint = self._load_checkpoint(self.config.resume_from)
            hidden_pres = checkpoint["hidden_pre"]
            hidden_posts = list(checkpoint["hidden_post"])
            seq_lengths = checkpoint["seq_lengths"]
            num_thinking_tokens_list = checkpoint["num_thinking_tokens"]
            successful_problems = checkpoint["problems"]
            start_idx = len(successful_problems)
            print(f"Resumed with {start_idx} samples, continuing...")

        # Create output directory for checkpoints
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Process in batches
        batch_size = self.config.batch_size
        num_batches = (len(problems) - start_idx + batch_size - 1) // batch_size

        pbar = tqdm(total=len(problems) - start_idx, desc="Generating pairs")

        for batch_idx in range(num_batches):
            batch_start = start_idx + batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(problems))
            batch_problems = problems[batch_start:batch_end]

            if not batch_problems:
                break

            try:
                # Generate batch with hidden states
                batch_results = self._generate_batch_with_hidden_states(batch_problems)

                # Process each result in batch
                for i, result in enumerate(batch_results):
                    if result is None:
                        failures += 1
                        continue

                    generated_ids, sample_hidden, input_len = result

                    # Extract hidden_pre and hidden_post
                    extraction = self._extract_from_hidden_states(
                        generated_ids, sample_hidden, input_len
                    )

                    if extraction is None:
                        failures += 1
                        continue

                    hidden_pre, hidden_post, seq_len, num_thinking = extraction

                    # Store results (CPU, float32)
                    hidden_pres.append(hidden_pre.cpu().float())
                    hidden_posts.append(hidden_post.cpu().float())
                    seq_lengths.append(seq_len)
                    num_thinking_tokens_list.append(num_thinking)
                    successful_problems.append(batch_problems[i])

                pbar.update(len(batch_problems))

            except torch.cuda.OutOfMemoryError:
                print(f"\nOOM at batch {batch_idx}, reducing batch size...")
                # Skip this batch and continue
                failures += len(batch_problems)
                pbar.update(len(batch_problems))
                torch.cuda.empty_cache()
                continue

            except Exception as e:
                if not self.config.skip_failures:
                    raise
                print(f"\nError processing batch {batch_idx}: {e}")
                failures += len(batch_problems)
                pbar.update(len(batch_problems))
                continue

            # Checkpoint saving
            total_processed = batch_end
            if (
                self.config.checkpoint_every > 0
                and total_processed % self.config.checkpoint_every < batch_size
                and len(hidden_pres) > 0
            ):
                checkpoint_data = {
                    "hidden_pre": hidden_pres,
                    "hidden_post": torch.stack(hidden_posts) if hidden_posts else torch.tensor([]),
                    "seq_lengths": seq_lengths,
                    "num_thinking_tokens": num_thinking_tokens_list,
                    "problems": successful_problems,
                    "hidden_size": self.hidden_size,
                }
                self._save_checkpoint(checkpoint_data, total_processed, self.config.output_dir)

        pbar.close()

        print("\nGeneration complete!")
        print(f"  Successful: {len(hidden_pres)}")
        print(f"  Failures: {failures}")
        print(f"  Success rate: {len(hidden_pres) / (len(hidden_pres) + failures) * 100:.1f}%")

        if len(hidden_pres) == 0:
            raise ValueError("No successful samples generated!")

        hidden_post_tensor = torch.stack(hidden_posts)

        return {
            "hidden_pre": hidden_pres,
            "hidden_post": hidden_post_tensor,
            "seq_lengths": seq_lengths,
            "num_thinking_tokens": num_thinking_tokens_list,
            "problems": successful_problems,
            "hidden_size": self.hidden_size,
        }

    def run(self) -> None:
        """Generate and save the dataset."""
        problems = self._load_problems()
        data = self.generate_pairs(problems)

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(self.config.output_dir, self.config.output_filename)
        torch.save(data, output_path)

        print(f"\nDataset saved to: {output_path}")
        print(f"  Number of samples: {len(data['hidden_pre'])}")
        print(f"  hidden_post shape: {data['hidden_post'].shape}")
        print(f"  Avg context length: {sum(data['seq_lengths']) / len(data['seq_lengths']):.1f}")
        print(
            f"  Avg thinking tokens: "
            f"{sum(data['num_thinking_tokens']) / len(data['num_thinking_tokens']):.1f}"
        )

        # Clean up checkpoints
        for f in Path(self.config.output_dir).glob("checkpoint_*.pt"):
            f.unlink()
            print(f"Cleaned up checkpoint: {f}")


def generate_hidden_state_pairs(config: DataGenConfig | None = None) -> None:
    """Main entry point for generating hidden state pairs."""
    config = config or DataGenConfig()
    generator = ThinkingDataGenerator(config)
    generator.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2: Generate Hidden State Pairs (Optimized)")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--dataset_subset", type=str, default="main")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="./data/hidden_pairs")
    parser.add_argument("--output_filename", type=str, default="hidden_pairs.pt")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM3-3B")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generation (increase for more VRAM)",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=100,
        help="Save checkpoint every N samples (0 to disable)",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--skip_failures",
        action="store_true",
        default=True,
        help="Skip samples where thinking tags not found",
    )
    parser.add_argument(
        "--no_skip_failures",
        action="store_false",
        dest="skip_failures",
        help="Raise error when thinking tags not found",
    )
    args = parser.parse_args()

    config = DataGenConfig(
        dataset=args.dataset,
        dataset_subset=args.dataset_subset,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume_from,
        skip_failures=args.skip_failures,
    )

    generate_hidden_state_pairs(config)
