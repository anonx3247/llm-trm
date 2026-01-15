"""
Phase 2 Data Generation: Extract Hidden State Sequences

Generate training data for TRM iteration training by:
1. Running SmolLM3 in thinking mode on reasoning problems
2. Capturing hidden states:
   - hidden_pre: Full sequence [L, D] before <think> token (context)
   - hidden_post: Single vector [D] after </think> token (target)
3. Saving dataset for variable-length sequence training

The TRM learns to take [B, L, D'] and output [B, L+1, D'] where
position L+1 matches what was at position L+M (after M thinking tokens).

Usage:
    python -m src.train.phase2_datagen \\
        --dataset gsm8k \\
        --output_dir ./data/hidden_pairs \\
        --num_samples 1000
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

    # Generation
    temperature: float = 0.7
    max_new_tokens: int = 1024
    do_sample: bool = True

    # Processing
    batch_size: int = 1  # Process one at a time for hidden state extraction
    skip_failures: bool = True  # Skip samples where thinking tags not found


class ThinkingDataGenerator:
    """
    Generate hidden state sequences from SmolLM3 thinking trajectories.

    Process:
    1. Load SmolLM3 with enable_thinking=True
    2. For each problem:
       a. Generate response with thinking
       b. Find <think> and </think> token positions
       c. Re-run forward pass and capture hidden states
       d. Extract hidden_pre (full context) and hidden_post (target)
    3. Save dataset

    Output format:
        {
            "hidden_pre": List[Tensor[L_i, D]],  # Variable-length context sequences
            "hidden_post": Tensor[N, D],          # Target states after thinking
            "seq_lengths": List[int],             # Length of each context sequence
            "num_thinking_tokens": List[int],     # Number of thinking tokens skipped
            "problems": List[str],                # Original problems (for reference)
        }

    Training goal:
        TRM takes [B, L, D'] -> outputs [B, L+1, D']
        Loss: MSE between output[:, -1, :] and hidden_post
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

    def _init_special_tokens(self) -> None:
        """Initialize special token IDs for thinking boundaries."""
        # SmolLM3 uses <think> and </think> tags
        self.think_start_token = "<think>"
        self.think_end_token = "</think>"

        # Tokenize the special tokens to get their IDs
        # Note: These may be multiple tokens depending on tokenizer
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

    def _format_problem(self, problem: str) -> str:
        """Format problem using chat template with thinking enabled."""
        messages = [{"role": "user", "content": problem}]

        # Apply chat template with thinking enabled
        formatted: str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        return formatted

    def _generate_with_thinking(self, problem: str) -> tuple[str, list[int]] | None:
        """
        Generate response with thinking for a single problem.

        Returns:
            Tuple of (generated_text, input_ids) or None if generation fails
        """
        # Format with chat template
        formatted = self._format_problem(problem)

        # Tokenize
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated_ids = outputs[0].tolist()
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

        return generated_text, generated_ids

    @torch.no_grad()
    def _extract_hidden_states(
        self, input_ids: list[int], think_start_pos: int, think_end_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Extract hidden state sequences at thinking boundaries.

        Args:
            input_ids: Full sequence including generation
            think_start_pos: Position of first token of <think>
            think_end_pos: Position of first token of </think>

        Returns:
            hidden_pre: Full context sequence before <think> [L, D]
            hidden_post: Target state after </think> [D]
            seq_length: Length of context sequence (L)
            num_thinking_tokens: Number of tokens in thinking section (M)
        """
        # Convert to tensor
        input_tensor = torch.tensor([input_ids], device=self.device)

        # Forward pass to get all hidden states
        outputs = self.model(
            input_ids=input_tensor,
            output_hidden_states=True,
            return_dict=True,
        )

        # Get last layer hidden states [1, total_len, D]
        hidden_states = outputs.hidden_states[-1]

        # hidden_pre: Full sequence from position 0 to think_start_pos (exclusive)
        # This is the context before thinking begins
        # We include up to (but not including) the <think> token
        context_end = think_start_pos  # Position where <think> starts
        hidden_pre = hidden_states[0, :context_end, :]  # [L, D]
        seq_length = context_end

        # hidden_post: Target state after </think>
        # Position right after the </think> token ends
        post_pos = think_end_pos + len(self.think_end_ids) - 1
        if post_pos >= hidden_states.shape[1]:
            post_pos = hidden_states.shape[1] - 1
        hidden_post = hidden_states[0, post_pos, :]  # [D]

        # Number of thinking tokens (M) = tokens between <think> and </think> (inclusive)
        # This is what the TRM will learn to "skip"
        num_thinking_tokens = think_end_pos + len(self.think_end_ids) - think_start_pos

        return hidden_pre, hidden_post, seq_length, num_thinking_tokens

    def _load_problems(self) -> list[str]:
        """Load reasoning problems from dataset."""
        print(f"Loading dataset: {self.config.dataset}...")

        if self.config.dataset == "gsm8k":
            dataset = load_dataset(
                "gsm8k", self.config.dataset_subset, split="train", streaming=True
            )
            problems = []
            for i, item in enumerate(
                tqdm(dataset, total=self.config.num_samples, desc="Loading problems")
            ):
                if i >= self.config.num_samples:
                    break
                problems.append(item["question"])
        elif self.config.dataset == "math":
            dataset = load_dataset("lighteval/MATH", split="train", streaming=True)
            problems = []
            for i, item in enumerate(
                tqdm(dataset, total=self.config.num_samples, desc="Loading problems")
            ):
                if i >= self.config.num_samples:
                    break
                problems.append(item["problem"])
        else:
            # Generic: try to find "question" or "problem" field
            dataset = load_dataset(self.config.dataset, split="train", streaming=True)
            problems = []
            for i, item in enumerate(
                tqdm(dataset, total=self.config.num_samples, desc="Loading problems")
            ):
                if i >= self.config.num_samples:
                    break
                if "question" in item:
                    problems.append(item["question"])
                elif "problem" in item:
                    problems.append(item["problem"])
                elif "text" in item:
                    problems.append(item["text"])
                else:
                    problems.append(str(list(item.values())[0]))

        print(f"Loaded {len(problems)} problems")
        return problems

    def generate_pairs(self, problems: list[str]) -> dict[str, Any]:
        """
        Generate hidden state sequence pairs for a list of problems.

        Args:
            problems: List of reasoning problems

        Returns:
            Dictionary with hidden_pre (list), hidden_post, seq_lengths, etc.
        """
        hidden_pres: list[torch.Tensor] = []  # List of [L_i, D] tensors
        hidden_posts: list[torch.Tensor] = []  # List of [D] tensors
        seq_lengths: list[int] = []
        num_thinking_tokens_list: list[int] = []
        successful_problems: list[str] = []
        failures = 0

        for i, problem in enumerate(tqdm(problems, desc="Generating pairs")):
            try:
                # Generate response with thinking
                result = self._generate_with_thinking(problem)
                if result is None:
                    failures += 1
                    continue

                generated_text, generated_ids = result

                # Find thinking boundaries
                think_start_pos = self._find_token_sequence(generated_ids, self.think_start_ids)
                think_end_pos = self._find_token_sequence(generated_ids, self.think_end_ids)

                if think_start_pos is None or think_end_pos is None:
                    if not self.config.skip_failures:
                        raise ValueError(
                            f"Could not find thinking boundaries in: {generated_text[:200]}..."
                        )
                    failures += 1
                    continue

                if think_end_pos <= think_start_pos:
                    failures += 1
                    continue

                # Skip if context is too short (need at least 1 token)
                if think_start_pos < 1:
                    failures += 1
                    continue

                # Extract hidden states
                hidden_pre, hidden_post, seq_len, num_thinking = self._extract_hidden_states(
                    generated_ids, think_start_pos, think_end_pos
                )

                # Store results (move to CPU and convert to float32 for storage)
                hidden_pres.append(hidden_pre.cpu().float())
                hidden_posts.append(hidden_post.cpu().float())
                seq_lengths.append(seq_len)
                num_thinking_tokens_list.append(num_thinking)
                successful_problems.append(problem)

            except Exception as e:
                if not self.config.skip_failures:
                    raise
                print(f"\nError processing problem {i}: {e}")
                failures += 1
                continue

            # Progress update every 100 samples
            if (i + 1) % 100 == 0:
                print(
                    f"\nProcessed {i + 1}/{len(problems)}, "
                    f"Successful: {len(hidden_pres)}, Failures: {failures}"
                )

        print("\nGeneration complete!")
        print(f"  Successful: {len(hidden_pres)}")
        print(f"  Failures: {failures}")

        if len(hidden_pres) == 0:
            raise ValueError("No successful samples generated!")

        # Stack hidden_posts since they're all [D]
        hidden_post_tensor = torch.stack(hidden_posts)  # [N, D]

        return {
            "hidden_pre": hidden_pres,  # List of [L_i, D] tensors (variable length)
            "hidden_post": hidden_post_tensor,  # [N, D]
            "seq_lengths": seq_lengths,  # List of L_i values
            "num_thinking_tokens": num_thinking_tokens_list,  # List of M_i values
            "problems": successful_problems,
            "hidden_size": self.hidden_size,
        }

    def run(self) -> None:
        """Generate and save the dataset."""
        # Load problems
        problems = self._load_problems()

        # Generate pairs
        data = self.generate_pairs(problems)

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Save dataset
        output_path = os.path.join(self.config.output_dir, self.config.output_filename)
        torch.save(data, output_path)

        print(f"\nDataset saved to: {output_path}")
        print(f"  Number of samples: {len(data['hidden_pre'])}")
        print(f"  hidden_post shape: {data['hidden_post'].shape}")
        print(f"  Avg context length: {sum(data['seq_lengths']) / len(data['seq_lengths']):.1f}")
        print(
            f"  Avg thinking tokens: {sum(data['num_thinking_tokens']) / len(data['num_thinking_tokens']):.1f}"
        )


def generate_hidden_state_pairs(config: DataGenConfig | None = None) -> None:
    """
    Main entry point for generating hidden state pairs.
    """
    config = config or DataGenConfig()
    generator = ThinkingDataGenerator(config)
    generator.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2: Generate Hidden State Pairs")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--dataset_subset", type=str, default="main")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="./data/hidden_pairs")
    parser.add_argument("--output_filename", type=str, default="hidden_pairs.pt")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM3-3B")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_seq_length", type=int, default=2048)
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
        skip_failures=args.skip_failures,
    )

    generate_hidden_state_pairs(config)
