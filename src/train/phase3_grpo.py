"""
Phase 3: GRPO Training for TRM

Group Relative Policy Optimization for improving TRM reasoning.

After Phase 2, the TRM can compress reasoning into latent space.
Phase 3 uses GRPO to improve beyond mimicking:
- Freeze: LLM weights completely frozen
- Train: TRM only (learns to produce better hidden state enhancements)
- Reward: Answer correctness on reasoning tasks

GRPO Algorithm (from DeepSeek):
1. For each prompt, generate G responses with TRM-enhanced reasoning
2. Score each response (correct=1, incorrect=0)
3. Compute advantage as normalized reward within group
4. Update TRM to increase probability of high-reward responses

Reference: https://huggingface.co/docs/trl/main/en/grpo_trainer

Usage:
    # With compressor (default):
    python -m src.train.phase3_grpo \\
        --trm_checkpoint ./checkpoints/phase2/best.pt \\
        --dataset gsm8k

    # Without compressor (direct TRM):
    python -m src.train.phase3_grpo \\
        --trm_checkpoint ./checkpoints/phase2_direct/best.pt \\
        --no_compressor \\
        --dataset aime
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.aime import (
    extract_answer as extract_answer_aime,
    format_aime_prompt,
    is_correct as is_correct_aime,
    load_aime,
)
from src.evaluation.gsm8k import (
    extract_answer as extract_answer_gsm8k,
    format_gsm8k_prompt,
    is_correct as is_correct_gsm8k,
    load_gsm8k,
)
from src.models.compression import DimensionCompressor
from src.train.phase1_compressor import Phase1Config
from src.train.phase2_trm import Phase2Config, SequenceTRM

# Optional wandb import
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore[assignment]


@dataclass
class Phase3Config:
    """Configuration for Phase 3 GRPO training"""

    # Model
    model_name: str = "HuggingFaceTB/SmolLM3-3B"
    trm_checkpoint: str = "./checkpoints/phase2/best.pt"
    compressor_checkpoint: str = "anonx3247/llm-trm-compressor"
    use_compressor: bool = True  # False = direct TRM on full hidden dim

    # GRPO parameters
    group_size: int = 4  # Number of responses per prompt (G)
    clip_epsilon: float = 0.2  # PPO-style clipping
    kl_coef: float = 0.01  # KL penalty coefficient
    reward_baseline: float = 0.0  # Baseline for advantage computation

    # Training
    batch_size: int = 2  # Prompts per batch (total samples = batch_size * group_size)
    learning_rate: float = 1e-5  # Lower LR for RL
    weight_decay: float = 0.01
    num_epochs: int = 10
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1

    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.7  # Higher temp for diverse samples
    top_p: float = 0.9

    # Data
    dataset: str = "gsm8k"
    num_samples: int | None = None  # None = use all

    # Output
    output_dir: str = "./checkpoints/phase3"
    log_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 50

    # Wandb
    use_wandb: bool = True
    wandb_project: str = "llm-trm-phase3"
    wandb_run_name: str | None = None

    # Reproducibility
    seed: int = 42


class TRMForGRPO(nn.Module):
    """
    TRM wrapper for GRPO training.

    Handles generation with TRM-enhanced hidden states and computes
    log probabilities for GRPO loss.
    """

    def __init__(
        self,
        model_name: str,
        trm: SequenceTRM,
        compressor: DimensionCompressor | None = None,
        device: torch.device | str = "cuda",
    ):
        super().__init__()
        self.device = device

        # Load base model (frozen)
        print(f"Loading base model: {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            self.model = self.model.to(device)
        self.model.eval()

        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size

        # TRM (trainable)
        self.trm = trm
        self.trm.train()

        # Compressor (frozen, optional)
        self.compressor = compressor
        if compressor is not None:
            for param in compressor.parameters():
                param.requires_grad = False

    def get_trm_enhanced_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get TRM-enhanced hidden state for the prompt."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states[-1]  # [B, L, H]

        if self.compressor is not None:
            # Compressed mode
            compressed = self.compressor(hidden_states.float())
            trm_output = self.trm(compressed, n_steps=8)
            decompressed = self.compressor.decompress(trm_output)
            reasoning_hidden = decompressed[:, -1:, :]
        else:
            # Direct mode
            trm_output = self.trm(hidden_states.float(), n_steps=8)
            reasoning_hidden = trm_output[:, -1:, :]

        return reasoning_hidden.to(hidden_states.dtype)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> tuple[str, list[int]]:
        """
        Generate response with TRM enhancement.

        Returns:
            response: Generated text
            token_ids: List of generated token IDs
        """
        # Format prompt with /no_think mode
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(formatted, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Get TRM-enhanced hidden state
        reasoning_hidden = self.get_trm_enhanced_hidden(input_ids, attention_mask)

        # Get first token logits
        logits = self.model.lm_head(reasoning_hidden)[:, -1, :]

        # Generate tokens
        generated_ids = []
        all_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            # Sample
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum - sorted_probs > top_p
                    sorted_probs[mask] = 0
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    next_token = sorted_indices[0, torch.multinomial(sorted_probs[0], 1)]
                else:
                    next_token = torch.multinomial(probs, num_samples=1)[0]
            else:
                next_token = logits.argmax(dim=-1)

            next_token = next_token.view(1, 1)
            generated_ids.append(next_token.item())
            all_ids = torch.cat([all_ids, next_token], dim=1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # Get next logits
            outputs = self.model(input_ids=all_ids, return_dict=True)
            logits = outputs.logits[:, -1, :]

        # Decode
        response = self.tokenizer.decode(all_ids[0], skip_special_tokens=False)
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0]

        return response.strip(), generated_ids

    def compute_log_probs(
        self,
        prompt: str,
        response_token_ids: list[int],
    ) -> torch.Tensor:
        """
        Compute log probabilities of the response tokens.

        This is differentiable through the TRM.
        """
        # Format prompt
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(formatted, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Get TRM-enhanced hidden state (differentiable)
        reasoning_hidden = self.get_trm_enhanced_hidden(input_ids, attention_mask)

        # First token log prob
        logits = self.model.lm_head(reasoning_hidden)[:, -1, :]
        log_probs_first = F.log_softmax(logits, dim=-1)

        if len(response_token_ids) == 0:
            return torch.tensor(0.0, device=self.device)

        total_log_prob = log_probs_first[0, response_token_ids[0]]

        # Remaining tokens (standard forward, no TRM)
        if len(response_token_ids) > 1:
            response_tensor = torch.tensor(
                response_token_ids[:-1], device=self.device
            ).unsqueeze(0)
            all_ids = torch.cat([input_ids, response_tensor], dim=1)

            with torch.no_grad():
                outputs = self.model(input_ids=all_ids, return_dict=True)

            # Get log probs for each position
            for i, token_id in enumerate(response_token_ids[1:]):
                pos = input_ids.shape[1] + i
                logits_i = outputs.logits[0, pos, :]
                log_probs_i = F.log_softmax(logits_i, dim=-1)
                total_log_prob = total_log_prob + log_probs_i[token_id]

        return total_log_prob


class GRPOTrainer:
    """
    Phase 3 trainer: GRPO for TRM.

    GRPO (Group Relative Policy Optimization):
    - Sample G responses per prompt
    - Score each response (correct/incorrect)
    - Compute advantage = (reward - mean_reward) / std_reward within group
    - Update TRM to increase probability of high-advantage responses
    """

    def __init__(self, config: Phase3Config):
        self.config = config
        self.device = self._get_device()
        self.best_accuracy = 0.0

        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Load components
        self._load_trm()
        self._load_compressor()
        self._init_model()
        self._init_optimizer()
        self._load_data()
        self._init_wandb()

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_trm(self) -> None:
        """Load TRM from checkpoint."""
        print(f"Loading TRM from {self.config.trm_checkpoint}...")

        # Fix pickle compatibility
        sys.modules["__main__"].Phase2Config = Phase2Config  # type: ignore[attr-defined]
        checkpoint = torch.load(
            self.config.trm_checkpoint,
            map_location=self.device,
            weights_only=False,
        )

        # Get config
        ckpt_config = checkpoint.get("config")
        if ckpt_config:
            d_compressed = ckpt_config.d_compressed
            n_layers = getattr(ckpt_config, "n_layers", 2)
            n_heads = getattr(ckpt_config, "n_heads", 8)
            n_latent = getattr(ckpt_config, "n_latent_steps", 6)
            n_deep = getattr(ckpt_config, "n_deep_recursions", 3)
        else:
            d_compressed = 2048 if not self.config.use_compressor else 256
            n_layers, n_heads = 2, 8
            n_latent, n_deep = 6, 3

        self.d_compressed = d_compressed

        # Initialize TRM
        self.trm = SequenceTRM(
            d_compressed=d_compressed,
            n_layers=n_layers,
            n_heads=n_heads,
            n_latent_steps=n_latent,
            n_deep_recursions=n_deep,
        )

        # Load weights
        state_dict = (
            checkpoint.get("trm_state_dict")
            or checkpoint.get("trm")
            or checkpoint.get("ema_trm")
        )
        if state_dict:
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            self.trm.load_state_dict(state_dict)

        self.trm.to(self.device)
        print(f"TRM loaded: d={d_compressed}, layers={n_layers}")

    def _load_compressor(self) -> None:
        """Load compressor if using compressed mode."""
        if not self.config.use_compressor:
            self.compressor = None
            print("Direct mode: no compressor")
            return

        from huggingface_hub import hf_hub_download

        ckpt_source = self.config.compressor_checkpoint
        if "/" in ckpt_source and not os.path.exists(ckpt_source):
            ckpt_path = hf_hub_download(repo_id=ckpt_source, filename="compressor.pt")
        else:
            ckpt_path = ckpt_source

        sys.modules["__main__"].Phase1Config = Phase1Config  # type: ignore[attr-defined]
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        ckpt_config = checkpoint.get("config")
        hidden_size = ckpt_config.hidden_size if ckpt_config else 2048
        d_compressed = ckpt_config.d_compressed if ckpt_config else 256

        self.compressor = DimensionCompressor(
            d_model=hidden_size,
            d_compressed=d_compressed,
        )

        state_dict = checkpoint["compressor"]
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        self.compressor.load_state_dict(state_dict)
        self.compressor.to(self.device)
        self.compressor.eval()
        print(f"Compressor loaded: {hidden_size} -> {d_compressed}")

    def _init_model(self) -> None:
        """Initialize the TRM+LLM wrapper."""
        self.model = TRMForGRPO(
            model_name=self.config.model_name,
            trm=self.trm,
            compressor=self.compressor,
            device=self.device,
        )

    def _init_optimizer(self) -> None:
        """Initialize optimizer for TRM only."""
        # Only optimize TRM parameters
        self.optimizer = AdamW(
            self.trm.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _load_data(self) -> None:
        """Load dataset."""
        print(f"Loading {self.config.dataset} dataset...")

        if self.config.dataset == "gsm8k":
            self.samples = load_gsm8k(split="train", num_samples=self.config.num_samples)
            self.format_prompt = format_gsm8k_prompt
            self.extract_answer = extract_answer_gsm8k
            self.is_correct = is_correct_gsm8k
        elif self.config.dataset == "aime":
            self.samples = load_aime(num_samples=self.config.num_samples)
            self.format_prompt = format_aime_prompt
            self.extract_answer = extract_answer_aime
            self.is_correct = is_correct_aime
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")

        print(f"Loaded {len(self.samples)} samples")

    def _init_wandb(self) -> None:
        """Initialize wandb logging."""
        if not self.config.use_wandb or not WANDB_AVAILABLE:
            return

        mode = "direct" if not self.config.use_compressor else "compressed"
        run_name = self.config.wandb_run_name or f"phase3_grpo_{mode}"
        wandb.init(
            project=self.config.wandb_project,
            name=run_name,
            config={
                "dataset": self.config.dataset,
                "group_size": self.config.group_size,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "use_compressor": self.config.use_compressor,
                "temperature": self.config.temperature,
            },
        )

    def _compute_reward(self, response: str, gold: str) -> float:
        """Compute reward for a response."""
        predicted = self.extract_answer(response)
        correct = self.is_correct(predicted, gold)
        return 1.0 if correct else 0.0

    def _grpo_step(
        self,
        prompts: list[str],
        golds: list[str],
    ) -> dict[str, float]:
        """
        Single GRPO training step.

        Args:
            prompts: List of formatted prompts
            golds: List of gold answers

        Returns:
            Dict with loss and metrics
        """
        G = self.config.group_size
        B = len(prompts)

        all_responses: list[list[str]] = []
        all_token_ids: list[list[list[int]]] = []
        all_rewards: list[list[float]] = []

        # Generate G responses per prompt
        self.trm.eval()
        for i, (prompt, gold) in enumerate(zip(prompts, golds)):
            responses = []
            token_ids = []
            rewards = []

            for _ in range(G):
                response, ids = self.model.generate(
                    prompt,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
                reward = self._compute_reward(response, gold)

                responses.append(response)
                token_ids.append(ids)
                rewards.append(reward)

            all_responses.append(responses)
            all_token_ids.append(token_ids)
            all_rewards.append(rewards)

        # Compute advantages within each group
        all_advantages: list[list[float]] = []
        for rewards in all_rewards:
            rewards_tensor = torch.tensor(rewards)
            mean_r = rewards_tensor.mean()
            std_r = rewards_tensor.std() + 1e-8
            advantages = ((rewards_tensor - mean_r) / std_r).tolist()
            all_advantages.append(advantages)

        # Compute GRPO loss
        self.trm.train()
        total_loss = torch.tensor(0.0, device=self.device)
        n_samples = 0

        for i, prompt in enumerate(prompts):
            for j in range(G):
                advantage = all_advantages[i][j]
                if abs(advantage) < 1e-6:
                    continue  # Skip zero advantage

                token_ids = all_token_ids[i][j]
                if len(token_ids) == 0:
                    continue

                # Compute log prob (differentiable through TRM)
                log_prob = self.model.compute_log_probs(prompt, token_ids)

                # GRPO loss: -advantage * log_prob
                loss = -advantage * log_prob
                total_loss = total_loss + loss
                n_samples += 1

        if n_samples > 0:
            total_loss = total_loss / n_samples

        # Backward and update
        self.optimizer.zero_grad()
        if total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.trm.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()

        # Compute metrics
        flat_rewards = [r for rewards in all_rewards for r in rewards]
        accuracy = sum(flat_rewards) / len(flat_rewards) if flat_rewards else 0.0

        return {
            "loss": total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
            "accuracy": accuracy,
            "mean_reward": sum(flat_rewards) / len(flat_rewards) if flat_rewards else 0.0,
            "n_correct": sum(1 for r in flat_rewards if r > 0.5),
            "n_total": len(flat_rewards),
        }

    def _save_checkpoint(self, step: int, accuracy: float, is_best: bool = False) -> None:
        """Save checkpoint."""
        checkpoint = {
            "step": step,
            "trm_state_dict": self.trm.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "accuracy": accuracy,
            "config": self.config,
        }

        if is_best:
            path = os.path.join(self.config.output_dir, "best.pt")
            torch.save(checkpoint, path)
            print(f"Saved best checkpoint (accuracy={accuracy:.2%})")

        path = os.path.join(self.config.output_dir, f"checkpoint_step{step}.pt")
        torch.save(checkpoint, path)

    def train(self) -> None:
        """Main GRPO training loop."""
        num_steps = (len(self.samples) // self.config.batch_size) * self.config.num_epochs
        warmup_steps = int(num_steps * self.config.warmup_ratio)

        scheduler = CosineAnnealingLR(
            self.optimizer, T_max=num_steps - warmup_steps
        )

        print("\nStarting Phase 3 GRPO training...")
        print(f"  Dataset: {self.config.dataset}")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Group size: {self.config.group_size}")
        print(f"  Epochs: {self.config.num_epochs}")
        print(f"  Total steps: {num_steps}")

        global_step = 0

        for epoch in range(self.config.num_epochs):
            # Shuffle samples
            import random
            random.shuffle(self.samples)

            epoch_metrics = {"loss": 0.0, "accuracy": 0.0, "n_correct": 0, "n_total": 0}

            pbar = tqdm(
                range(0, len(self.samples), self.config.batch_size),
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
            )

            for batch_start in pbar:
                batch_end = min(batch_start + self.config.batch_size, len(self.samples))
                batch = self.samples[batch_start:batch_end]

                prompts = [self.format_prompt(s["question"]) for s in batch]
                golds = [s["gold"] for s in batch]

                # GRPO step
                metrics = self._grpo_step(prompts, golds)
                global_step += 1

                # Accumulate metrics
                epoch_metrics["loss"] += metrics["loss"]
                epoch_metrics["accuracy"] += metrics["accuracy"]
                epoch_metrics["n_correct"] += metrics["n_correct"]
                epoch_metrics["n_total"] += metrics["n_total"]

                # Update progress bar
                n_batches = (batch_start // self.config.batch_size) + 1
                avg_acc = epoch_metrics["accuracy"] / n_batches
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "acc": f"{avg_acc:.1%}",
                })

                # Logging
                if global_step % self.config.log_steps == 0:
                    if self.config.use_wandb and WANDB_AVAILABLE:
                        wandb.log({
                            "train/loss": metrics["loss"],
                            "train/accuracy": metrics["accuracy"],
                            "train/lr": scheduler.get_last_lr()[0],
                        }, step=global_step)

                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    acc = epoch_metrics["n_correct"] / max(epoch_metrics["n_total"], 1)
                    is_best = acc > self.best_accuracy
                    if is_best:
                        self.best_accuracy = acc
                    self._save_checkpoint(global_step, acc, is_best)

                scheduler.step()

            # End of epoch
            n_batches = len(self.samples) // self.config.batch_size
            avg_loss = epoch_metrics["loss"] / max(n_batches, 1)
            avg_acc = epoch_metrics["n_correct"] / max(epoch_metrics["n_total"], 1)

            print(f"Epoch {epoch + 1} complete. Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.1%}")

            # Save epoch checkpoint
            is_best = avg_acc > self.best_accuracy
            if is_best:
                self.best_accuracy = avg_acc
            self._save_checkpoint(global_step, avg_acc, is_best)

        print("\nTraining complete!")
        print(f"Best accuracy: {self.best_accuracy:.1%}")

        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()


def run_phase3_training(config: Phase3Config | None = None) -> None:
    """Main entry point for Phase 3 GRPO training."""
    config = config or Phase3Config()
    trainer = GRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3: GRPO Training")
    parser.add_argument(
        "--trm_checkpoint",
        type=str,
        default="./checkpoints/phase2/best.pt",
        help="Path to TRM checkpoint from Phase 2",
    )
    parser.add_argument(
        "--compressor_checkpoint",
        type=str,
        default="anonx3247/llm-trm-compressor",
        help="HF Hub repo or local path to compressor checkpoint",
    )
    parser.add_argument(
        "--no_compressor",
        action="store_true",
        default=False,
        help="Direct mode: TRM operates on full hidden dim",
    )
    parser.add_argument("--output_dir", type=str, default="./checkpoints/phase3")
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "aime"],
    )
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    # Wandb
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--no_wandb", action="store_false", dest="use_wandb")
    parser.add_argument("--wandb_project", type=str, default="llm-trm-phase3")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = Phase3Config(
        trm_checkpoint=args.trm_checkpoint,
        compressor_checkpoint=args.compressor_checkpoint,
        use_compressor=not args.no_compressor,
        output_dir=args.output_dir,
        dataset=args.dataset,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        group_size=args.group_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        seed=args.seed,
    )

    run_phase3_training(config)
