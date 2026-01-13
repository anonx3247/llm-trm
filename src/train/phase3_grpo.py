"""
Phase 3: GRPO Training (Placeholder)

Group Relative Policy Optimization for improving TRM+Compressor.

After Phase 2, the TRM can replicate CoT reasoning in latent space.
Phase 3 uses GRPO to improve beyond mimicking:
- Freeze: LLM weights completely frozen
- Train: TRM + finetune compressors only
- Reward: Answer correctness on reasoning tasks

When <thinking> is triggered:
1. TRM iterates in latent space (replacing text CoT)
2. Model generates answer conditioned on TRM states
3. GRPO rewards trajectories that produce correct answers

Usage:
    python -m src.train.phase3_grpo \\
        --trm_checkpoint ./checkpoints/phase2/best.pt \\
        --output_dir ./checkpoints/phase3 \\
        --dataset gsm8k

TODO: Implement GRPO training
      Reference: TRL library GRPOTrainer
      https://huggingface.co/docs/trl/main/en/grpo_trainer
"""

import torch
from typing import Optional
from dataclasses import dataclass


@dataclass
class Phase3Config:
    """Configuration for Phase 3 GRPO training"""
    # Model
    model_name: str = "HuggingFaceTB/SmolLM3-3B"
    trm_checkpoint: str = "./checkpoints/phase2/best.pt"
    compressor_checkpoint: str = "./checkpoints/phase1/best.pt"

    # Training
    batch_size: int = 4
    learning_rate: float = 1e-5  # Lower LR for RL fine-tuning
    num_epochs: int = 10

    # GRPO specific
    group_size: int = 8  # Number of samples per group
    kl_coef: float = 0.1  # KL penalty coefficient

    # Data
    dataset: str = "gsm8k"
    num_samples: int = 10000

    # Output
    output_dir: str = "./checkpoints/phase3"


class GRPOTrainer:
    """
    Phase 3 trainer: GRPO for TRM + Compressor.

    GRPO (Group Relative Policy Optimization) from DeepSeek:
    - Sample G responses per prompt
    - Rank by reward (answer correctness)
    - Update policy to increase probability of high-reward responses
    - No need for reference model (unlike DPO)

    Training setup:
    - LLM: Frozen (no gradients)
    - TRM: Trainable
    - Compressor: Light finetuning

    Flow:
    1. For each prompt, generate G responses with TRM-enhanced reasoning
    2. Evaluate correctness of each response
    3. Compute GRPO loss based on relative rankings
    4. Update TRM and compressor weights

    TODO: Implement using TRL's GRPOTrainer as reference
    """

    def __init__(self, config: Phase3Config):
        self.config = config
        # TODO: Load models, setup GRPO

    def _compute_reward(self, response: str, target: str) -> float:
        """
        Compute reward for a response.

        For math problems (GSM8K):
        - Extract numerical answer
        - Compare to ground truth
        - Reward = 1.0 if correct, 0.0 if incorrect

        Could also use partial credit for intermediate steps.

        TODO: Implement
        """
        raise NotImplementedError("Reward computation not implemented")

    def _grpo_loss(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        group_size: int
    ) -> torch.Tensor:
        """
        Compute GRPO loss.

        GRPO groups responses and updates based on relative rankings:
        - Higher reward responses should have higher probability
        - No need for reference model

        TODO: Implement following DeepSeek GRPO paper
        """
        raise NotImplementedError("GRPO loss not implemented")

    def train(self) -> None:
        """
        Main GRPO training loop.

        TODO: Implement
        """
        raise NotImplementedError(
            "Phase 3 (GRPO training) not yet implemented.\n"
            "Reference implementations:\n"
            "- TRL GRPOTrainer: https://huggingface.co/docs/trl/main/en/grpo_trainer\n"
            "- DeepSeek GRPO paper\n"
        )


def run_phase3_training(config: Optional[Phase3Config] = None) -> None:
    """
    Main entry point for Phase 3 GRPO training.

    TODO: Implement
    """
    config = config or Phase3Config()
    trainer = GRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3: GRPO Training")
    parser.add_argument("--trm_checkpoint", type=str, default="./checkpoints/phase2/best.pt")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/phase3")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    args = parser.parse_args()

    config = Phase3Config(
        trm_checkpoint=args.trm_checkpoint,
        output_dir=args.output_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
    )

    run_phase3_training(config)
