"""
Phase 2 Data Generation: Extract Hidden State Pairs

Generate training data for TRM iteration training by:
1. Running SmolLM3 in thinking mode on reasoning problems
2. Capturing hidden states at two points:
   - hidden_pre: Hidden state right before <thinking> token
   - hidden_post: Hidden state right after </thinking> token
3. Saving dataset as tensors

The resulting dataset is used in phase2_trm.py to train TRM
to map hidden_pre -> hidden_post via iterative reasoning.

Usage:
    python -m src.train.phase2_datagen \\
        --dataset gsm8k \\
        --output_dir ./data/hidden_pairs \\
        --num_samples 1000

TODO: Implement data generation logic
"""

from dataclasses import dataclass

import torch


@dataclass
class DataGenConfig:
    """Configuration for hidden state pair generation"""

    # Model
    model_name: str = "HuggingFaceTB/SmolLM3-3B"

    # Data source
    dataset: str = "gsm8k"  # Dataset to use for reasoning problems
    num_samples: int = 1000
    max_seq_length: int = 2048

    # Output
    output_dir: str = "./data/hidden_pairs"
    save_format: str = "pt"  # "pt" for PyTorch tensors

    # Generation
    temperature: float = 0.7
    max_thinking_tokens: int = 1024


class ThinkingDataGenerator:
    """
    Generate hidden state pairs from SmolLM3 thinking trajectories.

    Process:
    1. Load SmolLM3 with enable_thinking=True
    2. For each problem:
       a. Generate response with thinking
       b. Hook into hidden states during generation
       c. Capture hidden_pre (before <thinking>) and hidden_post (after </thinking>)
       d. Record num_thinking_tokens
    3. Save dataset

    Output format:
        {
            "hidden_pre": Tensor[N, D],     # Hidden states before thinking
            "hidden_post": Tensor[N, D],    # Hidden states after thinking
            "num_tokens": List[int],        # Number of thinking tokens
            "problems": List[str],          # Original problems (for reference)
        }

    Note: SmolLM3 uses enable_thinking=True, NOT special <thinking> tokens.
    We may need to map the chat template's thinking markers.

    TODO: Implement generation logic
    """

    def __init__(self, config: DataGenConfig):
        self.config = config
        # TODO: Load model, tokenizer

    def _setup_hooks(self) -> None:
        """
        Set up forward hooks to capture hidden states.

        Need to capture:
        - Hidden state at the position where thinking begins
        - Hidden state at the position where thinking ends

        SmolLM3 uses chat template with enable_thinking=True,
        so we need to identify thinking boundaries in the generated text.

        TODO: Implement hook registration
        """
        raise NotImplementedError("Hook setup not implemented")

    def _extract_thinking_boundaries(self, text: str) -> tuple:
        """
        Find the start and end positions of thinking in generated text.

        SmolLM3 uses a chat template approach, not explicit tokens.
        Need to investigate the exact format of thinking output.

        TODO: Implement boundary detection
        """
        raise NotImplementedError("Thinking boundary detection not implemented")

    def generate_pairs(self, problems: list[str]) -> dict[str, torch.Tensor]:
        """
        Generate hidden state pairs for a list of problems.

        Args:
            problems: List of reasoning problems

        Returns:
            Dictionary with hidden_pre, hidden_post, num_tokens

        TODO: Implement
        """
        raise NotImplementedError(
            "Hidden state pair generation not yet implemented.\n"
            "See SmolLM3 blog for thinking mode details:\n"
            "https://huggingface.co/blog/smollm3"
        )

    def run(self) -> None:
        """Generate and save the dataset"""
        raise NotImplementedError("Data generation not implemented")


def generate_hidden_state_pairs(config: DataGenConfig | None = None) -> None:
    """
    Main entry point for generating hidden state pairs.

    TODO: Implement
    """
    config = config or DataGenConfig()
    generator = ThinkingDataGenerator(config)
    generator.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2: Generate Hidden State Pairs")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="./data/hidden_pairs")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM3-3B")
    args = parser.parse_args()

    config = DataGenConfig(
        dataset=args.dataset,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        model_name=args.model_name,
    )

    generate_hidden_state_pairs(config)
