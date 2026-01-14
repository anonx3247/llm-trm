#!/usr/bin/env python3
"""
Push trained compressor weights to the Hugging Face Hub.

Usage:
    python scripts/push_to_hub.py --checkpoint ./checkpoints/phase1/stage1a_best.pt --repo username/llm-trm-compressor-256

    # With custom model card
    python scripts/push_to_hub.py --checkpoint ./checkpoints/phase1/d256/stage1a_best.pt \
        --repo username/llm-trm-compressor-256 \
        --description "Compressor trained on fineweb-edu"
"""

import argparse
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo


def create_model_card(
    checkpoint: dict,
    repo_id: str,
    description: str | None = None,
) -> str:
    """Generate a model card for the compressor."""
    config = checkpoint.get("config")
    metrics = checkpoint.get("best_metrics", {})

    d_compressed = getattr(config, "d_compressed", "unknown") if config else "unknown"
    hidden_size = getattr(config, "hidden_size", "unknown") if config else "unknown"
    compression_ratio = getattr(config, "compression_ratio", "unknown") if config else "unknown"

    card = f"""---
tags:
- llm-trm
- compressor
- dimensionality-reduction
library_name: pytorch
---

# LLM-TRM Dimension Compressor

{description or "Trained dimension compressor for LLM-TRM (Tiny Recursive Models)."}

## Model Details

- **Architecture**: Linear compression with weight-tied decompression
- **Input dimension**: {hidden_size}
- **Compressed dimension**: {d_compressed}
- **Compression ratio**: {compression_ratio}x

## Training Metrics

| Metric | Value |
|--------|-------|
| MSE Loss | {metrics.get("mse", "N/A"):.6f if isinstance(metrics.get('mse'), float) else 'N/A'} |
| Cosine Similarity | {metrics.get("cosine_similarity", "N/A"):.4f if isinstance(metrics.get('cosine_similarity'), float) else 'N/A'} |
| Relative Error | {metrics.get("relative_error", "N/A"):.4f if isinstance(metrics.get('relative_error'), float) else 'N/A'} |
| Variance Ratio | {metrics.get("variance_ratio", "N/A"):.4f if isinstance(metrics.get('variance_ratio'), float) else 'N/A'} |

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from src.models.compression import DimensionCompressor

# Download and load
checkpoint_path = hf_hub_download(repo_id="{repo_id}", filename="compressor.pt")
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Initialize compressor
compressor = DimensionCompressor(
    d_model={hidden_size},
    d_compressed={d_compressed},
)
compressor.load_state_dict(checkpoint["compressor"])

# Use
hidden_states = ...  # [B, L, {hidden_size}]
compressed = compressor(hidden_states)  # [B, L, {d_compressed}]
reconstructed = compressor.decompress(compressed)  # [B, L, {hidden_size}]
```

## Part of LLM-TRM

This compressor is part of the [LLM-TRM project](https://github.com/anonx3247/llm-trm) for integrating Tiny Recursive Models with language models.
"""
    return card


def push_to_hub(
    checkpoint_path: str | Path,
    repo_id: str,
    description: str | None = None,
    private: bool = False,
) -> str:
    """
    Push compressor checkpoint to the Hugging Face Hub.

    Args:
        checkpoint_path: Path to the .pt checkpoint file
        repo_id: HF repo ID (e.g., "username/model-name")
        description: Optional description for the model card
        private: Whether to create a private repo

    Returns:
        URL of the uploaded model
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint from {ckpt_path}...")
    # weights_only=False needed because checkpoint contains Phase1Config dataclass
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Validate checkpoint
    if "compressor" not in checkpoint:
        raise ValueError("Checkpoint does not contain 'compressor' state dict")

    # Create repo
    api = HfApi()
    print(f"Creating/accessing repo: {repo_id}")
    create_repo(repo_id, repo_type="model", exist_ok=True, private=private)

    # Generate model card
    print("Generating model card...")
    model_card = create_model_card(checkpoint, repo_id, description)

    # Save model card locally
    readme_path = ckpt_path.parent / "README.md"
    readme_path.write_text(model_card)

    # Upload files
    print("Uploading checkpoint...")
    api.upload_file(
        path_or_fileobj=str(ckpt_path),
        path_in_repo="compressor.pt",
        repo_id=repo_id,
        repo_type="model",
    )

    print("Uploading model card...")
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )

    # Clean up local README
    readme_path.unlink()

    url = f"https://huggingface.co/{repo_id}"
    print(f"\nModel uploaded to: {url}")
    return url


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push trained compressor weights to Hugging Face Hub"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint file (.pt)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="HF repo ID (e.g., 'username/llm-trm-compressor-256')",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Optional description for the model card",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository",
    )

    args = parser.parse_args()

    push_to_hub(
        checkpoint_path=args.checkpoint,
        repo_id=args.repo,
        description=args.description,
        private=args.private,
    )


if __name__ == "__main__":
    main()
