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
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from huggingface_hub import HfApi, create_repo

# Import config classes so pickle can find them when loading checkpoints
# (checkpoints were saved with configs from __main__)
from src.train.phase1_compressor import Phase1Config
from src.train.phase2_trm import Phase2Config

# Make them available as __main__.* for pickle compatibility
sys.modules["__main__"].Phase1Config = Phase1Config  # type: ignore[attr-defined]
sys.modules["__main__"].Phase2Config = Phase2Config  # type: ignore[attr-defined]


def _fmt_metric(metrics: dict, key: str, fmt: str = ".4f") -> str:
    """Format a metric value or return N/A."""
    val = metrics.get(key)
    if isinstance(val, float):
        return f"{val:{fmt}}"
    return "N/A"


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

    # Format metrics
    mse = _fmt_metric(metrics, "mse", ".6f")
    cos_sim = _fmt_metric(metrics, "cosine_similarity")
    rel_err = _fmt_metric(metrics, "relative_error")
    var_ratio = _fmt_metric(metrics, "variance_ratio")

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
| MSE Loss | {mse} |
| Cosine Similarity | {cos_sim} |
| Relative Error | {rel_err} |
| Variance Ratio | {var_ratio} |

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


def create_trm_model_card(
    checkpoint: dict,
    repo_id: str,
    description: str | None = None,
) -> str:
    """Generate a model card for the TRM (Phase 2)."""
    config = checkpoint.get("config")
    metrics = checkpoint.get("best_metrics", {})

    d_compressed = getattr(config, "d_compressed", "unknown") if config else "unknown"
    n_layers = getattr(config, "n_layers", "unknown") if config else "unknown"
    n_heads = getattr(config, "n_heads", "unknown") if config else "unknown"
    n_latent_steps = getattr(config, "n_latent_steps", "unknown") if config else "unknown"
    n_deep_recursions = getattr(config, "n_deep_recursions", "unknown") if config else "unknown"

    # Format metrics
    mse = _fmt_metric(metrics, "mse", ".6f")
    cos_sim = _fmt_metric(metrics, "cosine_similarity")

    card = f"""---
tags:
- llm-trm
- trm
- recursive-reasoning
- thinking-model
library_name: pytorch
---

# LLM-TRM Sequence TRM

{description or "Trained TRM for sequence-to-sequence reasoning (Phase 2)."}

## Model Details

- **Architecture**: Tiny Recursive Network with transformer blocks
- **Compressed dimension**: {d_compressed}
- **Layers**: {n_layers}
- **Heads**: {n_heads}
- **Latent steps (n)**: {n_latent_steps}
- **Deep recursions (T)**: {n_deep_recursions}

## Training Metrics

| Metric | Value |
|--------|-------|
| MSE Loss | {mse} |
| Cosine Similarity | {cos_sim} |

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from src.train.phase2_trm import SequenceTRM

# Download and load
checkpoint_path = hf_hub_download(repo_id="{repo_id}", filename="trm.pt")
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Initialize TRM
trm = SequenceTRM(
    d_compressed={d_compressed},
    n_layers={n_layers},
    n_heads={n_heads},
)
trm.load_state_dict(checkpoint["model"])

# Use: takes [B, L, D'] context, outputs [B, L+1, D']
compressed_hidden = ...  # [B, L, {d_compressed}]
output = trm(compressed_hidden, n_steps=4)  # [B, L+1, {d_compressed}]
reasoning_result = output[:, -1, :]  # [B, {d_compressed}]
```

## Part of LLM-TRM

This TRM is part of the [LLM-TRM project](https://github.com/anonx3247/llm-trm) for integrating Tiny Recursive Models with language models.
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
    # weights_only=False needed because checkpoint contains config dataclasses
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Validate checkpoint - support both Phase 1 (compressor) and Phase 2 (model)
    if "compressor" not in checkpoint and "model" not in checkpoint:
        raise ValueError("Checkpoint does not contain 'compressor' or 'model' state dict")

    # Determine checkpoint type
    is_phase2 = "model" in checkpoint and "compressor" not in checkpoint

    # Create repo
    api = HfApi()
    print(f"Creating/accessing repo: {repo_id}")
    create_repo(repo_id, repo_type="model", exist_ok=True, private=private)

    # Generate model card
    print("Generating model card...")
    if is_phase2:
        model_card = create_trm_model_card(checkpoint, repo_id, description)
        upload_filename = "trm.pt"
    else:
        model_card = create_model_card(checkpoint, repo_id, description)
        upload_filename = "compressor.pt"

    # Save model card locally
    readme_path = ckpt_path.parent / "README.md"
    readme_path.write_text(model_card)

    # Upload files
    print(f"Uploading checkpoint as {upload_filename}...")
    api.upload_file(
        path_or_fileobj=str(ckpt_path),
        path_in_repo=upload_filename,
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
