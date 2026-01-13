# LLM-TRM: Recursive Reasoning for Language Models

Add iterative reasoning capabilities to language models using Tiny Recursive Models (TRM) with minimal parameter overhead.

**Based on:** "Less is More: Recursive Reasoning with Tiny Networks" by Alexia Jolicoeur-Martineau

## Overview

This implementation integrates TRM with SmolLMv3-3B using:
- **LoRA adapters** for efficient LLM fine-tuning
- **Latent attention compression** via Perceiver-style cross-attention (256x compression)
- **Recursive reasoning** in compressed hidden state space

**Key Innovation:** TRM reasons in latent space instead of generating chain-of-thought tokens, achieving effective depth of 672 layers with only a 2-layer network.

## Architecture

```
Input → SmolLMv3 → Hidden States [B, L, 3072]
                          ↓
         Compress via CrossAttention → [B, 256, 3072]
                          ↓
              TRM Recursive Reasoning
              (n=6 latent steps × T=3 deep × N_sup=16 supervision)
                          ↓
         Sliding Window Output → [B, L, 3072]
                          ↓
                 Continue Generation
```

## Installation

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

## Quick Start

```python
from src.models import create_smollm_trm_model

model = create_smollm_trm_model(
    model_name="HuggingFaceTB/SmolLM3-3B",
    use_lora=True,
    num_latents=256,
    trm_kwargs={
        "n_layers": 2,
        "n_latent_steps": 6,
        "n_deep_recursions": 3,
        "n_supervision_steps": 8,
    }
)

outputs = model(input_ids=x, labels=y, use_trm=True)
```

## Project Structure

```
llm-trm/
├── src/
│   ├── models/
│   │   ├── MODELS.md        # Model documentation
│   │   ├── trm.py           # Core TRM implementation
│   │   ├── compression.py   # Latent attention compressor
│   │   └── smollm.py        # SmolLMv3 + TRM integration
│   └── train/
│       ├── phase1_compressor.py  # Compressor pretraining
│       ├── phase2_datagen.py     # Generate hidden state pairs
│       ├── phase2_trm.py         # TRM iteration training
│       └── phase3_grpo.py        # GRPO training (placeholder)
├── papers/
│   └── less-is-more-TRM/    # TRM paper source
├── colab_notebook.ipynb     # Self-contained notebook for Colab
├── requirements.txt
└── README.md
```

## Training Pipeline

Training consists of three phases:

### Phase 1: Compressor Pretraining
```bash
python -m src.train.phase1_compressor --stage 1a  # Identity training
python -m src.train.phase1_compressor --stage 1b  # CoT finetuning
```

### Phase 2: TRM Iteration Training
```bash
# Generate hidden state pairs from thinking trajectories
python -m src.train.phase2_datagen --dataset gsm8k --output_dir ./data/hidden_pairs

# Train TRM to map hidden_pre -> hidden_post
python -m src.train.phase2_trm --data_path ./data/hidden_pairs
```

### Phase 3: GRPO Training (Placeholder)
```bash
python -m src.train.phase3_grpo --trm_checkpoint ./checkpoints/phase2/best.pt
```

## Using Colab

Upload `colab_notebook.ipynb` to Google Colab. All code is inlined - no external imports needed.

## Key Hyperparameters

From the TRM paper:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_layers` | 2 | Transformer layers (less is more!) |
| `n_latent_steps` | 6 | Latent reasoning iterations (n) |
| `n_deep_recursions` | 3 | Deep recursions (T) |
| `n_supervision_steps` | 16 | Max supervision steps (N_sup) |
| `num_latents` | 256 | Compression ratio (256x) |
| `ema_decay` | 0.999 | Critical for stability |

**Effective depth:** `n_layers × (n+1) × T × N_sup = 2 × 7 × 3 × 16 = 672 layers`

## References

- **TRM Paper:** `papers/less-is-more-TRM/paper.tex`
- **SmolLM3 Blog:** https://huggingface.co/blog/smollm3
- **SmolLM Repo:** https://github.com/huggingface/smollm

## Citation

```bibtex
@article{jolicoeur2025trm,
  title={Less is More: Recursive Reasoning with Tiny Networks},
  author={Jolicoeur-Martineau, Alexia},
  journal={arXiv preprint},
  year={2025}
}
```

## License

See LICENSE file.
