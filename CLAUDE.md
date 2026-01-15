# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LLM-TRM**: Tiny Recursive Models for iterative reasoning in language models. Integrates TRM with SmolLMv3-3B using dimension compression (12x: 3072 → 256) and recursive reasoning in compressed hidden state space.

Based on "Less is More: Recursive Reasoning with Tiny Networks" by Alexia Jolicoeur-Martineau.

## Commands

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Phase 1: Compressor training
python -m src.train.phase1_compressor --stage 1a --d_compressed 256

# Phase 2: Data generation
uv run accelerate launch -m src.train.phase2_datagen \
    --dataset gsm8k --num_samples 512 --batch_size 8 --max_new_tokens 2048

# Phase 2: TRM training
uv run accelerate launch --mixed_precision bf16 -m src.train.phase2_trm \
    --data data/hidden_pairs/hidden_pairs.pt --batch_size 8 --bce_halt_loss

# Phase 3: GRPO (placeholder)
python -m src.train.phase3_grpo

# Type checking and linting
uv run mypy src/ --ignore-missing-imports
uv run ruff check src/
uv run ruff format src/
```

## Architecture

```
src/
├── models/
│   ├── trm.py            # Core TRM: TransformerBlock, TinyRecursiveNetwork
│   ├── compression.py    # DimensionCompressor (weight-tied linear)
│   └── smollm.py         # SmolLMv3 + TRM integration
└── train/
    ├── TRAINING.md       # Detailed training guide
    ├── phase1_compressor.py  # Compressor pretraining
    ├── phase2_datagen.py     # Hidden state extraction from SmolLM3
    ├── phase2_trm.py         # SequenceTRM training with halting
    └── phase3_grpo.py        # GRPO (placeholder)
```

### Training Pipeline

| Phase | Script | Goal |
|-------|--------|------|
| 1a | `phase1_compressor.py` | Train compressor on regular hidden states |
| 2-datagen | `phase2_datagen.py` | Extract hidden states from SmolLM3 thinking |
| 2-trm | `phase2_trm.py` | TRM learns: compressed_pre → compressed_post |
| 3 | `phase3_grpo.py` | GRPO: fine-tune with task rewards |

### Data Flow

1. **Datagen**: SmolLM3 generates thinking on reasoning problems (GSM8K)
2. **Extract**: Hidden states before `<think>` and after `</think>`
3. **Compress**: `[L, 3072]` → `[L, 256]` with trained compressor
4. **TRM Training**: Learn mapping from context to post-thinking state
5. **Inference**: TRM reasons in compressed space, decompress for LM head

## Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `DimensionCompressor` | compression.py | Linear compression 3072 → 256 |
| `SequenceTRM` | phase2_trm.py | TRM for variable-length sequences |
| `TRMSequenceTrainer` | phase2_trm.py | Training loop with per-step updates |
| `ThinkingDataGenerator` | phase2_datagen.py | Extract hidden states from thinking |

## Python Standards

- Use `uv` for package management
- Type hints on all functions
- Run `uv run mypy` for type checking
- Run `uv run ruff check` and `uv run ruff format`

## Key Hyperparameters

From TRM paper, adapted for hidden state prediction:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_layers` | 2 | Transformer layers (less is more!) |
| `n_latent_steps` | 6 | Latent reasoning iterations (n) |
| `n_deep_recursions` | 3 | Deep recursions (T) |
| `n_supervision_steps` | 8 | Supervision steps (N_sup) |
| `d_compressed` | 256 | Compressed dimension |
| `ema_decay` | 0.999 | Critical for stability |
| `halt_threshold` | 0.7 | Initial (adaptive rises to ~0.95) |

## Training Features

- **Per-step gradient updates**: Update after each supervision step (paper style)
- **Adaptive halt threshold**: Curriculum learning - threshold rises with cos_sim
- **BCE vs MSE halt loss**: BCE with threshold or MSE on raw cos_sim
- **Auto-resume**: Datagen resumes from checkpoints automatically

## Results (Phase 2)

| Metric | Value |
|--------|-------|
| Cosine Similarity | 0.9991 |
| MSE Loss | 0.00488 |
| Halt Probability | 1.0 |

## References

- **TRM Paper:** `papers/less-is-more-TRM/paper.tex`
- **SmolLM3:** https://huggingface.co/blog/smollm3
- **Training Guide:** `src/train/TRAINING.md`
