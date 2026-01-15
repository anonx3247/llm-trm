# LLM-TRM: Recursive Reasoning for Language Models

Add iterative reasoning capabilities to language models using Tiny Recursive Models (TRM) with minimal parameter overhead.

**Based on:** "Less is More: Recursive Reasoning with Tiny Networks" by Alexia Jolicoeur-Martineau

## Overview

This implementation integrates TRM with SmolLMv3-3B using:
- **Dimension compression** via weight-tied linear layers (12x compression: 3072 → 256)
- **Recursive reasoning** in compressed hidden state space
- **Curriculum learning** with adaptive halt thresholds

**Key Innovation:** TRM reasons in latent space instead of generating chain-of-thought tokens, achieving effective depth of 672 layers with only a 2-layer network.

## Architecture

```
                        TRAINING PIPELINE
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Phase 1: Compressor Pretraining                               │
│  ────────────────────────────────                              │
│  Hidden States [B, L, 3072] ─→ Compress ─→ [B, L, 256]        │
│                              ─→ Decompress ─→ [B, L, 3072]    │
│  Loss: MSE(original, reconstructed)                            │
│                                                                 │
│  Phase 2: TRM Iteration Training                               │
│  ───────────────────────────────                               │
│  1. SmolLM3 generates thinking on reasoning problems (GSM8K)   │
│  2. Extract hidden states: pre-<think> and post-</think>       │
│  3. Compress both with trained compressor                       │
│  4. TRM learns: compressed_pre ─→ compressed_post              │
│                                                                 │
│  Phase 3: GRPO Training (Planned)                              │
│  ────────────────────────────────                              │
│  Fine-tune TRM + compressor with task rewards                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

                        INFERENCE PIPELINE
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Input ─→ SmolLMv3 ─→ Hidden States [B, L, 3072]              │
│                              ↓                                  │
│           Compress ─→ [B, L, 256]                              │
│                              ↓                                  │
│           TRM Recursive Reasoning                               │
│           (n=6 latent × T=3 deep × N_sup=8 supervision)        │
│                              ↓                                  │
│           Decompress ─→ [B, L, 3072]                           │
│                              ↓                                  │
│           Continue Generation                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

## Training Pipeline

### Phase 1: Compressor Pretraining

Train a dimension compressor to preserve information through a bottleneck.

```bash
# Stage 1a: Identity training on regular text
python -m src.train.phase1_compressor --stage 1a --d_compressed 256

# Stage 1b: Finetune on thinking trajectories (optional)
python -m src.train.phase1_compressor --stage 1b
```

**Results:** Cosine similarity > 0.99 with 12x compression (3072 → 256)

### Phase 2: TRM Iteration Training

#### Step 1: Generate Hidden State Pairs

Extract hidden states from SmolLM3 thinking trajectories:

```bash
# Generate from GSM8K math problems
uv run accelerate launch -m src.train.phase2_datagen \
    --dataset gsm8k \
    --num_samples 512 \
    --batch_size 8 \
    --max_new_tokens 2048

# Supports multiple datasets: gsm8k, aime, humaneval, mbpp, apps
# Auto-resumes from checkpoints if interrupted
```

This extracts:
- `hidden_pre`: Hidden states before `<think>` token (context)
- `hidden_post`: Hidden state after `</think>` token (target)

#### Step 2: Train TRM

Train the SequenceTRM to predict post-thinking hidden states:

```bash
uv run accelerate launch --mixed_precision bf16 -m src.train.phase2_trm \
    --data data/hidden_pairs/hidden_pairs.pt \
    --batch_size 8 \
    --bce_halt_loss  # Use BCE with adaptive threshold
```

**Key Features:**
- **Per-step gradient updates** (paper style): Update weights after each supervision step
- **Adaptive halt threshold**: Starts at 0.7, rises to match avg cosine similarity (curriculum learning)
- **EMA** (0.999): Critical for stability

**Results:**
| Metric | Value |
|--------|-------|
| Cosine Similarity | 0.9991 |
| MSE Loss | 0.00488 |
| Halt Probability | 1.0 |
| Relative Error | 0.13 |

### Phase 3: GRPO Training (Planned)

Fine-tune TRM + compressor with actual task rewards.

```bash
python -m src.train.phase3_grpo --trm_checkpoint ./checkpoints/phase2/best.pt
```

## Pre-trained Models

| Model | HuggingFace Hub | Description |
|-------|-----------------|-------------|
| Compressor | `anonx3247/llm-trm-compressor` | Dimension compressor (3072 → 256) |
| TRM | `anonx3247/llm-trm-sequence-trm` | Sequence TRM trained on GSM8K |

## Project Structure

```
llm-trm/
├── src/
│   ├── models/
│   │   ├── trm.py           # Core TRM: TransformerBlock, TinyRecursiveNetwork
│   │   ├── compression.py   # DimensionCompressor (weight-tied linear)
│   │   └── smollm.py        # SmolLMv3 + TRM integration
│   └── train/
│       ├── TRAINING.md      # Detailed training guide
│       ├── phase1_compressor.py  # Compressor pretraining
│       ├── phase2_datagen.py     # Hidden state extraction
│       ├── phase2_trm.py         # TRM training with halting
│       └── phase3_grpo.py        # GRPO training (placeholder)
├── scripts/
│   └── push_to_hub.py       # Push models to HuggingFace
├── papers/
│   └── less-is-more-TRM/    # TRM paper source
└── requirements.txt
```

## Key Hyperparameters

From the TRM paper, adapted for hidden state prediction:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_layers` | 2 | Transformer layers (less is more!) |
| `n_latent_steps` | 6 | Latent reasoning iterations (n) |
| `n_deep_recursions` | 3 | Deep recursions (T) |
| `n_supervision_steps` | 8 | Max supervision steps (N_sup) |
| `d_compressed` | 256 | Compressed dimension |
| `ema_decay` | 0.999 | Critical for stability |
| `halt_threshold` | 0.7 (adaptive) | Initial threshold, rises with training |

**Effective depth:** `n_layers × (n+1) × T × N_sup = 2 × 7 × 3 × 8 = 336 layers`

## SequenceTRM Architecture

The TRM operates on variable-length compressed sequences:

```python
class SequenceTRM(nn.Module):
    """
    Input: compressed context [B, L, D']
    Output: [B, L+1, D'] with appended reasoning token

    Features:
    - Learnable reasoning token appended to sequence
    - Deep recursion with gradient checkpointing (T-1 without grad, 1 with)
    - Halt head predicts when to stop (trained with MSE on cosine sim)
    """
```

## Halting Mechanism

Two options for training the halt head:

1. **MSE on cosine similarity** (default, stable):
   - `halt_target = cosine_similarity(output, target)`
   - Smooth gradient signal

2. **BCE with adaptive threshold** (`--bce_halt_loss`):
   - `halt_target = (cos_sim > threshold)`
   - Threshold rises each epoch: `new_threshold = avg_cos_sim * 0.95`
   - Curriculum learning effect

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
