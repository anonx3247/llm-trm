# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LLM-TRM**: Tiny Recursive Models for iterative reasoning in language models. Integrates TRM with SmolLMv3-3B using LoRA adapters and Perceiver-style latent attention compression for 256x sequence compression.

Based on "Less is More: Recursive Reasoning with Tiny Networks" by Alexia Jolicoeur-Martineau.

## Commands

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Training phases (all placeholders currently)
python -m src.train.phase1_compressor --stage 1a   # Compressor identity training
python -m src.train.phase1_compressor --stage 1b   # Compressor CoT finetuning
python -m src.train.phase2_datagen --dataset gsm8k # Generate hidden state pairs
python -m src.train.phase2_trm --data_path ./data  # TRM iteration training
python -m src.train.phase3_grpo                    # GRPO training (placeholder)

# Type checking and linting
mypy src/ --ignore-missing-imports
ruff check src/
ruff format src/
```

## Architecture

```
src/
├── models/
│   ├── MODELS.md         # Detailed model documentation
│   ├── trm.py            # Core TRM: TransformerBlock, TinyRecursiveNetwork, RecursiveReasoningBase
│   ├── compression.py    # LatentAttentionCompressor (Perceiver-style)
│   └── smollm.py         # HiddenStateTRM, SmolLMv3WithTRM (integration)
└── train/
    ├── phase1_compressor.py  # Stage 1a: Identity, Stage 1b: CoT
    ├── phase2_datagen.py     # Generate hidden_pre/hidden_post pairs
    ├── phase2_trm.py         # TRM learns hidden_pre → hidden_post
    └── phase3_grpo.py        # GRPO: freeze LLM, train TRM+compressor
```

### Training Pipeline

| Phase | Script | Goal |
|-------|--------|------|
| 1a | `phase1_compressor.py` | Identity training on regular hidden states |
| 1b | `phase1_compressor.py` | Finetune on CoT thinking trajectories |
| 2 | `phase2_datagen.py` + `phase2_trm.py` | TRM learns to map hidden_pre → hidden_post |
| 3 | `phase3_grpo.py` | GRPO: improve beyond mimicking CoT |

### Data Flow

1. Input → SmolLMv3 (frozen) → LoRA adapters → Hidden States `[B, L, 3072]`
2. Thinking triggers TRM processing
3. Compress via cross-attention: `[B, L, D]` → `[B, 256, D]`
4. TRM recursive reasoning (n=6 latent × T=3 deep × N_sup=16 supervision)
5. Sliding window: drop first 256, append 256 TRM states → `[B, L, D]`
6. Continue generation conditioned on TRM reasoning

## Python Standards

- Use `uv` for package management (not pip directly, not nix)
- Type hints on all functions
- Run `mypy` for type checking
- Run `ruff check` for linting and `ruff format` for formatting

## Key References

### TRM Paper (`papers/less-is-more-TRM/paper.tex`)

**Recursion Algorithm:**
```python
def latent_recursion(x, y, z, n=6):
    for _ in range(n):
        z = net(x + y + z)
    y = net(y + z)
    return y, z

def deep_recursion(x, y, z, T=3):
    with torch.no_grad():
        for _ in range(T-1):
            y, z = latent_recursion(x, y, z)
    y, z = latent_recursion(x, y, z)  # 1 with gradients
    return y, z
```

**Training Hyperparameters:**
- n_layers=2 (more layers → overfitting)
- n_latent_steps=6 (n)
- n_deep_recursions=3 (T)
- n_supervision_steps=16 (N_sup)
- EMA=0.999 (critical for stability)
- AdamW: β1=0.9, β2=0.95
- LR: 1e-4 (1e-2 for embeddings)

**Effective depth:** `n_layers × (n+1) × T × N_sup = 2 × 7 × 3 × 16 = 672`

### SmolLM3 (https://huggingface.co/blog/smollm3)

**Thinking Mode:**
- Uses `enable_thinking=True/False` in chat template (NOT special tokens)
- Mode control via `/think` and `/no_think` in system prompt

**Model:**
- 3B parameters, 128k context
- GQA with 4 groups
- NoPE in every 4th layer

**Resources:**
- Model: HuggingFaceTB/SmolLM3-3B
- Repo: https://github.com/huggingface/smollm
