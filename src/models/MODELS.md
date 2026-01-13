# LLM-TRM Model Architecture

This document explains the models in this directory and how they interoperate.

## Overview

LLM-TRM integrates Tiny Recursive Models (TRM) with SmolLMv3-3B to enable latent reasoning. Instead of generating chain-of-thought tokens, the model reasons in hidden state space through recursive iterations.

```
Input → SmolLM3 → Hidden States → [Compress] → [TRM Reasoning] → [Slide] → Output
                                      ↓              ↓
                                  [B,L,D]→[B,M,D]  Iterate y,z
```

## Files

| File | Purpose |
|------|---------|
| `trm.py` | Core TRM: recursive reasoning network |
| `compression.py` | Perceiver-style latent attention compression |
| `smollm.py` | SmolLMv3 + TRM integration |

---

## TRM (trm.py)

Based on ["Less is More: Recursive Reasoning with Tiny Networks"](https://arxiv.org/abs/2505.00000) by Alexia Jolicoeur-Martineau.

### Key Insight

TRM achieves deep effective computation (672 layers) with a tiny 2-layer network through:
1. **Latent recursion**: Update latent `z` given input `x`, prediction `y`, and current `z`
2. **Deep recursion**: Run T-1 iterations without gradients, 1 with
3. **Deep supervision**: Multiple supervision steps with loss at each

### Classes

#### `TransformerBlock`
Standard transformer block with RMSNorm, self-attention, and SwiGLU MLP.

#### `TinyRecursiveNetwork`
Stack of TransformerBlocks. Paper finding: **2 layers is optimal** (more layers → overfitting).

#### `RecursiveReasoningBase`
Base class with core recursion logic. Subclassed by both `TRM` and `HiddenStateTRM`.

**Key methods:**
```python
def latent_recursion(self, x, y, z) -> (y, z):
    """
    1. Update z n times: z = net(x + y + z)
    2. Update y once: y = net(y + z)
    """

def run_deep_recursion(self, x, y, z) -> (y, z):
    """
    T-1 recursions without gradients (improve y,z)
    1 recursion with gradients (for backprop)
    """
```

#### `TRM`
Full standalone TRM with embeddings and output head. Used for token-level tasks (Sudoku, ARC-AGI).

### Hyperparameters (from paper)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_layers` | 2 | Transformer layers (less is more!) |
| `n_latent_steps` | 6 | Latent reasoning iterations (n) |
| `n_deep_recursions` | 3 | Deep recursions (T) |
| `n_supervision_steps` | 16 | Max supervision steps (N_sup) |
| `ema_decay` | 0.999 | EMA for stability |

**Effective depth**: `n_layers × (n+1) × T × N_sup = 2 × 7 × 3 × 16 = 672`

---

## LatentAttentionCompressor (compression.py)

Perceiver-style compression that converts variable-length sequences to fixed latents.

### Purpose

SmolLM3 supports 128k context, but processing full sequences through TRM is expensive. The compressor:
- Reduces `[B, L, D]` → `[B, M, D]` where M << L
- Uses learned latent queries + cross-attention
- Handles variable-length inputs (no fixed L)

### Architecture

```
Latent Queries [M, D]
       ↓
Cross-Attention(Q=latents, K=sequence, V=sequence)
       ↓
LayerNorm + Feed-Forward
       ↓
Compressed [B, M, D]
```

### Compression Ratios

| num_latents | Context | Compression |
|-------------|---------|-------------|
| 256 | 65k | 256× |
| 128 | 65k | 512× |
| 512 | 65k | 128× |

---

## SmolLMv3WithTRM (smollm.py)

Integration of SmolLMv3-3B with TRM for enhanced reasoning.

### SmolLMv3 Background

From [HuggingFace Blog](https://huggingface.co/blog/smollm3):
- **Size**: 3B parameters
- **Context**: 128k tokens
- **Thinking**: Uses `enable_thinking=True` in chat template (NOT special tokens)
- **Architecture**: GQA with 4 groups, NoPE in every 4th layer

### HiddenStateTRM

TRM adapted for LLM hidden states with sliding window output.

**Flow:**
```
Hidden States [B, L, D]
       ↓
   Compress
       ↓
    [B, M, D]
       ↓
  TRM Reasoning
  (iterate y, z)
       ↓
    [B, M, D]
       ↓
 Sliding Window
       ↓
Hidden States [B, L, D]
```

**Sliding Window**: Drop first M positions, append M TRM reasoning states:
```python
output = torch.cat([
    hidden_states[:, M:, :],  # [B, L-M, D] - original (truncated)
    y                          # [B, M, D] - TRM reasoning
], dim=1)
```

### SmolLMv3WithTRM

Full integration with LoRA adapters.

**Components:**
- SmolLMv3-3B base model (frozen or LoRA)
- LatentAttentionCompressor
- HiddenStateTRM
- `<think>` token trigger

**Training modes:**
- Base model: Frozen (or LoRA adapters)
- TRM: Trainable
- Compressor: Trainable

---

## How Models Interoperate

### Inference Flow

```
1. User prompt → Tokenize
2. SmolLM3 forward pass → Hidden states [B, L, 3072]
3. If <think> token encountered:
   a. Compress: [B, L, 3072] → [B, 256, 3072]
   b. TRM reasoning:
      - Initialize y, z = zeros
      - For each supervision step:
        - Latent recursion (n=6 times)
        - Deep recursion (T-1 no grad, 1 with grad)
   c. Sliding window: Replace first 256 with TRM output
4. LM head on modified hidden states → Logits
5. Generate answer tokens
```

### Training Pipeline

| Phase | What's Trained | Goal |
|-------|---------------|------|
| 1a | Compressor | Identity: compress/decompress hidden states |
| 1b | Compressor | Finetune on CoT trajectories |
| 2 | TRM | Map hidden_pre → hidden_post (replicate CoT) |
| 3 | TRM + Compressor | GRPO to improve beyond mimicking |

---

## References

### TRM Paper
- Title: "Less is More: Recursive Reasoning with Tiny Networks"
- Author: Alexia Jolicoeur-Martineau
- Location: `papers/less-is-more-TRM/paper.tex`

### SmolLM3
- Blog: https://huggingface.co/blog/smollm3
- Repo: https://github.com/huggingface/smollm
- Model: HuggingFaceTB/SmolLM3-3B
