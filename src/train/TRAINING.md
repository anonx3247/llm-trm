# Training Guide

This document explains how to run the training phases for the LLM-TRM project.

## Overview

The training pipeline consists of three phases:

```
Phase 1: Compressor Pretraining
         ↓
Phase 2: TRM Iteration Training
         ↓
Phase 3: GRPO Training (Planned)
```

---

## Phase 1: Compressor Pretraining

Train a `DimensionCompressor` to faithfully compress and reconstruct LLM hidden states. This is a prerequisite for TRM training.

### Architecture

The compressor performs dimension compression (not sequence compression):

```
[B, L, D] → compress → [B, L, D'] → decompress → [B, L, D]
```

Where:
- `D = 3072` (SmolLM3-3B hidden size)
- `D'` = compressed dimension (e.g., 256 for 12x compression)
- Sequence length `L` is preserved

The compressor uses weight-tied (symmetric) linear layers for stability.

### Two-Stage Training

| Stage | Purpose | Data |
|-------|---------|------|
| **1a** | Identity reconstruction | Regular text from fineweb-edu |
| **1b** | CoT trajectory finetuning | Thinking sequences (builds on 1a checkpoint) |

### Running Training

```bash
# Stage 1a with default settings (D' = 256)
python -m src.train.phase1_compressor --stage 1a

# Custom compression dimension
python -m src.train.phase1_compressor --stage 1a --d_compressed 512

# Multi-GPU training
accelerate launch -m src.train.phase1_compressor --stage 1a --d_compressed 256
```

### Configuration Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--stage` | `1a` | Training stage (`1a` or `1b`) |
| `--d_compressed` | `256` | Compressed dimension D' |
| `--batch_size` | `8` | Batch size per GPU |
| `--num_epochs` | `10` | Maximum training epochs |
| `--learning_rate` | `1e-3` | Learning rate |
| `--num_samples` | `50000` | Number of training samples |

### Expected Results

| D' | Compression | Expected Cosine Sim | Parameters |
|----|-------------|---------------------|------------|
| 64 | 48x | ~0.85 | 197K |
| 128 | 24x | ~0.92 | 393K |
| 256 | 12x | ~0.96 | 786K |
| 512 | 6x | ~0.98 | 1.6M |

### Pushing to Hub

```bash
python scripts/push_to_hub.py \
    --checkpoint ./checkpoints/phase1/stage1a_best.pt \
    --repo username/llm-trm-compressor
```

---

## Phase 2: TRM Iteration Training

Train the SequenceTRM to predict post-thinking hidden states from pre-thinking context.

### Pipeline Overview

```
1. Reasoning Problems (GSM8K)
         ↓
2. SmolLM3 generates with <think>...</think>
         ↓
3. Extract hidden states:
   - hidden_pre: [L, D] before <think>
   - hidden_post: [D] after </think>
         ↓
4. Compress with trained compressor:
   - [L, D] → [L, D']
   - [D] → [D']
         ↓
5. TRM learns: compressed_pre → compressed_post
```

### Step 1: Generate Hidden State Pairs

Extract hidden states from SmolLM3 thinking trajectories.

```bash
# Basic usage with GSM8K
uv run accelerate launch -m src.train.phase2_datagen \
    --dataset gsm8k \
    --num_samples 512 \
    --batch_size 8 \
    --max_new_tokens 2048
```

#### Supported Datasets

| Dataset | Description | Typical Thinking Tokens |
|---------|-------------|-------------------------|
| `gsm8k` | Grade school math | 200-500 |
| `aime` | Competition math (harder) | 500-2000 |
| `humaneval` | Python coding | 300-800 |
| `mbpp` | Python coding | 200-600 |
| `apps` | Competitive programming | 500-1500 |

#### Configuration Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `gsm8k` | Dataset name (comma-separated for mixing) |
| `--num_samples` | `1000` | Number of samples to generate |
| `--batch_size` | `4` | Generation batch size |
| `--max_new_tokens` | `512` | Max tokens to generate |
| `--output_dir` | `./data/hidden_pairs` | Output directory |
| `--checkpoint_every` | `100` | Save checkpoint frequency |

#### Auto-Resume

The datagen script automatically resumes from checkpoints:

```bash
# First run - creates checkpoints
uv run accelerate launch -m src.train.phase2_datagen --num_samples 1000

# If interrupted, just re-run - auto-resumes
uv run accelerate launch -m src.train.phase2_datagen --num_samples 1000

# Force fresh start
uv run accelerate launch -m src.train.phase2_datagen --num_samples 1000 --no_auto_resume
```

Configuration is saved to `datagen_config.json` for validation on resume.

#### Output Format

```python
{
    "hidden_pre": List[Tensor],    # List of [L_i, 3072] context sequences
    "hidden_post": Tensor,          # [N, 3072] target states
    "seq_lengths": List[int],       # Context lengths
    "num_thinking_tokens": List[int], # Thinking tokens per sample
    "problems": List[str],          # Original problems
    "hidden_size": int,             # 3072 for SmolLM3
}
```

### Step 2: Train TRM

Train the SequenceTRM on the generated hidden state pairs.

```bash
# Recommended: BCE halt loss with adaptive threshold
uv run accelerate launch --mixed_precision bf16 -m src.train.phase2_trm \
    --data data/hidden_pairs/hidden_pairs.pt \
    --batch_size 8 \
    --bce_halt_loss

# Alternative: MSE halt loss (more stable, less interpretable)
uv run accelerate launch --mixed_precision bf16 -m src.train.phase2_trm \
    --data data/hidden_pairs/hidden_pairs.pt \
    --batch_size 8
```

#### SequenceTRM Architecture

```python
class SequenceTRM(nn.Module):
    """
    Operates on variable-length compressed sequences.

    Input: [B, L, D'] compressed context
    Output: [B, L+1, D'] with appended reasoning token

    Components:
    - reasoning_token: Learnable [1, D'] appended to input
    - net: TinyRecursiveNetwork (2-layer transformer)
    - halt_head: Linear(D', 1) predicts when to stop
    """
```

#### Training Algorithm (Paper Style)

The training follows the TRM paper with per-step gradient updates:

```python
for batch in dataloader:
    x_aug, y, z = setup(batch)  # Initialize state

    for step in range(N_supervision):
        # Single deep recursion
        y, z = deep_recursion(x_aug, y, z)

        # Compute loss
        loss = mse_loss + halt_loss_weight * halt_loss

        # Per-step gradient update (paper style)
        loss.backward()
        optimizer.step()

        # Detach state for next step
        y, z = y.detach(), z.detach()

        # Early stop if confident
        if halt_prob > 0.5:
            break
```

#### Halting Mechanism

Two options for training the halt head:

**Option 1: MSE on Cosine Similarity (default)**
```python
halt_target = cosine_similarity(output, target)  # [0, 1]
halt_loss = mse_loss(halt_prob, halt_target)
```
- Smooth gradient signal
- More stable training

**Option 2: BCE with Adaptive Threshold (`--bce_halt_loss`)**
```python
halt_target = (cos_sim > threshold).float()  # Binary
halt_loss = bce_loss(halt_prob, halt_target)

# Curriculum: threshold rises each epoch
threshold = avg_cos_sim * 0.95
```
- Interpretable halt decision
- Curriculum learning effect

#### Configuration Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | `./data/hidden_pairs/hidden_pairs.pt` | Path to hidden state pairs |
| `--compressor_checkpoint` | `anonx3247/llm-trm-compressor` | Compressor weights |
| `--batch_size` | `32` | Training batch size |
| `--num_epochs` | `100` | Number of epochs |
| `--learning_rate` | `1e-4` | Learning rate |

**TRM Architecture:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_layers` | `2` | Transformer layers |
| `--n_latent_steps` | `6` | Latent recursions (n) |
| `--n_deep_recursions` | `3` | Deep recursions (T) |
| `--n_supervision_steps` | `8` | Supervision steps (N_sup) |

**Halting Options:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_halting` | `True` | Train halt head |
| `--halt_threshold` | `0.7` | Initial threshold for BCE |
| `--adaptive_halt_threshold` | `True` | Update threshold each epoch |
| `--bce_halt_loss` | `False` | Use BCE instead of MSE |
| `--halt_loss_weight` | `0.5` | Weight for halt loss |

**Training Style:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--per_step_updates` | `True` | Gradient update per supervision step |
| `--use_ema` | `True` | Exponential moving average |
| `--ema_decay` | `0.999` | EMA decay rate |

#### Results (GSM8K, 278 samples)

| Metric | Final Value |
|--------|-------------|
| Cosine Similarity | **0.9991** |
| MSE Loss | 0.00488 |
| Halt Probability | 1.0 |
| Relative Error | 0.13 |
| Halt Threshold (adaptive) | 0.949 |

### Pushing to Hub

```bash
python scripts/push_to_hub.py \
    --checkpoint ./checkpoints/phase2/best.pt \
    --repo username/llm-trm-sequence-trm
```

---

## Phase 3: GRPO Training (Planned)

Fine-tune TRM + compressor with Group Relative Policy Optimization using task rewards.

```bash
python -m src.train.phase3_grpo \
    --trm_checkpoint ./checkpoints/phase2/best.pt \
    --compressor_checkpoint ./checkpoints/phase1/stage1a_best.pt
```

### Planned Features

- Freeze LLM, train TRM + compressor
- Reward based on answer correctness
- Multiple sampling for relative ranking
- Curriculum on problem difficulty

---

## Tips and Troubleshooting

### Out of Memory

```bash
# Reduce batch size
--batch_size 4

# Use mixed precision
accelerate launch --mixed_precision bf16 ...

# For datagen, reduce max_new_tokens
--max_new_tokens 1024
```

### NaN Loss

- Ensure compressor is loaded correctly
- Check data statistics (should have reasonable mean/std)
- EMA helps stability (`--use_ema`)
- RMSNorm is used for stability (already default)

### Slow Datagen

- Increase batch_size (limited by VRAM)
- Early stopping on `</think>` is automatic
- Use checkpointing (`--checkpoint_every 100`)

### Poor Halt Training

- Use `--bce_halt_loss` with `--adaptive_halt_threshold`
- Start with low threshold (0.7)
- Check that cos_sim is improving (halt target becomes meaningful)
