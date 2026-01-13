# Training Guide

This document explains how to run the training phases for the LLM-TRM project.

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

### Features

| Feature | Description | Default |
|---------|-------------|---------|
| **Early Stopping** | Stops training when loss plateaus | Enabled, patience=3 |
| **Best Checkpoint Only** | Only saves best model (saves storage) | Enabled |
| **Torch Compile** | Uses `torch.compile` for faster training | Enabled |
| **Multi-GPU** | Distributed training via Accelerate | Supported |
| **Wandb Logging** | Tracks metrics and saves artifacts | Enabled |

### Running Training

#### Single GPU

```bash
# Stage 1a with default settings (D' = 256)
python -m src.train.phase1_compressor --stage 1a

# Custom compression dimension
python -m src.train.phase1_compressor --stage 1a --d_compressed 512

# Stage 1b (requires stage 1a checkpoint)
python -m src.train.phase1_compressor --stage 1b

# Disable early stopping (train all epochs)
python -m src.train.phase1_compressor --stage 1a --no_early_stopping

# Disable torch.compile (for debugging)
python -m src.train.phase1_compressor --stage 1a --no_torch_compile
```

#### Multi-GPU (Scaleway / Cloud)

First, configure accelerate for your setup:

```bash
accelerate config
```

Select your multi-GPU configuration (number of GPUs, distributed strategy, etc.).

Then launch training:

```bash
# Multi-GPU training
accelerate launch -m src.train.phase1_compressor --stage 1a --d_compressed 256

# With specific GPU count
accelerate launch --num_processes 4 -m src.train.phase1_compressor --stage 1a
```

#### Compression Ratio Sweep

To find the optimal compression-quality tradeoff, run a sweep over different D' values:

```bash
# Default sweep: D' = [64, 128, 256, 512, 1024]
python -m src.train.phase1_compressor --sweep

# Custom D' values
python -m src.train.phase1_compressor --sweep --sweep_d_values 128 256 384 512

# Multi-GPU sweep
accelerate launch -m src.train.phase1_compressor --sweep
```

Each D' value creates a separate training run with its own checkpoint directory and wandb run.

### Configuration Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--stage` | `1a` | Training stage (`1a` or `1b`) |
| `--d_compressed` | `256` | Compressed dimension D' |
| `--batch_size` | `8` | Batch size per GPU |
| `--num_epochs` | `10` | Maximum training epochs |
| `--learning_rate` | `1e-3` | Learning rate |
| `--num_samples` | `50000` | Number of training samples |
| `--max_seq_length` | `512` | Maximum sequence length |
| `--output_dir` | `./checkpoints/phase1` | Checkpoint directory |
| `--seed` | `42` | Random seed |

#### Early Stopping Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--early_stopping` | `True` | Enable early stopping |
| `--no_early_stopping` | - | Disable early stopping |
| `--early_stopping_patience` | `3` | Epochs without improvement before stopping |

#### Torch Compile Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--torch_compile` | `True` | Enable torch.compile |
| `--no_torch_compile` | - | Disable torch.compile (for debugging) |

#### Wandb Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_wandb` | `True` | Enable wandb logging |
| `--no_wandb` | - | Disable wandb logging |
| `--wandb_project` | `llm-trm-phase1` | Wandb project name |
| `--wandb_run_name` | Auto-generated | Custom run name |

### Metrics

The following metrics are logged to wandb:

| Metric | Description | Goal |
|--------|-------------|------|
| `mse` | Mean squared reconstruction error | Lower is better |
| `cosine_similarity` | Direction preservation between original and reconstructed | Higher is better (target: >0.95) |
| `relative_error` | `‖reconstructed - original‖ / ‖original‖` | Lower is better |
| `variance_ratio` | How much variance is preserved | Closer to 1.0 is better |

### Checkpoints

Only the best checkpoint is saved (overwrites on improvement):

```
checkpoints/phase1/
├── stage1a_best.pt      # Best stage 1a model
└── stage1b_best.pt      # Best stage 1b model (after finetuning)
```

For sweeps, each D' gets its own subdirectory:

```
checkpoints/phase1/
├── d64/
│   └── stage1a_best.pt
├── d128/
│   └── stage1a_best.pt
├── d256/
│   └── stage1a_best.pt
└── ...
```

### Expected Results

| D' | Compression | Expected Cosine Sim | Parameters |
|----|-------------|---------------------|------------|
| 64 | 48x | ~0.85 | 197K |
| 128 | 24x | ~0.92 | 393K |
| 256 | 12x | ~0.96 | 786K |
| 512 | 6x | ~0.98 | 1.6M |
| 1024 | 3x | ~0.99 | 3.1M |

These are rough estimates. Actual results depend on the data distribution.

---

## Phase 2: TRM Training (Coming Soon)

Train the TRM to map pre-thinking hidden states to post-thinking hidden states.

## Phase 3: GRPO Training (Coming Soon)

RL fine-tuning with answer correctness as reward.
