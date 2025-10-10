# LLM-TRM: Recursive Reasoning for Language Models

Add iterative reasoning capabilities to small language models using Tiny Recursive Models (TRM) with minimal parameter overhead.

**Based on:** "Less is More: Recursive Reasoning with Tiny Networks" by Alexia Jolicoeur-Martineau

## Overview

This implementation integrates TRM with SmolLMv3-3B using:
- **LoRA adapters** for efficient LLM fine-tuning (~0.5% params)
- **Latent attention compression** via Perceiver-style cross-attention (~0.1% params)
- **Recursive reasoning** in compressed hidden state space

**Key Innovation:** Handles SmolLMv3's 65k token context by compressing to 256 learned latent queries via cross-attention, achieving 256x compression with variable-length support.


## Architecture

```
Input → SmolLMv3 (frozen) → LoRA adapters
                                ↓
                     Hidden States [B, L, 3072] (L up to 65k)
                                ↓
            Compress: CrossAttention(256 latents, hidden) → [B, 256, 3072]
                                ↓
            TRM Recursive Reasoning (40+ iterations on compressed space)
                                ↓
            Sliding Window: Drop first 256, Append 256 TRM states → [B, L, 3072]
                                ↓
                     Hidden States (with TRM reasoning at end)
                                ↓
                    Continue Generation
```

### When `<think>` Token Appears

Model learns to output `<think>` token when it needs to reason:
1. `<think>` token triggers TRM processing
2. Hidden states are compressed via learned latent queries (variable → 256 latents)
3. TRM iterates 40+ times in compressed space (256 reasoning states, not 65k tokens!)
4. Sliding window: Drop first 256 positions, append 256 TRM reasoning states
5. Model generates answer conditioned on TRM reasoning

## Installation

```bash
pip install torch transformers peft tqdm
# or with uv
uv pip install -r requirements.txt
```

## Quick Start

### Standalone TRM (for supervised learning)

```python
from src.trm import create_trm_model
import torch

model = create_trm_model(
    vocab_size=1000,
    d_model=256,
    n_layers=2,
    n_latent_steps=6,
    n_deep_recursions=3,
    n_supervision_steps=16
)

# Training
x = torch.randint(0, 1000, (4, 64))
y = torch.randint(0, 1000, (4, 64))
loss = model.compute_loss(x, y)
loss.backward()
```

### SmolLMv3 + TRM Integration

```python
from src.integration import create_smollm_trm_model

model = create_smollm_trm_model(
    model_name="HuggingFaceTB/SmolLM3-3B",
    use_lora=True,
    lora_r=16,
    num_latents=256,  # Compress up to 65k → 256 latents (256x!)
    trm_kwargs={
        "n_layers": 2,
        "n_latent_steps": 4,
        "n_deep_recursions": 2,
        "n_supervision_steps": 4,
        "compression_heads": 8  # Attention heads for compression
    }
)

# Use in training
outputs = model(input_ids=x, labels=y, use_trm=True)
loss = outputs.loss
```

## Project Structure

```
llm-trm/
├── src/
│   ├── trm/
│   │   ├── __init__.py
│   │   └── base.py          # Core TRM implementation
│   └── integration/
│       ├── __init__.py
│       └── smollm.py         # SmolLMv3 + TRM with compression
├── examples/
│   ├── train_standalone.py  # Train standalone TRM
│   └── train_smollm.py      # Train SmolLMv3 + TRM
├── requirements.txt
└── README.md
```

## Training

### Quick Start with PyTorch Lightning

```bash
# Install dependencies
pip install -r requirements.txt

# Login to wandb
wandb login

# Train with default settings
python train_lightning.py

# Or customize training
python train_lightning.py \
  --batch_size 2 \
  --num_epochs 10 \
  --learning_rate 2e-4 \
  --num_latents 256 \
  --wandb_project "my-project" \
  --wandb_name "experiment-1"
```

### Using Jupyter Notebook

For interactive training or Colab:

```bash
jupyter notebook train_notebook.ipynb
```

Or upload `train_notebook.ipynb` to Google Colab.

### Features

- **Automatic checkpointing**: Saves top 3 models based on validation loss
- **Early stopping**: Stops if validation loss doesn't improve for 3 epochs
- **Weights & Biases logging**: Track metrics, visualize training
- **Multi-GPU support**: Automatically uses all available GPUs
- **Mixed precision**: bfloat16 for faster training
- **Gradient accumulation**: Simulate larger batch sizes

### Training on Your Dataset

1. Prepare JSON dataset:
```json
[
  {"question": "What is 15 × 23?", "answer": "345"},
  {"question": "If x + 5 = 12, what is x?", "answer": "x = 7"}
]
```

2. Train:
```bash
python train_lightning.py --dataset_path your_data.json
```

### Resume Training

```bash
python train_lightning.py --resume ./checkpoints/last.ckpt
```

## How It Works

### 1. Latent Attention Compression (Perceiver-Style)

```python
class LatentAttentionCompressor(nn.Module):
    def __init__(self, hidden_size: int, num_latents: int):
        # Learned latent queries
        self.latent_queries = nn.Parameter(torch.randn(num_latents, hidden_size))
        
        # Cross-attention for compression
        self.compress_attn = nn.MultiheadAttention(...)
    
    def forward(self, x, attention_mask):
        # Latents attend to input sequence (variable length!)
        latents = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
        compressed = self.compress_attn(query=latents, key=x, value=x)
        return compressed  # [B, 256, D] regardless of input length
```

**Why this works:** 
- **Variable length:** Works with 100 or 65k tokens seamlessly
- **Semantic:** Attention focuses on important information
- **Proven:** Same approach as Perceiver, Q-Former (BLIP-2), Flamingo
- **Efficient:** No expansion network needed with sliding window

### 2. Sliding Window Output

Instead of expanding back to original length, use sliding window:

```python
# After TRM reasoning: [B, M, D] where M=256
# Original hidden states: [B, L, D]

# Sliding window: Drop first M, append M TRM states
output = torch.cat([
    hidden_states[:, M:, :],  # Drop first 256 positions
    trm_reasoning            # Append 256 TRM reasoning states
], dim=1)  # Result: [B, L, D]
```

**Why this works:**
- **No expansion network:** Saves ~1M parameters
- **Natural labels:** Labels align with last positions (TRM outputs)
- **Reasoning as "hidden tokens":** TRM generates states like new tokens
- **Context rarely lost:** First 256 positions usually less critical

### 3. Recursive Reasoning

Three levels of recursion:

**Latent Recursion** (n=6 steps):
```python
for _ in range(n):
    z = net(x + y + z)  # Update reasoning
y = net(y + z)          # Update answer
```

**Deep Recursion** (T=2 iterations):
```python
# T-1 without gradients (fast)
with torch.no_grad():
    y, z = latent_recursion(x, y, z)

# 1 with gradients (for backprop)
y, z = latent_recursion(x, y, z)
```

**Deep Supervision** (N_sup=4 steps):
```python
for step in range(N_sup):
    y, z = deep_recursion(x, y, z)
    compute_loss(y, target)
    y, z = y.detach(), z.detach()  # Carry forward
```

**Total effective depth:** 2 layers × (6+1) × 2 × 4 = **112 layers**

### 4. Training Strategy

1. **Freeze SmolLMv3** - preserve pre-trained knowledge
2. **Train LoRA adapters** - learn when to use `<think>` token
3. **Train TRM** - learn to generate reasoning in hidden space
4. **Train compression** - learn to aggregate sequence info

All trained end-to-end simultaneously! No expansion network needed with sliding window.

## Hyperparameters

Key parameters to tune:

```python
{
    # Compression
    "num_latents": 256,       # 65k → 256 (256x compression, recommended)
                              # Try 128 (512x) if memory constrained
                              # Try 512 (128x) if quality issues
    "compression_heads": 8,   # Attention heads for compression
    
    # TRM
    "n_layers": 2,            # Number of transformer layers
    "n_latent_steps": 4,      # Iterations for reasoning
    "n_deep_recursions": 2,   # T recursions (1 with grad)
    "n_supervision_steps": 4, # Deep supervision steps
    
    # LoRA
    "lora_r": 16,             # LoRA rank
    "lora_alpha": 32,         # LoRA alpha
    
    # Training
    "learning_rate": 2e-4,    # Train end-to-end (compressor + TRM + LoRA)
    "batch_size": 4,          # Smaller due to 65k context
    "gradient_clip": 1.0
}
```

**Effective depth per step:** `T × (n + 1) × n_layers`  
**Example:** 2 × 5 × 2 = 20 layers per supervision step

## Dataset Format

For reasoning tasks:

```json
{
  "question": "What is 15 × 23?",
  "answer": "345"
}
```

Format as:
```
Question: {question}
Answer: <think> {answer}
```

The model learns to output `<think>` to trigger TRM reasoning, then generate the answer.

## Tips

1. **Use EMA** - Exponential Moving Average for stability
2. **Clip gradients** - Essential due to recursive nature
3. **Start small** - Begin with GSM8K before AIME
4. **Monitor `<think>`** - Track how often model uses reasoning
5. **Heavy augmentation** - Crucial for small datasets
6. **Train end-to-end** - Compressor + TRM + LoRA together (no pre-training needed)
7. **Monitor compression** - Log reconstruction error, but don't optimize for it
8. **Use 256 latents** - Sweet spot for 65k context (adjust if memory constrained)

## Key Design Decisions

### Sliding Window vs Expansion
- **No expansion network:** TRM outputs directly replace oldest positions via sliding window
- **Saves 1M parameters:** Simpler, faster, more elegant
- **Natural training:** Labels align directly with TRM outputs

### Single `<think>` Token
- **Simple trigger:** `<think>` activates TRM reasoning
- **No end marker:** Model naturally continues after reasoning
- **Hidden space reasoning:** TRM generates reasoning states, not text tokens

## Citation

Original TRM paper:
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
