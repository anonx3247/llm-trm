# Training Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Login to Weights & Biases

```bash
wandb login
```

Enter your API key from https://wandb.ai/authorize

### 3. Train

```bash
python train_lightning.py
```

That's it! The script will:
- Use sample dataset for testing
- Train for 3 epochs
- Save checkpoints to `./checkpoints/`
- Log metrics to wandb

## Command Line Options

```bash
python train_lightning.py \
  --dataset_path data.json \      # Your dataset (JSON format)
  --batch_size 2 \                 # Batch size per GPU
  --num_epochs 10 \                # Number of epochs
  --learning_rate 2e-4 \           # Learning rate
  --num_latents 256 \              # 256 for 65k context, 128 for less memory
  --accumulate_grad_batches 4 \    # Gradient accumulation
  --devices 1 \                    # Number of GPUs
  --precision bf16-mixed \         # Mixed precision (bf16 or 16)
  --wandb_project "my-project" \   # Wandb project name
  --wandb_name "experiment-1" \    # Run name
  --output_dir ./checkpoints \     # Where to save models
  --resume ./checkpoints/last.ckpt # Resume from checkpoint
```

## Dataset Format

Create a JSON file with this format:

```json
[
  {
    "question": "What is 15 × 23?",
    "answer": "345"
  },
  {
    "question": "If x + 5 = 12, what is x?",
    "answer": "x = 7"
  }
]
```

The training script will automatically format as:
```
Question: {question}
Answer: <think> {answer}
```

## Using Jupyter Notebook

### Locally

```bash
jupyter notebook train_notebook.ipynb
```

### On Google Colab

1. Upload `train_notebook.ipynb` to Google Drive
2. Open with Google Colab
3. Enable GPU: Runtime → Change runtime type → GPU
4. Run cells sequentially

## Training on Colab Free Tier

T4 GPU (15GB VRAM):

```python
config = {
    "batch_size": 1,                  # Reduce to 1
    "num_latents": 128,               # Reduce to 128
    "accumulate_grad_batches": 8,     # Increase to 8
    "precision": "16-mixed",          # Use 16-bit instead of bfloat16
}
```

## Monitoring Training

### Weights & Biases

The training script logs to wandb automatically. View:
- Training/validation loss
- Learning rate schedule
- GPU usage
- Model checkpoints

### Checkpoints

Saved to `./checkpoints/`:
- `smollm-trm-epoch=XX-val/loss=X.XX.ckpt` - Top 3 models
- `last.ckpt` - Most recent checkpoint

## Resume Training

```bash
python train_lightning.py --resume ./checkpoints/last.ckpt
```

## Load Trained Model

```python
from train_lightning import SmolLMTRMLightningModule

# Load from checkpoint
model = SmolLMTRMLightningModule.load_from_checkpoint(
    "./checkpoints/best.ckpt"
)
model.eval()

# Generate
output = model.model.generate_with_thinking(
    "Question: What is 25 × 17?\nAnswer: <think>",
    max_new_tokens=64
)
print(output)
```

## Tips

### Memory Issues

If you run out of memory:
1. Reduce `batch_size` (try 1)
2. Reduce `num_latents` (try 128)
3. Increase `accumulate_grad_batches` (try 8 or 16)
4. Reduce sequence length in dataset

### Training Not Improving

If validation loss not decreasing:
1. Check if `<think>` token is being used
2. Increase learning rate (try 5e-4)
3. Reduce `trm_loss_weight` (in code, default 0.3)
4. Try more training data

### Slow Training

To speed up:
1. Use multiple GPUs: `--devices 2`
2. Increase batch size if memory allows
3. Use mixed precision: `--precision bf16-mixed`

### Overfitting

If training loss much lower than validation:
1. Use early stopping (automatic)
2. Add more training data
3. Reduce model size (decrease num_latents)

## Hyperparameters

Recommended starting points:

**For GSM8K (math problems):**
```bash
python train_lightning.py \
  --batch_size 4 \
  --num_epochs 10 \
  --learning_rate 2e-4 \
  --num_latents 256
```

**For MATH (harder problems):**
```bash
python train_lightning.py \
  --batch_size 2 \
  --num_epochs 20 \
  --learning_rate 1e-4 \
  --num_latents 256
```

**For AIME (very hard):**
```bash
python train_lightning.py \
  --batch_size 1 \
  --num_epochs 30 \
  --learning_rate 5e-5 \
  --num_latents 512
```

## Advanced: Multi-GPU Training

Automatic with PyTorch Lightning:

```bash
# Use all available GPUs
python train_lightning.py --devices -1

# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1 python train_lightning.py --devices 2
```

## Troubleshooting

### Import Error

```bash
pip install -r requirements.txt
```

### CUDA Out of Memory

Reduce batch size and num_latents, increase gradient accumulation.

### Wandb Login Issues

```bash
wandb login --relogin
```

### Checkpoint Not Found

Check `--output_dir` path and ensure training ran successfully.

## Next Steps

1. **Prepare your dataset** in JSON format
2. **Start with small dataset** to verify training works
3. **Monitor wandb dashboard** for metrics
4. **Adjust hyperparameters** based on results
5. **Scale to full dataset** once validated

Happy training! 🚀
