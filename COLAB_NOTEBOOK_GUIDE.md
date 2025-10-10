# Colab Notebook Guide

## File: `smollm_trm_colab.ipynb`

This is a **complete, self-contained** Google Colab notebook for training SmolLMv3 with the Tiny Recursive Model (TRM). You don't need any external Python files - everything is included.

## What's Inside

### Setup (Cells 1-6)
- **Cell 1-2**: Environment setup and dependency installation
- **Cell 3**: GPU verification
- **Cell 4-5**: Weights & Biases login
- **Cell 6**: Import core dependencies

### Core TRM Implementation (Cells 7-13)
All the code you'd normally import from `src/` is included:

- **TransformerBlock**: Standard transformer with self-attention
- **TinyRecursiveNetwork**: The core tiny network for reasoning
- **LatentAttentionCompressor**: Perceiver-style compression (variable length → fixed latents)
- **RecursiveReasoningBase**: Base class with recursion logic
- **HiddenStateTRM**: TRM adapted for processing LLM hidden states

### SmolLM Integration (Cells 14-18)
- **SmolLMv3WithTRM**: Complete integration with LoRA support
- **ReasoningDataset**: Dataset class for reasoning tasks
- **SmolLMTRMLightningModule**: PyTorch Lightning training module

### Training & Testing (Cells 19-33)
- **Sample dataset**: 10 math problems for quick testing
- **Training function**: Complete training pipeline with callbacks
- **Configuration**: Easy-to-modify training config
- **Testing**: Generate responses from trained model
- **Model saving/loading**: Persist your trained models

## How to Use

### 1. Open in Google Colab

Upload `smollm_trm_colab.ipynb` to Google Colab or open it directly from GitHub.

### 2. Enable GPU

- Go to Runtime → Change runtime type
- Select GPU (T4, V100, or A100)
- Save

### 3. Run the Cells

Simply run cells in order:

1. **Cells 1-6**: Setup environment
2. **Cell 3**: Login to wandb with your API key
3. **Cells 7-21**: Define all classes and functions
4. **Cell 22-23**: Configure and start training

### 4. Customize

Modify the configuration in Cell 22:

```python
config = {
    "model_name": "HuggingFaceTB/SmolLM3-3B",
    "batch_size": 2,              # Increase with more VRAM
    "num_epochs": 3,
    "learning_rate": 2e-4,
    "num_latents": 256,           # Compression ratio
    "accumulate_grad_batches": 4,
    "precision": "bf16-mixed",
    "devices": 1,
    "wandb_project": "smollm-trm",
    "wandb_name": "my-run-name",  # Change this!
}
```

### 5. Use Your Own Dataset

Replace the sample dataset with your own:

```python
# Instead of using create_sample_dataset()
# Load from file:
config["dataset_path"] = "path/to/your/dataset.json"
```

Expected JSON format:
```json
[
  {"question": "What is 2+2?", "answer": "4"},
  {"question": "What is 15 × 23?", "answer": "345"}
]
```

## Key Features

### Self-Contained
- ✅ No external imports needed
- ✅ All TRM code included
- ✅ All integration code included
- ✅ Complete training pipeline

### Production-Ready
- ✅ PyTorch Lightning for clean training
- ✅ Weights & Biases logging
- ✅ Automatic checkpointing
- ✅ Early stopping
- ✅ Learning rate scheduling
- ✅ Gradient clipping
- ✅ Mixed precision training

### Efficient
- ✅ LoRA for parameter-efficient fine-tuning
- ✅ Latent attention compression
- ✅ Sliding window TRM output
- ✅ Gradient accumulation

## Architecture Overview

```
Input: "Question: What is 15 × 23?\nAnswer: <think> 345"
         ↓
SmolLMv3 processes text → hidden states [B, L, D]
         ↓
<think> token triggers TRM:
  1. Compress: [B, L, D] → [B, 256, D]  (via cross-attention)
  2. Reason: TRM recursive refinement
  3. Sliding window: Drop first 256, append 256 TRM states
         ↓
Model generates " 345" conditioned on TRM reasoning
```

## Training Tips

### Memory Optimization
- Reduce `batch_size` if OOM
- Increase `accumulate_grad_batches` to maintain effective batch size
- Reduce `num_latents` (256 → 128) for less VRAM

### Better Results
- Use real datasets: GSM8K, MATH, AIME
- Increase `num_epochs` (3 → 10+)
- Tune `learning_rate` (try 1e-4 to 5e-4)
- Increase TRM parameters in `trm_kwargs`

### Debugging
- Set `return_all_steps=True` in TRM forward pass
- Check wandb logs for loss curves
- Test with sample prompts (Cell 24)

## Troubleshooting

### "CUDA out of memory"
- Reduce `batch_size` to 1
- Reduce `num_latents` to 128
- Reduce `max_length` to 256

### "LoRA not training"
- Check `model.base_model.print_trainable_parameters()`
- Ensure LoRA modules match model architecture

### "Loss not decreasing"
- Check if `<think>` token is in your data
- Verify dataset format
- Try higher learning rate
- Check wandb logs for gradients

## Next Steps

After training:

1. **Test the model** (Cell 24)
2. **Save checkpoint** (Cell 25)
3. **Upload to HuggingFace Hub** (add your own code)
4. **Evaluate on benchmarks** (GSM8K, MATH)
5. **Fine-tune on your domain**

## Support

- Check the main README for architecture details
- See TRAINING_GUIDE.md for advanced tips
- Open an issue if you find bugs

---

**Happy training! 🚀**



