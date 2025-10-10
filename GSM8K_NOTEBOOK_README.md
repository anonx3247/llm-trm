# GSM8K Training Notebook - Updated

## What's New

### ✅ Proper Train/Test Split (80/20)
The notebook now uses a **proper train/test separation** for development:
- **Training**: 80% of GSM8K train set (~5,978 examples)
- **Test**: 20% of GSM8K train set (~1,495 examples)
- **Final Evaluation**: Original GSM8K test set (1,319 examples) - kept separate for benchmarking

This prevents data leakage and gives you a proper validation/test set during development!

### ✅ Bug Fix: Removed Dead Code
Fixed the issue you spotted! The `shifted_attention_mask` variable was being created but never used. This dead code has been removed from:
- `src/integration/smollm.py` (source code)
- `smollm_trm_colab.ipynb` (notebook)

**Before:**
```python
# We also need to shift the attention mask to match
if attention_mask is not None:
    shifted_mask = attention_mask[:, self.trm.num_latents:]
    trm_mask = torch.ones(...)
    shifted_attention_mask = torch.cat([shifted_mask, trm_mask], dim=1)
else:
    shifted_attention_mask = None

trm_logits = self.base_model.lm_head(shifted_states)  # Never uses shifted_attention_mask!
```

**After:**
```python
# Compute logits from shifted states
trm_logits = self.base_model.lm_head(shifted_states)
```

Since `lm_head` is just a linear projection layer, it doesn't need an attention mask.

## Notebook Structure (39 cells)

### Setup (Cells 1-6)
1. Environment detection and dependency installation
2. GPU verification
3. Weights & Biases login

### GSM8K Dataset (Cell 7)
- Loads GSM8K from HuggingFace datasets
- **Creates 80/20 train/test split** from training data
- Keeps official test set separate for final evaluation
- Shows sample questions and answers

### TRM Components (Cells 8-13)
All core TRM code (same as before):
- TransformerBlock
- TinyRecursiveNetwork
- LatentAttentionCompressor
- RecursiveReasoningBase
- HiddenStateTRM

### SmolLM Integration (Cells 14-17)
- SmolLMv3WithTRM (with bug fix!)
- GSM8KDataset class
- SmolLMTRMLightningModule

### Training & Evaluation (Cells 18-39)
- **train_gsm8k()**: Train on 80% split, validate on 20% split
- **evaluate_gsm8k()**: Compute accuracy on test sets
- **extract_answer()**: Parse numerical answers from text
- Configuration and training execution
- Test set evaluation (20% split) with accuracy metrics
- Final evaluation on official GSM8K test set (optional)
- Individual problem testing
- Model saving

## Quick Start

### 1. Upload to Google Colab
Upload `smollm_trm_colab.ipynb`

### 2. Run Setup Cells (1-6)
Install dependencies and login to wandb

### 3. Configure Training (Cell 33)
```python
config = {
    "train_dataset": gsm8k_train,  # 80% split (~5,978 examples)
    "test_dataset": gsm8k_test,    # 20% split (~1,495 examples)
    "model_name": "HuggingFaceTB/SmolLM3-3B",
    "batch_size": 2,
    "num_epochs": 3,
    "learning_rate": 2e-4,
    "num_latents": 256,
}
```

### 4. Train (Cell 34)
```python
trainer, model = train_gsm8k(**config)
# Trains on 80% split, validates on 20% split
```

### 5. Evaluate on Test Set (Cell 35)
```python
# Evaluate on our 20% test split
results = evaluate_gsm8k(
    model.model,
    gsm8k_test,  # 20% split
    max_samples=100
)
```

### 6. Final Evaluation (Cell 36 - Optional)
```python
# Evaluate on official GSM8K test set (use sparingly!)
final_results = evaluate_gsm8k(
    model.model,
    gsm8k_final_test,  # Official 1,319 test examples
    max_samples=100
)
```

## Key Features

### Smart Dataset Handling
- **Proper split**: 80/20 train/test split from GSM8K training data
- **No data leakage**: Test set is completely separate during training
- **Final evaluation**: Original GSM8K test set kept separate for benchmarking
- **Training set**: ~5,978 examples (80% of 7,473)
- **Test set**: ~1,495 examples (20% of 7,473)

### Evaluation Metrics
- **Accuracy**: Percentage of correct numerical answers
- **Answer extraction**: Regex-based extraction from GSM8K format (`#### 123`)
- **Example tracking**: Saves first 10 predictions for inspection

### GSM8K Data Format
Each example has:
- `question`: "Natalia sold clips to 48 of her friends..."
- `answer`: "She sold 48/2 = 24 clips in May...\n#### 72"

The model learns to:
1. Read the question
2. Trigger reasoning with `<think>` token
3. Generate step-by-step solution
4. Provide final numerical answer

## Expected Performance

### Baseline (no TRM)
SmolLM3-3B typically achieves ~40-50% on GSM8K

### With TRM (this implementation)
Expected improvement of 5-15% depending on:
- Training epochs
- Dataset size
- TRM configuration (num_latents, n_supervision_steps)

### Training Time
- **Quick test** (1000 samples, 3 epochs): ~30-60 minutes on T4 GPU
- **Full dataset** (7473 samples, 3 epochs): ~3-5 hours on T4 GPU
- **Full dataset** (7473 samples, 10 epochs): ~10-15 hours on T4 GPU

## Customization

### Increase TRM Reasoning Steps
```python
trm_kwargs={
    "n_latent_steps": 6,        # 4 → 6 for more reasoning
    "n_deep_recursions": 3,     # 2 → 3 for deeper recursion
    "n_supervision_steps": 8,   # 4 → 8 for more supervision
}
```

### Adjust Compression
```python
num_latents = 128  # Lower for less memory, 512 for more capacity
```

### Tune Learning Rate
```python
learning_rate = 1e-4  # Lower for stability
learning_rate = 5e-4  # Higher for faster convergence
```

## Files

- `smollm_trm_colab.ipynb` - Complete self-contained notebook
- `src/integration/smollm.py` - Source code (bug fixed)
- `COLAB_NOTEBOOK_GUIDE.md` - General usage guide
- `GSM8K_NOTEBOOK_README.md` - This file

## Troubleshooting

### "Accuracy is 0%"
- Check if model is generating answers at all
- Verify `<think>` token is in training data
- Try lowering temperature in generation (0.1 instead of 0.7)
- Check answer extraction regex

### "OOM during training"
- Reduce `batch_size` to 1
- Reduce `num_latents` to 128
- Reduce `max_length` to 256
- Use `accumulate_grad_batches` to maintain effective batch size

### "Training is too slow"
- Use smaller `max_samples` for testing
- Reduce `num_epochs`
- Skip validation: `val_check_interval=1.0`

## Next Steps

1. **Baseline**: Train without TRM (set `use_trm=False`) to compare
2. **Hyperparameter search**: Try different `num_latents`, learning rates
3. **More data**: Use full training set with more epochs
4. **Other datasets**: Adapt for MATH, AIME, or custom problems
5. **Ensemble**: Combine multiple checkpoints for better accuracy

---

Happy training on GSM8K! 🚀📊

