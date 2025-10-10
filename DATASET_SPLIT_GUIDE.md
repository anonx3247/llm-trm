# GSM8K Dataset Split Strategy

## Overview

The notebook now implements a **proper train/test separation** to prevent data leakage and enable accurate evaluation during development.

## Split Strategy

### Three-Way Split

```
GSM8K Dataset
│
├── Original Training Set (7,473 examples)
│   ├── Our Train Split (80%) → ~5,978 examples
│   │   └── Used for training the model
│   │
│   └── Our Test Split (20%) → ~1,495 examples
│       └── Used for validation during training & testing
│
└── Original Test Set (1,319 examples)
    └── Reserved for final evaluation only
    └── Use sparingly to avoid overfitting to benchmark
```

## Why This Approach?

### ✅ Prevents Data Leakage
- Test set is **completely separate** from training data
- No information from test set leaks into model training
- Valid evaluation during development

### ✅ Proper Validation
- 20% split gives ~1,495 examples for testing
- Large enough for meaningful accuracy metrics
- PyTorch Lightning uses this for validation during training

### ✅ Benchmark Protection
- Original GSM8K test set (1,319 examples) stays untouched
- Use only for final evaluation/reporting
- Prevents overfitting to the benchmark

### ✅ Standard ML Practice
- 80/20 split is industry standard
- Aligns with best practices in ML research
- Makes results more reproducible and trustworthy

## Usage in Notebook

### Cell 7: Create Split
```python
# Load GSM8K
gsm8k_full = load_dataset("gsm8k", "main", split="train")
gsm8k_final_test = load_dataset("gsm8k", "main", split="test")

# Split training set 80/20
total_examples = len(gsm8k_full)
test_size = int(0.2 * total_examples)
train_size = total_examples - test_size

gsm8k_train = gsm8k_full.select(range(train_size))
gsm8k_test = gsm8k_full.select(range(train_size, total_examples))
```

**Result:**
- `gsm8k_train`: 5,978 examples (training)
- `gsm8k_test`: 1,495 examples (testing/validation)
- `gsm8k_final_test`: 1,319 examples (final evaluation)

### Cell 34: Train on Split
```python
trainer, model = train_gsm8k(
    train_dataset=gsm8k_train,  # 80% for training
    test_dataset=gsm8k_test,    # 20% for validation
    ...
)
```

During training:
- **Trains** on `gsm8k_train` (5,978 examples)
- **Validates** on `gsm8k_test` (1,495 examples) after each epoch
- **Checkpoints** best model based on validation loss
- **Early stopping** if validation loss stops improving

### Cell 35: Evaluate on Test Split
```python
# Evaluate on our 20% test split
results = evaluate_gsm8k(
    model.model,
    gsm8k_test,  # 20% split - fair game to use frequently
    max_samples=100
)
```

This gives you:
- Accuracy on held-out test data
- No data leakage concerns
- Can run as often as needed during development

### Cell 36: Final Benchmark (Use Sparingly!)
```python
# Evaluate on official GSM8K test set
final_results = evaluate_gsm8k(
    model.model,
    gsm8k_final_test,  # Official test set - use only for final reporting
    max_samples=100
)
```

**When to use:**
- Final evaluation before publishing results
- Comparing with other papers/models
- Reporting official benchmark numbers

**When NOT to use:**
- During development iterations
- Hyperparameter tuning
- Model architecture experiments
- Frequent testing (causes overfitting to benchmark)

## Comparison: Before vs After

### ❌ Before (No Clear Split)
```python
# Used whatever data was available
# Mix of train/test, unclear separation
# Risk of data leakage
# Can't trust evaluation metrics
```

### ✅ After (Proper 80/20 Split)
```python
# Clear separation: train vs test
# No data leakage
# Valid evaluation metrics
# Benchmark protected
# Follows ML best practices
```

## Expected Performance

### On 20% Test Split
- Should see similar performance to training
- Good indicator of model quality
- ~5-15% improvement with TRM over baseline

### On Official Test Set
- Slightly different distribution (collected separately)
- May be slightly harder/easier
- Use for final benchmarking only

## Tips

### During Development
1. **Always use 20% test split** for evaluation
2. **Monitor validation loss** during training
3. **Iterate freely** - no data leakage concerns
4. **Tune hyperparameters** based on 20% split performance

### For Final Evaluation
1. **Train best model** using optimal hyperparameters
2. **Evaluate once** on official test set
3. **Report both numbers**: 20% split + official test
4. **Don't iterate** based on official test performance

### Good Practice
- ✅ Train on 80% split
- ✅ Validate on 20% split
- ✅ Tune based on 20% split
- ✅ Final eval on official test (once)

### Bad Practice
- ❌ Training on test data
- ❌ Tuning based on official test set
- ❌ Repeated evaluation on benchmark
- ❌ No clear train/test separation

## References

- **GSM8K Paper**: [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
- **Standard Practice**: 80/20 or 70/30 train/test splits
- **Benchmark Guidelines**: Minimize evaluation on official test sets

---

**Summary**: The notebook now follows ML best practices with a clear 80/20 train/test split, protecting the official benchmark while enabling proper development and evaluation.



