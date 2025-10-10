"""
PyTorch Lightning training script for SmolLMv3 + TRM

Features:
- Weights & Biases (wandb) logging
- Automatic checkpointing with model snapshots
- Early stopping
- Multi-GPU support
- Learning rate scheduling
- Gradient accumulation
- Mixed precision training (bfloat16)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Optional
import json

from src.integration import create_smollm_trm_model


class ReasoningDataset(Dataset):
    """
    Dataset for reasoning tasks with <think> token support.
    
    Format: Each example has:
    - question: The problem to solve
    - answer: The final answer
    """
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 512,
        add_think_token: bool = True
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_think_token = add_think_token
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Format the prompt
        question = item['question']
        answer = item['answer']
        
        if self.add_think_token:
            # Add <think> token to trigger TRM reasoning
            text = f"Question: {question}\nAnswer: <think> {answer}"
        else:
            text = f"Question: {question}\nAnswer: {answer}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # Create labels (same as input_ids for causal LM)
        labels = input_ids.clone()
        
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class SmolLMTRMLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for SmolLMv3 + TRM training.
    
    Handles:
    - Training and validation steps
    - Optimizer and scheduler configuration
    - Metric logging to wandb
    - Gradient clipping
    """
    
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM3-3B",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        num_latents: int = 256,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        trm_loss_weight: float = 0.3,
        trm_kwargs: Optional[Dict] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Create model
        self.model = create_smollm_trm_model(
            model_name=model_name,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            num_latents=num_latents,
            trm_kwargs=trm_kwargs or {}
        )
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.trm_loss_weight = trm_loss_weight
        
        # For scheduler
        self.total_steps = None
    
    def forward(self, input_ids, attention_mask, labels):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_trm=True
        )
    
    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        loss = outputs.loss
        
        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/perplexity", torch.exp(loss), on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        loss = outputs.loss
        
        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/perplexity", torch.exp(loss), on_step=False, on_epoch=True, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        # Get trainable parameters
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Setup scheduler
        if self.total_steps is None:
            # Estimate total steps if not set
            self.total_steps = self.trainer.estimated_stepping_batches
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }


def load_dataset(dataset_path: str) -> List[Dict]:
    """
    Load dataset from JSON file.
    
    Expected format:
    [
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "What is 15 × 23?", "answer": "345"},
        ...
    ]
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return data


def create_sample_dataset() -> List[Dict]:
    """
    Create a small sample dataset for testing.
    In practice, load from GSM8K, MATH, AIME, etc.
    """
    return [
        {
            "question": "What is 15 × 23?",
            "answer": "345"
        },
        {
            "question": "If x + 5 = 12, what is x?",
            "answer": "x = 7"
        },
        {
            "question": "What is the sum of the first 10 positive integers?",
            "answer": "The sum is 1+2+3+...+10 = 55"
        },
        {
            "question": "A rectangle has length 8 and width 5. What is its area?",
            "answer": "Area = length × width = 8 × 5 = 40"
        },
        {
            "question": "If 3x - 7 = 14, what is the value of x?",
            "answer": "3x = 21, so x = 7"
        },
        {
            "question": "What is 25 × 17?",
            "answer": "425"
        },
        {
            "question": "If y - 3 = 10, what is y?",
            "answer": "y = 13"
        },
        {
            "question": "What is 144 ÷ 12?",
            "answer": "12"
        },
    ]


def train(
    model_name: str = "HuggingFaceTB/SmolLM3-3B",
    dataset_path: Optional[str] = None,
    output_dir: str = "./checkpoints",
    batch_size: int = 2,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    num_latents: int = 256,
    accumulate_grad_batches: int = 4,
    val_check_interval: float = 0.5,
    precision: str = "bf16-mixed",
    devices: int = 1,
    wandb_project: str = "smollm-trm",
    wandb_name: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None
):
    """
    Main training function.
    
    Args:
        model_name: HuggingFace model name
        dataset_path: Path to JSON dataset file (if None, uses sample data)
        output_dir: Directory to save checkpoints
        batch_size: Batch size per device
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        num_latents: Number of latents for compression (256 for 65k context)
        accumulate_grad_batches: Gradient accumulation steps
        val_check_interval: How often to run validation (fraction of epoch)
        precision: Training precision (bf16-mixed, 16-mixed, or 32)
        devices: Number of GPUs to use
        wandb_project: Weights & Biases project name
        wandb_name: Run name for wandb (optional)
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    
    # Load or create dataset
    if dataset_path:
        data = load_dataset(dataset_path)
        print(f"Loaded {len(data)} examples from {dataset_path}")
    else:
        data = create_sample_dataset()
        print(f"Using sample dataset with {len(data)} examples")
    
    # Split into train/val
    split_idx = int(0.8 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"Train: {len(train_data)} examples, Val: {len(val_data)} examples")
    
    # Initialize model
    pl_module = SmolLMTRMLightningModule(
        model_name=model_name,
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        num_latents=num_latents,
        learning_rate=learning_rate,
        warmup_steps=100,
        trm_kwargs={
            "n_layers": 2,
            "n_latent_steps": 4,
            "n_deep_recursions": 2,
            "n_supervision_steps": 4,
            "compression_heads": 8
        }
    )
    
    # Create datasets
    train_dataset = ReasoningDataset(
        train_data,
        pl_module.model.tokenizer,
        max_length=512,
        add_think_token=True
    )
    
    val_dataset = ReasoningDataset(
        val_data,
        pl_module.model.tokenizer,
        max_length=512,
        add_think_token=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Setup wandb logger
    wandb_logger = WandbLogger(
        project=wandb_project,
        name=wandb_name,
        log_model=True  # Log model checkpoints to wandb
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="smollm-trm-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        patience=3,
        mode="min",
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        devices=devices,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=1.0,
        val_check_interval=val_check_interval,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train
    trainer.fit(
        pl_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_from_checkpoint
    )
    
    print(f"\nTraining complete!")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val loss: {checkpoint_callback.best_model_score:.4f}")
    
    return trainer, pl_module


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SmolLMv3 + TRM with PyTorch Lightning")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM3-3B")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to JSON dataset")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_latents", type=int, default=256)
    parser.add_argument("--accumulate_grad_batches", type=int, default=4)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--wandb_project", type=str, default="smollm-trm")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    train(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        num_latents=args.num_latents,
        accumulate_grad_batches=args.accumulate_grad_batches,
        devices=args.devices,
        precision=args.precision,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        resume_from_checkpoint=args.resume
    )
