"""
Example training script for TRM on a simple task.

This demonstrates how to train a TRM model on a synthetic task.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from src.trm import create_trm_model
from tqdm import tqdm
from typing import Tuple


class SyntheticReasoningDataset(Dataset):
    """
    A simple synthetic dataset for testing TRM.
    Task: Given a sequence of numbers, predict the sequence reversed.
    """
    
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Generate random sequences
        self.inputs = torch.randint(1, vocab_size, (num_samples, seq_len))
        # Target is the reversed sequence
        self.targets = torch.flip(self.inputs, dims=[1])
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


class EMA:
    """Exponential Moving Average for model parameters"""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Apply EMA parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    ema: EMA = None,
    device: str = "cpu",
    use_act: bool = True
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Compute loss with deep supervision
        loss = model.compute_loss(x, y, use_act=use_act)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Update EMA
        if ema is not None:
            ema.update()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu"
) -> Tuple[float, float]:
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    pbar = tqdm(dataloader, desc="Evaluating")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        logits = model(x)
        
        # Compute loss
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1)
        )
        total_loss += loss.item()
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=-1)
        correct = (preds == y).sum().item()
        total_correct += correct
        total_tokens += y.numel()
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100 * correct / y.numel():.2f}%"
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total_tokens
    
    return avg_loss, avg_acc


def main():
    # Hyperparameters
    vocab_size = 50
    seq_len = 16
    d_model = 128
    n_layers = 2
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-3
    
    # Determine best available device: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = SyntheticReasoningDataset(
        num_samples=1000,
        seq_len=seq_len,
        vocab_size=vocab_size
    )
    
    val_dataset = SyntheticReasoningDataset(
        num_samples=200,
        seq_len=seq_len,
        vocab_size=vocab_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Create model
    model = create_trm_model(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_latent_steps=6,
        n_deep_recursions=3,
        n_supervision_steps=16
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Create EMA
    ema = EMA(model, decay=0.999)
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, ema, device, use_act=True
        )
        print(f"Train loss: {train_loss:.4f}")
        
        # Evaluate with EMA
        ema.apply_shadow()
        val_loss, val_acc = evaluate(model, val_loader, device)
        ema.restore()
        
        print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc*100:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_shadow': ema.shadow,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pt')
            print(f"Saved best model with val acc: {val_acc*100:.2f}%")
    
    print(f"\nTraining complete! Best val acc: {best_val_acc*100:.2f}%")


if __name__ == "__main__":
    main()

