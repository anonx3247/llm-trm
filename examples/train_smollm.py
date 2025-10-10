"""
Training script for SmolLMv3 + TRM on reasoning tasks.

This script demonstrates how to train LoRA adapters and TRM simultaneously
on datasets requiring complex reasoning (like AIME).
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from src.integration import create_smollm_trm_model
from tqdm import tqdm
from typing import Dict, List
import json


class ReasoningDataset(Dataset):
    """
    Dataset for reasoning tasks.
    
    Format: Each example has:
    - question: The problem to solve
    - reasoning: Optional chain of thought
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
            # Add <think> token to encourage TRM usage
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


class EMA:
    """Exponential Moving Average for model parameters"""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters for TRM only
        for name, param in model.named_parameters():
            if param.requires_grad and 'trm' in name:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'trm' in name:
                if name in self.shadow:
                    self.shadow[name] = (
                        self.decay * self.shadow[name] + 
                        (1 - self.decay) * param.data
                    )
    
    def apply_shadow(self):
        """Apply EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'trm' in name:
                if name in self.shadow:
                    self.backup[name] = param.data.clone()
                    param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'trm' in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}


def train_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    ema: EMA = None,
    device: str = "cuda",
    gradient_accumulation_steps: int = 1
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_trm=True
        )
        
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update EMA
            if ema is not None:
                ema.update()
        
        total_loss += outputs.loss.item()
        pbar.set_postfix({
            "loss": f"{outputs.loss.item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })
    
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    device: str = "cuda"
) -> float:
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Evaluating")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_trm=True
        )
        
        total_loss += outputs.loss.item()
        pbar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})
    
    return total_loss / len(dataloader)


def create_sample_dataset() -> List[Dict]:
    """
    Create a small sample dataset for demonstration.
    In practice, you'd load from AIME, GSM8K, MATH, etc.
    """
    return [
        {
            "question": "What is 15 * 23?",
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
        }
    ]


def main():
    # Configuration
    # Determine best available device: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    config = {
        "model_name": "HuggingFaceTB/SmolLM3-3B",
        "use_lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_epochs": 3,
        "learning_rate": 2e-4,
        "warmup_steps": 100,
        "max_length": 256,
        "device": device
    }
    
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Create model
    print("\nCreating model...")
    model = create_smollm_trm_model(
        model_name=config["model_name"],
        use_lora=config["use_lora"],
        lora_r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        trm_kwargs={
            "n_layers": 2,
            "n_latent_steps": 4,
            "n_deep_recursions": 2,
            "n_supervision_steps": 4,
            "dropout": 0.1
        }
    )
    
    # Print trainable parameters
    print("\nTrainable parameters:")
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if 'trm' in name or 'lora' in name.lower():
                print(f"  {name}: {param.numel():,}")
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    # Create datasets
    print("\nCreating datasets...")
    sample_data = create_sample_dataset()
    
    train_dataset = ReasoningDataset(
        data=sample_data,
        tokenizer=model.tokenizer,
        max_length=config["max_length"],
        add_think_token=True
    )
    
    val_dataset = ReasoningDataset(
        data=sample_data[:2],  # Small validation set for demo
        tokenizer=model.tokenizer,
        max_length=config["max_length"],
        add_think_token=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False
    )
    
    # Create optimizer and scheduler
    print("\nSetting up training...")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=0.01
    )
    
    total_steps = len(train_loader) * config["num_epochs"] // config["gradient_accumulation_steps"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=total_steps
    )
    
    # Create EMA
    ema = EMA(model, decay=0.999)
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            ema,
            config["device"],
            config["gradient_accumulation_steps"]
        )
        print(f"Train loss: {train_loss:.4f}")
        
        # Evaluate with EMA
        ema.apply_shadow()
        val_loss = evaluate(model, val_loader, config["device"])
        ema.restore()
        
        print(f"Val loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best! Saving model...")
            
            # Save model
            model.base_model.save_pretrained("./smollm_trm_checkpoint")
            model.tokenizer.save_pretrained("./smollm_trm_checkpoint")
            torch.save({
                'trm_state_dict': model.trm.state_dict(),
                'ema_shadow': ema.shadow,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, "./smollm_trm_checkpoint/trm_components.pt")
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    
    # Test generation
    print("\n" + "="*50)
    print("Testing generation with TRM reasoning:")
    print("="*50)
    
    model.eval()
    test_prompts = [
        "Question: What is 25 * 17?\nAnswer:",
        "Question: If y - 3 = 10, what is y?\nAnswer:",
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        output = model.generate_with_thinking(
            prompt,
            max_new_tokens=64,
            temperature=0.7
        )
        print(f"Output: {output}")


if __name__ == "__main__":
    main()

