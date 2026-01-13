"""
Tiny Recursive Model (TRM) - PyTorch Implementation

A simplified recursive reasoning model that recursively improves predictions
through latent reasoning features.

Based on "Less is More: Recursive Reasoning with Tiny Networks" by Alexia Jolicoeur-Martineau
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TransformerBlock(nn.Module):
    """Standard transformer block with self-attention"""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()

        self.norm1 = nn.RMSNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x)
        x = x + residual

        # MLP with residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual

        return x


class TinyRecursiveNetwork(nn.Module):
    """
    The core tiny network used for both latent reasoning and prediction refinement.

    Uses Transformer blocks with self-attention.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class RecursiveReasoningBase(nn.Module):
    """
    Base class for recursive reasoning models.

    Contains the core recursion logic that can be reused by different model types.
    Subclasses just need to define:
    - self.net: The tiny recursive network
    - self.n_latent_steps: Number of latent reasoning steps
    - self.n_deep_recursions: Number of deep recursions
    - self.halt_head: Halting mechanism
    """

    def latent_recursion(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single latent recursion process:
        1. Update z given x, y, z (n times) - latent reasoning
        2. Update y given y, z (once) - refine output answer

        Args:
            x: Input question embeddings [B, L, D]
            y: Current prediction embeddings [B, L, D]
            z: Current latent reasoning embeddings [B, L, D]

        Returns:
            Updated (y, z)
        """
        # Step 1: Recursively update latent z given x, y, z (n times)
        for _ in range(self.n_latent_steps):
            # Combine input, prediction, and latent
            combined = x + y + z
            z = self.net(combined)

        # Step 2: Update prediction y given y, z (once)
        # Note: No x here - this tells the network to refine the answer
        combined = y + z
        y = self.net(combined)

        return y, z

    def run_deep_recursion(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        with_gradients: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Deep recursion with optional gradient computation:
        - Run T-1 recursions without gradients to improve (y, z)
        - Run 1 recursion with gradients for backpropagation

        Args:
            x: Input embeddings [B, L, D]
            y: Current prediction embeddings [B, L, D]
            z: Current latent embeddings [B, L, D]
            with_gradients: Whether to compute gradients for the final recursion

        Returns:
            (y, z): Updated prediction and latent embeddings
        """
        # Run T-1 recursions without gradients (if T > 1)
        if self.n_deep_recursions > 1:
            with torch.no_grad():
                for _ in range(self.n_deep_recursions - 1):
                    y, z = self.latent_recursion(x, y, z)

        # Run final recursion with gradients
        if with_gradients:
            y, z = self.latent_recursion(x, y, z)
        else:
            with torch.no_grad():
                y, z = self.latent_recursion(x, y, z)

        return y, z

    def compute_halt_probability(self, y: torch.Tensor) -> torch.Tensor:
        """Compute halting probability from prediction embeddings"""
        halt_logits = self.halt_head(y.mean(dim=1))  # [B, 1]
        return torch.sigmoid(halt_logits)


class TRM(RecursiveReasoningBase):
    """
    Tiny Recursive Model (TRM)

    Recursively improves predictions through latent reasoning.
    Inherits core recursion logic from RecursiveReasoningBase.

    Args:
        vocab_size: Size of the vocabulary
        d_model: Embedding dimension
        n_layers: Number of layers in the tiny network (default: 2)
        n_heads: Number of attention heads
        dropout: Dropout probability
        n_latent_steps: Number of latent reasoning recursions (n in the paper)
        n_deep_recursions: Number of deep recursions without gradients (T-1 in the paper)
        n_supervision_steps: Maximum number of supervision steps (N_sup in the paper)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.0,
        n_latent_steps: int = 6,
        n_deep_recursions: int = 2,
        n_supervision_steps: int = 16,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_latent_steps = n_latent_steps
        self.n_deep_recursions = n_deep_recursions
        self.n_supervision_steps = n_supervision_steps

        # Input embedding
        self.input_embedding = nn.Embedding(vocab_size, d_model)

        # Single tiny network for both latent reasoning and prediction refinement
        self.net = TinyRecursiveNetwork(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
        )

        # Output head (reverse embedding)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

        # Halting head for early stopping (ACT)
        self.halt_head = nn.Linear(d_model, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights"""
        nn.init.normal_(self.input_embedding.weight, std=0.02)
        nn.init.normal_(self.output_head.weight, std=0.02)
        nn.init.normal_(self.halt_head.weight, std=0.02)

    def deep_recursion(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        with_gradients: bool = True
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Deep recursion with output computation for TRM.

        Returns:
            ((y, z), logits, halt_prob)
        """
        # Run deep recursion from base class
        y, z = self.run_deep_recursion(x, y, z, with_gradients)

        # Compute output logits from prediction embedding
        logits = self.output_head(y)

        # Compute halting probability
        halt_prob = self.compute_halt_probability(y)

        return (y.detach(), z.detach()), logits, halt_prob

    def forward(
        self,
        x_input: torch.Tensor,
        return_all_steps: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through TRM with deep supervision.

        Args:
            x_input: Input token ids [B, L]
            return_all_steps: Whether to return predictions from all supervision steps

        Returns:
            Final logits [B, L, vocab_size] or list of logits if return_all_steps=True
        """
        batch_size, seq_len = x_input.shape

        # Embed input
        x = self.input_embedding(x_input)  # [B, L, D]

        # Initialize prediction and latent embeddings
        y = torch.zeros_like(x)
        z = torch.zeros_like(x)

        all_logits = []

        # Deep supervision loop
        for step in range(self.n_supervision_steps):
            # Perform deep recursion
            (y, z), logits, halt_prob = self.deep_recursion(x, y, z)

            if return_all_steps:
                all_logits.append(logits)

            # Early stopping based on halt probability
            if halt_prob.mean() > 0.5 and not self.training:
                break

        if return_all_steps:
            return all_logits
        else:
            return logits

    def compute_loss(
        self,
        x_input: torch.Tensor,
        y_true: torch.Tensor,
        use_act: bool = True
    ) -> torch.Tensor:
        """
        Compute training loss with deep supervision.

        Args:
            x_input: Input token ids [B, L]
            y_true: Target token ids [B, L]
            use_act: Whether to use Adaptive Computational Time (early stopping)

        Returns:
            Total loss
        """
        batch_size, seq_len = x_input.shape

        # Embed input
        x = self.input_embedding(x_input)  # [B, L, D]

        # Initialize prediction and latent embeddings
        y = torch.zeros_like(x)
        z = torch.zeros_like(x)

        total_loss = 0.0

        # Deep supervision loop
        for step in range(self.n_supervision_steps):
            # Perform deep recursion
            (y, z), logits, halt_prob = self.deep_recursion(x, y, z)

            # Cross-entropy loss
            ce_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y_true.reshape(-1)
            )
            total_loss += ce_loss

            # Halting loss (ACT)
            if use_act:
                # Check if prediction matches target
                y_pred = torch.argmax(logits, dim=-1)
                target_halt = (y_pred == y_true).all(dim=-1).float().unsqueeze(-1)
                halt_loss = F.binary_cross_entropy(halt_prob.clamp(1e-7, 1-1e-7), target_halt)
                total_loss += 0.5 * halt_loss

            # Early stopping during training
            if use_act and halt_prob.mean() > 0.5:
                break

        return total_loss


def create_trm_model(
    vocab_size: int,
    d_model: int = 512,
    **kwargs
) -> TRM:
    """
    Factory function to create a TRM model with sensible defaults.

    Args:
        vocab_size: Size of the vocabulary
        d_model: Embedding dimension
        **kwargs: Additional arguments passed to TRM

    Returns:
        TRM model
    """
    return TRM(
        vocab_size=vocab_size,
        d_model=d_model,
        **kwargs
    )
