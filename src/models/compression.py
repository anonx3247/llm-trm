"""
Dimension Compression for TRM

MLA-inspired linear compression that reduces hidden state dimensionality
while preserving information through weight-tied symmetric compression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DimensionCompressor(nn.Module):
    """
    Linear dimension compression inspired by DeepSeek MLA.

    Compresses hidden states from D to D' dimensions using a simple linear projection.
    Uses weight-tied (symmetric) decompression for faster convergence and stability.

    Compression: [B, L, D] -> [B, L, D'] where D' << D

    Benefits:
    - Simple and efficient (just linear ops)
    - Weight-tied decompression ensures symmetry
    - Preserves sequence length L
    - Easy to train with reconstruction loss
    """

    def __init__(self, d_model: int = 3072, d_compressed: int = 256):
        """
        Args:
            d_model: Original hidden dimension (e.g., 3072 for SmolLM3-3B)
            d_compressed: Compressed dimension (e.g., 256 for 12x compression)
        """
        super().__init__()

        self.d_model = d_model
        self.d_compressed = d_compressed

        # Single linear layer for compression (no bias for cleaner transpose)
        self.compress = nn.Linear(d_model, d_compressed, bias=False)

        # Initialize with orthogonal initialization for better reconstruction
        nn.init.orthogonal_(self.compress.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress hidden states.

        Args:
            x: [B, L, D] - Input hidden states

        Returns:
            compressed: [B, L, D'] - Compressed hidden states
        """
        result: torch.Tensor = self.compress(x)
        return result

    def decompress(self, x_compressed: torch.Tensor) -> torch.Tensor:
        """
        Decompress hidden states using weight-tied transpose.

        Args:
            x_compressed: [B, L, D'] - Compressed hidden states

        Returns:
            reconstructed: [B, L, D] - Reconstructed hidden states
        """
        # Weight-tied: decompress.weight = compress.weight.T
        return F.linear(x_compressed, self.compress.weight.T)

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss for training.

        Args:
            x: [B, L, D] - Original hidden states

        Returns:
            loss: MSE between original and reconstructed
        """
        compressed = self.forward(x)
        reconstructed = self.decompress(compressed)
        return F.mse_loss(reconstructed, x)

    @property
    def compression_ratio(self) -> float:
        """Return the compression ratio D / D'"""
        return self.d_model / self.d_compressed
