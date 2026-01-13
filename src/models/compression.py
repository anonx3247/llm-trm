"""
Latent Attention Compression for TRM

Perceiver-style compression that uses learned latent queries to compress
variable-length sequences via cross-attention.
"""

import torch
import torch.nn as nn
from typing import Optional


class LatentAttentionCompressor(nn.Module):
    """
    Attention-based sequence compression inspired by Perceiver/MLA.

    Uses learned latent queries to compress variable-length sequences via cross-attention.
    Handles sequences of any length (up to max_seq_len) with constant parameters.

    Compression: [B, L, D] -> CrossAttention(latents, sequence) -> [B, M, D]

    Benefits:
    - Variable length inputs (no fixed L)
    - Semantic aggregation via attention
    - Much fewer parameters for long sequences
    - Interpretable (attention weights show what's compressed)
    """

    def __init__(
        self,
        hidden_size: int,
        num_latents: int,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_latents = num_latents
        self.n_heads = n_heads

        # Learned latent queries for compression
        # These act as "summary" tokens that aggregate information
        self.latent_queries = nn.Parameter(torch.randn(num_latents, hidden_size))

        # Compression: latents attend to full sequence
        self.compress_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.compress_norm = nn.LayerNorm(hidden_size)
        self.compress_ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        self.compress_ff_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compress variable-length sequence to fixed number of latents.

        Args:
            x: [B, L, D] - Input sequence (L can vary between batches)
            attention_mask: [B, L] - Optional mask for padding (True = valid, False = padding)

        Returns:
            compressed: [B, M, D] - Fixed number of latent representations
        """
        batch_size, seq_len, _ = x.shape

        # Expand latent queries for batch: [M, D] -> [B, M, D]
        latents = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attention: latents (Q) attend to sequence (K, V)
        # This semantically aggregates information from variable-length input
        # Convert attention_mask to key_padding_mask format (True for positions to ignore)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # True for padding positions

        attn_out, _ = self.compress_attn(
            query=latents,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask
        )
        latents = self.compress_norm(latents + attn_out)

        # Feed-forward for further processing
        ff_out = self.compress_ff(latents)
        latents = self.compress_ff_norm(latents + ff_out)

        return latents
