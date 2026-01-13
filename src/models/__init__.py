"""
LLM-TRM Models

Core model components for Tiny Recursive Models integrated with LLMs.
"""

# TRM core components
from src.models.trm import (
    TransformerBlock,
    TinyRecursiveNetwork,
    RecursiveReasoningBase,
    TRM,
    create_trm_model
)

# Compression
from src.models.compression import LatentAttentionCompressor

# SmolLM integration
from src.models.smollm import (
    HiddenStateTRM,
    SmolLMv3WithTRM,
    create_smollm_trm_model
)

__all__ = [
    # TRM
    "TransformerBlock",
    "TinyRecursiveNetwork",
    "RecursiveReasoningBase",
    "TRM",
    "create_trm_model",
    # Compression
    "LatentAttentionCompressor",
    # SmolLM
    "HiddenStateTRM",
    "SmolLMv3WithTRM",
    "create_smollm_trm_model",
]
