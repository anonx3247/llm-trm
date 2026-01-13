"""
LLM-TRM Models

Core model components for Tiny Recursive Models integrated with LLMs.
"""

# Compression
from src.models.compression import DimensionCompressor

# SmolLM integration
from src.models.smollm import HiddenStateTRM, SmolLMv3WithTRM, create_smollm_trm_model

# TRM core components
from src.models.trm import (
    TRM,
    RecursiveReasoningBase,
    TinyRecursiveNetwork,
    TransformerBlock,
    create_trm_model,
)

__all__ = [
    # TRM
    "TransformerBlock",
    "TinyRecursiveNetwork",
    "RecursiveReasoningBase",
    "TRM",
    "create_trm_model",
    # Compression
    "DimensionCompressor",
    # SmolLM
    "HiddenStateTRM",
    "SmolLMv3WithTRM",
    "create_smollm_trm_model",
]
