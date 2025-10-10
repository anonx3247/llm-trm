"""SmolLM3 + TRM integration with LoRA and sequence compression"""

from .smollm import (
    SequenceCompressor,
    HiddenStateTRM,
    SmolLMv3WithTRM,
    create_smollm_trm_model
)

__all__ = [
    'SequenceCompressor',
    'HiddenStateTRM',
    'SmolLMv3WithTRM',
    'create_smollm_trm_model'
]

