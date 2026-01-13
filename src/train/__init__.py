"""
LLM-TRM Training Scripts

Multi-phase training pipeline for integrating TRM with SmolLMv3.

Training Phases:
    Phase 1 (phase1_compressor.py): Compressor pretraining
        - Stage 1a: Identity training on regular hidden states
        - Stage 1b: Finetune on CoT thinking trajectories

    Phase 2 (phase2_trm.py): TRM iteration training
        - Generate hidden state pairs (phase2_datagen.py)
        - Train TRM to map hidden_pre_<thinking> -> hidden_post_</thinking>

    Phase 3 (phase3_grpo.py): GRPO training
        - Freeze LLM, train TRM + compressor
        - Group Relative Policy Optimization on CoT trajectories
"""

from src.train.phase1_compressor import (
    CompressorPretrainer,
    run_phase1_training,
)
from src.train.phase2_datagen import (
    ThinkingDataGenerator,
    generate_hidden_state_pairs,
)
from src.train.phase2_trm import (
    TRMIterationTrainer,
    run_phase2_training,
)
from src.train.phase3_grpo import (
    GRPOTrainer,
    run_phase3_training,
)

__all__ = [
    # Phase 1
    "CompressorPretrainer",
    "run_phase1_training",
    # Phase 2 datagen
    "ThinkingDataGenerator",
    "generate_hidden_state_pairs",
    # Phase 2 training
    "TRMIterationTrainer",
    "run_phase2_training",
    # Phase 3
    "GRPOTrainer",
    "run_phase3_training",
]
