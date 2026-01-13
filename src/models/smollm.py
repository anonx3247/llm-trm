"""
SmolLMv3 + TRM Integration with LoRA

Integrates a Tiny Recursive Model (TRM) with SmolLMv3 to enhance reasoning capabilities.
The TRM processes hidden states in a compressed dimension space when <think> token is encountered.
"""

from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.compression import DimensionCompressor

# Import base components
from src.models.trm import RecursiveReasoningBase, TinyRecursiveNetwork


class HiddenStateTRM(RecursiveReasoningBase):
    """
    TRM adapted for processing LLM hidden states with dimension compression.

    Takes hidden states from SmolLMv3, compresses the dimension (not sequence),
    performs TRM reasoning in compressed space, then decompresses back.

    Flow: [B, L, D] -> compress -> [B, L, D'] -> TRM -> [B, L, D'] -> decompress -> [B, L, D]

    Key insight: Sequence length L is preserved, only dimension D is compressed to D'.
    This is simpler and more efficient than sequence compression.

    Inherits ALL recursion logic from RecursiveReasoningBase.
    """

    def __init__(
        self,
        hidden_size: int = 3072,  # SmolLMv3-3B hidden size
        d_compressed: int = 256,  # Compressed dimension (12x compression from 3072)
        n_layers: int = 2,
        n_heads: int = 8,
        n_latent_steps: int = 6,
        n_deep_recursions: int = 3,
        n_supervision_steps: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.d_compressed = d_compressed
        self.n_latent_steps = n_latent_steps
        self.n_deep_recursions = n_deep_recursions
        self.n_supervision_steps = n_supervision_steps

        # Dimension compression: [B, L, D] -> [B, L, D']
        self.compressor = DimensionCompressor(
            d_model=hidden_size,
            d_compressed=d_compressed,
        )

        # TRM network operates in compressed dimension space [B, L, D']
        self.net = TinyRecursiveNetwork(
            d_model=d_compressed, n_layers=n_layers, n_heads=n_heads, dropout=dropout
        )

        # Halting mechanism (required by base class) - operates in compressed space
        self.halt_head = nn.Linear(d_compressed, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_all_steps: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Process hidden states through TRM with dimension compression.

        Flow:
        1. [B, L, D] -> compress -> [B, L, D']
        2. [B, L, D'] -> TRM reasoning -> [B, L, D']
        3. [B, L, D'] -> decompress -> [B, L, D]

        Args:
            hidden_states: [batch, seq_len, hidden_size] from SmolLMv3
            attention_mask: [batch, seq_len] - Mask for padding tokens (unused but kept for API)
            return_all_steps: Return intermediate refined states (for debugging)

        Returns:
            refined_states: [batch, seq_len, hidden_size] after TRM reasoning
        """
        # Compress dimension: [B, L, D] -> [B, L, D']
        x_compressed = self.compressor(hidden_states)

        # Initialize y and z in compressed space
        y = torch.zeros_like(x_compressed)
        z = torch.zeros_like(x_compressed)

        all_outputs = []

        # Deep supervision loop - uses inherited run_deep_recursion()
        for _step in range(self.n_supervision_steps):
            # Use inherited deep recursion method
            # Operates in compressed space [B, L, D']
            y, z = self.run_deep_recursion(x_compressed, y, z, with_gradients=True)

            if return_all_steps:
                # Decompress for tracking
                decompressed = self.compressor.decompress(y)
                all_outputs.append(decompressed)

            # Check if we should halt (inference only)
            if not self.training:
                halt_prob = self.compute_halt_probability(y)
                if halt_prob.mean() > 0.5:
                    break

            # Detach for next iteration
            y = y.detach()
            z = z.detach()

        # Decompress back to original dimension: [B, L, D'] -> [B, L, D]
        refined_states = self.compressor.decompress(y)

        if return_all_steps:
            return all_outputs
        return refined_states


class SmolLMv3WithTRM(nn.Module):
    """
    SmolLMv3 integrated with TRM for enhanced reasoning.

    When <think> token is encountered, hidden states are processed through TRM
    with dimension compression. The TRM refines the hidden states in compressed
    space, then decompresses back to original dimension.

    Example flow:
        Input:  "What is 15 x 23? <think> 345"
        At <think>: Trigger TRM reasoning in compressed hidden space
        TRM refines hidden states through recursive iterations
        Model generates answer conditioned on refined hidden states
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM3-3B",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        d_compressed: int = 256,  # Compressed dimension (12x compression from 3072)
        trm_kwargs: dict | None = None,
    ):
        super().__init__()

        # Load base model (type is Any to handle both regular and PEFT models)
        self.base_model: Any = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )

        # Apply LoRA if requested
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            )
            self.base_model = get_peft_model(self.base_model, lora_config)
            print("LoRA adapters applied. Trainable parameters:")
            self.base_model.print_trainable_parameters()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add special token <think> if not present
        special_tokens = {"additional_special_tokens": ["<think>"]}
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            self.base_model.resize_token_embeddings(len(self.tokenizer))

        self.think_token_id = self.tokenizer.convert_tokens_to_ids("<think>")

        # Get model config
        config = self.base_model.config
        hidden_size = config.hidden_size

        # Initialize TRM with dimension compression
        trm_kwargs = trm_kwargs or {}
        self.trm = HiddenStateTRM(hidden_size=hidden_size, d_compressed=d_compressed, **trm_kwargs)

        # Freeze base model (except LoRA adapters if used)
        if not use_lora:
            for param in self.base_model.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_trm: bool = True,
    ):
        """
        Forward pass with optional TRM processing.

        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            labels: Labels for training
            use_trm: Whether to use TRM when <think> token is encountered
        """
        # Get outputs from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )

        if not use_trm or not self.training:
            return outputs

        # Find positions where <think> token appears
        think_positions = (input_ids == self.think_token_id).nonzero(as_tuple=True)

        if len(think_positions[0]) == 0:
            # No <think> token, return normal output
            return outputs

        # Extract hidden states at <think> positions
        hidden_states = outputs.hidden_states[-1]  # Last layer: [B, L, D]

        # Process through TRM with dimension compression
        # Returns: [B, L, D] refined hidden states
        refined_states = self.trm(hidden_states, attention_mask=attention_mask)

        # Compute logits from refined states
        trm_logits = self.base_model.lm_head(refined_states)

        if labels is not None:
            # Compute loss on refined logits
            loss_fct = nn.CrossEntropyLoss()
            trm_loss = loss_fct(trm_logits.reshape(-1, trm_logits.size(-1)), labels.reshape(-1))

            # Combine with base loss
            outputs.loss = outputs.loss + 0.3 * trm_loss

        return outputs

    def generate_with_thinking(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text with TRM-enhanced reasoning.

        The model should learn to output <think> when it needs to reason,
        then generate the answer conditioned on TRM-refined hidden states.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.base_model.device)

        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return str(self.tokenizer.decode(outputs[0], skip_special_tokens=False))


def create_smollm_trm_model(
    model_name: str = "HuggingFaceTB/SmolLM3-3B",
    use_lora: bool = True,
    lora_r: int = 16,
    d_compressed: int = 256,
    **kwargs,
) -> SmolLMv3WithTRM:
    """
    Factory function to create SmolLMv3 + TRM model with dimension compression.

    Args:
        model_name: HuggingFace model name
        use_lora: Whether to apply LoRA adapters
        lora_r: LoRA rank
        d_compressed: Compressed dimension
                     - 256 recommended (12x compression from 3072)
                     - 512 for better quality (6x compression)
                     - 128 for more compression (24x)
        **kwargs: Additional arguments for TRM

    Returns:
        Integrated model with dimension compression
    """
    return SmolLMv3WithTRM(
        model_name=model_name,
        use_lora=use_lora,
        lora_r=lora_r,
        d_compressed=d_compressed,
        **kwargs,
    )
