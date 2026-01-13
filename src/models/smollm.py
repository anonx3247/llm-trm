"""
SmolLMv3 + TRM Integration with LoRA

Integrates a Tiny Recursive Model (TRM) with SmolLMv3 to enhance reasoning capabilities.
The TRM processes hidden states when a <think> token is encountered, using a sliding
window approach where TRM reasoning states replace the oldest positions.
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Import base components
from src.models.trm import TinyRecursiveNetwork, RecursiveReasoningBase
from src.models.compression import LatentAttentionCompressor


class HiddenStateTRM(RecursiveReasoningBase):
    """
    TRM adapted for processing LLM hidden states with sliding window output.

    Takes hidden states from SmolLMv3, compresses them, refines through TRM,
    then uses sliding window: truncate first M positions, append M TRM states.

    Result: [B, L, D] -> compress -> [B, M, D] -> TRM -> sliding window -> [B, L, D]

    The TRM reasoning states act as "new hidden states" appended to the sequence,
    similar to generating new tokens but in hidden space.

    Inherits ALL recursion logic from RecursiveReasoningBase - no duplication!
    """

    def __init__(
        self,
        hidden_size: int = 3072,  # SmolLMv3-3B hidden size
        num_latents: int = 256,  # Compress sequence L -> M (recommended 256 for 65k context)
        n_layers: int = 2,
        n_heads: int = 8,
        compression_heads: int = 8,  # Attention heads for compression
        n_latent_steps: int = 6,
        n_deep_recursions: int = 3,
        n_supervision_steps: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_latents = num_latents
        self.n_latent_steps = n_latent_steps
        self.n_deep_recursions = n_deep_recursions
        self.n_supervision_steps = n_supervision_steps

        # Latent attention compression (handles variable-length inputs)
        self.compressor = LatentAttentionCompressor(
            hidden_size=hidden_size,
            num_latents=num_latents,
            n_heads=compression_heads,
            dropout=dropout
        )

        # Reuse TinyRecursiveNetwork from base implementation
        # Operates on compressed sequence [B, M, D]
        self.net = TinyRecursiveNetwork(
            d_model=hidden_size,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
        )

        # Halting mechanism (required by base class)
        self.halt_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_steps: bool = False
    ) -> torch.Tensor:
        """
        Process hidden states through TRM with sliding window output.

        Flow:
        1. [B, L, D] -> compress -> [B, M, D]
        2. [B, M, D] -> TRM reasoning -> [B, M, D]
        3. Sliding window: drop first M from input, append M TRM states -> [B, L, D]

        Args:
            hidden_states: [batch, seq_len, hidden_size] from SmolLMv3 (variable length)
            attention_mask: [batch, seq_len] - Mask for padding tokens
            return_all_steps: Return intermediate refined states (for debugging)

        Returns:
            shifted_states: [batch, seq_len, hidden_size] with sliding window applied
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compress sequence: [B, L, D] -> [B, M, D]
        # Handles variable L via cross-attention
        x_compressed = self.compressor(hidden_states, attention_mask=attention_mask)

        # Initialize y and z in compressed space
        y = torch.zeros_like(x_compressed)
        z = torch.zeros_like(x_compressed)

        all_outputs = []

        # Deep supervision loop - uses inherited run_deep_recursion()
        for step in range(self.n_supervision_steps):
            # Use inherited deep recursion method
            # Operates on compressed space [B, M, D]
            y, z = self.run_deep_recursion(x_compressed, y, z, with_gradients=True)

            if return_all_steps:
                # For debugging: apply sliding window and track
                shifted = torch.cat([
                    hidden_states[:, self.num_latents:, :],
                    y
                ], dim=1)
                all_outputs.append(shifted)

            # Check if we should halt
            if not self.training:
                halt_prob = self.compute_halt_probability(y)
                if halt_prob.mean() > 0.5:
                    break

            # Detach for next iteration
            y = y.detach()
            z = z.detach()

        # Apply sliding window:
        # - Truncate first M positions from input
        # - Append M TRM reasoning states
        # Result: [B, L, D] (same shape as input)
        shifted_states = torch.cat([
            hidden_states[:, self.num_latents:, :],  # [B, L-M, D] - drop first M
            y                                         # [B, M, D] - append TRM reasoning
        ], dim=1)

        if return_all_steps:
            return all_outputs
        return shifted_states


class SmolLMv3WithTRM(nn.Module):
    """
    SmolLMv3 integrated with TRM for enhanced reasoning.

    When <think> token is encountered, hidden states are processed through TRM
    using a sliding window approach. The TRM generates M reasoning states that
    replace the oldest M positions in the sequence.

    Example flow:
        Input:  "What is 15 x 23? <think> 345"
        At <think>: Trigger TRM reasoning in hidden space
        TRM generates M hidden states (reasoning steps)
        Model generates " 345" conditioned on TRM reasoning
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM3-3B",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        num_latents: int = 256,  # For 65k context, use 256 latents (256x compression)
        trm_kwargs: Optional[dict] = None
    ):
        super().__init__()

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # Apply LoRA if requested
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none"
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

        # Initialize TRM with latent attention compression
        trm_kwargs = trm_kwargs or {}
        self.trm = HiddenStateTRM(
            hidden_size=hidden_size,
            num_latents=num_latents,
            **trm_kwargs
        )

        # Freeze base model (except LoRA adapters if used)
        if not use_lora:
            for param in self.base_model.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_trm: bool = True
    ):
        """
        Forward pass with optional TRM processing using sliding window.

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
            return_dict=True
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

        # Process through TRM with sliding window
        # Returns: [B, L, D] with first M truncated, last M are TRM reasoning states
        shifted_states = self.trm(hidden_states, attention_mask=attention_mask)

        # Compute logits from shifted states
        # The last M positions now contain TRM reasoning!
        trm_logits = self.base_model.lm_head(shifted_states)

        if labels is not None:
            # Ensure dimensions match: TRM logits should match the shifted labels
            shifted_labels = labels[:, self.trm.num_latents:]
            # Make sure TRM logits and shifted labels have the same sequence length
            if trm_logits.size(1) != shifted_labels.size(1):
                # If dimensions don't match, adjust TRM logits to match shifted labels
                trm_logits = trm_logits[:, :shifted_labels.size(1), :]

            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            trm_loss = loss_fct(
                trm_logits.reshape(-1, trm_logits.size(-1)),
                shifted_labels.reshape(-1)
            )

            # Combine with base loss
            outputs.loss = outputs.loss + 0.3 * trm_loss

        return outputs

    def generate_with_thinking(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """
        Generate text with TRM-enhanced reasoning.

        The model should learn to output <think> when it needs to reason,
        then generate the answer conditioned on TRM reasoning states.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.base_model.device)

        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=False)


def create_smollm_trm_model(
    model_name: str = "HuggingFaceTB/SmolLM3-3B",
    use_lora: bool = True,
    lora_r: int = 16,
    num_latents: int = 256,
    **kwargs
) -> SmolLMv3WithTRM:
    """
    Factory function to create SmolLMv3 + TRM model with latent attention compression
    and sliding window output.

    Args:
        model_name: HuggingFace model name
        use_lora: Whether to apply LoRA adapters
        lora_r: LoRA rank
        num_latents: Number of latents for compression
                     - 256 recommended for 65k context (256x compression)
                     - 128 for memory-constrained setups (512x compression)
                     - 512 if quality issues (128x compression)
        **kwargs: Additional arguments for TRM

    Returns:
        Integrated model with latent attention compression and sliding window
    """
    return SmolLMv3WithTRM(
        model_name=model_name,
        use_lora=use_lora,
        lora_r=lora_r,
        num_latents=num_latents,
        **kwargs
    )
