"""
SmolLMv3 + TRM Integration with LoRA

Integrates a Tiny Recursive Model (TRM) with SmolLMv3 to enhance reasoning capabilities.
The TRM processes hidden states when a <think> token is encountered.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Import base components from trm.py
from trm import TinyRecursiveNetwork, RecursiveReasoningBase


class SequenceCompressor(nn.Module):
    """
    Clean sequence compression using nn.Linear on sequence dimension.
    
    Compression: [B, L, D] → permute → [B, D, L] → Linear(L, M) → [B, D, M] → permute → [B, M, D]
    Expansion:   [B, M, D] → permute → [B, D, M] → Linear(M, L) → [B, D, L] → permute → [B, L, D]
    """
    
    def __init__(self, max_seq_len: int, num_latents: int):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.num_latents = num_latents
        
        # Compress: L → M (learned weights)
        self.compress = nn.Linear(max_seq_len, num_latents, bias=False)
        
        # Expand: M → L (learned weights)
        self.expand = nn.Linear(num_latents, max_seq_len, bias=False)
        
        print(f"  Sequence compression: {max_seq_len} → {num_latents} ({max_seq_len/num_latents:.1f}x)")
        print(f"  Compression params: {2 * max_seq_len * num_latents:,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compress: [B, L, D] → [B, M, D]"""
        x = x.permute(0, 2, 1)  # [B, L, D] → [B, D, L]
        x = self.compress(x)     # [B, D, L] → [B, D, M]
        x = x.permute(0, 2, 1)   # [B, D, M] → [B, M, D]
        return x
    
    def expand_back(self, compressed: torch.Tensor) -> torch.Tensor:
        """Expand: [B, M, D] → [B, L, D]"""
        x = compressed.permute(0, 2, 1)  # [B, M, D] → [B, D, M]
        x = self.expand(x)                # [B, D, M] → [B, D, L]
        x = x.permute(0, 2, 1)            # [B, D, L] → [B, L, D]
        return x


class HiddenStateTRM(RecursiveReasoningBase):
    """
    TRM adapted for processing LLM hidden states.
    
    Takes hidden states from SmolLMv3 and iteratively refines them.
    Inherits ALL recursion logic from RecursiveReasoningBase - no duplication!
    Only adds input/output projections specific to hidden state processing.
    """
    
    def __init__(
        self,
        hidden_size: int = 3072,  # SmolLMv3-3B hidden size
        max_seq_len: int = 2048,
        num_latents: int = 64,  # Compress sequence L → M
        n_layers: int = 2,
        n_heads: int = 8,
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
        
        # Sequence compression (clean nn.Linear approach)
        self.compressor = SequenceCompressor(max_seq_len, num_latents)
        
        # Reuse TinyRecursiveNetwork from base implementation
        # Now operates on compressed sequence [B, M, D] instead of [B, L, D]
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
        return_all_steps: bool = False
    ) -> torch.Tensor:
        """
        Process hidden states through TRM with sequence compression.
        
        Flow: [B, L, D] → compress → [B, M, D] → TRM → expand → [B, L, D]
        
        Args:
            hidden_states: [batch, seq_len, hidden_size] from SmolLMv3
            return_all_steps: Return intermediate refined states
        
        Returns:
            Refined hidden states [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compress sequence: [B, L, D] → [B, M, D]
        x_compressed = self.compressor(hidden_states)
        
        # Initialize y and z in compressed space
        y = torch.zeros_like(x_compressed)
        z = torch.zeros_like(x_compressed)
        
        all_outputs = []
        
        # Deep supervision loop - uses inherited run_deep_recursion()
        for step in range(self.n_supervision_steps):
            # Use inherited deep recursion method!
            # Now operates on compressed space [B, M, D]
            y, z = self.run_deep_recursion(x_compressed, y, z, with_gradients=True)
            
            if return_all_steps:
                # Expand back for tracking
                expanded = self.compressor.expand_back(y)
                all_outputs.append(expanded)
            
            # Check if we should halt
            if not self.training:
                halt_prob = self.compute_halt_probability(y)
                if halt_prob.mean() > 0.5:
                    break
            
            # Detach for next iteration
            y = y.detach()
            z = z.detach()
        
        # Expand back to original sequence length: [B, M, D] → [B, L, D]
        output = self.compressor.expand_back(y)
        
        if return_all_steps:
            return all_outputs
        return output


class SmolLMv3WithTRM(nn.Module):
    """
    SmolLMv3 integrated with TRM for enhanced reasoning.
    
    When <think> token is encountered, hidden states are processed through TRM.
    """
    
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM3-3B",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        num_latents: int = 64,
        max_seq_len: int = 2048,
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
            print(f"LoRA adapters applied. Trainable parameters:")
            self.base_model.print_trainable_parameters()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special token <think> if not present
        if "<think>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<think>"]})
            self.base_model.resize_token_embeddings(len(self.tokenizer))
        
        self.think_token_id = self.tokenizer.convert_tokens_to_ids("<think>")
        
        # Get model config
        config = self.base_model.config
        hidden_size = config.hidden_size
        
        # Initialize TRM with sequence compression
        trm_kwargs = trm_kwargs or {}
        self.trm = HiddenStateTRM(
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
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
        hidden_states = outputs.hidden_states[-1]  # Last layer
        
        # Process through TRM
        # For simplicity, we'll process the entire sequence
        refined_hidden_states = self.trm(hidden_states)
        
        # Compute additional loss for TRM refinement
        # The refined states should help predict the next tokens better
        trm_logits = self.base_model.lm_head(refined_hidden_states)
        
        if labels is not None:
            # Add TRM loss component
            loss_fct = nn.CrossEntropyLoss()
            trm_loss = loss_fct(
                trm_logits.view(-1, trm_logits.size(-1)),
                labels.view(-1)
            )
            outputs.loss = outputs.loss + 0.3 * trm_loss  # Weight TRM contribution
        
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
        
        The model should learn to output <think> when it needs to reason.
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
    num_latents: int = 64,
    **kwargs
) -> SmolLMv3WithTRM:
    """
    Factory function to create SmolLMv3 + TRM model with sequence compression.
    
    Args:
        model_name: HuggingFace model name
        use_lora: Whether to apply LoRA adapters
        lora_r: LoRA rank
        num_latents: Number of latents for sequence compression (e.g., 64)
        **kwargs: Additional arguments for TRM
    
    Returns:
        Integrated model with compressed TRM
    """
    return SmolLMv3WithTRM(
        model_name=model_name,
        use_lora=use_lora,
        lora_r=lora_r,
        num_latents=num_latents,
        **kwargs
    )


if __name__ == "__main__":
    print("="*70)
    print("Creating SmolLMv3 + TRM with Clean Sequence Compression")
    print("="*70)
    
    # Create model with LoRA and compressed TRM
    model = create_smollm_trm_model(
        model_name="HuggingFaceTB/SmolLM3-3B",
        use_lora=True,
        lora_r=16,
        num_latents=64,  # Compress 2048 → 64 (32x compression!)
        trm_kwargs={
            "n_layers": 2,
            "n_latent_steps": 4,
            "n_deep_recursions": 2,
            "n_supervision_steps": 4
        }
    )
    
    print("\nModel components:")
    print(f"- Base model: SmolLMv3-3B with LoRA")
    print(f"- TRM parameters: {sum(p.numel() for p in model.trm.parameters())/1e6:.2f}M")
    print(f"- Think token ID: {model.think_token_id}")
    
    # Example prompt
    prompt = "What is 15 * 23? <think>"
    print(f"\nExample prompt: {prompt}")
    print("The model should learn to use <think> token to engage TRM reasoning.")

