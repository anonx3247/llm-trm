"""
TRM Inference Integration for SmolLM3

Provides inference-time TRM integration where the TRM replaces chain-of-thought
reasoning entirely. When <think> is detected, TRM processes the context and
produces a post-thinking hidden state that is used for all subsequent generation.

Key insight: TRM output = hidden state AFTER </think> (post-reasoning state).
This hidden state encodes all the reasoning and must persist for all future tokens.
"""

import sys
from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.models.compression import DimensionCompressor
from src.train.phase1_compressor import Phase1Config
from src.train.phase2_trm import Phase2Config, SequenceTRM


def load_compressor(
    repo_id: str = "anonx3247/llm-trm-compressor",
    device: str | torch.device = "cuda",
) -> tuple[DimensionCompressor, int, int]:
    """
    Load pretrained compressor from HuggingFace Hub.

    Args:
        repo_id: HuggingFace Hub repository ID
        device: Device to load model to

    Returns:
        Tuple of (compressor, hidden_size, d_compressed)
    """
    print(f"Downloading compressor from HF Hub: {repo_id}...")
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="compressor.pt")

    # Fix pickle compatibility
    sys.modules["__main__"].Phase1Config = Phase1Config  # type: ignore[attr-defined]
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Get config from checkpoint
    config = checkpoint.get("config")
    hidden_size = config.hidden_size if config else 2048
    d_compressed = config.d_compressed if config else 256

    print(f"Compressor config: hidden_size={hidden_size}, d_compressed={d_compressed}")

    # Initialize compressor
    compressor = DimensionCompressor(
        d_model=hidden_size,
        d_compressed=d_compressed,
    )

    # Load state dict (handle torch.compile prefix)
    state_dict = checkpoint["compressor"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    compressor.load_state_dict(state_dict)
    compressor.to(device)
    compressor.eval()

    # Freeze
    for param in compressor.parameters():
        param.requires_grad = False

    print(f"Compressor loaded. Compression ratio: {hidden_size / d_compressed:.1f}x")
    return compressor, hidden_size, d_compressed


def load_trm(
    repo_id: str = "anonx3247/llm-trm-pretrained-trm",
    d_compressed: int = 256,
    device: str | torch.device = "cuda",
) -> SequenceTRM:
    """
    Load pretrained TRM from HuggingFace Hub.

    Args:
        repo_id: HuggingFace Hub repository ID
        d_compressed: Compressed dimension (must match compressor)
        device: Device to load model to

    Returns:
        Loaded SequenceTRM model
    """
    print(f"Downloading TRM from HF Hub: {repo_id}...")
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="trm.pt")

    # Fix pickle compatibility
    sys.modules["__main__"].Phase2Config = Phase2Config  # type: ignore[attr-defined]
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Get config from checkpoint
    config = checkpoint.get("config")
    if config:
        n_layers = getattr(config, "n_layers", 2)
        n_heads = getattr(config, "n_heads", 8)
        n_latent_steps = getattr(config, "n_latent_steps", 6)
        n_deep_recursions = getattr(config, "n_deep_recursions", 3)
    else:
        n_layers, n_heads = 2, 8
        n_latent_steps, n_deep_recursions = 6, 3

    print(
        f"TRM config: d_compressed={d_compressed}, n_layers={n_layers}, "
        f"n_latent={n_latent_steps}, n_deep={n_deep_recursions}"
    )

    # Initialize TRM
    trm = SequenceTRM(
        d_compressed=d_compressed,
        n_layers=n_layers,
        n_heads=n_heads,
        n_latent_steps=n_latent_steps,
        n_deep_recursions=n_deep_recursions,
    )

    # Load state dict (handle different checkpoint formats)
    state_dict = (
        checkpoint.get("trm_state_dict")
        or checkpoint.get("trm")
        or checkpoint.get("ema_trm")
        or checkpoint.get("model")
    )
    if state_dict is None:
        raise ValueError(f"Could not find TRM state dict in checkpoint. Keys: {checkpoint.keys()}")

    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    trm.load_state_dict(state_dict)
    trm.to(device)
    trm.eval()

    # Freeze
    for param in trm.parameters():
        param.requires_grad = False

    print("TRM loaded and frozen.")
    return trm


class SmolLMWithTRMInference(nn.Module):
    """
    SmolLM3 with TRM inference integration.

    When the model generates a <think> token, the TRM intercepts and replaces
    the chain-of-thought with its latent reasoning. The TRM output (post-thinking
    hidden state) is used for subsequent generation.

    Flow:
    1. Generate tokens until <think> is produced
    2. Run TRM on context hidden states (excluding <think>)
    3. TRM output = post-thinking hidden state
    4. Use this for logits AND populate KV cache for future tokens
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM3-3B",
        compressor_repo: str = "anonx3247/llm-trm-compressor",
        trm_repo: str = "anonx3247/llm-trm-pretrained-trm",
        device: str | torch.device | None = None,
        torch_dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device: str | torch.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Auto-select dtype based on device (MPS doesn't support bfloat16 well)
        if torch_dtype is None:
            if self.device == "cuda":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32

        # Load base model
        print(f"Loading base model: {model_name}...")
        if self.device == "cuda":
            self.model: Any = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
            )
        else:
            self.model: Any = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
            )
            self.model = self.model.to(self.device)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Get special token IDs
        # SmolLM3 uses <think> and </think> for thinking mode
        self.think_token_id = self.tokenizer.convert_tokens_to_ids("<think>")
        self.end_think_token_id = self.tokenizer.convert_tokens_to_ids("</think>")

        # Load compressor
        self.compressor, self.hidden_size, self.d_compressed = load_compressor(
            compressor_repo, self.device
        )

        # Load TRM
        self.trm = load_trm(trm_repo, self.d_compressed, self.device)

        # Track if we've done TRM intervention
        self._trm_activated = False

        print("\nSmolLMWithTRMInference initialized:")
        print(f"  - Model: {model_name}")
        print(f"  - Think token ID: {self.think_token_id}")
        print(f"  - Hidden size: {self.hidden_size}")
        print(f"  - Compressed dim: {self.d_compressed}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass with TRM intervention at <think> token.

        When the last token in input_ids is <think>:
        1. Get hidden states for context (excluding <think>)
        2. Run through compressor → TRM → decompressor
        3. TRM output = post-thinking hidden state
        4. Use this for logits AND inject into KV cache
        """
        # Check if we need TRM intervention
        if (
            input_ids.shape[1] > 0
            and input_ids[:, -1].item() == self.think_token_id
            and not self._trm_activated
        ):
            return self._forward_with_trm(input_ids, attention_mask, **kwargs)

        # Normal forward pass
        result: CausalLMOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        return result

    def _forward_with_trm(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass with TRM replacing thinking.

        This method:
        1. Gets hidden states for context (excluding <think>)
        2. Compresses → TRM → Decompresses
        3. Returns logits from TRM output
        4. Sets up KV cache with TRM-enhanced context
        """
        # Get hidden states for everything BEFORE <think>
        context_ids = input_ids[:, :-1]  # Exclude <think>
        context_mask = attention_mask[:, :-1] if attention_mask is not None else None

        # Forward through model to get hidden states
        with torch.no_grad():
            outputs = self.model(
                input_ids=context_ids,
                attention_mask=context_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=True,
            )

        # Get last layer hidden states [B, L, hidden_size]
        assert outputs.hidden_states is not None, "Model must return hidden states"
        hidden_states = outputs.hidden_states[-1]

        # TRM processing
        # Compress: [B, L, 2048] → [B, L, 256]
        compressed = self.compressor(hidden_states.to(self.compressor.compress.weight.dtype))

        # TRM: [B, L, 256] → [B, L+1, 256] (appends reasoning token)
        trm_output = self.trm(compressed, n_steps=8)  # Use 8 supervision steps

        # Decompress: [B, L+1, 256] → [B, L+1, 2048]
        decompressed = self.compressor.decompress(trm_output)

        # The last position (reasoning token) represents post-thinking state
        reasoning_hidden = decompressed[:, -1:, :]  # [B, 1, 2048]

        # Convert to model dtype and get logits
        reasoning_hidden = reasoning_hidden.to(outputs.hidden_states[-1].dtype)
        logits = self.model.lm_head(reasoning_hidden)  # [B, 1, vocab_size]

        # Mark TRM as activated so we don't trigger again
        self._trm_activated = True

        # Return with the original KV cache from context
        # Future tokens will attend to the context, not the TRM output directly
        # (The TRM output is reflected in the logits we return)
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states if kwargs.get("output_hidden_states") else None,
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
        enable_thinking: bool = True,
    ) -> dict[str, Any]:
        """
        Generate text with TRM intervention at <think> token.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample or greedy decode
            enable_thinking: Whether to use thinking mode in the prompt

        Returns:
            Dict with 'text', 'tokens_generated', 'trm_activated'
        """
        # Reset TRM activation flag
        self._trm_activated = False

        # Format prompt with thinking mode
        if enable_thinking:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        else:
            formatted_prompt = prompt

        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Custom generation loop to intercept <think>
        generated_ids = input_ids.clone()
        past_key_values = None
        tokens_generated = 0

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass (will trigger TRM if <think> detected)
                if past_key_values is None:
                    outputs = self.forward(
                        input_ids=generated_ids,
                        attention_mask=attention_mask,
                    )
                else:
                    # Use only the last token when we have KV cache
                    outputs = self.forward(
                        input_ids=generated_ids[:, -1:],
                        past_key_values=past_key_values,
                    )

                past_key_values = outputs.past_key_values
                assert outputs.logits is not None, "Model must return logits"
                logits = outputs.logits[:, -1, :]

                # After TRM activation, mask out <think> and </think> tokens
                # to prevent the model from trying to "think again"
                if self._trm_activated:
                    logits[:, self.think_token_id] = float("-inf")
                    logits[:, self.end_think_token_id] = float("-inf")

                # Sample next token
                if do_sample and temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

                # Check if <think> was just generated - DON'T append it
                # TRM will be activated on next forward pass, and we don't want
                # <think> in the output since TRM replaces the thinking
                if next_token.item() == self.think_token_id and not self._trm_activated:
                    # Append <think> temporarily so forward() can detect it
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)
                    if attention_mask is not None:
                        attention_mask = torch.cat(
                            [attention_mask, torch.ones(1, 1, device=self.device)], dim=1
                        )

                    # Run forward with <think> to trigger TRM
                    outputs = self.forward(
                        input_ids=generated_ids,
                        attention_mask=attention_mask,
                    )

                    # Now remove <think> from generated_ids - TRM replaced it
                    generated_ids = generated_ids[:, :-1]
                    if attention_mask is not None:
                        attention_mask = attention_mask[:, :-1]

                    # Update past_key_values from TRM output
                    past_key_values = outputs.past_key_values

                    # Get next token from TRM output (this is the answer token)
                    assert outputs.logits is not None
                    logits = outputs.logits[:, -1, :]
                    # Mask thinking tokens
                    logits[:, self.think_token_id] = float("-inf")
                    logits[:, self.end_think_token_id] = float("-inf")

                    if do_sample and temperature > 0:
                        probs = torch.softmax(logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = logits.argmax(dim=-1, keepdim=True)

                # Append to generated
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                tokens_generated += 1

                # Update attention mask
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones(1, 1, device=self.device)], dim=1
                    )

                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        # Decode
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)

        return {
            "text": generated_text,
            "tokens_generated": tokens_generated,
            "trm_activated": self._trm_activated,
            "prompt": formatted_prompt,
        }

    def reset(self) -> None:
        """Reset TRM activation state for new generation."""
        self._trm_activated = False


class CompressorOnlyInference(nn.Module):
    """
    SmolLM3 with compressor roundtrip only (no TRM).

    This is for evaluating whether the compressor alone degrades capabilities.
    Hidden states are compressed and decompressed before being passed to lm_head.
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM3-3B",
        compressor_repo: str = "anonx3247/llm-trm-compressor",
        device: str | torch.device | None = None,
        torch_dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device: str | torch.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Auto-select dtype based on device (MPS doesn't support bfloat16 well)
        if torch_dtype is None:
            if self.device == "cuda":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32

        # Load base model
        print(f"Loading base model: {model_name}...")
        if self.device == "cuda":
            self.model: Any = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
            )
        else:
            self.model: Any = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
            )
            self.model = self.model.to(self.device)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load compressor
        self.compressor, self.hidden_size, self.d_compressed = load_compressor(
            compressor_repo, self.device
        )

        print("\nCompressorOnlyInference initialized:")
        print(f"  - Model: {model_name}")
        print(f"  - Compression ratio: {self.hidden_size / self.d_compressed:.1f}x")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass with compressor roundtrip.

        Hidden states are compressed → decompressed before lm_head.
        """
        # Get hidden states
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        # Get last layer hidden states
        assert outputs.hidden_states is not None, "Model must return hidden states"
        hidden_states = outputs.hidden_states[-1]

        # Compress → Decompress roundtrip
        compressed = self.compressor(hidden_states.to(self.compressor.compress.weight.dtype))
        decompressed = self.compressor.decompress(compressed)

        # Get logits from roundtripped hidden states
        decompressed = decompressed.to(hidden_states.dtype)
        logits = self.model.lm_head(decompressed)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
        enable_thinking: bool = True,
    ) -> dict[str, Any]:
        """Generate with compressor roundtrip on every forward pass."""
        # Format prompt
        if enable_thinking:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        else:
            formatted_prompt = prompt

        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        # Generate token by token (no KV cache since we modify hidden states)
        generated_ids = input_ids.clone()
        tokens_generated = 0

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.forward(input_ids=generated_ids)
                assert outputs.logits is not None, "Model must return logits"
                logits = outputs.logits[:, -1, :]

                # Sample next token
                if do_sample and temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                tokens_generated += 1

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)

        return {
            "text": generated_text,
            "tokens_generated": tokens_generated,
            "prompt": formatted_prompt,
        }


# Quick test function
def test_trm_inference() -> None:
    """Quick test of TRM inference on a GSM8K-style problem."""
    print("=" * 60)
    print("Testing TRM Inference")
    print("=" * 60)

    # Initialize model
    model = SmolLMWithTRMInference()

    # Test problem
    problem = "What is 15 + 27?"

    print(f"\nProblem: {problem}")
    print("-" * 40)

    # Generate with TRM
    result = model.generate(
        prompt=problem,
        max_new_tokens=100,
        temperature=0.7,
        enable_thinking=True,
    )

    print(f"Generated text:\n{result['text']}")
    print(f"\nTokens generated: {result['tokens_generated']}")
    print(f"TRM activated: {result['trm_activated']}")


if __name__ == "__main__":
    test_trm_inference()
