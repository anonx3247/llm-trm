"""
TRM Inference Integration for SmolLM3

Provides inference-time TRM integration where the TRM replaces chain-of-thought
reasoning entirely. Uses /no_think mode with TRM enhancement on prompt hidden states.

Key insight: With /no_think mode, the model skips thinking and goes straight to answer.
We enhance the prompt's hidden states with TRM before generation, giving the model
"pre-computed reasoning" in latent space.
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

    Uses /no_think mode and enhances prompt hidden states with TRM before generation.
    The TRM provides "latent reasoning" that the model uses to generate answers directly.

    Flow:
    1. Format prompt with /no_think mode (skips CoT, goes straight to answer)
    2. Get hidden states for the prompt
    3. Run TRM: compress → reasoning → decompress
    4. Generate from TRM-enhanced hidden state
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

        # Load compressor
        self.compressor, self.hidden_size, self.d_compressed = load_compressor(
            compressor_repo, self.device
        )

        # Load TRM
        self.trm = load_trm(trm_repo, self.d_compressed, self.device)

        print("\nSmolLMWithTRMInference initialized:")
        print(f"  - Model: {model_name}")
        print(f"  - Hidden size: {self.hidden_size}")
        print(f"  - Compressed dim: {self.d_compressed}")
        print(f"  - Mode: /no_think + TRM enhancement")

    def _get_trm_enhanced_hidden(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Get TRM-enhanced hidden state for the prompt.

        Args:
            input_ids: Tokenized prompt [B, L]
            attention_mask: Optional attention mask [B, L]

        Returns:
            TRM-enhanced hidden state for the last position [B, 1, hidden_size]
        """
        # Get hidden states from the model
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # Get last layer hidden states [B, L, hidden_size]
        hidden_states = outputs.hidden_states[-1]

        # TRM processing
        # Compress: [B, L, 2048] → [B, L, 256]
        compressed = self.compressor(hidden_states.to(self.compressor.compress.weight.dtype))

        # TRM: [B, L, 256] → [B, L+1, 256] (appends reasoning token)
        trm_output = self.trm(compressed, n_steps=8)

        # Decompress: [B, L+1, 256] → [B, L+1, 2048]
        decompressed = self.compressor.decompress(trm_output)

        # The last position represents the "post-reasoning" state
        reasoning_hidden = decompressed[:, -1:, :]  # [B, 1, 2048]

        # Convert to model dtype
        return reasoning_hidden.to(hidden_states.dtype)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> dict[str, Any]:
        """
        Generate text with TRM-enhanced reasoning.

        Uses /no_think mode - the model skips CoT and goes straight to answer.
        TRM enhancement provides latent reasoning in the hidden state.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample or greedy decode

        Returns:
            Dict with 'text', 'tokens_generated'
        """
        # Format prompt with /no_think mode
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # /no_think mode
        )

        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Get TRM-enhanced hidden state for the prompt
        with torch.no_grad():
            reasoning_hidden = self._get_trm_enhanced_hidden(input_ids, attention_mask)

        # Get first token logits from TRM-enhanced hidden state
        logits = self.model.lm_head(reasoning_hidden)[:, -1, :]

        # Generate tokens
        generated_ids = input_ids.clone()
        tokens_generated = 0

        with torch.no_grad():
            for step in range(max_new_tokens):
                # Sample next token
                if do_sample and temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

                # Append to generated
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                tokens_generated += 1

                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Get next logits (normal forward, no TRM after first token)
                # We only need the last token since we're generating autoregressively
                outputs = self.model(
                    input_ids=generated_ids,
                    attention_mask=None,
                    return_dict=True,
                )
                logits = outputs.logits[:, -1, :]

        # Decode
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)

        return {
            "text": generated_text,
            "tokens_generated": tokens_generated,
            "prompt": formatted_prompt,
        }


class CompressorOnlyInference(nn.Module):
    """
    SmolLM3 with compressor roundtrip only (no TRM).

    Uses /no_think mode and applies compress→decompress once on prompt hidden states.
    This tests whether the compressor alone degrades capabilities.

    Flow:
    1. Format prompt with /no_think mode
    2. Get hidden states for the prompt
    3. Compress → Decompress (roundtrip)
    4. Generate from roundtripped hidden state
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
        print(f"  - Mode: /no_think + compressor roundtrip")

    def _get_compressed_hidden(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Get compressor-roundtripped hidden state for the prompt.

        Args:
            input_ids: Tokenized prompt [B, L]
            attention_mask: Optional attention mask [B, L]

        Returns:
            Roundtripped hidden state for the last position [B, 1, hidden_size]
        """
        # Get hidden states from the model
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # Get last layer hidden states [B, L, hidden_size]
        hidden_states = outputs.hidden_states[-1]

        # Compress → Decompress roundtrip
        compressed = self.compressor(hidden_states.to(self.compressor.compress.weight.dtype))
        decompressed = self.compressor.decompress(compressed)

        # Return last position
        return decompressed[:, -1:, :].to(hidden_states.dtype)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> dict[str, Any]:
        """
        Generate text with compressor-roundtripped prompt.

        Uses /no_think mode - applies compressor once on prompt, then normal generation.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample or greedy decode

        Returns:
            Dict with 'text', 'tokens_generated'
        """
        # Format prompt with /no_think mode
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # /no_think mode
        )

        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Get compressor-roundtripped hidden state for the prompt
        with torch.no_grad():
            roundtripped_hidden = self._get_compressed_hidden(input_ids, attention_mask)

        # Get first token logits from roundtripped hidden state
        logits = self.model.lm_head(roundtripped_hidden)[:, -1, :]

        # Generate tokens
        generated_ids = input_ids.clone()
        tokens_generated = 0

        with torch.no_grad():
            for step in range(max_new_tokens):
                # Sample next token
                if do_sample and temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

                # Append to generated
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                tokens_generated += 1

                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Get next logits (normal forward, no compressor after first token)
                outputs = self.model(
                    input_ids=generated_ids,
                    attention_mask=None,
                    return_dict=True,
                )
                logits = outputs.logits[:, -1, :]

        # Decode
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)

        return {
            "text": generated_text,
            "tokens_generated": tokens_generated,
            "prompt": formatted_prompt,
        }


def load_trm_direct(
    d_model: int = 2048,
    n_layers: int = 2,
    n_heads: int = 8,
    n_latent_steps: int = 6,
    n_deep_recursions: int = 3,
    device: str | torch.device = "cuda",
    checkpoint_path: str | None = None,
) -> SequenceTRM:
    """
    Create a TRM that works directly on full hidden dimension (no compression).

    Args:
        d_model: Hidden dimension (2048 for SmolLM3)
        n_layers: Number of transformer layers in TRM
        n_heads: Number of attention heads
        n_latent_steps: Number of latent recursion steps
        n_deep_recursions: Number of deep recursion steps
        device: Device to load model to
        checkpoint_path: Optional path to pretrained weights

    Returns:
        SequenceTRM model
    """
    print(f"Creating direct TRM (d={d_model}, no compression)...")

    trm = SequenceTRM(
        d_compressed=d_model,  # Use full hidden dim
        n_layers=n_layers,
        n_heads=n_heads,
        n_latent_steps=n_latent_steps,
        n_deep_recursions=n_deep_recursions,
    )

    if checkpoint_path:
        print(f"Loading TRM weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = (
            checkpoint.get("trm_state_dict")
            or checkpoint.get("trm")
            or checkpoint.get("ema_trm")
            or checkpoint.get("model")
            or checkpoint
        )
        if isinstance(state_dict, dict) and "trm_state_dict" not in state_dict:
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        trm.load_state_dict(state_dict)

    trm.to(device)
    trm.eval()

    # Freeze
    for param in trm.parameters():
        param.requires_grad = False

    print(f"Direct TRM created: d={d_model}, layers={n_layers}, heads={n_heads}")
    return trm


class SmolLMWithDirectTRMInference(nn.Module):
    """
    SmolLM3 with TRM working directly on full hidden dimension (no compressor).

    This variant skips compression entirely - TRM operates on 2048-dim hidden states.
    Use this to test if compression is causing information loss.

    Flow:
    1. Format prompt with /no_think mode
    2. Get hidden states for the prompt [B, L, 2048]
    3. Run TRM directly on hidden states [B, L, 2048] → [B, L+1, 2048]
    4. Generate from TRM-enhanced hidden state
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM3-3B",
        trm_checkpoint: str | None = None,
        n_layers: int = 2,
        n_heads: int = 16,  # More heads for larger dim
        n_latent_steps: int = 6,
        n_deep_recursions: int = 3,
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

        # Auto-select dtype based on device
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

        # Get hidden size from model config
        self.hidden_size = self.model.config.hidden_size  # 2048 for SmolLM3

        # Create TRM that works on full hidden dimension
        self.trm = load_trm_direct(
            d_model=self.hidden_size,
            n_layers=n_layers,
            n_heads=n_heads,
            n_latent_steps=n_latent_steps,
            n_deep_recursions=n_deep_recursions,
            device=self.device,
            checkpoint_path=trm_checkpoint,
        )

        print("\nSmolLMWithDirectTRMInference initialized:")
        print(f"  - Model: {model_name}")
        print(f"  - Hidden size: {self.hidden_size}")
        print(f"  - TRM dim: {self.hidden_size} (no compression)")
        print(f"  - Mode: /no_think + direct TRM")

    def _get_trm_enhanced_hidden(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Get TRM-enhanced hidden state for the prompt (no compression).

        Args:
            input_ids: Tokenized prompt [B, L]
            attention_mask: Optional attention mask [B, L]

        Returns:
            TRM-enhanced hidden state for the last position [B, 1, hidden_size]
        """
        # Get hidden states from the model
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # Get last layer hidden states [B, L, 2048]
        hidden_states = outputs.hidden_states[-1]

        # Run TRM directly on hidden states (no compression)
        # TRM: [B, L, 2048] → [B, L+1, 2048]
        trm_input = hidden_states.to(torch.float32)  # TRM expects float32
        trm_output = self.trm(trm_input, n_steps=8)

        # The last position represents the "post-reasoning" state
        reasoning_hidden = trm_output[:, -1:, :]  # [B, 1, 2048]

        # Convert back to model dtype
        return reasoning_hidden.to(hidden_states.dtype)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> dict[str, Any]:
        """
        Generate text with direct TRM-enhanced reasoning (no compression).

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample or greedy decode

        Returns:
            Dict with 'text', 'tokens_generated'
        """
        # Format prompt with /no_think mode
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Get TRM-enhanced hidden state
        with torch.no_grad():
            reasoning_hidden = self._get_trm_enhanced_hidden(input_ids, attention_mask)

        # Get first token logits from TRM-enhanced hidden state
        logits = self.model.lm_head(reasoning_hidden)[:, -1, :]

        # Generate tokens
        generated_ids = input_ids.clone()
        tokens_generated = 0

        with torch.no_grad():
            for step in range(max_new_tokens):
                # Sample next token
                if do_sample and temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

                # Append to generated
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                tokens_generated += 1

                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Get next logits (normal forward, no TRM after first token)
                outputs = self.model(
                    input_ids=generated_ids,
                    attention_mask=None,
                    return_dict=True,
                )
                logits = outputs.logits[:, -1, :]

        # Decode
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
    print("Testing TRM Inference (/no_think + TRM)")
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
        temperature=0.0,  # Greedy for reproducibility
        do_sample=False,
    )

    print(f"Generated text:\n{result['text']}")
    print(f"\nTokens generated: {result['tokens_generated']}")

    # Extract assistant response
    if "<|im_start|>assistant" in result["text"]:
        assistant_response = result["text"].split("<|im_start|>assistant")[-1]
        clean_response = assistant_response.strip()
        print(f"\nAssistant response: {repr(clean_response[:200])}")
    else:
        print("\nCould not find assistant response marker")


def test_comparison() -> None:
    """Compare outputs: base (no think) vs base (think) vs TRM."""
    print("=" * 70)
    print("Comparison Test: Base vs Base+Think vs TRM")
    print("=" * 70)

    problem = "What is 2 + 2?"
    max_tokens = 100

    # Load base model (no TRM)
    print("\n[1/3] Loading base model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "HuggingFaceTB/SmolLM3-3B"
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Test 1: Base model WITHOUT thinking
    print("\n" + "=" * 70)
    print("TEST 1: Base model WITHOUT thinking")
    print("=" * 70)
    messages = [{"role": "user", "content": problem}]
    prompt_no_think = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt_no_think, return_tensors="pt").to(base_model.device)
    with torch.no_grad():
        outputs = base_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response_no_think = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Extract just the assistant response
    if "<|im_start|>assistant" in response_no_think:
        response_no_think = response_no_think.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in response_no_think:
            response_no_think = response_no_think.split("<|im_end|}")[0]
    print(f"Response:\n{response_no_think[:500]}")

    # Test 2: Base model WITH thinking
    print("\n" + "=" * 70)
    print("TEST 2: Base model WITH thinking (native CoT)")
    print("=" * 70)
    prompt_think = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    inputs = tokenizer(prompt_think, return_tensors="pt").to(base_model.device)
    with torch.no_grad():
        outputs = base_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response_think = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if "<|im_start|>assistant" in response_think:
        response_think = response_think.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in response_think:
            response_think = response_think.split("<|im_end|>")[0]
    print(f"Response:\n{response_think[:500]}")

    # Clean up base model
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Test 3: TRM model (/no_think + TRM enhancement)
    print("\n" + "=" * 70)
    print("TEST 3: TRM model (/no_think + TRM enhancement)")
    print("=" * 70)
    trm_model = SmolLMWithTRMInference()
    result = trm_model.generate(
        prompt=problem,
        max_new_tokens=max_tokens,
        temperature=0.0,
        do_sample=False,
    )
    response_trm = result["text"]
    if "<|im_start|>assistant" in response_trm:
        response_trm = response_trm.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in response_trm:
            response_trm = response_trm.split("<|im_end|>")[0]
    print(f"Response:\n{response_trm[:500]}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Base (no think): {len(response_no_think)} chars")
    print(f"Base (think):    {len(response_think)} chars")
    print(f"TRM:             {len(response_trm)} chars")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        test_comparison()
    else:
        test_trm_inference()
