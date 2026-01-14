"""
Phase 1: Compressor Pretraining

Train DimensionCompressor to faithfully compress and reconstruct LLM hidden states.

Two-stage training:
    Stage 1a: Identity training on regular LLM hidden state outputs
    Stage 1b: Finetune on CoT thinking trajectories

The goal is for the compressor to learn to preserve information through
the dimensionality bottleneck before full TRM training begins.

Usage:
    # Single GPU
    python -m src.train.phase1_compressor --stage 1a --d_compressed 256

    # Multi-GPU with accelerate
    accelerate launch -m src.train.phase1_compressor --stage 1a --d_compressed 256

    # Sweep over compression ratios
    python -m src.train.phase1_compressor --stage 1a --sweep
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.compression import DimensionCompressor

# Optional wandb import
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore[assignment]


@dataclass
class Phase1Config:
    """Configuration for Phase 1 training"""

    # Model
    model_name: str = "HuggingFaceTB/SmolLM3-3B"
    hidden_size: int = 3072  # SmolLM3-3B hidden size
    d_compressed: int = 256  # Compressed dimension (12x compression)

    # Training
    batch_size: int = 8
    learning_rate: float = 1e-3  # Higher LR since simple linear layer
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 3  # Stop after N epochs without improvement
    early_stopping_min_delta: float = 1e-6  # Minimum improvement to count

    # Torch compile
    use_torch_compile: bool = True

    # Stage
    stage: str = "1a"  # "1a" for identity, "1b" for CoT

    # Pretrained compressor (for Stage 1b)
    # Can be HF Hub repo (e.g. "anonx3247/llm-trm-compressor") or local path
    compressor_checkpoint: str | None = None

    # Data
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_subset: str = "sample-10BT"  # Use 10BT sample for reasonable size
    # CoT dataset for Stage 1b
    cot_dataset_name: str = "teknium/OpenHermes-2.5"
    max_seq_length: int = 512  # Shorter for faster training
    num_samples: int = 50000  # Number of samples to use
    num_workers: int = 4

    # Output
    output_dir: str = "./checkpoints/phase1"
    log_steps: int = 10

    # Wandb
    use_wandb: bool = True
    wandb_project: str = "llm-trm-phase1"
    wandb_run_name: str | None = None

    # Sweep config
    sweep_d_compressed: list[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024])

    # Reproducibility
    seed: int = 42

    @property
    def compression_ratio(self) -> float:
        return self.hidden_size / self.d_compressed


class EarlyStopping:
    """Early stopping handler to stop training when loss stops improving."""

    def __init__(self, patience: int = 3, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, loss: float) -> bool:
        """Check if training should stop. Returns True if improved."""
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            return True  # Improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False  # Did not improve


class CompressorPretrainer:
    """
    Phase 1 trainer for DimensionCompressor with multi-GPU support.

    Features:
    - Early stopping when loss plateaus
    - Best checkpoint saving only (saves storage)
    - Torch compile for faster training
    - Multi-GPU via Accelerate

    Stage 1a: Identity Training
        - Run LLM on regular text
        - Compress hidden dimension: [B, L, D] -> [B, L, D']
        - Decompress back: [B, L, D'] -> [B, L, D]
        - Loss: MSE between original and reconstructed hidden states

    Stage 1b: CoT Trajectory Finetuning
        - Load LLM in thinking mode
        - Extract hidden states from thinking sequences
        - Same compression/decompression training
        - Goal: Compressor learns to handle reasoning trajectories
    """

    def __init__(self, config: Phase1Config):
        self.config = config
        self.best_loss = float("inf")
        self.best_metrics: dict[str, float] = {}

        # Determine mixed precision based on device
        # MPS has issues with mixed precision autocast, disable it
        if torch.cuda.is_available():
            mixed_precision = "bf16"
        else:
            # MPS and CPU: no mixed precision (MPS autocast causes dtype mismatches)
            mixed_precision = "no"

        # Initialize accelerator for multi-GPU
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="wandb" if config.use_wandb and WANDB_AVAILABLE else None,
            mixed_precision=mixed_precision,
        )

        # Set seed for reproducibility
        set_seed(config.seed)

        # Create output directory
        if self.accelerator.is_main_process:
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._init_model()
        self._init_compressor()
        self._init_optimizer()
        self._init_wandb()

    def _get_device_and_dtype(self) -> tuple[str, torch.dtype]:
        """Determine the best device and dtype for training."""
        if torch.cuda.is_available():
            return "cuda", torch.bfloat16
        elif torch.backends.mps.is_available():
            # MPS has issues with mixed precision, use float32 for stability
            return "mps", torch.float32
        else:
            return "cpu", torch.float32

    def _init_model(self) -> None:
        """Initialize frozen LLM for hidden state extraction"""
        self._print(f"Loading {self.config.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine device and dtype
        self.device, self.dtype = self._get_device_and_dtype()
        self._print(f"Using device: {self.device}, dtype: {self.dtype}")

        # Load model with appropriate settings per device
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=self.dtype,
                device_map="auto",
            )
        else:
            # MPS and CPU don't support device_map="auto"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=self.dtype,
            )
            self.model = self.model.to(torch.device(self.device))  # type: ignore[arg-type]

        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Get actual hidden size from model (override config if different)
        actual_hidden_size = self.model.config.hidden_size
        if self.config.hidden_size != actual_hidden_size:
            self._print(
                f"Warning: Config hidden_size ({self.config.hidden_size}) doesn't match "
                f"model ({actual_hidden_size}). Using model's hidden_size."
            )
            self.config.hidden_size = actual_hidden_size

        self._print(f"Model loaded. Hidden size: {actual_hidden_size}")

    def _init_compressor(self) -> None:
        """Initialize DimensionCompressor with optional torch.compile"""
        self.compressor = DimensionCompressor(
            d_model=self.config.hidden_size,
            d_compressed=self.config.d_compressed,
        )

        # Keep compressor in float32 - accelerate handles mixed precision casting
        # (master weights stay in fp32, forward pass uses autocast)

        # Apply torch.compile for faster training (not supported well on MPS)
        if self.config.use_torch_compile and self.device == "cuda":
            try:
                self.compressor = torch.compile(self.compressor)  # type: ignore[assignment]
                self._print("Torch compile enabled for compressor")
            except Exception as e:
                self._print(f"Torch compile failed, using eager mode: {e}")
        elif self.config.use_torch_compile and self.device != "cuda":
            self._print(f"Torch compile disabled on {self.device} (not well supported)")

        # Get parameters (handle both compiled and regular modules)
        params = self.compressor.parameters()
        num_params = sum(p.numel() for p in params)
        self._print(f"Compressor initialized. Parameters: {num_params:,}")
        self._print(f"Compression ratio: {self.config.compression_ratio:.1f}x")

    def _init_optimizer(self) -> None:
        """Initialize optimizer and scheduler"""
        self.optimizer = AdamW(
            self.compressor.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

    def _init_wandb(self) -> None:
        """Initialize wandb logging"""
        if not self.config.use_wandb or not WANDB_AVAILABLE:
            return

        if self.accelerator.is_main_process:
            run_name = self.config.wandb_run_name or (
                f"d{self.config.d_compressed}_"
                f"bs{self.config.batch_size}_"
                f"lr{self.config.learning_rate}_"
                f"{self.config.stage}"
            )

            self.accelerator.init_trackers(
                project_name=self.config.wandb_project,
                config={
                    "d_compressed": self.config.d_compressed,
                    "compression_ratio": self.config.compression_ratio,
                    "hidden_size": self.config.hidden_size,
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                    "num_epochs": self.config.num_epochs,
                    "num_samples": self.config.num_samples,
                    "max_seq_length": self.config.max_seq_length,
                    "stage": self.config.stage,
                    "model_name": self.config.model_name,
                    "early_stopping": self.config.early_stopping,
                    "early_stopping_patience": self.config.early_stopping_patience,
                    "use_torch_compile": self.config.use_torch_compile,
                },
                init_kwargs={"wandb": {"name": run_name}},
            )

    def _print(self, msg: str) -> None:
        """Print only on main process"""
        if self.accelerator.is_main_process:
            print(msg)

    def _adapt_state_dict_keys(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Adapt state dict keys to match current model (handles torch.compile prefix)."""
        # Check if current model is compiled (has _orig_mod prefix)
        model_keys = list(self.compressor.state_dict().keys())
        model_has_prefix = any(k.startswith("_orig_mod.") for k in model_keys)
        ckpt_has_prefix = any(k.startswith("_orig_mod.") for k in state_dict.keys())

        if model_has_prefix == ckpt_has_prefix:
            # Keys match, no adaptation needed
            return state_dict

        new_state_dict = {}
        for key, value in state_dict.items():
            if model_has_prefix and not ckpt_has_prefix:
                # Model is compiled, checkpoint is not - add prefix
                new_key = f"_orig_mod.{key}"
            elif not model_has_prefix and ckpt_has_prefix:
                # Model is not compiled, checkpoint is - strip prefix
                new_key = key.replace("_orig_mod.", "")
            else:
                new_key = key
            new_state_dict[new_key] = value
        return new_state_dict

    def _load_dataset(self, use_thinking: bool = False) -> DataLoader:
        """Load dataset for training.

        Args:
            use_thinking: If True, load CoT dataset for Stage 1b
        """
        if use_thinking:
            # Stage 1b: Use CoT dataset with reasoning traces
            self._print(f"Loading CoT dataset: {self.config.cot_dataset_name}...")
            dataset = load_dataset(
                self.config.cot_dataset_name,
                split="train",
                streaming=True,
            )

            def extract_text(item: dict[str, Any]) -> str:
                """Extract text from CoT dataset item."""
                # Handle OpenHermes format (conversations list)
                if "conversations" in item:
                    texts = []
                    for conv in item["conversations"]:
                        if isinstance(conv, dict):
                            texts.append(conv.get("value", ""))
                    return " ".join(texts)
                # Handle other formats
                elif "text" in item:
                    return item["text"]
                elif "content" in item:
                    return item["content"]
                else:
                    # Fallback: join all string values
                    return " ".join(str(v) for v in item.values() if isinstance(v, str))

            def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
                texts = [extract_text(item) for item in batch]
                return dict(
                    self.tokenizer(
                        texts,
                        truncation=True,
                        max_length=self.config.max_seq_length,
                        padding="max_length",
                        return_tensors="pt",
                    )
                )
        else:
            # Stage 1a: Use regular text dataset
            self._print(
                f"Loading dataset: {self.config.dataset_name}/{self.config.dataset_subset}..."
            )
            dataset = load_dataset(
                self.config.dataset_name,
                self.config.dataset_subset,
                split="train",
                streaming=True,
            )

            def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
                texts = [item["text"] for item in batch]
                return dict(
                    self.tokenizer(
                        texts,
                        truncation=True,
                        max_length=self.config.max_seq_length,
                        padding="max_length",
                        return_tensors="pt",
                    )
                )

        # Convert streaming dataset to list for DataLoader
        # IMPORTANT: Use .take() to limit streaming dataset to num_samples
        self._print("Processing dataset...")
        limited_dataset = dataset.take(self.config.num_samples)
        if self.accelerator.is_main_process:
            data_list: list[dict[str, Any]] = list(
                tqdm(limited_dataset, total=self.config.num_samples, desc="Loading data")
            )
        else:
            data_list = list(limited_dataset)

        dataloader: DataLoader[dict[str, Any]] = DataLoader(
            data_list,  # type: ignore[arg-type]
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        self._print(f"Dataset loaded. Batches: {len(dataloader)}")
        return dataloader

    @torch.no_grad()
    def _extract_hidden_states(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract hidden states from LLM"""
        outputs = self.model(
            input_ids=input_ids.to(self.model.device),
            attention_mask=attention_mask.to(self.model.device),
            output_hidden_states=True,
            return_dict=True,
        )
        # Get last layer hidden states
        hidden_states: torch.Tensor = outputs.hidden_states[-1]
        return hidden_states

    def _compute_metrics(
        self, original: torch.Tensor, reconstructed: torch.Tensor
    ) -> dict[str, float]:
        """Compute reconstruction quality metrics"""
        with torch.no_grad():
            # MSE loss
            mse = F.mse_loss(reconstructed, original).item()

            # Relative error
            rel_error = (torch.norm(reconstructed - original) / torch.norm(original)).item()

            # Cosine similarity (averaged over sequence)
            cos_sim = (
                F.cosine_similarity(
                    original.reshape(-1, original.size(-1)),
                    reconstructed.reshape(-1, reconstructed.size(-1)),
                    dim=-1,
                )
                .mean()
                .item()
            )

            # Per-dimension variance preserved
            orig_var = original.var(dim=(0, 1))
            recon_var = reconstructed.var(dim=(0, 1))
            var_ratio = (recon_var / (orig_var + 1e-8)).mean().item()

        return {
            "mse": mse,
            "relative_error": rel_error,
            "cosine_similarity": cos_sim,
            "variance_ratio": var_ratio,
        }

    def _train_step(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Single training step with metrics"""
        compressed = self.compressor(hidden_states)
        reconstructed = self.compressor.decompress(compressed)
        loss = F.mse_loss(reconstructed, hidden_states)

        metrics = self._compute_metrics(hidden_states, reconstructed)
        return loss, metrics

    def _save_best_checkpoint(
        self, stage_name: str, loss: float, metrics: dict[str, float]
    ) -> bool:
        """Save checkpoint only if it's the best so far. Returns True if saved."""
        if loss >= self.best_loss:
            return False

        self.best_loss = loss
        self.best_metrics = metrics.copy()

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            # Unwrap model if distributed
            unwrapped = self.accelerator.unwrap_model(self.compressor)
            checkpoint = {
                "compressor": unwrapped.state_dict(),
                "config": self.config,
                "best_loss": self.best_loss,
                "best_metrics": self.best_metrics,
            }

            # Save as best checkpoint (overwrites previous best)
            path = os.path.join(self.config.output_dir, f"stage{stage_name}_best.pt")
            torch.save(checkpoint, path)
            self._print(
                f"New best checkpoint! Loss: {loss:.6f}, "
                f"Cosine Sim: {metrics['cosine_similarity']:.4f}"
            )

            # Also save to wandb if enabled
            if self.config.use_wandb and WANDB_AVAILABLE and wandb is not None:
                artifact = wandb.Artifact(
                    name=f"compressor-stage{stage_name}-best",
                    type="model",
                    metadata={
                        "d_compressed": self.config.d_compressed,
                        "compression_ratio": self.config.compression_ratio,
                        "best_loss": self.best_loss,
                        "cosine_similarity": metrics["cosine_similarity"],
                    },
                )
                artifact.add_file(path)
                wandb.log_artifact(artifact)

        return True

    def _run_training_loop(self, stage_name: str) -> None:
        """Common training loop for both stages"""
        self._print(f"\n{'=' * 50}")
        self._print(f"Stage {stage_name}")
        self._print(f"{'=' * 50}")

        dataloader = self._load_dataset(use_thinking=(stage_name == "1b"))

        # Create scheduler
        num_training_steps = len(dataloader) * self.config.num_epochs
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        scheduler = CosineAnnealingLR(self.optimizer, T_max=num_training_steps - num_warmup_steps)

        # Prepare for distributed training
        self.compressor, self.optimizer, dataloader, scheduler = self.accelerator.prepare(
            self.compressor, self.optimizer, dataloader, scheduler
        )

        # Initialize early stopping
        early_stopper = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
        )

        self.compressor.train()
        global_step = 0
        self.best_loss = float("inf")

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            epoch_metrics: dict[str, float] = {
                "mse": 0.0,
                "relative_error": 0.0,
                "cosine_similarity": 0.0,
                "variance_ratio": 0.0,
            }
            num_batches = 0

            pbar = tqdm(
                dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
                disable=not self.accelerator.is_main_process,
            )

            for batch in pbar:
                with self.accelerator.accumulate(self.compressor):
                    # Extract hidden states from LLM
                    hidden_states = self._extract_hidden_states(
                        batch["input_ids"],
                        batch["attention_mask"],
                    )

                    # Move to compressor device and cast to float32
                    # (accelerate mixed precision keeps master weights in fp32)
                    hidden_states = hidden_states.to(
                        device=self.accelerator.device, dtype=torch.float32
                    )

                    # Compute loss and metrics
                    loss, metrics = self._train_step(hidden_states)

                    # Backward
                    self.accelerator.backward(loss)

                    # Clip gradients
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.compressor.parameters(),
                            self.config.max_grad_norm,
                        )

                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()

                epoch_loss += loss.item()
                for k, v in metrics.items():
                    epoch_metrics[k] += v
                num_batches += 1

                # Logging
                if self.accelerator.sync_gradients:
                    global_step += 1

                    if global_step % self.config.log_steps == 0:
                        avg_loss = epoch_loss / num_batches
                        avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}

                        pbar.set_postfix(
                            {
                                "loss": f"{avg_loss:.6f}",
                                "cos_sim": f"{avg_metrics['cosine_similarity']:.4f}",
                            }
                        )

                        # Log to wandb
                        if self.config.use_wandb and WANDB_AVAILABLE:
                            self.accelerator.log(
                                {
                                    "train/loss": avg_loss,
                                    "train/mse": avg_metrics["mse"],
                                    "train/relative_error": avg_metrics["relative_error"],
                                    "train/cosine_similarity": avg_metrics["cosine_similarity"],
                                    "train/variance_ratio": avg_metrics["variance_ratio"],
                                    "train/learning_rate": scheduler.get_last_lr()[0],
                                    "train/epoch": epoch,
                                    "train/global_step": global_step,
                                },
                                step=global_step,
                            )

            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            avg_epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}

            self._print(
                f"Epoch {epoch + 1} complete. "
                f"Loss: {avg_epoch_loss:.6f}, "
                f"Cosine Sim: {avg_epoch_metrics['cosine_similarity']:.4f}"
            )

            # Log epoch metrics to wandb
            if self.config.use_wandb and WANDB_AVAILABLE:
                self.accelerator.log(
                    {
                        "epoch/loss": avg_epoch_loss,
                        "epoch/cosine_similarity": avg_epoch_metrics["cosine_similarity"],
                        "epoch/relative_error": avg_epoch_metrics["relative_error"],
                        "epoch/best_loss": self.best_loss,
                    },
                    step=global_step,
                )

            # Save best checkpoint
            self._save_best_checkpoint(stage_name, avg_epoch_loss, avg_epoch_metrics)

            # Early stopping check
            if self.config.early_stopping:
                improved = early_stopper(avg_epoch_loss)
                if early_stopper.should_stop:
                    self._print(
                        f"Early stopping triggered after {epoch + 1} epochs. "
                        f"No improvement for {self.config.early_stopping_patience} epochs."
                    )
                    break
                elif not improved:
                    self._print(
                        f"No improvement. Patience: {early_stopper.counter}/"
                        f"{self.config.early_stopping_patience}"
                    )

        self._print(f"\nStage {stage_name} training complete!")
        self._print(
            f"Best loss: {self.best_loss:.6f}, "
            f"Best cosine sim: {self.best_metrics.get('cosine_similarity', 0):.4f}"
        )

    def train_stage_1a(self) -> None:
        """Stage 1a: Identity training on regular hidden states."""
        self._run_training_loop("1a")

    def train_stage_1b(self) -> None:
        """Stage 1b: Finetune on CoT thinking trajectories."""
        checkpoint_loaded = False

        # Priority 1: Load from specified checkpoint (HF Hub or local path)
        if self.config.compressor_checkpoint:
            ckpt_source = self.config.compressor_checkpoint

            # Check if it's an HF Hub repo (contains "/" and doesn't look like a path)
            if "/" in ckpt_source and not os.path.exists(ckpt_source):
                self._print(f"Downloading compressor from HF Hub: {ckpt_source}...")
                try:
                    ckpt_path = hf_hub_download(repo_id=ckpt_source, filename="compressor.pt")
                    self._print(f"Downloaded to: {ckpt_path}")
                except Exception as e:
                    self._print(f"Failed to download from HF Hub: {e}")
                    ckpt_path = None
            else:
                # Treat as local path
                ckpt_path = ckpt_source

            if ckpt_path and os.path.exists(ckpt_path):
                self._print(f"Loading compressor checkpoint from {ckpt_path}")
                # Fix pickle compatibility: checkpoints may contain Phase1Config
                # saved from __main__ when running as a script
                sys.modules["__main__"].Phase1Config = Phase1Config  # type: ignore[attr-defined]
                checkpoint = torch.load(
                    ckpt_path, map_location=self.accelerator.device, weights_only=False
                )
                # Handle torch.compile prefix in state dict keys
                state_dict = checkpoint["compressor"]
                state_dict = self._adapt_state_dict_keys(state_dict)
                self.compressor.load_state_dict(state_dict)
                checkpoint_loaded = True
            elif ckpt_path:
                self._print(f"Warning: Checkpoint not found at {ckpt_path}")

        # Priority 2: Fall back to local stage 1a checkpoint
        if not checkpoint_loaded:
            stage1a_path = os.path.join(self.config.output_dir, "stage1a_best.pt")
            if os.path.exists(stage1a_path):
                self._print(f"Loading Stage 1a checkpoint from {stage1a_path}")
                # Fix pickle compatibility for local checkpoints too
                sys.modules["__main__"].Phase1Config = Phase1Config  # type: ignore[attr-defined]
                checkpoint = torch.load(
                    stage1a_path, map_location=self.accelerator.device, weights_only=False
                )
                # Handle torch.compile prefix in state dict keys
                state_dict = checkpoint["compressor"]
                state_dict = self._adapt_state_dict_keys(state_dict)
                self.compressor.load_state_dict(state_dict)
                checkpoint_loaded = True
            else:
                self._print("Warning: No checkpoint found. Training from scratch.")

        self._run_training_loop("1b")

    def train(self) -> None:
        """Run the appropriate training stage"""
        try:
            if self.config.stage == "1a":
                self.train_stage_1a()
            elif self.config.stage == "1b":
                self.train_stage_1b()
            else:
                raise ValueError(f"Unknown stage: {self.config.stage}")
        finally:
            # End wandb run
            if self.config.use_wandb and WANDB_AVAILABLE:
                self.accelerator.end_training()


def run_phase1_training(config: Phase1Config | None = None) -> None:
    """Main entry point for Phase 1 training."""
    config = config or Phase1Config()
    trainer = CompressorPretrainer(config)
    trainer.train()


def run_compression_sweep(base_config: Phase1Config) -> None:
    """
    Run a sweep over different compression ratios.

    This trains separate compressors for each D' value to find the
    optimal compression-quality tradeoff.
    """
    print(f"Running compression sweep over D' values: {base_config.sweep_d_compressed}")

    results: list[dict[str, Any]] = []

    for d_compressed in base_config.sweep_d_compressed:
        print(f"\n{'#' * 60}")
        print(f"Training with D' = {d_compressed}")
        print(f"Compression ratio: {base_config.hidden_size / d_compressed:.1f}x")
        print(f"{'#' * 60}")

        # Create config for this run
        config = Phase1Config(
            model_name=base_config.model_name,
            hidden_size=base_config.hidden_size,
            d_compressed=d_compressed,
            batch_size=base_config.batch_size,
            learning_rate=base_config.learning_rate,
            num_epochs=base_config.num_epochs,
            early_stopping=base_config.early_stopping,
            early_stopping_patience=base_config.early_stopping_patience,
            use_torch_compile=base_config.use_torch_compile,
            stage=base_config.stage,
            dataset_name=base_config.dataset_name,
            dataset_subset=base_config.dataset_subset,
            max_seq_length=base_config.max_seq_length,
            num_samples=base_config.num_samples,
            output_dir=os.path.join(base_config.output_dir, f"d{d_compressed}"),
            use_wandb=base_config.use_wandb,
            wandb_project=base_config.wandb_project,
            wandb_run_name=f"sweep_d{d_compressed}_{base_config.stage}",
            seed=base_config.seed,
        )

        trainer = CompressorPretrainer(config)
        trainer.train()

        results.append(
            {
                "d_compressed": d_compressed,
                "compression_ratio": config.compression_ratio,
                "best_loss": trainer.best_loss,
                "best_metrics": trainer.best_metrics,
            }
        )

    print("\n" + "=" * 60)
    print("Sweep complete! Results:")
    print("=" * 60)
    for r in results:
        print(
            f"  D'={r['d_compressed']:4d} ({r['compression_ratio']:5.1f}x): "
            f"loss={r['best_loss']:.6f}, "
            f"cos_sim={r['best_metrics'].get('cosine_similarity', 0):.4f}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1: Compressor Pretraining")
    parser.add_argument("--stage", type=str, default="1a", choices=["1a", "1b"])
    parser.add_argument("--output_dir", type=str, default="./checkpoints/phase1")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--d_compressed", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)

    # Early stopping
    parser.add_argument(
        "--early_stopping", action="store_true", default=True, help="Enable early stopping"
    )
    parser.add_argument("--no_early_stopping", action="store_false", dest="early_stopping")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Epochs without improvement before stopping",
    )

    # Torch compile
    parser.add_argument(
        "--torch_compile", action="store_true", default=True, help="Use torch.compile"
    )
    parser.add_argument("--no_torch_compile", action="store_false", dest="torch_compile")

    # Wandb options
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--no_wandb", action="store_false", dest="use_wandb")
    parser.add_argument("--wandb_project", type=str, default="llm-trm-phase1")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    # Sweep option
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run sweep over compression ratios",
    )
    parser.add_argument(
        "--sweep_d_values",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024],
        help="D' values for sweep",
    )

    parser.add_argument("--seed", type=int, default=42)

    # Stage 1b: Pretrained compressor checkpoint
    parser.add_argument(
        "--compressor_checkpoint",
        type=str,
        default=None,
        help="HF Hub repo (e.g. 'anonx3247/llm-trm-compressor') or local path to compressor checkpoint for Stage 1b",
    )

    args = parser.parse_args()

    config = Phase1Config(
        stage=args.stage,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        d_compressed=args.d_compressed,
        num_samples=args.num_samples,
        max_seq_length=args.max_seq_length,
        num_workers=args.num_workers,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        use_torch_compile=args.torch_compile,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        sweep_d_compressed=args.sweep_d_values,
        seed=args.seed,
        compressor_checkpoint=args.compressor_checkpoint,
    )

    if args.sweep:
        run_compression_sweep(config)
    else:
        run_phase1_training(config)
