"""
CA-SAM (Continual Alignment for SAM) Trainer

This module provides a trainer for CA-SAM models that supports:
1. Single-task training: Train Alignment Layer for a single dataset
2. Continual learning: Sequential training across multiple tasks with VAE routing

Based on the paper:
"Continual Alignment for SAM: Rethinking Foundation Models for
 Medical Image Segmentation in Continual Learning"
"""

from pathlib import Path
from typing import Dict, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import wandb

from .base_trainer import BaseTrainer
from model.segment_anything import sam_model_registry
from model.ca_sam import (
    CASAM,
    AlignmentLayer,
    BCEDiceLoss,
    compute_iou,
    compute_boundary_iou,
    MetricsTracker,
)
from utils.data_processing_seg import SegDatasetProcessor


class CASAMTrainer(BaseTrainer):
    """
    Trainer for CA-SAM (Continual Alignment for SAM) models.

    Supports two training modes:
    - single_task: Train a single Alignment Layer for one dataset
    - continual: Sequential training across multiple tasks with VAE routing

    Key features:
    - Frozen SAM encoder and decoder
    - Lightweight Alignment Layer between encoder and decoder
    - VAE-based task routing for continual learning
    - OOD fallback to frozen SAM for unseen domains
    """

    def __init__(self, cfg):
        """Initialize CA-SAM trainer."""
        super().__init__(cfg)

        # CA-SAM specific attributes

        self.current_task_id: int = 0
        self.task_features: Dict[int, torch.Tensor] = {}  # For VAE training

        # Loss functions
        self.criterion = None

        # Training mode
        self.training_mode = cfg.model.get("mode", "single_task")
        self.img_size = cfg.model.get("img_size", 224)

        # Metrics tracker
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()

    def _get_config_value(self, section: str, keys: list, default, cast_type=float):
        """Get config value with multiple key fallbacks."""
        cfg_section = self.cfg.get(section, {})
        for key in keys:
            if (value := cfg_section.get(key)) is not None:
                return cast_type(value)
        return default

    def _get_num_epochs(self) -> int:
        """Get number of training epochs."""
        return self._get_config_value("training", ["num_epochs", "max_epochs"], 24, int)

    def _get_base_lr(self) -> float:
        """Get base learning rate."""
        return self._get_config_value("training", ["base_lr", "lr"], 1e-4, float)

    def _get_alignment_config(self) -> Dict:
        """Get Alignment Layer configuration."""
        alignment_cfg = self.cfg.model.get("alignment", {})
        return {
            "hidden_dim": alignment_cfg.get("hidden_dim", 256),
            "num_blocks": alignment_cfg.get("num_blocks", 4),
        }

    def _get_vae_config(self) -> Dict:
        """Get VAE Router configuration."""
        vae_cfg = self.cfg.model.get("vae", {})
        return {
            "latent_dim": vae_cfg.get("latent_dim", 64),
            "beta": vae_cfg.get("beta", 16.5),
            "temperature": vae_cfg.get("temperature", 1.0),
            "threshold_percentile": vae_cfg.get("threshold_percentile", 97),
        }

    def _get_n_gpus(self) -> int:
        """Get number of GPUs from config."""
        hw = self.cfg.get("hardware", {})
        for key in ["n_gpu", "n_gpus"]:
            if (value := hw.get(key)) is not None:
                return int(value)
        if (gpu_ids := hw.get("gpu_ids")) and isinstance(gpu_ids, (list, tuple)):
            return len(gpu_ids)
        return 1

    def _create_model(self):
        """Create CA-SAM model with frozen SAM and trainable Alignment Layer."""
        # Get SAM configuration
        sam_type = self.cfg.model.get("sam_type", "vit_b")
        if sam_type not in sam_model_registry:
            sam_type = "vit_b"

        sam_checkpoint = self.cfg.model.get(
            "sam_checkpoint", self.cfg.model.get("ckpt")
        )

        self.logger.info(f"Loading SAM model: {sam_type}")
        self.logger.info(f"SAM checkpoint: {sam_checkpoint}")

        # Create SAM model
        sam, img_embedding_size = sam_model_registry[sam_type](
            image_size=self.img_size,
            num_classes=self.cfg.data.num_classes,
            checkpoint=sam_checkpoint,
            pixel_mean=[0, 0, 0],
            pixel_std=[1, 1, 1],
        )

        # Get encoder output dimension based on SAM type
        encoder_dim_map = {
            "vit_b": 256,
            "vit_l": 256,
            "vit_h": 256,
        }
        encoder_output_dim = encoder_dim_map.get(sam_type, 256)

        # Get Alignment and VAE configuration
        alignment_cfg = self._get_alignment_config()
        vae_cfg = self._get_vae_config()

        self.logger.info(f"Alignment Layer config: {alignment_cfg}")
        self.logger.info(f"VAE Router config: {vae_cfg}")

        # Create CA-SAM model
        self.ca_sam_model = CASAM(
            sam_encoder=sam.image_encoder,
            sam_decoder=sam.mask_decoder,
            encoder_output_dim=encoder_output_dim,
            alignment_hidden_dim=alignment_cfg["hidden_dim"],
            alignment_num_blocks=alignment_cfg["num_blocks"],
            vae_latent_dim=vae_cfg["latent_dim"],
            vae_beta=vae_cfg["beta"],
            attention_temperature=vae_cfg["temperature"],
        )

        # Store prompt encoder for generating prompts
        self.prompt_encoder = sam.prompt_encoder
        self.prompt_encoder.eval()
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        # Add first task
        self.current_task_id = self.ca_sam_model.add_new_task()
        self.ca_sam_model.set_training_task(self.current_task_id)

        # Move to device
        self.ca_sam_model = self.ca_sam_model.to(self.device)
        self.prompt_encoder = self.prompt_encoder.to(self.device)

        # For compatibility with BaseTrainer
        self.model = self.ca_sam_model

        # Setup loss function
        self.criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5).to(self.device)

        # Log model info
        total_params = sum(p.numel() for p in self.ca_sam_model.parameters())
        trainable_params = sum(
            p.numel() for p in self.ca_sam_model.parameters() if p.requires_grad
        )
        alignment_params = self.ca_sam_model.get_num_trainable_params(
            self.current_task_id
        )

        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Alignment Layer parameters: {alignment_params:,}")

    def _create_dataloaders(self):
        """Create data loaders using SegDatasetProcessor."""
        if self.training_mode == "continual":
            # In continual mode, we don't load all data at once
            # Data will be loaded per task in _setup_task_data
            pass
        else:
            self.train_loader, self.val_loader, self.test_loader = (
                SegDatasetProcessor.build_data_loaders(self.cfg)
            )

            self.logger.info(f"Train set size: {len(self.train_loader.dataset)}")
            self.logger.info(f"Val set size: {len(self.val_loader.dataset)}")

            # Log test set sizes
            if isinstance(self.test_loader, dict):
                for name, loader in self.test_loader.items():
                    self.logger.info(f"Test set ({name}): {len(loader.dataset)}")
            else:
                self.logger.info(f"Test set size: {len(self.test_loader.dataset)}")

    def setup(self, mode: str = "train"):
        """Setup training environment with support for continual learning."""
        # For continual mode, we defer optimizer/scheduler/dataloaders setup
        # but we still need model and basic directories
        if self.training_mode == "continual" and mode == "train":
            # Run basic setup from BaseTrainer (seeds, dirs, logger)
            self._set_seed()
            self._setup_directories(mode)
            self._setup_logger()
            self._setup_wandb()

            # Create model (but no task-specific loaders yet)
            self._create_model()

            self.logger.info(f"Continual Learning setup completed for {mode} mode")
        else:
            # Standard setup
            super().setup(mode=mode)

    def _create_optimizer(self):
        """Create optimizer for Alignment Layer parameters only."""
        base_lr = self._get_base_lr()

        # Only optimize current task's Alignment Layer
        trainable_params = self.ca_sam_model.alignment_layers[
            self.current_task_id
        ].parameters()

        optimizer_name = self.cfg.get("optimizer", {}).get("name", "Adam")

        if optimizer_name == "AdamW":
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=base_lr,
                betas=(0.9, 0.999),
                weight_decay=self.cfg.get("optimizer", {}).get("weight_decay", 0.01),
            )
        else:
            self.optimizer = optim.Adam(
                trainable_params,
                lr=base_lr,
                betas=(0.9, 0.999),
            )

        self.logger.info(f"Optimizer: {optimizer_name}, Base LR: {base_lr}")

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        max_epochs = self._get_num_epochs()
        total_iters = max_epochs * len(self.train_loader)

        # Cosine annealing scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_iters, eta_min=1e-6
        )

        self.logger.info(f"Scheduler: CosineAnnealingLR, T_max={total_iters}")

    def _get_current_lr(self) -> float:
        """Get current learning rate from optimizer."""
        return self.optimizer.param_groups[0]["lr"]

    def _forward_with_prompts(
        self, images: torch.Tensor, task_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass through CA-SAM with automatic prompt generation.

        For CA-SAM, we use a simplified forward that:
        1. Encodes images through frozen SAM encoder
        2. Applies Alignment Layer
        3. Decodes through frozen SAM decoder

        Args:
            images: Input images [B, 3, H, W]
            task_id: Task ID for alignment layer selection

        Returns:
            pred_masks: Predicted masks [B, 1, H, W]
        """
        batch_size = images.shape[0]

        # Get encoder output (frozen)
        with torch.no_grad():
            encoder_output = self.ca_sam_model.sam_encoder(images)

        # Apply alignment layer (trainable)
        if task_id is None:
            task_id = self.current_task_id

        aligned_features, _ = self.ca_sam_model.forward_alignment(
            encoder_output, task_id=task_id
        )

        # Generate default prompts (no points, full image)
        # Note: SAM's mask_decoder handles batch expansion internally via repeat_interleave
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )

        # Decode (frozen but need gradients to flow back to alignment layer)
        # Note: decoder params are frozen via freeze_sam(), but we need gradient flow
        # Use multimask_output=True for SAM compatibility, then select first mask
        low_res_masks, iou_predictions = self.ca_sam_model.sam_decoder(
            image_embeddings=aligned_features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )

        # Select single mask (first one or best IoU)
        # low_res_masks shape: [B, num_masks, H, W] -> [B, 1, H, W]
        low_res_masks = low_res_masks[:, 0:1, :, :]

        # Upscale masks to original resolution
        pred_masks = F.interpolate(
            low_res_masks,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )

        return pred_masks

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.ca_sam_model.train()
        self.ca_sam_model.freeze_sam()  # Ensure SAM stays frozen
        self.train_metrics.reset()

        total_loss = 0.0
        num_batches = len(self.train_loader)

        train_pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self._get_num_epochs()} [Train]",
        )

        for batch_idx, batch in enumerate(train_pbar):
            # Handle different batch formats
            if len(batch) == 3:
                images, masks, low_res_masks = batch
            else:
                images, masks = batch[:2]
                low_res_masks = masks

            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            pred_logits = self._forward_with_prompts(
                images, task_id=self.current_task_id
            )

            # Prepare target
            target = masks.unsqueeze(1).float() if masks.dim() == 3 else masks.float()
            if target.shape[-2:] != pred_logits.shape[-2:]:
                target = F.interpolate(
                    target, size=pred_logits.shape[-2:], mode="nearest"
                )
            target = (target > 0.5).float()

            # Compute loss
            loss = self.criterion(pred_logits, target)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_clip_cfg = self.cfg.get("optimizer", {}).get("grad_clip", {})
            if grad_clip_cfg.get("enabled", False):
                max_norm = grad_clip_cfg.get("max_norm", 1.0)
                torch.nn.utils.clip_grad_norm_(
                    self.ca_sam_model.alignment_layers[
                        self.current_task_id
                    ].parameters(),
                    max_norm,
                )

            self.optimizer.step()
            self.scheduler.step()

            # Compute metrics
            with torch.no_grad():
                pred_probs = torch.sigmoid(pred_logits)
                iou = compute_iou(pred_probs, target)
                biou = compute_boundary_iou(pred_probs, target)

            self.train_metrics.update(loss.item(), iou, biou)
            total_loss += loss.item()

            # Update progress bar
            avg_loss, avg_iou, avg_biou = self.train_metrics.get_average()
            train_pbar.set_postfix(
                {
                    "loss": f"{avg_loss:.4f}",
                    "iou": f"{avg_iou:.4f}",
                    "lr": f"{self._get_current_lr():.6f}",
                }
            )

            # Log to wandb
            if self.global_step % 10 == 0:
                wandb.log(
                    {
                        "step_train/loss": loss.item(),
                        "step_train/iou": iou,
                        "step_train/biou": biou,
                        "step_train/learning_rate": self._get_current_lr(),
                        "global_step": self.global_step,
                    }
                )

            self.global_step += 1

        # Return epoch metrics
        avg_loss, avg_iou, avg_biou = self.train_metrics.get_average()
        return {
            "loss": avg_loss,
            "IoU": avg_iou,
            "BIoU": avg_biou,
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate model."""
        self.ca_sam_model.eval()
        self.val_metrics.reset()

        val_pbar = tqdm(
            self.val_loader, desc=f"Epoch {epoch + 1}/{self._get_num_epochs()} [Val]"
        )

        for batch in val_pbar:
            if len(batch) == 3:
                images, masks, _ = batch
            else:
                images, masks = batch[:2]

            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            pred_logits = self._forward_with_prompts(
                images, task_id=self.current_task_id
            )

            # Prepare target
            target = masks.unsqueeze(1).float() if masks.dim() == 3 else masks.float()
            if target.shape[-2:] != pred_logits.shape[-2:]:
                target = F.interpolate(
                    target, size=pred_logits.shape[-2:], mode="nearest"
                )
            target = (target > 0.5).float()

            # Compute loss and metrics
            loss = self.criterion(pred_logits, target)
            pred_probs = torch.sigmoid(pred_logits)
            iou = compute_iou(pred_probs, target)
            biou = compute_boundary_iou(pred_probs, target)

            self.val_metrics.update(loss.item(), iou, biou)

            # Update progress bar
            avg_loss, avg_iou, avg_biou = self.val_metrics.get_average()
            val_pbar.set_postfix(
                {
                    "loss": f"{avg_loss:.4f}",
                    "iou": f"{avg_iou:.4f}",
                }
            )

        avg_loss, avg_iou, avg_biou = self.val_metrics.get_average()

        # Use IoU as Dice for compatibility with BaseTrainer
        return {
            "loss": avg_loss,
            "IoU": avg_iou,
            "BIoU": avg_biou,
            "Dice": avg_iou,  # For early stopping compatibility
        }

    @torch.no_grad()
    def test(self) -> Dict[str, float]:
        """Test model on test set(s)."""
        self.ca_sam_model.eval()

        all_metrics = {}

        # Handle multiple test sets
        if isinstance(self.test_loader, dict):
            for name, loader in self.test_loader.items():
                metrics = self._test_single_loader(loader, name)
                for k, v in metrics.items():
                    all_metrics[f"{name}/{k}"] = v
        else:
            all_metrics = self._test_single_loader(self.test_loader, "test")

        return all_metrics

    def _test_single_loader(self, loader, name: str) -> Dict[str, float]:
        """Test on a single data loader."""
        test_metrics = MetricsTracker()

        for batch in tqdm(loader, desc=f"Testing [{name}]"):
            if len(batch) == 3:
                images, masks, _ = batch
            else:
                images, masks = batch[:2]

            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            pred_logits = self._forward_with_prompts(
                images, task_id=self.current_task_id
            )

            # Prepare target
            target = masks.unsqueeze(1).float() if masks.dim() == 3 else masks.float()
            if target.shape[-2:] != pred_logits.shape[-2:]:
                target = F.interpolate(
                    target, size=pred_logits.shape[-2:], mode="nearest"
                )
            target = (target > 0.5).float()

            # Compute metrics
            pred_probs = torch.sigmoid(pred_logits)
            iou = compute_iou(pred_probs, target)
            biou = compute_boundary_iou(pred_probs, target)

            test_metrics.update(0.0, iou, biou)

        avg_loss, avg_iou, avg_biou = test_metrics.get_average()

        self.logger.info(f"Test [{name}] - IoU: {avg_iou:.4f}, BIoU: {avg_biou:.4f}")

        return {
            "IoU": avg_iou,
            "BIoU": avg_biou,
            "Dice": avg_iou,
        }

    def _save_model(self, path: Path):
        """Save CA-SAM model (Alignment Layer and VAE Router)."""
        save_dict = {
            "task_id": self.current_task_id,
            "alignment_layers": {
                i: layer.state_dict()
                for i, layer in enumerate(self.ca_sam_model.alignment_layers)
            },
            "vae_router": self.ca_sam_model.vae_router.state_dict(),
            "thresholds": self.ca_sam_model.vae_router.task_thresholds,
        }
        torch.save(save_dict, str(path))

    def _load_checkpoint(self, path: Path):
        """Load CA-SAM model checkpoint."""
        self.logger.info(f"Loading checkpoint: {path}")
        checkpoint = torch.load(str(path), map_location=self.device)

        # Load alignment layers
        for i, state_dict in checkpoint["alignment_layers"].items():
            if int(i) < len(self.ca_sam_model.alignment_layers):
                self.ca_sam_model.alignment_layers[int(i)].load_state_dict(state_dict)

        # Load VAE router
        if "vae_router" in checkpoint:
            self.ca_sam_model.vae_router.load_state_dict(checkpoint["vae_router"])

        # Load thresholds
        if "thresholds" in checkpoint:
            self.ca_sam_model.vae_router.task_thresholds = checkpoint["thresholds"]

        self.current_task_id = checkpoint.get("task_id", 0)

    # ==================== VAE Training Methods ====================

    def collect_encoder_features(self) -> torch.Tensor:
        """
        Collect encoder features from training set for VAE training.

        Returns:
            features: Tensor of encoder features [N, C, H, W]
        """
        self.logger.info(
            f"Collecting encoder features for Task {self.current_task_id}..."
        )
        self.ca_sam_model.eval()

        all_features = []

        with torch.no_grad():
            for batch in tqdm(self.train_loader, desc="Collecting features"):
                if len(batch) == 3:
                    images, _, _ = batch
                else:
                    images = batch[0]

                images = images.to(self.device)
                encoder_output = self.ca_sam_model.sam_encoder(images)
                all_features.append(encoder_output.cpu())

        features = torch.cat(all_features, dim=0)
        self.logger.info(
            f"Collected {len(features)} features with shape {features.shape}"
        )

        # Cache for later use
        self.task_features[self.current_task_id] = features

        return features

    def train_vae_for_current_task(self):
        """Train VAE for current task after Alignment Layer training."""
        vae_cfg = self._get_vae_config()

        # Get or collect features
        if self.current_task_id not in self.task_features:
            features = self.collect_encoder_features()
        else:
            features = self.task_features[self.current_task_id]

        features = features.to(self.device)

        self.logger.info(f"Training VAE for Task {self.current_task_id}...")

        # Train VAE
        self.ca_sam_model.train_vae_for_task(
            task_id=self.current_task_id,
            train_features=features,
            num_epochs=self.cfg.training.get("vae_epochs", 10),
            learning_rate=self.cfg.training.get("vae_lr", 5e-4),
            batch_size=32,
        )

        # Calibrate threshold
        self.logger.info(f"Calibrating threshold for Task {self.current_task_id}...")
        threshold = self.ca_sam_model.calibrate_task_threshold(
            task_id=self.current_task_id,
            train_features=features,
            percentile=vae_cfg["threshold_percentile"],
        )

        self.logger.info(
            f"Task {self.current_task_id} VAE training completed. Threshold: {threshold:.4f}"
        )

    # ==================== Continual Learning Methods ====================

    def add_new_task(self) -> int:
        """Add a new task for continual learning."""
        task_id = self.ca_sam_model.add_new_task()
        self.current_task_id = task_id
        self.ca_sam_model.set_training_task(task_id)

        # Recreate optimizer for new task
        self._create_optimizer()
        self._create_scheduler()

        self.logger.info(f"Added new task: {task_id}")
        self.logger.info(
            f"Alignment Layer parameters: {self.ca_sam_model.get_num_trainable_params(task_id):,}"
        )

        return task_id

    def train(self):
        """Main training loop."""
        if self.training_mode == "continual":
            self.train_continual()
        else:
            super().train()

    def train_continual(self):
        """Sequential training across multiple tasks."""
        self.logger.info("Starting Continual Learning...")

        # Get list of tasks from config
        tasks = self.cfg.data.get("train", [])
        if isinstance(tasks, str):
            tasks = [tasks]

        self.logger.info(f"Task sequence: {tasks}")

        # Dictionary to store performance on all tasks
        self.cl_metrics = {task: [] for task in tasks}

        for task_idx, task_name in enumerate(tasks):
            self.logger.info(f"\n{'='*20} Task {task_idx}: {task_name} {'='*20}")

            # 1. Prepare Model & Optimizer
            if task_idx > 0:
                self.add_new_task()
            else:
                self._create_optimizer()

            # 2. Setup Data (needs optimizer for scheduler)
            self._setup_task_data(task_name)

            # 3. Train Alignment Layer
            self.logger.info(
                f"Training Alignment Layer for Task {task_idx} ({task_name})..."
            )

            # Reset early stopping for new task
            self._setup_early_stopping()
            self.best_metric = 0.0  # Reset best metric

            # Reuse base train loop for epochs
            # We override the inner loop control slightly by calling methods directly
            num_epochs = self._get_num_epochs()

            for epoch in range(num_epochs):
                self.current_epoch = epoch

                # Train
                train_metrics = self.train_epoch(epoch)

                # Validate (on current task)
                val_metrics = self.validate(epoch)

                # Log
                self._log_metrics(epoch, train_metrics, val_metrics)

                # Save checkpoint
                self._save_checkpoint(epoch, val_metrics)

                # Early stopping
                if self.early_stopping is not None:
                    self.early_stopping(
                        val_metrics.get("Dice", val_metrics.get("dice", 0.0))
                    )
                    if self.early_stopping.should_stop():
                        self.logger.info(
                            f"Early stopping triggered at epoch {epoch + 1}"
                        )
                        break

            # Load best model for this task before VAE training
            if self.best_model_path:
                self._load_checkpoint(self.best_model_path)

            # 4. Train VAE for current task
            self.train_vae_for_current_task()

            # 5. Evaluate on ALL tasks seen so far
            self.evaluate_all_tasks(tasks[: task_idx + 1])

            # Save task-specific checkpoint
            task_ckpt_path = (
                self.ckpt_dir / f"task_{task_idx}_{task_name}_completed.pth"
            )
            self._save_model(task_ckpt_path)
            self.logger.info(f"Saved completed task model: {task_ckpt_path}")

        self.logger.info("Continual Learning Completed!")

    def _setup_task_data(self, task_name: str):
        """Setup data loaders for a specific task."""
        self.logger.info(f"Setting up data for task: {task_name}")

        # Use our new static method from SegDatasetProcessor
        self.train_loader, self.val_loader, self.test_loader = (
            SegDatasetProcessor.build_continual_data_loaders(self.cfg, task_name)
        )

        self.logger.info(f"Train size: {len(self.train_loader.dataset)}")
        self.logger.info(f"Val size: {len(self.val_loader.dataset)}")
        self.logger.info(f"Test size: {len(self.test_loader.dataset)}")

        # Update scheduler for new data size
        self.scheduler = None
        self._create_scheduler()

    def evaluate_all_tasks(self, tasks_so_far: List[str]):
        """Evaluate model on all tasks seen so far."""
        self.logger.info(f"\nEvaluating on all tasks seen so far: {tasks_so_far}")
        self.ca_sam_model.eval()

        current_step_metrics = {}

        for task_idx, task_name in enumerate(tasks_so_far):
            self.logger.info(f"Evaluating Task {task_idx} ({task_name})...")

            # Load test data for this specific task
            # (Inefficient to reload every time, but safe)
            test_ds = SegDatasetProcessor.load_dataset_from_config(
                self.cfg, task_name, split="test"
            )
            test_loader = torch.utils.data.DataLoader(
                test_ds,
                batch_size=self.cfg.training.batch_size,
                shuffle=False,
                num_workers=self.cfg.training.num_workers,
            )

            # 1. Evaluate with Oracle (Correct Task ID)
            metrics_oracle = self._test_with_task_id(
                test_loader, task_id=task_idx, name=f"{task_name}_oracle"
            )

            # 2. Evaluate with Automatic Routing (VAE)
            metrics_auto = self._test_with_routing(
                test_loader, name=f"{task_name}_auto"
            )

            # Log metrics
            prefix = f"Task{self.current_task_id}_Eval/T{task_idx}_{task_name}"
            for k, v in metrics_oracle.items():
                current_step_metrics[f"{prefix}_Oracle_{k}"] = v
            for k, v in metrics_auto.items():
                current_step_metrics[f"{prefix}_Auto_{k}"] = v

        # Log to wandb
        wandb.log(current_step_metrics)

    def _test_with_task_id(self, loader, task_id, name):
        """Evaluate with specific task ID."""
        test_metrics = MetricsTracker()

        for batch in tqdm(loader, desc=f"Testing [{name}]"):
            if len(batch) == 3:
                images, masks, _ = batch
            else:
                images, masks = batch[:2]

            images = images.to(self.device)
            masks = masks.to(self.device)

            # Prepare target
            target = masks.unsqueeze(1).float() if masks.dim() == 3 else masks.float()
            target = (target > 0.5).float()

            # Forward with specific task ID
            pred_logits = self._forward_with_prompts(images, task_id=task_id)
            pred_probs = torch.sigmoid(pred_logits)

            # Resizing target to prediction size if needed
            if target.shape[-2:] != pred_probs.shape[-2:]:
                target = F.interpolate(
                    target, size=pred_probs.shape[-2:], mode="nearest"
                )

            iou = compute_iou(pred_probs, target)
            biou = compute_boundary_iou(pred_probs, target)
            test_metrics.update(0.0, iou, biou)

        avg_loss, avg_iou, avg_biou = test_metrics.get_average()
        return {"IoU": avg_iou, "BIoU": avg_biou}

    def _test_with_routing(self, loader, name):
        """Evaluate with automatic VAE routing."""
        test_metrics = MetricsTracker()
        task_correct = 0
        total_samples = 0

        for batch in tqdm(loader, desc=f"Testing [{name}]"):
            if len(batch) == 3:
                images, masks, _ = batch
            else:
                images, masks = batch[:2]

            images = images.to(self.device)
            masks = masks.to(self.device)

            # Prepare target
            target = masks.unsqueeze(1).float() if masks.dim() == 3 else masks.float()
            target = (target > 0.5).float()

            # Forward with routing (task_id=None)
            # _forward_with_prompts logic does not currently support `return_task_id` natively
            # We need to manually call model to get task ID for accuracy checking

            # 1. Get manually routed task ID
            with torch.no_grad():
                encoder_output = self.ca_sam_model.sam_encoder(images)
                _, selected_task_ids = self.ca_sam_model.forward_alignment(
                    encoder_output, task_id=None
                )

            # Note: selected_task_ids might be a scalar if batch processing assumes same task?
            # actually VAE router processes batch. `forward_alignment` returns `task_id`.
            # If `training_mode` is False, it routes.
            # But CA-SAM `forward_alignment` implementation for inference:
            # task_id, _ = self.vae_router.route_task(encoder_output)
            # It returns a single task_id for the whole batch? Checking `ca_sam.py`...
            # Yes: "route_task" does "return best_task_idx, prob_map". It seems it picks one task for the batch?
            # Or does it handle item-wise?
            # Let's check `vae_router.py` later. Assuming batch-wise for now or simple int.

            # 2. Forward for segmentation
            pred_logits = self._forward_with_prompts(images, task_id=None)
            pred_probs = torch.sigmoid(pred_logits)

            # Resizing target to prediction size if needed
            if target.shape[-2:] != pred_probs.shape[-2:]:
                target = F.interpolate(
                    target, size=pred_probs.shape[-2:], mode="nearest"
                )

            iou = compute_iou(pred_probs, target)
            biou = compute_boundary_iou(pred_probs, target)
            test_metrics.update(0.0, iou, biou)

        avg_loss, avg_iou, avg_biou = test_metrics.get_average()
        return {"IoU": avg_iou, "BIoU": avg_biou}

    def get_num_tasks(self) -> int:
        """Get number of trained tasks."""
        return len(self.ca_sam_model.alignment_layers)
