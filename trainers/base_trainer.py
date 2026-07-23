"""
Base Trainer for Multi-Model Training Framework

This module provides a base class for training different models with common functionalities.
"""

import os
import json
import subprocess
import numbers
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
import wandb

from utils.logger import setup_logger
from utils.evaluate import Evaluator_seg
from utils.utils import set_seed
from utils.checkpoint import load_model_weights
from utils.data_processing import SegDatasetProcessor
from utils.wandb_utils import resolve_wandb_identity, build_experiment_dir

_UNSET = object()  # sentinel for "use config default"



class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "max"):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics like Dice (higher is better), 'min' for loss (lower is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """Check if score improved."""
        if self.best_score is None:
            self.best_score = score
            return True

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

    def should_stop(self) -> bool:
        """Check if training should stop."""
        return self.early_stop


class BaseTrainer(ABC):
    """
    Base trainer class that provides common training infrastructure.

    All model-specific trainers should inherit from this class and implement
    the abstract methods for model creation, data loading, and training logic.
    """

    # =========================================================================
    # 1. Construction
    # =========================================================================

    def __init__(self, cfg: DictConfig, model: Optional[nn.Module] = None):
        """
        Initialize base trainer.

        Args:
            cfg: Configuration object (OmegaConf DictConfig)
            model: Pre-built model instance (optional)
        """
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize attributes
        self.model = model
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.logger = None
        self.evaluator = Evaluator_seg()
        self.early_stopping = None

        # Directories
        self.exp_dir = None
        self.ckpt_dir = None
        self.log_dir = None

        # Training state
        self.best_metric = 0.0      # best Dice
        self.best_iou = 0.0
        self.best_hd95 = float("inf")
        self.best_model_path = None
        self.best_iou_checkpoint = None
        self.best_hd95_checkpoint = None
        self.current_epoch = 0
        self.global_step = 0

    # =========================================================================
    # 2. Setup (directories / logging / wandb)
    # =========================================================================

    def _setup_directories(self):
        """
        Setup experiment directories.

        Structure: ``logs/train/{model_name}/[{group}/]{run_name}``. ``run_name``
        is ``self._identity["dir_name"]`` — the same hyperparameter tags and
        timestamp used for the W&B run name, so folder and run are
        self-consistent. ``group`` only appears when a sweep pins an explicit
        ``wandb.group`` (ad hoc single runs stay flat). See
        ``utils.wandb_utils`` for the shared identity resolution.
        """
        model_name = self.cfg.model.name
        self.exp_dir_name = self._identity["dir_name"]
        self.exp_dir = build_experiment_dir(
            self.cfg, root_segment=model_name, identity=self._identity
        )
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint directory
        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Log directory
        self.log_dir = self.exp_dir

        # Save config
        config_file = self.exp_dir / "config.yaml"
        with open(config_file, "w") as f:
            OmegaConf.save(self.cfg, f)

    def _setup_logger(self):
        """Setup logger."""
        log_file = self.exp_dir / "train.log"
        self.logger = setup_logger(str(log_file), logger_name="medfm.train")

    def _setup_wandb(self):
        """Setup Weights & Biases logging.

        Run identity (project/group/name/tags) is composed by the shared helper
        in ``utils.wandb_utils`` so supervised and distillation runs follow one
        consistent scheme; reuses ``self._identity`` (resolved once in
        ``setup()``) so the run name matches the directory already created by
        ``_setup_directories``. The full resolved config is logged as the run
        config.
        """
        identity = {k: v for k, v in self._identity.items() if k != "dir_name"}

        self.wandb_run = wandb.init(
            config=self._build_wandb_config(),
            dir=str(self.exp_dir),
            **identity,
        )

        self._define_wandb_metrics()

    def _define_wandb_metrics(self) -> None:
        """Bind each metric family to its proper x-axis so charts render correctly.

        W&B ``define_metric`` only accepts a single trailing ``*`` glob, which
        matches the remainder of the key (including ``/``). So ``train/*`` is
        enough to cover ``train/loss`` and ``train/BUID/Dice`` alike.
        """
        if wandb.run is None:
            return

        wandb.define_metric("epoch")
        wandb.define_metric("global_step")

        epoch_families = ("train", "val", "test", "final_test")
        step_families = ("step",)

        def _patterns(family: str) -> tuple[str, ...]:
            return (f"{family}/*", f"distill/{family}/*")

        for fam in epoch_families:
            for pat in _patterns(fam):
                wandb.define_metric(pat, step_metric="epoch")

        for fam in step_families:
            for pat in _patterns(fam):
                wandb.define_metric(pat, step_metric="global_step")

        for pat in ("lr", "lr/*", "distill/lr", "distill/lr/*"):
            wandb.define_metric(pat, step_metric="global_step")

    def _build_wandb_config(self) -> Dict[str, Any]:
        """Build the W&B run config: the full resolved Hydra config.

        Logging the whole config avoids the drift that hand-maintained
        ``wandb.log_config`` blocks suffered from; the run identity (name/group/
        tags) is handled separately by :func:`utils.wandb_utils.resolve_wandb_identity`.
        """
        result = OmegaConf.to_container(self.cfg, resolve=True)
        return result if isinstance(result, dict) else {}

    def _setup_early_stopping(self):
        """Setup early stopping."""
        early_stop_cfg = self.cfg.get("training", {}).get("early_stopping", {})

        if early_stop_cfg.get("enabled", False):
            patience = early_stop_cfg.get("patience", 15)
            min_delta = early_stop_cfg.get("min_delta", 0.001)
            self.early_stopping = EarlyStopping(
                patience=patience, min_delta=min_delta, mode="max"
            )
            self.logger.info(
                f"Early stopping enabled: patience={patience}, min_delta={min_delta}"
            )

    # =========================================================================
    # 3. Model / optimizer construction
    # =========================================================================

    @abstractmethod
    def _create_model(self):
        """Create and initialize model. Must be implemented by subclasses."""
        raise NotImplementedError

    def _create_dataloaders(self):
        """Create data loaders."""
        self.train_loader, self.val_loader, self.test_loader = (
            SegDatasetProcessor.build_data_loaders(self.cfg)
        )
        self._log_data_sizes()

    @abstractmethod
    def _create_optimizer(self):
        """Create optimizer. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def _create_scheduler(self):
        """Create learning rate scheduler. Must be implemented by subclasses."""
        raise NotImplementedError

    def _log_data_sizes(self):
        """Log dataset sizes. Call from _create_dataloaders()."""
        self.logger.info(f"Train set size: {len(self.train_loader.dataset)}")
        self.logger.info(f"Val set size: {len(self.val_loader.dataset)}")
        if isinstance(self.test_loader, dict):
            total = sum(len(l.dataset) for l in self.test_loader.values())
            self.logger.info(f"Test set size (Total): {total}")
        for name, loader in self._iter_test_loaders():
            prefix = "  - " if isinstance(self.test_loader, dict) else ""
            self.logger.info(f"{prefix}{name}: {len(loader.dataset)}")

    def _log_model_info(self):
        """Log total and trainable parameter counts."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total:,}")
        self.logger.info(f"Trainable parameters: {trainable:,}")
        if wandb.run is not None:
            wandb.summary["model/total_params"] = total
            wandb.summary["model/trainable_params"] = trainable

    def _get_model_type(self) -> str:
        """Return the model_type string for visualization inference.

        Subclasses can override this to specify 'sam', 'segformer', etc.
        """
        return "default"

    # =========================================================================
    # 4. Training step (abstract)
    # =========================================================================

    @abstractmethod
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        raise NotImplementedError

    # =========================================================================
    # 5. Evaluation (abstract)
    # =========================================================================

    @abstractmethod
    def validate(self, epoch: int, return_predictions: bool = False):
        """
        Validate model.

        Args:
            epoch: Current epoch number
            return_predictions: If True, also return a predictions cache reusable
                by ``_visualize_validation`` so visualization shares the same
                forward pass as metric computation.

        Returns:
            If ``return_predictions`` is False, a dict of validation metrics.
            Otherwise, a tuple ``(metrics, predictions_cache)`` where the cache
            has the form ``{"__val__": (images_list, preds_list, masks_list, filenames_list)}``.
        """
        raise NotImplementedError

    @abstractmethod
    def test(self) -> Dict[str, float]:
        """
        Test model.

        Returns:
            Dictionary of test metrics
        """
        raise NotImplementedError

    def _iter_test_loaders(self):
        """Iterate over test loaders yielding (name, loader) tuples."""
        if isinstance(self.test_loader, dict):
            yield from self.test_loader.items()
        else:
            yield "test", self.test_loader

    # =========================================================================
    # 6. Metrics logging & checkpointing
    # =========================================================================

    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict,
        test_metrics: Dict = None,
    ):
        """Log training and validation metrics."""
        num_epochs = self.cfg.training.get("num_epochs", 100)
        train_items = {k: v for k, v in train_metrics.items() if isinstance(v, (float, int))}
        val_items = {k: v for k, v in val_metrics.items() if isinstance(v, (float, int))}

        # Log to console (val metrics are already summarized by Evaluator_seg)
        self.logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        self.logger.info(
            "Train:\n    " + ", ".join(f"{k}: {v:.4f}" for k, v in train_items.items())
        )

        # Log to wandb (epoch-level metrics)
        wandb_metrics = {"epoch": epoch + 1}
        wandb_metrics.update({f"train/{k}": v for k, v in train_items.items()})
        wandb_metrics.update({f"val/{k}": v for k, v in val_items.items()})
        if wandb.run is not None:
            wandb.log(self._round_log_values(wandb_metrics))

    def _log_step_metrics(self, metrics: Dict, step: int):
        """Log step-level metrics to wandb with step/ prefix."""
        if wandb.run is not None:
            data = {"global_step": step}
            data.update({f"step/{k}": v for k, v in metrics.items()})
            wandb.log(self._round_log_values(data))

    def _supervised_epoch(self, model, compute_batch_loss, clip_params_fn, cadence, epoch):
        """One supervised training epoch, shared by the normal trainers and the
        distill task_only path so the two run bit-identical code.

        ``compute_batch_loss(model, batch, device) -> (loss, loss_dict)`` does the
        model's forward + supervised loss; this loop owns AMP (autocast +
        GradScaler), gradient clipping, the optimizer step, and per-batch LR
        stepping. AMP is a no-op (exact fp32) unless ``self.amp_enabled``.
        """
        from tqdm import tqdm

        model.train()
        totals: Dict[str, float] = {}
        num_epochs = self.cfg.training.num_epochs
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in pbar:
            self.optimizer.zero_grad()
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=getattr(self, "amp_enabled", False),
                dtype=getattr(self, "amp_dtype", torch.bfloat16),
            ):
                loss, loss_dict = compute_batch_loss(model, batch, self.device)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            max_norm = self.cfg.optimizer.get("gradient_clip", {}).get("max_norm", 1.0)
            nn.utils.clip_grad_norm_(clip_params_fn(), max_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if cadence == "batch" and self.scheduler is not None:
                self.scheduler.step()

            for k, v in loss_dict.items():
                totals[k] = totals.get(k, 0.0) + (v.item() if hasattr(v, "item") else float(v))
            lr = self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.6f}"})
            if self.global_step % 10 == 0:
                self._log_step_metrics(
                    {**{k: (v.item() if hasattr(v, "item") else float(v))
                        for k, v in loss_dict.items()}, "lr": lr},
                    self.global_step,
                )
            self.global_step += 1

        n = len(self.train_loader)
        out = {k: v / n for k, v in totals.items()}
        out["lr"] = self.optimizer.param_groups[0]["lr"]
        return out

    @staticmethod
    def _round_log_values(data: Dict, ndigits: int = 4) -> Dict:
        """Round float values before sending them to W&B.

        Only real (non-integer) numbers are rounded, so ``epoch``/``global_step``
        and parameter counts stay intact. Learning-rate keys are skipped so small
        values such as ``4e-5`` are not flattened to ``0.0`` by 4-decimal rounding.
        """
        def _skip(key) -> bool:
            return "lr" in str(key).split("/")

        out = {}
        for k, v in data.items():
            if (
                isinstance(v, numbers.Real)
                and not isinstance(v, (bool, int))
                and not _skip(k)
            ):
                out[k] = round(float(v), ndigits)
            else:
                out[k] = v
        return out

    def _save_checkpoint(self, epoch: int, val_metrics: Dict):
        """Save model checkpoint."""
        dice_score = val_metrics.get("Dice", 0.0)
        IoU_score = val_metrics.get("IoU", 0.0)
        hd95_score = val_metrics.get("HD95", 0.0)

        prev_best_model_path = self.best_model_path
        prev_best_iou_checkpoint = getattr(self, "best_iou_checkpoint", None)
        prev_best_hd95_checkpoint = getattr(self, "best_hd95_checkpoint", None)

        # Save best model (Dice)
        if dice_score > self.best_metric:
            self.best_metric = dice_score
            self.best_model_path = (
                self.ckpt_dir / f"best_epoch_{epoch + 1}_dice_{dice_score:.4f}.pth"
            )
            self._save_model(self.best_model_path)
            self.logger.info(f"Saved best model: {self.best_model_path}")
            if wandb.run is not None:
                wandb.run.summary["checkpoint_path"] = str(self.best_model_path)
                wandb.run.summary["best_dice"] = round(float(dice_score), 4)
            if (
                prev_best_model_path is not None
                and prev_best_model_path != self.best_model_path
                and prev_best_model_path.exists()
            ):
                prev_best_model_path.unlink()

        # Save best model (IoU) — tracked independently of wandb
        if IoU_score > self.best_iou:
            self.best_iou = IoU_score
            self.best_iou_checkpoint = (
                self.ckpt_dir / f"best_epoch_{epoch + 1}_iou_{IoU_score:.4f}.pth"
            )
            self._save_model(self.best_iou_checkpoint)
            self.logger.info(f"Saved best IoU model: {self.best_iou_checkpoint}")
            if wandb.run is not None:
                wandb.run.summary["best_iou"] = round(float(IoU_score), 4)
                wandb.run.summary["best_iou_checkpoint"] = str(self.best_iou_checkpoint)
            if (
                prev_best_iou_checkpoint is not None
                and prev_best_iou_checkpoint != self.best_iou_checkpoint
                and prev_best_iou_checkpoint.exists()
            ):
                prev_best_iou_checkpoint.unlink()

        # Save best model (HD95) — tracked independently of wandb
        if hd95_score < self.best_hd95:
            self.best_hd95 = hd95_score
            self.best_hd95_checkpoint = (
                self.ckpt_dir / f"best_epoch_{epoch + 1}_hd95_{hd95_score:.4f}.pth"
            )
            self._save_model(self.best_hd95_checkpoint)
            self.logger.info(f"Saved best HD95 model: {self.best_hd95_checkpoint}")
            if wandb.run is not None:
                wandb.run.summary["best_hd95"] = round(float(hd95_score), 4)
                wandb.run.summary["best_hd95_checkpoint"] = str(self.best_hd95_checkpoint)
            if (
                prev_best_hd95_checkpoint is not None
                and prev_best_hd95_checkpoint != self.best_hd95_checkpoint
                and prev_best_hd95_checkpoint.exists()
            ):
                prev_best_hd95_checkpoint.unlink()

        # Periodic checkpoint
        save_interval = self.cfg.get("training", {}).get("save_interval", 20)
        if (epoch + 1) % save_interval == 0:
            ckpt_path = self.ckpt_dir / f"epoch_{epoch + 1}.pth"
            self._save_model(ckpt_path)
            self.logger.info(f"Saved checkpoint: {ckpt_path}")

    def _save_model(self, path: Path):
        """Save model to path. Can be overridden by subclasses for custom saving."""
        torch.save(self.model.state_dict(), str(path))

    def _load_checkpoint(self, path: Path):
        """Load model from checkpoint (any project schema). Overridable by subclasses."""
        self.logger.info(f"Loading checkpoint: {path}")
        load_model_weights(self.model, path, map_location=self.device, strict=True)

    # =========================================================================
    # 7. Visualization helpers
    # =========================================================================

    def _run_vis_inference(self, loader, num_samples: Optional[int] = None):
        """Run forward passes over ``loader`` and collect tensors for visualization.

        Returns ``(images_list, preds_list, masks_list, filenames_list)`` where
        ``preds_list`` items are dicts ``{"pred": tensor[B,1,H,W]}``.
        """
        import torch
        import torch.nn.functional as F
        from tqdm import tqdm

        model_type = self._get_model_type()
        img_size = getattr(self, "img_size", None)
        num_classes = self.cfg.data.num_classes

        self.model.eval()
        images_list, preds_list, masks_list, fnames_list = [], [], [], []
        collected = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc="Visualization inference"):
                images = batch[0].to(self.device)
                masks = batch[1].to(self.device)
                last = batch[-1]
                fnames = (
                    list(last)
                    if isinstance(last, (list, tuple)) and last and isinstance(last[0], str)
                    else None
                )

                if model_type == "sam":
                    outputs = self.model(images, num_classes > 1, img_size)
                    logits = outputs.get("masks", outputs.get("low_res_logits"))
                    if logits is None:
                        logits = list(outputs.values())[0]
                elif model_type == "segformer":
                    logits = self.model(images).logits
                else:
                    logits = self.model(images)
                    if logits.dim() == 3:
                        logits = logits.unsqueeze(1)

                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(
                        logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                    )

                if num_classes == 1:
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    masks_vis = masks
                else:
                    preds = torch.argmax(logits, dim=1, keepdim=True).float()
                    masks_vis = torch.argmax(masks, dim=1, keepdim=True).float()

                images_list.append(images.cpu())
                preds_list.append({"pred": preds.cpu()})
                masks_list.append(masks_vis.cpu())
                fnames_list.append(fnames)

                collected += images.size(0)
                if num_samples is not None and collected >= num_samples:
                    break

        if not any(fnames_list):
            fnames_list = None
        return images_list, preds_list, masks_list, fnames_list

    def _visualize_predictions(
        self,
        loader=None,
        vis_dir=None,
        predictions_cache: Dict = None,
        num_samples=_UNSET,
        epoch: int = None,
        log_to_wandb: bool = True,
        max_wandb_images: Optional[int] = None,
    ):
        """Visualize predictions with a unified implementation.

        Supports three modes:
        1. loader provided -> run model inference on loader (validation).
        2. predictions_cache provided -> use pre-computed results (test).
        3. Neither -> iterate test_loaders with model inference (fallback).

        Args:
            num_samples: Max samples to visualize. Defaults to config value.
                         Pass None explicitly to visualize all samples.
            log_to_wandb: If True, upload saved images to W&B.
            max_wandb_images: Max number of images to upload to W&B.
                Does not affect how many files are saved to disk.
        """
        from utils.visualize import visualize_segmentation

        if num_samples is _UNSET:
            num_vis_samples = self.cfg.get("visualization", {}).get("num_samples", 10)
        else:
            num_vis_samples = num_samples

        # Mode 1: validation visualization
        if loader is not None or (predictions_cache is not None and "__val__" in predictions_cache):
            if predictions_cache is not None and "__val__" in predictions_cache:
                cached = predictions_cache["__val__"]
                images_l, preds_raw, masks_l = cached[0], cached[1], cached[2]
                fnames_l = cached[3] if len(cached) > 3 else None
                preds_l = [{"pred": p} for p in preds_raw]
            else:
                images_l, preds_l, masks_l, fnames_l = self._run_vis_inference(
                    loader, num_samples=num_vis_samples
                )
            visualize_segmentation(
                images_l, preds_l, masks_l,
                num_classes=self.cfg.data.num_classes,
                save_dir=vis_dir,
                num_samples=num_vis_samples,
                phase_name="val",
                filenames_list=fnames_l,
                log_to_wandb=log_to_wandb,
                max_wandb_images=max_wandb_images,
            )
            return

        # Mode 2/3: test visualization
        if vis_dir is None:
            vis_dir = self.exp_dir / "visualizations" / f"epoch_{self.current_epoch + 1}"
        if log_to_wandb and max_wandb_images is None:
            max_wandb_images = 10

        sample_msg = "all" if num_vis_samples is None else f"{num_vis_samples}"
        self.logger.info(
            f"Generating {sample_msg} visualizations for epoch {self.current_epoch + 1}..."
        )

        for name, test_loader in self._iter_test_loaders():
            save_dir = vis_dir / name if isinstance(self.test_loader, dict) else vis_dir
            phase_name = f"test_{name}" if isinstance(self.test_loader, dict) else "test"

            if predictions_cache and name in predictions_cache:
                cached = predictions_cache[name]
                images_l, preds_raw, masks_l = cached[0], cached[1], cached[2]
                fnames_l = cached[3] if len(cached) > 3 else None
                preds_l = [{"pred": p} for p in preds_raw]
            else:
                images_l, preds_l, masks_l, fnames_l = self._run_vis_inference(
                    test_loader, num_samples=num_vis_samples
                )

            visualize_segmentation(
                images_l, preds_l, masks_l,
                num_classes=self.cfg.data.num_classes,
                save_dir=save_dir,
                num_samples=num_vis_samples,
                phase_name=phase_name,
                filenames_list=fnames_l,
                log_to_wandb=log_to_wandb,
                max_wandb_images=max_wandb_images,
            )

        self.logger.info(f"Visualizations saved to {vis_dir}")

    def _visualize_validation(self, epoch: int, predictions_cache: Dict = None):
        """Visualize validation predictions.

        If ``predictions_cache`` is provided (with key ``"__val__"``), reuse those
        predictions instead of running another forward pass — keeps visualization
        consistent with the metrics computed in ``validate()``.
        """
        vis_dir = self.exp_dir / "visualizations" / f"epoch_{epoch + 1}_val"
        vis_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            f"Generating validation visualizations for epoch {epoch + 1}..."
        )
        if predictions_cache is not None:
            self._visualize_predictions(
                predictions_cache=predictions_cache,
                vis_dir=vis_dir,
                epoch=epoch,
                log_to_wandb=False,
            )
        else:
            self._visualize_predictions(
                loader=self.val_loader,
                vis_dir=vis_dir,
                epoch=epoch,
                log_to_wandb=False,
            )

    # =========================================================================
    # 8. Test result persistence
    # =========================================================================

    def _save_test_results(self, test_metrics: Dict, predictions_cache: dict = None):
        """Save test results to file and log to wandb as final_test/.

        Args:
            test_metrics: Dict of metric name → scalar value. Keys may be prefixed
                with dataset name (e.g. ``BUID/Dice``) for multi-dataset runs.
            predictions_cache: Optional precomputed predictions cache. When provided,
                per-sample CSV files are written via _save_per_sample_metrics.
        """
        test_results_path = self.exp_dir / "test_results.txt"
        with open(test_results_path, "w") as f:
            f.write("Test Results\n")
            f.write(f"Best Model: {self.best_model_path.name}\n")
            for key, value in test_metrics.items():
                if isinstance(value, (float, int)):
                    f.write(f"{key}: {value:.4f}\n")

        self.logger.info(f"Test results saved to {test_results_path}")

        # Log to wandb with final_test/ prefix
        if wandb.run is not None:
            final_metrics = {
                f"final_test/{k}": v
                for k, v in test_metrics.items()
                if isinstance(v, (float, int))
            }
            # final_test/* is bound to step_metric="epoch"; pass an explicit
            # "epoch" so these points land on the run's final epoch on the
            # x-axis instead of W&B's last-seen internal step.
            rounded_final = self._round_log_values(final_metrics)
            wandb.log({**rounded_final, "epoch": self.cfg.training.get("num_epochs", 0)})
            wandb.run.summary.update(rounded_final)

        # Save LaTeX table from test_metrics (authoritative values from evaluate_model)
        self._save_latex_metrics_table_from_metrics(test_metrics)

        # Machine-readable aggregate (full precision) for cross-run analysis.
        self._save_final_metrics_json(test_metrics)

        if predictions_cache is not None:
            self._save_per_sample_metrics(predictions_cache)

    def _save_final_metrics_json(self, test_metrics: Dict) -> None:
        """Persist final-test metrics as structured JSON for cross-run analysis.

        Replaces brittle parsing of ``test_results.txt``: metrics are nested as
        ``{dataset: {metric: value}}`` (dataset split from ``"<ds>/<metric>"``
        keys; flat keys go under ``"_overall"``) at full precision, alongside a
        ``_meta`` block (run label, models, key hyperparameters, best ckpt,
        seed, git commit) so a run is self-describing.
        """
        per_ds: Dict[str, Dict[str, float]] = {}
        for key, value in test_metrics.items():
            if not isinstance(value, numbers.Number):
                continue
            ds, _, metric = key.partition("/")
            if not metric:  # flat key (single-dataset run)
                ds, metric = "_overall", key
            per_ds.setdefault(ds, {})[metric] = float(value)

        m = self.cfg.get("method", {})
        meta = {
            "run_name": self.cfg.get("run_name"),
            "method": m.get("name"),
            "teacher": self.cfg.get("teacher"),
            "student": self.cfg.get("student"),
            "num_classes": self.cfg.get("data", {}).get("num_classes"),
            # Record the EFFECTIVE seed actually used (same default as the setup()
            # set_seed call, base_trainer.py). The bare .get() logged null when a
            # config omitted hardware.seed even though the run ran at 42 —
            # misrepresenting the run in final_metrics.json.
            "seed": self.cfg.get("hardware", {}).get("seed", 42),
            "best_checkpoint": (
                self.best_model_path.name if self.best_model_path else None
            ),
            "git_commit": self._git_commit(),
            "hparams": {
                "temperature": m.get("temperature"),
                "w_task": m.get("w_task"),
                "w_logit_kd": m.get("w_logit_kd"),
                "w_reliability_kd": m.get("w_reliability_kd"),
                "w_uncertainty_kd": m.get("w_uncertainty_kd"),
                "lr": self.cfg.get("training", {}).get("lr"),
                "batch_size": self.cfg.get("training", {}).get("batch_size"),
            },
        }
        payload = {"_meta": meta, "metrics": per_ds}
        out_path = self.exp_dir / "final_metrics.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        self.logger.info(f"Final metrics JSON saved → {out_path}")

    @staticmethod
    def _git_commit() -> Optional[str]:
        """Short git commit hash of the working tree, or None if unavailable."""
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            return None

    def _save_per_sample_metrics(self, predictions_cache: dict) -> None:
        """Save per-sample segmentation metrics to CSV for every test dataset.

        Reuses the per-sample lists already computed inside ``evaluate_model``
        (no recomputation), so this is essentially I/O-bound.

        Args:
            predictions_cache: Dict mapping dataset name →
                ``(images_list, preds_list, masks_list, filenames_list, per_sample)``
                where ``per_sample`` is the metrics dict produced by
                ``Evaluator_seg._evaluate_*`` when ``return_predictions=True``.
        """
        import pandas as pd

        out_dir = self.exp_dir / "test_results"
        out_dir.mkdir(parents=True, exist_ok=True)

        all_dfs = []
        for ds_name, cached in predictions_cache.items():
            if len(cached) < 5 or cached[4] is None:
                self.logger.warning(
                    f"Per-sample metrics unavailable for {ds_name}; skipping CSV."
                )
                continue
            per_sample = cached[4]
            df = self._per_sample_dict_to_df(per_sample)
            df.insert(0, "dataset", ds_name)
            csv_path = out_dir / f"metrics_{ds_name}.csv"
            df.to_csv(csv_path, index=False, float_format="%.6f")
            self.logger.info(f"Per-sample metrics saved → {csv_path}")
            all_dfs.append(df)

        if len(all_dfs) > 1:
            combined_path = out_dir / "metrics_all.csv"
            pd.concat(all_dfs, ignore_index=True).to_csv(
                combined_path, index=False, float_format="%.6f"
            )
            self.logger.info(f"Combined per-sample metrics saved → {combined_path}")

    @staticmethod
    def _per_sample_dict_to_df(per_sample: dict):
        """Flatten the per-sample dict (incl. optional ``_per_class``) into a DataFrame."""
        import pandas as pd

        flat = {k: v for k, v in per_sample.items() if k != "_per_class"}
        df = pd.DataFrame(flat)

        per_class = per_sample.get("_per_class")
        if per_class:
            class_ids = sorted({cid for d in per_class for cid in d.keys()})
            for cid in class_ids:
                for metric in ("Dice", "HD95", "IoU", "BoundaryIoU", "Sensitivity", "Specificity"):
                    col = f"{metric}_class_{cid}"
                    df[col] = [d.get(cid, {}).get(metric, float("nan")) for d in per_class]
        return df

    def _save_latex_metrics_table_from_metrics(self, test_metrics: Dict) -> None:
        """Save a LaTeX table from the authoritative test_metrics dict.

        Uses the mean/std values already computed by evaluate_model (same values
        as test_results.txt and wandb), avoiding re-aggregation from CSV.

        test_metrics keys follow two formats:
          - Multi-dataset: ``<dataset>/<Metric>`` and ``<dataset>/<Metric>_std``
          - Single-dataset: ``<Metric>`` and ``<Metric>_std``

        Each cell: ``mean \\pm std`` (×100 for ratio metrics, raw for HD95).
        """
        import re

        RATIO_COLS = {"Dice", "IoU", "BIoU", "Sensitivity", "Specificity", "PixelAcc"}
        BASE_COLS = ["Dice", "HD95", "IoU", "BIoU", "Sensitivity", "Specificity", "PixelAcc"]

        # Determine if keys are prefixed with dataset name
        has_prefix = any("/" in k for k in test_metrics)

        # Group metrics by dataset
        datasets: Dict[str, Dict[str, float]] = {}
        for key, value in test_metrics.items():
            if not isinstance(value, (int, float)):
                continue
            if has_prefix:
                ds_name, metric = key.split("/", 1)
            else:
                ds_name, metric = "test", key
            datasets.setdefault(ds_name, {})[metric] = value

        if not datasets:
            return

        # Collect all column names (union across datasets, preserving order)
        def _col_order(ds_metrics):
            # Detect per-class ids from keys like Dice_class_1
            class_ids = sorted({
                int(m.group(1))
                for k in ds_metrics
                for m in [re.search(r"_class_(\d+)(?:_std)?$", k)]
                if m
            })
            cols = [c for c in BASE_COLS if c in ds_metrics or f"{c}_std" in ds_metrics]
            for c in class_ids:
                for base in ["Dice", "HD95", "IoU"]:
                    col = f"{base}_class_{c}"
                    if col in ds_metrics or f"{col}_std" in ds_metrics:
                        cols.append(col)
            return cols

        # Use column list from first dataset
        first_ds_metrics = next(iter(datasets.values()))
        final_cols = _col_order(first_ds_metrics)

        if not final_cols:
            return

        rows = []
        for ds_name, ds_metrics in datasets.items():
            cells = []
            for col in final_cols:
                mean_v = ds_metrics.get(col)
                std_v = ds_metrics.get(f"{col}_std")
                if mean_v is None:
                    cells.append("--")
                    continue
                base_metric = col.split("_class_")[0]
                if base_metric in RATIO_COLS:
                    std_str = f" $\\pm$ {std_v * 100:.2f}" if std_v is not None else ""
                    cells.append(f"{mean_v * 100:.2f}{std_str}")
                else:
                    std_str = f" $\\pm$ {std_v:.2f}" if std_v is not None else ""
                    cells.append(f"{mean_v:.2f}{std_str}")
            rows.append((ds_name, cells))

        out_dir = self.exp_dir / "test_results"
        out_dir.mkdir(parents=True, exist_ok=True)

        header_cols = " & ".join(final_cols)
        lines = [
            "% Auto-generated LaTeX metrics table",
            "\\begin{table}[h]",
            "  \\centering",
            "  \\caption{Final test metrics (mean $\\pm$ std)}",
            "  \\begin{tabular}{l" + "c" * len(final_cols) + "}",
            "    \\hline",
            f"    Dataset & {header_cols} \\\\",
            "    \\hline",
        ]
        for ds_name, cells in rows:
            lines.append(f"    {ds_name} & {' & '.join(cells)} \\\\")
        lines += [
            "    \\hline",
            "  \\end{tabular}",
            "\\end{table}",
        ]

        out_path = out_dir / "metrics_latex.txt"
        out_path.write_text("\n".join(lines) + "\n")
        self.logger.info(f"LaTeX metrics table saved → {out_path}")

    # =========================================================================
    # 9. Entry points (main loops)
    # =========================================================================

    def setup(self, mode: str = "train"):
        """
        Setup training environment.

        Args:
            mode: 'train' or 'test'
        """
        # Set random seeds
        seed = self.cfg.get("hardware", {}).get("seed", 42)
        deterministic = self.cfg.get("hardware", {}).get("deterministic", True)
        set_seed(seed, deterministic=deterministic)

        # Resolve run identity once: the W&B name/group/tags and the on-disk
        # directory name are derived from this single call (one timestamp, one
        # hparam-tag computation) so they never drift apart.
        self._identity = resolve_wandb_identity(self.cfg, default_job_type="train")

        # Setup directories
        self._setup_directories()

        # Setup logger
        self._setup_logger()

        # Setup wandb
        if mode == "train":
            self._setup_wandb()

        # Create data loaders
        self._create_dataloaders()

        # Create model
        self._create_model()

        # Setup training components (only for training mode)
        if mode == "train":
            self._create_optimizer()
            self._create_scheduler()
            self._setup_early_stopping()

        self.logger.info(f"Setup completed for {mode} mode")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Experiment directory: {self.exp_dir}")

    def train(self):
        """Main training loop."""
        self.logger.info("Starting training")

        num_epochs = self.cfg.training.get("num_epochs", 100)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate (request predictions on visualization epochs to avoid a 2nd forward pass)
            should_visualize = (epoch + 1) % 5 == 0
            val_result = self.validate(epoch, return_predictions=should_visualize)
            if isinstance(val_result, tuple):
                val_metrics, val_predictions_cache = val_result
            else:
                val_metrics, val_predictions_cache = val_result, None

            # Visualize validation every 5 epochs (reuse predictions from validate())
            if should_visualize:
                self._visualize_validation(epoch, predictions_cache=val_predictions_cache)

            # Log metrics (removed test_metrics from per-epoch loop)
            self._log_metrics(epoch, train_metrics, val_metrics)

            # Save checkpoint
            self._save_checkpoint(epoch, val_metrics)

            # Early stopping check
            if self.early_stopping is not None:
                self.early_stopping(
                    val_metrics.get("Dice", val_metrics.get("dice", 0.0))
                )

                if self.early_stopping.should_stop():
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        # Test with best model
        if self.best_model_path is not None:
            self.logger.info("Testing with best model")
            self._load_checkpoint(self.best_model_path)
            result = self.test()
            if isinstance(result, tuple):
                test_metrics, predictions_cache = result
            else:
                test_metrics, predictions_cache = result, None
            self._save_test_results(test_metrics, predictions_cache=predictions_cache)
        else:
            self.logger.warning(
                "No best model checkpoint found (training may have ended at epoch 0 "
                "or all checkpoints failed). Skipping final test evaluation."
            )

        self.logger.info("Training completed!")

    def run_test_only(self, checkpoint_path: str):
        """Run test-only mode with a specific checkpoint."""
        self.logger.info("TEST-ONLY MODE")
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")

        # Load checkpoint
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self._load_checkpoint(checkpoint_path)

        # Initialize wandb for test-only run (same identity scheme, eval job).
        identity = resolve_wandb_identity(self.cfg, default_job_type="eval")
        identity["tags"] = ["test-only", *identity["tags"]]
        identity.pop("dir_name", None)
        self.wandb_run = wandb.init(
            config=self._build_wandb_config(),
            **identity,
        )

        # Run test
        test_metrics = self.test()

        # Save results
        self.best_model_path = checkpoint_path
        self._save_test_results(test_metrics)

        wandb.finish()
        self.logger.info("Test-only evaluation completed!")
