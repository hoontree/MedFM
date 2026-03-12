"""
PyTorch Lightning Module for SAM-based Medical Image Segmentation.

Wraps the existing SAM training logic (SAMTrainer) into a LightningModule,
preserving all loss functions, metrics, and model behaviour.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from medpy.metric.binary import dc, hd95
from medpy.metric.binary import recall as medpy_recall
from omegaconf import DictConfig
from torch.nn.modules.loss import CrossEntropyLoss

import lightning as L

from utils.evaluate import Evaluator_seg
from utils.sam_utils import DiceLoss

log = logging.getLogger(__name__)


class SAMLitModule(L.LightningModule):
    """Lightning wrapper around the SAM segmentation model.

    Re-uses the same loss calculation, optimizer, and scheduler logic
    from ``trainers.sam_trainer.SAMTrainer`` so that the training
    semantics are identical.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # ---- hyper-parameters (mirror SAMTrainer) ----
        self.img_size: int = cfg.model.img_size
        self.num_classes: int = cfg.data.num_classes
        self.num_epochs: int = cfg.training.get("num_epochs", 100)
        self.base_lr: float = float(cfg.training.get("lr", 1e-4))
        self.dice_loss_weight: float = float(cfg.training.get("dice_loss_weight", 0.8))
        self.moe_loss_weight: float = float(
            cfg.get("training", {}).get("moe_loss_weight", 0.0)
        )
        self.gradient_clip_max_norm = float(
            cfg.optimizer.get("gradient_clip", {}).get("max_norm", 1.0)
        )
        self.warmup_config = cfg.training.get(
            "warmup", {"enabled": False, "steps": 0}
        )
        self.step_log_interval: int = 10

        # ---- model ----
        self.model: nn.Module = instantiate(cfg.model)
        self._log_param_counts()

        # ---- losses ----
        self.ce_loss = CrossEntropyLoss()
        pos_weight = torch.tensor([5.0])
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice_loss = DiceLoss(self.num_classes)

        # ---- evaluator (for val / test) ----
        self.evaluator = Evaluator_seg()

        # ---- save hyper-parameters (logged to W&B / TB automatically) ----
        self.save_hyperparameters(ignore=["cfg"])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(images, False, self.img_size)

    # ------------------------------------------------------------------
    # Loss (identical to SAMTrainer._calc_loss)
    # ------------------------------------------------------------------
    def _calc_loss(self, outputs, low_res_label_batch):
        dice_weight = self.dice_loss_weight

        logits = outputs["low_res_logits"]
        target = low_res_label_batch

        if logits.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")

        loss_ce = self.bce_loss(logits, target)
        loss_dice = self.dice_loss(logits, target)
        loss_moe = outputs.get("moe_loss", torch.tensor(0.0, device=logits.device))
        if not torch.is_tensor(loss_moe):
            loss_moe = torch.tensor(float(loss_moe), device=logits.device)

        loss = (
            (1 - dice_weight) * loss_ce
            + dice_weight * loss_dice
            + self.moe_loss_weight * loss_moe
        )
        return loss, loss_ce, loss_dice, loss_moe

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        image_batch, label_batch, low_res_label_batch = batch

        outputs = self.model(image_batch, False, self.img_size)
        loss, loss_ce, loss_dice, loss_moe = self._calc_loss(
            outputs, low_res_label_batch
        )

        # Logging (Lightning handles aggregation automatically)
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_ce", loss_ce, on_step=False, on_epoch=True)
        self.log("train/loss_dice", loss_dice, on_step=False, on_epoch=True)
        self.log("train/loss_moe", loss_moe, on_step=False, on_epoch=True)
        self.log("train/lr", lr, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    # ------------------------------------------------------------------
    # Validation step  (sample-level metrics, aggregated at epoch end)
    # ------------------------------------------------------------------
    def on_validation_epoch_start(self):
        self._val_dice_list = []
        self._val_hd95_list = []
        self._val_iou_list = []
        self._val_sens_list = []
        self._val_spec_list = []
        self._val_pixacc_list = []
        self._val_bf_list = []

    def validation_step(self, batch, batch_idx):
        images, labels, low_res_labels = batch

        outputs = self.model(images, False, self.img_size)
        logits = outputs["masks"]

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        # Per-sample metrics
        for pred, gt in zip(preds, labels):
            pred_np = pred.squeeze().cpu().numpy().astype(bool)
            gt_np = gt.squeeze().cpu().numpy().astype(bool)

            self._val_dice_list.append(dc(pred_np, gt_np))

            if pred_np.any() and gt_np.any():
                self._val_hd95_list.append(hd95(pred_np, gt_np))
            elif not pred_np.any() and not gt_np.any():
                self._val_hd95_list.append(0.0)
            else:
                self._val_hd95_list.append(224.0)

            self._val_iou_list.append(
                Evaluator_seg.compute_jaccard(pred_np, gt_np)
            )
            self._val_sens_list.append(medpy_recall(pred_np, gt_np))
            self._val_spec_list.append(
                Evaluator_seg.compute_specificity(pred_np, gt_np)
            )
            self._val_pixacc_list.append(
                (pred_np == gt_np).sum() / gt_np.size
            )
            self._val_bf_list.append(
                Evaluator_seg.compute_boundary_score(pred_np, gt_np)
            )

    def on_validation_epoch_end(self):
        metrics = {
            "val/Dice": np.mean(self._val_dice_list),
            "val/HD95": np.mean(self._val_hd95_list),
            "val/IoU": np.mean(self._val_iou_list),
            "val/Sensitivity": np.mean(self._val_sens_list),
            "val/Specificity": np.mean(self._val_spec_list),
            "val/PixelAcc": np.mean(self._val_pixacc_list),
            "val/BFScore": np.mean(self._val_bf_list),
        }
        for k, v in metrics.items():
            self.log(k, v, prog_bar=(k == "val/Dice"), sync_dist=True)

    # ------------------------------------------------------------------
    # Test step  (mirrors validation but with final_test/ prefix)
    # ------------------------------------------------------------------
    def on_test_epoch_start(self):
        self._test_dice_list = []
        self._test_hd95_list = []
        self._test_iou_list = []
        self._test_sens_list = []
        self._test_spec_list = []
        self._test_pixacc_list = []
        self._test_bf_list = []

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        images, labels, low_res_labels = batch

        outputs = self.model(images, False, self.img_size)
        logits = outputs["masks"]

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        for pred, gt in zip(preds, labels):
            pred_np = pred.squeeze().cpu().numpy().astype(bool)
            gt_np = gt.squeeze().cpu().numpy().astype(bool)

            self._test_dice_list.append(dc(pred_np, gt_np))
            if pred_np.any() and gt_np.any():
                self._test_hd95_list.append(hd95(pred_np, gt_np))
            elif not pred_np.any() and not gt_np.any():
                self._test_hd95_list.append(0.0)
            else:
                self._test_hd95_list.append(224.0)

            self._test_iou_list.append(
                Evaluator_seg.compute_jaccard(pred_np, gt_np)
            )
            self._test_sens_list.append(medpy_recall(pred_np, gt_np))
            self._test_spec_list.append(
                Evaluator_seg.compute_specificity(pred_np, gt_np)
            )
            self._test_pixacc_list.append(
                (pred_np == gt_np).sum() / gt_np.size
            )
            self._test_bf_list.append(
                Evaluator_seg.compute_boundary_score(pred_np, gt_np)
            )

    def on_test_epoch_end(self):
        metrics = {
            "final_test/Dice": np.mean(self._test_dice_list),
            "final_test/Dice_std": np.std(self._test_dice_list),
            "final_test/HD95": np.mean(self._test_hd95_list),
            "final_test/HD95_std": np.std(self._test_hd95_list),
            "final_test/IoU": np.mean(self._test_iou_list),
            "final_test/IoU_std": np.std(self._test_iou_list),
            "final_test/Sensitivity": np.mean(self._test_sens_list),
            "final_test/Specificity": np.mean(self._test_spec_list),
            "final_test/PixelAcc": np.mean(self._test_pixacc_list),
            "final_test/BFScore": np.mean(self._test_bf_list),
        }
        for k, v in metrics.items():
            self.log(k, v, sync_dist=True)

        # Console summary
        log.info("Test results:")
        for k, v in metrics.items():
            log.info("  %s = %.4f", k, v)

    # ------------------------------------------------------------------
    # Optimizer & Scheduler (mirrors SAMTrainer)
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        cfg = self.cfg

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.base_lr,
            weight_decay=float(cfg.optimizer.get("weight_decay", 0.1)),
        )

        # Build LR lambda (identical to SAMTrainer._create_scheduler)
        warmup_cfg = self.warmup_config
        warmup_steps = warmup_cfg["steps"]
        warmup_enabled = warmup_cfg["enabled"]
        total_iters = self.num_epochs * self.trainer.estimated_stepping_batches // self.num_epochs  # steps per epoch
        total_iters = self.num_epochs * total_iters  # re-compute (need trainer)
        # Fallback: use trainer.estimated_stepping_batches directly
        total_iters = self.trainer.estimated_stepping_batches

        power = float(cfg.get("scheduler", {}).get("power", 0.9))
        min_lr = float(cfg.get("scheduler", {}).get("min_lr", 1e-6))
        min_lr_ratio = min_lr / self.base_lr if self.base_lr > 0 else 0.0

        def lr_lambda(current_step: int) -> float:
            if warmup_enabled and warmup_steps > 0 and current_step < warmup_steps:
                return (current_step + 1) / warmup_steps
            shift_iter = (
                current_step - warmup_steps
                if (warmup_enabled and warmup_steps > 0)
                else current_step
            )
            shift_iter = min(max(0, shift_iter), total_iters)
            decay = (1.0 - shift_iter / total_iters) ** power
            return max(min_lr_ratio, decay)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    # Gradient clipping (delegated to Lightning Trainer via config)
    # ------------------------------------------------------------------
    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        """Override to use the same gradient clipping as SAMTrainer."""
        self.clip_gradients(
            optimizer,
            gradient_clip_val=self.gradient_clip_max_norm,
            gradient_clip_algorithm="norm",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _log_param_counts(self):
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        log.info("Total parameters: %s", f"{total:,}")
        log.info("Trainable parameters: %s", f"{trainable:,}")

    def save_lora_parameters(self, path: str):
        """Delegate LoRA saving to the underlying model."""
        base = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(base, "save_lora_parameters"):
            base.save_lora_parameters(path)
        else:
            torch.save(self.model.state_dict(), path)

    def load_lora_parameters(self, path: str):
        """Delegate LoRA loading to the underlying model."""
        base = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(base, "load_lora_parameters"):
            base.load_lora_parameters(path)
        else:
            self.model.load_state_dict(torch.load(path, map_location="cpu"))
