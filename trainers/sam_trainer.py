"""
SAM (Segment Anything Model) Trainer

This module provides a trainer for SAM models with LoRA adaptation.
"""

from pathlib import Path
from typing import Dict, override
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm
import wandb

from hydra.utils import instantiate
from .base_trainer import BaseTrainer
from utils.data_processing import SegDatasetProcessor
from utils.criterion import DiceLoss
from monai.losses import DiceLoss as MonaiDiceLoss


class SAMTrainer(BaseTrainer):
    """Trainer for SAM models."""

    def __init__(self, cfg):
        """Initialize SAM trainer."""
        super().__init__(cfg)

        # SAM-specific attributes
        self.ce_loss = None
        self.bce_loss = None
        self.dice_loss = None
        self.img_size = cfg.model.img_size
        self.step_log_interval = 10

        self.num_epochs = self.cfg.training.get("num_epochs", 100)
        self.base_lr = float(self.cfg.training.get("lr", 1e-4))
        self.dice_loss_weight = self.cfg.training.get("dice_loss_weight", 0.8)
        self.moe_loss_weight = float(self.cfg.get("training", {}).get("moe_loss_weight", 0.0))
        self.gradient_clip_max_norm = float(
            self.cfg.optimizer.get("gradient_clip", {}).get("max_norm", 1.0)
        )
        self.warmup_config = self.cfg.training.get(
            "warmup", {"enabled": False, "steps": 0}
        )

        # Mixed precision. bf16 (default) needs no loss scaling; fp16 does.
        # Enables full fine-tuning of the larger backbones (vit_l/vit_h at
        # img_size 1024) within memory. Disabled → exact fp32 behavior as before.
        self.amp_enabled = bool(self.cfg.training.get("amp", False))
        amp_dtype_str = str(self.cfg.training.get("amp_dtype", "bfloat16")).lower()
        self.amp_dtype = torch.float16 if amp_dtype_str in ("float16", "fp16", "half") else torch.bfloat16
        # GradScaler is a no-op passthrough unless enabled (fp16 only).
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_enabled and self.amp_dtype == torch.float16
        )
        # Down-scale the backbone (image encoder) LR relative to the decoder/
        # prompt head. 1.0 preserves the previous single-LR behavior.
        self.backbone_lr_scale = float(self.cfg.training.get("backbone_lr_scale", 1.0))

    @override
    def _create_model(self):
        """Create SAM model using ModelBuilder."""
        # Refresh img_size from the (now synced) data config. __init__ cached
        # cfg.model.img_size before _create_dataloaders ran
        # _sync_img_size_with_sam_type, so for vit_l/vit_h it was stale at the
        # 224 default; the sync bumped data/model img_size to 1024. Using the
        # stale value made the eval forward postprocess masks to 224 while GT
        # labels are 1024 → shape-mismatch crash in the evaluator.
        self.img_size = int(self.cfg.data.img_size)

        # Keep trainer runtime img_size aligned with any pre-dataloader sync.
        self.model = instantiate(self.cfg.model).to(self.device)

        # Setup DataParallel
        if len(self.cfg.get("hardware", {}).get("gpu_ids", [0])) > 1:
            self.model = nn.DataParallel(self.model)

        self._setup_loss_functions()

        self._log_model_info()

    @override
    def _create_optimizer(self):
        """Create optimizer with separate LR for backbone vs decoder/prompt.

        The ViT image encoder ("backbone") is trained at ``base_lr *
        backbone_lr_scale`` (default 1.0 → identical to a single-group AdamW),
        while the mask decoder / prompt encoder use the full ``base_lr``. This
        discriminative schedule stabilizes full fine-tuning of the large
        backbones, mirroring the SAM3 recipe.
        """
        weight_decay = self.cfg.optimizer.get("weight_decay", 0.1)
        base_model = self._get_base_model()

        backbone_params, other_params = [], []
        for name, p in base_model.named_parameters():
            if not p.requires_grad:
                continue
            if "image_encoder" in name:
                backbone_params.append(p)
            else:
                other_params.append(p)

        param_groups = [
            {"params": other_params, "lr": self.base_lr},
            {"params": backbone_params, "lr": self.base_lr * self.backbone_lr_scale},
        ]
        param_groups = [g for g in param_groups if g["params"]]

        self.optimizer = optim.AdamW(
            param_groups, lr=self.base_lr, weight_decay=weight_decay
        )

        self.logger.info(
            "Optimizer: AdamW, decoder/prompt LR=%s, backbone LR=%s (scale=%s), "
            "backbone params=%d, other params=%d",
            self.base_lr,
            self.base_lr * self.backbone_lr_scale,
            self.backbone_lr_scale,
            len(backbone_params),
            len(other_params),
        )

    @override
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup and polynomial decay."""
        warmup_cfg = self.warmup_config
        warmup_steps = warmup_cfg["steps"]
        warmup_enabled = warmup_cfg["enabled"]

        total_iters = self.num_epochs * len(self.train_loader)
        power = float(self.cfg.get("scheduler", {}).get("power", 0.9))

        # Minimum learning rate ratio (prevents LR from reaching 0)
        min_lr = float(self.cfg.get("scheduler", {}).get("min_lr", 1e-6))
        min_lr_ratio = min_lr / self.base_lr

        def lr_lambda(current_step: int) -> float:
            """Calculate learning rate multiplier for given step."""
            # Warmup phase: linear increase
            if warmup_enabled and warmup_steps > 0 and current_step < warmup_steps:
                return (current_step + 1) / warmup_steps

            # Polynomial decay phase
            shift_iter = (
                current_step - warmup_steps
                if (warmup_enabled and warmup_steps > 0)
                else current_step
            )
            shift_iter = min(max(0, shift_iter), total_iters)  # Clamp to valid range
            decay = (1.0 - shift_iter / total_iters) ** power
            return max(min_lr_ratio, decay)

        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda
        )
        self._sched_cadence = "batch"  # stepped per-batch inside _supervised_epoch
        self.logger.info(
            f"Scheduler: LambdaLR with warmup={warmup_enabled}, warmup_steps={warmup_steps}, "
            f"max_iterations={total_iters}, power={power}, min_lr={min_lr}"
        )

    @override
    def _log_model_info(self):
        """Override: also log SAM adaptation configuration (encoder/decoder mode)."""
        super()._log_model_info()

        encoder_mode = getattr(self.model, "encoder_mode", "N/A")
        decoder_mode = getattr(self.model, "decoder_mode", "N/A")

        self.logger.info("SAM Configuration:")
        self.logger.info(
            "encoder_mode=%s, decoder_mode=%s",
            encoder_mode,
            decoder_mode,
        )

        if wandb.run is not None:
            wandb.config.update(
                {
                    "model.encoder_mode": encoder_mode,
                    "model.decoder_mode": decoder_mode,
                },
                allow_val_change=True,
            )

    def _setup_loss_functions(self):
        """Setup loss functions based on number of classes."""
        num_classes = self.cfg.data.num_classes

        # Targets arrive already one-hot encoded ([B, C, H, W] float), so
        # to_onehot_y stays False. The activation is fixed at construction time
        # (MonaiDiceLoss has no per-call `activation` arg): sigmoid for binary,
        # softmax for multi-class.
        if num_classes == 1:
            pos_weight = torch.tensor([5.0], device=self.device)
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self.dice_loss = MonaiDiceLoss(
                include_background=True,
                to_onehot_y=False,
                sigmoid=True,
            )
        else:
            self.ce_loss = CrossEntropyLoss()
            self.dice_loss = MonaiDiceLoss(
                include_background=False,  # Exclude background from Dice loss
                to_onehot_y=False,
                softmax=True,
            )

    def _calc_loss(self, outputs, label_batch, low_res_label_batch):
        """Calculate loss using unified channel-based approach."""
        dice_weight = self.dice_loss_weight
        moe_loss_weight = self.moe_loss_weight
        num_classes = self.cfg.data.num_classes

        logits = outputs["low_res_logits"]
        target = low_res_label_batch

        # Ensure target and logits have the same resolution.
        # Expected: target is downsampled to match low-res logits.
        # Upsampling (target smaller than logits) should never happen in normal use.
        if logits.shape[-2:] != target.shape[-2:]:
            if target.shape[-2] < logits.shape[-2] or target.shape[-1] < logits.shape[-1]:
                import warnings
                warnings.warn(
                    f"_calc_loss: target {tuple(target.shape[-2:])} is smaller than "
                    f"logits {tuple(logits.shape[-2:])} — upsampling target, which may "
                    "silently degrade training. Check data pipeline resolution.",
                    stacklevel=2,
                )
            target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")

        if num_classes == 1:
            loss_ce = self.bce_loss(logits, target)
            loss_dice = self.dice_loss(logits, target)
        else:
            # target is one-hot [B, C, H, W] float; CE expects class index [B, H, W] long
            target_idx = target.argmax(dim=1).long()
            loss_ce = self.ce_loss(logits, target_idx)
            loss_dice = self.dice_loss(logits, target)

        loss_moe = outputs.get("moe_loss", torch.tensor(0.0, device=logits.device))
        if not torch.is_tensor(loss_moe):
            loss_moe = torch.tensor(float(loss_moe), device=logits.device)

        loss = (
            (1 - dice_weight) * loss_ce
            + dice_weight * loss_dice
            + moe_loss_weight * loss_moe
        )

        return loss, loss_ce, loss_dice, loss_moe

    @override
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """One epoch via the shared BaseTrainer._supervised_epoch loop, using the
        SAM recipe's per-batch compute. This is the SAME loop + SAME per-batch
        compute a distillation task_only run of a SAM student uses, so the two are
        bit-identical (task_only ≡ normal training)."""
        if getattr(self, "_recipe", None) is None:
            from trainers.recipes import get_recipe
            self._recipe = get_recipe("sam", self.cfg)
        return self._supervised_epoch(
            self.model, self._recipe.compute_batch_loss,
            self.model.parameters, "batch", epoch,
        )

    @override
    def validate(self, epoch: int, return_predictions: bool = False):
        """Validate model.

        When ``return_predictions=True``, also returns a predictions cache so that
        visualization reuses the same forward pass as metric computation.
        """
        self.model.eval()

        result = self.evaluator.evaluate_model_sam(
            self.model,
            self.val_loader,
            self.device,
            self.cfg.data.num_classes,
            img_size=self.img_size,
            return_predictions=return_predictions,
        )

        if return_predictions:
            val_metrics, images_l, preds_l, masks_l, fnames_l, _ = result
        else:
            val_metrics = result

        self.evaluator.print_metrics(val_metrics, phase="validation")

        if return_predictions:
            return val_metrics, {"__val__": (images_l, preds_l, masks_l, fnames_l)}
        return val_metrics

    @override
    def test(self) -> Dict[str, float]:
        """Test model and visualize predictions using cached inference results."""
        self.model.eval()

        test_metrics = {}
        # Store predictions for visualization (avoid redundant forward pass)
        predictions_cache = {}

        for name, loader in self._iter_test_loaders():
            if isinstance(self.test_loader, dict):
                self.logger.info(f"Testing on dataset: {name}")

            # Get metrics and predictions in one forward pass
            result = self.evaluator.evaluate_model_sam(
                self.model,
                loader,
                self.device,
                self.cfg.data.num_classes,
                img_size=self.img_size,
                return_predictions=True,
            )
            metrics, images_list, preds_list, masks_list, filenames_list, per_sample = result

            self.evaluator.print_metrics(
                metrics,
                phase=f"test_{name}" if isinstance(self.test_loader, dict) else "test",
            )

            # Store metrics with appropriate keys
            if isinstance(self.test_loader, dict):
                for k, v in metrics.items():
                    test_metrics[f"{name}/{k}"] = v
            else:
                test_metrics = metrics

            # Cache predictions for visualization + per-sample metrics for CSV
            predictions_cache[name] = (images_list, preds_list, masks_list, filenames_list, per_sample)

        # Visualize predictions using cached results (all samples)
        vis_dir = self.exp_dir / "visualizations" / f"epoch_{self.current_epoch + 1}"
        self._visualize_predictions(
            predictions_cache=predictions_cache, vis_dir=vis_dir, num_samples=10
        )

        return test_metrics, predictions_cache

    def _get_base_model(self):
        """Return the underlying LoRA_Sam, unwrapping DataParallel if needed."""
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    @override
    def _get_model_type(self) -> str:
        """SAM model type for visualization inference."""
        return "sam"

    @override
    def _save_model(self, path: Path):
        """Save SAM model (LoRA parameters)."""
        self._get_base_model().save_lora_parameters(str(path))

    @override
    def _load_checkpoint(self, path: Path):
        """Load SAM model checkpoint."""
        self.logger.info(f"Loading checkpoint: {path}")
        self._get_base_model().load_lora_parameters(str(path))
