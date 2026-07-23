"""Per-model supervised-training recipe — the single source of truth for a
model's task loss, optimizer, and LR scheduler.

Shared by a model's normal trainer and by ``DistillTrainer`` so that a
distillation run with every KD weight = 0 (``task_only``) reduces **exactly** to
that model's normal supervised training. Previously the two diverged silently
(different loss / optimizer / scheduler code), so no amount of HP-matching made
them agree.

The task loss is owned by the *trainer*, not the distiller: it takes the full
student output dict and the full batch, so it can reproduce losses computed on
outputs the KD pipeline doesn't carry — notably SAM, whose supervised loss is on
the LOW-RES mask logits vs a low-res label (batch[2]), while KD compares the
high-res ``masks``. The distiller then computes KD terms only.

A recipe exposes:
  - ``task_loss(student_output, batch, device) -> scalar`` : the model's
    supervised loss, matching its normal trainer exactly.
  - ``build_optimizer(model, cfg) -> Optimizer``
  - ``build_scheduler(optimizer, cfg) -> (scheduler, cadence)`` where cadence is
    ``"epoch"`` | ``"batch"`` | ``"plateau"``.

``get_recipe(name, cfg)`` returns ``None`` for unregistered models, so the
trainer keeps its previous behavior for those.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim


@dataclass
class Recipe:
    task_loss: Callable          # (student_output: dict, batch: tuple, device) -> scalar
    build_optimizer: Callable    # (model, cfg) -> optim.Optimizer
    build_scheduler: Callable    # (optimizer, cfg, total_iters) -> (scheduler, cadence:str)


# --------------------------------------------------------------------------- #
# TinyUSFM — dense high-res segmenter (bare BCE / CE), layer-wise-LR-decay AdamW,
# epoch (poly) or plateau scheduler. Reuses TinyUSFMTrainer's own functions.
# --------------------------------------------------------------------------- #
def _tinyusfm_recipe(cfg) -> Recipe:
    from trainers.tinyusfm_trainer import _build_criterion, _compute_loss, _build_optimizer
    from utils.schedule import build_scheduler as _build_scheduler

    num_classes = int(cfg.data.num_classes)
    criterion = _build_criterion(num_classes)

    def task_loss(student_output, batch, device):
        logits = student_output["masks"]
        masks = batch[1].to(device)
        return _compute_loss(criterion, logits, masks, num_classes)

    def build_optimizer(model, c):
        return _build_optimizer(model, c)

    def build_scheduler(optimizer, c, total_iters) -> Tuple[object, str]:
        sched_cfg = c.get("scheduler", {})
        if sched_cfg.get("use_reduce_on_plateau", False):
            sched = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max",
                factor=sched_cfg.get("factor", 0.5),
                patience=sched_cfg.get("patience", 5),
                min_lr=sched_cfg.get("min_lr", 1e-7),
            )
            return sched, "plateau"
        return _build_scheduler(optimizer, c), "epoch"

    return Recipe(task_loss=task_loss, build_optimizer=build_optimizer, build_scheduler=build_scheduler)


# --------------------------------------------------------------------------- #
# SAM (LoRA_Sam) — supervised loss on the LOW-RES logits vs low-res label
# (SamTrainer._calc_loss), backbone/other LR split, per-batch LambdaLR.
# --------------------------------------------------------------------------- #
def _sam_recipe(cfg) -> Recipe:
    from monai.losses import DiceLoss as MonaiDiceLoss

    num_classes = int(cfg.data.num_classes)
    dice_weight = float(cfg.training.get("dice_loss_weight", 0.8))
    moe_weight = float(cfg.get("training", {}).get("moe_loss_weight", 0.0))
    backbone_lr_scale = float(cfg.training.get("backbone_lr_scale", 1.0))
    base_lr = float(cfg.training.get("lr", 1e-4))
    weight_decay = float(cfg.optimizer.get("weight_decay", 0.1))
    num_epochs = int(cfg.training.get("num_epochs", 100))

    # MonaiDiceLoss is stateless w.r.t. device (fixed sigmoid/softmax); pos_weight
    # is applied functionally so no buffer needs moving.
    if num_classes == 1:
        dice = MonaiDiceLoss(include_background=True, to_onehot_y=False, sigmoid=True)
    else:
        dice = MonaiDiceLoss(include_background=False, to_onehot_y=False, softmax=True)

    def task_loss(student_output, batch, device):
        logits = student_output["low_res_logits"]        # SAM's native mask resolution
        target = batch[2].to(device)                     # low-res label (3rd batch element)
        if logits.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
        if num_classes == 1:
            loss_ce = F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=torch.tensor([5.0], device=logits.device)
            )
            loss_dice = dice(logits, target)
        else:
            loss_ce = F.cross_entropy(logits, target.argmax(dim=1).long())
            loss_dice = dice(logits, target)
        moe = student_output.get("moe_loss", torch.tensor(0.0, device=logits.device))
        if not torch.is_tensor(moe):
            moe = torch.tensor(float(moe), device=logits.device)
        return (1 - dice_weight) * loss_ce + dice_weight * loss_dice + moe_weight * moe

    def build_optimizer(model, c):
        # backbone (image_encoder) vs other, per SamTrainer._create_optimizer.
        backbone, other = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (backbone if "image_encoder" in name else other).append(p)
        groups = [
            {"params": other, "lr": base_lr},
            {"params": backbone, "lr": base_lr * backbone_lr_scale},
        ]
        groups = [g for g in groups if g["params"]]
        return optim.AdamW(groups, lr=base_lr, weight_decay=weight_decay)

    def build_scheduler(optimizer, c, total_iters) -> Tuple[object, str]:
        # per-batch WarmupPoly LambdaLR (SamTrainer._create_scheduler).
        total_iters = int(total_iters) or 1
        warmup_cfg = c.get("training", {}).get("warmup", {"enabled": False, "steps": 0})
        warmup_steps = int(warmup_cfg.get("steps", 0))
        warmup_enabled = bool(warmup_cfg.get("enabled", False))
        power = float(c.get("scheduler", {}).get("power", 0.9))
        min_lr = float(c.get("scheduler", {}).get("min_lr", 1e-6))
        min_lr_ratio = min_lr / base_lr

        def lr_lambda(step):
            if warmup_enabled and warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps
            shift = step - warmup_steps if (warmup_enabled and warmup_steps > 0) else step
            shift = min(max(0, shift), total_iters)
            return max(min_lr_ratio, (1.0 - shift / total_iters) ** power)

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), "batch"

    return Recipe(task_loss=task_loss, build_optimizer=build_optimizer, build_scheduler=build_scheduler)


# --------------------------------------------------------------------------- #
# SegFormer — dense high-res loss ((1-w)*BCE + w*Dice), single-group AdamW,
# per-batch WarmupPolyLR. (Not currently used as a distill student, but kept for
# completeness / future use.)
# --------------------------------------------------------------------------- #
def _segformer_recipe(cfg) -> Recipe:
    import torch.nn as nn
    from utils.criterion import DiceLoss
    from utils.schedule import WarmupPolyLR

    num_classes = int(cfg.data.num_classes)
    dice_weight = float(cfg.training.get("dice_weight", 0.8))
    base_lr = float(cfg.training.base_lr)
    weight_decay = float(cfg.optimizer.get("weight_decay", 0.1))
    num_epochs = int(cfg.training.num_epochs)
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss(num_classes)  # matches SegformerTrainer exactly (num_classes as first arg)

    def task_loss(student_output, batch, device):
        logits = student_output["masks"]
        target = batch[1].to(device).float()
        if logits.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
        return (1 - dice_weight) * bce(logits, target) + dice_weight * dice(logits, target)

    def build_optimizer(model, c):
        return optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=base_lr, weight_decay=weight_decay,
        )

    def build_scheduler(optimizer, c, total_iters) -> Tuple[object, str]:
        total_iters = int(total_iters) or 1
        warmup = c.get("training", {}).get("warmup", False)
        warmup_iters = int(c.get("training", {}).get("warmup_period", 250)) if warmup else 0
        return WarmupPolyLR(
            optimizer, warmup_epochs=warmup_iters, num_epochs=total_iters,
            base_lr=base_lr, power=0.9,
        ), "batch"

    return Recipe(task_loss=task_loss, build_optimizer=build_optimizer, build_scheduler=build_scheduler)


# model name (config/model/<name>.yaml) -> recipe factory
_RECIPES = {
    "tinyusfm": _tinyusfm_recipe,
    "sam": _sam_recipe,
    "segformer": _segformer_recipe,
}


def get_recipe(model_name: str, cfg) -> Optional[Recipe]:
    """Recipe for ``model_name``, or None if none is registered."""
    factory = _RECIPES.get(model_name)
    return factory(cfg) if factory is not None else None
