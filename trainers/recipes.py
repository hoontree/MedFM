"""Per-model supervised-training recipe — the single source of truth for a
model's task loss, optimizer, and LR scheduler.

Shared by a model's normal trainer and by ``DistillTrainer`` so that a
distillation run with every KD weight set to 0 (``task_only``) reduces
**exactly** to that model's normal supervised training. Previously the two
diverged silently: distillation used ``TaskLoss`` (BCE pos_weight=5 + Dice) while
TinyUSFM's normal path uses bare ``BCEWithLogitsLoss``; distillation built a
single-group AdamW while TinyUSFM builds a layer-wise-LR-decay optimizer; and
distillation ignored ``use_reduce_on_plateau``. Routing both through one recipe
removes all three.

A recipe exposes:
  - ``task_loss(logits, masks) -> scalar`` : the model's supervised loss, matching
    its normal trainer exactly. Injected into the distiller as ``task_loss_fn``.
  - ``build_optimizer(model, cfg) -> Optimizer``
  - ``build_scheduler(optimizer, cfg) -> (scheduler, cadence)`` where cadence is
    ``"epoch"`` | ``"batch"`` | ``"plateau"`` and tells the trainer when to step.

``get_recipe(name, cfg)`` returns ``None`` for models without a registered recipe,
so ``DistillTrainer`` keeps its previous behavior for those until they are added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch.nn as nn
import torch.optim as optim


class _TaskLossModule(nn.Module):
    """Wrap a ``(logits, masks) -> scalar`` callable as an nn.Module.

    The distiller registers ``task_loss_fn`` as a submodule, so it must be an
    nn.Module (a bare function can't be assigned to it). The wrapped criterion is
    stateless w.r.t. device for the current recipes (CE / BCE without weights), so
    it needs no explicit ``.to(device)``.
    """

    def __init__(self, fn: Callable):
        super().__init__()
        self._fn = fn

    def forward(self, logits, masks):
        return self._fn(logits, masks)


@dataclass
class Recipe:
    task_loss: nn.Module         # nn.Module: (logits, masks) -> scalar
    build_optimizer: Callable    # (model, cfg) -> optim.Optimizer
    build_scheduler: Callable    # (optimizer, cfg) -> (scheduler, cadence:str)


def _tinyusfm_recipe(cfg) -> Recipe:
    # Reuse TinyUSFMTrainer's own construction functions verbatim — the exact same
    # code the normal path runs, so task_only distillation is bit-for-bit its
    # supervised training.
    from trainers.tinyusfm_trainer import _build_criterion, _compute_loss, _build_optimizer
    from utils.schedule import build_scheduler as _build_scheduler

    num_classes = int(cfg.data.num_classes)
    criterion = _build_criterion(num_classes)

    def _task_loss(logits, masks):
        return _compute_loss(criterion, logits, masks, num_classes)

    task_loss = _TaskLossModule(_task_loss)

    def build_optimizer(model, c):
        return _build_optimizer(model, c)

    def build_scheduler(optimizer, c) -> Tuple[object, str]:
        sched_cfg = c.get("scheduler", {})
        if sched_cfg.get("use_reduce_on_plateau", False):
            sched = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=sched_cfg.get("factor", 0.5),
                patience=sched_cfg.get("patience", 5),
                min_lr=sched_cfg.get("min_lr", 1e-7),
            )
            return sched, "plateau"
        return _build_scheduler(optimizer, c), "epoch"

    return Recipe(task_loss=task_loss, build_optimizer=build_optimizer, build_scheduler=build_scheduler)


# model name (config/model/<name>.yaml) -> recipe factory
_RECIPES = {
    "tinyusfm": _tinyusfm_recipe,
}


def get_recipe(model_name: str, cfg) -> Optional[Recipe]:
    """Recipe for ``model_name``, or None if none is registered yet."""
    factory = _RECIPES.get(model_name)
    return factory(cfg) if factory is not None else None
