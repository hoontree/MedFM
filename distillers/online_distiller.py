import logging
import torch
import torch.nn as nn
from typing import Dict, Any

from distillers.unified_distiller import UnifiedDistiller

logger = logging.getLogger(__name__)


class OnlineDistiller(UnifiedDistiller):
    """
    Online distillation: teacher and student are trained simultaneously.

    In addition to the standard UnifiedDistiller losses (task + logit + feature),
    the teacher also receives a task loss from ground truth so that it continues
    to improve during distillation.

    Loss structure:
        student_loss = alpha * task_loss(student) + beta * logit_kd + gamma * feature_kd
        teacher_loss = teacher_alpha * task_loss(teacher)
        total_loss   = student_loss + teacher_loss

    The teacher_alpha weight controls how strongly the teacher is updated.
    Set teacher_alpha=0.0 to fall back to offline behaviour (teacher frozen).
    """

    def __init__(self, cfg: Any, **kwargs):
        super().__init__(cfg, **kwargs)
        self.teacher_alpha = float(cfg.method.get("teacher_alpha", 1.0))
        logger.info(
            f"[OnlineDistiller] teacher_alpha={self.teacher_alpha}  "
            "(teacher is updated jointly with student)"
        )

    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        losses = super().forward(student_outputs, teacher_outputs, targets)

        # Teacher task loss (ground-truth supervision on teacher output)
        if self.teacher_alpha > 0:
            teacher_logits = teacher_outputs["masks"]
            teacher_task_loss = self._compute_task_loss(teacher_logits, targets)
            losses["teacher_task_loss"] = teacher_task_loss
            losses["loss"] = losses["loss"] + self.teacher_alpha * teacher_task_loss
        else:
            losses["teacher_task_loss"] = self._zero(
                student_outputs["masks"].device
            )

        return losses
