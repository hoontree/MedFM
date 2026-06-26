import logging
import torch
import torch.nn as nn
from typing import Dict, Any

from distillers.unified_distiller import UnifiedDistiller

logger = logging.getLogger(__name__)


class OnlineDistiller(UnifiedDistiller):
    """
    Online distillation: teacher and student are trained simultaneously.

    In addition to the standard UnifiedDistiller losses (task + logit + CWD +
    reliability/uncertainty KD), the teacher also receives a task loss from
    ground truth so that it continues to improve during distillation.

    Loss structure:
        student_loss = UnifiedDistiller.forward(...)["loss"]
        teacher_loss = w_teacher_task * task_loss(teacher)
        total_loss   = student_loss + teacher_loss

    ``w_teacher_task`` controls how strongly the teacher is updated. Set it
    to ``0.0`` to fall back to offline behaviour (teacher frozen). The
    legacy ``teacher_alpha`` key is still accepted with a deprecation
    warning.
    """

    def __init__(self, cfg: Any, **kwargs):
        super().__init__(cfg, **kwargs)
        m = cfg.method
        if "w_teacher_task" in m:
            self.w_teacher_task = float(m.get("w_teacher_task"))
        elif "teacher_alpha" in m:
            logger.warning(
                "method.teacher_alpha is deprecated; rename to method.w_teacher_task"
            )
            self.w_teacher_task = float(m.get("teacher_alpha"))
        else:
            self.w_teacher_task = 1.0
        logger.info(
            "[OnlineDistiller] w_teacher_task=%s  "
            "(teacher is updated jointly with student)",
            self.w_teacher_task,
        )

    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        losses = super().forward(student_outputs, teacher_outputs, targets)

        # Teacher task loss (ground-truth supervision on teacher output).
        # Weight 0 → neither computed nor stored, keeping the loss dict free
        # of disabled components (see UnifiedDistiller.forward).
        if self.w_teacher_task > 0:
            teacher_logits = teacher_outputs["masks"]
            teacher_task_loss = self._compute_task_loss(teacher_logits, targets)
            losses["teacher_task_loss"] = teacher_task_loss
            losses["loss"] = losses["loss"] + self.w_teacher_task * teacher_task_loss

        return losses
