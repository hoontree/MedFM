import logging
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Dict

logger = logging.getLogger(__name__)


# Loss weight keys are semantic (``w_<component>``). Each entry lists the
# canonical name and any legacy aliases that should still be accepted with
# a deprecation warning.
LOSS_WEIGHT_ALIASES: Dict[str, tuple] = {
    "w_task":            ("alpha",),
    "w_logit_kd":        ("beta",),
    "w_logit_cwd":       ("delta",),
    "w_feature_cwd":     ("zeta",),
    "w_reliability_kd":  ("eta",),
    "w_uncertainty_kd":  ("kd_lambda",),
}


def resolve_loss_weight(method_cfg: Any, canonical: str, default: float = 0.0) -> float:
    """Read a loss weight from ``method_cfg`` accepting legacy aliases.

    Looks up ``canonical`` first; if absent, falls back to any registered
    alias (warning once per call) so old yamls keep working. Returns
    ``default`` when neither is provided.
    """
    if canonical in method_cfg:
        return float(method_cfg.get(canonical))
    for alias in LOSS_WEIGHT_ALIASES.get(canonical, ()):
        if alias in method_cfg:
            logger.warning(
                "method.%s is deprecated; rename to method.%s", alias, canonical,
            )
            return float(method_cfg.get(alias))
    return float(default)


class BaseDistiller(nn.Module, ABC):
    """
    Abstract base class for all distillers.
    Each distiller should implement its own loss calculation logic.
    """

    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.temperature = cfg.method.get("temperature", 4.0)
        self.w_task = resolve_loss_weight(cfg.method, "w_task", default=1.0)
        self.w_logit_kd = resolve_loss_weight(cfg.method, "w_logit_kd", default=0.0)

    def prepare(self, student: nn.Module, teacher: nn.Module):
        """
        Prepare the distiller by registering hooks, initializing adapters, etc.
        Called once before training starts.
        """
        pass

    def on_step_begin(self):
        """
        Called at the beginning of each training step.
        Used to clear feature buffers or reset internal state.
        """
        pass

    @abstractmethod
    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distillation losses.

        Args:
            student_outputs: Dictionary of student outputs (logits, features, etc.)
            teacher_outputs: Dictionary of teacher outputs (logits, features, etc.)
            targets: Ground truth labels

        Returns:
            Dictionary of losses:
                'loss': Total combined loss
                'task_loss': Loss related to ground truth
                'distill_loss': Loss related to teacher-student alignment
        """
        pass
