import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseDistiller(nn.Module, ABC):
    """
    Abstract base class for all distillers.
    Each distiller should implement its own loss calculation logic.
    """

    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.temperature = cfg.method.get("temperature", 4.0)
        self.alpha = cfg.method.get("alpha", 0.5)
        self.beta = cfg.method.get("beta", 0.5)
        self.gamma = cfg.method.get("gamma", 0.0)

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
                'feature_loss': Loss related to feature alignment (optional)
        """
        pass
