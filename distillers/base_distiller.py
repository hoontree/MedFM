import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseDistiller(nn.Module, ABC):
    """
    Abstract base class for all distillers.
    Each distiller should implement its own loss calculation logic.
    """

    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.temperature = cfg.method.get("temperature", 4.0)
        self.alpha = cfg.method.get("alpha", 1.0)
        self.beta = cfg.method.get("beta", 0.0)
        self.gamma = cfg.method.get("gamma", 0.0)

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
                'feature_loss': Loss related to feature alignment (optional)
        """
        pass
