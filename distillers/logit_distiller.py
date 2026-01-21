import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from distillers.base_distiller import BaseDistiller
from distillers.registry import DistillerRegistry
from utils.sam_utils import DiceLoss


@DistillerRegistry.register("logit")
class LogitDistiller(BaseDistiller):
    """
    Standard logit-based distillation.
    Computes Task Loss (BCE/CE + Dice) and Distillation Loss (KL Divergence).
    """

    def __init__(self, cfg: Any):
        super().__init__(cfg)
        self.num_classes = cfg.data.num_classes

        # Task losses
        if self.num_classes == 2:
            self.task_criterion = nn.BCEWithLogitsLoss()
        else:
            self.task_criterion = nn.CrossEntropyLoss()

        self.dice_loss = DiceLoss(self.num_classes)
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        student_logits = student_outputs["masks"]
        teacher_logits = teacher_outputs["masks"]

        # 1. Task loss (hard targets)
        if self.num_classes == 2:
            # Binary case: targets is [B, H, W]
            task_loss = self.task_criterion(
                student_logits, targets.unsqueeze(1).float()
            )
            dice_loss = self.dice_loss(student_logits, targets.unsqueeze(1).float())
        else:
            # Multi-class case: targets is [B, H, W]
            task_loss = self.task_criterion(student_logits, targets.long())
            dice_loss = self.dice_loss(student_logits, targets, softmax=True)

        total_task_loss = 0.5 * task_loss + 0.5 * dice_loss

        # 2. Distillation loss (soft targets)
        # Resize teacher logits to match student if needed
        if teacher_logits.shape != student_logits.shape:
            teacher_logits = F.interpolate(
                teacher_logits,
                size=student_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # Apply temperature scaling and compute KL divergence
        if student_logits.shape[1] == 1:
            # Binary segmentation case: expand to 2 channels for KL divergence
            student_logits_expanded = torch.cat(
                [torch.zeros_like(student_logits), student_logits], dim=1
            )
            teacher_logits_expanded = torch.cat(
                [torch.zeros_like(teacher_logits), teacher_logits], dim=1
            )

            student_soft = F.log_softmax(
                student_logits_expanded / self.temperature, dim=1
            )
            teacher_soft = F.softmax(teacher_logits_expanded / self.temperature, dim=1)
        else:
            student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
            teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)

        distill_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature**2)

        # Normalize by spatial dimensions
        num_pixels = student_logits.shape[-2] * student_logits.shape[-1]
        distill_loss = distill_loss / num_pixels

        # Total loss
        total_loss = self.alpha * total_task_loss + self.beta * distill_loss

        return {
            "loss": total_loss,
            "task_loss": total_task_loss,
            "distill_loss": distill_loss,
        }
