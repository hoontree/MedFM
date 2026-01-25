import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
from distillers.base_distiller import BaseDistiller
from distillers.registry import DistillerRegistry
from utils.sam_utils import DiceLoss


@DistillerRegistry.register("hybrid")
class HybridDistiller(BaseDistiller):
    """
    Hybrid distillation supporting:
    1. Prediction KL Divergence + GT Dice/CE
    2. Encoder Attention Map MSE + Prediction KL + GT Loss
    3. Encoder Attention Map MSE + Prediction KL + Alignment Feature MSE + GT Dice
    """

    def __init__(self, cfg: Any):
        super().__init__(cfg)
        self.num_classes = cfg.data.num_classes

        # Hyperparameters for different components
        self.alpha = cfg.method.get("alpha", 1.0)  # Task loss weight
        self.beta = cfg.method.get("beta", 1.0)  # Logit KD weight
        self.gamma_attn = cfg.method.get("gamma_attn", 0.0)  # Attention MSE weight
        self.gamma_align = cfg.method.get(
            "gamma_align", 0.0
        )  # Alignment feature MSE weight

        self.temperature = cfg.method.get("temperature", 4.0)

        # Flags for task loss components
        self.use_dice = cfg.method.get("use_dice", True)
        self.use_ce = cfg.method.get("use_ce", True)

        # Task losses
        if self.num_classes == 2:
            self.task_criterion = nn.BCEWithLogitsLoss()
        else:
            self.task_criterion = nn.CrossEntropyLoss()

        self.dice_loss = DiceLoss(self.num_classes)
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss = nn.MSELoss()

        # Projection layer for alignment feature MSE if channels differ
        self.align_proj = None
        s_channels = cfg.method.get("student_channels", 48)
        t_channels = cfg.method.get("teacher_alignment_channels", 256)
        if self.gamma_align > 0 and s_channels != t_channels:
            self.align_proj = nn.Conv2d(s_channels, t_channels, kernel_size=1)

    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        student_logits = student_outputs["masks"]
        teacher_logits = teacher_outputs["masks"]

        losses = {}

        # 1. Task Loss (GT)
        ce_loss = torch.tensor(0.0).to(student_logits.device)
        dice_loss = torch.tensor(0.0).to(student_logits.device)

        if self.num_classes == 2:
            target_mask = targets.unsqueeze(1).float()
            if self.use_ce:
                ce_loss = self.task_criterion(student_logits, target_mask)
            if self.use_dice:
                dice_loss = self.dice_loss(student_logits, target_mask)
        else:
            if self.use_ce:
                ce_loss = self.task_criterion(student_logits, targets.long())
            if self.use_dice:
                dice_loss = self.dice_loss(student_logits, targets, softmax=True)

        task_loss = 0.0
        if self.use_ce and self.use_dice:
            task_loss = 0.5 * ce_loss + 0.5 * dice_loss
        elif self.use_ce:
            task_loss = ce_loss
        elif self.use_dice:
            task_loss = dice_loss

        losses["task_ce"] = ce_loss
        losses["task_dice"] = dice_loss

        # 2. Logit Distillation Loss (KL Divergence)
        if self.beta > 0:
            if teacher_logits.shape != student_logits.shape:
                teacher_logits_resized = F.interpolate(
                    teacher_logits,
                    size=student_logits.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                teacher_logits_resized = teacher_logits

            if self.num_classes == 2:
                # Binary: expand to 2 channels
                student_soft = F.log_softmax(
                    torch.cat([torch.zeros_like(student_logits), student_logits], dim=1)
                    / self.temperature,
                    dim=1,
                )
                teacher_soft = F.softmax(
                    torch.cat(
                        [
                            torch.zeros_like(teacher_logits_resized),
                            teacher_logits_resized,
                        ],
                        dim=1,
                    )
                    / self.temperature,
                    dim=1,
                )
            else:
                student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
                teacher_soft = F.softmax(
                    teacher_logits_resized / self.temperature, dim=1
                )

            distill_loss = self.kl_div(student_soft, teacher_soft) * (
                self.temperature**2
            )
            # Normalize by spatial dimensions
            num_pixels = student_logits.shape[-2] * student_logits.shape[-1]
            distill_loss = distill_loss / num_pixels
            losses["distill_loss"] = distill_loss
        else:
            distill_loss = torch.tensor(0.0).to(student_logits.device)

        # 3. Attention Map MSE Loss
        attn_loss = torch.tensor(0.0).to(student_logits.device)
        if (
            self.gamma_attn > 0
            and "attn_maps" in student_outputs
            and "attn_maps" in teacher_outputs
        ):
            s_attns = student_outputs["attn_maps"]
            t_attns = teacher_outputs["attn_maps"]

            num_blocks = min(len(s_attns), len(t_attns))
            if num_blocks > 0:
                for i in range(num_blocks):
                    s_attn = s_attns[i]
                    t_attn = t_attns[i]

                    if s_attn.dim() == 4 and t_attn.dim() == 3:
                        B, H, N, _ = s_attn.shape
                        s_attn = s_attn.view(B * H, N, N)
                    elif s_attn.dim() == 3 and t_attn.dim() == 4:
                        B, H, N, _ = t_attn.shape
                        t_attn = t_attn.view(B * H, N, N)

                    if s_attn.shape != t_attn.shape:
                        if s_attn.shape[-1] != t_attn.shape[-1]:
                            # Bilinear interpolation for attention map [B*H, 1, N, N]
                            t_attn = F.interpolate(
                                t_attn.unsqueeze(1),
                                size=s_attn.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze(1)

                    attn_loss += self.mse_loss(s_attn, t_attn)
                attn_loss /= num_blocks
            losses["attn_loss"] = attn_loss

        # 4. Alignment Layer Feature Map MSE Loss
        align_loss = torch.tensor(0.0).to(student_logits.device)
        if (
            self.gamma_align > 0
            and "features" in student_outputs
            and "image_embeddings" in teacher_outputs
        ):
            s_feat = student_outputs["features"]
            t_align = teacher_outputs["image_embeddings"]

            if self.align_proj is not None:
                s_feat = self.align_proj(s_feat)

            if s_feat.shape != t_align.shape:
                s_feat = F.interpolate(
                    s_feat,
                    size=t_align.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            align_loss = self.mse_loss(s_feat, t_align)
            losses["align_loss"] = align_loss

        # Total combined loss
        total_loss = (
            self.alpha * task_loss
            + self.beta * distill_loss
            + self.gamma_attn * attn_loss
            + self.gamma_align * align_loss
        )

        losses["loss"] = total_loss
        return losses
