import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from distillers.base_distiller import BaseDistiller

logger = logging.getLogger(__name__)


from utils.criterion import TaskLoss, LogitDistillLoss, FeatureDistillLoss, UncertaintyWeightedKDLoss
from utils.feature_extractor import FeatureExtractor


class FeatureAdapter(nn.Module):
    """Adapter to match student feature dimension to teacher."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UnifiedDistiller(BaseDistiller):
    """
    Unified distillation supporting multiple components controlled by coefficients.
    1. Task Loss (GT Dice/CE) - alpha
    2. Logit Distillation (KL Div) - beta
    3. Feature Distillation (MSE on intermediate layers) - gamma
    """

    def __init__(self, cfg: Any, **kwargs):
        super().__init__(cfg)
        self.num_classes = cfg.data.num_classes

        # Task Loss components
        self.use_dice = cfg.method.get("use_dice", True)
        self.use_ce = cfg.method.get("use_ce", True)

        self.task_loss_fn = TaskLoss(
            num_classes=self.num_classes,
            use_ce=self.use_ce,
            use_dice=self.use_dice,
        )
        self.logit_loss_fn = LogitDistillLoss(
            num_classes=self.num_classes,
            temperature=self.temperature,
        )
        self.feature_loss_fn = FeatureDistillLoss()

        # Uncertainty-weighted KD (optional)
        self.use_uncertainty_kd = cfg.method.get("use_uncertainty_weighted_kd", False)
        self.kd_lambda = cfg.method.get("kd_lambda", 1.0)
        if self.use_uncertainty_kd:
            self.uncertainty_kd_fn = UncertaintyWeightedKDLoss(
                num_classes=self.num_classes,
                tau=cfg.method.get("kd_tau", self.temperature),
                weight_type=cfg.method.get("uncertainty_weight_type", "linear"),
                beta=cfg.method.get("uncertainty_beta", 1.0),
                eps=cfg.method.get("uncertainty_eps", 1e-8),
            )

        # Feature adapters (for multiple layer mapping)
        self.layer_mapping = cfg.method.get("layer_mapping", {})
        self.adapters = nn.ModuleDict()
        layer_channels = cfg.method.get("layer_channels", {})

        for s_layer, t_layer in self.layer_mapping.items():
            s_ch = layer_channels.get(s_layer, 48)
            t_ch = layer_channels.get(t_layer, 256)
            if s_ch != t_ch:
                self.adapters[s_layer.replace(".", "_")] = FeatureAdapter(s_ch, t_ch)

        # Extractor placeholders
        self.teacher_extractor: Optional[FeatureExtractor] = None
        self.student_extractor: Optional[FeatureExtractor] = None

    def prepare(self, student: nn.Module, teacher: nn.Module):
        """Setup hooks for feature extraction."""
        t_layers = list(self.layer_mapping.values())
        s_layers = list(self.layer_mapping.keys())

        if t_layers:
            self.teacher_extractor = FeatureExtractor(teacher, t_layers)
        if s_layers:
            self.student_extractor = FeatureExtractor(student, s_layers)

    def on_step_begin(self):
        """Clear extracted features."""
        if self.teacher_extractor:
            self.teacher_extractor.clear()
        if self.student_extractor:
            self.student_extractor.clear()

    def _zero(self, device):
        """Return a zero scalar tensor on the given device."""
        return torch.tensor(0.0, device=device)

    def _compute_task_loss(self, student_logits, targets):
        """Task Loss (GT Dice/CE) - alpha"""
        if self.alpha <= 0:
            return self._zero(student_logits.device)
        return self.task_loss_fn(student_logits, targets)

    def _compute_logit_loss(self, student_logits, teacher_logits):
        """Logit Distillation (KL Div) - beta"""
        if self.beta <= 0:
            return self._zero(student_logits.device)
        return self.logit_loss_fn(student_logits, teacher_logits)

    def _compute_uncertainty_kd_loss(self, student_logits, teacher_logits):
        """Uncertainty-weighted KD loss — weighted by kd_lambda."""
        if not self.use_uncertainty_kd:
            return self._zero(student_logits.device), {}
        loss, uncertainty, weight = self.uncertainty_kd_fn(student_logits, teacher_logits)
        diagnostics = {
            "uncertainty_kd_loss_raw": loss.item(),
            "uncertainty_kd_loss_weighted": (self.kd_lambda * loss).item(),
            "mean_teacher_uncertainty": uncertainty.mean().item(),
            "mean_kd_weight": weight.mean().item(),
        }
        return loss, diagnostics

    def _compute_feature_loss(self, device):
        """Feature Distillation (MSE on intermediate layers) - gamma"""
        if (
            self.gamma <= 0
            or self.teacher_extractor is None
            or self.student_extractor is None
        ):
            return self._zero(device)

        s_feats = self.student_extractor.get_features()
        t_feats = self.teacher_extractor.get_features()

        feature_loss = self._zero(device)
        count = 0
        for s_layer, t_layer in self.layer_mapping.items():
            s_f = s_feats.get(s_layer)
            t_f = t_feats.get(t_layer)

            if s_f is not None and t_f is not None:
                # Convert student feature from sequence to spatial if needed
                if s_f.dim() == 3:  # [B, N, C] format
                    B, N, C = s_f.shape
                    # CLS token 여부에 따라 분기: N이 perfect square이면 CLS 없음
                    H_sq = int(N ** 0.5)
                    if H_sq * H_sq == N:
                        H = W = H_sq
                    else:
                        # CLS token이 있는 경우 제거 후 square 여부 재확인
                        N_no_cls = N - 1
                        H_sq = int(N_no_cls ** 0.5)
                        if H_sq * H_sq == N_no_cls:
                            s_f = s_f[:, 1:, :]  # Remove CLS token
                            H = W = H_sq
                            N = N_no_cls
                        else:
                            # reshape 불가 — 이 layer는 skip
                            continue
                    s_f = s_f.transpose(1, 2).reshape(B, C, H, W)

                # Convert teacher feature from SAM format if needed
                if (
                    t_f.dim() == 4 and t_f.shape[1] < t_f.shape[3]
                ):  # Likely [B, H, W, C]
                    t_f = t_f.permute(0, 3, 1, 2)

                # Adapt student if needed
                adapter_key = s_layer.replace(".", "_")
                if adapter_key in self.adapters:
                    s_f = self.adapters[adapter_key](s_f)

                feature_loss += self.feature_loss_fn(s_f, t_f)
                count += 1

        return feature_loss / count if count > 0 else feature_loss

    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        losses = {}
        student_logits = student_outputs["masks"]
        teacher_logits = teacher_outputs["masks"]
        device = student_logits.device

        # Resize teacher logits once to match student shape
        if teacher_logits.shape != student_logits.shape:
            teacher_logits = F.interpolate(
                teacher_logits,
                size=student_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # 1. Task Loss
        losses["task_loss"] = self._compute_task_loss(student_logits, targets)

        # 2. Logit Distillation
        losses["distill_loss"] = self._compute_logit_loss(
            student_logits, teacher_logits
        )

        # 3. Feature Distillation
        losses["feature_loss"] = self._compute_feature_loss(device)

        # Total Loss (Weighted sum with fixed coefficients from cfg)
        total_loss = (
            self.alpha * losses["task_loss"]
            + self.beta * losses["distill_loss"]
            + self.gamma * losses["feature_loss"]
        )

        # 4. Uncertainty-weighted KD (optional, disabled by default)
        unc_kd_loss, unc_diagnostics = self._compute_uncertainty_kd_loss(
            student_logits, teacher_logits
        )
        total_loss = total_loss + self.kd_lambda * unc_kd_loss
        losses["uncertainty_kd_loss"] = unc_kd_loss
        losses.update(unc_diagnostics)  # flat floats flow into existing step/epoch logging

        losses["loss"] = total_loss

        # Log per-component raw and weighted values
        weights = {"task_loss": self.alpha, "distill_loss": self.beta, "feature_loss": self.gamma}
        for key, weight_val in weights.items():
            if key in losses:
                raw_val = losses[key].item()
                losses[f"{key}_raw"] = raw_val
                losses[f"{key}_weight"] = weight_val
                losses[f"{key}_weighted"] = weight_val * raw_val

        return losses

    def __del__(self):
        """Cleanup hooks."""
        if self.teacher_extractor:
            self.teacher_extractor.remove()
        if self.student_extractor:
            self.student_extractor.remove()
