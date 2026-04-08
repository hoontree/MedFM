import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from distillers.base_distiller import BaseDistiller

logger = logging.getLogger(__name__)


from utils.sam_utils import DiceLoss
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

    LOSS_WEIGHT_MAP = {
        "task_loss": "alpha",
        "distill_loss": "beta",
        "feature_loss": "gamma",
    }

    def __init__(self, cfg: Any, **kwargs):
        super().__init__(cfg)
        self.num_classes = cfg.data.num_classes

        # GradNorm settings
        self.use_gradnorm = cfg.method.get("use_gradnorm", False)
        self.gradnorm_alpha = cfg.method.get("gradnorm_alpha", 0.1)
        self.theta_ref_name = cfg.method.get("reference_layer", "backbone.blocks.11")

        # Initial weights for GradNorm (if enabled, these will be optimized)
        self.loss_weights = nn.ParameterDict(
            {
                "alpha": nn.Parameter(torch.tensor(float(self.alpha))),
                "beta": nn.Parameter(torch.tensor(float(self.beta))),
                "gamma": nn.Parameter(torch.tensor(float(self.gamma))),
            }
        )
        for p in self.loss_weights.values():
            p.requires_grad = (
                False  # Weights are updated via GradNorm, not optimizer directly
            )

        # Task Loss components
        self.use_dice = cfg.method.get("use_dice", True)
        self.use_ce = cfg.method.get("use_ce", True)

        if self.num_classes == 1:
            self.register_buffer("pos_weight", torch.tensor([5.0]))
            self.task_criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            self.task_criterion = nn.CrossEntropyLoss()

        self.dice_loss = DiceLoss(self.num_classes)
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss = nn.MSELoss()

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

        # GradNorm reference parameters
        self.theta_ref = None

    def prepare(self, student: nn.Module, teacher: nn.Module):
        """Setup hooks for feature extraction."""
        # 1. Intermediate Feature Extraction
        t_layers = list(self.layer_mapping.values())
        s_layers = list(self.layer_mapping.keys())

        if t_layers:
            self.teacher_extractor = FeatureExtractor(teacher, t_layers)
        if s_layers:
            self.student_extractor = FeatureExtractor(student, s_layers)

        # 2. GradNorm reference parameters
        if self.use_gradnorm:
            for name, module in student.named_modules():
                if name == self.theta_ref_name:
                    # Use all parameters in the module as reference
                    self.theta_ref = list(module.parameters())
                    logger.info(
                        f"[UnifiedDistiller] GradNorm reference module: {name} ({len(self.theta_ref)} parameters)"
                    )
                    break
            if self.theta_ref is None:
                # Fallback: search in named_parameters if module name didn't match exactly
                self.theta_ref = [
                    p for n, p in student.named_parameters() if self.theta_ref_name in n
                ]
                if self.theta_ref:
                    logger.info(
                        f"[UnifiedDistiller] GradNorm reference parameters found by name pattern: {self.theta_ref_name} ({len(self.theta_ref)} parameters)"
                    )
                else:
                    logger.warning(
                        f"GradNorm reference layer '{self.theta_ref_name}' not found in student. GradNorm will be disabled."
                    )
                    self.use_gradnorm = False

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

        target_mask = targets.float()
        losses = []
        if self.use_ce:
            if self.num_classes == 1:
                losses.append(self.task_criterion(student_logits, target_mask))
            else:
                target_idx = targets.argmax(dim=1).long()
                losses.append(self.task_criterion(student_logits, target_idx))
        if self.use_dice:
            use_softmax = self.num_classes > 1
            losses.append(self.dice_loss(student_logits, target_mask, softmax=use_softmax, sigmoid=not use_softmax))
        return (
            sum(losses) / len(losses) if losses else self._zero(student_logits.device)
        )

    def _compute_logit_loss(self, student_logits, teacher_logits):
        """Logit Distillation (KL Div) - beta"""
        if self.beta <= 0:
            return self._zero(student_logits.device)

        if self.num_classes == 1:
            s_soft = F.log_softmax(
                torch.cat([torch.zeros_like(student_logits), student_logits], dim=1)
                / self.temperature,
                dim=1,
            )
            t_soft = F.softmax(
                torch.cat([torch.zeros_like(teacher_logits), teacher_logits], dim=1)
                / self.temperature,
                dim=1,
            )
        else:
            s_soft = F.log_softmax(student_logits / self.temperature, dim=1)
            t_soft = F.softmax(teacher_logits / self.temperature, dim=1)

        distill_loss = self.kl_div(s_soft, t_soft) * (self.temperature**2)
        
        # PyTorch KLDivLoss with batchmean divides by batch_size, but sums over spatial dimensions.
        # We need to average over spatial dimensions to match typical segmentation task loss scale.
        if student_logits.dim() == 4:
            distill_loss = distill_loss / (student_logits.shape[2] * student_logits.shape[3])

        return distill_loss

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

                if s_f.shape != t_f.shape:
                    t_f = F.interpolate(
                        t_f,
                        size=s_f.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                feature_loss += self.mse_loss(s_f, t_f)
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

        # Total Loss (Weighted)
        total_loss = (
            self.loss_weights["alpha"] * losses["task_loss"]
            + self.loss_weights["beta"] * losses["distill_loss"]
            + self.loss_weights["gamma"] * losses["feature_loss"]
        )

        losses["loss"] = total_loss

        # 9. GradNorm Balancing & Detailed Logging
        if self.theta_ref and self.training:
            self._apply_gradnorm(losses)
        else:
            # Basic logging even if not training or GradNorm is not applicable
            for key, weight_key in self.LOSS_WEIGHT_MAP.items():
                if key in losses:
                    raw_val = losses[key].item()
                    weight_val = self.loss_weights[weight_key].item()
                    losses[f"{key}_raw"] = raw_val
                    losses[f"{key}_weight"] = weight_val
                    losses[f"{key}_weighted"] = weight_val * raw_val

        return losses

    def _apply_gradnorm(self, losses: Dict[str, torch.Tensor]):
        """Adjust loss weights and log detailed gradient contributions."""
        norms = []
        active_keys = []

        # 1. Capture basic metrics and compute gradient norms
        # Iterate over list to avoid "dictionary changed size during iteration"
        for key, loss_val in list(losses.items()):
            if (
                key == "loss"
                or not isinstance(loss_val, torch.Tensor)
                or loss_val.numel() == 0
                or abs(loss_val.item()) < 1e-8
            ):
                continue

            weight_key = self.LOSS_WEIGHT_MAP.get(key)
            if weight_key is None:
                continue

            weight = self.loss_weights[weight_key]
            weight_val = weight.item()
            raw_val = loss_val.item()

            # Log basic metrics
            losses[f"{key}_raw"] = raw_val
            losses[f"{key}_weight"] = weight_val
            losses[f"{key}_weighted"] = weight_val * raw_val

            if not self.use_gradnorm or weight_val <= 0:
                continue

            # Compute gradient of weighted loss: || ∇_{θ_s} (w_i * L_i) ||
            grads_weighted = torch.autograd.grad(
                weight * loss_val, self.theta_ref, retain_graph=True, allow_unused=True
            )
            valid_grads_weighted = [
                g.contiguous().view(-1) for g in grads_weighted if g is not None
            ]

            if valid_grads_weighted:
                norm_weighted_val = torch.norm(
                    torch.cat(valid_grads_weighted), p=2
                ).item()
                losses[f"{key}_grad_norm_weighted"] = norm_weighted_val
                losses[f"{key}_grad_norm_unweighted"] = norm_weighted_val / (
                    weight_val + 1e-8
                )

                norms.append(torch.tensor(norm_weighted_val, device=loss_val.device))
                active_keys.append(weight_key)

        if not norms or not self.use_gradnorm:
            return

        # 2. Update weights to equalize norms (GradNorm Multiplicative Update)
        norms_stack = torch.stack(norms)
        avg_norm = norms_stack.mean().item()

        for i, weight_key in enumerate(active_keys):
            current_norm = norms[i].item()
            if current_norm > 0:
                ratio = avg_norm / current_norm
                new_weight = self.loss_weights[weight_key].item() * (
                    ratio**self.gradnorm_alpha
                )
                # Avoid extreme values
                new_weight = max(1e-4, min(10.0, new_weight))
                self.loss_weights[weight_key].data.fill_(new_weight)

        # 3. Log normalized weights
        current_weights = [self.loss_weights[k].item() for k in active_keys]
        w_sum = sum(current_weights)
        if w_sum > 0:
            norm_factor = len(active_keys) / w_sum
            for k in active_keys:
                losses[f"weight_normalized/{k}"] = (
                    self.loss_weights[k].item() * norm_factor
                )

    def __del__(self):
        """Cleanup hooks."""
        if self.teacher_extractor:
            self.teacher_extractor.remove()
        if self.student_extractor:
            self.student_extractor.remove()
