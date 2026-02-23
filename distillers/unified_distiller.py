import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Dict, Any, List, Optional
from distillers.base_distiller import BaseDistiller


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
    4. Attention Map Distillation (MSE) - gamma_attn
    5. Alignment Layer Distillation (MSE on specific SAM output) - gamma_align
    """

    def __init__(self, cfg: Any, **kwargs):
        super().__init__(cfg)
        self.num_classes = cfg.data.num_classes

        # Hyperparameters
        self.gamma_align = cfg.method.get(
            "gamma_align", 0.0
        )  # alignment layer distillation
        self.gamma_attn = 1 - self.gamma_align  # attention map distillation

        # Advanced KD Coefficients
        self.lambda_boundary = cfg.method.get("lambda_boundary", 0.0)
        self.lambda_shape = cfg.method.get("lambda_shape", 0.0)
        self.lambda_uncertainty = cfg.method.get("lambda_uncertainty", 0.0)

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
                "gamma_attn": nn.Parameter(torch.tensor(float(self.gamma_attn))),
                "gamma_align": nn.Parameter(torch.tensor(float(self.gamma_align))),
                "lambda_boundary": nn.Parameter(
                    torch.tensor(float(self.lambda_boundary))
                ),
                "lambda_shape": nn.Parameter(torch.tensor(float(self.lambda_shape))),
                "lambda_uncertainty": nn.Parameter(
                    torch.tensor(float(self.lambda_uncertainty))
                ),
            }
        )
        for p in self.loss_weights.values():
            p.requires_grad = (
                False  # Weights are updated via GradNorm, not optimizer directly
            )

        # Fixed Sobel filters for Boundary KD
        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
            ).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
            ).view(1, 1, 3, 3),
        )

        # Task Loss components
        self.use_dice = cfg.method.get("use_dice", True)
        self.use_ce = cfg.method.get("use_ce", True)

        self.task_criterion = nn.BCEWithLogitsLoss()

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

        # Single feature align projection (for general features vs image_embeddings)
        self.align_proj = None
        if self.gamma_align > 0:
            # Use student_channels from student config if available, fallback to method config
            s_channels = getattr(cfg.student, "student_channels", 256)
            t_channels = cfg.method.get("teacher_alignment_channels", 256)

            if s_channels != t_channels:
                print(
                    f"[UnifiedDistiller] Creating align_proj: {s_channels} -> {t_channels}"
                )
                self.align_proj = nn.Conv2d(s_channels, t_channels, kernel_size=1)
            else:
                print(
                    f"[UnifiedDistiller] Skipping align_proj as dimensions match ({s_channels} == {t_channels})"
                )

        # Extractor placeholders
        self.teacher_extractor: Optional[FeatureExtractor] = None
        self.student_extractor: Optional[FeatureExtractor] = None
        self.teacher_attn_hooks = []
        self.student_attn_hooks = []
        self.teacher_attn_maps = []
        self.student_attn_maps = []

        # GradNorm reference parameters
        self.theta_ref = None

    def prepare(self, student: nn.Module, teacher: nn.Module):
        """Setup hooks for feature and attention extraction."""
        # 1. Intermediate Feature Extraction
        t_layers = list(self.layer_mapping.values())
        s_layers = list(self.layer_mapping.keys())

        if t_layers:
            self.teacher_extractor = FeatureExtractor(teacher, t_layers)
        if s_layers:
            self.student_extractor = FeatureExtractor(student, s_layers)

        # 2. Attention Map Extraction (for SAM-like models)
        if self.gamma_attn > 0:
            self._setup_attn_hooks(student, teacher)

        # 3. GradNorm reference parameters
        if self.use_gradnorm:
            for name, module in student.named_modules():
                if name == self.theta_ref_name:
                    # Use all parameters in the module as reference
                    self.theta_ref = list(module.parameters())
                    print(
                        f"[UnifiedDistiller] GradNorm reference module: {name} ({len(self.theta_ref)} parameters)"
                    )
                    break
            if self.theta_ref is None:
                # Fallback: search in named_parameters if module name didn't match exactly
                self.theta_ref = [
                    p for n, p in student.named_parameters() if self.theta_ref_name in n
                ]
                if self.theta_ref:
                    print(
                        f"[UnifiedDistiller] GradNorm reference parameters found by name pattern: {self.theta_ref_name} ({len(self.theta_ref)} parameters)"
                    )
                else:
                    print(
                        f"Warning: GradNorm reference layer '{self.theta_ref_name}' not found in student. GradNorm will be disabled."
                    )
                    self.use_gradnorm = False

    def _get_sobel_edge(self, p):
        """Compute edge map using Sobel filters."""
        B, C, H, W = p.shape
        p_flat = p.view(B * C, 1, H, W)
        edge_x = F.conv2d(p_flat, self.sobel_x, padding=1)
        edge_y = F.conv2d(p_flat, self.sobel_y, padding=1)
        edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        return edge.view(B, C, H, W)

    def _get_entropy(self, p):
        """Compute entropy map."""
        if self.num_classes == 1:
            # Binary entropy (p: probability [B, 1, H, W])
            entropy = -p * torch.log(p + 1e-6) - (1 - p) * torch.log(1 - p + 1e-6)
            return entropy
        else:
            # Multi-class entropy (p: probability [B, C, H, W])
            return -torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)

    def _get_sdt(self, p):
        """Compute Signed Distance Transform (SDT) map.
        Note: This is non-differentiable if computed with EDT,
        but we can use the teacher's SDT as a target.
        """
        device = p.device
        mask = (p > 0.5).cpu().numpy()
        sdt_maps = []
        for i in range(mask.shape[0]):  # Batch
            batch_sdt = []
            for j in range(mask.shape[1]):  # Channel
                m = mask[i, j]
                pos_dist = distance_transform_edt(m)
                neg_dist = distance_transform_edt(1 - m)
                sdt = pos_dist - neg_dist
                batch_sdt.append(sdt)
            sdt_maps.append(np.stack(batch_sdt))

        sdt_tensor = torch.from_numpy(np.stack(sdt_maps)).float().to(device)
        return sdt_tensor

    def _setup_attn_hooks(self, student, teacher):
        """Setup hooks to capture attention maps."""

        def get_attn_hook(target_list):
            def hook(module, input, output):
                if hasattr(module, "last_attn"):
                    target_list.append(module.last_attn)

            return hook

        # Teacher attention hooks (SAM image encoder)
        t_model = teacher.module if hasattr(teacher, "module") else teacher
        image_encoder = getattr(t_model, "image_encoder", None)
        if hasattr(t_model, "sam") and hasattr(t_model.sam, "image_encoder"):
            image_encoder = t_model.sam.image_encoder

        if image_encoder and hasattr(image_encoder, "blocks"):
            for blk in image_encoder.blocks:
                if hasattr(blk, "attn"):
                    self.teacher_attn_hooks.append(
                        blk.attn.register_forward_hook(
                            get_attn_hook(self.teacher_attn_maps)
                        )
                    )

        # Student attention hooks
        s_model = student.module if hasattr(student, "module") else student
        backbone = getattr(s_model, "backbone", s_model)
        if hasattr(backbone, "blocks"):
            for blk in backbone.blocks:
                if hasattr(blk, "attn"):
                    self.student_attn_hooks.append(
                        blk.attn.register_forward_hook(
                            get_attn_hook(self.student_attn_maps)
                        )
                    )

    def on_step_begin(self):
        """Clear extracted features and attention maps."""
        if self.teacher_extractor:
            self.teacher_extractor.clear()
        if self.student_extractor:
            self.student_extractor.clear()
        self.teacher_attn_maps.clear()
        self.student_attn_maps.clear()

    def _compute_task_loss(self, student_logits, targets):
        """Task Loss (GT Dice/CE) - alpha"""
        if self.alpha <= 0:
            return torch.tensor(0.0, device=student_logits.device)

        target_mask = targets.float()
        ce_loss = torch.tensor(0.0, device=student_logits.device)
        dice_loss = torch.tensor(0.0, device=student_logits.device)

        if self.use_ce:
            ce_loss = self.task_criterion(student_logits, target_mask)
        if self.use_dice:
            dice_loss = self.dice_loss(student_logits, target_mask)

        if self.use_ce and self.use_dice:
            return 0.5 * ce_loss + 0.5 * dice_loss
        return ce_loss if self.use_ce else dice_loss

    def _compute_logit_loss(self, student_logits, teacher_logits):
        """Logit Distillation (KL Div) - beta"""
        if self.beta <= 0:
            return torch.tensor(0.0, device=student_logits.device)

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
        distill_loss = distill_loss / (
            student_logits.shape[-2] * student_logits.shape[-1]
        )
        return distill_loss

    def _compute_feature_loss(self, device):
        """Feature Distillation (MSE on intermediate layers) - gamma"""
        if (
            self.gamma <= 0
            or self.teacher_extractor is None
            or self.student_extractor is None
        ):
            return torch.tensor(0.0, device=device)

        s_feats = self.student_extractor.get_features()
        t_feats = self.teacher_extractor.get_features()

        feature_loss = torch.tensor(0.0, device=device)
        count = 0
        for s_layer, t_layer in self.layer_mapping.items():
            s_f = s_feats.get(s_layer)
            t_f = t_feats.get(t_layer)

            if s_f is not None and t_f is not None:
                # Convert student feature from sequence to spatial if needed
                if s_f.dim() == 3:  # [B, N, C] format
                    B, N, C = s_f.shape
                    s_f = s_f[:, 1:, :]  # Remove CLS token
                    H = W = int((N - 1) ** 0.5)
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

    def _compute_attn_loss(self, device):
        """Attention Distillation (MSE) - gamma_attn"""
        if (
            self.gamma_attn <= 0
            or not self.teacher_attn_maps
            or not self.student_attn_maps
        ):
            return torch.tensor(0.0, device=device)

        num_blocks = min(len(self.teacher_attn_maps), len(self.student_attn_maps))
        attn_loss = torch.tensor(0.0, device=device)
        for i in range(num_blocks):
            t_a = self.teacher_attn_maps[i]
            s_a = self.student_attn_maps[i]

            if t_a.dim() == 4 and s_a.dim() == 3:
                B, H, N, _ = t_a.shape
                t_a = t_a.view(B * H, N, N)
            elif s_a.dim() == 4 and t_a.dim() == 3:
                B, H, N, _ = s_a.shape
                s_a = s_a.view(B * H, N, N)

            if t_a.shape != s_a.shape:
                t_a = F.interpolate(
                    t_a.unsqueeze(1),
                    size=s_a.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

            attn_loss += self.mse_loss(s_a, t_a)

        return attn_loss / num_blocks if num_blocks > 0 else attn_loss

    def _compute_align_loss(self, student_outputs, teacher_outputs, device):
        """Alignment Layer Distillation (MSE) - gamma_align"""
        if self.gamma_align <= 0:
            return torch.tensor(0.0, device=device)

        s_f = student_outputs.get("features")
        t_f = teacher_outputs.get("image_embeddings")

        if s_f is not None and t_f is not None:
            if self.align_proj:
                s_f = self.align_proj(s_f)
            if s_f.shape != t_f.shape:
                s_f = F.interpolate(
                    s_f, size=t_f.shape[-2:], mode="bilinear", align_corners=False
                )
            return self.mse_loss(s_f, t_f)
        return torch.tensor(0.0, device=device)

    def _compute_advanced_kd_losses(self, student_logits, teacher_logits):
        """Compute Boundary, Shape, and Uncertainty KD losses."""
        device = student_logits.device
        res = {
            "boundary": torch.tensor(0.0, device=device),
            "shape": torch.tensor(0.0, device=device),
            "uncertainty": torch.tensor(0.0, device=device),
        }

        if self.num_classes == 1:
            s_p = torch.sigmoid(student_logits)
            t_p = torch.sigmoid(teacher_logits)
        else:
            s_p = torch.softmax(student_logits, dim=1)
            t_p = torch.softmax(teacher_logits, dim=1)

        # Boundary KD
        if self.loss_weights["lambda_boundary"] > 0:
            s_edge = self._get_sobel_edge(s_p)
            t_edge = self._get_sobel_edge(t_p)
            res["boundary"] = F.l1_loss(s_edge, t_edge)

        # Shape KD
        if self.loss_weights["lambda_shape"] > 0:
            with torch.no_grad():
                t_sdt = self._get_sdt(t_p)
            res["shape"] = F.smooth_l1_loss(student_logits, t_sdt)

        # Uncertainty KD
        if self.loss_weights["lambda_uncertainty"] > 0:
            s_unc = self._get_entropy(s_p)
            t_unc = self._get_entropy(t_p)
            res["uncertainty"] = F.l1_loss(s_unc, t_unc)

        return res

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

        # 4. Attention Distillation
        losses["attn_loss"] = self._compute_attn_loss(device)

        # 5. Alignment Layer Distillation
        losses["align_loss"] = self._compute_align_loss(
            student_outputs, teacher_outputs, device
        )

        # 6-8. Advanced KD Losses
        adv_losses = self._compute_advanced_kd_losses(student_logits, teacher_logits)
        losses["boundary_loss"] = adv_losses["boundary"]
        losses["shape_loss"] = adv_losses["shape"]
        losses["uncertainty_loss"] = adv_losses["uncertainty"]

        # Total Loss (Weighted)
        total_loss = (
            self.loss_weights["alpha"] * losses["task_loss"]
            + self.loss_weights["beta"] * losses["distill_loss"]
            + self.loss_weights["gamma"] * losses["feature_loss"]
            + self.loss_weights["gamma_attn"] * losses["attn_loss"]
            + self.loss_weights["gamma_align"] * losses["align_loss"]
            + self.loss_weights["lambda_boundary"] * losses["boundary_loss"]
            + self.loss_weights["lambda_shape"] * losses["shape_loss"]
            + self.loss_weights["lambda_uncertainty"] * losses["uncertainty_loss"]
        )

        losses["loss"] = total_loss

        # 9. GradNorm Balancing & Detailed Logging
        if self.theta_ref and self.training:
            self._apply_gradnorm(losses)
        else:
            # Basic logging even if not training or GradNorm is not applicable
            key_map = {
                "task_loss": "alpha",
                "distill_loss": "beta",
                "feature_loss": "gamma",
                "attn_loss": "gamma_attn",
                "align_loss": "gamma_align",
                "boundary_loss": "lambda_boundary",
                "shape_loss": "lambda_shape",
                "uncertainty_loss": "lambda_uncertainty",
            }
            for key, weight_key in key_map.items():
                if key in losses:
                    raw_val = losses[key].item()
                    weight_val = self.loss_weights[weight_key].item()
                    losses[f"{key}_raw"] = raw_val
                    losses[f"{key}_weight"] = weight_val
                    losses[f"{key}_weighted"] = weight_val * raw_val

        return losses

    def _apply_gradnorm(self, losses: Dict[str, torch.Tensor]):
        """Adjust loss weights and log detailed gradient contributions."""
        # Map loss keys to weight parameter names
        key_map = {
            "task_loss": "alpha",
            "distill_loss": "beta",
            "feature_loss": "gamma",
            "attn_loss": "gamma_attn",
            "align_loss": "gamma_align",
            "boundary_loss": "lambda_boundary",
            "shape_loss": "lambda_shape",
            "uncertainty_loss": "lambda_uncertainty",
        }

        norms = []
        active_keys = []

        # 1. Capture basic metrics and compute gradient norms
        # Iterate over list to avoid "dictionary changed size during iteration"
        for key, loss_val in list(losses.items()):
            if (
                key == "loss"
                or not isinstance(loss_val, torch.Tensor)
                or loss_val.numel() == 0
                or loss_val.item() == 0
            ):
                continue

            weight_key = key_map.get(key)
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
        for hook in self.teacher_attn_hooks:
            hook.remove()
        for hook in self.student_attn_hooks:
            hook.remove()
        if self.teacher_extractor:
            self.teacher_extractor.remove()
        if self.student_extractor:
            self.student_extractor.remove()
