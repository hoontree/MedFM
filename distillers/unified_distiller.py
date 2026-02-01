import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.alpha = cfg.method.get("alpha", 1.0)
        self.beta = cfg.method.get("beta", 0.0)
        self.gamma = cfg.method.get("gamma", 0.0)
        self.gamma_attn = cfg.method.get("gamma_attn", 0.0)
        self.gamma_align = cfg.method.get("gamma_align", 0.0)

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
            s_channels = cfg.method.get("student_channels", 48)
            t_channels = cfg.method.get("teacher_alignment_channels", 256)
            if s_channels != t_channels:
                self.align_proj = nn.Conv2d(s_channels, t_channels, kernel_size=1)

        # Extractor placeholders
        self.teacher_extractor: Optional[FeatureExtractor] = None
        self.student_extractor: Optional[FeatureExtractor] = None
        self.teacher_attn_hooks = []
        self.student_attn_hooks = []
        self.teacher_attn_maps = []
        self.student_attn_maps = []

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

        # 1. Task Loss
        ce_loss = torch.tensor(0.0, device=device)
        dice_loss = torch.tensor(0.0, device=device)
        if self.alpha > 0:
            target_mask = targets.float()
            if self.use_ce:
                ce_loss = self.task_criterion(student_logits, target_mask)
            if self.use_dice:
                dice_loss = self.dice_loss(student_logits, target_mask)

            task_loss = 0.0
            if self.use_ce and self.use_dice:
                task_loss = 0.5 * ce_loss + 0.5 * dice_loss
            else:
                task_loss = ce_loss if self.use_ce else dice_loss
            losses["task_loss"] = task_loss
        else:
            task_loss = torch.tensor(0.0, device=device)

        # 2. Logit Distillation
        distill_loss = torch.tensor(0.0, device=device)
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

            if self.num_classes == 1:
                s_soft = F.log_softmax(
                    torch.cat([torch.zeros_like(student_logits), student_logits], dim=1)
                    / self.temperature,
                    dim=1,
                )
                t_soft = F.softmax(
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
                s_soft = F.log_softmax(student_logits / self.temperature, dim=1)
                t_soft = F.softmax(teacher_logits_resized / self.temperature, dim=1)

            distill_loss = self.kl_div(s_soft, t_soft) * (self.temperature**2)
            distill_loss = distill_loss / (
                student_logits.shape[-2] * student_logits.shape[-1]
            )
            losses["distill_loss"] = distill_loss

        # 3. Feature Distillation
        feature_loss = torch.tensor(0.0, device=device)
        if self.gamma > 0 and self.teacher_extractor and self.student_extractor:
            s_feats = self.student_extractor.get_features()
            t_feats = self.teacher_extractor.get_features()

            count = 0
            for s_layer, t_layer in self.layer_mapping.items():
                s_key = s_layer  # extractor uses the full name usually
                t_key = t_layer

                s_f = s_feats.get(s_key)
                t_f = t_feats.get(t_key)

                if s_f is not None and t_f is not None:
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
            if count > 0:
                feature_loss /= count
            losses["feature_loss"] = feature_loss

        # 4. Attention Distillation
        attn_loss = torch.tensor(0.0, device=device)
        if self.gamma_attn > 0 and self.teacher_attn_maps and self.student_attn_maps:
            num_blocks = min(len(self.teacher_attn_maps), len(self.student_attn_maps))
            for i in range(num_blocks):
                t_a = self.teacher_attn_maps[i]
                s_a = self.student_attn_maps[i]

                # Reshape if needed [B, H, N, N] vs [B*H, N, N]
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
            if num_blocks > 0:
                attn_loss /= num_blocks
            losses["attn_loss"] = attn_loss

        # 5. Alignment Layer Distillation
        align_loss = torch.tensor(0.0, device=device)
        if self.gamma_align > 0:
            s_f = student_outputs.get("features")
            t_f = teacher_outputs.get(
                "image_embeddings"
            )  # Alignment layer output from LoRA_Sam

            if s_f is not None and t_f is not None:
                if self.align_proj:
                    s_f = self.align_proj(s_f)
                if s_f.shape != t_f.shape:
                    s_f = F.interpolate(
                        s_f, size=t_f.shape[-2:], mode="bilinear", align_corners=False
                    )
                align_loss = self.mse_loss(s_f, t_f)
                losses["align_loss"] = align_loss

        # Total Loss
        total_loss = (
            self.alpha * task_loss
            + self.beta * distill_loss
            + self.gamma * feature_loss
            + self.gamma_attn * attn_loss
            + self.gamma_align * align_loss
        )
        losses["loss"] = total_loss

        return losses

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
