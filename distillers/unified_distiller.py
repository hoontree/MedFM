import logging
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Callable
from distillers.base_distiller import BaseDistiller, resolve_loss_weight

logger = logging.getLogger(__name__)


from utils.criterion import (
    TaskLoss,
    LogitDistillLoss,
    UncertaintyWeightedKDLoss,
    ChannelWiseDistillLoss,
    ReliabilityWeightedKDLoss,
)
from hydra.utils import instantiate
from monai.losses import DiceLoss, MaskedDiceLoss, FocalLoss, GeneralizedDiceLoss, DiceFocalLoss
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
    """Unified distillation supporting multiple loss components.

    Each loss is gated by a non-negative weight in ``cfg.method``; setting
    the weight to 0 disables (and skips constructing) the corresponding
    module to save memory. Weight keys are semantic ``w_<component>``:

        w_task            - Task loss (GT Dice / CE)
        w_logit_kd        - Logit KD (KL divergence)
        w_logit_cwd       - Channel-wise distillation on the final logit map
        w_feature_cwd     - Channel-wise distillation on intermediate features
        w_reliability_kd  - Reliability-weighted KD (uses build_reliability_map)
        w_uncertainty_kd  - Uncertainty-weighted KD (entropy-driven weighting)

    Legacy aliases (``alpha`` / ``beta`` / ``delta`` / ``zeta`` / ``eta`` /
    ``kd_lambda``) are still accepted with a deprecation warning.

    Optional ``cfg.method.normalize_weights=true`` divides every active
    weight by their sum so the total loss is a convex combination of
    components. Disabled by default to preserve the original scale.
    """

    def __init__(self, cfg: Any, **kwargs):
        super().__init__(cfg)
        self.num_classes = cfg.data.num_classes
        m = cfg.method

        # --- Loss weights -------------------------------------------------
        # (w_task / w_logit_kd come from BaseDistiller; the rest are added here.)
        self.w_logit_cwd = resolve_loss_weight(m, "w_logit_cwd", 0.0)
        self.w_feature_cwd = resolve_loss_weight(m, "w_feature_cwd", 0.0)
        self.w_reliability_kd = resolve_loss_weight(m, "w_reliability_kd", 0.0)
        self.w_uncertainty_kd = resolve_loss_weight(m, "w_uncertainty_kd", 1.0)
        self.use_uncertainty_kd = bool(m.get("use_uncertainty_weighted_kd", False))

        # Optional convex-combination normalisation
        if m.get("normalize_weights", False):
            self._normalize_weights()

        # --- Task loss (w_task) ------------------------------------------
        self.use_dice = m.get("use_dice", True)
        self.use_ce = m.get("use_ce", True)
        self.task_loss_fn = (
            TaskLoss(
                use_ce=self.use_ce,
                use_dice=self.use_dice,
                pos_weight=float(m.get("pos_weight", 5.0)),
            )
            if self.w_task > 0
            else None
        )

        # --- Logit KD (w_logit_kd) ---------------------------------------
        self.logit_loss_fn = (
            LogitDistillLoss(temperature=self.temperature)
            if self.w_logit_kd > 0
            else None
        )

        # --- Channel-wise distillation (w_logit_cwd / w_feature_cwd) -----
        self.cwd_temperature = m.get("cwd_temperature", self.temperature)
        # CWD module is shared by both logit-CWD and feature-CWD; also used
        # as the default logit-KD when num_classes >= 2.
        self.cwd_loss_fn = (
            ChannelWiseDistillLoss(temperature=self.cwd_temperature)
            if (
                self.w_logit_cwd > 0
                or self.w_feature_cwd > 0
                or (self.num_classes >= 2 and self.w_logit_kd > 0)
            )
            else None
        )

        # --- Reliability-weighted KD (w_reliability_kd) ------------------
        # `reliability_kd` is a `_partial_` instantiate spec for
        # build_reliability_map: Hydra binds the static config args here and
        # the per-step tensors (teacher/student logits, gt) are supplied in
        # forward. `temperature` / `eps` are reused for the loss module.
        rkd_cfg = m.get("reliability_kd", {}) or {}
        self.reliability_kd_fn = (
            ReliabilityWeightedKDLoss(
                temperature=float(rkd_cfg.get("temperature", self.temperature)),
                eps=float(rkd_cfg.get("eps", 1e-8)),
            )
            if self.w_reliability_kd > 0
            else None
        )
        self._rkd_use_student_bypass = bool(rkd_cfg.get("use_student_bypass", False))
        self._build_reliability_map: Optional[Callable] = (
            instantiate(rkd_cfg) if self.w_reliability_kd > 0 else None
        )

        # --- Uncertainty-weighted KD (w_uncertainty_kd) ------------------
        self.uncertainty_kd_fn = (
            UncertaintyWeightedKDLoss(
                tau=m.get("kd_tau", self.temperature),
                weight_type=m.get("uncertainty_weight_type", "linear"),
                beta=m.get("uncertainty_beta", 1.0),
                eps=m.get("uncertainty_eps", 1e-8),
            )
            if self.use_uncertainty_kd and self.w_uncertainty_kd > 0
            else None
        )

        # --- Feature alignment infrastructure ----------------------------
        self.feature_pairs: List[int] = self._parse_feature_pairs(m)
        self.adapters = nn.ModuleDict()  # lazy-initialized in first forward
        self._adapters_built = False
        self.student_layers: Dict[int, str] = {}
        self.teacher_layers: Dict[int, str] = {}
        self.teacher_extractor: Optional[FeatureExtractor] = None
        self.student_extractor: Optional[FeatureExtractor] = None

    def _normalize_weights(self):
        """Rescale active loss weights so the components sum to ~1.

        Iterates over the canonical ``w_*`` weights, including the
        uncertainty-KD weight when enabled. Zero weights are treated as
        disabled and left untouched. Skipped when no positive weight is
        configured.
        """
        names = [
            "w_task",
            "w_logit_kd",
            "w_logit_cwd",
            "w_feature_cwd",
            "w_reliability_kd",
        ]
        if self.use_uncertainty_kd:
            names.append("w_uncertainty_kd")

        pairs = [(n, float(getattr(self, n))) for n in names]
        total = sum(v for _, v in pairs if v > 0)
        if total <= 0:
            return
        normalized = []
        for n, v in pairs:
            new_v = v / total if v > 0 else v
            setattr(self, n, new_v)
            normalized.append(new_v)
        logger.info(
            "Normalised loss weights: " + " ".join(f"{n}=%.4f" for n in names),
            *normalized,
        )

    @staticmethod
    def _parse_feature_pairs(method_cfg: Any) -> List[int]:
        """Accept the new ``feature_pairs: [7, 9, 11]`` form, falling back to
        the legacy ``layer_mapping`` dict (block indices parsed from path)."""
        pairs = method_cfg.get("feature_pairs", None)
        if pairs:
            return [int(p) for p in pairs]

        legacy = method_cfg.get("layer_mapping", {}) or {}
        if legacy:
            logger.warning(
                "`layer_mapping` is deprecated; use `feature_pairs: [<block_idx>, ...]` "
                "instead. Auto-extracting block indices from legacy keys."
            )
            indices = []
            for key in legacy.keys():
                m = re.search(r"blocks\.(\d+)", key)
                if m:
                    indices.append(int(m.group(1)))
            return sorted(set(indices))
        return []

    @staticmethod
    def _resolve_block_paths(model: nn.Module, indices: List[int]) -> Dict[int, str]:
        """Find module paths matching `*blocks.{idx}` for each requested index.

        Returns a dict {block_idx: dotted_path}. Picks the shortest matching
        path when multiple candidates exist (e.g. prefers `blocks.7` over
        `image_encoder.blocks.7.attn` — though the regex already excludes
        the latter)."""
        if not indices:
            return {}
        wanted = set(indices)
        candidates: Dict[int, List[str]] = {i: [] for i in wanted}
        pattern = re.compile(r"(?:^|\.)blocks\.(\d+)$")
        for name, _ in model.named_modules():
            m = pattern.search(name)
            if m:
                idx = int(m.group(1))
                if idx in wanted:
                    candidates[idx].append(name)
        resolved: Dict[int, str] = {}
        for idx, paths in candidates.items():
            if paths:
                resolved[idx] = min(paths, key=len)
        return resolved

    def prepare(self, student: nn.Module, teacher: nn.Module):
        """Resolve per-model block paths and set up extraction hooks."""
        # Skip all feature-hook setup when feature-CWD is disabled: otherwise the
        # extractors capture intermediate features every forward step only for
        # them to be discarded (wasted compute/memory on every non-feature run).
        if not self.feature_pairs or self.w_feature_cwd <= 0:
            return

        self.student_layers = self._resolve_block_paths(student, self.feature_pairs)
        self.teacher_layers = self._resolve_block_paths(teacher, self.feature_pairs)

        missing_s = [i for i in self.feature_pairs if i not in self.student_layers]
        missing_t = [i for i in self.feature_pairs if i not in self.teacher_layers]
        if missing_s:
            logger.warning("Student has no `*.blocks.%s` modules", missing_s)
        if missing_t:
            logger.warning("Teacher has no `*.blocks.%s` modules", missing_t)

        usable = [i for i in self.feature_pairs
                  if i in self.student_layers and i in self.teacher_layers]
        if not usable:
            logger.warning("No usable feature pairs after resolution; "
                           "feature distillation will be skipped.")
            return

        logger.info("Feature pairs resolved (block_idx → student / teacher):")
        for i in usable:
            logger.info("  %d: %s  ↔  %s", i, self.student_layers[i], self.teacher_layers[i])

        self.student_extractor = FeatureExtractor(
            student, [self.student_layers[i] for i in usable]
        )
        self.teacher_extractor = FeatureExtractor(
            teacher, [self.teacher_layers[i] for i in usable]
        )

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
        """Task Loss (GT Dice/CE) - w_task."""
        if self.task_loss_fn is None:
            return self._zero(student_logits.device)
        return self.task_loss_fn(student_logits, targets)

    def _compute_logit_loss(self, student_logits, teacher_logits):
        """Logit Distillation - w_logit_kd.

        Binary segmentation (num_classes == 1): standard Hinton-style KD
        (sigmoid + KL) via LogitDistillLoss.
        Multi-class (num_classes >= 2): Channel-wise Distillation (CWD,
        ICCV 2021) on the logit map — softmax over spatial dims per class
        channel is a better fit than per-pixel softmax-KL for dense
        segmentation.
        """
        if self.w_logit_kd <= 0:
            return self._zero(student_logits.device)
        if self.num_classes >= 2:
            return self.cwd_loss_fn(student_logits, teacher_logits)
        return self.logit_loss_fn(student_logits, teacher_logits)

    def _compute_reliability_kd_loss(self, student_logits, teacher_logits, targets):
        """Reliability-weighted KD - w_reliability_kd.

        Builds a per-pixel reliability map from the teacher prediction
        and ground truth, then weights ``ReliabilityWeightedKDLoss`` by
        it. GT-dependent factors (boundary attenuation / student-bypass)
        use ``targets`` reduced to a class-index mask when multi-class.
        The student-bypass gate additionally consumes ``student_logits``.
        """
        # build_reliability_map expects [B, H, W] GT — convert one-hot.
        gt = targets
        if gt.dim() == 4:
            gt = gt.argmax(dim=1) if gt.shape[1] > 1 else gt[:, 0]
        gt = gt.long()

        with torch.no_grad():
            reliability, components = self._build_reliability_map(
                teacher_logits=teacher_logits,
                student_logits=student_logits if self._rkd_use_student_bypass else None,
                gt=gt,
                return_components=True,
            )

        loss = self.reliability_kd_fn(
            student_logits, teacher_logits=teacher_logits, reliability=reliability
        )
        diagnostics = {
            "reliability_kd_loss_weighted": (self.w_reliability_kd * loss).item(),
            "mean_reliability": reliability.mean().item(),
        }
        # Log the scale of each reliability component (mean over pixels) so
        # the relative contribution of confidence / entropy / gates is visible.
        for name, factor in components.items():
            diagnostics[f"reliability/{name}_mean"] = factor.mean().item()
        return loss, diagnostics

    def _compute_uncertainty_kd_loss(self, student_logits, teacher_logits):
        """Uncertainty-weighted KD loss - w_uncertainty_kd."""
        if self.uncertainty_kd_fn is None:
            return self._zero(student_logits.device), {}
        loss, uncertainty, weight = self.uncertainty_kd_fn(student_logits, teacher_logits)
        diagnostics = {
            "uncertainty_kd_loss_weighted": (self.w_uncertainty_kd * loss).item(),
            "mean_teacher_uncertainty": uncertainty.mean().item(),
            "mean_kd_weight": weight.mean().item(),
        }
        return loss, diagnostics

    @staticmethod
    def _to_bchw(feat: torch.Tensor) -> Optional[torch.Tensor]:
        """Normalize a hooked tensor to [B, C, H, W]. Returns None if the
        shape can't be interpreted as a square spatial token grid."""
        if feat.dim() == 3:
            B, N, C = feat.shape
            H_sq = int(N ** 0.5)
            if H_sq * H_sq == N:
                H = W = H_sq
            else:
                N_no_cls = N - 1
                H_sq = int(N_no_cls ** 0.5)
                if H_sq * H_sq == N_no_cls:
                    feat = feat[:, 1:, :]
                    H = W = H_sq
                else:
                    return None
            return feat.transpose(1, 2).reshape(B, feat.shape[-1], H, W)
        if feat.dim() == 4 and feat.shape[1] < feat.shape[3]:
            # [B, H, W, C] (SAM-style) → [B, C, H, W]
            return feat.permute(0, 3, 1, 2)
        return feat

    def _build_adapters(self, pairs: List[Tuple[int, torch.Tensor, torch.Tensor]]):
        """Lazily create 1×1 channel adapters on student side using the
        actually-observed channel counts."""
        device = next(iter(p[1] for p in pairs)).device
        for idx, s_f, t_f in pairs:
            s_ch, t_ch = s_f.shape[1], t_f.shape[1]
            key = f"block_{idx}"
            if s_ch != t_ch and key not in self.adapters:
                self.adapters[key] = FeatureAdapter(s_ch, t_ch).to(device)
        self._adapters_built = True

    def _collect_feature_pairs(self):
        """Yield (student_feat, teacher_feat) tensors aligned to [B,C,H,W]
        with channel adaptation applied. Adapters are built lazily on the
        first forward, using the observed channel counts."""
        s_feats = self.student_extractor.get_features()
        t_feats = self.teacher_extractor.get_features()

        # Stage 1: collect & normalize shapes
        staged: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
        for idx in self.feature_pairs:
            s_path = self.student_layers.get(idx)
            t_path = self.teacher_layers.get(idx)
            if s_path is None or t_path is None:
                continue
            s_f = s_feats.get(s_path)
            t_f = t_feats.get(t_path)
            if s_f is None or t_f is None:
                continue
            s_f = self._to_bchw(s_f)
            t_f = self._to_bchw(t_f)
            if s_f is None or t_f is None:
                continue
            staged.append((idx, s_f, t_f))

        if not staged:
            return

        # Stage 2: lazy adapter init on first call
        if not self._adapters_built:
            self._build_adapters(staged)

        # Stage 3: apply adapters and yield
        for idx, s_f, t_f in staged:
            key = f"block_{idx}"
            if key in self.adapters:
                s_f = self.adapters[key](s_f)
            yield s_f, t_f

    def _compute_logit_cwd_loss(self, student_logits, teacher_logits):
        """Channel-wise distillation on the final logit map - w_logit_cwd."""
        if self.w_logit_cwd <= 0 or self.cwd_loss_fn is None:
            return self._zero(student_logits.device)
        # Binary segmentation produces 1-channel fg logits, which collapses
        # CWD's channel-softmax into a degenerate single-channel signal.
        # Expand to [bg, fg] = [-x, x] so spatial saliency is contrasted
        # against an explicit background channel before channel-wise KL.
        student_logits = self._expand_binary_logits(student_logits)
        teacher_logits = self._expand_binary_logits(teacher_logits)
        return self.cwd_loss_fn(student_logits, teacher_logits)

    @staticmethod
    def _expand_binary_logits(logits):
        if logits.dim() == 4 and logits.shape[1] == 1:
            return torch.cat([-logits, logits], dim=1)
        return logits

    def _compute_feature_cwd_loss(self, device):
        """Channel-wise distillation on intermediate features - w_feature_cwd."""
        if (
            self.w_feature_cwd <= 0
            or self.cwd_loss_fn is None
            or self.teacher_extractor is None
            or self.student_extractor is None
        ):
            return self._zero(device)

        cwd_loss = self._zero(device)
        count = 0
        for s_f, t_f in self._collect_feature_pairs():
            cwd_loss = cwd_loss + self.cwd_loss_fn(s_f, t_f)
            count += 1
        return cwd_loss / count if count > 0 else cwd_loss

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

        # Compute only the components whose weight is non-zero so that
        # disabled losses never appear in the returned dict. The total loss
        # accumulates each active component as it is computed.
        total_loss = self._zero(device)

        # 1. Task Loss
        if self.w_task > 0:
            losses["task_loss"] = self._compute_task_loss(student_logits, targets)
            total_loss = total_loss + self.w_task * losses["task_loss"]

        # 2. Logit Distillation
        if self.w_logit_kd > 0:
            losses["distill_loss"] = self._compute_logit_loss(
                student_logits, teacher_logits
            )
            total_loss = total_loss + self.w_logit_kd * losses["distill_loss"]

        # 3. Channel-wise distillation (CWD) on logits and/or features
        if self.w_logit_cwd > 0:
            losses["logit_cwd_loss"] = self._compute_logit_cwd_loss(
                student_logits, teacher_logits
            )
            total_loss = total_loss + self.w_logit_cwd * losses["logit_cwd_loss"]
        if self.w_feature_cwd > 0:
            losses["feature_cwd_loss"] = self._compute_feature_cwd_loss(device)
            total_loss = total_loss + self.w_feature_cwd * losses["feature_cwd_loss"]

        # 4. Reliability-weighted KD
        if self.w_reliability_kd > 0:
            rkd_loss, rkd_diagnostics = self._compute_reliability_kd_loss(
                student_logits, teacher_logits, targets
            )
            losses["reliability_kd_loss"] = rkd_loss
            losses.update(rkd_diagnostics)
            total_loss = total_loss + self.w_reliability_kd * rkd_loss

        # 5. Uncertainty-weighted KD
        if self.w_uncertainty_kd > 0:
            unc_kd_loss, unc_diagnostics = self._compute_uncertainty_kd_loss(
                student_logits, teacher_logits
            )
            losses["uncertainty_kd_loss"] = unc_kd_loss
            losses.update(unc_diagnostics)
            total_loss = total_loss + self.w_uncertainty_kd * unc_kd_loss

        losses["loss"] = total_loss

        # Log per-component weighted values (active components only)
        weights = {
            "task_loss": self.w_task,
            "distill_loss": self.w_logit_kd,
            "logit_cwd_loss": self.w_logit_cwd,
            "feature_cwd_loss": self.w_feature_cwd,
            "reliability_kd_loss": self.w_reliability_kd,
            "uncertainty_kd_loss": self.w_uncertainty_kd,
        }
        for key, weight_val in weights.items():
            if key in losses:
                losses[f"{key}_weighted"] = weight_val * losses[key].item()

        return losses

    def __del__(self):
        """Cleanup hooks."""
        if self.teacher_extractor:
            self.teacher_extractor.remove()
        if self.student_extractor:
            self.student_extractor.remove()
