"""Meta-Reliability Distillation.

This module turns the hand-crafted, *multiplicative* reliability map of
``utils.reliability_kd.build_reliability_map`` into a **learnable per-pixel
function** ``r = Rθ(z)`` and provides the loss machinery to train ``θ`` either

    * by a supervised prior (``mode="learned"``) — ``θ`` imitates a simple
      teacher-correctness / confidence target, the classic "learn the
      teacher-correct pseudo-label" baseline, or
    * by a **meta-objective** (``mode="meta"``) — ``θ`` is updated so that a
      one-step virtual student update weighted by ``rθ`` *reduces supervised
      loss on a held-out meta set*. This is the headline method
      (Meta-Reliability Distillation): the predictor learns which teacher
      signals actually help the student generalise, not merely where the
      teacher is confident.

Conceptually (per the study design):

    z       = [confidence, entropy_penalty, teacher_gt_agreement,
               student_confidence, teacher_student_disagreement, ...]   # [B, K, H, W]
    r_kd    = sigmoid(MLP(z))            # scalar reliability  ([B, H, W])  OR
    [a_gt, a_kd, a_neg] = softmax(MLP(z))  # supervision mixture (mixture mode)

The hand-crafted factors are **not** discarded; they are demoted to *input
features* (the multiplicative composition is removed). The final combination is
decided by ``θ`` via the meta-objective.

Student training loss (φ-update; ``r`` is detached so it acts as a fixed weight
on this step — θ is trained separately):

    L_train(φ) = L_sup(Sφ, y) + λ · r · L_KD(Sφ, T)

θ objective (mode-dependent):

    learned:  min_θ  γ · L_prior(θ) + β · L_sparsity(θ)
    meta:     min_θ  L_sup_meta(Sφ', y_meta) + γ · L_prior(θ) + β · L_sparsity(θ)
              with  φ' = φ - η ∇φ [L_sup_train + λ rθ L_KD_train]   (one step)

The bilevel optimisation itself (virtual update + meta gradient) lives in
``trainers.meta_distill_trainer.MetaDistillTrainer``; this module is the
loss/feature/predictor layer that trainer drives.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from distillers.unified_distiller import UnifiedDistiller
from utils.criterion import ReliabilityWeightedKDLoss
from utils.reliability_kd import (
    _foreground_prob,
    _infer_is_binary,
    _normalized_entropy_from_probs,
    boundary_uncertainty_from_probs,
    compute_disagreement_uncertainty,
    pixel_confidence_from_probs,
    probs_from_logits,
)

logger = logging.getLogger(__name__)


# Canonical per-pixel signal set fed to the reliability predictor.
#
# Each name maps to a function (teacher_probs, student_probs, gt) -> [B, H, W]
# in [0, 1]. Keeping the registry here (single source of truth) means the
# predictor's input width is fully determined by the configured `feature_names`,
# and the analysis tooling can request the exact same stack.
DEFAULT_FEATURE_NAMES: Tuple[str, ...] = (
    "teacher_confidence",
    "teacher_gt_agreement",
    "teacher_student_disagreement",
    "student_confidence",
)

ALL_FEATURE_NAMES: Tuple[str, ...] = (
    "teacher_confidence",
    "teacher_entropy",
    "teacher_margin",
    "teacher_gt_agreement",
    "student_confidence",
    "student_entropy",
    "teacher_student_disagreement",
    "boundary_score",
)


def _gt_class_prob(probs: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Soft teacher↔GT agreement: probability mass on the GT class.

    A smooth replacement for the hard ``teacher_pred == gt`` gate that keeps
    gradient-friendly information: ``1`` means the teacher puts all mass on the
    correct class, ``0`` means none. Binary GT uses ``p`` where ``gt==1`` and
    ``1-p`` where ``gt==0``.
    """
    if _infer_is_binary(probs):
        p = _foreground_prob(probs)
        gt_f = gt.to(p.dtype)
        return p * gt_f + (1.0 - p) * (1.0 - gt_f)
    gt_idx = gt.long().clamp(min=0, max=probs.shape[1] - 1).unsqueeze(1)
    return probs.gather(1, gt_idx).squeeze(1)


def build_reliability_features(
    teacher_logits: torch.Tensor,
    student_logits: Optional[torch.Tensor],
    gt: torch.Tensor,
    feature_names: Tuple[str, ...] | List[str] = DEFAULT_FEATURE_NAMES,
    temperature: float = 1.0,
    boundary_kernel_size: int = 3,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, List[str]]:
    """Assemble the per-pixel feature stack ``z`` for the reliability predictor.

    Reuses the existing ``utils.reliability_kd`` signal helpers so the math is
    single-sourced with the hand-crafted map. Returns a ``[B, K, H, W]`` tensor
    (``K = len(feature_names)``) of features in ``[0, 1]`` plus the resolved
    name list (the channel order).

    All features are computed under ``no_grad`` w.r.t. the teacher; the student
    features are detached too — the predictor consumes *signals*, it must not
    backprop into the student through its own inputs (that path is the KD loss).

    Args:
        teacher_logits: Teacher logits ``[B, C, H, W]`` (or binary 1-ch / 3-D).
        student_logits: Student logits, same shape. Required if any
            student-dependent feature is requested.
        gt: GT mask ``[B, H, W]`` (class indices; binary uses 0/1).
        feature_names: Ordered subset of ``ALL_FEATURE_NAMES``.
        temperature: Softening temperature for the logit→prob conversion.
        boundary_kernel_size: Window for the boundary-score feature.
        eps: Numerical stability constant.

    Returns:
        ``(z, names)`` where ``z`` is ``[B, K, H, W]`` and ``names`` lists the
        channels in order.
    """
    names = list(feature_names)
    with torch.no_grad():
        t_probs = probs_from_logits(teacher_logits.detach(), temperature=temperature)
        s_probs = (
            probs_from_logits(student_logits.detach(), temperature=temperature)
            if student_logits is not None
            else None
        )

        # Disagreement needs an explicit channel dim on a shared simplex; reuse
        # the [bg, fg] expansion convention for the binary case.
        def _two_class(probs: torch.Tensor) -> torch.Tensor:
            if not _infer_is_binary(probs):
                return probs
            p = _foreground_prob(probs).unsqueeze(1)
            return torch.cat([1.0 - p, p], dim=1)

        channels: List[torch.Tensor] = []
        for name in names:
            if name == "teacher_confidence":
                f = pixel_confidence_from_probs(t_probs, mode="max_prob", eps=eps)
            elif name == "teacher_entropy":
                f = _normalized_entropy_from_probs(t_probs, eps=eps)
            elif name == "teacher_margin":
                f = pixel_confidence_from_probs(t_probs, mode="margin", eps=eps)
            elif name == "teacher_gt_agreement":
                f = _gt_class_prob(t_probs, gt)
            elif name == "student_confidence":
                _require_student(s_probs, name)
                f = pixel_confidence_from_probs(s_probs, mode="max_prob", eps=eps)
            elif name == "student_entropy":
                _require_student(s_probs, name)
                f = _normalized_entropy_from_probs(s_probs, eps=eps)
            elif name == "teacher_student_disagreement":
                _require_student(s_probs, name)
                f = compute_disagreement_uncertainty(
                    _two_class(t_probs), _two_class(s_probs), mode="l1", eps=eps
                ).clamp(0.0, 1.0)
            elif name == "boundary_score":
                fg = _foreground_prob(t_probs) if _infer_is_binary(t_probs) else t_probs[:, 1:].sum(dim=1)
                f = boundary_uncertainty_from_probs(fg, kernel_size=boundary_kernel_size)
            else:
                raise ValueError(f"Unknown reliability feature: {name!r}")
            channels.append(f)

        z = torch.stack(channels, dim=1)  # [B, K, H, W]
    return z, names


def _require_student(s_probs: Optional[torch.Tensor], name: str) -> None:
    if s_probs is None:
        raise ValueError(
            f"feature {name!r} needs student_logits; pass it to build_reliability_features."
        )


def simple_reliability_target(
    teacher_logits: torch.Tensor,
    gt: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """The warm-up / prior target ``r_simple = teacher_correct * teacher_confidence``.

    Used by the prior loss (and the warm-up schedule) as the initial inductive
    bias the predictor imitates before the meta-objective takes over. This is a
    *soft* version: ``teacher_gt_agreement * teacher_confidence`` in ``[0, 1]``.
    """
    with torch.no_grad():
        t_probs = probs_from_logits(teacher_logits.detach(), temperature=temperature)
        conf = pixel_confidence_from_probs(t_probs, mode="max_prob", eps=eps)
        agree = _gt_class_prob(t_probs, gt)
    return (conf * agree).clamp(0.0, 1.0)


class ReliabilityPredictor(nn.Module):
    """Per-pixel reliability predictor ``Rθ`` (a 1×1-conv MLP).

    Operates pointwise on a ``[B, K, H, W]`` feature stack. Two output modes:

        * ``"scalar"``  → ``r = sigmoid(MLP(z))`` shape ``[B, H, W]`` in ``[0, 1]``;
          a single KD reliability weight.
        * ``"mixture"`` → ``[a_gt, a_kd, a_neg] = softmax(MLP(z))`` shape
          ``[B, 3, H, W]``; a per-pixel supervision mixture (GT / positive-KD /
          negative-KD), the stronger extension that chooses *which* supervision
          to apply rather than only whether to trust the teacher.

    The scalar head is bias-initialised so the initial reliability sits near
    ``init_reliability`` (default 0.5) — a neutral start that the prior / meta
    loss then shapes, avoiding a random map that wrecks early student training.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        output_mode: str = "scalar",
        init_reliability: float = 0.5,
        dropout: float = 0.0,
    ):
        super().__init__()
        if output_mode not in ("scalar", "mixture"):
            raise ValueError(f"output_mode must be 'scalar' or 'mixture', got {output_mode!r}")
        self.output_mode = output_mode
        out_channels = 1 if output_mode == "scalar" else 3

        layers: List[nn.Module] = []
        c = in_channels
        for _ in range(max(1, num_layers) - 1):
            layers += [nn.Conv2d(c, hidden_dim, kernel_size=1), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            c = hidden_dim
        self.head = nn.Conv2d(c, out_channels, kernel_size=1)
        self.body = nn.Sequential(*layers) if layers else nn.Identity()

        # Initialise the scalar head so sigmoid(bias) ≈ init_reliability.
        if output_mode == "scalar":
            p = min(max(init_reliability, 1e-3), 1 - 1e-3)
            nn.init.zeros_(self.head.weight)
            nn.init.constant_(self.head.bias, float(torch.logit(torch.tensor(p))))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map a feature stack ``[B, K, H, W]`` to reliability / mixture.

        Returns ``[B, H, W]`` (scalar mode) or ``[B, 3, H, W]`` (mixture mode).
        """
        h = self.body(z)
        out = self.head(h)
        if self.output_mode == "scalar":
            return torch.sigmoid(out).squeeze(1)
        return F.softmax(out, dim=1)


# Per-pixel supervision losses (shared by scalar + mixture paths)
def per_pixel_gt_loss(
    student_logits: torch.Tensor,
    gt: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-pixel GT supervision loss map ``[B, H, W]`` (CE / BCE, unreduced)."""
    if student_logits.dim() == 4 and student_logits.shape[1] > 1:
        return F.cross_entropy(student_logits, gt.long(), reduction="none")
    s = student_logits[:, 0] if student_logits.dim() == 4 else student_logits
    return F.binary_cross_entropy_with_logits(s, gt.to(s.dtype), reduction="none")


def per_pixel_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-pixel KD loss map ``[B, H, W]`` (KL / BCE-to-teacher), ``T^2`` scaled.

    Mirrors ``ReliabilityWeightedKDLoss`` so the mixture path shares the exact
    KD definition with the scalar path.
    """
    T = temperature
    if student_logits.dim() == 4 and student_logits.shape[1] > 1:
        t = F.softmax(teacher_logits / T, dim=1).detach()
        s = F.log_softmax(student_logits / T, dim=1)
        kd = F.kl_div(s, t, reduction="none").sum(dim=1)
    else:
        s = student_logits[:, 0] if student_logits.dim() == 4 else student_logits
        t = torch.sigmoid(teacher_logits / T).detach()
        t = (t[:, 0] if t.dim() == 4 else t).clamp(eps, 1.0 - eps)
        kd = F.binary_cross_entropy_with_logits(s / T, t, reduction="none")
    return kd * (T ** 2)


def per_pixel_negative_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    gt: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-pixel *negative*-KD map: penalise following a confidently-wrong teacher.

    Where the teacher's hard prediction disagrees with GT, push the student's
    probability on the teacher's (wrong) class **down**:
    ``L_neg = -log(1 - p_student[wrong_class])``. Where the teacher is correct
    the map is ``0`` (the mixture's ``a_neg`` should learn to stay low there,
    but we zero it explicitly so a stray weight cannot inject noise).
    """
    if student_logits.dim() == 4 and student_logits.shape[1] > 1:
        s_probs = F.softmax(student_logits, dim=1)
        t_pred = teacher_logits.argmax(dim=1)
        wrong = (t_pred != gt.long())
        p_wrong = s_probs.gather(1, t_pred.unsqueeze(1)).squeeze(1)
    else:
        s = student_logits[:, 0] if student_logits.dim() == 4 else student_logits
        s_fg = torch.sigmoid(s)
        t = teacher_logits[:, 0] if teacher_logits.dim() == 4 else teacher_logits
        t_pred = (t >= 0).long()
        wrong = (t_pred != gt.long())
        # probability the student assigns to the teacher's predicted (wrong) class
        p_wrong = torch.where(t_pred.bool(), s_fg, 1.0 - s_fg)
    neg = -torch.log((1.0 - p_wrong).clamp(eps, 1.0))
    return neg * wrong.to(neg.dtype)


class MetaReliabilityDistiller(UnifiedDistiller):
    """UnifiedDistiller variant whose reliability map is a learnable ``Rθ``.

    Drop-in for ``UnifiedDistiller`` (same loss-weight surface) that replaces
    the multiplicative ``build_reliability_map`` with the
    :class:`ReliabilityPredictor`. The φ-update path is unchanged in spirit —
    ``L = w_task·L_task + w_reliability_kd · r · L_KD`` — but ``r`` is produced
    by ``Rθ`` and **detached** on the φ step (θ is trained separately, either by
    a supervised prior or by the meta-objective driven from the trainer).

    Config lives under ``cfg.method.meta_reliability``; see
    ``config/method/meta_reliability.yaml`` for the documented surface.
    """

    def __init__(self, cfg: Any, **kwargs):
        super().__init__(cfg, **kwargs)
        m = cfg.method
        mr = m.get("meta_reliability", {}) or {}

        self.mr_mode = str(mr.get("mode", "meta"))  # "meta" | "learned" | "fixed"
        if self.mr_mode not in ("meta", "learned", "fixed"):
            raise ValueError(f"meta_reliability.mode must be meta/learned/fixed, got {self.mr_mode!r}")

        self.mr_output_mode = str(mr.get("output_mode", "scalar"))
        self.mr_feature_names = tuple(mr.get("feature_names", DEFAULT_FEATURE_NAMES))
        self.mr_temperature = float(mr.get("temperature", self.temperature))
        self.mr_boundary_kernel = int(mr.get("boundary_kernel_size", 3))
        self.mr_eps = float(mr.get("eps", 1e-8))

        # θ-objective weights.
        self.mr_lambda = self.w_reliability_kd  # λ in the doc (KD weight)
        self.mr_prior_weight = float(mr.get("prior_weight", 0.1))
        self.mr_sparsity_weight = float(mr.get("sparsity_weight", 0.1))
        self.mr_target_density = float(mr.get("target_density", 0.5))  # ρ
        self.mr_reliability_floor = float(mr.get("reliability_floor", 0.0))  # lower bound

        # Staged schedule (epoch-aware; set via set_epoch()).
        self.mr_warmup_epochs = int(mr.get("warmup_epochs", 0))
        self.mr_prior_anneal_epochs = int(mr.get("prior_anneal_epochs", 0))
        self._current_epoch = 0
        self._total_epochs = int(cfg.training.get("num_epochs", 1)) if "training" in cfg else 1

        # The predictor itself. in_channels is fixed by the feature set.
        self.reliability_predictor = ReliabilityPredictor(
            in_channels=len(self.mr_feature_names),
            hidden_dim=int(mr.get("hidden_dim", 32)),
            num_layers=int(mr.get("num_layers", 2)),
            output_mode=self.mr_output_mode,
            init_reliability=float(mr.get("init_reliability", 0.5)),
            dropout=float(mr.get("dropout", 0.0)),
        )

        # The KD loss the scalar reliability reweights (reuse the existing one).
        if self.reliability_kd_fn is None:
            self.reliability_kd_fn = ReliabilityWeightedKDLoss(
                temperature=self.mr_temperature, eps=self.mr_eps
            )
        # Make build_reliability_map unused on this path; we compute r via Rθ.
        self._build_reliability_map = None
        logger.info(
            "MetaReliabilityDistiller: mode=%s output=%s features=%s λ=%.3f "
            "prior_w=%.3f sparsity_w=%.3f ρ=%.2f",
            self.mr_mode, self.mr_output_mode, self.mr_feature_names,
            self.mr_lambda, self.mr_prior_weight, self.mr_sparsity_weight,
            self.mr_target_density,
        )

    # --- schedule -----------------------------------------------------------
    def set_epoch(self, epoch: int, total_epochs: Optional[int] = None) -> None:
        """Trainer hook: update epoch for warm-up / prior annealing."""
        self._current_epoch = int(epoch)
        if total_epochs is not None:
            self._total_epochs = int(total_epochs)

    @property
    def in_warmup(self) -> bool:
        return self._current_epoch < self.mr_warmup_epochs

    @property
    def prior_weight_now(self) -> float:
        """Prior weight with linear anneal ``γ↓`` over ``prior_anneal_epochs``."""
        if self.mr_prior_anneal_epochs <= 0:
            return self.mr_prior_weight
        e = max(0, self._current_epoch - self.mr_warmup_epochs)
        frac = min(1.0, e / float(self.mr_prior_anneal_epochs))
        return self.mr_prior_weight * (1.0 - frac)

    # --- reliability / mixture computation ---------------------------------
    def compute_reliability(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run ``Rθ`` on the assembled features. Gradient flows to θ.

        During warm-up the scalar reliability is replaced by the simple
        teacher-correct×confidence target (predictor not yet trusted for
        weighting); the predictor is still pre-trained by the prior loss.

        Returns ``(r, info)`` where ``r`` is ``[B, H, W]`` (scalar) or
        ``[B, 3, H, W]`` (mixture) and ``info`` carries the feature stack and
        simple target for diagnostics / prior.
        """
        z, names = build_reliability_features(
            teacher_logits,
            student_logits,
            gt,
            feature_names=self.mr_feature_names,
            temperature=self.mr_temperature,
            boundary_kernel_size=self.mr_boundary_kernel,
            eps=self.mr_eps,
        )
        r = self.reliability_predictor(z)

        info: Dict[str, torch.Tensor] = {"features": z, "feature_names": names}
        if self.mr_output_mode == "scalar":
            if self.mr_reliability_floor > 0:
                r = self.mr_reliability_floor + (1.0 - self.mr_reliability_floor) * r
            if self.in_warmup:
                simple = simple_reliability_target(
                    teacher_logits, gt, temperature=self.mr_temperature, eps=self.mr_eps
                )
                info["r_predicted"] = r
                r = simple
        return r, info

    def reliability_regularization(
        self,
        r: torch.Tensor,
        teacher_logits: torch.Tensor,
        gt: torch.Tensor,
        info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """θ regularisers: sparsity (mean→ρ) + prior (BCE to simple target).

        Returns ``(reg_loss, diagnostics)``. ``reg_loss`` already folds in the
        current (possibly annealed) prior weight and the sparsity weight; the
        meta loss / learned-mode loss adds it directly.
        """
        # For mixture mode, sparsity/prior act on the KD-supervision share.
        r_scalar = r if r.dim() == 3 else r[:, 1]  # a_kd channel
        r_used = info.get("r_predicted", r_scalar) if info else r_scalar

        sparsity = (r_used.mean() - self.mr_target_density).abs()

        simple = simple_reliability_target(
            teacher_logits, gt, temperature=self.mr_temperature, eps=self.mr_eps
        )
        prior = F.binary_cross_entropy(
            r_used.clamp(self.mr_eps, 1.0 - self.mr_eps), simple.clamp(0.0, 1.0)
        )

        gamma = self.prior_weight_now
        reg = self.mr_sparsity_weight * sparsity + gamma * prior
        diag = {
            "meta/sparsity": sparsity.item(),
            "meta/prior_bce": prior.item(),
            "meta/prior_weight": gamma,
            "meta/mean_reliability": r_used.mean().item(),
        }
        return reg, diag

    def reliability_kd_term(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        gt: torch.Tensor,
        r: torch.Tensor,
        detach_reliability: bool = True,
    ) -> torch.Tensor:
        """The KD term weighted by ``r`` (scalar) or the mixture supervision.

        ``detach_reliability`` controls whether ``r`` carries gradient to θ:
        the φ-update detaches it (θ is fixed on that step); the meta virtual
        update keeps it attached so the meta gradient can flow.
        """
        if self.mr_output_mode == "mixture":
            return self._mixture_supervision(student_logits, teacher_logits, gt, r, detach_reliability)
        rel = r.detach() if detach_reliability else r
        return self.reliability_kd_fn(
            student_logits, teacher_logits=teacher_logits, reliability=rel
        )

    def _mixture_supervision(self, student_logits, teacher_logits, gt, mixture, detach):
        """Per-pixel ``a_gt·L_gt + a_kd·L_kd + a_neg·L_neg`` mean."""
        a = mixture.detach() if detach else mixture
        a_gt, a_kd, a_neg = a[:, 0], a[:, 1], a[:, 2]
        l_gt = per_pixel_gt_loss(student_logits, gt, eps=self.mr_eps)
        l_kd = per_pixel_kd_loss(
            student_logits, teacher_logits, temperature=self.mr_temperature, eps=self.mr_eps
        )
        l_neg = per_pixel_negative_kd_loss(student_logits, teacher_logits, gt, eps=self.mr_eps)
        return (a_gt * l_gt + a_kd * l_kd + a_neg * l_neg).mean()

    # --- forward (φ-update loss) -------------------------------------------
    def _compute_reliability_kd_loss(self, student_logits, teacher_logits, targets):
        """Override: build r via Rθ and weight KD by it (r detached on φ step).

        Keeps the same return contract as ``UnifiedDistiller`` so the rest of
        ``forward`` is unchanged. In ``learned`` mode the θ regularisers are
        folded into the returned loss (θ rides the main optimiser); in ``meta``
        mode they are *not* (the trainer owns the θ update), and in ``fixed``
        mode θ never updates.
        """
        gt = targets
        if gt.dim() == 4:
            gt = gt.argmax(dim=1) if gt.shape[1] > 1 else gt[:, 0]
        gt = gt.long()

        r, info = self.compute_reliability(student_logits, teacher_logits, gt)
        kd_loss = self.reliability_kd_term(
            student_logits, teacher_logits, gt, r, detach_reliability=True
        )

        diagnostics = {
            "reliability_kd_loss_weighted": (self.w_reliability_kd * kd_loss).item(),
        }
        r_scalar = r if r.dim() == 3 else r[:, 1]
        diagnostics["mean_reliability"] = (
            info.get("r_predicted", r_scalar) if info else r_scalar
        ).mean().item()

        loss = kd_loss
        if self.mr_mode == "learned":
            # θ is trained by the prior/sparsity here (no meta update).
            reg, reg_diag = self.reliability_regularization(r, teacher_logits, gt, info)
            loss = kd_loss + reg
            diagnostics.update(reg_diag)

        return loss, diagnostics
