"""Loss functions for Seg-TransMiter-lite.

Composition rules
-----------------
* Supervised: re-use ``utils.criterion.TaskLoss`` (Dice + CE/BCE) — see
  :func:`dice_ce_loss` / :func:`bce_dice_loss` below for thin wrappers.
* SAM-prior objectness: BCE + Dice against the *binary lesion mask* the
  SAM teacher proposes from a GT bounding box.  Always gated by SAM's
  predicted IoU/confidence — see :func:`confidence_gate`.
* Boundary: Sobel edge map of the predicted lesion probability vs. the
  SAM teacher mask — MSE in edge space.  Sharpens lesion contours.
* Feature alignment: cosine distance between the projected student feature
  and the SAM image embedding (or vice-versa for the reverse direction).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.criterion import TaskLoss


# --------------------------------------------------------------------- #
# Supervised losses                                                     #
# --------------------------------------------------------------------- #

def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice on multiclass logits.

    Args:
        logits: ``[B, C, H, W]`` raw class logits (``C >= 2``).
        target: ``[B, C, H, W]`` float one-hot or ``[B, H, W]`` long indices.

    Returns mean Dice loss over classes (1 - mean dice).
    """
    if target.dim() == 3:
        # [B, H, W] long -> one-hot [B, C, H, W]
        n_cls = logits.shape[1]
        target = F.one_hot(target.long().clamp(0, n_cls - 1), n_cls).permute(0, 3, 1, 2).float()
    probs = F.softmax(logits, dim=1)
    dims = (0, 2, 3)
    inter = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def dice_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 3,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> torch.Tensor:
    """Cross-entropy + Dice supervised loss for multiclass segmentation.

    Delegates to the project's :class:`utils.criterion.TaskLoss` so we share
    the exact same averaging convention as the rest of the codebase.

    Args:
        logits: ``[B, C, H, W]``.
        target: ``[B, C, H, W]`` one-hot float.
    """
    # TaskLoss averages CE and Dice; we honor the weights by composing manually
    # to keep them tunable from outside.
    if num_classes == 1:
        # Fallback: use bce_dice for binary
        return bce_dice_loss(torch.sigmoid(logits), target, ce_weight=ce_weight, dice_weight=dice_weight)

    ce_fn = nn.CrossEntropyLoss()
    if target.dim() == 4:
        target_idx = target.argmax(dim=1).long()
    else:
        target_idx = target.long()
    ce = ce_fn(logits, target_idx)
    d = dice_loss(logits, target)
    return ce_weight * ce + dice_weight * d


def bce_dice_loss(
    prob: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> torch.Tensor:
    """BCE + Dice on a single-channel probability map.

    Args:
        prob:   ``[B, 1, H, W]`` or ``[B, H, W]`` probabilities in ``[0, 1]``.
        target: same shape, binary 0/1.
    """
    if prob.dim() == 3:
        prob = prob.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    target = target.float()

    bce = F.binary_cross_entropy(prob.clamp(eps, 1.0 - eps), target)

    dims = (0, 2, 3)
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    d_loss = 1.0 - dice.mean()
    return ce_weight * bce + dice_weight * d_loss


# --------------------------------------------------------------------- #
# Boundary loss (Sobel edges)                                           #
# --------------------------------------------------------------------- #


def _sobel_kernels(device, dtype):
    kx = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], device=device, dtype=dtype)
    ky = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], device=device, dtype=dtype)
    return kx, ky


def _sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """Compute Sobel gradient magnitude for a single-channel map.

    Args:
        x: ``[B, 1, H, W]`` in ``[0, 1]``.
    Returns:
        ``[B, 1, H, W]`` edge magnitude.
    """
    kx, ky = _sobel_kernels(x.device, x.dtype)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def boundary_loss(
    pred_prob: torch.Tensor,
    teacher_mask: torch.Tensor,
    mode: str = "mse",
) -> torch.Tensor:
    """Sobel edge loss between predicted probability and teacher mask.

    Both inputs should be in ``[0, 1]``.  ``teacher_mask`` is typically the
    SAM teacher's binary lesion mask cast to float.

    Args:
        pred_prob:    ``[B, 1, H, W]`` predicted lesion probability.
        teacher_mask: ``[B, 1, H, W]`` binary mask from SAM teacher.
        mode:         ``"mse"`` (L2 on edge magnitudes) or ``"bce"``.
    """
    if pred_prob.dim() == 3:
        pred_prob = pred_prob.unsqueeze(1)
    if teacher_mask.dim() == 3:
        teacher_mask = teacher_mask.unsqueeze(1)
    teacher_mask = teacher_mask.to(pred_prob.dtype)

    if pred_prob.shape[-2:] != teacher_mask.shape[-2:]:
        teacher_mask = F.interpolate(
            teacher_mask, size=pred_prob.shape[-2:], mode="nearest"
        )

    pred_edge = _sobel_edges(pred_prob)
    target_edge = _sobel_edges(teacher_mask)

    # Normalize each map to [0, 1] for a stable signal.
    pred_edge = pred_edge / (pred_edge.amax(dim=(2, 3), keepdim=True) + 1e-6)
    target_edge = target_edge / (target_edge.amax(dim=(2, 3), keepdim=True) + 1e-6)

    if mode == "bce":
        return F.binary_cross_entropy(pred_edge.clamp(1e-6, 1 - 1e-6), target_edge)
    return F.mse_loss(pred_edge, target_edge)


# --------------------------------------------------------------------- #
# Feature alignment                                                     #
# --------------------------------------------------------------------- #


def feature_align_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    mode: str = "cosine",
) -> torch.Tensor:
    """Align a projected student feature to a teacher feature map.

    Args:
        student_feat: ``[B, C, H, W]``.
        teacher_feat: ``[B, C, H, W]`` (must match channels; resize handled).
        mode: ``"cosine"`` (1 - mean cosine similarity over pixels) or
              ``"mse"`` (mean L2 on L2-normalized features).
    """
    if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
        teacher_feat = F.interpolate(
            teacher_feat, size=student_feat.shape[-2:], mode="bilinear", align_corners=False
        )
    s = F.normalize(student_feat, dim=1)
    t = F.normalize(teacher_feat, dim=1)
    if mode == "mse":
        return F.mse_loss(s, t)
    # cosine: per-pixel cosine similarity averaged
    cos = (s * t).sum(dim=1)  # [B, H, W]
    return (1.0 - cos).mean()


# --------------------------------------------------------------------- #
# Confidence gate                                                       #
# --------------------------------------------------------------------- #


def confidence_gate(
    sam_score: torch.Tensor,
    threshold: float = 0.75,
    soft: bool = False,
    sharpness: float = 20.0,
) -> torch.Tensor:
    """Per-sample SAM-teacher confidence gate.

    Args:
        sam_score:  ``[B]`` predicted IoU/objectness from SAM mask decoder.
        threshold:  Hard threshold above which the SAM mask is trusted.
        soft:       If True, use a sigmoid soft gate around the threshold
                    instead of a hard step.
        sharpness:  Slope of the soft gate.

    Returns:
        Tensor of shape ``[B]`` with values in ``[0, 1]`` to be broadcast
        against the per-sample SAM-prior loss terms.
    """
    score = sam_score.float().view(-1)
    if soft:
        return torch.sigmoid(sharpness * (score - threshold))
    return (score >= threshold).to(score.dtype)


def gated_mean(per_sample_loss: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Average per-sample losses with a confidence gate.

    ``per_sample_loss`` is reduced over non-batch dims, then weighted by
    ``gate``.  If every sample is gated out, returns 0.

    Args:
        per_sample_loss: ``[B]`` or ``[B, ...]`` scalar-per-sample loss.
        gate:            ``[B]`` weights in ``[0, 1]``.
    """
    if per_sample_loss.dim() > 1:
        per_sample_loss = per_sample_loss.flatten(1).mean(dim=1)
    w = gate.to(per_sample_loss.dtype)
    s = w.sum()
    if s <= 0:
        return torch.zeros((), device=per_sample_loss.device, dtype=per_sample_loss.dtype)
    return (per_sample_loss * w).sum() / s


def supervised_task_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 3,
) -> torch.Tensor:
    """Project-aligned supervised loss using existing :class:`TaskLoss`.

    Use this when you want the *exact* averaging convention the rest of the
    codebase uses (matches distill_trainer/sam_trainer).
    """
    fn = TaskLoss(num_classes=num_classes, use_ce=True, use_dice=True)
    fn = fn.to(logits.device)
    return fn(logits, target)
