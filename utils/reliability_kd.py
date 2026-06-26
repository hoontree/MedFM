"""Reliability-aware Knowledge Distillation utilities.

This module provides building blocks for a per-pixel **reliability map**
``r in [0, 1]`` used to reweight logit distillation losses. The default
composition is

    r = confidence
        * gt_agreement_gate
        * entropy_penalty

so that pixels where the teacher is uncertain or disagrees with the GT
contribute less to the KD loss. The map is then
broadcast against KL / BCE pixelwise losses (see ``utils/criterion.py``).

Most helpers accept both binary and multi-class segmentation outputs and
dispatch on tensor shape:
    * binary: ``[B, 1, H, W]`` or ``[B, H, W]`` (foreground probability).
    * multi-class: ``[B, C, H, W]`` with ``C > 1`` (softmax probability).

Related entry points (in ``utils/criterion.py``):
    * ``LogitDistillLoss``: plain logit-KD without reliability weighting.
    * ``UncertaintyWeightedKDLoss``: entropy-weighted KD using a linear or
      exponential weight schedule.

Conventions:
    * All ``*_from_probs`` helpers expect already-normalised probabilities.
    * Functions that consume logits accept a ``temperature`` argument and
      perform the sigmoid/softmax internally.
    * Output reliability/gate maps always have shape ``[B, H, W]``.
    * Individual factor functions compute *only* their factor. Cross-cutting
      concerns — shape/logit→prob conversion and ``ignore_index`` masking —
      are owned by ``build_reliability_map``, which applies the ignore mask
      exactly once at the end (the "ignore is a final-map policy" rule).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# Reliability map composition
def build_reliability_map(
    teacher_logits: torch.Tensor | None = None,
    teacher_probs: torch.Tensor | None = None,
    student_logits: torch.Tensor | None = None,
    student_probs: torch.Tensor | None = None,
    gt: torch.Tensor | None = None,
    confidence_mode: str = "max_prob",
    temperature: float = 1.0,
    use_entropy_penalty: bool = True,
    use_teacher_correctness_gate: bool = False,
    teacher_correctness_wrong_weight: float = 0.0,
    teacher_correctness_correct_weight: float = 1.0,
    teacher_correctness_binary_threshold: float = 0.5,
    use_student_bypass: bool = False,
    student_bypass_margin: float = 0.0,
    student_bypass_weight: float = 0.1,
    student_bypass_default_weight: float = 1.0,
    student_bypass_wrong_teacher_weight: float = 0.0,
    student_bypass_binary_threshold: float = 0.5,
    use_reliability_smoothing: bool = False,
    smoothing_kernel_size: int = 5,
    smoothing_sigma: float = 0.5,
    confidence_clip: tuple[float, float] | None = None,
    ignore_index: int | None = None,
    eps: float = 1e-8,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compose the per-pixel reliability map used by reliability-aware KD.

    Combines the reliability factors implemented in this module:

        reliability = confidence
                      * entropy_penalty             (if use_entropy_penalty)
                      * teacher_correctness_gate    (if use_teacher_correctness_gate)
                      * student_bypass_gate         (if use_student_bypass)

    and, when ``use_reliability_smoothing`` is set, the composed map is then
    passed through ``prediction_aware_reliability_smoothing`` before the
    ignore mask is applied.

    This function owns the cross-cutting concerns the individual factors
    deliberately omit:
        * logit → probability conversion (teacher and student), and
        * ``ignore_index`` masking, applied **once** at the very end so that
          "ignore is a final reliability-mask policy" stays a single,
          unambiguous statement.

    ``confidence_mode="entropy"`` and ``use_entropy_penalty=True`` compute
    the identical normalised-entropy factor (see
    ``pixel_confidence_from_probs``); enabling both would square it, so the
    combination raises ``ValueError`` instead of silently double-counting.

    Pixels with ``gt == ignore_index`` are forced to ``0`` and the final
    map is clamped to ``[0, 1]``. Either ``teacher_logits`` or
    ``teacher_probs`` must be provided; if only logits are given, they are
    converted with sigmoid (binary) or softmax (multi-class) using
    ``temperature``.

    The student-bypass factor (``gt_correct_student_bypass_gate``) compares
    student and teacher predictions against the GT to down-weight KD where
    the student is already confidently correct and rescue/disable it
    elsewhere. It needs the student distribution, supplied via
    ``student_probs`` or ``student_logits`` (converted with the same
    ``temperature`` as the teacher); the student tensor must match the
    shape of the teacher distribution used for the gate.

    Binary / multi-class is auto-detected from the channel dimension of
    the teacher tensor:
        * binary: ``[B, 1, H, W]`` or ``[B, H, W]``.
        * multi-class: ``[B, C, H, W]`` with ``C > 1``.

    Args:
        teacher_logits: Raw teacher logits. Required if
            ``teacher_probs`` is not given.
        teacher_probs: Pre-computed teacher probabilities. Takes
            precedence over ``teacher_logits`` when both are provided.
        student_logits: Raw student logits. Used only when
            ``use_student_bypass=True`` and ``student_probs`` is not given.
        student_probs: Pre-computed student probabilities. Takes precedence
            over ``student_logits``. Required (directly or via logits) when
            ``use_student_bypass=True``.
        gt: GT mask ``[B, H, W]``; binary uses ``0/1``, multi-class uses
            class indices. Required when any GT-dependent factor is on.
        confidence_mode: Forwarded to ``pixel_confidence_from_probs``.
        temperature: Softening temperature applied when converting logits
            (teacher and student). Ignored for whichever distribution is
            supplied directly as probabilities.
        use_entropy_penalty: Enable the entropy penalty factor. Must not be
            combined with ``confidence_mode == "entropy"`` (raises
            ``ValueError`` — it would duplicate the factor).
        use_teacher_correctness_gate: Enable the GT-conditioned teacher
            correctness gate. Unlike the confidence factors, this looks at
            ``teacher_pred == gt`` directly and is the only factor that can
            suppress a confidently-wrong teacher pixel. Requires ``gt``.
        teacher_correctness_wrong_weight: ``wrong_weight`` forwarded to
            ``teacher_correctness_gate`` (default ``0.0`` disables KD where
            the teacher is wrong).
        teacher_correctness_correct_weight: ``correct_weight`` forwarded to
            the gate (default ``1.0`` leaves correct pixels untouched).
        teacher_correctness_binary_threshold: ``binary_threshold`` forwarded
            to the gate.
        use_student_bypass: Enable the GT-conditioned student-bypass gate.
        student_bypass_margin: ``margin`` forwarded to
            ``gt_correct_student_bypass_gate``.
        student_bypass_weight: ``bypass_weight`` forwarded to the gate.
        student_bypass_default_weight: ``default_weight`` forwarded to the gate.
        student_bypass_wrong_teacher_weight: ``wrong_teacher_weight``
            forwarded to the gate.
        student_bypass_binary_threshold: ``binary_threshold`` forwarded to
            the gate.
        use_reliability_smoothing: Enable prediction-aware spatial smoothing
            of the composed map (see
            ``prediction_aware_reliability_smoothing``). Applied after all
            factors but before the ignore mask.
        smoothing_kernel_size: Odd local window size for the smoothing.
        smoothing_sigma: Affinity bandwidth for the smoothing; smaller keeps
            edges sharper.
        confidence_clip: Optional ``(low, high)`` clamp applied to the
            raw confidence before composition. Useful for stabilising
            extreme values, e.g. ``(0.05, 0.95)``.
        ignore_index: Label value treated as ignore. Applied once at the end.
        eps: Numerical stability constant forwarded to helpers.
        return_components: If ``True``, also return a dict mapping each
            enabled factor's name to its per-pixel ``[B, H, W]`` map (the
            exact tensors used in the composition, before the final clamp).
            Useful for logging the scale/contribution of each factor.

    Returns:
        Reliability map ``[B, H, W]`` with values in ``[0, 1]``. When
        ``return_components=True``, a ``(reliability, components)`` tuple
        where ``components`` is a ``dict[str, torch.Tensor]`` of the
        individual factor maps (always includes ``"confidence"``; other
        keys present only for the factors that were enabled, plus
        ``"ignore_mask"`` when ``ignore_index`` is set).

    Raises:
        ValueError: If ``use_entropy_penalty`` is combined with
            ``confidence_mode == "entropy"``, neither logits nor probs are
            provided, the teacher tensor has an unsupported shape, a
            GT-dependent factor is requested without supplying ``gt``,
            ``use_student_bypass`` is set without a student distribution, or
            ``ignore_index`` is set without ``gt``.
    """
    # `confidence_mode="entropy"` already *is* the entropy penalty (both
    # delegate to `_normalized_entropy_from_probs`), so enabling both would
    # square the identical factor. Reject the combination explicitly rather
    # than silently dropping one of them.
    if use_entropy_penalty and confidence_mode == "entropy":
        raise ValueError(
            "use_entropy_penalty=True is redundant with confidence_mode='entropy' "
            "(both compute 1 - H(p)/log(C_eff)). Set use_entropy_penalty=False, "
            "or pick a different confidence_mode."
        )

    source = teacher_probs if teacher_probs is not None else teacher_logits
    if source is None:
        raise ValueError("Either teacher_logits or teacher_probs must be provided.")

    # Validates shape and raises on anything but 3-D / 4-D.
    _infer_is_binary(source)

    if teacher_probs is None:
        teacher_probs = probs_from_logits(teacher_logits, temperature=temperature)

    confidence = pixel_confidence_from_probs(
        teacher_probs, mode=confidence_mode, eps=eps,
    )

    if confidence_clip is not None:
        low, high = confidence_clip
        confidence = confidence.clamp(low, high)

    # Each factor's per-pixel map is stashed here as it is computed so the
    # caller can log the scale/contribution of every component without
    # recomputing anything (the math stays single-sourced).
    components: dict[str, torch.Tensor] = {"confidence": confidence}
    reliability = confidence

    # The redundant entropy combination is rejected above, so by here
    # use_entropy_penalty implies a non-entropy confidence_mode.
    if use_entropy_penalty:
        entropy_factor = entropy_penalty_from_probs(teacher_probs, eps=eps)
        components["entropy_penalty"] = entropy_factor
        reliability = reliability * entropy_factor

    if use_teacher_correctness_gate:
        if gt is None:
            raise ValueError("gt is required when use_teacher_correctness_gate=True.")
        teacher_gate = teacher_correctness_gate(
            teacher_probs=teacher_probs,
            gt=gt,
            wrong_weight=teacher_correctness_wrong_weight,
            correct_weight=teacher_correctness_correct_weight,
            binary_threshold=teacher_correctness_binary_threshold,
        )
        components["teacher_correctness_gate"] = teacher_gate
        reliability = reliability * teacher_gate

    if use_student_bypass:
        if gt is None:
            raise ValueError("gt is required when use_student_bypass=True.")
        if student_probs is None:
            if student_logits is None:
                raise ValueError(
                    "student_probs or student_logits is required when use_student_bypass=True."
                )
            student_probs = probs_from_logits(student_logits, temperature=temperature)
        student_gate = gt_correct_student_bypass_gate(
            student_probs=student_probs,
            teacher_probs=teacher_probs,
            gt=gt,
            margin=student_bypass_margin,
            bypass_weight=student_bypass_weight,
            default_weight=student_bypass_default_weight,
            wrong_teacher_weight=student_bypass_wrong_teacher_weight,
            binary_threshold=student_bypass_binary_threshold,
            eps=eps,
        )
        components["student_bypass_gate"] = student_gate
        reliability = reliability * student_gate

    # Prediction-aware spatial smoothing of the composed map. Runs after all
    # multiplicative factors (so it smooths the *final* reliability) but before
    # the ignore mask, keeping "ignore is a final-map policy" intact.
    if use_reliability_smoothing:
        reliability = prediction_aware_reliability_smoothing(
            reliability,
            teacher_probs=_as_multiclass_probs(teacher_probs),
            kernel_size=smoothing_kernel_size,
            sigma=smoothing_sigma,
            eps=eps,
        )
        components["reliability_smoothed"] = reliability

    # Ignore masking is a final-map policy: apply it exactly once, here.
    if ignore_index is not None:
        if gt is None:
            raise ValueError("gt is required when ignore_index is set.")
        ignore_mask = (gt != ignore_index).to(reliability.dtype)
        components["ignore_mask"] = ignore_mask
        reliability = reliability * ignore_mask

    reliability = reliability.clamp(0.0, 1.0)
    if return_components:
        return reliability, components
    return reliability


# Shape / task dispatch helpers
#
# Every public helper used to re-derive "binary vs multi-class" and reshape
# its inputs. These helpers centralise that logic so factor functions can stay
# focused on the actual math.


def _infer_is_binary(x: torch.Tensor) -> bool:
    """Return ``True`` for a binary segmentation tensor, ``False`` otherwise.

    Dispatches on shape: ``[B, H, W]`` and ``[B, 1, H, W]`` are binary,
    ``[B, C, H, W]`` with ``C > 1`` is multi-class.

    Raises:
        ValueError: If ``x`` is neither 3-D nor 4-D.
    """
    if x.dim() == 3:
        return True
    if x.dim() == 4:
        return x.shape[1] == 1
    raise ValueError("tensor must have shape [B, H, W] or [B, C, H, W].")


# Pixel-wise confidence
def pixel_confidence_from_probs(
    probs: torch.Tensor,
    mode: str = "max_prob",
    foreground_class: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Pixel-wise teacher confidence from a probability map.

    Dispatches on the shape of ``probs``:
        * binary: ``[B, 1, H, W]`` or ``[B, H, W]`` (foreground prob).
        * multi-class: ``[B, C, H, W]`` with ``C > 1`` (softmax prob).

    Supported ``mode`` definitions (returning ``[B, H, W]`` in ``[0, 1]``):
        * ``"max_prob"``: multi-class ``max_c p_c``; binary ``max(p, 1-p)``.
        * ``"foreground"``: multi-class ``p[foreground_class]``;
          binary returns ``p`` itself.
        * ``"entropy"``: ``1 - H(p) / log(C_eff)`` (normalised entropy
          inverted into a confidence). ``C_eff = 2`` for binary, ``C``
          otherwise. Shares its definition with
          ``entropy_penalty_from_probs`` via the same internal helper, so
          combining ``confidence_mode="entropy"`` with the entropy penalty
          would multiply the *same* factor twice (``build_reliability_map``
          guards against this).
        * ``"margin"``: multi-class ``p_top1 - p_top2``;
          binary ``|p - 0.5| * 2``.

    Args:
        probs: Probability map, see shape rules above. Probabilities are
            assumed to already be normalised (sigmoid / softmax output).
        mode: One of ``"max_prob"``, ``"foreground"``, ``"entropy"``,
            ``"margin"``.
        foreground_class: Class index used by the multi-class
            ``"foreground"`` mode. Ignored for binary inputs.
        eps: Stability clamp applied before ``log`` in the binary entropy
            mode and to ``log`` arguments in general.

    Returns:
        Confidence map of shape ``[B, H, W]`` with values in ``[0, 1]``.

    Raises:
        ValueError: If ``probs`` has an unsupported shape, ``mode`` is
            unknown, or ``foreground_class`` is out of range.
    """
    if _infer_is_binary(probs):
        if mode == "entropy":
            return (1.0 - _normalized_entropy_from_probs(probs, eps=eps)).clamp(0.0, 1.0)

        p = _foreground_prob(probs).clamp(eps, 1.0 - eps)
        if mode == "max_prob":
            confidence = torch.maximum(p, 1.0 - p)
        elif mode == "foreground":
            confidence = p
        elif mode == "margin":
            confidence = ((p - 0.5).abs() * 2.0).clamp(0.0, 1.0)
        else:
            raise ValueError(f"Unknown confidence mode: {mode}")
        return confidence

    c = probs.shape[1]

    if mode == "max_prob":
        confidence = probs.max(dim=1).values

    elif mode == "foreground":
        if not (0 <= foreground_class < c):
            raise ValueError(f"foreground_class must be in [0, {c - 1}].")
        confidence = probs[:, foreground_class]

    elif mode == "entropy":
        confidence = (1.0 - _normalized_entropy_from_probs(probs, eps=eps)).clamp(0.0, 1.0)

    elif mode == "margin":
        if c < 2:
            raise ValueError("margin confidence requires at least 2 classes.")
        top2 = probs.topk(k=2, dim=1).values
        confidence = (top2[:, 0] - top2[:, 1]).clamp(0.0, 1.0)

    else:
        raise ValueError(f"Unknown confidence mode: {mode}")

    return confidence


# Entropy penalty
def entropy_penalty_from_probs(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Confidence factor based on normalised entropy.

    Computes ``1 - H(p) / log(C_eff)``, which is close to ``1`` for
    confident pixels and drops towards ``0`` as the teacher distribution
    approaches uniform. Use as a multiplicative factor in the reliability
    map to suppress high-entropy regions.

    This is numerically identical to ``pixel_confidence_from_probs(probs,
    mode="entropy")`` — both delegate to ``_normalized_entropy_from_probs``.
    Do not stack the two on the same map.

    Dispatches on ``probs`` shape:
        * binary: ``[B, 1, H, W]`` or ``[B, H, W]`` (foreground prob).
          ``C_eff = 2``.
        * multi-class: ``[B, C, H, W]`` with ``C > 1``. ``C_eff = C``.

    Args:
        probs: Already-normalised probability map.
        eps: Stability constant used for log clamping.

    Returns:
        Penalty tensor of shape ``[B, H, W]`` clamped to ``[0, 1]``.

    Raises:
        ValueError: If ``probs`` has an unsupported shape.
    """
    return (1.0 - _normalized_entropy_from_probs(probs, eps=eps)).clamp(0.0, 1.0)


def entropy_penalty_from_logits(
    teacher_logits: torch.Tensor,
    tau: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Entropy penalty computed directly from teacher logits.

    Convenience wrapper around ``pixelwise_entropy`` that softens logits
    by ``tau``, computes the entropy, and returns
    ``1 - H(p) / log(C_eff)``. Equivalent to first converting logits to
    probabilities and calling ``entropy_penalty_from_probs``, but avoids
    materialising the probability tensor outside ``no_grad``. Binary vs.
    multi-class is inferred from ``teacher_logits.shape[1]``.

    Args:
        teacher_logits: Raw teacher logits ``[B, C, H, W]``.
        tau: Softmax temperature for entropy computation.
        eps: Stability constant for ``max_entropy`` clamping.

    Returns:
        Penalty tensor of shape ``[B, H, W]`` in ``[0, 1]``.
    """
    entropy = pixelwise_entropy(teacher_logits, tau=tau, eps=eps)
    c_eff = _effective_num_classes(teacher_logits)
    max_entropy = torch.log(
        torch.tensor(float(c_eff), device=entropy.device, dtype=entropy.dtype)
    ).clamp(min=eps)
    return (1.0 - entropy / max_entropy).clamp(0.0, 1.0)

# Teacher correctness gate
def teacher_correctness_gate(
    teacher_probs: torch.Tensor,
    gt: torch.Tensor,
    wrong_weight: float = 0.0,
    correct_weight: float = 1.0,
    binary_threshold: float = 0.5,
) -> torch.Tensor:
    """Down-weight KD where the teacher's hard prediction disagrees with GT.

    The other factors in this module are GT-agnostic confidence scalers:
    they can only attenuate the teacher's own confidence, never override
    it. That leaves the most damaging case unhandled — a teacher that is
    confidently *wrong*. A high ``max_prob`` on an incorrect pixel keeps
    reliability high and distils the error straight into the student.

    This gate closes that hole by looking directly at ``teacher_pred == gt``:

        * teacher correct  → ``correct_weight`` (default ``1.0``, a no-op).
        * teacher wrong    → ``wrong_weight`` (default ``0.0``, KD disabled).

    Unlike ``gt_correct_student_bypass_gate`` it does not consult the
    student or relative confidence — the decision is purely "is the teacher
    signal trustworthy here?". The two are complementary: this gate kills
    confidently-wrong teacher pixels regardless of student state, while the
    bypass gate decides what to do where the student is more confident.

    This is a pure factor: it does not apply ``ignore_index`` masking. The
    final reliability map masks ignore pixels once in
    ``build_reliability_map``.

    Binary vs multi-class is dispatched from the channel dim of
    ``teacher_probs``:
        * multi-class: prediction ``argmax_c p_c``.
        * binary: prediction ``p >= binary_threshold``.

    Args:
        teacher_probs: Teacher probability map. ``[B, C, H, W]`` for
            multi-class (``C > 1``) or ``[B, 1, H, W]`` / ``[B, H, W]``
            for binary.
        gt: GT mask ``[B, H, W]``; binary uses ``0/1``, multi-class uses
            class indices.
        wrong_weight: Weight assigned where the teacher prediction is
            wrong. ``0.0`` fully disables KD on those pixels; a small
            positive value keeps a damped signal.
        correct_weight: Weight assigned where the teacher prediction is
            correct (default ``1.0`` leaves those pixels untouched).
        binary_threshold: Foreground threshold in the binary branch.

    Returns:
        Gate ``[B, H, W]`` with values in
        ``[min(wrong_weight, correct_weight), max(wrong_weight, correct_weight)]``.
    """
    if _infer_is_binary(teacher_probs):
        t_p = _foreground_prob(teacher_probs)
        teacher_pred = (t_p >= binary_threshold).long()
    else:
        teacher_pred = teacher_probs.argmax(dim=1)

    teacher_correct = teacher_pred == gt.to(teacher_pred.dtype)
    return torch.where(
        teacher_correct,
        torch.full_like(teacher_pred, 0, dtype=teacher_probs.dtype) + correct_weight,
        torch.full_like(teacher_pred, 0, dtype=teacher_probs.dtype) + wrong_weight,
    )

# Consistency weight
def consistency_weight(
    teacher_probs_a: torch.Tensor,
    teacher_probs_b: torch.Tensor,
    mode: str = "l1",
    sharpness: float = 10.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Augmentation-consistency reliability weight.

    Measures pixel-wise disagreement between two teacher views and maps
    it through ``w = exp(-sharpness * disagreement)``, so consistent
    pixels approach ``1`` and inconsistent pixels approach ``0``. The
    disagreement itself is delegated to
    ``compute_disagreement_uncertainty`` so the ``"l1"`` / ``"kl"`` metric
    has a single definition.

    Both inputs must be aligned to the same pixel grid: if the views
    were obtained via flip/resize augmentations, apply the inverse
    transform before passing them here.

    Modes:
        * ``"l1"``: ``mean_c |p_a - p_b|``.
        * ``"kl"``: ``KL(p_a || p_b)``.

    Args:
        teacher_probs_a: View-A probabilities ``[B, C, H, W]``.
        teacher_probs_b: View-B probabilities, same shape as
            ``teacher_probs_a``.
        mode: Disagreement metric (``"l1"`` or ``"kl"``).
        sharpness: Decay rate of the exponential. Higher values penalise
            disagreement more aggressively.
        eps: Lower clamp used by the KL branch.

    Returns:
        Weight tensor ``[B, H, W]`` clamped to ``[0, 1]``.

    Raises:
        ValueError: If shapes differ or ``mode`` is unknown.
    """
    disagreement = compute_disagreement_uncertainty(
        teacher_probs_a, teacher_probs_b, mode=mode, eps=eps
    )
    weight = torch.exp(-sharpness * disagreement)
    return weight.clamp(0.0, 1.0)

def gt_correct_student_bypass_gate(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    gt: torch.Tensor,
    margin: float = 0.0,
    bypass_weight: float = 0.1,
    default_weight: float = 1.0,
    wrong_teacher_weight: float = 0.0,
    binary_threshold: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute a per-pixel KD gate from GT-conditioned student/teacher agreement.

    Returns a multiplicative weight in ``[0, 1]`` for the KD loss at each
    pixel, decided by comparing student and teacher correctness against the
    GT and their relative confidence. A pixel is "student-more-confident"
    when ``student_conf > teacher_conf + margin``. The gate
    defaults to ``default_weight`` and is overridden by the following cases,
    applied in order:

    1. **Bypass** — student-more-confident AND student correct →
       ``bypass_weight``. An already-correct, confident student should not
       be dragged back toward a less-confident teacher, so KD is suppressed.

    2. **Rescue** — student-more-confident AND student wrong AND teacher
       correct → a weight ramped from ``bypass_weight`` up to ``default_weight`` in
       proportion to the confidence gap ``student_conf - teacher_conf``
       (normalised by the batch-max gap). A confidently-wrong student that
       the teacher can correct gets *more* KD the more overconfident it is.

    3. **Both wrong** — neither student nor teacher matches GT → ``wrong_teacher_weight``. KD
       is disabled where the teacher offers no reliable signal.

    Cases applied later override earlier ones where they overlap. All other
    pixels keep the default weight of ``default_weight``.

    This is a pure factor: it does not apply ``ignore_index`` masking. The
    final reliability map masks ignore pixels once in
    ``build_reliability_map``.

    Binary vs multi-class is dispatched from the channel dim of
    ``student_probs`` (which must match ``teacher_probs``).

    Confidence and prediction definition:
        * multi-class: confidence ``max_c p_c``, prediction ``argmax_c p_c``.
        * binary: confidence ``max(p, 1 - p)``, prediction
          ``p >= binary_threshold``.

    Args:
        student_probs: Student probability map. ``[B, C, H, W]`` for
            multi-class (``C > 1``) or ``[B, 1, H, W]`` / ``[B, H, W]``
            for binary.
        teacher_probs: Teacher probabilities, same shape as
            ``student_probs``.
        gt: GT mask ``[B, H, W]``.
        margin: Confidence margin the student must exceed over the
            teacher to count as "more confident"; larger is more conservative.
        bypass_weight: Floor weight used for the bypass case and as the lower
            bound of the rescue ramp (e.g. ``0`` to fully disable KD there).
        default_weight: Default weight for pixels that don't match any special case.
        wrong_teacher_weight: Weight for pixels where both student and teacher are wrong.
        binary_threshold: Foreground threshold in the binary branch.
        eps: Small constant for numerical stability.

    Returns:
        Gate ``[B, H, W]`` with values in ``[0, 1]``.

    Raises:
        ValueError: If ``student_probs`` and ``teacher_probs`` have
            different shapes.
    """
    if student_probs.shape != teacher_probs.shape:
        raise ValueError("student_probs and teacher_probs must have the same shape.")

    if _infer_is_binary(student_probs):
        s_p = _foreground_prob(student_probs)
        t_p = _foreground_prob(teacher_probs)
        # binary confidence: foreground/background 중 더 확신하는 쪽
        student_conf = torch.maximum(s_p, 1.0 - s_p)
        teacher_conf = torch.maximum(t_p, 1.0 - t_p)
        student_pred = (s_p >= binary_threshold).long()
        teacher_pred = (t_p >= binary_threshold).long()
    else:
        student_conf, student_pred = student_probs.max(dim=1)
        teacher_conf, teacher_pred = teacher_probs.max(dim=1)

    gt_cast = gt.to(student_pred.dtype)

    student_more_confident = student_conf > (teacher_conf + margin)
    student_correct = student_pred == gt_cast
    teacher_correct = teacher_pred == gt_cast

    case_bypass = student_more_confident & student_correct
    gate = torch.full_like(student_conf, default_weight)

    gate = torch.where(
        case_bypass,
        torch.full_like(student_conf, bypass_weight),
        gate
    )

    case_student_wrong_teacher_correct = student_more_confident & ~student_correct & teacher_correct
    conf_gap = (student_conf - teacher_conf).clamp(min=0.0)
    rescue_weight = bypass_weight + conf_gap.clamp(0.0, 1.0) * (
        default_weight - bypass_weight
    )
    rescue_weight = rescue_weight.clamp(0.0, default_weight)
    gate = torch.where(
        case_student_wrong_teacher_correct,
        rescue_weight,
        gate
    )

    case_both_wrong = ~student_correct & ~teacher_correct
    gate = torch.where(
        case_both_wrong,
        torch.full_like(student_conf, wrong_teacher_weight),
        gate
    )

    return gate

def _foreground_prob(probs: torch.Tensor) -> torch.Tensor:
    """Collapse a binary probability map to ``[B, H, W]`` foreground prob.

    Accepts ``[B, 1, H, W]`` or ``[B, H, W]`` and returns ``[B, H, W]``.
    """
    if not _infer_is_binary(probs):
        raise ValueError("_foreground_prob expects a binary probability map.")

    return probs[:, 0] if probs.dim() == 4 else probs

def _effective_num_classes(probs: torch.Tensor) -> int:
    """Number of classes on the simplex (``2`` for binary, ``C`` otherwise)."""
    return 2 if _infer_is_binary(probs) else probs.shape[1]


def _as_multiclass_probs(probs: torch.Tensor) -> torch.Tensor:
    """Ensure a probability map carries an explicit channel dim ``[B, C, H, W]``.

    The reliability factors accept binary probs as ``[B, H, W]``, but the
    spatial smoothing helper needs a 4-D tensor. A 3-D binary map is promoted
    to ``[B, 1, H, W]`` (the single foreground channel is enough to measure
    local prediction similarity); 4-D maps pass through unchanged.
    """
    return probs.unsqueeze(1) if probs.dim() == 3 else probs


def _expand_binary_logits(logits: torch.Tensor) -> torch.Tensor:
    """Expand a single-channel logit to a 2-class ``[neg, pos]`` simplex.

    Matches the convention used by ``LogitDistillLoss`` so entropy / margin
    statistics computed here share the same simplex as the KD loss. No-op for
    tensors that are already multi-class.
    """
    if not _infer_is_binary(logits):
        return logits
    if logits.dim() == 3:
        logits = logits.unsqueeze(1)  # [B, 1, H, W]
        
    return torch.cat([torch.zeros_like(logits), logits], dim=1)  # [B, 2, H, W]


def _normalized_entropy_from_probs(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-pixel entropy ``H(p)`` normalised to ``[0, 1]`` by ``log(C_eff)``.

    Single source of truth for the entropy-based statistics: both the
    ``"entropy"`` confidence mode and ``entropy_penalty_from_probs`` derive
    their value as ``1 - H(p)/log(C_eff)`` from this. Dispatches binary vs
    multi-class internally and returns ``[B, H, W]``.
    """
    if _infer_is_binary(probs):
        p = _foreground_prob(probs).clamp(eps, 1.0 - eps)
        entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
        c_eff = 2
    else:
        entropy = -(probs * (probs + eps).log()).sum(dim=1)
        c_eff = probs.shape[1]
    max_entropy = torch.log(
        torch.tensor(float(c_eff), device=entropy.device, dtype=entropy.dtype)
    ).clamp(min=eps)
    return (entropy / max_entropy).clamp(0.0, 1.0)


def pixelwise_entropy(
    teacher_logits: torch.Tensor,
    tau: float = 4.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Predictive entropy of the temperature-softened teacher distribution.

    Computes ``H(p) = -sum_c p_c * log p_c`` per pixel, with ``p`` obtained
    by ``softmax(logits / tau)``. Binary vs. multi-class is inferred from
    ``teacher_logits.shape[1]``; for ``C == 1`` the single logit is
    expanded to ``[0, logit]`` so that entropy and KD share the same
    simplex used by ``LogitDistillLoss``.

    The computation is wrapped in ``torch.no_grad()`` since the teacher
    distribution is treated as a fixed target.

    Args:
        teacher_logits: Raw teacher logits, shape ``[B, C, H, W]``.
        tau: Softmax temperature. Larger ``tau`` produces a softer
            distribution and a larger entropy.
        eps: Log-stability constant added inside ``log``.

    Returns:
        Tensor of shape ``[B, H, W]`` holding entropy in nats, with values
        in ``[0, log(C_eff)]`` where ``C_eff = 2`` if binary else ``C``.
    """
    with torch.no_grad():
        logits = _expand_binary_logits(teacher_logits)
        p = F.softmax(logits / tau, dim=1)              # [B, C_eff, H, W]
        entropy = -(p * (p + eps).log()).sum(dim=1)      # [B, H, W]
    return entropy


def compute_margin_uncertainty(
    teacher_logits: torch.Tensor,
    tau: float = 4.0,
) -> torch.Tensor:
    """Margin-based uncertainty from teacher logits.

    Defined as ``u = 1 - (p_top1 - p_top2)``, where probabilities come from
    ``softmax(logits / tau)``. A small top-1/top-2 gap implies high
    uncertainty (``u -> 1``); a confident pixel yields ``u -> 0``.

    Binary vs. multi-class is inferred from ``teacher_logits.shape[1]``;
    for ``C == 1`` the logit is expanded to ``[0, logit]`` before softmax,
    matching the convention used by ``LogitDistillLoss``.

    Args:
        teacher_logits: Raw teacher logits, shape ``[B, C, H, W]``.
        tau: Softmax temperature.

    Returns:
        Tensor of shape ``[B, H, W]`` with values in ``[0, 1]``.
    """
    with torch.no_grad():
        logits = _expand_binary_logits(teacher_logits)
        p = F.softmax(logits / tau, dim=1)              # [B, C_eff, H, W]
        top2 = p.topk(k=2, dim=1).values                # [B, 2, H, W]
        margin = top2[:, 0] - top2[:, 1]                # [B, H, W]
        uncertainty = 1.0 - margin                      # invert margin to get uncertainty
    return uncertainty

def compute_disagreement_uncertainty(
    teacher_probs_a: torch.Tensor,
    teacher_probs_b: torch.Tensor,
    mode: str = "l1",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-pixel disagreement between two teacher predictions.

    Used for consistency-based reliability weighting: the same teacher is
    evaluated on two augmented views (after inverse-aligning to the same
    coordinate system), and pixels where the predictions disagree are
    flagged as unreliable. This is the single source of truth for the
    disagreement metric; ``consistency_weight`` maps the value returned here
    through an exponential decay.

    Modes:
        * ``"l1"``: mean absolute difference across classes,
          ``mean_c |p_a - p_b|``.
        * ``"kl"``: ``KL(p_a || p_b)``.

    Args:
        teacher_probs_a: Softmax probabilities of view A, ``[B, C, H, W]``.
        teacher_probs_b: Softmax probabilities of view B, ``[B, C, H, W]``.
            Must have the same shape as ``teacher_probs_a`` and be aligned
            to the same pixel grid.
        mode: Disagreement metric (``"l1"`` or ``"kl"``).
        eps: Lower clamp applied before ``log`` in the KL branch.

    Returns:
        Tensor of shape ``[B, H, W]`` with values in ``[0, inf)``. Larger
        values indicate stronger disagreement.

    Raises:
        ValueError: If shapes differ or ``mode`` is not recognised.
    """
    if teacher_probs_a.shape != teacher_probs_b.shape:
        raise ValueError("teacher_probs_a and teacher_probs_b must have the same shape.")

    if mode == "l1":
        uncertainty = (teacher_probs_a - teacher_probs_b).abs().mean(dim=1)  # [B, H, W]

    elif mode == "kl":
        p = teacher_probs_a.clamp(eps, 1.0)
        q = teacher_probs_b.clamp(eps, 1.0)
        uncertainty = (p * (p.log() - q.log())).sum(dim=1)  # [B, H, W]

    else:
        raise ValueError(f"Unknown disagreement mode: {mode}")

    return uncertainty

def augmentation_variance_uncertainty(
    probs_list: list[torch.Tensor],
    reduce_class: str = "mean",
) -> torch.Tensor:
    """Per-pixel variance across augmented teacher predictions.

    Given ``K`` teacher probability maps obtained from augmented views (and
    inverse-aligned to a common coordinate system), computes the across-view
    variance per class and reduces over the class dimension.

    Args:
        probs_list: List of ``K`` probability tensors, each
            ``[B, C, H, W]``. ``K >= 2`` and all entries must share shape
            and coordinate frame.
        reduce_class: Reduction across classes, ``"mean"`` or ``"max"``.

    Returns:
        Tensor of shape ``[B, H, W]`` clamped to ``[0, 1]``.

    Raises:
        ValueError: If fewer than two maps are provided, shapes mismatch,
            or ``reduce_class`` is invalid.
    """
    if len(probs_list) < 2:
        raise ValueError("At least two probability maps are required.")

    ref_shape = probs_list[0].shape
    for p in probs_list:
        if p.shape != ref_shape:
            raise ValueError("All probability maps must have the same shape.")

    stacked = torch.stack(probs_list, dim=0)  # [K, B, C, H, W]
    var = stacked.var(dim=0, unbiased=False)  # [B, C, H, W]

    if reduce_class == "mean":
        uncertainty = var.mean(dim=1)
    elif reduce_class == "max":
        uncertainty = var.max(dim=1).values
    else:
        raise ValueError(f"Unknown reduce_class: {reduce_class}")

    return uncertainty.clamp(0.0, 1.0)

def _entropy_from_probs_raw(
    probs: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Compute ``H(p) = -sum_c p_c * log p_c`` over an already-normalised map.

    Internal helper used by the ensemble uncertainty functions to avoid
    double-softmaxing inputs. Operates on ``[B, C, H, W]`` and returns
    ``[B, H, W]``. Unlike ``_normalized_entropy_from_probs`` this returns the
    raw (un-normalised) entropy and does not special-case binary inputs.
    """
    return -(probs * (probs + eps).log()).sum(dim=1)


def ensemble_predictive_entropy_uncertainty(
    probs_list: list[torch.Tensor],
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Predictive entropy of an ensemble / MC-dropout mean prediction.

    Total uncertainty in the Bayesian sense: ``H(E[p])`` where the
    expectation is the mean over members.

    Args:
        probs_list: List of ``K >= 2`` already-normalised probability
            tensors ``[B, C, H, W]``.
        normalize: If ``True``, divide by ``log(C)`` to get values in
            ``[0, 1]``.
        eps: Log-stability constant.

    Returns:
        Tensor of shape ``[B, H, W]``.

    Raises:
        ValueError: If fewer than two maps are provided.
    """
    if len(probs_list) < 2:
        raise ValueError("At least two probability maps are required.")

    mean_probs = torch.stack(probs_list, dim=0).mean(dim=0)  # [B, C, H, W]
    entropy = _entropy_from_probs_raw(mean_probs, eps=eps)

    if normalize:
        c = mean_probs.shape[1]
        entropy = entropy / torch.log(
            torch.tensor(c, device=entropy.device, dtype=entropy.dtype)
        ).clamp(min=eps)

    return entropy


def ensemble_mutual_information_uncertainty(
    probs_list: list[torch.Tensor],
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mutual information (epistemic uncertainty) of an ensemble.

    Decomposes total uncertainty ``H(E[p])`` into aleatoric and epistemic
    components and returns the epistemic part:

        I[y; theta] = H(E[p]) - E[H(p)]

    Args:
        probs_list: List of ``K >= 2`` already-normalised probability
            tensors ``[B, C, H, W]``.
        normalize: If ``True``, divide by ``log(C)``.
        eps: Log-stability constant.

    Returns:
        Tensor of shape ``[B, H, W]`` clamped to ``>= 0``.

    Raises:
        ValueError: If fewer than two maps are provided.
    """
    if len(probs_list) < 2:
        raise ValueError("At least two probability maps are required.")

    stacked = torch.stack(probs_list, dim=0)  # [K, B, C, H, W]
    mean_probs = stacked.mean(dim=0)

    predictive_entropy = _entropy_from_probs_raw(mean_probs, eps=eps)
    expected_entropy = torch.stack(
        [_entropy_from_probs_raw(p, eps=eps) for p in probs_list],
        dim=0,
    ).mean(dim=0)

    mutual_info = predictive_entropy - expected_entropy

    if normalize:
        c = mean_probs.shape[1]
        mutual_info = mutual_info / torch.log(
            torch.tensor(c, device=mutual_info.device, dtype=mutual_info.dtype)
        ).clamp(min=eps)

    return mutual_info.clamp(min=0.0)

def boundary_uncertainty_from_probs(
    foreground_probs: torch.Tensor,
    kernel_size: int = 3,
) -> torch.Tensor:
    """Boundary-like uncertainty from a binary foreground probability map.

    Computes the local range ``max_pool(p) - min_pool(p)`` within a
    ``kernel_size`` window. Pixels where probability changes rapidly
    (object boundaries) get a higher score.

    Args:
        foreground_probs: Foreground probability, ``[B, 1, H, W]`` or
            ``[B, H, W]``.
        kernel_size: Odd window size for the local range operator.

    Returns:
        Tensor of shape ``[B, H, W]`` clamped to ``[0, 1]``.

    Raises:
        ValueError: If ``kernel_size`` is even or the input has an
            unsupported shape.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")

    if foreground_probs.dim() == 3:
        p = foreground_probs.unsqueeze(1)
    elif foreground_probs.dim() == 4 and foreground_probs.shape[1] == 1:
        p = foreground_probs
    else:
        raise ValueError("foreground_probs must have shape [B, H, W] or [B, 1, H, W].")

    pad = kernel_size // 2
    local_max = F.max_pool2d(p, kernel_size=kernel_size, stride=1, padding=pad)
    local_min = -F.max_pool2d(-p, kernel_size=kernel_size, stride=1, padding=pad)

    local_range = (local_max - local_min).squeeze(1)
    return local_range.clamp(0.0, 1.0)


def prediction_aware_reliability_smoothing(
    reliability: torch.Tensor,
    teacher_probs: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Smooth a reliability map guided by local teacher-prediction similarity.

    Reliability factors are computed per pixel and can be spatially noisy —
    an isolated low-confidence pixel inside an otherwise reliable region (or
    vice-versa) makes the KD weight jitter across neighbouring pixels that
    the teacher predicts almost identically. This applies a *prediction-aware*
    (bilateral-style) smoothing: each pixel is replaced by a weighted average
    of its neighbours, where the weight decays with how different the
    neighbour's teacher distribution is:

        r_smooth(i) = sum_j w(i, j) r(j) / sum_j w(i, j)
        w(i, j)     = exp(-||p_t(i) - p_t(j)||^2 / sigma^2)

    Pixels straddling an object boundary (where ``p_t`` changes sharply) get
    near-zero affinity, so reliability does **not** bleed across edges — only
    within regions the teacher predicts consistently.

    This is a pure post-processing factor: it does not apply ``ignore_index``
    masking. ``build_reliability_map`` runs it before the final ignore mask.

    Args:
        reliability: Per-pixel reliability map ``[B, H, W]`` in ``[0, 1]``.
        teacher_probs: Teacher probability map ``[B, C, H, W]`` used to
            measure local prediction similarity. ``C == 1`` (binary) is
            supported; ``C > 1`` uses the full class distribution.
        kernel_size: Odd local window size.
        sigma: Controls how strongly prediction disagreement blocks
            smoothing. Smaller ``sigma`` keeps edges sharper.
        eps: Numerical stability constant.

    Returns:
        Smoothed reliability ``[B, H, W]`` clamped to ``[0, 1]``.

    Raises:
        ValueError: If shapes are inconsistent or ``kernel_size`` is even.
    """
    if reliability.dim() != 3:
        raise ValueError("reliability must have shape [B, H, W].")
    if teacher_probs.dim() != 4:
        raise ValueError("teacher_probs must have shape [B, C, H, W].")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")

    b, c, h, w = teacher_probs.shape
    if reliability.shape != (b, h, w):
        raise ValueError("reliability and teacher_probs spatial shapes must match.")

    pad = kernel_size // 2
    k2 = kernel_size * kernel_size

    # [B, C*K*K, H*W] -> [B, C, K*K, H, W]
    prob_patches = F.unfold(
        teacher_probs,
        kernel_size=kernel_size,
        padding=pad,
    ).view(b, c, k2, h, w)

    center_probs = teacher_probs.unsqueeze(2)  # [B, C, 1, H, W]

    # squared prediction distance within local window: [B, K*K, H, W]
    dist2 = ((prob_patches - center_probs) ** 2).sum(dim=1)

    # local affinity (bilateral weight)
    weights = torch.exp(-dist2 / (sigma ** 2 + eps))  # [B, K*K, H, W]

    # zero-padding introduces phantom out-of-bounds neighbours; mask them out
    # so border pixels are averaged only over real neighbours (otherwise a
    # uniform map would darken at the edges toward the padded zeros).
    valid = F.unfold(
        torch.ones(b, 1, h, w, device=reliability.device, dtype=reliability.dtype),
        kernel_size=kernel_size,
        padding=pad,
    ).view(b, k2, h, w)
    weights = weights * valid

    # reliability patches: [B, K*K, H, W]
    r_patches = F.unfold(
        reliability.unsqueeze(1),
        kernel_size=kernel_size,
        padding=pad,
    ).view(b, k2, h, w)

    r_smooth = (weights * r_patches).sum(dim=1) / weights.sum(dim=1).clamp_min(eps)

    return r_smooth.clamp(0.0, 1.0)


# Probability helpers
def temperature_softmax(logits: torch.Tensor, temperature: float = 1.0, dim: int = 1) -> torch.Tensor:
    """Temperature-scaled softmax for multi-class logits.

    Computes ``softmax(logits / T)`` along ``dim``. Higher temperatures
    yield smoother distributions, which is the standard Hinton-KD setup.

    Args:
        logits: Logits tensor, typically ``[B, C, H, W]``.
        temperature: Positive scalar ``T``. ``T == 1`` reduces to plain
            softmax; ``T > 1`` softens the distribution.
        dim: Class dimension.

    Returns:
        Probability tensor with the same shape as ``logits``.

    Raises:
        ValueError: If ``temperature <= 0``.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    return F.softmax(logits / temperature, dim=dim)


def binary_temperature_sigmoid(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Temperature-scaled sigmoid for binary segmentation logits.

    Computes ``sigmoid(logits / T)`` and returns the foreground
    probability. Counterpart to ``temperature_softmax`` for the binary
    case.

    Args:
        logits: Logits tensor of shape ``[B, 1, H, W]`` or ``[B, H, W]``.
        temperature: Positive scalar ``T``. ``T > 1`` softens the
            sigmoid response.

    Returns:
        Foreground probabilities with the same shape as ``logits``.

    Raises:
        ValueError: If ``temperature <= 0``.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    return torch.sigmoid(logits / temperature)


def probs_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Convert logits to probabilities, dispatching on binary vs multi-class.

    Wraps ``binary_temperature_sigmoid`` / ``temperature_softmax`` behind the
    shared shape inference so callers (notably ``build_reliability_map``)
    don't repeat the binary branch.

    Args:
        logits: Raw logits, ``[B, H, W]``, ``[B, 1, H, W]`` or ``[B, C, H, W]``.
        temperature: Softening temperature.

    Returns:
        Probabilities with the same shape as ``logits``.
    """
    if _infer_is_binary(logits):
        return binary_temperature_sigmoid(logits, temperature=temperature)
    return temperature_softmax(logits, temperature=temperature)

