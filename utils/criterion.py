import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Soft Dice loss for segmentation.

    Computes ``1 - Dice`` per (batch, class) using the standard
    Sorensen-Dice formula

        Dice = (2 * sum(p * y) + smooth) / (sum(p^2) + sum(y^2) + smooth)

    where ``p`` is the activated probability map and ``y`` is the
    one-hot mask. With ``squared=False`` the denominator uses linear
    sums instead of squared sums.

    Args:
        smooth: Numerical stabiliser added to both numerator and
            denominator.
        squared: Use squared sums in the denominator when ``True``
            (default, matches V-Net style).
    """

    def __init__(self, smooth=1e-5, squared=True):
        super().__init__()
        self.smooth = smooth
        self.squared = squared

    def forward(self, inputs, target, weight=None, activation="auto"):
        """Compute the Dice loss.

        Args:
            inputs: Logits or probabilities, shape ``[B, C, H, W]``.
            target: One-hot mask with the same shape as ``inputs``,
                values in ``{0, 1}``.
            weight: Optional per-class weight vector of length ``C``.
                When provided, the loss is reweighted and normalised by
                ``weight.sum()``.
            activation: Activation applied to ``inputs`` before the
                Dice computation. One of:
                * ``"auto"`` (default): ``"sigmoid"`` if ``C == 1``,
                  otherwise ``"softmax"``.
                * ``"sigmoid"``: binary or multi-label segmentation.
                * ``"softmax"``: mutually exclusive multi-class.
                * ``"none"``: inputs are already probabilities.

        Returns:
            Scalar Dice loss averaged over batch and classes (or
            weight-normalised when ``weight`` is given).

        Raises:
            ValueError: If ``activation`` is unknown or the activated
                shape does not match ``target``.
        """
        if activation == "auto":
            is_multiclass = inputs.dim() == 4 and inputs.shape[1] > 1
            activation = "softmax" if is_multiclass else "sigmoid"

        if activation == "sigmoid":
            probs = torch.sigmoid(inputs)
        elif activation == "softmax":
            probs = torch.softmax(inputs, dim=1)
        elif activation == "none":
            probs = inputs
        else:
            raise ValueError(f"Unknown activation: {activation}")

        target = target.float()

        if probs.shape != target.shape:
            raise ValueError(f"predict {probs.shape} & target {target.shape} shape do not match")

        dims = tuple(range(2, probs.ndim))  # H, W 또는 H, W, D 등 spatial dims

        intersect = torch.sum(probs * target, dim=dims)

        if self.squared:
            denominator = torch.sum(probs ** 2, dim=dims) + torch.sum(target ** 2, dim=dims)
        else:
            denominator = torch.sum(probs, dim=dims) + torch.sum(target, dim=dims)

        dice = (2 * intersect + self.smooth) / (denominator + self.smooth)
        loss = 1 - dice  # [B, C]

        if weight is not None:
            weight = torch.as_tensor(weight, dtype=loss.dtype, device=loss.device)
            loss = loss * weight.view(1, -1)
            return loss.sum() / (loss.shape[0] * weight.sum())

        return loss.mean()


# Distillation losses
class TaskLoss(nn.Module):
    """Ground-truth supervised loss for the distillation student.

    Combines cross-entropy and Dice components, averaging the two when
    both are enabled. The binary vs. multi-class dispatch is inferred
    from ``logits.shape[1]`` at forward time:

    Binary (``C == 1``):
        * CE: ``BCEWithLogitsLoss`` with positive-class weighting.
        * Dice: ``DiceLoss`` with sigmoid activation.

    Multi-class (``C > 1``):
        * CE: ``CrossEntropyLoss``; one-hot targets are reduced to
          ``[B, H, W]`` class indices via ``argmax``.
        * Dice: ``DiceLoss`` with softmax activation.

    Args:
        use_ce: Include the CE component.
        use_dice: Include the Dice component.
        pos_weight: Positive-class weight for binary BCE. Ignored when
            the input has ``C > 1`` channels.
    """

    def __init__(
        self,
        use_ce: bool = True,
        use_dice: bool = True,
        pos_weight: float = 5.0,
    ):
        super().__init__()
        self.use_ce = use_ce
        self.use_dice = use_dice

        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.cross_entropy = nn.CrossEntropyLoss()

        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the supervised task loss.

        Args:
            logits: Raw model output ``[B, C, H, W]``. ``C == 1``
                triggers the binary (BCE + sigmoid-Dice) path; ``C > 1``
                triggers the multi-class (CE + softmax-Dice) path.
            targets: One-hot float masks ``[B, C, H, W]``.

        Returns:
            Scalar loss equal to the mean of active components
            (``use_ce`` and/or ``use_dice``). Returns ``0`` if both
            components are disabled.
        """
        is_multiclass = logits.dim() == 4 and logits.shape[1] > 1
        losses = []

        if targets.dim() == logits.dim() - 1:
            target_mask = targets.unsqueeze(1).float()
        else:
            target_mask = targets.float()

        if self.use_ce:
            if is_multiclass:
                # CrossEntropyLoss expects [B, H, W] long class-index targets
                losses.append(self.cross_entropy(logits, target_mask.argmax(dim=1).long()))
            else:
                losses.append(self.bce(logits, target_mask))

        if self.use_dice:
            losses.append(self.dice(logits, target_mask))

        if not losses:
            return torch.tensor(0.0, device=logits.device)

        return sum(losses) / len(losses)


class LogitDistillLoss(nn.Module):
    """Hinton-style soft-label KD distillation on output logits.

    Binary vs. multi-class dispatch is inferred from
    ``student_logits.shape[1]`` at forward time.

    Multi-class (``C > 1``):
        ``L = KL(softmax(t / T) || softmax(s / T)) * T^2``, averaged
        over the batch via ``KLDivLoss(reduction='batchmean')`` and
        additionally divided by ``H * W`` so the magnitude is
        comparable to the per-pixel task loss.

    Binary (``C == 1`` or 3-D ``[B, H, W]``):
        ``L = BCE(s / T, sigmoid(t / T)) * T^2 / B``, summed over
        pixels and divided by the batch size (then further divided by
        ``H * W`` for 4-D tensors). Equivalent to per-pixel KL on the
        2-class simplex ``[0, logit]``, but evaluated directly through
        the BCE-with-logits formulation for numerical stability.

    Args:
        temperature: Distillation temperature ``T``. Higher values
            soften both distributions.
    """

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute the soft-label distillation loss.

        Args:
            student_logits: Student logits ``[B, C, H, W]``.
            teacher_logits: Teacher logits ``[B, C, H, W]``; must match
                the student's spatial size.

        Returns:
            Scalar KD loss with ``T^2`` rescaling applied.
        """
        T = self.temperature
        is_multiclass = student_logits.dim() == 4 and student_logits.shape[1] > 1

        if not is_multiclass:
            loss = F.binary_cross_entropy_with_logits(
                student_logits / T,
                torch.sigmoid(teacher_logits / T).detach(),
                reduction="sum",
            ) / student_logits.shape[0]
            loss = loss * (T ** 2)
        else:
            s_soft = F.log_softmax(student_logits / T, dim=1)
            t_soft = F.softmax(teacher_logits / T, dim=1)

            loss = self.kl_div(s_soft, t_soft) * (T ** 2)

        # Normalise by spatial size for 4-D tensors
        if student_logits.dim() == 4:
            loss = loss / (student_logits.shape[2] * student_logits.shape[3])

        return loss

class ReliabilityWeightedKDLoss(nn.Module):
    """Pixel-wise KD loss reweighted by a teacher reliability map.

    Companion to ``utils.reliability_kd.build_reliability_map``: each
    pixel's KD contribution is multiplied by a reliability weight
    ``r in [0, 1]`` and the final loss is a reliability-normalised mean

        L = sum(r * kd_map) / (sum(r) + eps)

    Auto-dispatches on the student's channel dimension:
        * multi-class (``C > 1``): KL divergence ``KL(p_t || p_s)``
          summed over classes per pixel.
        * binary (``C == 1`` or 3-D): BCE between foreground student
          logits and clipped teacher probabilities.

    ``T^2`` rescaling is applied to keep the gradient scale comparable
    to the GT loss, matching the Hinton convention.

    Args:
        temperature: Distillation temperature ``T``.
        eps: Numerical stability constant; used both for log clipping
            and for the denominator of the reliability-weighted mean.
    """

    def __init__(self, temperature: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.kl_div = nn.KLDivLoss(reduction="none")


    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor | None = None,
        teacher_probs: torch.Tensor | None = None,
        reliability: torch.Tensor | None = None,
        ignore_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the reliability-weighted KD loss.

        Args:
            student_logits: Student logits. Multi-class ``[B, C, H, W]``
                with ``C > 1``, or binary ``[B, 1, H, W]`` / ``[B, H, W]``.
            teacher_logits: Teacher logits with the same shape as
                ``student_logits``. Either this or ``teacher_probs`` is
                required.
            teacher_probs: Pre-tempered teacher probabilities (softmax
                for multi-class, sigmoid for binary). Takes precedence
                over ``teacher_logits`` when supplied.
            reliability: Per-pixel weight map ``[B, H, W]`` in
                ``[0, 1]``. Defaults to all-ones (unweighted).
            ignore_mask: Per-pixel inclusion mask ``[B, H, W]`` where
                ``1`` keeps the pixel and ``0`` drops it; multiplied
                into ``reliability`` before reduction.

        Returns:
            Scalar loss with ``T^2`` rescaling, normalised by the sum of
            (possibly masked) reliability weights.

        Raises:
            ValueError: If neither ``teacher_logits`` nor
                ``teacher_probs`` is provided.
        """
        T = self.temperature
        eps = self.eps

        is_multiclass = student_logits.dim() == 4 and student_logits.shape[1] > 1

        if is_multiclass:
            if teacher_probs is None:
                if teacher_logits is None:
                    raise ValueError(
                        "Either teacher_logits or teacher_probs must be provided."
                    )
                teacher_probs = F.softmax(teacher_logits / T, dim=1).detach()
            s_log = F.log_softmax(student_logits / T, dim=1)
            kd_map = self.kl_div(s_log, teacher_probs)
            kd_map = kd_map.sum(dim=1)  # sum over classes to get per-pixel map
        else:
            s_logits = (
                student_logits[:, 0] if student_logits.dim() == 4 else student_logits
            )

            if teacher_probs is None:
                if teacher_logits is None:
                    raise ValueError(
                        "Either teacher_logits or teacher_probs must be provided."
                    )
                teacher_probs = torch.sigmoid(teacher_logits / T).detach()

            t_probs = teacher_probs[:, 0] if teacher_probs.dim() == 4 else teacher_probs

            kd_map = F.binary_cross_entropy_with_logits(
                s_logits / T,
                t_probs.clamp(eps, 1.0 - eps),
                reduction="none",
            )

        # Hinton KD convention
        kd_map = kd_map * (T ** 2)

        if reliability is None:
            reliability = torch.ones_like(kd_map)

        if ignore_mask is not None:
            reliability = reliability * ignore_mask.to(reliability.dtype)

        return (kd_map * reliability).sum() / (reliability.sum() + eps)


class ChannelWiseDistillLoss(nn.Module):
    """Channel-wise Knowledge Distillation (CWD) for dense prediction.

    Reference: Shu et al., "Channel-wise Knowledge Distillation for
    Dense Prediction", ICCV 2021. Re-implemented from scratch because
    the official submodule targets PyTorch 0.4.1.

    For each channel the spatial activations are softmax-normalised
    into a probability map over pixels, and KL divergence is computed
    channel-by-channel between teacher and student:

        kl_c = KL(softmax(t_c / T) || softmax(s_c / T))
        L    = mean_{B, c} kl_c * T^2

    Channel alignment:
        The paper aligns teacher/student channels with a 1x1 conv on
        the student side. In this codebase that role is normally played
        by ``FeatureAdapter`` inside ``UnifiedDistiller`` — pass
        already-aligned tensors and leave ``student_channels=None``.
        For standalone use, set both ``student_channels`` and
        ``teacher_channels`` to build an internal projection
        (``Conv2d -> BN -> ReLU``, matching the paper).

    Binary segmentation (``C == 1``):
        CWD still gives a per-channel spatial-saliency alignment signal
        and avoids the near-trivial collapse of pixel-wise KL on a
        2-class distribution. However it only matches *relative*
        foreground saliency within each channel and does NOT distill
        the calibrated foreground/background probability or the
        decision boundary. Use alongside the task loss (BCE + Dice),
        not as a replacement.

    Args:
        temperature: Softmax temperature ``T`` for the per-channel
            spatial softmax.
        match_spatial: If ``True``, bilinearly resize the teacher map
            to the student's spatial size when they differ.
        student_channels: Optional input channel count for the internal
            1x1 alignment projection. Must be set together with
            ``teacher_channels``.
        teacher_channels: Output channel count for the alignment
            projection. See ``student_channels``.
        detach_teacher: Detach the teacher tensor so gradients do not
            flow into the teacher network.

    Raises:
        ValueError: If exactly one of ``student_channels`` /
            ``teacher_channels`` is provided.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        match_spatial: bool = True,
        student_channels: int = None,
        teacher_channels: int = None,
        detach_teacher: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.match_spatial = match_spatial
        self.detach_teacher = detach_teacher
        self.kl_div = nn.KLDivLoss(reduction="none")

        if (student_channels is None) ^ (teacher_channels is None):
            raise ValueError(
                "student_channels and teacher_channels must both be set or both be None"
            )
        if student_channels is not None and student_channels != teacher_channels:
            self.align = nn.Sequential(
                nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(teacher_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.align = None

    def forward(
        self, student_feat: torch.Tensor, teacher_feat: torch.Tensor
    ) -> torch.Tensor:
        """Compute the channel-wise distillation loss.

        Args:
            student_feat: Student logits or features
                ``[B, C_s, H, W]``.
            teacher_feat: Teacher features ``[B, C_t, H', W']``.
                After optional internal 1x1 alignment ``C_t`` must
                equal ``C_s``; ``H'`` / ``W'`` may differ and are
                resized when ``match_spatial=True``.

        Returns:
            Scalar CWD loss with ``T^2`` rescaling.

        Raises:
            AssertionError: If channel counts still mismatch after
                alignment.
        """
        if self.detach_teacher:
            teacher_feat = teacher_feat.detach()

        if self.align is not None:
            student_feat = self.align(student_feat)

        if student_feat.dtype != teacher_feat.dtype:
            teacher_feat = teacher_feat.to(student_feat.dtype)
        if student_feat.device != teacher_feat.device:
            teacher_feat = teacher_feat.to(student_feat.device)

        if self.match_spatial and student_feat.shape[-2:] != teacher_feat.shape[-2:]:
            teacher_feat = F.interpolate(
                teacher_feat,
                size=student_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        assert student_feat.shape[1] == teacher_feat.shape[1], (
            f"CWD requires matching channel count after alignment, "
            f"got student C={student_feat.shape[1]} vs teacher C={teacher_feat.shape[1]}. "
            f"Either pre-align externally (e.g. via FeatureAdapter) or set "
            f"student_channels/teacher_channels on this loss."
        )

        T = self.temperature
        B, C, H, W = student_feat.shape

        # Per-channel spatial softmax: each channel becomes a probability map
        s_log = F.log_softmax(student_feat.reshape(B, C, -1) / T, dim=-1)
        t_prob = F.softmax(teacher_feat.reshape(B, C, -1) / T, dim=-1)

        # KL(t || s) accumulated over spatial positions, mean over (B, C), τ²-rescaled
        kl = self.kl_div(s_log, t_prob)
        return kl.sum() * (T ** 2)


class UncertaintyWeightedKDLoss(nn.Module):
    """Entropy-weighted KD loss wrapped as a single ``nn.Module``.

    Pipeline:
        1. Compute teacher pixel-wise entropy via
           ``reliability_kd.pixelwise_entropy``.
        2. Map entropy to a per-pixel confidence weight in ``[0, 1]``
           using either a linear or an exponential schedule.
        3. Apply the weight to a per-pixel KL map and reduce with
           ``T^2`` rescaling.

    Binary vs. multi-class dispatch is inferred from
    ``student_logits.shape[1]`` at forward time. For binary (``C == 1``)
    logits are expanded to ``[0, logit]`` before softmax to keep entropy
    and KD on the same 2-class simplex.

    ``forward`` returns ``(loss, uncertainty_map, weight_map)`` so the
    caller can log diagnostic statistics without recomputing them.

    Args:
        tau: Distillation temperature.
        weight_type: ``"linear"`` (``w = 1 - u / log(C_eff)``) or
            ``"exp"`` (``w = exp(-beta * u)``).
        beta: Exponential decay rate. Ignored in linear mode.
        eps: Log-stability epsilon used in the entropy computation.
    """

    def __init__(
        self,
        tau: float = 4.0,
        weight_type: str = "linear",
        beta: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.tau = tau
        self.weight_type = weight_type
        self.beta = beta
        self.eps = eps

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ):
        """Compute the entropy-weighted KD loss with diagnostic maps.

        Args:
            student_logits: Student logits ``[B, C, H, W]``. ``C == 1``
                triggers the binary path; ``C > 1`` triggers
                multi-class.
            teacher_logits: Teacher logits ``[B, C, H, W]``.

        Returns:
            Tuple ``(loss, uncertainty, weight)`` where
                * ``loss``: scalar weighted KD loss.
                * ``uncertainty``: ``[B, H, W]`` teacher entropy
                  (detached).
                * ``weight``: ``[B, H, W]`` confidence weight in
                  ``[0, 1]`` (detached).
        """
        from utils.reliability_kd import pixelwise_entropy

        is_multiclass = student_logits.dim() == 4 and student_logits.shape[1] > 1
        # C_eff is num_classes for multi-class, 2 for binary (after expansion)
        c_eff = student_logits.shape[1] if is_multiclass else 2

        uncertainty = pixelwise_entropy(teacher_logits, tau=self.tau, eps=self.eps)
        weight = self.uncertainty_weight(
            uncertainty,
            weight_type=self.weight_type,
            beta=self.beta,
            num_classes_effective=c_eff,
        )
        loss = self.uncertainty_weighted_kd_loss(
            student_logits, teacher_logits, weight, tau=self.tau,
        )
        return loss, uncertainty, weight
    
    def uncertainty_weight(
        self,
        uncertainty: torch.Tensor,
        weight_type: str = "linear",
        beta: float = 1.0,
        num_classes_effective: int = 2,
    ) -> torch.Tensor:
        """Map pixel-wise entropy to a KD confidence weight in ``[0, 1]``.

        Schedules:
            * ``"linear"``: ``w = clamp(1 - u / log(C_eff), 0, 1)``.
              Zero weight at maximum entropy, one at zero entropy.
            * ``"exp"``: ``w = exp(-beta * u)``. Smoother decay; larger
              ``beta`` drops the weight faster with entropy.

        Args:
            uncertainty: Pixel-wise entropy ``[B, H, W]`` (e.g. from
                ``pixelwise_entropy``).
            weight_type: ``"linear"`` or ``"exp"``.
            beta: Decay rate used by the exponential schedule.
            num_classes_effective: ``C_eff`` for ``log(C_eff)``
                normalisation (use ``2`` for binary).

        Returns:
            Detached weight tensor ``[B, H, W]`` with values in
            ``[0, 1]``.
        """
        if weight_type == "exp":
            weight = torch.exp(-beta * uncertainty)
        else:
            # linear: normalise entropy to [0, 1] then invert
            max_entropy = torch.tensor(
                num_classes_effective, dtype=uncertainty.dtype
            ).log().to(uncertainty.device)
            weight = (1.0 - uncertainty / max_entropy.clamp(min=1e-8)).clamp(0.0, 1.0)
        return weight.detach()


    def uncertainty_weighted_kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        weight: torch.Tensor,
        tau: float = 4.0,
    ) -> torch.Tensor:
        """Per-pixel KL divergence weighted by teacher confidence.

        KL is first accumulated over the class dimension at each pixel,
        then multiplied by the weight map, and only then reduced — the
        reverse order would distort the per-pixel reweighting.

        Binary vs. multi-class dispatch is inferred from
        ``student_logits.shape[1]``. For binary (``C == 1``) logits are
        expanded to ``[0, logit]`` before softmax so the simplex matches
        ``pixelwise_entropy``.

        Args:
            student_logits: Student logits ``[B, C, H, W]``.
            teacher_logits: Teacher logits ``[B, C, H, W]``; same
                spatial size as student.
            weight: Detached confidence map ``[B, H, W]`` (no grad).
            tau: Softmax temperature.

        Returns:
            Scalar KD loss with ``T^2`` rescaling applied.
        """
        is_multiclass = student_logits.dim() == 4 and student_logits.shape[1] > 1
        if not is_multiclass:
            student_logits = torch.cat(
                [torch.zeros_like(student_logits), student_logits], dim=1
            )
            teacher_logits = torch.cat(
                [torch.zeros_like(teacher_logits), teacher_logits], dim=1
            )

        # tempered distributions
        p_s_log = F.log_softmax(student_logits / tau, dim=1)    # [B, C_eff, H, W]
        p_t = F.softmax(teacher_logits / tau, dim=1).detach()   # [B, C_eff, H, W]

        # pixel-wise KL: KL(p_t || p_s) = sum_c p_t * (log p_t - log p_s)
        kl_map = (p_t * (p_t.clamp(min=1e-8).log() - p_s_log)).sum(dim=1)  # [B, H, W]
        kl_map = kl_map.clamp(min=0.0)                          # numerical safety

        # weight and reduce; tau^2 restores scale (standard KD convention)
        return (weight * kl_map).mean() * (tau ** 2)


class CriterionAdv(nn.Module):
    """Adversarial discriminator loss for KD with a teacher-vs-student GAN.

    The teacher discriminator output is treated as the "real" sample
    and the student's as "fake"; the loss form is selected by
    ``adv_type``:

        * ``"wgan-gp"``: ``L = -mean(D(t)) + mean(D(s))``
          (Wasserstein loss; pair with a gradient penalty applied
          externally).
        * ``"hinge"``: ``L = mean(relu(1 - D(t))) + mean(relu(1 + D(s)))``.

    Args:
        adv_type: Loss variant, ``"wgan-gp"`` or ``"hinge"``.

    Raises:
        ValueError: If ``adv_type`` is not one of the supported variants.
    """

    def __init__(self, adv_type):
        super(CriterionAdv, self).__init__()
        if (adv_type != 'wgan-gp') and (adv_type != 'hinge'):
            raise ValueError('adv_type should be wgan-gp or hinge')
        self.adv_loss = adv_type

    def forward(self, d_out_S, d_out_T):
        """Compute the adversarial loss.

        Args:
            d_out_S: Sequence of student discriminator outputs; only
                ``d_out_S[0]`` is used (the discriminator's primary
                scalar map / logit tensor).
            d_out_T: Sequence of teacher discriminator outputs with the
                same convention.

        Returns:
            Scalar adversarial loss.
        """
        assert d_out_S[0].shape == d_out_T[0].shape,'the output dim of D with teacher and student as input differ'
        '''teacher output'''
        d_out_real = d_out_T[0]
        if self.adv_loss == 'wgan-gp':
            d_loss_real = - torch.mean(d_out_real)
        elif self.adv_loss == 'hinge':
            d_loss_real = nn.ReLU()(1.0 - d_out_real).mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')

        # apply Gumbel Softmax
        '''student output'''
        d_out_fake = d_out_S[0]
        if self.adv_loss == 'wgan-gp':
            d_loss_fake = d_out_fake.mean()
        elif self.adv_loss == 'hinge':
            d_loss_fake = nn.ReLU()(1.0 + d_out_fake).mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')
        return d_loss_real + d_loss_fake
