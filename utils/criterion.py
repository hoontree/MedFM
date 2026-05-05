import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.sam_utils import DiceLoss


class MaskDiceLoss(DiceLoss):
    def __init__(self):
        super().__init__(n_classes=1)

    def forward(self, pred, target, weight=None, sigmoid=False):
        if sigmoid:
            pred = torch.sigmoid(pred)  # b 1 h w
        assert pred.size() == target.size(), "predict {} & target {} shape do not match".format(
            pred.size(), target.size()
        )
        dice_loss = self._dice_loss(pred[:, 0], target[:, 0])
        return dice_loss


class Mask_DC_and_BCE_loss(nn.Module):
    def __init__(self, dice_weight=0.8):
        super(Mask_DC_and_BCE_loss, self).__init__()

        self.ce = torch.nn.BCEWithLogitsLoss()
        self.dc = MaskDiceLoss()
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        loss_ce = self.ce(pred, target)
        loss_dice = self.dc(pred, target, sigmoid=True)
        loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice

        return loss


# ---------------------------------------------------------------------------
# Distillation losses
# ---------------------------------------------------------------------------


class TaskLoss(nn.Module):
    """
    Ground-truth supervised loss for distillation student.

    Supports binary (num_classes=1) and multiclass segmentation.

    Binary:
        - CE: BCEWithLogitsLoss (pos_weight=5.0)
        - Dice: sigmoid-activated DiceLoss

    Multiclass:
        - CE: CrossEntropyLoss  (targets: one-hot [B,C,H,W] → argmax → [B,H,W] long)
        - Dice: softmax-activated DiceLoss

    Args:
        num_classes: number of output channels.
        use_ce: include CE component (default True).
        use_dice: include Dice component (default True).
        pos_weight: scalar weight for positive class in binary BCE (default 5.0).
    """

    def __init__(
        self,
        num_classes: int = 1,
        use_ce: bool = True,
        use_dice: bool = True,
        pos_weight: float = 5.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_ce = use_ce
        self.use_dice = use_dice

        if num_classes == 1:
            self.register_buffer("pos_weight", torch.tensor([pos_weight]))
            self.ce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            self.ce = nn.CrossEntropyLoss()

        self.dice = DiceLoss(num_classes)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C, H, W] raw model output.
            targets: [B, C, H, W] float one-hot masks.
        Returns:
            Scalar loss averaged over active components.
        """
        target_mask = targets.float()
        losses = []

        if self.use_ce:
            if self.num_classes == 1:
                losses.append(self.ce(logits, target_mask))
            else:
                # CrossEntropyLoss expects [B, H, W] long class-index targets
                target_idx = targets.argmax(dim=1).long()
                losses.append(self.ce(logits, target_idx))

        if self.use_dice:
            use_softmax = self.num_classes > 1
            losses.append(
                self.dice(logits, target_mask, softmax=use_softmax, sigmoid=not use_softmax)
            )

        if not losses:
            return torch.tensor(0.0, device=logits.device)

        return sum(losses) / len(losses)


class LogitDistillLoss(nn.Module):
    """
    Soft-label KL-divergence distillation loss between student and teacher logits.

    Binary case: logits are expanded to 2-channel [neg, pos] before softmax,
    so KL is computed over a proper 2-class distribution.

    Multiclass case: softmax is applied directly over the class channel.

    Spatial averaging: PyTorch KLDivLoss(reduction='batchmean') divides by
    batch size but sums over all other dims. For 4-D tensors we divide by
    H*W so the magnitude is comparable to the task loss.

    Args:
        num_classes:  number of output channels.
        temperature:  distillation temperature T (default 4.0).
    """

    def __init__(self, num_classes: int = 1, temperature: float = 4.0):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            student_logits: [B, C, H, W]
            teacher_logits: [B, C, H, W]  (must be same spatial size as student)
        Returns:
            Scalar distillation loss.
        """
        T = self.temperature

        if self.num_classes == 1:
            s_soft = F.log_softmax(
                torch.cat([torch.zeros_like(student_logits), student_logits], dim=1) / T,
                dim=1,
            )
            t_soft = F.softmax(
                torch.cat([torch.zeros_like(teacher_logits), teacher_logits], dim=1) / T,
                dim=1,
            )
        else:
            s_soft = F.log_softmax(student_logits / T, dim=1)
            t_soft = F.softmax(teacher_logits / T, dim=1)

        loss = self.kl_div(s_soft, t_soft) * (T ** 2)

        # Normalise by spatial size for 4-D tensors
        if student_logits.dim() == 4:
            loss = loss / (student_logits.shape[2] * student_logits.shape[3])

        return loss


class FeatureDistillLoss(nn.Module):
    """
    MSE loss between a pair of (adapter-projected) student and teacher feature maps.

    If the spatial resolutions differ, the teacher feature is bilinearly
    interpolated to match the student.

    Args:
        None  (stateless; adapters live in the distiller)
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self, student_feat: torch.Tensor, teacher_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            student_feat: [B, C, H, W]
            teacher_feat: [B, C, H', W']
        Returns:
            Scalar MSE loss.
        """
        if student_feat.shape != teacher_feat.shape:
            teacher_feat = F.interpolate(
                teacher_feat,
                size=student_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.mse(student_feat, teacher_feat)


class ChannelWiseDistillLoss(nn.Module):
    """
    Channel-wise Knowledge Distillation (CWD) for dense prediction.
    Reference: Shu et al., ICCV 2021 (re-implemented from scratch — submodule
    code targets PyTorch 0.4.1 and is not imported).

    Each channel's spatial activations are softmax-normalized into a probability
    map, then KL divergence is computed channel-by-channel between teacher and
    student. The result is averaged over channels and batch and τ²-rescaled.

    Channel alignment
        The paper aligns teacher/student channels via a 1×1 conv on the
        student side. In this codebase that role is normally played by
        ``FeatureAdapter`` inside ``UnifiedDistiller`` — pass already-aligned
        tensors and leave ``student_channels=None``. When the loss is used
        standalone, set ``student_channels`` and ``teacher_channels`` to build
        an internal 1×1 projection (Conv2d → BN → ReLU, paper-style).

    Binary segmentation (C=1)
        CWD still yields a per-channel spatial-saliency alignment signal and
        avoids the near-trivial collapse of per-pixel KL on a 2-class
        distribution. However it only matches the *relative* foreground
        saliency map within each channel — it does NOT distill the calibrated
        foreground/background probability or the decision boundary itself.
        Use it together with the task loss (BCE+Dice), not as a replacement.

    Args:
        temperature:       softmax temperature τ (default 4.0).
        match_spatial:     bilinearly resize teacher to student spatial size
                           when they differ (default True).
        student_channels:  if set with ``teacher_channels``, build an internal
                           1×1 projection from student_channels → teacher_channels.
        teacher_channels:  see above.
        detach_teacher:    detach teacher tensor before loss to keep teacher
                           out of the autograd graph (default True).
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
        """
        Args:
            student_feat: [B, C_s, H, W]   logits or feature map
            teacher_feat: [B, C_t, H', W'] (C_t must equal C_s after optional
                          internal 1×1 alignment; H'×W' may differ)
        Returns:
            Scalar CWD loss.
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
        kl = (t_prob * (t_prob.clamp(min=1e-8).log() - s_log)).sum(dim=-1)  # [B, C]
        return kl.mean() * (T ** 2)
    
# ---------------------------------------------------------------------------
# Uncertainty-Weighted Knowledge Distillation
# ---------------------------------------------------------------------------


def compute_teacher_uncertainty(
    teacher_logits: torch.Tensor,
    tau: float = 4.0,
    eps: float = 1e-8,
    num_classes: int = 1,
) -> torch.Tensor:
    """
    Compute pixel-wise entropy of the temperature-softened teacher distribution.

    For binary segmentation (num_classes=1) the single logit is first expanded
    to a 2-class representation [neg, pos] — exactly the same convention used
    in LogitDistillLoss — so both entropy and KD operate on the same simplex.

    Args:
        teacher_logits: [B, C, H, W]  raw teacher logits (no grad required)
        tau:            temperature for softening
        eps:            log-stability constant
        num_classes:    1 for binary, >1 for multiclass

    Returns:
        uncertainty: [B, H, W]  pixel-wise entropy in nats, range [0, log(C_eff)]
    """
    with torch.no_grad():
        if num_classes == 1:
            # expand to 2-class [neg, pos]
            teacher_logits = torch.cat(
                [torch.zeros_like(teacher_logits), teacher_logits], dim=1
            )
        p = F.softmax(teacher_logits / tau, dim=1)          # [B, C_eff, H, W]
        entropy = -(p * (p + eps).log()).sum(dim=1)          # [B, H, W]
    return entropy


def compute_uncertainty_weight(
    uncertainty: torch.Tensor,
    weight_type: str = "linear",
    beta: float = 1.0,
    num_classes_effective: int = 2,
) -> torch.Tensor:
    """
    Map pixel-wise entropy to a KD confidence weight in [0, 1].

    Option A (linear):  w = clamp(1 - u / log(C), 0, 1)
        — zero weight at maximum entropy, one at zero entropy.
    Option B (exp):     w = exp(-beta * u)
        — smoother decay; beta controls how fast weight drops with entropy.

    Args:
        uncertainty:           [B, H, W]  from compute_teacher_uncertainty
        weight_type:           "linear" or "exp"
        beta:                  exponential decay rate (ignored for linear)
        num_classes_effective: C for log(C) normalisation (use 2 for binary)

    Returns:
        weight: [B, H, W]  detached, values in [0, 1]
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


def compute_uncertainty_weighted_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    weight: torch.Tensor,
    tau: float = 4.0,
    num_classes: int = 1,
) -> torch.Tensor:
    """
    Pixel-wise KL divergence weighted by teacher confidence, scaled by tau^2.

    KL is accumulated per pixel first, then the weight map is applied before
    the final mean reduction — NOT the other way around.

    For binary (num_classes=1): logits are expanded to 2-channel [neg, pos]
    before softmax, matching compute_teacher_uncertainty.

    Args:
        student_logits: [B, C, H, W]
        teacher_logits: [B, C, H, W]  same spatial size as student
        weight:         [B, H, W]     detached weight map (no grad)
        tau:            temperature
        num_classes:    1 for binary, >1 for multiclass

    Returns:
        Scalar weighted KD loss (tau^2-scaled).
    """
    if num_classes == 1:
        student_logits = torch.cat(
            [torch.zeros_like(student_logits), student_logits], dim=1
        )
        teacher_logits = torch.cat(
            [torch.zeros_like(teacher_logits), teacher_logits], dim=1
        )

    # tempered distributions
    p_s_log = F.log_softmax(student_logits / tau, dim=1)    # [B, C_eff, H, W]
    p_t = F.softmax(teacher_logits / tau, dim=1)            # [B, C_eff, H, W]

    # pixel-wise KL: KL(p_t || p_s) = sum_c p_t * (log p_t - log p_s)
    kl_map = (p_t * (p_t.log() - p_s_log)).sum(dim=1)      # [B, H, W]
    kl_map = kl_map.clamp(min=0.0)                          # numerical safety

    # weight and reduce; tau^2 restores scale (standard KD convention)
    return (weight * kl_map).mean() * (tau ** 2)


class UncertaintyWeightedKDLoss(nn.Module):
    """
    Wraps the three uncertainty-KD helpers into a single nn.Module.

    forward() returns (loss, uncertainty_map, weight_map) so the caller
    can log diagnostic statistics without recomputing them.

    Args:
        num_classes:  1 for binary, >1 for multiclass
        tau:          distillation temperature
        weight_type:  "linear" or "exp"
        beta:         exponential decay rate (exp mode only)
        eps:          log-stability epsilon for entropy computation
    """

    def __init__(
        self,
        num_classes: int = 1,
        tau: float = 4.0,
        weight_type: str = "linear",
        beta: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.tau = tau
        self.weight_type = weight_type
        self.beta = beta
        self.eps = eps
        # C_eff is 2 for binary (after expansion), otherwise num_classes
        self._c_eff = 2 if num_classes == 1 else num_classes

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ):
        """
        Args:
            student_logits: [B, C, H, W]
            teacher_logits: [B, C, H, W]

        Returns:
            loss:        scalar weighted KD loss
            uncertainty: [B, H, W] teacher entropy map (detached)
            weight:      [B, H, W] confidence weight map (detached)
        """
        uncertainty = compute_teacher_uncertainty(
            teacher_logits, tau=self.tau, eps=self.eps, num_classes=self.num_classes
        )
        weight = compute_uncertainty_weight(
            uncertainty,
            weight_type=self.weight_type,
            beta=self.beta,
            num_classes_effective=self._c_eff,
        )
        loss = compute_uncertainty_weighted_kd_loss(
            student_logits, teacher_logits, weight,
            tau=self.tau, num_classes=self.num_classes,
        )
        return loss, uncertainty, weight


class PairWiseforWholeFeatAfterPool(nn.Module):
    def __init__(self, scale, feat_ind):
        '''inter pair-wise loss from inter feature maps'''
        super(PairWiseforWholeFeatAfterPool, self).__init__()
        self.criterion = self.sim_dis_compute
        self.feat_ind = feat_ind
        self.scale = scale

    def forward(self, preds_S, preds_T):
        feat_S = preds_S[self.feat_ind]
        feat_T = preds_T[self.feat_ind]
        feat_T.detach()

        total_w, total_h = feat_T.shape[2], feat_T.shape[3]
        patch_w, patch_h = int(total_w*self.scale), int(total_h*self.scale)
        maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True) # change
        loss = self.criterion(maxpool(feat_S), maxpool(feat_T))
        return loss
    
    def similarity(self, feat):
        feat = feat.float()
        tmp = self.L2(feat).detach()
        feat = feat/tmp
        feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
        return torch.einsum('icm,icn->imn', [feat, feat])

    def sim_dis_compute(self, f_S, f_T):
        sim_err = ((self.similarity(f_T) - self.similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
        sim_dis = sim_err.sum()
        return sim_dis

class CriterionAdv(nn.Module):
    def __init__(self, adv_type):
        super(CriterionAdv, self).__init__()
        if (adv_type != 'wgan-gp') and (adv_type != 'hinge'):
            raise ValueError('adv_type should be wgan-gp or hinge')
        self.adv_loss = adv_type

    def forward(self, d_out_S, d_out_T):
        assert d_out_S[0].shape == d_out_T[0].shape,'the output dim of D with teacher and student as input differ'
        '''teacher output'''
        d_out_real = d_out_T[0]
        if self.adv_loss == 'wgan-gp':
            d_loss_real = - torch.mean(d_out_real)
        elif self.adv_loss == 'hinge':
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')

        # apply Gumbel Softmax
        '''student output'''
        d_out_fake = d_out_S[0]
        if self.adv_loss == 'wgan-gp':
            d_loss_fake = d_out_fake.mean()
        elif self.adv_loss == 'hinge':
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')
        return d_loss_real + d_loss_fake