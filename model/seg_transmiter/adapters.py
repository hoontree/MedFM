"""Lightweight adapter modules.

These modules are intentionally small. They are trained on top of frozen
TinyUSFM / SAM backbones so that the final deployable model is one backbone
+ a few hundred-K parameter adapter, with no dependence on the other model
at inference time.

Tensor shapes
-------------
ResidualConvAdapter:
    in  : [B, C_in,   H,     W]
    out : [B, n_cls,  H_out, W_out]  # residual class logits to add to base

FeatureProjector:
    in  : [B, C_src,  H_src, W_src]
    out : [B, C_dst,  H_dst, W_dst]
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvAdapter(nn.Module):
    """Tiny residual segmentation head: 1x1 -> DW 3x3 -> 1x1.

    The output is *added* to the base model's class logits, so it lives in
    logit space rather than producing a stand-alone segmentation.

    Args:
        in_channels:  Channels of the backbone feature map.
        num_classes:  Number of output classes (3 for normal/benign/malignant).
        bottleneck:   Internal width (defaults to in_channels // 2, min 16).
        upsample_to:  Optional ``(H, W)`` to bilinear-upsample the output to,
                      matching the base model's final logit resolution.
        zero_init:    Initialize last 1x1 to zero so the adapter starts as a
                      no-op (final_logits == base_logits at step 0).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        bottleneck: Optional[int] = None,
        upsample_to: Optional[Tuple[int, int]] = None,
        zero_init: bool = True,
    ):
        super().__init__()
        r = bottleneck if bottleneck is not None else max(in_channels // 2, 16)
        self.upsample_to = upsample_to

        self.proj_in = nn.Conv2d(in_channels, r, kernel_size=1, bias=True)
        self.act1 = nn.GELU()
        # Depthwise 3x3 — keeps spatial mixing cheap.
        self.dwconv = nn.Conv2d(r, r, kernel_size=3, padding=1, groups=r, bias=True)
        self.act2 = nn.GELU()
        self.proj_out = nn.Conv2d(r, num_classes, kernel_size=1, bias=True)

        if zero_init:
            # Start as a no-op so the frozen base model defines the initial output.
            nn.init.zeros_(self.proj_out.weight)
            nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor, out_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        # x: [B, C_in, H, W]
        y = self.proj_in(x)
        y = self.act1(y)
        y = self.dwconv(y)
        y = self.act2(y)
        y = self.proj_out(y)  # [B, n_cls, H, W]

        target = out_size if out_size is not None else self.upsample_to
        if target is not None and (y.shape[-2], y.shape[-1]) != tuple(target):
            y = F.interpolate(y, size=target, mode="bilinear", align_corners=False)
        return y


class FeatureProjector(nn.Module):
    """Project a feature map between TinyUSFM <-> SAM embedding spaces.

    Structure: 1x1 Conv (channel match) -> bilinear resize -> GroupNorm + GELU.
    Lightweight by design — channel projection only, no extra spatial mixing.

    Args:
        in_channels:    Source channels.
        out_channels:   Destination channels (e.g. 256 for SAM vit_b).
        out_size:       Optional ``(H, W)`` to resize to.  ``None`` keeps the
                        spatial resolution of the input.
        groups:         GroupNorm groups (defaults to ``min(32, out_channels)``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        out_size: Optional[Tuple[int, int]] = None,
        groups: Optional[int] = None,
    ):
        super().__init__()
        self.out_size = out_size
        g = groups if groups is not None else min(32, out_channels)
        # GroupNorm with at most out_channels groups, and groups must divide out_channels.
        while out_channels % g != 0 and g > 1:
            g -= 1

        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=g, num_channels=out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, H_in, W_in]
        y = self.proj(x)
        if self.out_size is not None and (y.shape[-2], y.shape[-1]) != tuple(self.out_size):
            y = F.interpolate(y, size=self.out_size, mode="bilinear", align_corners=False)
        y = self.norm(y)
        y = self.act(y)
        return y


def ConvLoRAAdapterStub(
    in_features: int,
    out_features: int,
    r: int = 4,
    lora_alpha: int = 4,
    expert_num: int = 4,
    bias: bool = True,
):
    """Return a Conv-LoRA-adapted linear projection for SAM q/v.

    This is a thin wrapper around the existing
    :class:`model.adaptation_layers.ConvLoRALinear` so callers can opt in
    to SAM image encoder adaptation without re-implementing it.

    Use :func:`model.sam_hybrid_adapter.inject_adaptation_to_linear_layer`
    to actually inject these into the SAM encoder blocks.
    """
    from model.adaptation_layers import ConvLoRALinear

    return ConvLoRALinear(
        in_features=in_features,
        out_features=out_features,
        r=r,
        lora_alpha=lora_alpha,
        merge_weights=False,
        conv_lora_expert_num=expert_num,
        bias=bias,
    )
