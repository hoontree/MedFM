"""SAM Hybrid Adapter with flexible adaptation modes.

This module provides a unified SAM model that supports various combinations of
freeze/LoRA/full-finetune for both encoder and decoder components.

Supported adaptation modes:
    - dual_lora: LoRA on both encoder and decoder
    - dual_ft: Full fine-tune on both encoder and decoder
    - encoder_conv_lora_decoder_{ft,lora,frozen}: Conv-LoRA on encoder, chosen mode on decoder
    - encoder_conv_lora_alignment_decoder_{ft,lora,frozen}: Conv-LoRA on encoder, alignment layer, chosen mode on decoder
    - encoder_lora_decoder_ft: LoRA on encoder, full fine-tune decoder
    - encoder_lora_decoder_frozen: LoRA on encoder, freeze decoder
    - encoder_ft_decoder_lora: Full fine-tune encoder, LoRA on decoder
    - encoder_ft_decoder_frozen: Full fine-tune encoder, freeze decoder
    - encoder_frozen_decoder_ft: Freeze encoder, full fine-tune decoder
    - encoder_frozen_decoder_lora: Freeze encoder, LoRA on decoder
    - encoder_frozen_decoder_frozen: Freeze both (inference only)
    - encoder_frozen_alignment_decoder_ft: Freeze encoder, alignment layer, full fine-tune decoder
    - encoder_frozen_alignment_decoder_lora: Freeze encoder, alignment layer, LoRA on decoder
    - encoder_frozen_alignment_decoder_frozen: Freeze encoder and decoder, train alignment layer only
"""

import logging
import math
import re
from pathlib import Path
from typing import List, Optional
import torch
import torch.nn as nn
from model.segment_anything import sam_model_registry
from model.segment_anything.modeling import Sam
from model.ca_sam.alignment_layer import AlignmentLayer
from model.adaptation_layers import ConvLoRALinear

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAM_PRESET_CHECKPOINTS = {
    "vit_b": "checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": "checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": "checkpoints/sam_vit_h_4b8939.pth",
}


class _LoRA_qkv(nn.Module):
    """LoRA adapter for SAM's combined qkv projection in image encoder.

    In SAM's image encoder, qkv is implemented as:
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B, N, N, 3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim :] += new_v
        return qkv


class _LoRA_qkv_proj(nn.Module):
    """LoRA adapter for separate q/k/v projections in mask decoder."""

    def __init__(self, proj: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.proj = proj
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        return self.proj(x) + self.w_b(self.w_a(x))


class _ConvLoRA_qkv(nn.Module):
    """Adapter for Conv-LoRA qkv projection in image encoder.

    ConvLoRALinear returns `(result, moe_loss)`, but the SAM encoder attention
    expects a plain tensor. This wrapper keeps interface compatibility.
    """

    def __init__(self, conv_lora_qkv: ConvLoRALinear):
        super().__init__()
        self.conv_lora_qkv = conv_lora_qkv
        self.last_moe_loss = None

    def forward(self, x):
        qkv, moe_loss = self.conv_lora_qkv(x)
        self.last_moe_loss = moe_loss
        return qkv


def create_adaptation(
    peft: str, layer: nn.Module, lora_r: int, lora_alpha: int, **kwargs
):
    """Create Conv-LoRA adaptation for a linear layer."""
    if peft != "conv_lora":
        raise NotImplementedError(
            f"Unsupported adaptation strategy '{peft}'. Only 'conv_lora' is supported here."
        )
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"Conv-LoRA can only wrap nn.Linear. Got: {type(layer)}")

    conv_lora = ConvLoRALinear(
        in_features=layer.in_features,
        out_features=layer.out_features,
        r=lora_r,
        lora_alpha=lora_alpha,
        merge_weights=False,
        conv_lora_expert_num=kwargs["conv_lora_expert_num"],
        bias=layer.bias is not None,
    )
    conv_lora.weight.data.copy_(layer.weight.data)
    if layer.bias is not None and conv_lora.bias is not None:
        conv_lora.bias.data.copy_(layer.bias.data)
    return _ConvLoRA_qkv(conv_lora)


def inject_adaptation_to_linear_layer(
    model: nn.Module,
    peft: str,
    lora_r: int,
    lora_alpha: int,
    filter: Optional[List[str]] = None,
    module_filter: Optional[List[str]] = None,
    **kwargs,
) -> List[_ConvLoRA_qkv]:
    """Inject Conv-LoRA adapters into selected linear layers."""
    injected_layers: List[_ConvLoRA_qkv] = []

    for m_name, module in dict(model.named_modules()).items():
        if module_filter and not any(
            re.match(filter_module, m_name) for filter_module in module_filter
        ):
            continue

        for c_name, layer in dict(module.named_children()).items():
            if filter and not any(
                re.match(filter_layer, c_name) for filter_layer in filter
            ):
                continue
            if not isinstance(layer, nn.Linear):
                continue

            adaptation_layer = create_adaptation(
                peft, layer, lora_r, lora_alpha, **kwargs
            )
            setattr(module, c_name, adaptation_layer)
            injected_layers.append(adaptation_layer)

    return injected_layers


def _resolve_checkpoint_for_sam_type(sam_type: str) -> str:
    """Resolve fixed SAM checkpoint from sam_type."""
    if sam_type not in SAM_PRESET_CHECKPOINTS:
        raise ValueError(
            f"Unsupported sam_type='{sam_type}'. "
            f"Available: {sorted(SAM_PRESET_CHECKPOINTS.keys())}"
        )

    ckpt = (PROJECT_ROOT / SAM_PRESET_CHECKPOINTS[sam_type]).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Checkpoint for sam_type='{sam_type}' not found: {ckpt}"
        )
    return str(ckpt)


class LoRA_Sam(nn.Module):
    """SAM model with flexible adaptation modes for encoder and decoder.

    This class provides a unified interface for applying different adaptation
    strategies (freeze, LoRA, full fine-tune) to SAM's encoder and decoder.

    Args:
        sam_model: Base SAM model instance
        r: LoRA rank (used when LoRA is applied)
        adaptation_mode: String specifying the adaptation strategy
        lora_layer: Optional list of layer indices to apply LoRA (encoder only)

    Adaptation mode format:
        - 'encoder_{encoder_mode}_decoder_{decoder_mode}'
        - 'encoder_{encoder_mode}_alignment_decoder_{decoder_mode}'
        - Special: 'dual_lora', 'dual_ft'
        - encoder_mode: 'lora', 'conv_lora', 'ft', 'frozen'
        - decoder_mode: 'lora', 'ft', 'frozen'
    """

    VALID_ENCODER_MODES = {"lora", "conv_lora", "ft", "frozen"}
    VALID_DECODER_MODES = {"lora", "ft", "frozen"}

    def __init__(
        self,
        sam_model: Sam,
        r_e: int = 4,
        r_d: int = 4,
        adaptation_mode: str = "dual_lora",
        lora_layer=None,
        alignment_num_blocks: int = 4,
        alignment_hidden_channels: int = 256,
        conv_lora_expert_num: int = 4,
    ):
        super(LoRA_Sam, self).__init__()

        self.adaptation_mode = adaptation_mode
        self.r_e = r_e
        self.r_d = r_d
        self.conv_lora_expert_num = conv_lora_expert_num

        # Parse adaptation mode
        encoder_mode, decoder_mode, use_alignment = self._parse_adaptation_mode(
            adaptation_mode
        )
        self.encoder_mode = encoder_mode
        self.decoder_mode = decoder_mode
        self.use_alignment = use_alignment

        # Alignment layer configuration
        self.alignment_num_blocks = alignment_num_blocks
        self.alignment_hidden_channels = alignment_hidden_channels

        # Setup lora layers for encoder
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.blocks)))

        # Storage for LoRA parameters (for saving/loading)
        self.w_As = []
        self.w_Bs = []
        self.self_attn_As = []
        self.self_attn_Bs = []
        self.cross_attn_ti_As = []
        self.cross_attn_ti_Bs = []
        self.cross_attn_it_As = []
        self.cross_attn_it_Bs = []
        self.encoder_conv_qkv_adapters = []
        self.encoder_conv_qkv_modules = []

        # Apply adaptation to encoder
        self._adapt_encoder(sam_model, encoder_mode)

        # Apply adaptation to decoder
        self._adapt_decoder(sam_model, decoder_mode)

        # Setup alignment layer if needed
        self.alignment_layer = None
        if self.use_alignment:
            self._setup_alignment_layer()

        # Initialize LoRA parameters
        self.reset_parameters()

        self.sam = sam_model

        # Log trainable parameters
        self._log_trainable_params()

    def _parse_adaptation_mode(self, mode: str):
        """Parse adaptation mode string into encoder mode, decoder mode, and alignment flag."""
        if mode == "dual_lora":
            return "lora", "lora", False
        elif mode == "dual_ft":
            return "ft", "ft", False
        elif mode.startswith("encoder_"):
            # Robust parsing that supports encoder_mode tokens containing underscores
            # (e.g., conv_lora).
            remainder = mode[len("encoder_") :]
            use_alignment = "_alignment_decoder_" in remainder

            if use_alignment:
                if "_alignment_decoder_" not in remainder:
                    raise ValueError(
                        f"Invalid adaptation_mode format: {mode}. "
                        "Expected: encoder_{mode}_alignment_decoder_{mode}"
                    )
                encoder_mode, decoder_mode = remainder.split("_alignment_decoder_", 1)
            else:
                if "_decoder_" not in remainder:
                    raise ValueError(
                        f"Invalid adaptation_mode format: {mode}. "
                        "Expected: encoder_{mode}_decoder_{mode}"
                    )
                encoder_mode, decoder_mode = remainder.split("_decoder_", 1)

            if encoder_mode not in self.VALID_ENCODER_MODES:
                raise ValueError(
                    f"Invalid encoder mode '{encoder_mode}' in adaptation_mode='{mode}'. "
                    f"Valid encoder modes: {self.VALID_ENCODER_MODES}"
                )
            if decoder_mode not in self.VALID_DECODER_MODES:
                raise ValueError(
                    f"Invalid decoder mode '{decoder_mode}' in adaptation_mode='{mode}'. "
                    f"Valid decoder modes: {self.VALID_DECODER_MODES}"
                )

            # Conv-LoRA's implementation here is designed for encoder token maps (HxW).
            # Decoder query lengths are non-square; applying Conv-LoRA there would require
            # additional reshaping logic and is intentionally disabled for safety.
            if decoder_mode == "conv_lora":
                raise ValueError(
                    "Decoder Conv-LoRA is not supported in this SAM hybrid adapter. "
                    "Use encoder Conv-LoRA with decoder in {lora, ft, frozen}."
                )

            return encoder_mode, decoder_mode, use_alignment
        else:
            raise ValueError(
                f"Invalid adaptation_mode: {mode}. "
                "Examples: dual_lora, dual_ft, encoder_conv_lora_decoder_ft, "
                "encoder_frozen_alignment_decoder_lora"
            )

    def _setup_alignment_layer(self):
        """Setup alignment layer between encoder and decoder."""
        # SAM encoder output is always 256 channels (prompt_embed_dim)
        encoder_output_dim = 256
        self.alignment_layer = AlignmentLayer(
            in_channels=encoder_output_dim,
            hidden_channels=self.alignment_hidden_channels,
            num_blocks=self.alignment_num_blocks,
        )
        LOGGER.info(
            "[LoRA_Sam] Alignment Layer: %s parameters",
            f"{self.alignment_layer.get_num_params():,}",
        )

    def _adapt_encoder(self, sam_model: Sam, mode: str):
        """Apply adaptation strategy to the image encoder."""
        r = self.r_e
        if mode == "frozen":
            # Freeze all encoder parameters
            for param in sam_model.image_encoder.parameters():
                param.requires_grad = False

        elif mode == "ft":
            # Full fine-tune: all parameters trainable
            for param in sam_model.image_encoder.parameters():
                param.requires_grad = True

        elif mode == "lora":
            # Freeze base parameters, add LoRA adapters
            for param in sam_model.image_encoder.parameters():
                param.requires_grad = False

            # Apply LoRA to attention layers
            for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
                if t_layer_i not in self.lora_layer:
                    continue

                w_qkv_linear = blk.attn.qkv
                self.dim = w_qkv_linear.in_features

                w_a_linear_q = nn.Linear(self.dim, r, bias=False)
                w_b_linear_q = nn.Linear(r, self.dim, bias=False)
                w_a_linear_v = nn.Linear(self.dim, r, bias=False)
                w_b_linear_v = nn.Linear(r, self.dim, bias=False)

                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)

                blk.attn.qkv = _LoRA_qkv(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                )
        elif mode == "conv_lora":
            for param in sam_model.image_encoder.parameters():
                param.requires_grad = False

            if self.conv_lora_expert_num is None or self.conv_lora_expert_num < 1:
                raise ValueError(
                    f"conv_lora_expert_num must be >= 1 for encoder conv_lora mode. "
                    f"Got: {self.conv_lora_expert_num}"
                )

            module_filter = [
                r"^blocks\.(%s)\.attn$" % "|".join(map(str, self.lora_layer))
            ]
            self.encoder_conv_qkv_adapters = inject_adaptation_to_linear_layer(
                model=sam_model.image_encoder,
                peft="conv_lora",
                lora_r=r,
                lora_alpha=r,
                filter=[r"^qkv$"],
                module_filter=module_filter,
                conv_lora_expert_num=self.conv_lora_expert_num,
            )
            self.encoder_conv_qkv_modules = [
                adapter.conv_lora_qkv for adapter in self.encoder_conv_qkv_adapters
            ]
            if len(self.encoder_conv_qkv_modules) == 0:
                raise RuntimeError(
                    "Conv-LoRA injection failed: no encoder qkv layers were adapted."
                )

    def _adapt_decoder(self, sam_model: Sam, mode: str):
        """Apply adaptation strategy to the mask decoder."""
        r = self.r_d
        decoder_transformer = sam_model.mask_decoder.transformer

        if mode == "frozen":
            # Freeze all decoder parameters
            for param in sam_model.mask_decoder.parameters():
                param.requires_grad = False

        elif mode == "ft":
            # Full fine-tune: all parameters trainable
            for param in sam_model.mask_decoder.parameters():
                param.requires_grad = True

        elif mode == "lora":
            # Freeze base transformer parameters, add LoRA adapters
            for param in decoder_transformer.parameters():
                param.requires_grad = False

            # Keep non-transformer parts trainable (output tokens, hypernetworks, etc.)
            for name, param in sam_model.mask_decoder.named_parameters():
                if "transformer" not in name:
                    param.requires_grad = True

            # Apply LoRA to transformer layers
            for blk in decoder_transformer.layers:
                # Self attention
                self._apply_lora_to_attention(
                    blk.self_attn, r, self.self_attn_As, self.self_attn_Bs
                )

                # Cross attention: token to image
                self._apply_lora_to_attention(
                    blk.cross_attn_token_to_image,
                    r,
                    self.cross_attn_ti_As,
                    self.cross_attn_ti_Bs,
                )

                # Cross attention: image to token
                self._apply_lora_to_attention(
                    blk.cross_attn_image_to_token,
                    r,
                    self.cross_attn_it_As,
                    self.cross_attn_it_Bs,
                )

            # Final attention token to image
            block = decoder_transformer.final_attn_token_to_image
            fa_ti_q_proj = block.q_proj
            fa_ti_v_proj = block.v_proj
            in_dim, out_dim = block.embedding_dim, block.internal_dim

            self.fa_ti_q_proj_A = nn.Linear(in_dim, r, bias=False)
            self.fa_ti_q_proj_B = nn.Linear(r, out_dim, bias=False)
            self.fa_ti_v_proj_A = nn.Linear(in_dim, r, bias=False)
            self.fa_ti_v_proj_B = nn.Linear(r, out_dim, bias=False)

            block.q_proj = _LoRA_qkv_proj(
                fa_ti_q_proj, self.fa_ti_q_proj_A, self.fa_ti_q_proj_B
            )
            block.v_proj = _LoRA_qkv_proj(
                fa_ti_v_proj, self.fa_ti_v_proj_A, self.fa_ti_v_proj_B
            )

    def _apply_lora_to_attention(self, attn_module, r: int, a_list: list, b_list: list):
        """Apply LoRA to q and v projections of an attention module."""
        q_proj = attn_module.q_proj
        v_proj = attn_module.v_proj
        input_dim = attn_module.embedding_dim
        output_dim = attn_module.internal_dim

        w_a_q = nn.Linear(input_dim, r, bias=False)
        w_b_q = nn.Linear(r, output_dim, bias=False)
        w_a_v = nn.Linear(input_dim, r, bias=False)
        w_b_v = nn.Linear(r, output_dim, bias=False)

        a_list.append(w_a_q)
        b_list.append(w_b_q)
        a_list.append(w_a_v)
        b_list.append(w_b_v)

        attn_module.q_proj = _LoRA_qkv_proj(q_proj, w_a_q, w_b_q)
        attn_module.v_proj = _LoRA_qkv_proj(v_proj, w_a_v, w_b_v)

    def _log_trainable_params(self):
        """Log the number of trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        LOGGER.info("[LoRA_Sam] Adaptation mode: %s", self.adaptation_mode)
        LOGGER.info(
            "[LoRA_Sam] Encoder mode: %s, Decoder mode: %s",
            self.encoder_mode,
            self.decoder_mode,
        )
        if self.encoder_mode == "conv_lora":
            LOGGER.info(
                "[LoRA_Sam] Conv-LoRA (encoder): experts=%s",
                self.conv_lora_expert_num,
            )
        if self.use_alignment:
            LOGGER.info(
                "[LoRA_Sam] Alignment Layer: enabled (blocks=%s)",
                self.alignment_num_blocks,
            )
        LOGGER.info("[LoRA_Sam] Total parameters: %s", f"{total_params:,}")
        LOGGER.info(
            "[LoRA_Sam] Trainable parameters: %s (%.2f%%)",
            f"{trainable_params:,}",
            100 * trainable_params / total_params,
        )

    def reset_parameters(self) -> None:
        """Initialize LoRA parameters with Kaiming uniform for A and zeros for B."""
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)
        for w_A in self.self_attn_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.self_attn_Bs:
            nn.init.zeros_(w_B.weight)
        for w_A in self.cross_attn_ti_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.cross_attn_ti_Bs:
            nn.init.zeros_(w_B.weight)
        for w_A in self.cross_attn_it_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.cross_attn_it_Bs:
            nn.init.zeros_(w_B.weight)

        # Final attention (only if decoder uses LoRA)
        if hasattr(self, "fa_ti_q_proj_A"):
            nn.init.kaiming_uniform_(self.fa_ti_q_proj_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.fa_ti_q_proj_B.weight)
            nn.init.kaiming_uniform_(self.fa_ti_v_proj_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.fa_ti_v_proj_B.weight)

    def save_lora_parameters(self, filename: str) -> None:
        """Save LoRA parameters and other trainable components.

        Args:
            filename: Path to save the checkpoint (.pt or .pth)
        """
        assert str(filename).endswith(".pt") or str(filename).endswith(".pth")

        merged_dict = {"adaptation_mode": self.adaptation_mode}

        # Save encoder LoRA if applicable
        if self.encoder_mode == "lora":
            num_layer = len(self.w_As)
            merged_dict.update(
                {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
            )
            merged_dict.update(
                {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
            )
        elif self.encoder_mode == "conv_lora":
            for i, mod in enumerate(self.encoder_conv_qkv_modules):
                merged_dict[f"conv_w_a_{i:03d}"] = mod.lora_A
                merged_dict[f"conv_w_b_{i:03d}"] = mod.lora_B
                merged_dict[f"conv_gate_w_{i:03d}"] = mod.lora_moe_gating.w_gate
                merged_dict[f"conv_noise_w_{i:03d}"] = mod.lora_moe_gating.w_noise
                for j, expert in enumerate(mod.lora_moe_experts):
                    expert_conv = expert[0]
                    merged_dict[f"conv_exp_w_{i:03d}_{j:02d}"] = expert_conv.weight
                    merged_dict[f"conv_exp_b_{i:03d}_{j:02d}"] = expert_conv.bias
            merged_dict["conv_lora_expert_num"] = self.conv_lora_expert_num

        # Save decoder LoRA if applicable
        if self.decoder_mode == "lora":
            merged_dict.update(
                {
                    f"sa_a_{i:03d}": self.self_attn_As[i].weight
                    for i in range(len(self.self_attn_As))
                }
            )
            merged_dict.update(
                {
                    f"sa_b_{i:03d}": self.self_attn_Bs[i].weight
                    for i in range(len(self.self_attn_Bs))
                }
            )
            merged_dict.update(
                {
                    f"cti_a_{i:03d}": self.cross_attn_ti_As[i].weight
                    for i in range(len(self.cross_attn_ti_As))
                }
            )
            merged_dict.update(
                {
                    f"cti_b_{i:03d}": self.cross_attn_ti_Bs[i].weight
                    for i in range(len(self.cross_attn_ti_Bs))
                }
            )
            merged_dict.update(
                {
                    f"cit_a_{i:03d}": self.cross_attn_it_As[i].weight
                    for i in range(len(self.cross_attn_it_As))
                }
            )
            merged_dict.update(
                {
                    f"cit_b_{i:03d}": self.cross_attn_it_Bs[i].weight
                    for i in range(len(self.cross_attn_it_Bs))
                }
            )
            merged_dict.update(
                {
                    "fati_qa": self.fa_ti_q_proj_A.weight,
                    "fati_qb": self.fa_ti_q_proj_B.weight,
                    "fati_va": self.fa_ti_v_proj_A.weight,
                    "fati_vb": self.fa_ti_v_proj_B.weight,
                }
            )

        # Save alignment layer if applicable
        if self.use_alignment and self.alignment_layer is not None:
            merged_dict["alignment_layer"] = self.alignment_layer.state_dict()
            merged_dict["alignment_num_blocks"] = self.alignment_num_blocks
            merged_dict["alignment_hidden_channels"] = self.alignment_hidden_channels

        # Save prompt encoder and mask decoder (non-transformer) state
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(
            self.sam, torch.nn.parallel.DistributedDataParallel
        ):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()

        for key, value in state_dict.items():
            if "prompt_encoder" in key:
                merged_dict[key] = value
            if "mask_decoder" in key and "transformer" not in key:
                merged_dict[key] = value

        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        """Load LoRA parameters and other trainable components.

        Args:
            filename: Path to the checkpoint (.pt or .pth)

        Supports two checkpoint formats:
            1. Legacy format with __format__, __mode__, __sam_state_dict__ keys
            2. Current flat format with direct parameter keys
        """
        assert str(filename).endswith(".pt") or str(filename).endswith(".pth")

        state_dict = torch.load(filename, map_location="cpu")

        # Handle legacy checkpoint format (hybrid_lora_v1)
        if (
            "__format__" in state_dict
            and state_dict.get("__format__") == "hybrid_lora_v1"
        ):
            self._load_legacy_checkpoint(state_dict)
            return

        # Load encoder LoRA if applicable
        if self.encoder_mode == "lora":
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                if saved_key in state_dict:
                    w_A_linear.weight.data.copy_(state_dict[saved_key])

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                if saved_key in state_dict:
                    w_B_linear.weight.data.copy_(state_dict[saved_key])
        elif self.encoder_mode == "conv_lora":
            for i, mod in enumerate(self.encoder_conv_qkv_modules):
                key_a = f"conv_w_a_{i:03d}"
                key_b = f"conv_w_b_{i:03d}"
                if key_a in state_dict:
                    mod.lora_A.data.copy_(state_dict[key_a])
                if key_b in state_dict:
                    mod.lora_B.data.copy_(state_dict[key_b])

                key_gate = f"conv_gate_w_{i:03d}"
                key_noise = f"conv_noise_w_{i:03d}"
                if key_gate in state_dict:
                    mod.lora_moe_gating.w_gate.data.copy_(state_dict[key_gate])
                if key_noise in state_dict:
                    mod.lora_moe_gating.w_noise.data.copy_(state_dict[key_noise])

                for j, expert in enumerate(mod.lora_moe_experts):
                    expert_conv = expert[0]
                    key_exp_w = f"conv_exp_w_{i:03d}_{j:02d}"
                    key_exp_b = f"conv_exp_b_{i:03d}_{j:02d}"
                    if key_exp_w in state_dict:
                        expert_conv.weight.data.copy_(state_dict[key_exp_w])
                    if key_exp_b in state_dict:
                        expert_conv.bias.data.copy_(state_dict[key_exp_b])

        # Load decoder LoRA if applicable
        if self.decoder_mode == "lora":
            for i, sa_A_linear in enumerate(self.self_attn_As):
                saved_key = f"sa_a_{i:03d}"
                if saved_key in state_dict:
                    sa_A_linear.weight.data.copy_(state_dict[saved_key])

            for i, sa_B_linear in enumerate(self.self_attn_Bs):
                saved_key = f"sa_b_{i:03d}"
                if saved_key in state_dict:
                    sa_B_linear.weight.data.copy_(state_dict[saved_key])

            for i, cti_a_linear in enumerate(self.cross_attn_ti_As):
                saved_key = f"cti_a_{i:03d}"
                if saved_key in state_dict:
                    cti_a_linear.weight.data.copy_(state_dict[saved_key])

            for i, cti_b_linear in enumerate(self.cross_attn_ti_Bs):
                saved_key = f"cti_b_{i:03d}"
                if saved_key in state_dict:
                    cti_b_linear.weight.data.copy_(state_dict[saved_key])

            for i, cit_a_linear in enumerate(self.cross_attn_it_As):
                saved_key = f"cit_a_{i:03d}"
                if saved_key in state_dict:
                    cit_a_linear.weight.data.copy_(state_dict[saved_key])

            for i, cit_b_linear in enumerate(self.cross_attn_it_Bs):
                saved_key = f"cit_b_{i:03d}"
                if saved_key in state_dict:
                    cit_b_linear.weight.data.copy_(state_dict[saved_key])

            if "fati_qa" in state_dict:
                self.fa_ti_q_proj_A.weight.data.copy_(state_dict["fati_qa"])
                self.fa_ti_q_proj_B.weight.data.copy_(state_dict["fati_qb"])
                self.fa_ti_v_proj_A.weight.data.copy_(state_dict["fati_va"])
                self.fa_ti_v_proj_B.weight.data.copy_(state_dict["fati_vb"])

        # Load alignment layer if applicable
        if self.use_alignment and self.alignment_layer is not None:
            if "alignment_layer" in state_dict:
                self.alignment_layer.load_state_dict(state_dict["alignment_layer"])
                LOGGER.info("[LoRA_Sam] Loaded alignment layer from checkpoint")

        # Load prompt encoder and mask decoder state
        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        prompt_encoder_keys = [k for k in sam_keys if "prompt_encoder" in k]
        for k in prompt_encoder_keys:
            if k in state_dict:
                sam_dict[k] = state_dict[k]

        mask_decoder_keys = [
            k for k in sam_keys if "mask_decoder" in k and "transformer" not in k
        ]
        for k in mask_decoder_keys:
            if k in state_dict:
                sam_dict[k] = state_dict[k]

        self.sam.load_state_dict(sam_dict)

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint. Supports both LoRA and regular SAM checkpoints."""
        try:
            # First try loading as LoRA parameters
            self.load_lora_parameters(checkpoint_path)
            LOGGER.info(
                "[LoRA_Sam] Loaded LoRA/Adapter parameters from %s", checkpoint_path
            )
        except Exception as e:
            # Fallback to regular SAM load_state_dict if it's a full model checkpoint
            try:
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                self.load_state_dict(state_dict)
                LOGGER.info(
                    "[LoRA_Sam] Loaded full model state_dict from %s", checkpoint_path
                )
            except Exception as e2:
                LOGGER.warning("Failed to load checkpoint %s", checkpoint_path)
                LOGGER.warning("LoRA load error: %s", e)
                LOGGER.warning("Full model load error: %s", e2)
                raise RuntimeError(
                    f"Failed to load checkpoint: {checkpoint_path}"
                ) from e2

    def _load_legacy_checkpoint(self, state_dict: dict) -> None:
        """Load checkpoint in legacy hybrid_lora_v1 format.

        Legacy format contains:
            - __format__: 'hybrid_lora_v1'
            - __mode__: adaptation mode string
            - __sam_state_dict__: full SAM state dict (for decoder_ft modes)
            - LoRA keys (w_a_*, w_b_*, sa_*, cti_*, cit_*, fati_*)
            - prompt_encoder and mask_decoder keys
        """
        # Load full SAM state dict if present (used for decoder_ft modes)
        if "__sam_state_dict__" in state_dict:
            sam_state = state_dict["__sam_state_dict__"]
            current_sam_dict = self.sam.state_dict()

            # Only load prompt_encoder and mask_decoder (non-transformer) parts
            for k, v in sam_state.items():
                if "prompt_encoder" in k:
                    current_sam_dict[k] = v
                elif "mask_decoder" in k and "transformer" not in k:
                    current_sam_dict[k] = v
                # For decoder_ft mode, also load transformer weights
                elif self.decoder_mode == "ft" and "mask_decoder" in k:
                    current_sam_dict[k] = v

            self.sam.load_state_dict(current_sam_dict)

        # Load encoder LoRA if applicable
        if self.encoder_mode == "lora":
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                if saved_key in state_dict:
                    w_A_linear.weight.data.copy_(state_dict[saved_key])

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                if saved_key in state_dict:
                    w_B_linear.weight.data.copy_(state_dict[saved_key])

        # Load decoder LoRA if applicable
        if self.decoder_mode == "lora":
            for i, sa_A_linear in enumerate(self.self_attn_As):
                saved_key = f"sa_a_{i:03d}"
                if saved_key in state_dict:
                    sa_A_linear.weight.data.copy_(state_dict[saved_key])

            for i, sa_B_linear in enumerate(self.self_attn_Bs):
                saved_key = f"sa_b_{i:03d}"
                if saved_key in state_dict:
                    sa_B_linear.weight.data.copy_(state_dict[saved_key])

            for i, cti_a_linear in enumerate(self.cross_attn_ti_As):
                saved_key = f"cti_a_{i:03d}"
                if saved_key in state_dict:
                    cti_a_linear.weight.data.copy_(state_dict[saved_key])

            for i, cti_b_linear in enumerate(self.cross_attn_ti_Bs):
                saved_key = f"cti_b_{i:03d}"
                if saved_key in state_dict:
                    cti_b_linear.weight.data.copy_(state_dict[saved_key])

            for i, cit_a_linear in enumerate(self.cross_attn_it_As):
                saved_key = f"cit_a_{i:03d}"
                if saved_key in state_dict:
                    cit_a_linear.weight.data.copy_(state_dict[saved_key])

            for i, cit_b_linear in enumerate(self.cross_attn_it_Bs):
                saved_key = f"cit_b_{i:03d}"
                if saved_key in state_dict:
                    cit_b_linear.weight.data.copy_(state_dict[saved_key])

            if "fati_qa" in state_dict:
                self.fa_ti_q_proj_A.weight.data.copy_(state_dict["fati_qa"])
                self.fa_ti_q_proj_B.weight.data.copy_(state_dict["fati_qb"])
                self.fa_ti_v_proj_A.weight.data.copy_(state_dict["fati_va"])
                self.fa_ti_v_proj_B.weight.data.copy_(state_dict["fati_vb"])

    def forward(self, batched_input, multimask_output, image_size):
        """Forward pass through the SAM model.

        If alignment layer is enabled, applies it between encoder and decoder.
        """
        if self.use_alignment and self.alignment_layer is not None:
            return self._forward_with_alignment(
                batched_input, multimask_output, image_size
            )
        else:
            outputs = self.sam(batched_input, multimask_output, image_size)
            return self._attach_moe_loss(outputs)

    def _get_conv_lora_moe_loss(self):
        """Aggregate Conv-LoRA MoE losses from injected encoder qkv adapters."""
        if self.encoder_mode != "conv_lora":
            return None
        if len(self.encoder_conv_qkv_adapters) == 0:
            return None

        losses = []
        for adapter in self.encoder_conv_qkv_adapters:
            if adapter.last_moe_loss is not None:
                losses.append(adapter.last_moe_loss)
        if len(losses) == 0:
            return None
        return torch.stack(losses).sum()

    def _attach_moe_loss(self, outputs):
        """Attach aggregated moe_loss to outputs for training-time loss composition."""
        moe_loss = self._get_conv_lora_moe_loss()
        if moe_loss is None:
            return outputs

        if isinstance(outputs, dict):
            outputs["moe_loss"] = moe_loss
        elif isinstance(outputs, list):
            for item in outputs:
                if isinstance(item, dict):
                    item["moe_loss"] = moe_loss
        return outputs

    def _forward_with_alignment(self, batched_input, multimask_output, image_size):
        """Forward pass with alignment layer between encoder and decoder."""
        if isinstance(batched_input, list):
            return self._forward_test_with_alignment(batched_input, multimask_output)
        else:
            return self._forward_train_with_alignment(
                batched_input, multimask_output, image_size
            )

    def _forward_train_with_alignment(
        self, batched_input, multimask_output, image_size
    ):
        """Training forward pass with alignment layer."""
        input_images = self.sam.preprocess(batched_input)
        image_embeddings = self.sam.image_encoder(input_images)

        # Apply alignment layer
        image_embeddings = self.alignment_layer(image_embeddings)

        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None, boxes=None, masks=None
        )
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        masks = self.sam.postprocess_masks(
            low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size),
        )
        outputs = {
            "masks": masks,
            "iou_predictions": iou_predictions,
            "low_res_logits": low_res_masks,
            "image_embeddings": image_embeddings,
        }
        return self._attach_moe_loss(outputs)

    @torch.no_grad()
    def _forward_test_with_alignment(self, batched_input, multimask_output):
        """Test forward pass with alignment layer."""
        input_images = torch.stack(
            [self.sam.preprocess(x["image"]) for x in batched_input], dim=0
        )
        image_embeddings = self.sam.image_encoder(input_images)

        # Apply alignment layer
        image_embeddings = self.alignment_layer(image_embeddings)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.sam.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.sam.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return self._attach_moe_loss(outputs)


def build_sam_hybrid(
    sam_type="vit_b",
    img_size=224,
    num_classes=2,
    pixel_mean=[0, 0, 0],
    pixel_std=[1, 1, 1],
    r_e=4,
    r_d=4,
    adaptation_mode="dual_lora",
    lora_layer=None,
    alignment_num_blocks=4,
    alignment_hidden_channels=256,
    conv_lora_expert_num=4,
    **kwargs,
):
    """Factory function for Hydra instantiation."""

    resolved_checkpoint = _resolve_checkpoint_for_sam_type(sam_type)
    LOGGER.info(
        "[build_sam_hybrid] sam_type=%s, checkpoint=%s",
        sam_type,
        resolved_checkpoint,
    )

    # 1. Build base SAM model
    sam, _ = sam_model_registry[sam_type](
        image_size=img_size,
        num_classes=num_classes,
        checkpoint=resolved_checkpoint,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )

    # 2. Wrap with LoRA_Sam
    model = LoRA_Sam(
        sam,
        r_e=r_e,
        r_d=r_d,
        adaptation_mode=adaptation_mode,
        lora_layer=lora_layer,
        alignment_num_blocks=alignment_num_blocks,
        alignment_hidden_channels=alignment_hidden_channels,
        conv_lora_expert_num=conv_lora_expert_num,
    )
    model.sam_type = sam_type
    model.resolved_sam_checkpoint = resolved_checkpoint
    model.pretrained_weight_name = "fixed_by_sam_type"
    model.pretrained_weight_source = "sam_type"

    return model
