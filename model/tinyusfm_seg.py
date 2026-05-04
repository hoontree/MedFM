import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import timm.models.vision_transformer
from timm.models.vision_transformer import DropPath, Mlp, Attention as BaseAttn

from model.segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from model.segment_anything.modeling.common import LayerNorm2d


class Attention(BaseAttn):
    def __init__(self, *args, **kwargs):
        super(Attention, self).__init__(*args, **kwargs)
        self.identity = nn.Identity()

    def forward(self, x, return_latent=False):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.identity(attn)
        if return_latent:
            return attn
        attn = attn.softmax(dim=-1)
        self.last_attn = attn  # Store for visualization
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, return_latent=False):
        if return_latent:
            x = self.attn(self.norm1(x), return_latent=return_latent)
            return x
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling and distillation"""

    def __init__(
        self,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        act_layer=None,
        global_pool=False,
        distilled=False,
        **kwargs,
    ):
        super(VisionTransformer, self).__init__(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            act_layer=act_layer,
            **kwargs,
        )
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )
        self.num_heads = num_heads
        self.depth = depth
        self.global_pool = global_pool
        self.distilled = distilled
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm
            if self.distilled:
                self.dist_norm = norm_layer(embed_dim)
                if self.num_classes > 0:
                    self.head_dist = nn.Linear(self.embed_dim, self.num_classes)
                else:
                    self.head_dist = nn.Identity()
        self.init_weights("")

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            if self.distilled:
                x_dist, x = x[:, 0, :], x[:, 1:, :].mean(dim=1)
                return self.fc_norm(x), self.dist_norm(x_dist)
            else:
                x = x[:, 1:, :].mean(dim=1)
                return self.fc_norm(x)
        else:
            x = self.norm(x)
            return x[:, 0]


class MAEBackbone(VisionTransformer):
    def __init__(
        self,
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=192,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=(3, 5, 7, 11),
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_eval=False,
        init_values=None,
        **kwargs,
    ):

        super().__init__(
            img_size=img_size[0] if isinstance(img_size, tuple) else img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dims,
            depth=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            global_pool=False,
            distilled=False,
            num_classes=0,
            **kwargs,
        )

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dims = embed_dims
        self.out_indices = out_indices
        self.norm_eval = norm_eval

    def forward(self, x):
        B, C, H, W = x.shape
        # Patch embedding
        x = self.patch_embed(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Extract features at specified layers
        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                # Remove CLS token and reshape to spatial format
                feat = x[:, 1:, :]  # Remove CLS token
                H_out = H // self.patch_size
                W_out = W // self.patch_size
                feat = feat.transpose(1, 2).reshape(B, self.embed_dims, H_out, W_out)
                features.append(feat)

        return features

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()


class Feature2Pyramid_Scale(nn.Module):
    """Feature2Pyramid with scale"""

    def __init__(self, embed_dim=192, rescales=[4, 2, 1, 0.5], scale_factor=0.25):
        super().__init__()
        self.embed_dim = embed_dim
        self.rescales = rescales
        self.scale_factor = scale_factor

        out_channels = int(embed_dim * scale_factor)

        # Create projection layers for each scale
        self.projections = nn.ModuleList(
            [nn.Conv2d(embed_dim, out_channels, kernel_size=1) for _ in rescales]
        )

    def forward(self, features):
        outputs = []

        for i, (feat, rescale) in enumerate(zip(features, self.rescales)):
            # Project to 256 channels
            proj_feat = self.projections[i](feat)

            # Scale the feature map
            if rescale != 1.0:
                proj_feat = F.interpolate(
                    proj_feat,
                    scale_factor=rescale,
                    mode="bilinear",
                    align_corners=False,
                )

            outputs.append(proj_feat)

        return outputs


class FPNHead(nn.Module):
    """FPN Decode Head"""

    def __init__(
        self,
        in_channels=[48, 48, 48, 48],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=1,
        align_corners=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.in_index = in_index
        self.feature_strides = feature_strides
        self.channels = channels
        self.dropout_ratio = dropout_ratio
        self.num_classes = num_classes
        self.align_corners = align_corners

        # Lateral convolutions
        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, channels, kernel_size=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )

        # FPN convolutions
        self.fpn_convs = nn.ModuleList()
        for _ in range(len(in_channels)):
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Final classifier
        # binary: 1 output channel (sigmoid); multiclass: num_classes channels (softmax)
        out_ch = 1 if num_classes == 1 else num_classes
        self.conv_seg = nn.Conv2d(channels, out_ch, kernel_size=1)

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def forward(self, features):
        # Apply lateral convolutions
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(features[self.in_index[i]]))

        # Build FPN (top-down pathway)
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

        # Apply FPN convolutions
        fpn_outs = []
        for i, fpn_conv in enumerate(self.fpn_convs):
            fpn_outs.append(fpn_conv(laterals[i]))

        # Fuse all levels
        target_size = fpn_outs[0].shape[2:]
        fused = fpn_outs[0]

        for i in range(1, len(fpn_outs)):
            upsampled = F.interpolate(
                fpn_outs[i],
                size=target_size,
                mode="bilinear",
                align_corners=self.align_corners,
            )
            fused = fused + upsampled

        # Apply dropout
        if self.dropout is not None:
            fused = self.dropout(fused)

        # Final classification
        output = self.conv_seg(fused)

        return output


class SegmentationModel(nn.Module):
    """Complete segmentation model using the provided VisionTransformer.

    Args:
        decoder_type (str): Which decoder branch to use at runtime.
            ``"fpn"``  – the original Feature2Pyramid_Scale → FPNHead pipeline
                         (default, fully backward-compatible).
            ``"sam_mask"`` – experimental branch using SAM PromptEncoder +
                         MaskDecoder with prompt-free inputs.
        sam_transformer_dim (int): Channel width for the SAM decoder
            (transformer_dim).  256 matches SAM-vit_b pretrained weights.
    """

    def __init__(
        self,
        num_classes,
        checkpoint=None,
        use_alignment=False,
        alignment_out_channels=256,
        decoder_type="fpn",
        sam_transformer_dim=256,
        **kwargs,
    ):
        super().__init__()

        self.checkpoint_path = checkpoint
        self.decoder_type = decoder_type

        self.backbone = MAEBackbone(
            img_size=(224, 224),
            patch_size=16,
            in_channels=3,
            embed_dims=192,
            num_layers=12,
            num_heads=12,
            mlp_ratio=4,
            out_indices=(3, 5, 7, 11),
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_eval=False,
        )

        # Neck (always instantiated to keep checkpoint compatibility)
        self.neck = Feature2Pyramid_Scale(
            embed_dim=192, rescales=[4, 2, 1, 0.5], scale_factor=0.25
        )

        # Decode head (always instantiated to keep checkpoint compatibility)
        self.decode_head = FPNHead(
            in_channels=[48, 48, 48, 48],
            in_index=[0, 1, 2, 3],
            feature_strides=[4, 8, 16, 32],
            channels=128,
            dropout_ratio=0.1,
            num_classes=num_classes,
            align_corners=False,
        )

        # Student alignment layer (Proj/Adapter)
        self.use_alignment = use_alignment
        self.alignment_out_channels = alignment_out_channels

        if self.use_alignment:
            in_channels = int(192 * 0.25)  # matches neck output channels
            self.alignment_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels, self.alignment_out_channels, kernel_size=1, bias=False
                ),
                nn.BatchNorm2d(self.alignment_out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.alignment_layer = None

        # ------------------------------------------------------------------ #
        # SAM-mask decoder branch (experimental)                             #
        # ------------------------------------------------------------------ #
        if decoder_type == "sam_mask":
            num_out_feats = len(self.backbone.out_indices)   # 4
            fused_in_ch = num_out_feats * self.backbone.embed_dims  # 4*192=768

            # Fuse all backbone feature maps (same spatial size) into one map
            # at the SAM decoder width.  Only this conv + norm are trained from
            # scratch; SAM decoder weights load without name conflicts.
            self.sam_feature_proj = nn.Sequential(
                nn.Conv2d(fused_in_ch, sam_transformer_dim, kernel_size=1, bias=False),
                LayerNorm2d(sam_transformer_dim),
                nn.GELU(),
            )

            # Derive SAM prompt-encoder grid dimensions from backbone geometry
            img_h, img_w = (
                self.backbone.img_size
                if isinstance(self.backbone.img_size, (tuple, list))
                else (self.backbone.img_size, self.backbone.img_size)
            )
            h_tok = img_h // self.backbone.patch_size   # 14 for 224/16
            w_tok = img_w // self.backbone.patch_size   # 14 for 224/16

            self.prompt_encoder = PromptEncoder(
                embed_dim=sam_transformer_dim,
                image_embedding_size=(h_tok, w_tok),
                input_image_size=(img_h, img_w),
                mask_in_chans=16,
            )

            self.mask_decoder = MaskDecoder(
                transformer_dim=sam_transformer_dim,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=sam_transformer_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                num_multimask_outputs=3,
                iou_head_depth=3,
                iou_head_hidden_dim=sam_transformer_dim,
            )

            # Final channel projection to match the num_classes expected by
            # the training loop (SAM mask decoder always outputs 1 channel in
            # single-mask mode).
            self.sam_cls_conv = nn.Conv2d(1, num_classes, kernel_size=1)
        else:
            self.sam_feature_proj = None
            self.prompt_encoder = None
            self.mask_decoder = None
            self.sam_cls_conv = None

        import os
        if self.checkpoint_path:
            self.load_finetuned(self.checkpoint_path)
        else:
            default_pretrained = "checkpoints/TinyUSFM.pt"
            if os.path.exists(default_pretrained):
                self.load_pretrained(default_pretrained)

    def forward(self, x, return_features=False):
        # Extract features from backbone
        features = self.backbone(x)

        if self.decoder_type == "sam_mask":
            return self._forward_sam_mask(x, features, return_features)

        # ---- FPN path (original, fully intact) ----
        # Transform features through neck
        neck_features = self.neck(features)

        # Decode to segmentation map
        seg_logits = self.decode_head(neck_features)

        # Upsample to input size
        seg_logits = F.interpolate(
            seg_logits, size=x.shape[2:], mode="bilinear", align_corners=False
        )

        if return_features:
            # Return seg_logits and the 3rd scale feature (14x14 for 224 input)
            # neck_features[2] corresponds to rescale=1.0, matching ViT output size
            feat = neck_features[2]
            # if self.use_alignment:
            #     feat = self.alignment_layer(feat)
            return seg_logits, feat

        return seg_logits

    def _forward_sam_mask(self, x, features, return_features=False):
        """SAM-style prompt-free decoder branch.

        Uses the PromptEncoder (with all-None prompts) and MaskDecoder from SAM.
        The forward pass is written explicitly at batch level to avoid the
        ``repeat_interleave`` in ``MaskDecoder.predict_masks`` which assumes one
        image at a time and one prompt set replicated across the batch dimension.

        Concretely, for prompt-free batched inference we use:
            src     = image_embedding + dense_prompt_embeddings   (broadcast)
            pos_src = image_pe.expand(B, -1, -1, -1)
        """
        # 1. Fuse all backbone feature maps (same spatial size H_tok x W_tok)
        fused = torch.cat(features, dim=1)          # (B, 4*192=768, H_tok, W_tok)
        image_embedding = self.sam_feature_proj(fused)  # (B, transformer_dim=256, H_tok, W_tok)

        B, C, H_tok, W_tok = image_embedding.shape

        # 2. Get prompt-free embeddings from PromptEncoder (batch size = 1)
        #    sparse_pe: (1, 0, C)   dense_pe: (1, C, H_tok, W_tok)
        _sparse_pe, dense_pe = self.prompt_encoder(None, None, None)
        image_pe = self.prompt_encoder.get_dense_pe()   # (1, C, H_tok, W_tok)

        # 3. Build token sequence directly at batch dim B to avoid
        #    repeat_interleave incompatibility in predict_masks.
        iou_tok = self.mask_decoder.iou_token.weight        # (1, C)
        mask_toks = self.mask_decoder.mask_tokens.weight    # (num_mask_tokens, C)
        output_tokens = torch.cat([iou_tok, mask_toks], dim=0)          # (1+num_mask_tokens, C)
        tokens = output_tokens.unsqueeze(0).expand(B, -1, -1)           # (B, 1+num_mask_tokens, C)
        # No sparse prompt tokens (prompt-free), so tokens == output_tokens

        # 4. Batch-aligned src / pos_src (no repeat_interleave)
        src = image_embedding + dense_pe.expand(B, -1, -1, -1)          # (B, C, H_tok, W_tok)
        pos_src = image_pe.expand(B, -1, -1, -1)                        # (B, C, H_tok, W_tok)

        # 5. Two-way transformer
        hs, src_out = self.mask_decoder.transformer(src, pos_src, tokens)
        # hs:     (B, 1+num_mask_tokens, C)
        # src_out:(B, H_tok*W_tok, C)
        mask_tokens_out = hs[:, 1 : 1 + self.mask_decoder.num_mask_tokens, :]  # (B, num_mask_tokens, C)

        # 6. Upscale image embedding
        #    ConvTranspose2d x2: (B, C, H_tok, W_tok) → (B, C//8, H_tok*4, W_tok*4)
        src_2d = src_out.transpose(1, 2).view(B, C, H_tok, W_tok)
        upscaled = self.mask_decoder.output_upscaling(src_2d)           # (B, C//8, H_tok*4, W_tok*4)

        # 7. Hypernetwork: generate per-mask prediction kernels
        hyper_in_list = [
            self.mask_decoder.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            for i in range(self.mask_decoder.num_mask_tokens)
        ]
        hyper_in = torch.stack(hyper_in_list, dim=1)                    # (B, num_mask_tokens, C//8)

        b, c_up, h_up, w_up = upscaled.shape
        masks = (hyper_in @ upscaled.view(b, c_up, h_up * w_up)).view(b, -1, h_up, w_up)
        # masks: (B, num_mask_tokens, H_tok*4, W_tok*4)

        # 8. Single-mask mode: take token index 0 (same as multimask_output=False)
        seg_logits = masks[:, 0:1, :, :]                                # (B, 1, H_tok*4, W_tok*4)

        # 9. Project to num_classes and upsample to input resolution
        seg_logits = self.sam_cls_conv(seg_logits)                      # (B, num_classes, H_tok*4, W_tok*4)
        seg_logits = F.interpolate(
            seg_logits, size=x.shape[2:], mode="bilinear", align_corners=False
        )

        if return_features:
            return seg_logits, image_embedding

        return seg_logits

    def load_pretrained(self, path=None):
        """Load pretrained backbone weights with TinyUSFM key mapping (model. -> backbone.)."""
        load_path = path or self.checkpoint_path
        if not load_path:
            print("Warning: No checkpoint path provided for load_pretrained.")
            return

        try:
            checkpoint = torch.load(load_path, map_location="cpu")
            new_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith("model."):
                    new_state_dict[k.replace("model.", "backbone.")] = v
                else:
                    new_state_dict[k] = v
            load_info = self.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded pretrained weights from {load_path}")
            print(f"Missing keys: {len(load_info.missing_keys)}")
            print(f"Unexpected keys: {len(load_info.unexpected_keys)}")
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights. Error: {e}")

    def load_finetuned(self, path):
        """Load a finetuned checkpoint (keys already match this model's state dict)."""
        try:
            checkpoint = torch.load(path, map_location="cpu")
            # Support checkpoints saved as {'model': state_dict, ...} or plain state dicts
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
            load_info = self.load_state_dict(state_dict, strict=False)
            print(f"Loaded finetuned checkpoint from {path}")
            print(f"Missing keys: {len(load_info.missing_keys)}")
            print(f"Unexpected keys: {len(load_info.unexpected_keys)}")
        except Exception as e:
            print(f"Warning: Failed to load finetuned checkpoint. Error: {e}")

    def load_checkpoint(self, path=None):
        """Load checkpoint with TinyUSFM specific mapping (pretrained weights).

        Deprecated: use load_pretrained() or load_finetuned() instead.
        """
        self.load_pretrained(path)

    def load_sam_decoder_weights(self, path: str) -> None:
        """Load pretrained SAM weights into the sam_mask branch modules.

        Only ``prompt_encoder`` and ``mask_decoder`` are touched; backbone and
        FPN submodules are left unchanged.  The ``sam_feature_proj`` and
        ``sam_cls_conv`` projection layers are intentionally excluded and keep
        their random initialization so they can be trained from scratch.

        The SAM checkpoint can be a full SAM ``.pth`` file (keys prefixed with
        ``mask_decoder.`` / ``prompt_encoder.``) or a standalone decoder
        checkpoint (keys without prefix).

        Args:
            path: Path to the SAM checkpoint file.

        Raises:
            RuntimeError: If ``decoder_type != "sam_mask"`` (no modules to
                load into).
        """
        if self.decoder_type != "sam_mask":
            raise RuntimeError(
                "load_sam_decoder_weights() called but decoder_type is "
                f"'{self.decoder_type}', not 'sam_mask'."
            )

        raw = torch.load(path, map_location="cpu")
        # Raw SAM checkpoints are sometimes wrapped; unwrap if needed.
        state_dict = raw.get("model", raw)

        # Extract only the decoder/prompt-encoder keys, strip any leading
        # "image_encoder." prefix that might appear in full SAM files.
        targets = {
            "prompt_encoder": self.prompt_encoder,
            "mask_decoder": self.mask_decoder,
        }
        for module_name, module in targets.items():
            prefix = module_name + "."
            sub_dict = {
                k[len(prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            if not sub_dict:
                print(f"[load_sam_decoder_weights] No keys found for '{module_name}' – skipped.")
                continue
            info = module.load_state_dict(sub_dict, strict=True)
            print(
                f"[load_sam_decoder_weights] Loaded '{module_name}': "
                f"missing={info.missing_keys}, unexpected={info.unexpected_keys}"
            )
