"""Train a small adapter on top of a frozen TinyUSFM with SAM-prior signals.

Default direction: TinyUSFM is the deployable model.  At inference time
only TinyUSFM + ResidualConvAdapter are required — SAM does **not** need
to be loaded.

If ``direction == "sam_student"``, the *same* training loop trains an
adapter on top of a (mostly) frozen SAM image encoder, using the SAM
output dict instead of the TinyUSFM neck feature.

Losses
------
* ``L_sup``      – supervised DiceCE on the final 3-class logits
* ``L_obj``      – BCE+Dice between lesion prob and SAM teacher mask
                   (gated on SAM confidence)
* ``L_boundary`` – Sobel edge loss between lesion prob and teacher mask
                   (gated on SAM confidence)
* ``L_feat``     – cosine alignment between projected adapter feature and
                   the SAM image embedding (teacher cache)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model.seg_transmiter.adapters import ResidualConvAdapter, FeatureProjector  # noqa: E402
from project.seg_transmiter_lite.sam_cache_dataset import SAMCacheDataset  # noqa: E402
from utils.data_processing_seg import SegDatasetProcessor  # noqa: E402
from utils.evaluate import Evaluator_seg  # noqa: E402
from utils.seg_transmiter_losses import (  # noqa: E402
    bce_dice_loss,
    boundary_loss,
    confidence_gate,
    dice_ce_loss,
    feature_align_loss,
    gated_mean,
)

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Base + adapter composition                                            #
# --------------------------------------------------------------------- #


def _freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False
    module.eval()


def _tiny_forward(base, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run TinyUSFM ``SegmentationModel`` and return (logits, feat).

    ``return_features=True`` yields the 3rd-scale FPN feature
    (48ch, 14x14 @ 224 input).
    """
    return base(images, return_features=True)


def _sam_forward(base, images: torch.Tensor, img_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run ``LoRA_Sam`` prompt-free and expose (logits, image_embedding).

    We call ``LoRA_Sam.forward`` which already routes through the empty-prompt
    path (the same one used by the project's SAM trainer); the resulting
    ``outputs`` dict yields ``masks`` and — when alignment is on —
    ``image_embeddings``.  When alignment is off we recompute the embedding
    via the encoder directly (no extra forward of mask decoder).
    """
    outputs = base(images, False, img_size)
    if isinstance(outputs, list):
        outputs = outputs[0]
    logits = outputs["masks"]
    if "image_embeddings" in outputs:
        feat = outputs["image_embeddings"]
    else:
        with torch.no_grad():
            preprocessed = base.sam.preprocess(images)
            feat = base.sam.image_encoder(preprocessed)
    return logits, feat


class AdapterOnBase(nn.Module):
    """Frozen base (TinyUSFM or SAM) + residual adapter + feature projector.

    Only the adapter and projector hold trainable parameters by default.
    """

    def __init__(
        self,
        cfg,
        direction: str,
        num_classes: int,
        sam_feat_channels: int,
        sam_feat_size: Tuple[int, int],
        adapter_bottleneck=None,
    ):
        super().__init__()
        self.direction = direction
        self.img_size = int(cfg.data.img_size)
        self.num_classes = num_classes

        if direction == "tiny_student":
            from model.tinyusfm_seg import SegmentationModel
            self.base = SegmentationModel(
                num_classes=num_classes,
                checkpoint=cfg.tiny.get("checkpoint", None),
                use_alignment=False,
                decoder_type="fpn",
            )
            in_ch = int(192 * 0.25)  # FPN neck @ scale=1.0 -> 48ch
        elif direction == "sam_student":
            from model.sam_hybrid_adapter import LoRA_Sam
            self.base = LoRA_Sam(
                sam_type=cfg.sam.get("sam_type", "vit_b"),
                img_size=self.img_size,
                num_classes=num_classes,
                encoder_mode=cfg.sam.get("encoder_mode", "frozen"),
                decoder_mode=cfg.sam.get("decoder_mode", "frozen"),
                r_e=int(cfg.sam.get("r_e", 4)),
                r_d=int(cfg.sam.get("r_d", 4)),
                use_alignment=bool(cfg.sam.get("use_alignment", False)),
                checkpoint=cfg.sam.get("checkpoint", None),
            )
            in_ch = 256  # SAM image embedding always 256ch for vit_b
        else:
            raise ValueError(f"Unknown direction: {direction}")

        # Freeze base by default; reverse stage may still want LoRA params trainable
        # via train_sam_us_adapter.py (it does its own un-freezing).
        if bool(cfg.get("freeze_base", True)):
            _freeze(self.base)

        self.adapter = ResidualConvAdapter(
            in_channels=in_ch,
            num_classes=num_classes,
            bottleneck=adapter_bottleneck,
            zero_init=True,
        )
        self.projector = FeatureProjector(
            in_channels=in_ch,
            out_channels=sam_feat_channels,
            out_size=sam_feat_size,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep base in eval when its params are all frozen.
        if not any(p.requires_grad for p in self.base.parameters()):
            self.base.eval()
        return self

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.direction == "tiny_student":
            base_logits, feat = _tiny_forward(self.base, images)
        else:
            base_logits, feat = _sam_forward(self.base, images, self.img_size)

        residual = self.adapter(feat, out_size=base_logits.shape[-2:])
        final_logits = base_logits + residual
        projected = self.projector(feat)
        return {
            "logits": final_logits,
            "base_logits": base_logits,
            "residual": residual,
            "features": feat,
            "projected_features": projected,
        }

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# --------------------------------------------------------------------- #
# Training loop                                                         #
# --------------------------------------------------------------------- #


def _build_dataloaders(cfg, cache_dir: Path):
    train_ds, val_ds, _test_ds_dict = SegDatasetProcessor.build_dataset(cfg)
    embed_shape = tuple(cfg.sam_teacher.get("embedding_shape", [256, 14, 14]))
    train_wrapped = SAMCacheDataset(train_ds, cache_dir, embedding_shape=embed_shape)
    val_wrapped = SAMCacheDataset(val_ds, cache_dir, embedding_shape=embed_shape)
    bs = int(cfg.training.batch_size)
    nw = int(cfg.training.num_workers)
    return (
        DataLoader(train_wrapped, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True),
        DataLoader(val_wrapped, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True),
    )


def _lesion_probability(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    if num_classes <= 1:
        return torch.sigmoid(logits)
    p = F.softmax(logits, dim=1)
    return p[:, 1:].sum(dim=1, keepdim=True)


def _validate(model: AdapterOnBase, loader: DataLoader, device, num_classes: int) -> Dict[str, float]:
    model.eval()
    dice_lists = [[] for _ in range(num_classes)]
    iou_lists = [[] for _ in range(num_classes)]
    with torch.no_grad():
        for batch in loader:
            images, masks, _low_res, _fn, _sam = batch
            images = images.to(device)
            masks = masks.to(device)
            out = model(images)
            metrics = Evaluator_seg.evaluate_batch(
                out["logits"], masks, num_classes=num_classes
            )
            if num_classes == 1:
                dice_lists[0].extend(metrics["dice"])
                iou_lists[0].extend(metrics["iou"])
            else:
                for c in range(num_classes):
                    dice_lists[c].extend(metrics["dice_per_class"][c])
                    iou_lists[c].extend(metrics["iou_per_class"][c])

    target_classes = list(range(1, num_classes)) if num_classes > 1 else [0]
    fg_dice = [
        sum(dice_lists[c]) / max(len(dice_lists[c]), 1) for c in target_classes
    ]
    mean_fg_dice = sum(fg_dice) / len(fg_dice)
    summary = {"mean_fg_dice": mean_fg_dice}
    for c in target_classes:
        summary[f"dice_c{c}"] = sum(dice_lists[c]) / max(len(dice_lists[c]), 1)
    return summary


def train(cfg, output_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Output dir: %s", output_dir)

    direction = cfg.get("direction", "tiny_student").lower()
    num_classes = int(cfg.data.num_classes)

    if direction == "tiny_student":
        sam_feat_shape = cfg.sam_teacher.get("embedding_shape", [256, 14, 14])
        sam_feat_channels = int(sam_feat_shape[0])
        sam_feat_size = (int(sam_feat_shape[1]), int(sam_feat_shape[2]))
    else:
        # Reverse direction: align SAM embedding *into* TinyUSFM neck space.
        tok = int(cfg.data.img_size) // 16
        sam_feat_channels = int(192 * 0.25)
        sam_feat_size = (tok, tok)

    model = AdapterOnBase(
        cfg=cfg,
        direction=direction,
        num_classes=num_classes,
        sam_feat_channels=sam_feat_channels,
        sam_feat_size=sam_feat_size,
        adapter_bottleneck=cfg.adapter.get("bottleneck", None),
    ).to(device)

    trainable = model.trainable_parameters()
    n_trainable = sum(p.numel() for p in trainable)
    LOGGER.info("Trainable parameters: %s", f"{n_trainable:,}")

    cache_dir = Path(cfg.sam_teacher.cache_dir)
    train_loader, val_loader = _build_dataloaders(cfg, cache_dir)

    lr = float(cfg.training.base_lr)
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=float(cfg.training.weight_decay))
    num_epochs = int(cfg.training.num_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    use_amp = bool(cfg.training.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    w_sup = float(cfg.loss.get("sup", 1.0))
    w_obj = float(cfg.loss.get("obj", 0.5))
    w_boundary = float(cfg.loss.get("boundary", 0.2))
    w_feat = float(cfg.loss.get("feat", 0.1))
    sam_threshold = float(cfg.loss.get("sam_threshold", 0.75))

    best_dice = -1.0
    best_path = output_dir / "best.pth"

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        total = {"sup": 0.0, "obj": 0.0, "boundary": 0.0, "feat": 0.0, "loss": 0.0}
        n = 0
        for batch in pbar:
            images, masks, _low_res, _fn, sam = batch
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            sam_mask = sam["sam_mask"].to(device, non_blocking=True)
            sam_score = sam["sam_score"].to(device, non_blocking=True)
            sam_emb = sam["sam_embedding"].to(device, non_blocking=True)
            has_lesion = sam["has_lesion"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(images)
                final_logits = out["logits"]

                l_sup = dice_ce_loss(final_logits, masks, num_classes=num_classes)

                gate = confidence_gate(sam_score, threshold=sam_threshold) * has_lesion
                lesion_prob = _lesion_probability(final_logits, num_classes)

                if gate.sum() > 0:
                    obj_stack = torch.stack([
                        bce_dice_loss(lesion_prob[i:i+1], sam_mask[i:i+1])
                        if gate[i].item() > 0 else torch.zeros((), device=device)
                        for i in range(images.shape[0])
                    ])
                    bnd_stack = torch.stack([
                        boundary_loss(lesion_prob[i:i+1], sam_mask[i:i+1])
                        if gate[i].item() > 0 else torch.zeros((), device=device)
                        for i in range(images.shape[0])
                    ])
                    l_obj = gated_mean(obj_stack, gate)
                    l_boundary = gated_mean(bnd_stack, gate)
                else:
                    l_obj = torch.zeros((), device=device)
                    l_boundary = torch.zeros((), device=device)

                if has_lesion.sum() > 0:
                    proj = out["projected_features"]
                    feat_stack = torch.stack([
                        feature_align_loss(proj[i:i+1], sam_emb[i:i+1])
                        if has_lesion[i].item() > 0 else torch.zeros((), device=device)
                        for i in range(images.shape[0])
                    ])
                    l_feat = gated_mean(feat_stack, has_lesion)
                else:
                    l_feat = torch.zeros((), device=device)

                loss = (
                    w_sup * l_sup
                    + w_obj * l_obj
                    + w_boundary * l_boundary
                    + w_feat * l_feat
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            n += 1
            total["sup"] += float(l_sup.item())
            total["obj"] += float(l_obj.item())
            total["boundary"] += float(l_boundary.item())
            total["feat"] += float(l_feat.item())
            total["loss"] += float(loss.item())
            pbar.set_postfix({k: f"{v/n:.4f}" for k, v in total.items()})

        scheduler.step()

        val_metrics = _validate(model, val_loader, device, num_classes)
        LOGGER.info(
            "Epoch %d: train_loss=%.4f  val_mean_fg_dice=%.4f",
            epoch + 1, total["loss"] / max(n, 1), val_metrics["mean_fg_dice"],
        )

        if val_metrics["mean_fg_dice"] > best_dice:
            best_dice = val_metrics["mean_fg_dice"]
            state = {
                "adapter": model.adapter.state_dict(),
                "projector": model.projector.state_dict(),
                "epoch": epoch + 1,
                "mean_fg_dice": best_dice,
                "config": OmegaConf.to_container(cfg, resolve=True),
                "direction": direction,
            }
            torch.save(state, best_path)
            LOGGER.info("Saved best adapter -> %s (mean_fg_dice=%.4f)", best_path, best_dice)

    LOGGER.info("Training complete. Best mean_fg_dice = %.4f", best_dice)


def main():
    parser = argparse.ArgumentParser(description="Train base+adapter with SAM prior.")
    parser.add_argument(
        "--config",
        default=str(_PROJECT_ROOT / "config" / "seg_transmiter_lite.yaml"),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--direction", default=None, choices=["tiny_student", "sam_student"])
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("overrides", nargs="*", help="key=value overrides (dotted)")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    if args.direction:
        cfg.direction = args.direction

    output_dir = Path(
        args.output_dir
        or cfg.get("output_dir")
        or f"logs/seg_transmiter_lite/{int(time.time())}"
    )
    train(cfg, output_dir)


if __name__ == "__main__":
    main()
