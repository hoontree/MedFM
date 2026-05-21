"""Evaluate Seg-TransMiter-lite variants on the project's test splits.

Variants (selected by ``--variant``):

* ``base``          – frozen TinyUSFM (or SAM) baseline, no adapter.
* ``adapter_sup``   – base + adapter, trained with GT only
                      (``loss.obj=0 loss.boundary=0 loss.feat=0``).
* ``adapter_sam``   – base + adapter, trained with the SAM teacher cache
                      (default).
* ``adapter_sam2``  – base + adapter, trained against the cache produced
                      by an adapted-SAM checkpoint (``train_sam_us_adapter``
                      run, then re-cache).

Each variant points to a different checkpoint via the CLI; the rest of the
forward pass is shared with :class:`AdapterOnBase`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from project.seg_transmiter_lite.train_tiny_with_sam_prior import AdapterOnBase  # noqa: E402
from utils.data_processing_seg import SegDatasetProcessor  # noqa: E402
from utils.evaluate import Evaluator_seg  # noqa: E402
from utils.visualize import visualize_segmentation  # noqa: E402

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Model construction                                                    #
# --------------------------------------------------------------------- #


def _build_model(cfg, variant: str, ckpt: Optional[str], device) -> torch.nn.Module:
    num_classes = int(cfg.data.num_classes)
    direction = cfg.get("direction", "tiny_student")

    if direction == "tiny_student":
        sam_feat_shape = cfg.sam_teacher.get("embedding_shape", [256, 14, 14])
        sam_feat_channels = int(sam_feat_shape[0])
        sam_feat_size = (int(sam_feat_shape[1]), int(sam_feat_shape[2]))
    else:
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

    if variant == "base":
        # Zero-init residual already makes the adapter a no-op, so the base
        # output is returned unchanged.  Nothing else to load.
        pass
    elif ckpt is None:
        raise ValueError(f"variant={variant} requires --checkpoint")
    else:
        # weights_only=False: our own checkpoint stores a serialized OmegaConf
        # dict alongside the tensors, which trips PyTorch 2.6's default safe
        # unpickler.
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.adapter.load_state_dict(state["adapter"])
        if "projector" in state:
            model.projector.load_state_dict(state["projector"])
        LOGGER.info("Loaded adapter checkpoint: %s (epoch=%s, val=%s)",
                    ckpt, state.get("epoch"), state.get("mean_fg_dice"))

    model.eval()
    return model


# --------------------------------------------------------------------- #
# Evaluation                                                            #
# --------------------------------------------------------------------- #


def _aggregate_metrics(per_batch, num_classes: int) -> Dict[str, float]:
    """Combine batch-level metrics into a single summary dict."""
    if num_classes == 1:
        keys = ["dice", "hd95", "iou", "sensitivity", "specificity", "pixel_acc", "bf_score"]
        agg = {k: [] for k in keys}
        for m in per_batch:
            for k in keys:
                agg[k].extend(m.get(k, []))
        return {k: float(sum(v) / max(len(v), 1)) for k, v in agg.items()}

    dice_per_class = [[] for _ in range(num_classes)]
    iou_per_class = [[] for _ in range(num_classes)]
    hd95_per_class = [[] for _ in range(num_classes)]
    for m in per_batch:
        for c in range(num_classes):
            dice_per_class[c].extend(m["dice_per_class"][c])
            iou_per_class[c].extend(m["iou_per_class"][c])
            if "hd95_per_class" in m:
                hd95_per_class[c].extend(m["hd95_per_class"][c])
    summary = {}
    fg_dice, fg_iou = [], []
    for c in range(num_classes):
        d = float(sum(dice_per_class[c]) / max(len(dice_per_class[c]), 1))
        i = float(sum(iou_per_class[c]) / max(len(iou_per_class[c]), 1))
        summary[f"dice_c{c}"] = d
        summary[f"iou_c{c}"] = i
        if hd95_per_class[c]:
            summary[f"hd95_c{c}"] = float(
                sum(hd95_per_class[c]) / len(hd95_per_class[c])
            )
        if c >= 1:
            fg_dice.append(d)
            fg_iou.append(i)
    summary["mean_fg_dice"] = float(sum(fg_dice) / max(len(fg_dice), 1))
    summary["mean_fg_iou"] = float(sum(fg_iou) / max(len(fg_iou), 1))
    return summary


@torch.no_grad()
def evaluate_loader(model, loader, num_classes: int, device) -> Dict[str, float]:
    per_batch = []
    for batch in loader:
        # Tolerate both wrapped (5-tuple) and base (4-tuple) loaders.
        if len(batch) == 5:
            images, masks, _low_res, _fn, _sam = batch
        else:
            images, masks, _low_res, _fn = batch
        images = images.to(device)
        masks = masks.to(device)
        out = model(images)
        per_batch.append(
            Evaluator_seg.evaluate_batch(out["logits"], masks, num_classes=num_classes)
        )
    return _aggregate_metrics(per_batch, num_classes)


# --------------------------------------------------------------------- #
# Visualization                                                         #
# --------------------------------------------------------------------- #


@torch.no_grad()
def save_qualitative(model, loader, output_dir: Path, num_classes: int, device, max_n: int = 8):
    """Render Image | GT | Prediction panels using the project's viz helper.

    Collects up to ``max_n`` samples across the first few batches, then calls
    :func:`utils.visualize.visualize_segmentation` once with batch-aligned
    lists (its native input format).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_list, preds_list, masks_list, fn_list = [], [], [], []
    collected = 0
    for batch in loader:
        if len(batch) == 5:
            images, masks, _low_res, fns, _sam = batch
        else:
            images, masks, _low_res, fns = batch
        images = images.to(device)
        masks = masks.to(device)
        out = model(images)
        if num_classes == 1:
            pred = torch.sigmoid(out["logits"]).cpu()
        else:
            # Encode argmax-by-class as a 1-channel index map for the viz helper.
            pred = out["logits"].argmax(dim=1, keepdim=True).float().cpu()
        images_list.append(images.cpu())
        preds_list.append({"pred": pred})
        masks_list.append(masks.cpu())
        fn_list.append(list(fns))
        collected += images.shape[0]
        if collected >= max_n:
            break

    if not images_list:
        return
    try:
        visualize_segmentation(
            images_list=images_list,
            preds_list=preds_list,
            masks_list=masks_list,
            num_classes=num_classes,
            save_dir=output_dir,
            num_samples=max_n,
            phase_name="eval",
            filenames_list=fn_list,
            log_to_wandb=False,
        )
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("Viz failed: %s", e)


# --------------------------------------------------------------------- #
# CLI                                                                   #
# --------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description="Evaluate Seg-TransMiter-lite variants.")
    parser.add_argument(
        "--config",
        default=str(_PROJECT_ROOT / "config" / "seg_transmiter_lite.yaml"),
    )
    parser.add_argument(
        "--variant",
        default="adapter_sam",
        choices=["base", "adapter_sup", "adapter_sam", "adapter_sam2"],
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--max-viz", type=int, default=8)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("overrides", nargs="*", help="key=value overrides (dotted)")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir or f"logs/seg_transmiter_lite/eval/{args.variant}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset/test loaders using the same convention as the rest of the project.
    _train, _val, test_loaders = SegDatasetProcessor.build_data_loaders(cfg)

    model = _build_model(cfg, args.variant, args.checkpoint, device)

    summary: Dict[str, Dict[str, float]] = {}
    for name, loader in test_loaders.items():
        LOGGER.info("Evaluating %s ...", name)
        summary[name] = evaluate_loader(model, loader, int(cfg.data.num_classes), device)
        LOGGER.info("  %s", summary[name])

        if args.visualize:
            save_qualitative(
                model, loader, output_dir / "viz" / name,
                num_classes=int(cfg.data.num_classes),
                device=device, max_n=args.max_viz,
            )

    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"variant": args.variant, "checkpoint": args.checkpoint, "metrics": summary}, f, indent=2)
    LOGGER.info("Wrote metrics -> %s/metrics.json", output_dir)


if __name__ == "__main__":
    main()
