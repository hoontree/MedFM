"""Phase B sanity: does the SAM3 multiclass teacher actually DISCRIMINATE
benign vs malignant, or does it degenerately ground the same lesion for both
prompts? Runs the *real* Sam3Teacher(num_classes=3) per-class grounding path
(exactly what KD consumes) on the internal val split and reports per-class Dice
against the 3-class GT. A binary-foreground Dice would hide degeneracy; this
does not.

  uv run tools/sanity_sam3_mc_teacher.py \
      --checkpoint logs/train/sam3/<run>/checkpoints/best_*.pth [--max-batches 40]

Decision gate (plan): mean foreground Dice > 0.4 AND both classes non-trivial
→ keep SAM3 on the multiclass spectrum; else fall back to the SAM-only
3-teacher spectrum.
"""
import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import torch
from hydra import compose, initialize_config_dir

from utils.data_processing import SegDatasetProcessor
from model.sam3_teacher import Sam3Teacher
from model.sam3_prompts import PROMPT_SETS


def _class_index(mask: torch.Tensor) -> torch.Tensor:
    """Normalize a GT mask batch to [B,H,W] long class indices."""
    if mask.dim() == 4:
        return mask.argmax(dim=1) if mask.shape[1] > 1 else mask[:, 0]
    return mask


def dice_per_class(pred_idx, gt_idx, num_classes):
    """Per-class Dice accumulators (intersection, pred_sum, gt_sum)."""
    out = {}
    for c in range(1, num_classes):  # skip background (0)
        p = (pred_idx == c)
        g = (gt_idx == c)
        inter = (p & g).sum().item()
        out[c] = (inter, p.sum().item(), g.sum().item())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="SAM3 MC FT checkpoint; None → base HF weights (expect ~0 Dice)")
    ap.add_argument("--max-batches", type=int, default=40)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--score-threshold", type=float, default=0.2,
                    help="Sam3Teacher instance score gate; higher → less benign flooding")
    ap.add_argument("--aggregate", default="max", choices=["max", "top1"],
                    help="instance aggregation: max (all kept) | top1 (best-score instance)")
    ap.add_argument("--class-prompts", default=None,
                    help="comma-separated grounding prompts, one per foreground class. "
                         "MUST match the FT manifest's category names (e.g. the v2 "
                         "prompts). Default: benign,malignant")
    args = ap.parse_args()

    try:
        from config.schema import register_schemas
        register_schemas()
    except Exception:
        pass

    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_DIR / "config")):
        cfg = compose(
            config_name="distill_sam_to_usfm_binary",
            overrides=[
                "data.num_classes=3",
                f"data.img_size={args.img_size}",
                "data.auto_img_size_by_sam_type=false",
            ],
        )

    _, val_loader, _ = SegDatasetProcessor.build_data_loaders(cfg)
    num_classes = 3
    dev = "cuda"

    prompts = [p.strip() for p in args.class_prompts.split(",")] if args.class_prompts else None
    teacher = Sam3Teacher(num_classes=num_classes, checkpoint=args.checkpoint,
                          class_prompts=prompts,
                          score_threshold=args.score_threshold,
                          aggregate=args.aggregate,
                          load_from_hf=True).to(dev).eval()
    print(f"[sanity] class_prompts = {teacher.class_prompts}")

    acc = {c: [0, 0, 0] for c in range(1, num_classes)}  # inter, psum, gsum
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            image = batch[0].to(dev)
            gt_idx = _class_index(batch[1]).to(dev)
            logits = teacher(image)                       # [B,3,H,W]
            pred_idx = logits.argmax(dim=1)               # [B,H,W]
            for c, (i, ps, gs) in dice_per_class(pred_idx, gt_idx, num_classes).items():
                acc[c][0] += i; acc[c][1] += ps; acc[c][2] += gs
            n += 1
            if n >= args.max_batches:
                break

    print(f"\n[sanity] SAM3 MC teacher over {n} val batches (ckpt={args.checkpoint}):")
    dices = []
    names = PROMPT_SETS["v1"]
    for c in range(1, num_classes):
        i, ps, gs = acc[c]
        d = 2 * i / (ps + gs + 1e-6)
        dices.append(d)
        print(f"  class {c} ({names.get(c, c)}): Dice={d:.4f}  (pred_px={ps}, gt_px={gs})")
    print(f"  mean foreground Dice = {sum(dices)/len(dices):.4f}")
    print("  gate: >0.4 AND both classes non-trivial → keep SAM3 on the spectrum")


if __name__ == "__main__":
    main()
