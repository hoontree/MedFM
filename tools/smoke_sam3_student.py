"""Smoke test for the trainable SAM3 student.

Answers the three things that decide whether SAM-vit_h → SAM3 distillation is even
possible, none of which are visible from a loss curve:

1. Do gradients actually reach SAM3's weights — and specifically its **classification /
   presence head**? SAM3's class decision *is* the instance score, so if the aggregation
   feeds the score in only through an argmax or a `> threshold` boolean (as "top1"/"max"
   do), the classification head gets exactly zero gradient and the student can never
   learn the benign/malignant distinction — while the loss still goes down. That is the
   failure this test exists to catch.
2. Does back-prop through a 1008^2 ViT fit, and how fast is it?
3. Both must be checked on REAL images: on random noise SAM3 correctly grounds nothing,
   every instance score falls under the gate, the output collapses to all-background and
   the gradient is legitimately zero — which looks exactly like a broken graph.

  uv run tools/smoke_sam3_student.py --aggregate soft --batches 1,2,4,8
"""
import argparse
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir

from model.sam3_student import Sam3Student
from utils.data_processing import SegDatasetProcessor

CKPT = "logs/train/sam3/20260713_111233_bs1_lr0.0001/checkpoints/epoch_10.pth"
# Substrings identifying SAM3's classification / presence path — the head that must
# receive gradient for the student to be able to learn a class at all.
_CLS_KEYS = ("class_embed", "pred_logits", "presence", "cls", "score")


def real_batch(n, img_size=224):
    """A batch of real ultrasound images + 3-class GT (noise cannot test grounding)."""
    # The distill config pulls in structured schemas (wandb/base_schema, …); they have
    # to be in the ConfigStore before compose() or Hydra can't resolve the defaults.
    from config.schema import register_schemas
    register_schemas()

    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_DIR / "config")):
        cfg = compose(
            config_name="distill_sam_to_usfm_binary",
            overrides=[
                "data.num_classes=3",
                f"data.img_size={img_size}",
                "data.auto_img_size_by_sam_type=false",
            ],
        )
    _, val_loader, _ = SegDatasetProcessor.build_data_loaders(cfg)
    batch = next(iter(val_loader))
    img, mask = batch[0], batch[1]
    gt = mask.argmax(dim=1) if mask.dim() == 4 and mask.shape[1] > 1 else mask.squeeze(1)
    reps = (n + img.shape[0] - 1) // img.shape[0]
    return img.repeat(reps, 1, 1, 1)[:n], gt.long().repeat(reps, 1, 1)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--prompt-set", default="v2")
    ap.add_argument("--aggregate", default="soft")
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()

    dev = "cuda"
    student = Sam3Student(
        num_classes=args.num_classes, checkpoint=args.checkpoint,
        prompt_set=args.prompt_set, aggregate=args.aggregate, load_from_hf=True,
    ).to(dev)
    student.train()
    print(f"[cfg] aggregate={student.aggregate}  prompts={student.class_prompts}")

    named = [(n, p) for n, p in student.model.named_parameters() if p.requires_grad]
    cls_named = [(n, p) for n, p in named if any(k in n.lower() for k in _CLS_KEYS)]
    print(f"[params] trainable={sum(p.numel() for _, p in named):,}  "
          f"of which classification-head tensors={len(cls_named)}")

    opt = torch.optim.AdamW([p for _, p in named], lr=1e-5)
    x_all, gt_all = real_batch(max(int(b) for b in args.batches.split(",")))
    print(f"[data] real val images {tuple(x_all.shape)}  gt classes={sorted(set(gt_all.unique().tolist()))}")

    for B in [int(b) for b in args.batches.split(",")]:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        x, gt = x_all[:B].to(dev), gt_all[:B].to(dev)
        try:
            ts = []
            for s in range(args.steps):
                torch.cuda.synchronize(); t0 = time.time()
                opt.zero_grad(set_to_none=True)
                logits = student(x)
                assert logits.shape == (B, args.num_classes, 224, 224), logits.shape
                loss = F.cross_entropy(logits, gt)
                loss.backward()
                opt.step()
                torch.cuda.synchronize(); ts.append(time.time() - t0)

            live = sum(1 for _, p in named if p.grad is not None and p.grad.abs().sum() > 0)
            cls_live = sum(1 for _, p in cls_named
                           if p.grad is not None and p.grad.abs().sum() > 0)
            peak = torch.cuda.max_memory_allocated() / 1e9
            step = sum(ts[1:]) / max(len(ts) - 1, 1)
            flag = "OK" if cls_live > 0 else "!! CLS HEAD HAS NO GRADIENT !!"
            print(f"[B={B}] loss={loss.item():.4f}  grads: {live}/{len(named)} all, "
                  f"{cls_live}/{len(cls_named)} cls-head  {flag}  "
                  f"peak={peak:.1f}GB  step={step:.2f}s ({B/step:.2f} img/s)")
        except torch.cuda.OutOfMemoryError:
            print(f"[B={B}] OOM")
            break


if __name__ == "__main__":
    main()
