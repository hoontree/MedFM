"""Reliability-map analysis & visualization for reliability-weighted KD.

Loads the configured teacher/student exactly as ``DistillTrainer`` does,
runs forward passes on a chosen test/val loader, and for each sample builds
the per-pixel reliability map together with its individual factors via
``build_reliability_map(..., return_components=True)``.

Two outputs (under ``logs/reliability_analysis/<timestamp>/``):

* Qualitative per-sample panels — image / GT / teacher pred / student pred and
  one heatmap per reliability factor plus the final map.
* Quantitative ``stats.json`` + ``reliability_hist.png`` — per-factor means and,
  crucially, mean reliability split by teacher-correct vs teacher-wrong pixels
  (the evidence that the map suppresses confidently-wrong teacher pixels).

Run (reuses the distillation config; reliability factors come from
``config/method/unified.yaml`` and can be overridden on the CLI):

    uv run tools/analyze_reliability.py
    uv run tools/analyze_reliability.py analysis.loader=BUID analysis.num_batches=8
    uv run tools/analyze_reliability.py method.reliability_kd.use_student_bypass=false
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Allow `uv run tools/analyze_reliability.py` to import the project packages
# (the repo root is not on sys.path when the script lives under tools/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
import matplotlib
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils.hardware import set_gpu  # noqa: E402
from trainers.distill_trainer import DistillTrainer  # noqa: E402

# Factor maps rendered (in order) when present in the components dict.
_FACTOR_ORDER = [
    "confidence",
    "entropy_penalty",
    "teacher_correctness_gate",
    "student_bypass_gate",
    "ignore_mask",
]

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def _denorm_image(img_chw: torch.Tensor) -> np.ndarray:
    """Best-effort denormalize a [3,H,W] tensor to a displayable [H,W,3]."""
    x = img_chw.detach().cpu().float().numpy().transpose(1, 2, 0)
    # Heuristic: imagenet-normalized tensors are roughly zero-mean.
    if x.min() < 0.0:
        x = x * _IMAGENET_STD + _IMAGENET_MEAN
    if x.max() > 1.5:  # 0-255 range
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def _to_gt_indices(masks: torch.Tensor) -> torch.Tensor:
    """One-hot [B,C,H,W] or [B,H,W] -> class-index [B,H,W] long (matches distiller)."""
    gt = masks
    if gt.dim() == 4:
        gt = gt.argmax(dim=1) if gt.shape[1] > 1 else gt[:, 0]
    return gt.long()


def _hard_pred(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Teacher/student hard prediction [B,H,W]."""
    if num_classes == 1:
        return (torch.sigmoid(logits.squeeze(1)) > 0.5).long()
    return logits.argmax(dim=1).long()


@hydra.main(version_base=None, config_path="../config", config_name="distill_sam_to_usfm_binary")
def main(cfg: DictConfig):
    set_gpu(cfg)

    OmegaConf.set_struct(cfg, False)
    # Keep this read-only: no W&B run, and use the fine-tuned student so its
    # predictions (and the student-bypass gate) are meaningful.
    cfg.setdefault("wandb", {})
    cfg.wandb["disabled"] = True
    cfg.use_student_finetuned_ckpt = True
    cfg.setdefault("analysis", {})
    a = cfg.analysis
    loader_name = a.get("loader", None)        # None -> first test loader
    num_batches = int(a.get("num_batches", 4))
    num_panels = int(a.get("num_panels", 12))
    student_ckpt = a.get("student_checkpoint", None)

    if cfg.method.get("w_reliability_kd", 0.0) <= 0:
        raise SystemExit(
            "w_reliability_kd must be > 0 to build the reliability map. "
            "Run with method=unified (default) or set method.w_reliability_kd=1.0."
        )

    trainer = DistillTrainer(cfg)
    device = trainer.device
    num_classes = cfg.data.num_classes

    if student_ckpt:
        ckpt = torch.load(student_ckpt, map_location="cpu", weights_only=False)
        trainer.student.load_state_dict(ckpt.get("model_state_dict", ckpt))
        trainer.logger.info(f"Loaded analysis student checkpoint: {student_ckpt}")

    trainer.teacher.eval()
    trainer.student.eval()
    build_map = trainer.distiller._build_reliability_map
    if build_map is None:
        raise SystemExit("distiller._build_reliability_map is None (w_reliability_kd<=0).")

    # Resolve the loader to analyse.
    if isinstance(trainer.test_loader, dict):
        if loader_name and loader_name in trainer.test_loader:
            loader = trainer.test_loader[loader_name]
        else:
            loader_name = next(iter(trainer.test_loader))
            loader = trainer.test_loader[loader_name]
    else:
        loader, loader_name = trainer.test_loader, "test"

    out_dir = Path(cfg.output.dir) / "reliability_analysis" / datetime.now().strftime("%Y%m%d_%H%M%S")
    panel_dir = out_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    trainer.logger.info(f"Analysing loader='{loader_name}', writing to {out_dir}")

    # Accumulators for quantitative analysis.
    comp_sums: dict[str, float] = {}
    comp_count = 0.0
    sum_r_correct = n_correct = sum_r_wrong = n_wrong = 0.0
    gated_total = gated_wrong = 0.0
    # Pearson(reliability, teacher_correct) running sums.
    s_r = s_c = s_rc = s_rr = s_cc = s_n = 0.0
    hist_bins = np.linspace(0.0, 1.0, 21)
    hist_correct = np.zeros(len(hist_bins) - 1)
    hist_wrong = np.zeros(len(hist_bins) - 1)

    panels_made = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= num_batches:
                break
            images = batch[0].to(device)
            masks = batch[1].to(device)

            t_logits = trainer._call_teacher(images)["masks"]
            s_logits = trainer._call_student(images)["masks"]
            if t_logits.shape != s_logits.shape:
                t_logits = torch.nn.functional.interpolate(
                    t_logits, size=s_logits.shape[-2:], mode="bilinear", align_corners=False
                )
            gt = _to_gt_indices(masks).to(device)

            reliability, components = build_map(
                teacher_logits=t_logits,
                student_logits=s_logits,
                gt=gt,
                return_components=True,
            )

            t_pred = _hard_pred(t_logits, num_classes)
            s_pred = _hard_pred(s_logits, num_classes)
            teacher_correct = (t_pred == gt)

            # --- accumulate component means ---
            for name, fac in components.items():
                comp_sums[name] = comp_sums.get(name, 0.0) + fac.mean().item()
            comp_count += 1.0

            r = reliability.reshape(-1)
            c = teacher_correct.reshape(-1).float()
            corr_mask = c > 0.5
            sum_r_correct += r[corr_mask].sum().item()
            n_correct += corr_mask.sum().item()
            sum_r_wrong += r[~corr_mask].sum().item()
            n_wrong += (~corr_mask).sum().item()
            gated = (r < 0.1).float()
            gated_total += gated.sum().item()
            gated_wrong += gated[~corr_mask].sum().item()

            rn = r.cpu().numpy()
            hist_correct += np.histogram(rn[corr_mask.cpu().numpy()], bins=hist_bins)[0]
            hist_wrong += np.histogram(rn[~corr_mask.cpu().numpy()], bins=hist_bins)[0]

            s_r += r.sum().item(); s_c += c.sum().item()
            s_rc += (r * c).sum().item()
            s_rr += (r * r).sum().item(); s_cc += (c * c).sum().item()
            s_n += r.numel()

            # --- qualitative panels ---
            for j in range(images.size(0)):
                if panels_made >= num_panels:
                    break
                _save_panel(
                    panel_dir / f"{loader_name}_b{bi}_s{j}.png",
                    _denorm_image(images[j]),
                    gt[j].cpu().numpy(),
                    t_pred[j].cpu().numpy(),
                    s_pred[j].cpu().numpy(),
                    {k: components[k][j].cpu().numpy() for k in _FACTOR_ORDER if k in components},
                    reliability[j].cpu().numpy(),
                    num_classes,
                )
                panels_made += 1

    # --- finalize stats ---
    mean_r_correct = sum_r_correct / max(n_correct, 1.0)
    mean_r_wrong = sum_r_wrong / max(n_wrong, 1.0)
    pearson = _pearson(s_r, s_c, s_rc, s_rr, s_cc, s_n)
    stats = {
        "loader": loader_name,
        "num_batches": int(min(num_batches, bi + 1)),
        "component_mean": {k: v / comp_count for k, v in comp_sums.items()},
        "mean_reliability_teacher_correct": mean_r_correct,
        "mean_reliability_teacher_wrong": mean_r_wrong,
        "wrong_to_correct_ratio": mean_r_wrong / max(mean_r_correct, 1e-8),
        "frac_pixels_gated_lt0.1": gated_total / max(s_n, 1.0),
        "frac_wrong_pixels_gated_lt0.1": gated_wrong / max(n_wrong, 1.0),
        "pearson_reliability_vs_teacher_correct": pearson,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    _save_hist(out_dir / "reliability_hist.png", hist_bins, hist_correct, hist_wrong)

    trainer.logger.info("Reliability analysis complete:\n" + json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"\nPanels: {panel_dir}\nStats:  {out_dir / 'stats.json'}")


def _pearson(s_r, s_c, s_rc, s_rr, s_cc, n):
    if n <= 1:
        return float("nan")
    cov = s_rc / n - (s_r / n) * (s_c / n)
    var_r = s_rr / n - (s_r / n) ** 2
    var_c = s_cc / n - (s_c / n) ** 2
    denom = (var_r * var_c) ** 0.5
    return float(cov / denom) if denom > 1e-12 else float("nan")


def _save_panel(path, image, gt, t_pred, s_pred, factors, reliability, num_classes):
    seg_kw = dict(vmin=0, vmax=max(1, num_classes - 1), cmap="tab10" if num_classes > 1 else "gray")
    tiles = [("image", image, None), ("GT", gt, seg_kw),
             ("teacher pred", t_pred, seg_kw), ("student pred", s_pred, seg_kw)]
    for name, fac in factors.items():
        tiles.append((name, fac, dict(vmin=0.0, vmax=1.0, cmap="viridis")))
    tiles.append(("reliability r", reliability, dict(vmin=0.0, vmax=1.0, cmap="magma")))

    n = len(tiles)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, (title, data, kw) in zip(axes, tiles):
        if kw is None:
            ax.imshow(data)
        else:
            im = ax.imshow(data, **kw)
            if kw.get("cmap") in ("viridis", "magma"):
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _save_hist(path, bins, hist_correct, hist_wrong):
    centers = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) * 0.42
    fig, ax = plt.subplots(figsize=(7, 4))
    hc = hist_correct / max(hist_correct.sum(), 1.0)
    hw = hist_wrong / max(hist_wrong.sum(), 1.0)
    ax.bar(centers - width / 2, hc, width=width, label="teacher correct", color="#2a9d8f")
    ax.bar(centers + width / 2, hw, width=width, label="teacher wrong", color="#e76f51")
    ax.set_xlabel("reliability r")
    ax.set_ylabel("fraction of pixels")
    ax.set_title("Reliability distribution: teacher-correct vs teacher-wrong")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
