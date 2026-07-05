"""Evidence that the (frozen) teacher makes *confidently wrong* predictions.

Motivation figure for reliability-aware KD: before any reweighting, plain KD
distils every teacher pixel equally — including pixels where the teacher is
highly confident yet disagrees with the GT. This script quantifies and
visualizes exactly those pixels for a given teacher (no student / no reliability
map involved).

Outputs (under ``logs/confidently_wrong/<timestamp>/``):
  * ``confidence_hist.png`` — teacher max-prob confidence, split by correct vs
    wrong pixels. A heavy high-confidence tail on the *wrong* curve is the
    "confidently wrong" phenomenon.
  * ``stats.json`` — fraction of pixels that are confidently wrong, fraction of
    wrong pixels that are confident, mean confidence of wrong vs correct, at a
    few thresholds.
  * per-sample panels: image | GT | teacher pred | confidence | confidently-wrong
    mask (red = conf > tau AND teacher != GT).

Run (teacher chosen via the usual override; gpu4 example):
    uv run tools/analyze_confidently_wrong.py teacher=sam analysis.loader=BUID
    uv run tools/analyze_confidently_wrong.py teacher=sam_lora analysis.conf_tau=0.7
"""

import json
import sys
from datetime import datetime
from pathlib import Path

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
from utils.reliability_kd import probs_from_logits, pixel_confidence_from_probs  # noqa: E402
from config.schema import register_schemas  # noqa: E402

register_schemas()  # wandb/data/training groups default to base_schema

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def _denorm(img_chw):
    x = img_chw.detach().cpu().float().numpy().transpose(1, 2, 0)
    if x.min() < 0.0:
        x = x * _IMAGENET_STD + _IMAGENET_MEAN
    if x.max() > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def _to_gt(masks):
    gt = masks
    if gt.dim() == 4:
        gt = gt.argmax(dim=1) if gt.shape[1] > 1 else gt[:, 0]
    return gt.long()


@hydra.main(version_base=None, config_path="../config", config_name="distill_sam_to_usfm_binary")
def main(cfg: DictConfig):
    set_gpu(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.setdefault("wandb", {})
    cfg.wandb["disabled"] = True
    a = cfg.get("analysis", {}) or {}
    loader_name = a.get("loader", None)
    num_batches = int(a.get("num_batches", 8))
    num_panels = int(a.get("num_panels", 8))
    tau = float(a.get("conf_tau", 0.7))

    trainer = DistillTrainer(cfg)
    device = trainer.device
    num_classes = cfg.data.num_classes
    trainer.teacher.eval()

    if isinstance(trainer.test_loader, dict):
        loader_name = loader_name if loader_name in trainer.test_loader else next(iter(trainer.test_loader))
        loader = trainer.test_loader[loader_name]
    else:
        loader, loader_name = trainer.test_loader, "test"

    out = Path(cfg.output.dir) / "confidently_wrong" / datetime.now().strftime("%Y%m%d_%H%M%S")
    pan = out / "panels"
    pan.mkdir(parents=True, exist_ok=True)
    trainer.logger.info(f"[confidently-wrong] teacher={cfg.teacher} loader={loader_name} tau={tau} -> {out}")

    bins = np.linspace(0, 1, 26)
    h_correct = np.zeros(len(bins) - 1)
    h_wrong = np.zeros(len(bins) - 1)
    n_total = n_wrong = n_correct = 0.0
    sum_conf_wrong = sum_conf_correct = 0.0
    taus = [0.5, 0.7, 0.9]
    n_conf_wrong = {t: 0.0 for t in taus}   # wrong AND conf>t
    panels = 0

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= num_batches:
                break
            images = batch[0].to(device)
            gt = _to_gt(batch[1]).to(device)
            t_logits = trainer._call_teacher(images)["masks"]
            if t_logits.shape[-2:] != gt.shape[-2:]:
                t_logits = torch.nn.functional.interpolate(
                    t_logits.float(), size=gt.shape[-2:], mode="bilinear", align_corners=False)
            probs = probs_from_logits(t_logits, temperature=1.0)
            conf = pixel_confidence_from_probs(probs, mode="max_prob")  # [B,H,W]
            if num_classes == 1:
                pred = (torch.sigmoid(t_logits.squeeze(1)) > 0.5).long()
            else:
                pred = t_logits.argmax(dim=1).long()
            correct = pred == gt

            c = conf.reshape(-1)
            cb = correct.reshape(-1)
            cc = c[cb].cpu().numpy()
            cw = c[~cb].cpu().numpy()
            h_correct += np.histogram(cc, bins=bins)[0]
            h_wrong += np.histogram(cw, bins=bins)[0]
            n_total += c.numel(); n_correct += cb.sum().item(); n_wrong += (~cb).sum().item()
            sum_conf_correct += float(cc.sum()); sum_conf_wrong += float(cw.sum())
            for t in taus:
                n_conf_wrong[t] += float(((~cb) & (c > t)).sum().item())

            for j in range(images.size(0)):
                if panels >= num_panels:
                    break
                _panel(pan / f"{loader_name}_b{bi}_s{j}.png", _denorm(images[j]),
                       gt[j].cpu().numpy(), pred[j].cpu().numpy(), conf[j].cpu().numpy(),
                       (~correct[j] & (conf[j] > tau)).cpu().numpy(), num_classes, tau)
                panels += 1

    stats = {
        "teacher": cfg.teacher, "loader": loader_name, "conf_tau": tau,
        "num_batches": int(min(num_batches, bi + 1)),
        "pixel_accuracy": n_correct / max(n_total, 1),
        "mean_conf_correct": sum_conf_correct / max(n_correct, 1),
        "mean_conf_wrong": sum_conf_wrong / max(n_wrong, 1),
        "frac_confidently_wrong_of_all": {f"tau>{t}": n_conf_wrong[t] / max(n_total, 1) for t in taus},
        "frac_of_wrong_that_are_confident": {f"tau>{t}": n_conf_wrong[t] / max(n_wrong, 1) for t in taus},
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    _hist(out / "confidence_hist.png", bins, h_correct, h_wrong, tau, cfg.teacher)
    trainer.logger.info("[confidently-wrong] " + json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print("panels:", pan)


def _panel(path, image, gt, pred, conf, cw_mask, num_classes, tau):
    seg = dict(vmin=0, vmax=max(1, num_classes - 1), cmap="tab10" if num_classes > 1 else "gray")
    over = image.copy()
    over[cw_mask] = [1.0, 0.0, 0.0]  # red = confidently wrong
    tiles = [("image", image, None), ("GT", gt, seg), ("teacher pred", pred, seg),
             ("teacher confidence", conf, dict(vmin=0, vmax=1, cmap="magma")),
             (f"confidently wrong\n(conf>{tau} & wrong)", over, None)]
    fig, axes = plt.subplots(1, len(tiles), figsize=(3 * len(tiles), 3))
    for ax, (t, d, kw) in zip(axes, tiles):
        im = ax.imshow(d) if kw is None else ax.imshow(d, **kw)
        if kw and kw.get("cmap") == "magma":
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(t, fontsize=9); ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def _hist(path, bins, h_correct, h_wrong, tau, teacher):
    ctr = (bins[:-1] + bins[1:]) / 2
    w = (bins[1] - bins[0]) * 0.9
    fig, ax = plt.subplots(figsize=(7, 4))
    hc = h_correct / max(h_correct.sum(), 1)
    hw = h_wrong / max(h_wrong.sum(), 1)
    ax.bar(ctr, hc, width=w, alpha=0.55, label="teacher correct", color="#2a9d8f")
    ax.bar(ctr, hw, width=w, alpha=0.55, label="teacher wrong", color="#e76f51")
    ax.axvline(tau, ls="--", c="k", lw=1, label=f"tau={tau}")
    ax.set_xlabel("teacher max-prob confidence"); ax.set_ylabel("fraction of pixels (within group)")
    ax.set_title(f"Teacher confidence: correct vs wrong ({teacher})\nmass right of tau on the wrong curve = confidently wrong")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
