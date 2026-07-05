"""Inference / evaluation-only entry point.

Runs the model on every sample in the test dataset(s), saves one PNG
visualisation per sample, and writes per-sample metrics to CSV.

Usage examples
--------------
# SAM with LoRA checkpoint
python infer.py model=sam infer.checkpoint=/path/to/lora.pth

# TinyUSFM full-model checkpoint
python infer.py model=tinyusfm infer.checkpoint=/path/to/weights.pth

# Evaluate on a specific dataset
python infer.py model=sam data=BUID infer.checkpoint=...

# Override multiple test sets (dynamic config)
python infer.py model=sam data=dynamic data.test=[BUID,BUS_UCLM] infer.checkpoint=...

# Custom output directory and batch size
python infer.py model=tinyusfm infer.checkpoint=... \\
    infer.output_dir=results/my_run infer.batch_size=4

Output layout
-------------
<output_dir>/<model>/<dataset>/<timestamp>/
    visualizations/<ds_name>/
        sample_<filename>.png   # Image | GT | Prediction | Overlay
    metrics_<ds_name>.csv       # per-sample metrics, one row per sample
    metrics_all.csv             # combined (only when multiple test sets)
"""

import logging
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from medpy.metric.binary import dc, hd95
from medpy.metric.binary import recall as medpy_recall

from config.schema import register_schemas
from utils.data_processing import SegDatasetProcessor
from utils.evaluate import Evaluator_seg
from utils.hardware import set_gpu
from utils.visualize import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    denormalize_image,
    plot_segmentation_panel,
)

log = logging.getLogger(__name__)


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_model(cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    """Instantiate model from config and load the inference checkpoint.

    Supports three checkpoint formats:
    - SAM / LoRA:   model has ``load_lora_parameters`` → loads compact LoRA file.
    - Full state dict with wrapper keys (``model_state_dict`` / ``state_dict``).
    - Plain state dict (default ``torch.save(model.state_dict(), ...)``).
    """
    model = instantiate(cfg.model)

    # TinyUSFM: load pretrained MAE backbone weights first.
    # Note: For LoRA_Sam (SAM hybrid adapter), load_checkpoint is a unified loader
    # for both LoRA and full checkpoints. We skip the redundant MAE-specific 
    # check if the model is LoRA_Sam.
    is_lora_sam = model.__class__.__name__ == "LoRA_Sam"
    
    if not is_lora_sam and hasattr(model, "load_checkpoint") and callable(model.load_checkpoint):
        model.load_checkpoint()
        log.info("Loaded backbone weights via model.load_checkpoint().")

    ckpt_path = cfg.infer.get("checkpoint")
    if ckpt_path is None:
        raise ValueError(
            "infer.checkpoint is required. "
            "Pass it on the command line: infer.checkpoint=/path/to/file.pth"
        )
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # For LoRA_Sam, use its unified load_checkpoint method which handles
    # LoRA parameters, full state dicts, and SAM wrapper keys correctly.
    if is_lora_sam and hasattr(model, "load_checkpoint"):
        model.load_checkpoint(str(ckpt_path))
        log.info("Loaded LoRA_Sam checkpoint via model.load_checkpoint(): %s", ckpt_path)
    elif hasattr(model, "load_lora_parameters"):
        # SAM-family: compact LoRA file.
        model.load_lora_parameters(str(ckpt_path))
        log.info("Loaded LoRA checkpoint: %s", ckpt_path)
    else:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        # Unwrap common trainer wrapper keys.
        if isinstance(state, dict):
            state = state.get("model_state_dict", state.get("state_dict", state))
        model.load_state_dict(state, strict=False)
        log.info("Loaded checkpoint: %s", ckpt_path)

    return model.to(device)


# ── Forward pass dispatcher ────────────────────────────────────────────────────

def _forward(
    model: torch.nn.Module,
    images: torch.Tensor,
    model_name: str,
    img_size: int,
    target_size: tuple,
) -> torch.Tensor:
    """Run a forward pass and return logits resized to *target_size*.

    Args:
        model:       The segmentation model.
        images:      Input tensor [B, C, H, W] on the correct device.
        model_name:  ``"sam"``, ``"segformer"``, or any other string (default branch).
        img_size:    SAM's expected image size (used only when model_name="sam").
        target_size: (H, W) to resize logits to when they differ.

    Returns:
        Logits tensor [B, C, H, W].
    """
    is_lora_sam = model.__class__.__name__ == "LoRA_Sam"

    if model_name == "sam" or is_lora_sam:
        # LoRA_Sam (SAM hybrid adapter) expects: (images, multimask_output, image_size)
        out = model(images, False, img_size)
        if isinstance(out, dict):
            # Use explicit key checking to avoid "Boolean value of Tensor is ambiguous" error
            if "masks" in out:
                logits = out["masks"]
            elif "low_res_logits" in out:
                logits = out["low_res_logits"]
            else:
                logits = next(iter(out.values()))
        else:
            logits = out
    elif model_name == "segformer":
        out = model(images)
        logits = out.logits
    else:
        logits = model(images)
        if isinstance(logits, tuple):
            logits = logits[0]
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)

    if logits.shape[-2:] != target_size:
        logits = F.interpolate(
            logits, size=target_size, mode="bilinear", align_corners=False
        )
    return logits


# ── Per-sample metric computation ─────────────────────────────────────────────

def _metrics_binary(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute per-sample metrics for binary segmentation.

    Args:
        pred: 2-D predicted binary mask (float or bool).
        gt:   2-D ground-truth binary mask (float or bool).

    Returns:
        Dict with keys: dice, hd95, iou, sensitivity, specificity,
        pixel_acc, bf_score.
    """
    pb = pred.astype(bool)
    gb = gt.astype(bool)

    dice_v = float(dc(pb, gb))

    if pb.any() and gb.any():
        hd_v = float(hd95(pb, gb))
    elif not pb.any() and not gb.any():
        hd_v = 0.0
    else:
        hd_v = 224.0

    return {
        "dice":        dice_v,
        "hd95":        hd_v,
        "iou":         float(Evaluator_seg.compute_jaccard(pb, gb)),
        "sensitivity": float(medpy_recall(pb, gb)),
        "specificity": float(Evaluator_seg.compute_specificity(pred.astype(int), gt.astype(int))),
        "pixel_acc":   float((pb == gb).sum()) / gb.size,
        "bf_score":    float(Evaluator_seg.compute_boundary_score(pb, gb)),
    }


def _metrics_multiclass(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> dict:
    """Compute per-sample metrics for multiclass segmentation.

    Averages Dice, HD95, and IoU over all foreground classes (1 … num_classes-1).

    Args:
        pred:        2-D integer class prediction map.
        gt:          2-D integer ground-truth class map.
        num_classes: Total number of classes (including background at 0).

    Returns:
        Dict with keys: dice, hd95, iou, pixel_acc.
    """
    dice_vals, hd_vals, iou_vals = [], [], []
    for c in range(1, num_classes):
        p = pred == c
        g = gt == c
        if not (p.any() or g.any()):
            continue
        tp = np.logical_and(p, g).sum()
        dice_vals.append(float(2 * tp / (p.sum() + g.sum() + 1e-8)))
        iou_vals.append(float(Evaluator_seg.compute_jaccard(p, g)))
        if p.any() and g.any():
            hd_vals.append(float(hd95(p.astype(bool), g.astype(bool))))
        else:
            hd_vals.append(224.0)

    return {
        "dice":      float(np.mean(dice_vals)) if dice_vals else 0.0,
        "hd95":      float(np.mean(hd_vals))   if hd_vals  else 224.0,
        "iou":       float(np.mean(iou_vals))  if iou_vals else 0.0,
        "pixel_acc": float((pred == gt).mean()),
    }


def _format_latex_mean_std(df: pd.DataFrame, numeric_cols: list[str]) -> str:
    """Format per-metric mean and std as LaTeX-friendly ``$mean \\pm std$`` strings."""
    if not numeric_cols:
        return "(no numeric metrics)"

    mean_s = df[numeric_cols].mean()
    std_s = df[numeric_cols].std()
    parts = [
        f"{col}: ${mean_s[col]:.4f} \\pm {std_s[col]:.4f}$"
        for col in numeric_cols
    ]
    return ", ".join(parts)


# ── Main inference loop ────────────────────────────────────────────────────────

def _run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    model_name: str,
    img_size: int,
    threshold: float,
    vis_dir: Path,
) -> pd.DataFrame:
    """Infer on every sample; save a visualisation PNG; return metrics DataFrame.

    Each row in the returned DataFrame corresponds to one sample and is
    identified by its *filename* stem (taken from the last element of the
    batch tuple, which is the filename stem set by ``BaseUltrasoundDataset``).

    Args:
        model:       Loaded, eval-mode model.
        loader:      DataLoader for a single test set.
        device:      Computation device.
        num_classes: Number of segmentation classes.
        model_name:  Used by :func:`_forward` to dispatch the correct call.
        img_size:    Image size passed to SAM.
        threshold:   Binary decision threshold (ignored for multiclass).
        vis_dir:     Directory where PNG files are saved.

    Returns:
        DataFrame with columns: filename, + all metric names.
    """
    vis_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    records = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", unit="batch"):
            images = batch[0].to(device)
            labels = batch[1].to(device)

            # Filename stems are always the last batch element when the dataset
            # inherits from BaseUltrasoundDataset (returns a 4-tuple).
            last = batch[-1]
            if (
                isinstance(last, (list, tuple))
                and last
                and isinstance(last[0], str)
            ):
                fnames = list(last)
            else:
                fnames = [f"sample_{len(records) + i:05d}" for i in range(images.size(0))]

            target_size = tuple(labels.shape[-2:])
            logits = _forward(model, images, model_name, img_size, target_size)

            if num_classes == 1:
                preds = (torch.sigmoid(logits) > threshold).float()
            else:
                preds = torch.argmax(logits, dim=1, keepdim=True).float()

            imgs_np  = images.cpu().numpy()   # (B, C, H, W)
            preds_np = preds.cpu().numpy()    # (B, 1, H, W)
            lbls_np  = labels.cpu().numpy()   # (B, C, H, W)

            for i, fname in enumerate(fnames):
                pred_2d = preds_np[i].squeeze()

                if num_classes == 1:
                    gt_2d = lbls_np[i].squeeze()
                    metrics = _metrics_binary(pred_2d, gt_2d)
                else:
                    gt_2d = np.argmax(lbls_np[i], axis=0)
                    metrics = _metrics_multiclass(
                        pred_2d.astype(int), gt_2d.astype(int), num_classes
                    )

                # Visualisation: show the three core metrics in the panel title.
                vis_metrics = {k: metrics[k] for k in ("dice", "hd95", "iou") if k in metrics}
                img_disp = denormalize_image(imgs_np[i])
                # plot_segmentation_panel(
                #     img_disp,
                #     gt_2d,
                #     pred_2d,
                #     save_path=vis_dir / f"sample_{fname}.png",
                #     num_classes=num_classes,
                #     metrics=vis_metrics,
                #     filename=fname,
                # )

                records.append({"filename": fname, **metrics})

    return pd.DataFrame(records)


# ── Hydra entry point ──────────────────────────────────────────────────────────

# Register structured-config schemas before Hydra composes (validates the
# `data` group against config/schema.py; yaml remains the source of values).
register_schemas()


@hydra.main(version_base=None, config_path="config", config_name="infer")
def main(cfg: DictConfig) -> None:
    """Inference entry point: evaluate a checkpoint on test data and save results."""
    set_gpu(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Output directory ───────────────────────────────────────────────────────
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    mdl_name = str(cfg.model.get("name", "model")).lower()
    ds_tag   = cfg.data.get("name", "dataset")
    out_dir  = Path(cfg.infer.output_dir) / mdl_name / ds_tag / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir.resolve())

    # ── Build test data loaders ────────────────────────────────────────────────
    # build_dataset also builds train/val (needed for img_size sync side-effects)
    # but we only run inference on the test sets.
    _, _, test_ds_dict = SegDatasetProcessor.build_dataset(cfg)

    bs = int(cfg.infer.get("batch_size", 8))
    nw = int(cfg.infer.get("num_workers", 4))
    test_loaders = {
        name: DataLoader(
            ds,
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
        )
        for name, ds in test_ds_dict.items()
    }
    log.info(
        "Test sets: %s",
        ", ".join(f"{n}({len(ld.dataset)})" for n, ld in test_loaders.items()),
    )

    # ── Load model ─────────────────────────────────────────────────────────────
    model = _load_model(cfg, device)

    img_size    = int(cfg.data.get("img_size", 224))
    num_classes = int(cfg.data.get("num_classes", 1))
    threshold   = float(cfg.infer.get("threshold", 0.5))

    # ── Inference per test set ─────────────────────────────────────────────────
    all_dfs = []
    for name, loader in test_loaders.items():
        log.info("── %s  (%d samples) ──", name, len(loader.dataset))
        vis_dir = out_dir / "visualizations" / name

        df = _run_inference(
            model, loader, device, num_classes,
            mdl_name, img_size, threshold, vis_dir,
        )
        df.insert(0, "dataset", name)
        all_dfs.append(df)

        # Per-dataset CSV
        csv_path = out_dir / f"metrics_{name}.csv"
        df.to_csv(csv_path, index=False, float_format="%.6f")
        log.info("Saved per-sample metrics → %s", csv_path)

        # Summary statistics printed to log
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        summary = df[numeric_cols].agg(["mean", "std"])
        log.info("\n%s summary:\n%s", name, summary.to_string(float_format="%.4f"))
        log.info("%s latex (mean\\pmstd): %s", name, _format_latex_mean_std(df, numeric_cols))

    # ── Combined CSV when multiple test sets ───────────────────────────────────
    if len(all_dfs) > 1:
        combined_path = out_dir / "metrics_all.csv"
        pd.concat(all_dfs, ignore_index=True).to_csv(
            combined_path, index=False, float_format="%.6f"
        )
        log.info("Saved combined metrics → %s", combined_path)

    log.info("Done. All results in: %s", out_dir.resolve())


if __name__ == "__main__":
    main()
