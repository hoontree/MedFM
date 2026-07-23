"""
Unified visualization utilities for medical image segmentation.

This module provides reusable building blocks for segmentation visualization:
- Low-level helpers: denormalization, mask preparation, overlay creation
- Mid-level: panel rendering (segmentation, distillation comparison)
- High-level: batch visualization with model inference or cached predictions

All visualization functions share consistent styling (colormap, alpha, dpi, font).
"""

import torch
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

# ImageNet normalization (default for most models in this project)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Default visualization parameters
DEFAULT_CMAP = "jet"
DEFAULT_OVERLAY_ALPHA = 0.4
DEFAULT_DPI = 150
DEFAULT_FONT_SIZE = 14
DEFAULT_PANEL_WIDTH = 5  # width per panel in inches
DEFAULT_PANEL_HEIGHT = 5  # height per panel in inches


# ──────────────────────────────────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────────────────────────────────


def denormalize_image(
    img: np.ndarray,
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
) -> np.ndarray:
    """Denormalize a CHW image tensor (numpy) back to [0, 1] range for display.

    Args:
        img: Image array in CHW format (C=3 for RGB, C=1 for grayscale).
        mean: Per-channel mean used during normalization.
        std: Per-channel std used during normalization.

    Returns:
        HW or HWC image in [0, 1] range.
    """
    if img.shape[0] == 3:  # RGB
        img = img.transpose(1, 2, 0)
        img = std * img + mean
    elif img.shape[0] == 1:  # Grayscale
        img = img[0]
        img = std[0] * img + mean[0]
    else:
        # Already HW or unexpected shape – return as-is with min-max scaling
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return np.clip(img, 0, 1)


def prepare_mask_for_plot(mask: np.ndarray) -> np.ndarray:
    """Convert a mask array to a 2D (HW) index map for plotting.

    Args:
        mask: Mask array in (1, H, W), (C, H, W) one-hot, or (H, W) format.

    Returns:
        2D (HW) mask array with integer class indices.
    """
    if mask.ndim == 3 and mask.shape[0] == 1:
        return mask[0]
    if mask.ndim == 3 and mask.shape[0] > 1:
        # One-hot or multi-channel logits → argmax to index map
        return mask.argmax(axis=0).astype(np.float32)
    return mask


def create_overlay(
    pred: np.ndarray,
    num_classes: int = 1,
    alpha: float = DEFAULT_OVERLAY_ALPHA,
    cmap_name: str = DEFAULT_CMAP,
) -> np.ndarray:
    """Create an RGBA overlay from a prediction mask.

    Args:
        pred: 2D prediction mask (HW).
        num_classes: Number of segmentation classes.
        alpha: Overlay transparency.
        cmap_name: Matplotlib colormap name.

    Returns:
        RGBA array of shape (H, W, 4).
    """
    cmap = plt.get_cmap(cmap_name)
    normalized = pred / max(num_classes - 1, 1)
    colored = cmap(normalized)
    colored[..., 3] = alpha
    return colored


def log_wandb_images(
    wandb_images: list,
    phase_name: str = "test",
) -> None:
    """Log a list of WandB Images if WandB is active.

    Args:
        wandb_images: List of wandb.Image objects.
        phase_name: Phase prefix for the log key.
    """
    try:
        import wandb

        if wandb_images and wandb.run is not None:
            max_items = getattr(wandb.Image, "MAX_ITEMS", 108)
            wandb.log({f"{phase_name}/visualizations": wandb_images[:max_items]})
    except ImportError:
        pass


def _collect_wandb_image(
    save_path: Union[str, Path], caption: str
) -> Optional[object]:
    """Create a wandb.Image object, returning None if wandb is unavailable."""
    try:
        import wandb

        return wandb.Image(str(save_path), caption=caption)
    except ImportError:
        return None




# ──────────────────────────────────────────────────────────────────────
# Mid-level: panel rendering
# ──────────────────────────────────────────────────────────────────────


def plot_segmentation_panel(
    img: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    save_path: Union[str, Path],
    num_classes: int = 1,
    metrics: Optional[Dict[str, float]] = None,
    overlay: bool = False,
    dpi: int = DEFAULT_DPI,
    font_size: int = DEFAULT_FONT_SIZE,
    filename: Optional[str] = None,
) -> None:
    """Render a multi-panel segmentation figure: Image | GT | Prediction [| Overlay].

    Args:
        img: Denormalized image (HW for grayscale, HWC for RGB).
        gt: Ground truth mask (HW).
        pred: Prediction mask (HW).
        save_path: Path to save the figure.
        num_classes: Number of segmentation classes.
        metrics: Optional dict of metrics to display (e.g. {"dice": 0.85}).
        overlay: Whether to add a 4th overlay panel.
        dpi: Figure DPI.
        font_size: Title font size.
        filename: Optional source filename to display in the Image panel title.
    """
    n_panels = 4 if overlay else 3
    fig, axes = plt.subplots(
        1, n_panels, figsize=(DEFAULT_PANEL_WIDTH * n_panels, DEFAULT_PANEL_HEIGHT)
    )
    img_cmap = "gray" if img.ndim == 2 else None

    # Panel 1: Input Image
    image_title = f"Image\n{filename}" if filename else "Image"
    axes[0].imshow(img, cmap=img_cmap)
    axes[0].set_title(image_title, fontsize=font_size)
    axes[0].axis("off")

    # Panel 2: Ground Truth
    axes[1].imshow(img, cmap=img_cmap)
    axes[1].imshow(gt, cmap=DEFAULT_CMAP, alpha=DEFAULT_OVERLAY_ALPHA)
    axes[1].set_title("Ground Truth", fontsize=font_size)
    axes[1].axis("off")

    # Panel 3: Prediction
    pred_title = "Prediction"
    if metrics:
        metric_strs = [f"{k}: {v:.4f}" for k, v in metrics.items()]
        pred_title += "\n" + " | ".join(metric_strs)
    axes[2].imshow(img, cmap=img_cmap)
    axes[2].imshow(pred, cmap=DEFAULT_CMAP, alpha=DEFAULT_OVERLAY_ALPHA)
    axes[2].set_title(pred_title, fontsize=font_size)
    axes[2].axis("off")

    # Panel 4: Overlay (optional)
    if overlay:
        axes[3].imshow(img, cmap=img_cmap)
        colored_pred = create_overlay(pred, num_classes)
        axes[3].imshow(colored_pred)
        axes[3].set_title("Overlay", fontsize=font_size)
        axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_distillation_panel(
    img: np.ndarray,
    gt: np.ndarray,
    teacher_pred: np.ndarray,
    student_pred: np.ndarray,
    save_path: Union[str, Path],
    num_classes: int = 1,
    dpi: int = DEFAULT_DPI,
    font_size: int = DEFAULT_FONT_SIZE,
    filename: Optional[str] = None,
    labels: Tuple[str, str] = ("Teacher", "Student"),
) -> None:
    """Render a 4-panel distillation comparison: Image | GT | label0 | label1."""
    fig, axes = plt.subplots(
        1, 4, figsize=(DEFAULT_PANEL_WIDTH * 4, DEFAULT_PANEL_HEIGHT)
    )
    img_cmap = "gray" if img.ndim == 2 else None

    image_title = f"Image\n{filename}" if filename else "Image"
    axes[0].imshow(img, cmap=img_cmap)
    axes[0].set_title(image_title, fontsize=font_size)
    axes[0].axis("off")

    axes[1].imshow(img, cmap=img_cmap)
    axes[1].imshow(gt, cmap=DEFAULT_CMAP, alpha=DEFAULT_OVERLAY_ALPHA)
    axes[1].set_title("Ground Truth", fontsize=font_size)
    axes[1].axis("off")

    axes[2].imshow(img, cmap=img_cmap)
    axes[2].imshow(teacher_pred, cmap=DEFAULT_CMAP, alpha=DEFAULT_OVERLAY_ALPHA)
    axes[2].set_title(labels[0], fontsize=font_size)
    axes[2].axis("off")

    axes[3].imshow(img, cmap=img_cmap)
    axes[3].imshow(student_pred, cmap=DEFAULT_CMAP, alpha=DEFAULT_OVERLAY_ALPHA)
    axes[3].set_title(labels[1], fontsize=font_size)
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# High-level: batch visualization
# ──────────────────────────────────────────────────────────────────────


def _visualize_sample_arrays(
    img_chw: np.ndarray,
    mask: np.ndarray,
    pred: np.ndarray,
    save_path: Union[str, Path],
    num_classes: int,
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
    metrics: Optional[Dict[str, float]] = None,
    filename: Optional[str] = None,
) -> None:
    """Denormalize + prepare + render a single sample.

    Args:
        img_chw: Raw image in CHW format (pre-denormalization).
        mask: Ground truth mask array (possibly with channel dim).
        pred: Prediction mask array (possibly with channel dim).
        save_path: Output path.
        num_classes: Number of segmentation classes.
        mean: Denormalization mean.
        std: Denormalization std.
        metrics: Optional metrics to display.
        filename: Optional source filename to display.
    """
    img = denormalize_image(img_chw, mean, std)
    mask = prepare_mask_for_plot(mask)
    pred = prepare_mask_for_plot(pred)
    plot_segmentation_panel(
        img, mask, pred, save_path, num_classes, metrics=metrics, overlay=False, filename=filename,
    )


def visualize_segmentation(
    images_list: List[torch.Tensor],
    preds_list: List[Dict[str, torch.Tensor]],
    masks_list: List[torch.Tensor],
    num_classes: int,
    save_dir: Union[Path, str],
    num_samples: Optional[int] = None,
    phase_name: str = "test",
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
    filenames_list: Optional[List[List[str]]] = None,
    log_to_wandb: bool = True,
    max_wandb_images: Optional[int] = None,
    epoch: Optional[int] = None,
) -> None:
    """Render per-sample panels from precomputed predictions and (optionally) log to W&B.

    Each entry of ``preds_list`` is a dict mapping a label to a [B, 1, H, W] tensor:
    - 1 entry  -> standard segmentation panel (Image | GT | Prediction).
    - 2 entries -> distillation panel (Image | GT | first | second), labels become titles.

    Args:
        images_list: per-batch image tensors [B, C, H, W].
        preds_list:  per-batch prediction dicts; all batches must share the same key set.
        masks_list:  per-batch ground-truth masks [B, C, H, W] or [B, 1, H, W].
        num_classes: number of segmentation classes.
        save_dir:    output directory (a subdirectory ``epoch_{N}`` is added if epoch given).
        num_samples: max samples to render (None = all).
        phase_name:  W&B log key prefix.
        filenames_list: optional per-batch filename lists.
        log_to_wandb: upload rendered PNGs as a W&B Image gallery.
        max_wandb_images: cap on uploaded images (does not limit files saved to disk).
        epoch: if given, append ``epoch_{epoch+1}`` to ``save_dir``.
    """
    save_dir = Path(save_dir)
    if epoch is not None:
        save_dir = save_dir / f"epoch_{epoch + 1}"
    save_dir.mkdir(parents=True, exist_ok=True)

    pred_keys: Optional[List[str]] = None
    sample_count = 0
    wandb_images: list = []
    fnames_iter = iter(filenames_list) if filenames_list else None

    for batch_idx, (images, preds_dict, masks) in enumerate(
        zip(images_list, preds_list, masks_list)
    ):
        if pred_keys is None:
            pred_keys = list(preds_dict.keys())
            if len(pred_keys) not in (1, 2):
                raise ValueError(
                    f"preds_list entries must have 1 or 2 keys, got {len(pred_keys)}"
                )

        if num_classes > 1 and masks.shape[1] == num_classes:
            masks = torch.argmax(masks, dim=1, keepdim=True).float()

        batch_fnames = next(fnames_iter) if fnames_iter else None
        images_np = images.numpy() if isinstance(images, torch.Tensor) else images
        masks_np = masks.numpy() if isinstance(masks, torch.Tensor) else masks
        preds_np = {
            k: (v.numpy() if isinstance(v, torch.Tensor) else v)
            for k, v in preds_dict.items()
        }

        for i in range(images_np.shape[0]):
            fname = batch_fnames[i] if batch_fnames else None
            file_label = fname or f"{sample_count:03d}"
            save_path = save_dir / f"sample_{file_label}.png"

            img = denormalize_image(images_np[i], mean, std)
            gt = prepare_mask_for_plot(masks_np[i])
            sample_preds = [
                prepare_mask_for_plot(preds_np[k][i]) for k in pred_keys
            ]

            if len(pred_keys) == 1:
                plot_segmentation_panel(
                    img, gt, sample_preds[0], save_path, num_classes,
                    metrics=None, overlay=False, filename=fname,
                )
            else:
                plot_distillation_panel(
                    img, gt, sample_preds[0], sample_preds[1], save_path,
                    num_classes, filename=fname,
                    labels=(pred_keys[0].capitalize(), pred_keys[1].capitalize()),
                )

            if log_to_wandb and (
                max_wandb_images is None or len(wandb_images) < max_wandb_images
            ):
                caption = fname or f"Sample {sample_count}"
                wb_img = _collect_wandb_image(save_path, caption)
                if wb_img is not None:
                    wandb_images.append(wb_img)

            sample_count += 1
            if num_samples is not None and sample_count >= num_samples:
                break

        if num_samples is not None and sample_count >= num_samples:
            break

    if log_to_wandb:
        log_wandb_images(wandb_images, phase_name)

