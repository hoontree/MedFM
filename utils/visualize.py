"""
Unified visualization utilities for medical image segmentation.

This module provides reusable building blocks for segmentation visualization:
- Low-level helpers: denormalization, mask preparation, overlay creation
- Mid-level: panel rendering (segmentation, distillation comparison)
- High-level: batch visualization with model inference or cached predictions

All visualization functions share consistent styling (colormap, alpha, dpi, font).
"""

import torch
import torch.nn.functional as F
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
    """Squeeze a single-channel mask from CHW to HW for plotting.

    Args:
        mask: Mask array, possibly with a leading channel dim of 1.

    Returns:
        2D (HW) mask array.
    """
    if mask.ndim == 3 and mask.shape[0] == 1:
        return mask[0]
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
            wandb.log({f"{phase_name}/visualizations": wandb_images})
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


def _extract_filenames(batch) -> Optional[List[str]]:
    """Extract filenames from the last element of a dataloader batch.

    Returns a list of filename strings if the last batch element contains
    strings, otherwise None.
    """
    last = batch[-1]
    if isinstance(last, (list, tuple)) and len(last) > 0 and isinstance(last[0], str):
        return list(last)
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
    overlay: bool = True,
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
) -> None:
    """Render a 4-panel distillation comparison: Image | GT | Teacher | Student.

    Args:
        img: Denormalized image (HW or HWC).
        gt: Ground truth mask (HW).
        teacher_pred: Teacher prediction mask (HW).
        student_pred: Student prediction mask (HW).
        save_path: Path to save the figure.
        num_classes: Number of segmentation classes.
        dpi: Figure DPI.
        font_size: Title font size.
        filename: Optional source filename to display in the Image panel title.
    """
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
    axes[2].set_title("Teacher", fontsize=font_size)
    axes[2].axis("off")

    axes[3].imshow(img, cmap=img_cmap)
    axes[3].imshow(student_pred, cmap=DEFAULT_CMAP, alpha=DEFAULT_OVERLAY_ALPHA)
    axes[3].set_title("Student", fontsize=font_size)
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# High-level: batch visualization
# ──────────────────────────────────────────────────────────────────────


def _infer_predictions(
    model: torch.nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    model_type: str,
    num_classes: int,
    img_size: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run model inference and return (predictions, masks_for_vis).

    Returns:
        preds: [B, 1, H, W] binary/class predictions.
        masks_vis: [B, 1, H, W] ground truth for visualization.
    """
    # Convert multi-class masks to index form for visualization
    if num_classes > 1:
        masks_vis = torch.argmax(masks, dim=1, keepdim=True).float()
    else:
        masks_vis = masks

    if model_type == "sam":
        outputs = model(images, False, img_size)
        logits = outputs.get("masks", outputs.get("low_res_logits"))
        if logits is None:
            logits = list(outputs.values())[0]
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )
    elif model_type == "segformer":
        outputs = model(images)
        logits = outputs.logits
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )
    else:  # default (TinyUSFM, etc.)
        logits = model(images)
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)

    if num_classes == 1:
        preds = (torch.sigmoid(logits) > 0.5).float()
    else:
        preds = torch.argmax(logits, dim=1, keepdim=True).float()

    return preds, masks_vis


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
        img, mask, pred, save_path, num_classes, metrics=metrics, filename=filename,
    )


def visualize_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    save_dir: Union[Path, str],
    num_samples: Optional[int] = None,
    model_type: str = "default",
    img_size: Optional[int] = None,
    phase_name: str = "test",
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
) -> None:
    """Visualize model predictions with inference.

    Runs model forward pass on each batch and creates segmentation panels.

    Args:
        model: The torch model to evaluate.
        dataloader: DataLoader yielding (images, masks, ...).
        device: Device to run the model on.
        num_classes: Number of segmentation classes.
        save_dir: Directory to save visualization images.
        num_samples: Max samples to visualize (None = all).
        model_type: Model type ('default', 'sam', 'segformer').
        img_size: Image size for SAM model.
        phase_name: Phase name for WandB logging.
        mean: Denormalization mean.
        std: Denormalization std.
    """
    from tqdm import tqdm

    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    sample_count = 0
    wandb_images = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Visualizing {phase_name} predictions"):
            images = batch[0].to(device)
            masks = batch[1].to(device)
            filenames = _extract_filenames(batch)

            preds, masks_vis = _infer_predictions(
                model, images, masks, model_type, num_classes, img_size
            )

            images_np = images.cpu().numpy()
            masks_np = masks_vis.cpu().numpy()
            preds_np = preds.cpu().numpy()

            for i in range(images_np.shape[0]):
                fname = filenames[i] if filenames else None
                file_label = fname or f"{sample_count:03d}"
                save_path = save_dir / f"sample_{file_label}.png"
                _visualize_sample_arrays(
                    images_np[i], masks_np[i], preds_np[i],
                    save_path, num_classes, mean, std,
                    filename=fname,
                )

                caption = fname or f"Sample {sample_count}"
                wb_img = _collect_wandb_image(save_path, caption)
                if wb_img is not None:
                    wandb_images.append(wb_img)

                sample_count += 1
                if num_samples is not None and sample_count >= num_samples:
                    break

            if num_samples is not None and sample_count >= num_samples:
                break

    log_wandb_images(wandb_images, phase_name)


def visualize_from_predictions(
    images_list: List[torch.Tensor],
    preds_list: List[torch.Tensor],
    masks_list: List[torch.Tensor],
    num_classes: int,
    save_dir: Union[Path, str],
    num_samples: Optional[int] = None,
    phase_name: str = "test",
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
    filenames_list: Optional[List[List[str]]] = None,
) -> None:
    """Visualize pre-computed predictions (no model inference).

    Args:
        images_list: List of image tensors [B, C, H, W].
        preds_list: List of prediction tensors [B, 1, H, W].
        masks_list: List of mask tensors [B, C, H, W].
        num_classes: Number of segmentation classes.
        save_dir: Directory to save visualization images.
        num_samples: Max samples to visualize (None = all).
        phase_name: Phase name for WandB logging.
        mean: Denormalization mean.
        std: Denormalization std.
        filenames_list: Optional list of filename lists, one per batch.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    sample_count = 0
    wandb_images = []

    _fnames_iter = iter(filenames_list) if filenames_list else None

    for images, preds, masks in zip(images_list, preds_list, masks_list):
        if num_classes > 1 and masks.shape[1] == num_classes:
            masks = torch.argmax(masks, dim=1, keepdim=True).float()

        batch_fnames = next(_fnames_iter) if _fnames_iter else None

        images_np = images.numpy()
        preds_np = preds.numpy()
        masks_np = masks.numpy()

        for i in range(images_np.shape[0]):
            fname = batch_fnames[i] if batch_fnames else None
            file_label = fname or f"{sample_count:03d}"
            save_path = save_dir / f"sample_{file_label}.png"
            _visualize_sample_arrays(
                images_np[i], masks_np[i], preds_np[i],
                save_path, num_classes, mean, std,
                filename=fname,
            )

            caption = fname or f"Sample {sample_count}"
            wb_img = _collect_wandb_image(save_path, caption)
            if wb_img is not None:
                wandb_images.append(wb_img)

            sample_count += 1
            if num_samples is not None and sample_count >= num_samples:
                break

        if num_samples is not None and sample_count >= num_samples:
            break

    log_wandb_images(wandb_images, phase_name)


def visualize_distillation(
    teacher_model: torch.nn.Module,
    student_model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    teacher_img_size: int,
    save_dir: Union[Path, str],
    num_samples: int = 10,
    epoch: Optional[int] = None,
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
) -> None:
    """Visualize teacher vs student predictions side-by-side.

    Args:
        teacher_model: Teacher model (SAM-like or generic).
        student_model: Student model.
        test_loader: DataLoader yielding (images, masks, ...).
        device: Computation device.
        num_classes: Number of segmentation classes.
        teacher_img_size: Image size for SAM teacher inference.
        save_dir: Base directory for saving visualizations.
        num_samples: Max samples to visualize.
        epoch: Current epoch (None for final evaluation).
        mean: Denormalization mean.
        std: Denormalization std.
    """
    from tqdm import tqdm

    teacher_model.eval()
    student_model.eval()

    if epoch is not None:
        vis_dir = Path(save_dir) / f"epoch_{epoch + 1}"
    else:
        vis_dir = Path(save_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    sample_count = 0
    wandb_images = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Visualizing distillation"):
            images = batch[0].to(device)
            masks = batch[1].to(device)
            filenames = _extract_filenames(batch)

            # Teacher inference
            if hasattr(teacher_model, "image_encoder") or hasattr(
                teacher_model, "sam"
            ):
                teacher_outputs = teacher_model(images, False, teacher_img_size)
            else:
                teacher_outputs = {"masks": teacher_model(images)}

            # Student inference
            student_raw = student_model(images)
            student_logits = (
                student_raw[0] if isinstance(student_raw, tuple) else student_raw
            )
            teacher_logits = teacher_outputs["masks"]

            # Convert logits to predictions
            if num_classes == 1:
                teacher_preds = (torch.sigmoid(teacher_logits) > 0.5).float()
                student_preds = (torch.sigmoid(student_logits) > 0.5).float()
            else:
                teacher_preds = torch.argmax(
                    torch.softmax(teacher_logits, dim=1), dim=1, keepdim=True
                )
                student_preds = torch.argmax(
                    torch.softmax(student_logits, dim=1), dim=1, keepdim=True
                )

            for i in range(images.size(0)):
                if sample_count >= num_samples:
                    log_wandb_images(wandb_images, "distillation")
                    return

                fname = filenames[i] if filenames else None
                img = denormalize_image(images[i].cpu().numpy(), mean, std)
                t_pred = prepare_mask_for_plot(teacher_preds[i].cpu().numpy())
                s_pred = prepare_mask_for_plot(student_preds[i].cpu().numpy())
                gt = prepare_mask_for_plot(masks[i].cpu().numpy())

                file_label = fname or f"{sample_count:03d}"
                save_path = vis_dir / f"sample_{file_label}.png"
                plot_distillation_panel(
                    img, gt, t_pred, s_pred, save_path, num_classes,
                    filename=fname,
                )

                caption = fname or f"Sample {sample_count}"
                wb_img = _collect_wandb_image(save_path, caption)
                if wb_img is not None:
                    wandb_images.append(wb_img)

                sample_count += 1

    log_wandb_images(wandb_images, "distillation")
