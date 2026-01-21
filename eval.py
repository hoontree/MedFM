"""
Evaluation-only script for segmentation models.
Supports SAM and TinyUSFM models.

Usage:
    python eval.py checkpoint=/path/to/model.pth
    python eval.py checkpoint=/path/to/model.pth data=dynamic
    python eval.py checkpoint=/path/to/model.pth model.name=tinyusfm
"""

import os
import random
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from utils.data_processing_seg import SegDatasetProcessor
from utils.evaluate import Evaluator_seg
from utils.logger import setup_logger
from trainers.model_builder import ModelBuilder


# ImageNet normalization constants for denormalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def denormalize_image(img_tensor: np.ndarray) -> np.ndarray:
    """Denormalize image from ImageNet normalization."""
    if img_tensor.shape[0] == 3:
        img = img_tensor.transpose(1, 2, 0)
        img = img * IMAGENET_STD + IMAGENET_MEAN
        img = np.clip(img, 0, 1)
    else:
        img = img_tensor[0]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def get_model_short_name(cfg: DictConfig) -> str:
    """Create a short model identifier."""
    model_name = cfg.model.name.lower()

    if "sam" in model_name or "vit" in model_name:
        sam_type = cfg.model.get("sam_type", "vit_b")
        backbone = sam_type.replace("vit_", "")
        return f"sam_{backbone}"

    return model_name


def get_dataset_short_name(cfg: DictConfig) -> str:
    """Get dataset name, handling dynamic multi-dataset configs."""
    dataset_name = cfg.data.name
    if (
        dataset_name == "dynamic"
        and hasattr(cfg.data, "train")
        and isinstance(cfg.data.train, (list, ListConfig))
    ):
        dataset_name = "+".join(cfg.data.train)
    return dataset_name


def visualize_single_sample(
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    save_path: Path,
    sample_idx: int,
    metrics: dict = None,
):
    """Visualize a single sample with Image, GT, and Prediction."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Image
    axes[0].imshow(image)
    axes[0].set_title("Image", fontsize=12)
    axes[0].axis("off")

    # Ground Truth
    axes[1].imshow(image)
    axes[1].imshow(gt, cmap="jet", alpha=0.5)
    axes[1].set_title("Ground Truth", fontsize=12)
    axes[1].axis("off")

    # Prediction
    axes[2].imshow(image)
    axes[2].imshow(pred, cmap="jet", alpha=0.5)
    title = "Prediction"
    if metrics:
        title += f"\nDice: {metrics.get('dice', 0):.4f}"
    axes[2].set_title(title, fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_and_visualize(
    model,
    data_loader,
    device,
    num_classes: int,
    save_dir: Path,
    is_sam: bool = False,
    img_size: int = 224,
    threshold: float = 0.5,
    dataset_name: str = "test",
    logger=None,
):
    """
    Evaluate model and visualize ALL samples.

    Returns:
        dict: Aggregated metrics
    """
    model.eval()
    save_dir = Path(save_dir) / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Per-sample metrics lists
    dice_list = []
    hd95_list = []
    iou_list = []
    sensitivity_list = []
    specificity_list = []
    pixel_acc_list = []
    bf_score_list = []
    all_probs = []
    all_labels = []

    sample_idx = 0

    from medpy.metric.binary import hd95 as compute_hd95, dc, recall

    with torch.no_grad():
        for batch_idx, (images, masks, _) in enumerate(tqdm(data_loader, desc=f"Evaluating {dataset_name}")):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            if is_sam:
                outputs = model(images, False, img_size)
                logits = outputs["masks"]
            else:
                outputs = model(images)
                logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

            # Get predictions
            if num_classes == 2:
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).float()
            else:
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1, keepdim=True).float()

            # Process each sample in batch
            for i in range(images.size(0)):
                img_np = images[i].cpu().numpy()
                pred_np = preds[i].squeeze().cpu().numpy().astype(bool)
                gt_np = masks[i].cpu().numpy().astype(bool)

                # Compute per-sample metrics
                dice = dc(pred_np, gt_np)

                if pred_np.any() and gt_np.any():
                    hausdorff = compute_hd95(pred_np, gt_np)
                elif not pred_np.any() and not gt_np.any():
                    hausdorff = 0
                else:
                    hausdorff = img_size  # Max distance

                iou = Evaluator_seg.compute_jaccard(pred_np, gt_np)
                sens = recall(pred_np, gt_np)
                spec = Evaluator_seg.compute_specificity(pred_np, gt_np)
                pixel_acc = (pred_np == gt_np).sum() / gt_np.size
                bf_score = Evaluator_seg.compute_boundary_score(pred_np, gt_np)

                # Store metrics
                dice_list.append(dice)
                hd95_list.append(hausdorff)
                iou_list.append(iou)
                sensitivity_list.append(sens)
                specificity_list.append(spec)
                pixel_acc_list.append(pixel_acc)
                bf_score_list.append(bf_score)

                # Collect for ECE
                if num_classes == 2:
                    all_probs.append(probs[i].cpu().numpy().flatten())
                    all_labels.append(masks[i].cpu().numpy().flatten())

                # Visualize
                img_vis = denormalize_image(img_np)
                sample_metrics = {"dice": dice, "hd95": hausdorff, "iou": iou}

                vis_path = save_dir / f"sample_{sample_idx:04d}.png"
                visualize_single_sample(
                    img_vis,
                    gt_np.astype(float),
                    pred_np.astype(float),
                    vis_path,
                    sample_idx,
                    sample_metrics,
                )

                sample_idx += 1

    # Compute ECE
    if num_classes == 2 and all_probs:
        all_probs_flat = np.concatenate(all_probs)
        all_labels_flat = np.concatenate(all_labels)
        ece = Evaluator_seg.compute_ece(all_probs_flat, all_labels_flat)
    else:
        ece = 0.0

    # Aggregate metrics
    metrics = {
        "Dice": np.mean(dice_list),
        "Dice_std": np.std(dice_list),
        "HD95": np.mean(hd95_list),
        "HD95_std": np.std(hd95_list),
        "IoU": np.mean(iou_list),
        "IoU_std": np.std(iou_list),
        "Sensitivity": np.mean(sensitivity_list),
        "Sensitivity_std": np.std(sensitivity_list),
        "Specificity": np.mean(specificity_list),
        "Specificity_std": np.std(specificity_list),
        "PixelAcc": np.mean(pixel_acc_list),
        "PixelAcc_std": np.std(pixel_acc_list),
        "BFScore": np.mean(bf_score_list),
        "BFScore_std": np.std(bf_score_list),
        "ECE": ece,
        "num_samples": len(dice_list),
    }

    return metrics


def print_full_metrics(metrics: dict, dataset_name: str, logger):
    """Print all metrics in a formatted way."""
    logger.info("=" * 70)
    logger.info(f"EVALUATION RESULTS: {dataset_name}")
    logger.info("=" * 70)
    logger.info(f"  Samples: {metrics['num_samples']}")
    logger.info("-" * 70)
    logger.info(f"  Dice:        {metrics['Dice']:.4f} +/- {metrics['Dice_std']:.4f}")
    logger.info(f"  HD95:        {metrics['HD95']:.2f} +/- {metrics['HD95_std']:.2f}")
    logger.info(f"  IoU:         {metrics['IoU']:.4f} +/- {metrics['IoU_std']:.4f}")
    logger.info(f"  Sensitivity: {metrics['Sensitivity']:.4f} +/- {metrics['Sensitivity_std']:.4f}")
    logger.info(f"  Specificity: {metrics['Specificity']:.4f} +/- {metrics['Specificity_std']:.4f}")
    logger.info(f"  PixelAcc:    {metrics['PixelAcc']:.4f} +/- {metrics['PixelAcc_std']:.4f}")
    logger.info(f"  BFScore:     {metrics['BFScore']:.4f} +/- {metrics['BFScore_std']:.4f}")
    logger.info(f"  ECE:         {metrics['ECE']:.4f}")
    logger.info("=" * 70)


def save_metrics_to_file(all_metrics: dict, save_path: Path):
    """Save all metrics to a text file."""
    with open(save_path, "w") as f:
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 70 + "\n\n")

        for dataset_name, metrics in all_metrics.items():
            f.write(f"Dataset: {dataset_name}\n")
            f.write("-" * 50 + "\n")
            f.write(f"  Samples:     {metrics['num_samples']}\n")
            f.write(f"  Dice:        {metrics['Dice']:.4f} +/- {metrics['Dice_std']:.4f}\n")
            f.write(f"  HD95:        {metrics['HD95']:.2f} +/- {metrics['HD95_std']:.2f}\n")
            f.write(f"  IoU:         {metrics['IoU']:.4f} +/- {metrics['IoU_std']:.4f}\n")
            f.write(f"  Sensitivity: {metrics['Sensitivity']:.4f} +/- {metrics['Sensitivity_std']:.4f}\n")
            f.write(f"  Specificity: {metrics['Specificity']:.4f} +/- {metrics['Specificity_std']:.4f}\n")
            f.write(f"  PixelAcc:    {metrics['PixelAcc']:.4f} +/- {metrics['PixelAcc_std']:.4f}\n")
            f.write(f"  BFScore:     {metrics['BFScore']:.4f} +/- {metrics['BFScore_std']:.4f}\n")
            f.write(f"  ECE:         {metrics['ECE']:.4f}\n")
            f.write("\n")


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    # Validate checkpoint path
    checkpoint_path = cfg.get("checkpoint")
    if not checkpoint_path:
        raise ValueError("checkpoint path is required. Usage: python eval.py checkpoint=/path/to/model.pth")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")

    # Set environment and seed
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg.hardware.gpu_ids))
    set_seed(cfg.hardware.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup output directory
    model_short = get_model_short_name(cfg)
    dataset_name = get_dataset_short_name(cfg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(cfg.output.dir) / "eval" / model_short / dataset_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger(str(output_dir / "eval.log"))
    logger.info(f"Evaluation Script Started")
    logger.info(f"Model: {model_short}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Output: {output_dir}")

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # Load data
    logger.info("Loading datasets...")
    _, _, test_loader = SegDatasetProcessor.build_data_loaders(cfg)

    # Create model
    logger.info("Creating model...")
    model = ModelBuilder.create_model(cfg.model, cfg.data.num_classes, device)
    is_sam = hasattr(model, "image_encoder") or hasattr(model, "sam")
    img_size = cfg.model.get("img_size", 224)

    # Load checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    logger.info("Model loaded successfully")

    # Evaluate
    all_metrics = {}

    if isinstance(test_loader, dict):
        # Multiple test datasets
        for name, loader in test_loader.items():
            logger.info(f"\nEvaluating dataset: {name} ({len(loader.dataset)} samples)")

            metrics = evaluate_and_visualize(
                model=model,
                data_loader=loader,
                device=device,
                num_classes=cfg.data.num_classes,
                save_dir=vis_dir,
                is_sam=is_sam,
                img_size=img_size,
                dataset_name=name,
                logger=logger,
            )

            all_metrics[name] = metrics
            print_full_metrics(metrics, name, logger)
    else:
        # Single test dataset
        logger.info(f"\nEvaluating dataset: test ({len(test_loader.dataset)} samples)")

        metrics = evaluate_and_visualize(
            model=model,
            data_loader=test_loader,
            device=device,
            num_classes=cfg.data.num_classes,
            save_dir=vis_dir,
            is_sam=is_sam,
            img_size=img_size,
            dataset_name="test",
            logger=logger,
        )

        all_metrics["test"] = metrics
        print_full_metrics(metrics, "test", logger)

    # Save all metrics
    save_metrics_to_file(all_metrics, output_dir / "metrics.txt")
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
