import torch
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pandas as pd
from medpy.metric.binary import hd95, dc, precision, recall
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_erosion, generate_binary_structure


logger = logging.getLogger(__name__)


class Evaluator:
    @staticmethod
    def evaluate_model(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> Tuple[Dict[str, float], List[int], List[int]]:
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels, _ in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                num_classes = probs.size(1)

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = {}

        if num_classes == 1:
            try:
                auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
            except Exception as e:
                logging.warning(f"AUC calculation failed in binary classification: {e}")
                auc = 0.0

            metrics["accuracy"] = accuracy_score(all_labels, all_preds)
            metrics["precision"] = precision_score(
                all_labels, all_preds, pos_label=1, average="binary"
            )
            metrics["recall"] = recall_score(
                all_labels, all_preds, pos_label=1, average="binary"
            )
            metrics["f1"] = f1_score(
                all_labels, all_preds, pos_label=1, average="binary"
            )
            metrics["auc"] = auc
            metrics["mcc"] = matthews_corrcoef(all_labels, all_preds)
            metrics["b_accuracy"] = balanced_accuracy_score(all_labels, all_preds)

        elif num_classes > 2:
            try:
                auc = roc_auc_score(
                    all_labels, all_probs, multi_class="ovr", average="macro"
                )

            except Exception as e:

                logging.warning(
                    f"AUC calculation failed in multi-class classification: {e}"
                )
                auc = 0.0

            metrics["accuracy"] = accuracy_score(all_labels, all_preds)
            metrics["precision"] = precision_score(
                all_labels, all_preds, average="macro"
            )
            metrics["recall"] = recall_score(all_labels, all_preds, average="macro")
            metrics["f1"] = f1_score(all_labels, all_preds, average="macro")
            metrics["auc"] = auc
            metrics["mcc"] = matthews_corrcoef(all_labels, all_preds)
            metrics["b_accuracy"] = balanced_accuracy_score(all_labels, all_preds)
        else:
            raise ValueError(
                f"Invalid number of classes: {num_classes}. Check the model output."
            )

        return metrics, all_preds, all_labels

    @staticmethod
    def print_metrics(metrics: Dict[str, float], phase: str = "Validation") -> None:
        logger.info(
            f'{phase} Metrics: ACC: {metrics["accuracy"]:.4f}, Precision: {metrics["precision"]:.4f}, Recall: {metrics["recall"]:.4f}, F1: {metrics["f1"]:.4f}, AUC: {metrics["auc"]:.4f}, MCC: {metrics["mcc"]:.4f}, BACC: {metrics["b_accuracy"]:.4f}'
        )

    @staticmethod
    def generate_classification_report(
        labels: List[int], predictions: List[int], save_dir: str
    ) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        report = classification_report(labels, predictions)
        with open(save_dir / "classification_report.txt", "w") as f:
            f.write(report)
        logger.info("\nClassification Report:")
        logger.info(report)
        cm = confusion_matrix(labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig(save_dir / "confusion_matrix.png")
        plt.close()

    @staticmethod
    def save_predictions(
        image_names: List[str],
        predictions: List[int],
        labels: List[int],
        save_path: str,
    ) -> None:

        file_names = [Path(img_name).name for img_name in image_names]

        correctness = [
            "Correct" if pred == label else "Wrong"
            for pred, label in zip(predictions, labels)
        ]

        df = pd.DataFrame(
            {
                "image_name": file_names,
                "prediction": predictions,
                "label": labels,
                "correct": correctness,
            }
        )

        df.to_excel(save_path, index=False)

        print(f"Prediction save to {save_path}")


def _align_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Resize model logits to the GT mask's spatial size when they disagree.

    The model output and the loaded mask can differ in resolution (e.g. a SAM
    postprocess size vs. the dataset's mask size). Both the loss and the metrics
    are then computed at the GT resolution. A no-op when the sizes already match.
    """
    if logits.shape[-2:] == labels.shape[-2:]:
        return logits
    return torch.nn.functional.interpolate(
        logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
    )


class Evaluator_seg:
    @staticmethod
    def evaluate_batch(
        preds: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int = 1,
        threshold: float = 0.5,
    ) -> Dict[str, List[float]]:
        """
        Evaluate a single batch using unified channel-based approach.

        Args:
            preds: Model predictions [B, C, H, W] (logits)
            labels: Ground truth labels [B, C, H, W] (float 0/1)
            num_classes: Number of classes (C)
            threshold: Threshold for binary mask creation
        """
        if num_classes == 1:
            return Evaluator_seg._evaluate_batch_binary(preds, labels, threshold)
        else:
            # Multi-class: convert channel-based labels back to indices for the multi-class logic
            # Or refactor multi-class logic to handle channel-based labels
            labels_indices = torch.argmax(labels, dim=1)
            return Evaluator_seg._evaluate_batch_multiclass(
                preds, labels_indices, num_classes
            )

    @staticmethod
    def _evaluate_batch_binary(preds, labels, threshold):
        """Evaluate binary segmentation for a single batch."""
        dice_list = []
        hd95_list = []
        iou_list = []
        sensitivity_list = []
        specificity_list = []
        pixel_acc_list = []
        bf_score_list = []

        # Convert predictions to probabilities and binary masks
        probs = torch.sigmoid(preds)
        preds_binary = (probs > threshold).float()

        # Collect probs and labels for ECE calculation later
        all_probs = probs.cpu().numpy().flatten()
        all_labels_flat = labels.cpu().numpy().flatten()

        # Process each sample in batch
        for pred, gt in zip(preds_binary, labels):
            pred_np = pred.squeeze().cpu().numpy().astype(bool)
            gt_np = gt.squeeze().cpu().numpy().astype(bool)

            dice = dc(pred_np, gt_np)
            if pred_np.any() and gt_np.any():
                hausdorff = hd95(pred_np, gt_np)
            elif not pred_np.any() and not gt_np.any():
                hausdorff = 0
            else:
                hausdorff = float(max(gt_np.shape))
            iou = Evaluator_seg.compute_jaccard(pred_np, gt_np)
            sens = recall(pred_np, gt_np)
            spec = Evaluator_seg.compute_specificity(pred_np, gt_np)
            pixel_acc = (pred_np == gt_np).sum() / gt_np.size
            bf_score = Evaluator_seg.compute_boundary_score(pred_np, gt_np)

            dice_list.append(dice)
            hd95_list.append(hausdorff)
            iou_list.append(iou)
            sensitivity_list.append(sens)
            specificity_list.append(spec)
            pixel_acc_list.append(pixel_acc)
            bf_score_list.append(bf_score)

        return {
            "dice": dice_list,
            "hd95": hd95_list,
            "iou": iou_list,
            "sensitivity": sensitivity_list,
            "specificity": specificity_list,
            "pixel_acc": pixel_acc_list,
            "bf_score": bf_score_list,
            "probs": all_probs,
            "labels": all_labels_flat,
        }

    @staticmethod
    def _evaluate_batch_multiclass(preds, labels, num_classes):
        """Evaluate multiclass segmentation for a single batch."""
        dice_per_class = [[] for _ in range(num_classes)]
        iou_per_class = [[] for _ in range(num_classes)]
        pixel_acc_list = []
        sensitivity_per_class = [[] for _ in range(num_classes)]
        specificity_per_class = [[] for _ in range(num_classes)]
        hd95_per_class = [[] for _ in range(num_classes)]
        bfscore_per_class = [[] for _ in range(num_classes)]

        # Convert to class predictions
        preds_class = torch.argmax(preds, dim=1)  # [B, H, W]

        for pred, gt in zip(preds_class, labels):
            pred_np = pred.cpu().numpy()
            gt_np = gt.cpu().numpy()

            pixel_acc = (pred_np == gt_np).mean()
            pixel_acc_list.append(pixel_acc)

            for class_id in range(num_classes):
                pred_class = pred_np == class_id
                gt_class = gt_np == class_id

                # Only evaluate samples where gt contains this class
                if not gt_class.any():
                    continue

                tp = np.logical_and(pred_class, gt_class).sum()
                dice = 2 * tp / (pred_class.sum() + gt_class.sum() + 1e-8)
                iou = Evaluator_seg.compute_jaccard(pred_class, gt_class)
                fn = np.logical_and(~pred_class, gt_class).sum()
                sensitivity = tp / (tp + fn + 1e-8)
                tn = np.logical_and(~pred_class, ~gt_class).sum()
                fp = np.logical_and(pred_class, ~gt_class).sum()
                specificity = tn / (tn + fp + 1e-8)
                if pred_class.any():
                    hd = hd95(pred_class.astype(np.bool_), gt_class.astype(np.bool_))
                else:
                    hd = float(max(gt_np.shape))
                bfscore = Evaluator_seg.compute_boundary_score(pred_class, gt_class)

                dice_per_class[class_id].append(dice)
                iou_per_class[class_id].append(iou)
                sensitivity_per_class[class_id].append(sensitivity)
                specificity_per_class[class_id].append(specificity)
                hd95_per_class[class_id].append(hd)
                bfscore_per_class[class_id].append(bfscore)

        return {
            "dice_per_class": dice_per_class,
            "iou_per_class": iou_per_class,
            "pixel_acc": pixel_acc_list,
            "sensitivity_per_class": sensitivity_per_class,
            "specificity_per_class": specificity_per_class,
            "hd95_per_class": hd95_per_class,
            "bfscore_per_class": bfscore_per_class,
        }

    @staticmethod
    def aggregate_metrics(
        batch_metrics: List[Dict], num_classes: int = 1
    ) -> Dict[str, float]:
        """
        Aggregate batch metrics into epoch metrics for Lightning.

        Args:
            batch_metrics: List of metric dictionaries from evaluate_batch
            num_classes: Number of classes

        Returns:
            Aggregated metrics dictionary
        """
        if num_classes == 1:
            return Evaluator_seg._aggregate_binary(batch_metrics)
        else:
            return Evaluator_seg._aggregate_multiclass(batch_metrics, num_classes)

    @staticmethod
    def _aggregate_binary(batch_metrics):
        """Aggregate binary metrics from multiple batches."""
        all_dice = []
        all_hd95 = []
        all_iou = []
        all_sensitivity = []
        all_specificity = []
        all_pixel_acc = []
        all_bf_score = []
        all_probs = []
        all_labels = []

        for batch in batch_metrics:
            all_dice.extend(batch["dice"])
            all_hd95.extend(batch["hd95"])
            all_iou.extend(batch["iou"])
            all_sensitivity.extend(batch["sensitivity"])
            all_specificity.extend(batch["specificity"])
            all_pixel_acc.extend(batch["pixel_acc"])
            all_bf_score.extend(batch["bf_score"])
            all_probs.append(batch["probs"])
            all_labels.append(batch["labels"])

        # Compute ECE
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        ece = Evaluator_seg.compute_ece(all_probs, all_labels)

        return {
            "Dice": np.mean(all_dice),
            "Dice_std": np.std(all_dice),
            "HD95": np.mean(all_hd95),
            "HD95_std": np.std(all_hd95),
            "IoU": np.mean(all_iou),
            "IoU_std": np.std(all_iou),
            "Sensitivity": np.mean(all_sensitivity),
            "Sensitivity_std": np.std(all_sensitivity),
            "Specificity": np.mean(all_specificity),
            "Specificity_std": np.std(all_specificity),
            "PixelAcc": np.mean(all_pixel_acc),
            "PixelAcc_std": np.std(all_pixel_acc),
            "BFScore": np.mean(all_bf_score),
            "BFScore_std": np.std(all_bf_score),
            "ECE": ece,
        }

    @staticmethod
    def _aggregate_multiclass(batch_metrics, num_classes):
        """Aggregate multiclass metrics from multiple batches."""
        all_dice_per_class = [[] for _ in range(num_classes)]
        all_iou_per_class = [[] for _ in range(num_classes)]
        all_pixel_acc = []
        all_sensitivity_per_class = [[] for _ in range(num_classes)]
        all_specificity_per_class = [[] for _ in range(num_classes)]
        all_hd95_per_class = [[] for _ in range(num_classes)]
        all_bfscore_per_class = [[] for _ in range(num_classes)]

        for batch in batch_metrics:
            all_pixel_acc.extend(batch["pixel_acc"])
            for class_id in range(num_classes):
                all_dice_per_class[class_id].extend(batch["dice_per_class"][class_id])
                all_iou_per_class[class_id].extend(batch["iou_per_class"][class_id])
                all_sensitivity_per_class[class_id].extend(
                    batch["sensitivity_per_class"][class_id]
                )
                all_specificity_per_class[class_id].extend(
                    batch["specificity_per_class"][class_id]
                )
                all_hd95_per_class[class_id].extend(batch["hd95_per_class"][class_id])
                all_bfscore_per_class[class_id].extend(batch["bfscore_per_class"][class_id])

        foreground_ids = list(range(1, num_classes))

        return {
            "PixelAcc": np.nanmean(all_pixel_acc),
            "PixelAcc_std": np.nanstd(all_pixel_acc),
            "Dice": np.nanmean(
                [np.nanmean(all_dice_per_class[i]) for i in foreground_ids]
            ),
            "Dice_std": np.nanstd(
                [item for i in foreground_ids for item in all_dice_per_class[i]]
            ),
            "IoU": np.nanmean(
                [np.nanmean(all_iou_per_class[i]) for i in foreground_ids]
            ),
            "IoU_std": np.nanstd(
                [item for i in foreground_ids for item in all_iou_per_class[i]]
            ),
            "Sensitivity": np.nanmean(
                [np.nanmean(all_sensitivity_per_class[i]) for i in foreground_ids]
            ),
            "Sensitivity_std": np.nanstd(
                [item for i in foreground_ids for item in all_sensitivity_per_class[i]]
            ),
            "Specificity": np.nanmean(
                [np.nanmean(all_specificity_per_class[i]) for i in foreground_ids]
            ),
            "Specificity_std": np.nanstd(
                [item for i in foreground_ids for item in all_specificity_per_class[i]]
            ),
            "HD95": np.nanmean(
                [np.nanmean(all_hd95_per_class[i]) for i in foreground_ids]
            ),
            "HD95_std": np.nanstd(
                [item for i in foreground_ids for item in all_hd95_per_class[i]]
            ),
            "BFScore": np.nanmean(
                [np.nanmean(all_bfscore_per_class[i]) for i in foreground_ids]
            ),
            "BFScore_std": np.nanstd(
                [item for i in foreground_ids for item in all_bfscore_per_class[i]]
            ),
        }

    @staticmethod
    def evaluate_model(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        num_classes: int = 1,
        threshold: float = 0.5,
        return_predictions: bool = False,
        criterion=None,
    ) -> Dict[str, float] | tuple:
        model.eval()
        if num_classes == 1:
            return Evaluator_seg._evaluate_binary(
                model, data_loader, device, threshold, return_predictions, criterion=criterion
            )
        else:
            return Evaluator_seg._evaluate_multiclass(
                model, data_loader, device, num_classes, return_predictions, criterion=criterion
            )

    @staticmethod
    def evaluate_model_sam(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        num_classes: int = 1,
        img_size: int = 256,
        threshold: float = 0.5,
        return_predictions: bool = False,
    ) -> Dict[str, float] | tuple:
        """Evaluate SAM-based models that require img_size parameter."""
        model.eval()
        if num_classes == 1:
            return Evaluator_seg._evaluate_binary(
                model, data_loader, device, threshold, return_predictions, img_size=img_size
            )
        else:
            return Evaluator_seg._evaluate_multiclass(
                model, data_loader, device, num_classes, return_predictions, img_size=img_size
            )

    @staticmethod
    def _evaluate_binary(
        model, data_loader, device, threshold, return_predictions=False,
        img_size=None, criterion=None,
    ):
        dice_list = []
        hd95_list = []
        iou_list = []
        sensitivity_list = []
        specificity_list = []
        pixel_acc_list = []
        boundary_iou_list =  []
        bf_score_list = []
        per_sample_filenames = [] if return_predictions else None
        all_probs = []
        all_labels = []

        images_list = [] if return_predictions else None
        preds_list = [] if return_predictions else None
        masks_list = [] if return_predictions else None
        filenames_list = [] if return_predictions else None

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                images = batch[0].to(device)
                labels = batch[1].to(device)
                batch_fnames = list(batch[-1]) if isinstance(batch[-1], (list, tuple)) and batch[-1] and isinstance(batch[-1][0], str) else None

                if img_size is not None:
                    raw_out = model(images, False, img_size)
                    logits = raw_out["masks"]
                else:
                    logits = model(images)
                    if logits.shape[1] != 1:
                        raise ValueError("Only binary segmentation supported.")

                logits = _align_logits(logits, labels)

                if criterion is not None:
                    loss = criterion(logits, labels.float())
                    total_loss += loss.item() * images.size(0)
                    total_samples += images.size(0)

                probs = torch.sigmoid(logits)
                preds = (probs > threshold).float()

                # Collect for ECE
                all_probs.append(probs.cpu().numpy().flatten())
                all_labels.append(labels.cpu().numpy().flatten())

                if return_predictions:
                    images_list.append(images.cpu())
                    preds_list.append(preds.cpu())
                    masks_list.append(labels.cpu())
                    filenames_list.append(batch_fnames)

                for sample_idx, (pred, gt) in enumerate(zip(preds, labels)):
                    pred_np = pred.squeeze().cpu().numpy().astype(bool)
                    gt_np = gt.squeeze().cpu().numpy().astype(bool)

                    dice = dc(pred_np, gt_np)
                    if pred_np.any() and gt_np.any():
                        hausdorff = hd95(pred_np, gt_np)
                    elif not pred_np.any() and not gt_np.any():
                        hausdorff = 0
                    else:
                        hausdorff = float(max(gt_np.shape))
                    iou = Evaluator_seg.compute_jaccard(pred_np, gt_np)
                    biou = Evaluator_seg.boundary_iou(pred_np, gt_np)
                    bf = Evaluator_seg.compute_boundary_score(pred_np, gt_np)
                    sens = recall(pred_np, gt_np)
                    spec = Evaluator_seg.compute_specificity(pred_np, gt_np)
                    pixel_acc = (pred_np == gt_np).sum() / gt_np.size

                    dice_list.append(dice)
                    hd95_list.append(hausdorff)
                    iou_list.append(iou)
                    sensitivity_list.append(sens)
                    specificity_list.append(spec)
                    pixel_acc_list.append(pixel_acc)
                    boundary_iou_list.append(biou)
                    bf_score_list.append(bf)

                    if return_predictions:
                        if batch_fnames is not None and sample_idx < len(batch_fnames):
                            per_sample_filenames.append(batch_fnames[sample_idx])
                        else:
                            per_sample_filenames.append(f"sample_{len(per_sample_filenames):05d}")
                # Cleanup batch-level tensors
                if img_size is not None:
                    del raw_out
                del images, labels, logits, probs, preds

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        ece = Evaluator_seg.compute_ece(all_probs, all_labels)

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
            "ECE": ece,
            "BoundaryIoU": np.mean(boundary_iou_list),
            "BoundaryIoU_std": np.std(boundary_iou_list),
            "BFScore": np.mean(bf_score_list),
            "BFScore_std": np.std(bf_score_list),
        }
        if criterion is not None:
            metrics["loss"] = total_loss / total_samples if total_samples > 0 else 0.0
        if return_predictions:
            per_sample = {
                "filename": per_sample_filenames,
                "Dice": dice_list,
                "HD95": hd95_list,
                "IoU": iou_list,
                "BoundaryIoU": boundary_iou_list,
                "BFScore": bf_score_list,
                "Sensitivity": sensitivity_list,
                "Specificity": specificity_list,
                "PixelAcc": pixel_acc_list,
            }
            return metrics, images_list, preds_list, masks_list, filenames_list, per_sample
        return metrics

    @staticmethod
    def _evaluate_multiclass(
        model, data_loader, device, num_classes, return_predictions=False,
        img_size=None, criterion=None,
    ):
        dice_per_class = [[] for _ in range(num_classes)]
        boundary_iou_per_class = [[] for _ in range(num_classes)]
        bf_score_per_class = [[] for _ in range(num_classes)]
        iou_per_class = [[] for _ in range(num_classes)]
        pixel_acc_list = []
        sensitivity_per_class = [[] for _ in range(num_classes)]
        specificity_per_class = [[] for _ in range(num_classes)]
        hd95_per_class = [[] for _ in range(num_classes)]

        images_list = [] if return_predictions else None
        preds_list = [] if return_predictions else None
        masks_list = [] if return_predictions else None
        filenames_list = [] if return_predictions else None

        # Per-sample mean across foreground classes (for CSV).
        ps_filenames = [] if return_predictions else None
        ps_dice = [] if return_predictions else None
        ps_hd95 = [] if return_predictions else None
        ps_iou = [] if return_predictions else None
        ps_biou = [] if return_predictions else None
        ps_bf = [] if return_predictions else None
        ps_sens = [] if return_predictions else None
        ps_spec = [] if return_predictions else None
        ps_pixacc = [] if return_predictions else None
        ps_per_class = [] if return_predictions else None  # list of dicts: {class_id: {Dice, HD95, ...}}

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                images = batch[0].to(device)
                labels = batch[1].to(device)
                batch_fnames = list(batch[-1]) if isinstance(batch[-1], (list, tuple)) and batch[-1] and isinstance(batch[-1][0], str) else None
                # labels can be one-hot [B, num_classes, H, W] or index map [B, 1, H, W] / [B, H, W]
                if labels.dim() == 4 and labels.shape[1] == num_classes:
                    gt_indices = torch.argmax(labels, dim=1)  # [B, H, W]
                else:
                    gt_indices = labels.squeeze(1).long()  # [B, H, W]

                if img_size is not None:
                    raw_out = model(images, True, img_size)
                    output_masks = raw_out["masks"]
                else:
                    output_masks = model(images)  # [B, num_classes, H, W]

                output_masks = _align_logits(output_masks, gt_indices)
                preds = torch.argmax(output_masks, dim=1)  # [B, H, W]

                if criterion is not None:
                    loss = criterion(output_masks, gt_indices.long())
                    total_loss += loss.item() * images.size(0)
                    total_samples += images.size(0)

                if return_predictions:
                    images_list.append(images.cpu())
                    preds_list.append(preds.unsqueeze(1).cpu())
                    masks_list.append(labels.cpu())
                    filenames_list.append(batch_fnames)

                for sample_idx, (pred, gt) in enumerate(zip(preds, gt_indices)):
                    pred_np = pred.cpu().numpy()
                    gt_np = gt.cpu().numpy()

                    pixel_acc = (pred_np == gt_np).mean()
                    pixel_acc_list.append(pixel_acc)

                    sample_dice, sample_hd, sample_iou = [], [], []
                    sample_sens, sample_spec, sample_biou = [], [], []
                    sample_bf = []
                    sample_per_class = {}
                    for class_id in range(num_classes):
                        pred_class = pred_np == class_id
                        gt_class = gt_np == class_id

                        # Only evaluate samples where gt contains this class
                        if not gt_class.any():
                            continue

                        tp = np.logical_and(pred_class, gt_class).sum()
                        dice = 2 * tp / (pred_class.sum() + gt_class.sum() + 1e-8)
                        iou = Evaluator_seg.compute_jaccard(pred_class, gt_class)
                        biou = Evaluator_seg.boundary_iou(pred_class, gt_class)
                        bf = Evaluator_seg.compute_boundary_score(pred_class, gt_class)
                        fn = np.logical_and(~pred_class, gt_class).sum()
                        sensitivity = tp / (tp + fn + 1e-8)
                        tn = np.logical_and(~pred_class, ~gt_class).sum()
                        fp = np.logical_and(pred_class, ~gt_class).sum()
                        specificity = tn / (tn + fp + 1e-8)
                        if pred_class.any():
                            hd = hd95(
                                pred_class.astype(np.bool_),
                                gt_class.astype(np.bool_),
                            )
                        else:
                            hd = float(max(gt_np.shape))

                        dice_per_class[class_id].append(dice)
                        iou_per_class[class_id].append(iou)
                        sensitivity_per_class[class_id].append(sensitivity)
                        specificity_per_class[class_id].append(specificity)
                        hd95_per_class[class_id].append(hd)
                        boundary_iou_per_class[class_id].append(biou)
                        bf_score_per_class[class_id].append(bf)

                        if class_id >= 1:
                            sample_dice.append(float(dice))
                            sample_hd.append(float(hd))
                            sample_iou.append(float(iou))
                            sample_biou.append(float(biou))
                            sample_bf.append(float(bf))
                            sample_sens.append(float(sensitivity))
                            sample_spec.append(float(specificity))
                            sample_per_class[class_id] = {
                                "Dice": float(dice), "HD95": float(hd),
                                "IoU": float(iou), "BoundaryIoU": float(biou),
                                "BFScore": float(bf),
                                "Sensitivity": float(sensitivity),
                                "Specificity": float(specificity),
                            }

                    if return_predictions:
                        if batch_fnames is not None and sample_idx < len(batch_fnames):
                            ps_filenames.append(batch_fnames[sample_idx])
                        else:
                            ps_filenames.append(f"sample_{len(ps_filenames):05d}")
                        ps_dice.append(float(np.mean(sample_dice)) if sample_dice else 0.0)
                        ps_hd95.append(float(np.mean(sample_hd)) if sample_hd else 224.0)
                        ps_iou.append(float(np.mean(sample_iou)) if sample_iou else 0.0)
                        ps_biou.append(float(np.mean(sample_biou)) if sample_biou else 0.0)
                        ps_bf.append(float(np.mean(sample_bf)) if sample_bf else 0.0)
                        ps_sens.append(float(np.mean(sample_sens)) if sample_sens else float("nan"))
                        ps_spec.append(float(np.mean(sample_spec)) if sample_spec else float("nan"))
                        ps_pixacc.append(float(pixel_acc))
                        ps_per_class.append(sample_per_class)

        foreground_ids = list(range(1, num_classes))

        metrics = {}
        metrics["PixelAcc"] = np.nanmean(pixel_acc_list)
        metrics["PixelAcc_std"] = np.nanstd(pixel_acc_list)
        metrics["Dice"] = np.nanmean(
            [np.nanmean(dice_per_class[i]) for i in foreground_ids]
        )
        metrics["Dice_std"] = np.nanstd(
            [item for i in foreground_ids for item in dice_per_class[i]]
        )
        metrics["IoU"] = np.nanmean(
            [np.nanmean(iou_per_class[i]) for i in foreground_ids]
        )
        metrics["IoU_std"] = np.nanstd(
            [item for i in foreground_ids for item in iou_per_class[i]]
        )
        metrics["Sensitivity"] = np.nanmean(
            [np.nanmean(sensitivity_per_class[i]) for i in foreground_ids]
        )
        metrics["Sensitivity_std"] = np.nanstd(
            [item for i in foreground_ids for item in sensitivity_per_class[i]]
        )
        metrics["Specificity"] = np.nanmean(
            [np.nanmean(specificity_per_class[i]) for i in foreground_ids]
        )
        metrics["Specificity_std"] = np.nanstd(
            [item for i in foreground_ids for item in specificity_per_class[i]]
        )
        metrics["HD95"] = np.nanmean(
            [np.nanmean(hd95_per_class[i]) for i in foreground_ids]
        )
        metrics["HD95_std"] = np.nanstd(
            [item for i in foreground_ids for item in hd95_per_class[i]]
        )
        metrics["BoundaryIoU"] = np.nanmean(
            [np.nanmean(boundary_iou_per_class[i]) for i in foreground_ids]
        )
        metrics["BoundaryIoU_std"] = np.nanstd(
            [item for i in foreground_ids for item in boundary_iou_per_class[i]]
        )
        metrics["BFScore"] = np.nanmean(
            [np.nanmean(bf_score_per_class[i]) for i in foreground_ids]
        )
        metrics["BFScore_std"] = np.nanstd(
            [item for i in foreground_ids for item in bf_score_per_class[i]]
        )

        # Per-class metrics for each foreground class
        for i in foreground_ids:
            metrics[f"Dice_class_{i}"] = np.nanmean(dice_per_class[i])
            metrics[f"Dice_class_{i}_std"] = np.nanstd(dice_per_class[i])
            metrics[f"IoU_class_{i}"] = np.nanmean(iou_per_class[i])
            metrics[f"IoU_class_{i}_std"] = np.nanstd(iou_per_class[i])
            metrics[f"HD95_class_{i}"] = np.nanmean(hd95_per_class[i])
            metrics[f"HD95_class_{i}_std"] = np.nanstd(hd95_per_class[i])
            metrics[f"Sensitivity_class_{i}"] = np.nanmean(sensitivity_per_class[i])
            metrics[f"Sensitivity_class_{i}_std"] = np.nanstd(sensitivity_per_class[i])
            metrics[f"Specificity_class_{i}"] = np.nanmean(specificity_per_class[i])
            metrics[f"Specificity_class_{i}_std"] = np.nanstd(specificity_per_class[i])
            metrics[f"BoundaryIoU_class_{i}"] = np.nanmean(boundary_iou_per_class[i])
            metrics[f"BoundaryIoU_class_{i}_std"] = np.nanstd(boundary_iou_per_class[i])
            metrics[f"BFScore_class_{i}"] = np.nanmean(bf_score_per_class[i])
            metrics[f"BFScore_class_{i}_std"] = np.nanstd(bf_score_per_class[i])

        if criterion is not None:
            metrics["loss"] = total_loss / total_samples if total_samples > 0 else 0.0

        if return_predictions:
            per_sample = {
                "filename": ps_filenames,
                "Dice": ps_dice,
                "HD95": ps_hd95,
                "IoU": ps_iou,
                "BoundaryIoU": ps_biou,
                "BFScore": ps_bf,
                "Sensitivity": ps_sens,
                "Specificity": ps_spec,
                "PixelAcc": ps_pixacc,
                "_per_class": ps_per_class,
            }
            return metrics, images_list, preds_list, masks_list, filenames_list, per_sample
        return metrics

    @staticmethod
    def compute_specificity(pred: np.ndarray, gt: np.ndarray) -> float:
        tn = np.logical_and(pred == 0, gt == 0).sum()
        fp = np.logical_and(pred == 1, gt == 0).sum()
        return tn / (tn + fp + 1e-8)

    @staticmethod
    def compute_jaccard(pred, gt):
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        return intersection / union
    
    @staticmethod
    def mask_to_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
        """
        Binary mask를 boundary region으로 변환한다.

        Args:
            mask: shape (H, W), 0/1 또는 bool 배열
            dilation_ratio: boundary 두께를 이미지 대각선 대비 비율로 지정
                            논문 기본값은 대략 0.02

        Returns:
            boundary_mask: shape (H, W), bool 배열
        """
        if mask.ndim != 2:
            raise ValueError("mask must be a 2D array")

        mask = mask.astype(bool)
        h, w = mask.shape

        # 이미지 diagonal 기준으로 boundary 두께 d 계산
        img_diag = np.sqrt(h ** 2 + w ** 2)
        d = max(1, int(round(dilation_ratio * img_diag)))

        # erosion 후 원본과의 차이를 이용해 내부 기준 boundary region 생성
        # 논문의 (G_d ∩ G), (P_d ∩ P)에 해당하는 실용적 구현
        eroded = binary_erosion(mask, structure=np.ones((3, 3)), iterations=d, border_value=0)
        boundary = mask ^ eroded
        return boundary

    @staticmethod
    def boundary_iou(gt: np.ndarray, pred: np.ndarray, dilation_ratio: float = 0.02) -> float:
        """
        Boundary IoU 계산.

        Args:
            gt: ground truth binary mask, shape (H, W)
            pred: prediction binary mask, shape (H, W)
            dilation_ratio: boundary 두께 비율

        Returns:
            Boundary IoU score (float)
        """
        if gt.shape != pred.shape:
            raise ValueError("gt and pred must have the same shape")

        gt_boundary = Evaluator_seg.mask_to_boundary(gt, dilation_ratio)
        pred_boundary = Evaluator_seg.mask_to_boundary(pred, dilation_ratio)

        intersection = np.logical_and(gt_boundary, pred_boundary).sum()
        union = np.logical_or(gt_boundary, pred_boundary).sum()

        # 둘 다 비어 있는 특수 케이스
        if union == 0:
            return 1.0

        return intersection / union

    @staticmethod
    def compute_boundary_score(pred, gt, theta=2):
        """
        Compute Boundary F-Score.
        """
        # Get boundaries
        gt_boundary = Evaluator_seg._get_boundary(gt)
        pred_boundary = Evaluator_seg._get_boundary(pred)

        if not gt_boundary.any() and not pred_boundary.any():
            return 1.0
        if not gt_boundary.any() or not pred_boundary.any():
            return 0.0

        # Distance transform
        gt_dt = distance_transform_edt(~gt_boundary)
        pred_dt = distance_transform_edt(~pred_boundary)

        # Precision and Recall
        precision = np.sum(pred_boundary * (gt_dt < theta)) / (
            np.sum(pred_boundary) + 1e-8
        )
        recall = np.sum(gt_boundary * (pred_dt < theta)) / (np.sum(gt_boundary) + 1e-8)

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _get_boundary(mask):
        """
        Extract boundary from binary mask.
        """
        mask = mask.astype(bool)
        struct = generate_binary_structure(2, 1)
        eroded = binary_erosion(mask, struct)
        boundary = mask ^ eroded
        return boundary

    @staticmethod
    def compute_ece(probs, labels, n_bins=10):
        """
        Compute Expected Calibration Error.
        probs: [N] flattened probabilities
        labels: [N] flattened labels (0 or 1)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy = labels[in_bin].mean()
                avg_confidence = probs[in_bin].mean()
                ece += np.abs(avg_confidence - accuracy) * prop_in_bin

        return ece

    @staticmethod
    def print_metrics(metrics: Dict[str, float], phase: str = "Validation") -> None:
        # Detect per-class keys (e.g. Dice_class_1)
        class_ids = sorted({
            int(k.split("_class_")[1].split("_")[0])
            for k in metrics
            if "_class_" in k and not k.endswith("_std")
            and k.split("_class_")[1].split("_")[0].isdigit()
        })

        if phase.lower().startswith("test"):
            logger.info(
                f'{phase} Metrics - Dice: {metrics["Dice"]:.4f}±{metrics["Dice_std"]:.4f}, '
                f'HD95: {metrics["HD95"]:.2f}±{metrics["HD95_std"]:.2f}, '
                f'BFScore: {metrics.get("BFScore", 0):.4f}±{metrics.get("BFScore_std", 0):.4f}, '
                f'ECE: {metrics.get("ECE", 0):.4f}, '
                f'IOU: {metrics["IoU"]:.4f}±{metrics["IoU_std"]:.4f}, '
                f'Sensitivity: {metrics["Sensitivity"]:.4f}±{metrics["Sensitivity_std"]:.4f}, '
                f'Specificity: {metrics["Specificity"]:.4f}±{metrics["Specificity_std"]:.4f}, '
                f'PixelAcc: {metrics["PixelAcc"]:.4f}±{metrics["PixelAcc_std"]:.4f}'
            )
            for c in class_ids:
                logger.info(
                    f'  Class {c} - '
                    f'Dice: {metrics.get(f"Dice_class_{c}", float("nan")):.4f}±{metrics.get(f"Dice_class_{c}_std", float("nan")):.4f}, '
                    f'HD95: {metrics.get(f"HD95_class_{c}", float("nan")):.2f}±{metrics.get(f"HD95_class_{c}_std", float("nan")):.2f}, '
                    f'IoU: {metrics.get(f"IoU_class_{c}", float("nan")):.4f}±{metrics.get(f"IoU_class_{c}_std", float("nan")):.4f}'
                )
        else:
            logger.info(
                f'{phase} Metrics - Dice: {metrics["Dice"]:.4f}, '
                f'HD95: {metrics["HD95"]:.2f}, BFScore: {metrics.get("BFScore", 0):.4f}, ECE: {metrics.get("ECE", 0):.4f}, '
                f'IOU: {metrics["IoU"]:.4f}, Sensitivity: {metrics["Sensitivity"]:.4f}, '
                f'Specificity: {metrics["Specificity"]:.4f}, PixelAcc: {metrics["PixelAcc"]:.4f}'
            )
            for c in class_ids:
                logger.info(
                    f'  Class {c} - '
                    f'Dice: {metrics.get(f"Dice_class_{c}", float("nan")):.4f}, '
                    f'HD95: {metrics.get(f"HD95_class_{c}", float("nan")):.2f}, '
                    f'IoU: {metrics.get(f"IoU_class_{c}", float("nan")):.4f}'
                )

    @staticmethod
    def build_per_sample_df(
        preds_list: List[torch.Tensor],
        masks_list: List[torch.Tensor],
        filenames_list: List,
        num_classes: int = 1,
    ) -> "pd.DataFrame":
        """Build a per-sample metrics DataFrame from cached prediction tensors.

        Args:
            preds_list:     List of [B, 1, H, W] binary/class prediction tensors
                            (already thresholded/argmaxed as returned by evaluate_model).
            masks_list:     List of [B, 1, H, W] GT tensors (binary) or
                            [B, C, H, W] one-hot tensors (multiclass).
            filenames_list: List of per-batch filename lists (may be None entries).
            num_classes:    Number of segmentation classes.

        Returns:
            DataFrame with columns: ``filename``, ``Dice``, ``HD95``, ``IoU``,
            ``Sensitivity``, ``Specificity``, ``PixelAcc``, ``BFScore``.
        """
        records = []
        sample_idx = 0

        for batch_preds, batch_masks, batch_fnames in zip(
            preds_list, masks_list, filenames_list
        ):
            B = batch_preds.shape[0]
            for i in range(B):
                fname = (
                    batch_fnames[i]
                    if (batch_fnames and i < len(batch_fnames))
                    else f"sample_{sample_idx:05d}"
                )
                pred_np = batch_preds[i].squeeze().numpy()
                if num_classes == 1:
                    gt_np = batch_masks[i].squeeze().numpy()
                    pb = pred_np.astype(bool)
                    gb = gt_np.astype(bool)

                    dice_v = float(dc(pb, gb))
                    if pb.any() and gb.any():
                        hd_v = float(hd95(pb, gb))
                    elif not pb.any() and not gb.any():
                        hd_v = 0.0
                    else:
                        hd_v = 224.0

                    record = {
                        "filename":    fname,
                        "Dice":        dice_v,
                        "HD95":        hd_v,
                        "IoU":         float(Evaluator_seg.compute_jaccard(pb, gb)),
                        "Sensitivity": float(recall(pb, gb)),
                        "Specificity": float(Evaluator_seg.compute_specificity(pb, gb)),
                        "PixelAcc":    float((pb == gb).sum()) / gb.size,
                        "BFScore":     float(Evaluator_seg.compute_boundary_score(pb, gb)),
                    }
                else:
                    # pred_np is class-index map [H, W]; masks are one-hot [C, H, W]
                    pred_cls = pred_np.astype(int)
                    gt_cls = np.argmax(batch_masks[i].numpy(), axis=0).astype(int)

                    fg = range(1, num_classes)
                    dice_vals, hd_vals, iou_vals = [], [], []
                    sens_vals, spec_vals, bf_vals = [], [], []
                    per_class = {}
                    for c in fg:
                        p = pred_cls == c
                        g = gt_cls == c
                        if not (p.any() or g.any()):
                            per_class[c] = {
                                "Dice": np.nan, "HD95": np.nan, "IoU": np.nan,
                                "Sensitivity": np.nan, "Specificity": np.nan, "BFScore": np.nan,
                            }
                            continue
                        tp = np.logical_and(p, g).sum()
                        dice_c = float(2 * tp / (p.sum() + g.sum() + 1e-8))
                        iou_c = float(Evaluator_seg.compute_jaccard(p, g))
                        hd_c = float(hd95(p.astype(bool), g.astype(bool))) if (p.any() and g.any()) else 224.0
                        fn = np.logical_and(~p, g).sum()
                        sens_c = float(tp / (tp + fn + 1e-8)) if (tp + fn) > 0 else np.nan
                        tn = np.logical_and(~p, ~g).sum()
                        fp = np.logical_and(p, ~g).sum()
                        spec_c = float(tn / (tn + fp + 1e-8)) if (tn + fp) > 0 else np.nan
                        bf_c = float(Evaluator_seg.compute_boundary_score(p, g))

                        dice_vals.append(dice_c)
                        iou_vals.append(iou_c)
                        hd_vals.append(hd_c)
                        sens_vals.append(sens_c)
                        spec_vals.append(spec_c)
                        bf_vals.append(bf_c)
                        per_class[c] = {
                            "Dice": dice_c, "HD95": hd_c, "IoU": iou_c,
                            "Sensitivity": sens_c, "Specificity": spec_c, "BFScore": bf_c,
                        }

                    record = {
                        "filename":    fname,
                        "Dice":        float(np.nanmean(dice_vals)) if dice_vals else 0.0,
                        "HD95":        float(np.nanmean(hd_vals))   if hd_vals  else 224.0,
                        "IoU":         float(np.nanmean(iou_vals))  if iou_vals else 0.0,
                        "Sensitivity": float(np.nanmean(sens_vals)) if sens_vals else np.nan,
                        "Specificity": float(np.nanmean(spec_vals)) if spec_vals else np.nan,
                        "PixelAcc":    float((pred_cls == gt_cls).mean()),
                        "BFScore":     float(np.nanmean(bf_vals)) if bf_vals else np.nan,
                    }
                    for c, cm in per_class.items():
                        for metric_name, val in cm.items():
                            record[f"{metric_name}_class_{c}"] = val

                records.append(record)
                sample_idx += 1

        return pd.DataFrame(records)

    @staticmethod
    def save_visual_results(
        images: torch.Tensor,
        preds: torch.Tensor,
        gts: torch.Tensor,
        save_dir: str,
        file_names: List[str],
    ) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        images = images.cpu().permute(0, 2, 3, 1).numpy()
        preds = preds.cpu().squeeze(1).numpy()
        gts = gts.cpu().numpy()

        for i in range(len(images)):
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(images[i], cmap="gray")
            axs[0].set_title("Image")
            axs[1].imshow(preds[i], cmap="gray")
            axs[1].set_title("Prediction")
            axs[2].imshow(gts[i], cmap="gray")
            axs[2].set_title("Ground Truth")
            for ax in axs:
                ax.axis("off")

            fname = Path(file_names[i]).stem
            plt.tight_layout()
            plt.savefig(save_dir / f"{fname}_result.png")
            plt.close()
