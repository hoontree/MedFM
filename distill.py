import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import wandb
import json

from utils.data_processing_seg import SegDatasetProcessor
from utils.evaluate import Evaluator_seg
from utils.logger import setup_logger
from utils.schedule import build_scheduler, get_lr_decay_param_groups
from trainers.model_builder import ModelBuilder
from distillers import DistillerRegistry
from utils.feature_extractor import FeatureExtractor


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_adaptation_short(adaptation_mode: str) -> str:
    """Convert adaptation mode to short abbreviation.

    Notation: E=Encoder, D=Decoder, 0=Frozen, FT=FineTune, L=LoRA
    """
    mode_map = {
        "encoder_frozen_decoder_ft": "E0-DFT",
        "encoder_frozen_decoder_lora": "E0-DL",
        "encoder_ft_decoder_lora": "EFT-DL",
        "decoder_ft_encoder_lora": "EL-DFT",
        "dual_lora": "EL-DL",
        "dual_ft": "EFT-DFT",
    }
    return mode_map.get(adaptation_mode, adaptation_mode)


def get_teacher_short_name(cfg: DictConfig) -> str:
    """Create a short teacher model identifier."""
    teacher_name = cfg.teacher.name

    # For SAM hybrid models with adaptation mode
    if teacher_name == "sam_hybrid":
        sam_type = cfg.teacher.get("sam_type", "vit_b")
        backbone = sam_type.replace("vit_", "")
        adaptation = cfg.teacher.get("adaptation_mode", "")
        if adaptation:
            adapt_short = get_adaptation_short(adaptation)
            return f"sam_{backbone}_{adapt_short}"
        return f"sam_{backbone}"

    # For simple SAM teachers (vit_b, vit_l, vit_h)
    if teacher_name.startswith("vit_"):
        backbone = teacher_name.replace("vit_", "")
        return f"sam_{backbone}"

    return teacher_name.lower()


def get_student_short_name(cfg: DictConfig) -> str:
    """Create a short student model identifier."""
    return cfg.student.name.lower()


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


def create_log_dir(cfg: DictConfig) -> Path:
    """Create hierarchical log directory structure for distillation.

    Structure: logs/distill/{teacher}_{student}_{method}/{dataset}/{timestamp}/
    Example: logs/distill/sam_b_E0-DFT_tinyusfm_logit/BUSBRA/20240116_143052/
    """
    teacher_short = get_teacher_short_name(cfg)
    student_short = get_student_short_name(cfg)
    method_name = cfg.method.name
    dataset_name = get_dataset_short_name(cfg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_config = f"{teacher_short}_{student_short}_{method_name}"
    return Path(cfg.output.dir) / "distill" / exp_config / dataset_name / timestamp


def save_experiment_summary(cfg: DictConfig, log_dir: Path):
    """Save human-readable experiment summary for distillation."""
    summary_path = log_dir / "experiment_summary.txt"

    lines = [
        "=" * 60,
        "DISTILLATION EXPERIMENT SUMMARY",
        "=" * 60,
        "",
        "[Teacher Model]",
        f"  Name: {cfg.teacher.name}",
    ]

    # Teacher SAM-specific info
    if cfg.teacher.name == "sam_hybrid":
        lines.extend(
            [
                f"  Backbone: {cfg.teacher.get('sam_type', 'N/A')}",
                f"  Adaptation: {cfg.teacher.get('adaptation_mode', 'N/A')}",
                f"  LoRA Rank: {cfg.teacher.get('rank', 'N/A')}",
            ]
        )
    elif cfg.teacher.name.startswith("vit_"):
        lines.append(f"  Backbone: {cfg.teacher.name}")

    lines.extend(
        [
            f"  Image Size: {cfg.teacher.get('img_size', 'N/A')}",
            f"  LoRA Checkpoint: {cfg.teacher.get('lora_checkpoint', 'N/A')}",
            "",
            "[Student Model]",
            f"  Name: {cfg.student.name}",
            f"  Pretrained: {cfg.student.get('pretrained', 'N/A')}",
            "",
            "[Distillation Method]",
            f"  Name: {cfg.method.name}",
            f"  Temperature: {cfg.method.get('temperature', 'N/A')}",
            f"  Alpha (task): {cfg.method.get('alpha', 'N/A')}",
            f"  Beta (distill): {cfg.method.get('beta', 'N/A')}",
            f"  Gamma (feature): {cfg.method.get('gamma', 'N/A')}",
            "",
            "[Dataset]",
            f"  Name: {get_dataset_short_name(cfg)}",
            f"  Num Classes: {cfg.data.get('num_classes', 'N/A')}",
            "",
            "[Training]",
            f"  Epochs: {cfg.training.num_epochs}",
            f"  Batch Size: {cfg.training.batch_size}",
            f"  Learning Rate: {cfg.training.lr}",
            f"  Early Stopping: {cfg.training.early_stopping.enabled} (patience={cfg.training.early_stopping.patience})",
            "",
            "[Hardware]",
            f"  GPU IDs: {cfg.hardware.gpu_ids}",
            f"  Seed: {cfg.hardware.seed}",
            "",
            "=" * 60,
        ]
    )

    with open(summary_path, "w") as f:
        f.write("\n".join(lines))


def visualize_distillation(
    teacher_model,
    student_model,
    test_loader,
    device,
    num_classes,
    teacher_img_size,
    save_dir,
    num_samples=10,
    epoch=None,
):
    """Visualize teacher vs student predictions."""
    teacher_model.eval()
    student_model.eval()

    if epoch is not None:
        save_dir = Path(save_dir) / f"epoch_{epoch+1}"
    else:
        save_dir = Path(save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    sample_count = 0
    wandb_images = []

    with torch.no_grad():
        for batch_idx, (images, masks, _) in enumerate(
            tqdm(test_loader, desc="Visualizing distillation")
        ):
            images = images.to(device)
            masks = masks.to(device)

            # Get predictions
            if hasattr(teacher_model, "image_encoder") or hasattr(
                teacher_model, "sam"
            ):  # SAM-like
                teacher_outputs = teacher_model(images, False, teacher_img_size)
            else:
                teacher_outputs = {"masks": teacher_model(images)}

            student_raw = student_model(images)
            if isinstance(student_raw, tuple):
                student_logits = student_raw[0]
            else:
                student_logits = student_raw

            teacher_logits = teacher_outputs["masks"]

            # Convert to predictions
            if num_classes == 2:
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
                    if wandb_images:
                        wandb.log({"distillation/predictions": wandb_images})
                    return

                img = images[i].cpu().numpy()
                t_pred = teacher_preds[i].cpu().numpy()
                s_pred = student_preds[i].cpu().numpy()
                gt = masks[i].cpu().numpy()

                # Basic normalization for visualization
                if img.shape[0] == 3:
                    img = img.transpose(1, 2, 0)
                    img = (img - img.min()) / (img.max() - img.min())
                else:
                    img = img[0]
                    img = (img - img.min()) / (img.max() - img.min())

                if num_classes == 2:
                    t_pred = t_pred[0]
                    s_pred = s_pred[0]

                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                axes[0].imshow(img, cmap="gray" if img.ndim == 2 else None)
                axes[0].set_title("Image")
                axes[1].imshow(gt, cmap="jet", alpha=0.5)
                axes[1].set_title("GT")
                axes[2].imshow(t_pred, cmap="jet", alpha=0.5)
                axes[2].set_title("Teacher")
                axes[3].imshow(s_pred, cmap="jet", alpha=0.5)
                axes[3].set_title("Student")
                for ax in axes:
                    ax.axis("off")

                save_path = save_dir / f"sample_{sample_count:03d}.png"
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                wandb_images.append(
                    wandb.Image(str(save_path), caption=f"Sample {sample_count}")
                )
                plt.close()
                sample_count += 1


def run_evaluation(student, test_loader, evaluator, device, cfg, logger, phase="Test"):
    """Helper to run evaluation on test loader(s)."""
    all_metrics = {}
    if isinstance(test_loader, dict):
        for ds_name, loader in test_loader.items():
            metrics = evaluator.evaluate_model(
                student, loader, device, cfg.data.num_classes
            )
            logger.info(f"--- {phase} ({ds_name}) ---")
            evaluator.print_metrics(metrics, phase=f"{phase}_{ds_name}")
            for k, v in metrics.items():
                if isinstance(v, (float, int)):
                    all_metrics[f"{phase.lower()}/{ds_name}/{k.lower()}"] = v
    else:
        metrics = evaluator.evaluate_model(
            student, test_loader, device, cfg.data.num_classes
        )
        logger.info(f"--- {phase} ---")
        evaluator.print_metrics(metrics, phase=phase)
        for k, v in metrics.items():
            if isinstance(v, (float, int)):
                all_metrics[f"{phase.lower()}/{k.lower()}"] = v
    return all_metrics


@hydra.main(version_base=None, config_path="config", config_name="distill")
def main(cfg: DictConfig):
    # Set environment and seed
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg.gpu_ids))
    set_seed(cfg.hardware.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup directories with new structure: logs/distill/{teacher}_{student}_{method}/{dataset}/{timestamp}/
    log_dir = create_log_dir(cfg)
    log_dir.mkdir(parents=True, exist_ok=True)

    models_dir = log_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = log_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment summary for quick reference
    save_experiment_summary(cfg, log_dir)

    # Setup logger
    teacher_short = get_teacher_short_name(cfg)
    student_short = get_student_short_name(cfg)
    dataset_name = get_dataset_short_name(cfg)
    logger = setup_logger(str(log_dir / "distill.log"))
    logger.info(f"Starting Distillation: {cfg.method.name}")
    logger.info(f"Teacher: {teacher_short} -> Student: {student_short}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Log directory: {log_dir}")

    # Initialize wandb with descriptive experiment name
    exp_name = f"{teacher_short}_{student_short}_{cfg.method.name}_{dataset_name}"
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=exp_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # Load data
    distill_cfg = cfg.get("distillation", {})
    distill_enabled = distill_cfg.get("enabled", False)

    if distill_enabled:
        # Distillation mode: use split datasets (distillation subset)
        adaptation_ratio = distill_cfg.get("adaptation_ratio", 0.5)
        split_seed = distill_cfg.get("split_seed", 42)
        split_file = distill_cfg.get("split_file", None)

        logger.info(f"=== Distillation Split Mode ===")
        logger.info(f"Adaptation ratio: {adaptation_ratio}, Seed: {split_seed}")

        loaders = SegDatasetProcessor.build_distillation_data_loaders(
            cfg,
            adaptation_ratio=adaptation_ratio,
            seed=split_seed,
            split_file=split_file,
        )

        # Distill.py always uses the distillation subset
        train_loader = loaders["distillation_train"]
        val_loader = loaders["distillation_val"]
        test_loader = loaders["test"]

        logger.info(f"Using distillation split:")
        logger.info(
            f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}"
        )
    else:
        # Normal mode: use full dataset
        train_loader, val_loader, test_loader = SegDatasetProcessor.build_data_loaders(
            cfg
        )

    # Create models
    teacher = ModelBuilder.create_model(cfg.teacher, cfg.data.num_classes, device)
    student = ModelBuilder.create_model(cfg.student, cfg.data.num_classes, device)

    teacher.eval()  # Teacher always in eval
    for param in teacher.parameters():
        param.requires_grad = False

    # Create Distiller
    distiller = DistillerRegistry.create(cfg).to(device)

    # Setup optimizer
    if cfg.optimizer.name == "AdamW":
        param_groups = [{"params": student.parameters(), "lr": cfg.training.lr}]
        if list(distiller.parameters()):
            param_groups.append(
                {"params": distiller.parameters(), "lr": cfg.training.lr}
            )
        optimizer = optim.AdamW(param_groups, weight_decay=cfg.optimizer.weight_decay)
    else:
        params = list(student.parameters()) + list(distiller.parameters())
        optimizer = optim.Adam(params, lr=cfg.training.lr)

    scheduler = build_scheduler(optimizer, cfg)
    evaluator = Evaluator_seg()

    # Setup feature extraction if needed
    teacher_layers = []
    student_layers = []
    if cfg.method.name == "adaptive_layer":
        for s_layer, t_layer in cfg.method.layer_mapping.items():
            student_layers.append(s_layer)
            teacher_layers.append(t_layer)

    teacher_extractor = (
        FeatureExtractor(teacher, teacher_layers) if teacher_layers else None
    )
    student_extractor = (
        FeatureExtractor(student, student_layers) if student_layers else None
    )

    # Early stopping setup
    early_stopping_cfg = cfg.training.get("early_stopping")
    es_enabled = early_stopping_cfg and early_stopping_cfg.enabled
    if es_enabled:
        patience = early_stopping_cfg.patience
        min_delta = early_stopping_cfg.min_delta
        es_counter = 0

    # Training loop
    best_dice = 0.0
    best_model_path = None
    global_step = 0
    for epoch in range(cfg.training.num_epochs):
        student.train()
        distiller.train()

        running_losses = {}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        limit_batches = cfg.training.get("limit_train_batches")

        for i, (images, masks, low_res_masks) in enumerate(pbar):
            if limit_batches is not None and i >= limit_batches:
                break
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            if teacher_extractor:
                teacher_extractor.clear()
            if student_extractor:
                student_extractor.clear()

            with torch.no_grad():
                if hasattr(teacher, "image_encoder") or hasattr(
                    teacher, "sam"
                ):  # SAM or SAM-based wrapper
                    teacher_outputs = teacher(images, False, cfg.teacher.img_size)
                else:
                    teacher_outputs = {"masks": teacher(images)}

                if teacher_extractor:
                    teacher_outputs["layer_features"] = teacher_extractor.get_features()

            s_model = student.module if hasattr(student, "module") else student
            student_raw = (
                s_model(images, return_features=True)
                if hasattr(s_model, "backbone")
                else student(images)
            )
            if isinstance(student_raw, tuple):
                student_outputs = {"masks": student_raw[0], "features": student_raw[1]}
            else:
                student_outputs = {"masks": student_raw}

            if student_extractor:
                student_outputs["layer_features"] = student_extractor.get_features()

            # Hybrid Distillation: Collect attention maps and specific features
            if cfg.method.name == "hybrid":
                if cfg.method.get("gamma_attn", 0) > 0:
                    # Collect student attention maps
                    student_attn_maps = []
                    s_model = student.module if hasattr(student, "module") else student
                    # Student is SegmentationModel, its backbone is MAEBackbone
                    if hasattr(s_model, "backbone") and hasattr(
                        s_model.backbone, "blocks"
                    ):
                        for blk in s_model.backbone.blocks:
                            if hasattr(blk, "attn") and hasattr(blk.attn, "last_attn"):
                                student_attn_maps.append(blk.attn.last_attn)
                    elif hasattr(s_model, "blocks"):
                        for blk in s_model.blocks:
                            if hasattr(blk, "attn") and hasattr(blk.attn, "last_attn"):
                                student_attn_maps.append(blk.attn.last_attn)
                    student_outputs["attn_maps"] = student_attn_maps

                    # Collect teacher attention maps
                    teacher_attn_maps = []
                    t_model = teacher.module if hasattr(teacher, "module") else teacher
                    # Teacher can be LoRA_Sam (with .sam) or just Sam
                    image_encoder = None
                    if hasattr(t_model, "sam") and hasattr(
                        t_model.sam, "image_encoder"
                    ):
                        image_encoder = t_model.sam.image_encoder
                    elif hasattr(t_model, "image_encoder"):
                        image_encoder = t_model.image_encoder

                    if image_encoder and hasattr(image_encoder, "blocks"):
                        for blk in image_encoder.blocks:
                            if hasattr(blk, "attn") and hasattr(blk.attn, "last_attn"):
                                teacher_attn_maps.append(blk.attn.last_attn)
                    teacher_outputs["attn_maps"] = teacher_attn_maps

            # Distillation Loss
            loss_dict = distiller(student_outputs, teacher_outputs, masks)
            loss = loss_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record losses
            for k, v in loss_dict.items():
                running_losses[k] = running_losses.get(k, 0.0) + v.item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Step-level logging
            if global_step % 10 == 0:
                step_log = {"global_step": global_step}
                for k, v in loss_dict.items():
                    step_log[f"train_step/{k}"] = v.item()
                step_log["train_step/lr"] = optimizer.param_groups[0]["lr"]
                wandb.log(step_log)
            global_step += 1

        # Validation
        student.eval()
        val_metrics = evaluator.evaluate_model(
            student, val_loader, device, cfg.data.num_classes
        )
        logger.info(f"Epoch {epoch+1} Val Dice: {val_metrics['Dice']:.4f}")

        # Log to wandb
        log_data = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
        }
        # Add training losses (epoch average)
        for k, v in running_losses.items():
            log_data[f"train/{k}"] = v / len(train_loader)

        # Add all val metrics
        for k, v in val_metrics.items():
            if isinstance(v, (float, int)):
                log_data[f"val/{k.lower()}"] = v

        # Test Every Epoch (All metrics)
        test_metrics = run_evaluation(
            student, test_loader, evaluator, device, cfg, logger, phase="Test"
        )
        log_data.update(test_metrics)

        # Visualize Every Epoch
        vis_loader = (
            list(test_loader.values())[0]
            if isinstance(test_loader, dict)
            else test_loader
        )
        visualize_distillation(
            teacher,
            student,
            vis_loader,
            device,
            cfg.data.num_classes,
            cfg.teacher.img_size,
            vis_dir,
            num_samples=cfg.visualization.num_samples,
            epoch=epoch,
        )

        wandb.log(log_data)

        # Save best model and check early stopping
        if val_metrics["Dice"] > best_dice:
            if es_enabled and (val_metrics["Dice"] - best_dice) > min_delta:
                es_counter = 0

            # Delete previous best model to save space
            if best_model_path and best_model_path.exists():
                try:
                    best_model_path.unlink()
                except Exception as e:
                    logger.warning(f"Could not delete old best model: {e}")

            best_dice = val_metrics["Dice"]
            best_model_path = (
                models_dir / f"best_epoch_{epoch+1}_dice{best_dice:.4f}.pth"
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": student.state_dict(),
                    "distiller_state_dict": distiller.state_dict(),
                    "dice": best_dice,
                },
                best_model_path,
            )
            logger.info(f"Saved best model to {best_model_path}")
        elif es_enabled:
            es_counter += 1
            if es_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        scheduler.step()

    # Final Test
    if best_model_path and best_model_path.exists():
        checkpoint = torch.load(best_model_path, weights_only=False)
        student.load_state_dict(checkpoint["model_state_dict"])

        test_metrics = run_evaluation(
            student, test_loader, evaluator, device, cfg, logger, phase="Final_Test"
        )
        # Log all metrics to wandb
        wandb.log(test_metrics)

        # Final visualization
        vis_loader = (
            list(test_loader.values())[0]
            if isinstance(test_loader, dict)
            else test_loader
        )
        visualize_distillation(
            teacher,
            student,
            vis_loader,
            device,
            cfg.data.num_classes,
            cfg.teacher.img_size,
            vis_dir,
            num_samples=cfg.visualization.num_samples,
            epoch=None,  # Final one doesn't need epoch subfolder or we can use "final"
        )

    wandb.finish()


if __name__ == "__main__":
    main()
