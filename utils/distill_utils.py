import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Any
from omegaconf import DictConfig, ListConfig


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
    if teacher_name == "sam_hybrid" or "E0_" in teacher_name or "EL_" in teacher_name:
        sam_type = cfg.teacher.get("sam_type", "vit_b")
        backbone = sam_type.replace("vit_", "")
        adaptation = cfg.teacher.get("adaptation_mode", "")

        name = f"sam_{backbone}"
        if adaptation:
            adapt_short = get_adaptation_short(adaptation)
            name = f"{name}_{adapt_short}"

        # Add hyperparameters
        if "alignment_num_blocks" in cfg.teacher:
            name = f"{name}_al{cfg.teacher.alignment_num_blocks}"
        if "r_d" in cfg.teacher:
            name = f"{name}_rd{cfg.teacher.r_d}"

        return name

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


def resolve_distillation_split_path(
    cfg: DictConfig,
    adaptation_ratio: float = 0.5,
    seed: int = 42,
    split_file: Optional[str] = None,
) -> Path:
    """Resolve split file path for adaptation/distillation split.

    Priority:
    1) Explicit split_file
    2) Deterministic auto path based on dataset names, ratio, and seed
       -> splits/distill_{train_names}_r{adaptation_ratio}_s{seed}.json
    """
    if split_file:
        return Path(split_file)

    train_names = cfg.data.train
    if isinstance(train_names, (list, ListConfig)):
        data_name = "_".join(train_names)
    else:
        data_name = cfg.data.name

    return Path(f"splits/distill_{data_name}_r{adaptation_ratio}_s{seed}.json")


def get_experiment_tags(cfg: DictConfig) -> list:
    """Generate standardized tags for experiments based on hyperparameters."""
    tags = []

    # Training tags
    if (bs := cfg.get("training", {}).get("batch_size")) is not None:
        tags.append(f"bs{bs}")

    lr = cfg.get("training", {}).get("lr") or cfg.get("training", {}).get("base_lr")
    if lr is not None:
        tags.append(f"lr{lr}")

    # Model structural tags
    model_cfg = cfg.get(
        "model", cfg.get("student", {})
    )  # Use student if model not found (distillation case)
    if not model_cfg and "teacher" in cfg:
        model_cfg = cfg.teacher  # Fallback for teacher-only paths if needed

    if "alignment_num_blocks" in model_cfg:
        tags.append(f"al{model_cfg.alignment_num_blocks}")
    if "r_d" in model_cfg:
        tags.append(f"rd{model_cfg.r_d}")

    # Distillation coefficients
    method_cfg = cfg.get("method", {})
    coeff_map = {
        "alpha": "a",
        "beta": "b",
        "gamma": "g",
        "gamma_attn": "ga",
        "gamma_align": "galign",
        "temperature": "T",
    }
    for key, tag in coeff_map.items():
        if (val := method_cfg.get(key)) is not None:
            tags.append(f"{tag}{val}")

    return tags


def create_log_dir(cfg: DictConfig) -> Path:
    """Create hierarchical log directory structure with hyperparameter tags.

    Structure: logs/distill/{teacher}_{student}_{method}/{dataset}/{timestamp}_{tags}/
    """
    teacher_short = get_teacher_short_name(cfg)
    student_short = get_student_short_name(cfg)
    method_name = cfg.method.name
    dataset_name = get_dataset_short_name(cfg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    tags = get_experiment_tags(cfg)
    tag_suffix = "_" + "_".join(tags) if tags else ""

    exp_group = f"{teacher_short}_{student_short}_{method_name}"
    return (
        Path(cfg.output.dir)
        / "distill"
        / exp_group
        / dataset_name
        / f"{timestamp}{tag_suffix}"
    )


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
            f"  Early Stopping: {cfg.training.get('early_stopping', {}).get('enabled', 'N/A')} (patience={cfg.training.get('early_stopping', {}).get('patience', 'N/A')})",
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

                if num_classes == 1:
                    t_pred = t_pred[0]
                    s_pred = s_pred[0]
                    gt = gt[0]
                else:
                    # For multi-class, squeeze channel dimension
                    t_pred = t_pred.squeeze()
                    s_pred = s_pred.squeeze()
                    gt = gt.squeeze()

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


def find_best_checkpoint(
    model_cfg: DictConfig, logs_root: str = "logs"
) -> Optional[Path]:
    """Find the best checkpoint for a given model configuration automatically.

    Searches for: logs/{phase}/{model_name}/{dataset}/{timestamp}_{tags}/checkpoints/best_*.pth
    """
    adaptation_mode = model_cfg.get("adaptation_mode", model_cfg.get("name", "model"))

    logs_root_path = Path(logs_root)
    if not logs_root_path.exists():
        return None

    # We search in these subdirectories
    search_dirs = [
        logs_root_path / "adaptation" / adaptation_mode,
        logs_root_path / "train" / adaptation_mode,
        logs_root_path / adaptation_mode,  # Legacy support
    ]

    candidates = []

    # Required tags for structural matching
    required_tags = []
    if "alignment_num_blocks" in model_cfg:
        required_tags.append(f"al{model_cfg.alignment_num_blocks}")
    if "r_d" in model_cfg:
        required_tags.append(f"rd{model_cfg.r_d}")

    for logs_dir in search_dirs:
        if not logs_dir.exists():
            continue

        # Recursively find checkpoints
        for dataset_path in logs_dir.iterdir():
            if not dataset_path.is_dir():
                continue

            for exp_path in dataset_path.iterdir():
                if not exp_path.is_dir():
                    continue

                # Check if tags match
                exp_name = exp_path.name
                if all(tag in exp_name for tag in required_tags):
                    # Verify experiment configuration (distillation.enabled and phase)
                    config_path = exp_path / "config.yaml"
                    if config_path.exists():
                        try:
                            exp_cfg = OmegaConf.load(config_path)
                            # If we found it in 'adaptation' folder, it's likely correct.
                            # But if we found it elsewhere, we might want to check if it's strictly a teacher-ready model.
                            # For now, let's be loose if it matches structural tags.
                        except Exception:
                            continue

                    ckpt_dir = exp_path / "checkpoints"
                    if ckpt_dir.exists():
                        best_ckpts = list(ckpt_dir.glob("best_*.pth"))
                        if best_ckpts:

                            def get_dice(p):
                                try:
                                    # Handle dice scores in filename
                                    return float(p.stem.split("dice")[-1])
                                except:
                                    return 0.0

                            best_ckpts.sort(key=get_dice, reverse=True)
                            candidates.append((exp_path.stat().st_mtime, best_ckpts[0]))

    if not candidates:
        return None

    # Pick the one from the most recent experiment that matches requirements
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]
