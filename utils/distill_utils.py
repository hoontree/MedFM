import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
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
