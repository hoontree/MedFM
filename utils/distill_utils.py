import random
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from omegaconf import DictConfig, ListConfig, OmegaConf


def get_teacher_short_name(cfg: DictConfig) -> str:
    """Create a short teacher model identifier from encoder_mode/decoder_mode/use_alignment."""
    teacher_name = cfg.teacher.name

    # SAM-based models: build name from mode fields
    if teacher_name in ("sam", "sam_hybrid"):
        sam_type = cfg.teacher.get("sam_type", "vit_b")
        backbone = sam_type.replace("vit_", "")
        encoder_mode = cfg.teacher.get("encoder_mode", "frozen")
        decoder_mode = cfg.teacher.get("decoder_mode", "lora")
        use_alignment = cfg.teacher.get("use_alignment", False)

        name = f"sam_{backbone}"
        if encoder_mode != "frozen":
            name = f"{name}_e{encoder_mode}"
        if use_alignment:
            name = f"{name}_align"
        if decoder_mode != "frozen":
            name = f"{name}_d{decoder_mode}"

        # Add hyperparameters
        if cfg.teacher.get("alignment_num_blocks") is not None:
            name = f"{name}_al{cfg.teacher.alignment_num_blocks}"
        if cfg.teacher.get("r_d") is not None:
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
    if cfg.teacher.name in ("sam", "sam_hybrid"):
        lines.extend(
            [
                f"  Backbone: {cfg.teacher.get('sam_type', 'N/A')}",
                f"  Encoder Mode: {cfg.teacher.get('encoder_mode', 'N/A')}",
                f"  Decoder Mode: {cfg.teacher.get('decoder_mode', 'N/A')}",
                f"  Use Alignment: {cfg.teacher.get('use_alignment', 'N/A')}",
                f"  LoRA Rank (decoder): {cfg.teacher.get('r_d', 'N/A')}",
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
    """Visualize teacher vs student predictions.

    Delegates to the unified implementation in utils.visualize.
    """
    from utils.visualize import visualize_distillation as _visualize_distillation

    _visualize_distillation(
        teacher_model=teacher_model,
        student_model=student_model,
        test_loader=test_loader,
        device=device,
        num_classes=num_classes,
        teacher_img_size=teacher_img_size,
        save_dir=save_dir,
        num_samples=num_samples,
        epoch=epoch,
    )
