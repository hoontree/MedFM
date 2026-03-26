import random
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, List
import logging
from omegaconf import DictConfig, ListConfig, OmegaConf


def get_teacher_short_name(cfg: DictConfig) -> str:
    """Create a short teacher model identifier from encoder_mode/decoder_mode/use_alignment."""
    teacher_name = cfg.teacher.name

    # SAM-based models: build name from mode fields
    if teacher_name in ("sam", "sam_hybrid"):
        sam_type = cfg.teacher.get("sam_type", "vit_b")
        encoder_mode = cfg.teacher.get("encoder_mode", "frozen")
        decoder_mode = cfg.teacher.get("decoder_mode", "lora")
        use_alignment = cfg.teacher.get("use_alignment", False)

        name = f"sam_{sam_type}"
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

    return teacher_name.lower()


def _sync_teacher_cfg_from_exp_dir(cfg: DictConfig, exp_dir: Path, logger: logging.Logger) -> None:
    """Merge the original experiment's model config into cfg.teacher in-place.

    Fields that belong to the current distill config (checkpoint, exp_id) are
    preserved and never overwritten by the old experiment's values.
    """
    conf_file = exp_dir / "config.yaml"
    if not conf_file.exists():
        return
    try:
        old_cfg = OmegaConf.load(conf_file)
        # train.yaml stores the model under 'model:'; distill.yaml uses 'teacher:'
        src_model_cfg = old_cfg.get("model", None)
        if src_model_cfg is None:
            return

        keep = {k: v for k in ("checkpoint", "exp_id") if (v := cfg.teacher.get(k)) is not None}
        merged = OmegaConf.merge(cfg.teacher, src_model_cfg)

        # Restore protected fields
        OmegaConf.set_struct(merged, False)
        for k, v in keep.items():
            OmegaConf.update(merged, k, v, merge=False)

        # Write back to cfg.teacher in-place
        OmegaConf.set_struct(cfg.teacher, False)
        for key in merged:
            OmegaConf.update(cfg.teacher, key, merged[key], merge=False)

        logger.info(f"Synchronized teacher config from {conf_file} (protected: {list(keep)})")
    except Exception as e:
        logger.warning(f"Failed to sync teacher config from {conf_file}: {e}")


def _find_exp_dir_by_id(exp_id, cfg: DictConfig, logger: logging.Logger) -> Optional[Path]:
    """Search for an experiment directory matching exp_id under the logs tree."""
    # YAML may strip underscores from a timestamp-like value (e.g. 20260313_012805 → 20260313012805).
    exp_id_str = str(exp_id)
    if len(exp_id_str) == 14 and exp_id_str.isdigit():
        exp_id_fmt = exp_id_str[:8] + "_" + exp_id_str[8:]
    else:
        exp_id_fmt = exp_id_str
    variants = list(dict.fromkeys([exp_id_str, exp_id_fmt]))  # dedup, preserve order

    def _matches(d: Path) -> bool:
        return any(v in d.name for v in variants)

    logs_root = Path(cfg.get("output", {}).get("dir", "logs"))
    encoder_mode = cfg.teacher.get("encoder_mode", "frozen")
    decoder_mode = cfg.teacher.get("decoder_mode", "lora")
    peft_mode = f"e_{encoder_mode}_d_{decoder_mode}"
    logs_sam = logs_root / "sam"

    # Search order: specific peft dir → all of logs/sam/ → entire logs tree
    search_roots = [
        (logs_sam / peft_mode, False),
        (logs_sam, True),
        (logs_root, True),
    ]
    for root, recursive in search_roots:
        if not root.exists():
            continue
        candidates = (
            sorted(root.rglob("*")) if recursive else list(root.iterdir())
        )
        matching = [
            p for p in candidates
            if p.is_dir() and _matches(p) and (not recursive or (p / "config.yaml").exists())
        ]
        if matching:
            exp_dir = sorted(matching)[-1]
            logger.info(f"Resolved exp_id '{exp_id_str}' → {exp_dir}")
            return exp_dir
        if recursive:
            logger.info(f"exp_id '{exp_id_str}' not found under {root}, expanding search...")

    logger.warning(f"No experiment directory found for exp_id: {exp_id_str} (tried: {variants})")
    return None


def resolve_teacher_checkpoint(cfg: DictConfig) -> Optional[str]:
    """Resolve teacher checkpoint and synchronize teacher config from the
    original experiment's config.yaml.

    Priority:
    1) Explicit cfg.teacher.checkpoint → infer exp_dir from checkpoint path,
       sync config, then return the checkpoint.
    2) cfg.teacher.exp_id → locate exp_dir, sync config, find best checkpoint.
    """
    logger = logging.getLogger("medfm.distill")

    # 1. Explicit checkpoint path
    checkpoint = cfg.teacher.get("checkpoint")
    if checkpoint is not None:
        ckpt_path = Path(checkpoint)
        if ckpt_path.exists():
            # Infer exp_dir: checkpoint lives inside <exp_dir>/checkpoints/
            exp_dir = ckpt_path.parent.parent
            if (exp_dir / "config.yaml").exists():
                _sync_teacher_cfg_from_exp_dir(cfg, exp_dir, logger)
            return str(checkpoint)
        logger.warning(f"Explicit teacher checkpoint not found: {checkpoint}")

    # 2. Resolve via exp_id
    exp_id = cfg.teacher.get("exp_id")
    if not exp_id:
        return None

    exp_dir = _find_exp_dir_by_id(exp_id, cfg, logger)
    if exp_dir is None:
        return None

    _sync_teacher_cfg_from_exp_dir(cfg, exp_dir, logger)

    # Locate best checkpoint inside the experiment directory
    ckpt_dir = exp_dir / "checkpoints"
    if not ckpt_dir.exists():
        logger.warning(f"Checkpoint directory not found: {ckpt_dir}")
        return None

    checkpoints = list(ckpt_dir.glob("best_epoch_*_dice*.pth")) or list(ckpt_dir.glob("epoch_*.pth"))
    if not checkpoints:
        logger.warning(f"No checkpoints found in {ckpt_dir}")
        return None

    best_ckpt = sorted(checkpoints)[-1]
    logger.info(f"Automatically resolved teacher checkpoint: {best_ckpt}")
    return str(best_ckpt)


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

    if model_cfg.get("use_alignment", False):
        tags.append(f"al{model_cfg.alignment_num_blocks}")
    if model_cfg.get("encoder_mode", "frozen") != "frozen":
        tags.append(f"encoder_{model_cfg.encoder_mode}")
        tags.append(f"re{model_cfg.r_e}")
    if model_cfg.get("decoder_mode", "frozen") == "lora":
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

    Structure: logs/distill/{peft_mode}/{timestamp}_{tags}/
    where peft_mode mirrors the teacher's training directory (e_{enc}_d_{dec}).
    """
    encoder_mode = cfg.teacher.get("encoder_mode", "frozen")
    decoder_mode = cfg.teacher.get("decoder_mode", "frozen")
    peft_mode = f"e_{encoder_mode}_d_{decoder_mode}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tags = get_experiment_tags(cfg)
    tag_suffix = "_" + "_".join(tags) if tags else ""

    return Path(cfg.output.dir) / "distill" / peft_mode / f"{timestamp}{tag_suffix}"


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
