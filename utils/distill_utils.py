import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, ListConfig, OmegaConf


# Project-relative directory holding per-model yaml definitions.
_MODEL_CFG_DIR = Path(__file__).resolve().parent.parent / "config" / "model"


def load_model_cfg(name: str, num_classes: int) -> DictConfig:
    """Load ``config/model/{name}.yaml`` and extract the ``model:`` section.

    The model yaml is expected to expose ``binary_checkpoint`` and
    ``multiclass_checkpoint`` fields. The correct one is picked based on
    ``num_classes`` (``1`` → binary, otherwise → multiclass) and assigned to
    ``checkpoint`` so downstream model classes can consume it transparently.

    Variable interpolations like ``${data.num_classes}`` are intentionally left
    unresolved here — the caller is responsible for merging this into a parent
    config so interpolations resolve correctly.
    """
    cfg_path = _MODEL_CFG_DIR / f"{name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Model config not found: {cfg_path}")

    full_cfg = OmegaConf.load(cfg_path)
    model_cfg = full_cfg.get("model")
    if model_cfg is None:
        raise ValueError(
            f"{cfg_path} does not contain a 'model:' section."
        )

    OmegaConf.set_struct(model_cfg, False)

    # Select binary / multiclass checkpoint based on the task.
    binary_ckpt = model_cfg.pop("binary_checkpoint", None)
    multi_ckpt = model_cfg.pop("multiclass_checkpoint", None)
    selected = binary_ckpt if num_classes == 1 else multi_ckpt
    if selected is not None:
        model_cfg.checkpoint = selected

    return model_cfg


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


def _model_cfg_for_tags(cfg: DictConfig) -> Optional[DictConfig]:
    """Return the model-shaped DictConfig used for tag extraction.

    Prefers the resolved ``student_cfg`` (distillation), then ``model``
    (training), then ``teacher_cfg`` as a last resort.
    """
    for key in ("student_cfg", "model", "teacher_cfg"):
        sub = cfg.get(key)
        if isinstance(sub, DictConfig):
            return sub
    return None


def get_experiment_tags(cfg: DictConfig) -> list:
    """Generate standardized tags for experiments based on hyperparameters."""
    tags = []

    if (bs := cfg.get("training", {}).get("batch_size")) is not None:
        tags.append(f"bs{bs}")

    lr = cfg.get("training", {}).get("lr") or cfg.get("training", {}).get("base_lr")
    if lr is not None:
        tags.append(f"lr{lr}")

    model_cfg = _model_cfg_for_tags(cfg)
    if model_cfg is not None:
        if model_cfg.get("encoder_mode", "frozen") != "frozen":
            tags.append(f"encoder_{model_cfg.encoder_mode}")
            tags.append(f"re{model_cfg.r_e}")
        if model_cfg.get("decoder_mode", "frozen") == "lora":
            tags.append(f"rd{model_cfg.r_d}")

    method_cfg = cfg.get("method", {})
    # Map cfg.method key → short tag prefix used in log directory names.
    # New ``w_*`` keys are listed first; legacy greek names follow so old
    # configs still produce tags.
    coeff_map = {
        "w_task":           "task",
        "w_logit_kd":       "lkd",
        "w_logit_cwd":      "lcwd",
        "w_feature_cwd":    "fcwd",
        "w_reliability_kd": "rkd",
        "w_uncertainty_kd": "ukd",
        "temperature":      "T",
        # legacy aliases (kept until configs migrate)
        "alpha":     "a",
        "beta":      "b",
        "delta":     "d",
        "zeta":      "z",
        "eta":       "e",
        "kd_lambda": "ukd",
    }
    for key, tag in coeff_map.items():
        if (val := method_cfg.get(key)) is not None:
            tags.append(f"{tag}{val}")

    return tags


def create_log_dir(cfg: DictConfig) -> Path:
    """Create hierarchical log directory structure with hyperparameter tags.

    Structure: logs/distill/{teacher}_to_{student}/{timestamp}_{tags}/
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tags = get_experiment_tags(cfg)
    tag_suffix = "_" + "_".join(tags) if tags else ""

    # Human label (sweeps set cfg.run_name) goes into the dir so runs are
    # self-identifying without parsing logs.
    run_name = cfg.get("run_name")
    label = f"_{run_name}" if run_name else ""

    teacher_name = cfg.get("teacher")
    student_name = cfg.get("student")
    return (
        Path(cfg.output.dir)
        / "distill"
        / f"{teacher_name}_to_{student_name}"
        / f"{timestamp}{label}{tag_suffix}"
    )


def save_experiment_summary(cfg: DictConfig, log_dir: Path):
    """Save human-readable experiment summary for distillation."""
    summary_path = log_dir / "experiment_summary.txt"

    teacher_name = cfg.get("teacher")
    student_name = cfg.get("student")
    teacher_cfg = cfg.get("teacher_cfg", OmegaConf.create({}))
    student_cfg = cfg.get("student_cfg", OmegaConf.create({}))

    lines = [
        "=" * 60,
        "DISTILLATION EXPERIMENT SUMMARY",
        "=" * 60,
        "",
        "[Teacher Model]",
        f"  Name: {teacher_name}",
        f"  Checkpoint: {teacher_cfg.get('checkpoint', 'N/A')}",
    ]

    if teacher_name in ("sam", "sam_hybrid"):
        lines.extend(
            [
                f"  Backbone: {teacher_cfg.get('sam_type', 'N/A')}",
                f"  Encoder Mode: {teacher_cfg.get('encoder_mode', 'N/A')}",
                f"  Decoder Mode: {teacher_cfg.get('decoder_mode', 'N/A')}",
                f"  LoRA Rank (decoder): {teacher_cfg.get('r_d', 'N/A')}",
            ]
        )

    lines.extend(
        [
            f"  Image Size: {teacher_cfg.get('img_size', 'N/A')}",
            "",
            "[Student Model]",
            f"  Name: {student_name}",
            f"  Checkpoint: {student_cfg.get('checkpoint', 'N/A')}",
            "",
            "[Distillation Method]",
            f"  Name: {cfg.method.name}",
            f"  Temperature: {cfg.method.get('temperature', 'N/A')}",
            f"  w_task (task):                   {cfg.method.get('w_task', cfg.method.get('alpha', 'N/A'))}",
            f"  w_logit_kd (logit-KD):           {cfg.method.get('w_logit_kd', cfg.method.get('beta', 'N/A'))}",
            f"  w_logit_cwd (logit-CWD):         {cfg.method.get('w_logit_cwd', cfg.method.get('delta', 'N/A'))}",
            f"  w_feature_cwd (feature-CWD):     {cfg.method.get('w_feature_cwd', cfg.method.get('zeta', 'N/A'))}",
            f"  w_reliability_kd (reliability):  {cfg.method.get('w_reliability_kd', cfg.method.get('eta', 'N/A'))}",
            f"  w_uncertainty_kd (uncertainty):  {cfg.method.get('w_uncertainty_kd', cfg.method.get('kd_lambda', 'N/A'))}",
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
