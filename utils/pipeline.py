"""Teacher→Distillation pipeline utilities.

Provides config building and checkpoint validation for the
train.py pipeline mode (``pipeline.enabled=true``).
"""

import logging
from pathlib import Path
from typing import Optional

from hydra import compose
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def build_distill_cfg(
    train_cfg: DictConfig,
    teacher_ckpt: Path,
    *,
    teacher_run_id: Optional[str] = None,
    teacher_log_dir: Optional[str] = None,
) -> DictConfig:
    """Build distillation config by composing base distill config + train context.

    Merges the full teacher model config from the training run into the
    distillation teacher section so that every architecture key is carried
    over automatically.  Also forwards pipeline context (teacher_run_id,
    teacher_log_dir) for integrated logging.
    """
    teacher_name = train_cfg.model.get("name")
    if not teacher_name:
        raise ValueError("cfg.model.name is required for pipeline distillation.")

    # Verify that a teacher preset yaml exists
    preset = Path("config/teacher") / f"{teacher_name}.yaml"
    if not preset.exists():
        raise FileNotFoundError(f"Teacher preset not found: {preset}")

    # Compose distillation config with teacher override
    distill_cfg = compose(config_name="distill", overrides=[f"teacher={teacher_name}"])
    OmegaConf.set_struct(distill_cfg, False)
    OmegaConf.set_struct(distill_cfg.teacher, False)

    # Merge full teacher model arch (carries all keys automatically)
    distill_cfg.teacher = OmegaConf.merge(distill_cfg.teacher, train_cfg.model)
    distill_cfg.teacher.checkpoint = str(teacher_ckpt.resolve())

    # Carry over shared sections from training config
    distill_cfg.data = train_cfg.data
    distill_cfg.hardware = OmegaConf.merge(
        distill_cfg.get("hardware", OmegaConf.create({})),
        train_cfg.get("hardware", OmegaConf.create({})),
    )

    # Pipeline context for integrated logging in DistillTrainer
    pipeline_dict = (
        OmegaConf.to_container(train_cfg.get("pipeline", {}), resolve=False) or {}
    )
    if teacher_run_id is not None:
        pipeline_dict["teacher_run_id"] = teacher_run_id
    if teacher_log_dir is not None:
        pipeline_dict["teacher_log_dir"] = teacher_log_dir
    distill_cfg.pipeline = OmegaConf.create(pipeline_dict)

    return distill_cfg
