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
    """Build distillation config from a freshly-trained teacher.

    The distillation config selects the teacher by its model name
    (``cfg.model.name`` from the training run). The teacher's resolved
    checkpoint path is forwarded via ``pipeline.teacher_ckpt_override`` so
    DistillTrainer applies it after loading ``config/model/{name}.yaml``.
    """
    teacher_name = train_cfg.model.get("name")
    if not teacher_name:
        raise ValueError("cfg.model.name is required for pipeline distillation.")

    preset = Path("config/model") / f"{teacher_name}.yaml"
    if not preset.exists():
        raise FileNotFoundError(f"Model preset not found: {preset}")

    distill_cfg = compose(config_name="distill", overrides=[f"teacher={teacher_name}"])
    OmegaConf.set_struct(distill_cfg, False)

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
    pipeline_dict["teacher_ckpt_override"] = str(teacher_ckpt.resolve())
    if teacher_run_id is not None:
        pipeline_dict["teacher_run_id"] = teacher_run_id
    if teacher_log_dir is not None:
        pipeline_dict["teacher_log_dir"] = teacher_log_dir
    distill_cfg.pipeline = OmegaConf.create(pipeline_dict)

    return distill_cfg
