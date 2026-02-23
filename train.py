import gc
import os
from pathlib import Path
from typing import Optional

import hydra
import torch
from hydra import compose
from omegaconf import DictConfig, OmegaConf

from trainers.model_builder import ModelBuilder
from utils.distill_utils import resolve_distillation_split_path


TEACHER_SYNC_KEYS = (
    "sam_type",
    "adaptation_mode",
    "r_d",
    "alignment_num_blocks",
    "alignment_hidden_channels",
    "r_e",
    "conv_lora_expert_num",
)


def _set_cuda_visible_devices(cfg: DictConfig) -> None:
    gpu_ids = cfg.get("hardware", {}).get("gpu_ids", cfg.get("gpu_ids", [0]))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))


def _is_sweep_run() -> bool:
    return os.environ.get("WANDB_SWEEP_ID") is not None


def _maybe_disable_teacher_wandb_for_sweep(cfg: DictConfig) -> None:
    pipeline_cfg = cfg.get("pipeline", {})
    if not pipeline_cfg.get("enabled", False):
        return

    disable_teacher_wandb = pipeline_cfg.get("sweep", {}).get(
        "disable_teacher_wandb", True
    )
    if not (_is_sweep_run() and disable_teacher_wandb):
        return

    OmegaConf.set_struct(cfg, False)
    if "wandb" not in cfg or cfg.wandb is None:
        cfg.wandb = OmegaConf.create({})
    cfg.wandb.disabled = True
    print("[Pipeline] Sweep run detected. Teacher W&B logging is disabled.")


def _cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _release_trainer_resources(trainer) -> None:
    for attr_name in (
        "model",
        "optimizer",
        "scheduler",
        "train_loader",
        "val_loader",
        "test_loader",
    ):
        if not hasattr(trainer, attr_name):
            continue
        try:
            value = getattr(trainer, attr_name)
            if attr_name == "model" and value is not None and hasattr(value, "to"):
                value.to("cpu")
            setattr(trainer, attr_name, None)
        except Exception:
            continue


def _sync_train_context_to_distill_cfg(
    train_cfg: DictConfig,
    distill_cfg: DictConfig,
    teacher_checkpoint: Path,
) -> DictConfig:
    OmegaConf.set_struct(distill_cfg, False)
    OmegaConf.set_struct(distill_cfg.teacher, False)
    OmegaConf.set_struct(distill_cfg.distillation, False)

    model_name = train_cfg.model.get("name")
    distill_cfg.teacher.name = model_name
    distill_cfg.teacher.checkpoint = str(teacher_checkpoint.resolve())

    for key in TEACHER_SYNC_KEYS:
        if key in train_cfg.model:
            distill_cfg.teacher[key] = train_cfg.model.get(key)

    distill_cfg.data = OmegaConf.create(
        OmegaConf.to_container(train_cfg.data, resolve=False)
    )
    hardware_cfg = OmegaConf.to_container(train_cfg.get("hardware", {}), resolve=False)
    if hardware_cfg is None:
        hardware_cfg = {}
    if "gpu_ids" not in hardware_cfg:
        hardware_cfg["gpu_ids"] = train_cfg.get("gpu_ids", [0])
    if "seed" not in hardware_cfg:
        hardware_cfg["seed"] = 42
    distill_cfg.hardware = OmegaConf.create(hardware_cfg)

    train_distill_cfg = train_cfg.get("distillation", {})
    adaptation_ratio = train_distill_cfg.get(
        "adaptation_ratio",
        distill_cfg.distillation.get("adaptation_ratio", 0.3),
    )
    split_seed = train_distill_cfg.get(
        "split_seed",
        distill_cfg.distillation.get("split_seed", 42),
    )
    split_file = train_distill_cfg.get("split_file", None)
    split_path = resolve_distillation_split_path(
        train_cfg,
        adaptation_ratio=adaptation_ratio,
        seed=split_seed,
        split_file=split_file,
    )

    distill_cfg.distillation.enabled = True
    distill_cfg.distillation.phase = "distillation"
    distill_cfg.distillation.adaptation_ratio = adaptation_ratio
    distill_cfg.distillation.split_seed = split_seed
    distill_cfg.distillation.split_file = str(split_path)

    distill_cfg.pipeline = OmegaConf.create(
        OmegaConf.to_container(train_cfg.get("pipeline", {}), resolve=False)
    )

    return distill_cfg


def _build_distill_cfg(train_cfg: DictConfig, teacher_checkpoint: Path) -> DictConfig:
    pipeline_distill_cfg = train_cfg.get("pipeline", {}).get("distill", {})
    config_name = pipeline_distill_cfg.get("config_name", "distill")
    teacher_name = train_cfg.model.get("name")
    if not teacher_name:
        raise ValueError("Pipeline requires cfg.model.name to select teacher preset.")

    require_teacher_preset = pipeline_distill_cfg.get("require_teacher_preset", True)
    if require_teacher_preset:
        teacher_cfg_path = Path("config") / "teacher" / f"{teacher_name}.yaml"
        if not teacher_cfg_path.exists():
            raise FileNotFoundError(
                f"Teacher preset not found: {teacher_cfg_path}. "
                "Set pipeline.distill.require_teacher_preset=false to bypass."
            )

    distill_cfg = compose(config_name=config_name, overrides=[f"teacher={teacher_name}"])
    return _sync_train_context_to_distill_cfg(train_cfg, distill_cfg, teacher_checkpoint)


def _run_distillation_pipeline(
    cfg: DictConfig,
    teacher_trainer,
) -> Optional[dict]:
    pipeline_cfg = cfg.get("pipeline", {})
    if not pipeline_cfg.get("enabled", False):
        return None

    distill_pipeline_cfg = pipeline_cfg.get("distill", {})
    if not distill_pipeline_cfg.get("enabled", True):
        print("[Pipeline] Distillation stage is disabled. Skipping.")
        return None

    fail_fast = distill_pipeline_cfg.get("fail_fast", True)

    best_model_path = getattr(teacher_trainer, "best_model_path", None)
    if best_model_path is None:
        message = "Teacher training finished but best checkpoint was not created."
        if fail_fast:
            raise RuntimeError(message)
        print(f"[Pipeline] {message} Distillation stage is skipped.")
        return None

    teacher_checkpoint = Path(best_model_path)
    if not teacher_checkpoint.exists():
        message = f"Teacher best checkpoint does not exist: {teacher_checkpoint}"
        if fail_fast:
            raise FileNotFoundError(message)
        print(f"[Pipeline] {message} Distillation stage is skipped.")
        return None

    _release_trainer_resources(teacher_trainer)
    _cleanup_memory()

    try:
        distill_cfg = _build_distill_cfg(cfg, teacher_checkpoint)
    except Exception as exc:
        if fail_fast:
            raise
        print(
            f"[Pipeline] Failed to build distillation config ({exc}). Distillation stage is skipped."
        )
        return None

    print(
        f"[Pipeline] Starting distillation with teacher checkpoint: {teacher_checkpoint.resolve()}"
    )
    distill_trainer = ModelBuilder.create_trainer(distill_cfg)

    if distill_cfg.get("debug", False):
        distill_trainer.dry_run()
        return {
            "pipeline_metric": None,
            "pipeline_metric_key": distill_cfg.get("pipeline", {})
            .get("distill", {})
            .get("metric_key", "pipeline/distill_final_dice"),
            "pipeline_metric_dataset": None,
            "log_dir": str(getattr(distill_trainer, "log_dir", "")),
            "best_model_path": str(getattr(distill_trainer, "best_model_path", "")),
            "final_metrics": {},
        }

    try:
        result = distill_trainer.train()
    except Exception as exc:
        if fail_fast:
            raise
        print(f"[Pipeline] Distillation stage failed ({exc}).")
        return None

    if not isinstance(result, dict):
        return None
    return result


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    """Main training entry point using unified Trainer system."""
    _set_cuda_visible_devices(cfg)
    _maybe_disable_teacher_wandb_for_sweep(cfg)

    trainer = ModelBuilder.create_trainer(cfg)
    trainer.setup(mode="train")
    trainer.train()

    pipeline_result = _run_distillation_pipeline(cfg, trainer)
    if not pipeline_result:
        return

    metric_key = pipeline_result.get("pipeline_metric_key", "pipeline/distill_final_dice")
    metric_value = pipeline_result.get("pipeline_metric")
    metric_dataset = pipeline_result.get("pipeline_metric_dataset")
    log_dir = pipeline_result.get("log_dir")

    if metric_value is not None:
        print(
            f"[Pipeline] Completed. {metric_key}={metric_value:.6f} "
            f"(dataset={metric_dataset})"
        )
    else:
        print("[Pipeline] Completed. Pipeline metric was not found in distillation run.")

    if log_dir:
        print(f"[Pipeline] Distillation logs: {log_dir}")


if __name__ == "__main__":
    main()
