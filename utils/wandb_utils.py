"""Unified W&B run-identity helper shared by every trainer.

Goal: experiments *accumulate* in a small number of projects while each run is
individually identifiable and related runs cluster together.

  project   set by config (medfm-train / medfm-distill)
  group     "{job_type}/{method}/{datasets}"   <- runs of one study cluster here
  job_type  train | distill | ...
  name      compact, unique per run (key hparams + timestamp)
  tags      filterable dimensions (model/method, datasets, num_classes, hparams)

Any field explicitly set under ``cfg.wandb`` (group / name / tags) takes
precedence over the auto-composed value, so sweeps can pin their own grouping.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from utils.distill_utils import get_dataset_short_name, get_experiment_tags


def _is_sweep() -> bool:
    return os.environ.get("WANDB_SWEEP_ID") is not None


def _method_label(cfg: DictConfig, job_type: str) -> str:
    """Short label of *what* is being run: method name for distillation,
    model name for supervised training."""
    if job_type == "distill" and "method" in cfg:
        return str(cfg.method.get("name", "distill"))
    model = cfg.get("model")
    if isinstance(model, DictConfig):
        return str(model.get("name", "model"))
    return job_type


def _auto_group(cfg: DictConfig, job_type: str) -> str:
    return f"{job_type}/{_method_label(cfg, job_type)}/{get_dataset_short_name(cfg)}"


def _auto_name(cfg: DictConfig, job_type: str) -> str:
    ts = datetime.now().strftime("%m%d_%H%M%S")
    datasets = get_dataset_short_name(cfg)
    if job_type == "distill":
        teacher = cfg.get("teacher", "teacher")
        student = cfg.get("student", "student")
        method = _method_label(cfg, job_type)
        # An explicit experiment label (set by sweeps) stays in the name.
        label = cfg.get("run_name")
        core = f"{teacher}-{student}_{method}_{datasets}"
        return f"{core}_{label}_{ts}" if label else f"{core}_{ts}"

    # supervised training
    model = cfg.get("model", {})
    name = model.get("name", "model") if isinstance(model, DictConfig) else "model"
    enc = model.get("encoder_mode") if isinstance(model, DictConfig) else None
    dec = model.get("decoder_mode") if isinstance(model, DictConfig) else None
    if enc or dec:
        return f"{name}_{enc or 'na'}-{dec or 'na'}_{datasets}_{ts}"
    return f"{name}_{datasets}_{ts}"


def _auto_tags(cfg: DictConfig, job_type: str) -> List[str]:
    tags = [job_type, _method_label(cfg, job_type)]
    datasets = get_dataset_short_name(cfg)
    tags += [d for d in str(datasets).split("+") if d]
    nc = cfg.get("data", {}).get("num_classes")
    if nc is not None:
        tags.append("binary" if int(nc) == 1 else f"nc{int(nc)}")
    if job_type == "distill":
        tags += [f"teacher:{cfg.get('teacher')}", f"student:{cfg.get('student')}"]
    tags += get_experiment_tags(cfg)
    # de-dup, preserve order, drop falsy
    seen, out = set(), []
    for t in tags:
        t = str(t)
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def resolve_wandb_identity(
    cfg: DictConfig, default_job_type: str
) -> Dict[str, Any]:
    """Compose the W&B init kwargs (project/entity/group/name/job_type/tags/mode).

    Explicit ``cfg.wandb.{group,name,tags}`` win over the auto scheme. In a W&B
    sweep, ``name`` is left to W&B (returned as ``None``) but group/tags still
    apply so swept runs cluster.
    """
    wcfg = cfg.get("wandb", {})
    job_type = wcfg.get("job_type") or default_job_type

    group = wcfg.get("group") or _auto_group(cfg, job_type)

    explicit_name = wcfg.get("name")
    name = None if _is_sweep() else (explicit_name or _auto_name(cfg, job_type))

    explicit_tags = wcfg.get("tags")
    if explicit_tags is not None:
        explicit_tags = OmegaConf.to_container(explicit_tags, resolve=True) or []
    tags = list(explicit_tags or []) + _auto_tags(cfg, job_type)
    # de-dup preserving order
    seen, dedup = set(), []
    for t in tags:
        t = str(t)
        if t and t not in seen:
            seen.add(t)
            dedup.append(t)

    mode = wcfg.get("mode", None)
    if wcfg.get("disabled", False):
        mode = "disabled"

    return {
        "entity": wcfg.get("entity", "hheo"),
        "project": wcfg.get("project", f"medfm-{job_type}"),
        "job_type": job_type,
        "group": group,
        "name": name,
        "tags": dedup,
        "mode": mode,
    }
