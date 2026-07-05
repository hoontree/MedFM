"""Unified run-identity helper shared by every trainer.

Goal: experiments *accumulate* in a small number of projects while each run is
individually identifiable, related runs cluster together, and the on-disk
experiment directory tells the same story as the W&B run — because both are
built from the same resolved components, computed once per run.

  project   set by config (medfm)
  group     "{job_type}/{method}/{datasets}", or a sweep's explicit
            ``wandb.group`` (conventionally "{topic}/{key_param}") when a
            study wants its own consistent cluster label
  job_type  train | distill | ...
  name      compact, unique per run (model/teacher-student + key hparams +
            timestamp) — also the on-disk leaf directory name
  tags      filterable dimensions (model/method, datasets, num_classes, hparams)

Any field explicitly set under ``cfg.wandb`` (group / name / tags) takes
precedence over the auto-composed value, so sweeps can pin their own grouping.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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


@dataclass(frozen=True)
class _RunComponents:
    """Pieces shared by the W&B run name and the on-disk directory name.

    Computed exactly once per run (one ``datetime.now()`` call, one
    ``get_experiment_tags`` call) so the two never drift apart.
    """

    ts: str
    desc: str        # model name, or "{teacher}-{student}_{method}"
    datasets: str
    label: Optional[str]   # cfg.run_name, if a sweep set one
    tags: List[str]        # from get_experiment_tags (bs/lr/lora-rank/method coeffs)

    @property
    def tag_suffix(self) -> str:
        return ("_" + "_".join(self.tags)) if self.tags else ""

    @property
    def label_suffix(self) -> str:
        return f"_{self.label}" if self.label else ""


def _run_components(cfg: DictConfig, job_type: str) -> _RunComponents:
    datasets = get_dataset_short_name(cfg)
    if job_type == "distill":
        teacher = cfg.get("teacher", "teacher")
        student = cfg.get("student", "student")
        method = _method_label(cfg, job_type)
        desc = f"{teacher}-{student}_{method}"
    else:
        model = cfg.get("model", {})
        desc = model.get("name", "model") if isinstance(model, DictConfig) else "model"

    return _RunComponents(
        ts=datetime.now().strftime("%Y%m%d_%H%M%S"),
        desc=str(desc),
        datasets=str(datasets),
        label=cfg.get("run_name"),
        tags=get_experiment_tags(cfg),
    )


def _compose_name(c: _RunComponents) -> str:
    """Human-first ordering for W&B's run list: description, then hparams,
    then a timestamp to disambiguate re-runs."""
    return f"{c.desc}_{c.datasets}{c.label_suffix}{c.tag_suffix}_{c.ts}"


def _compose_dir_name(c: _RunComponents) -> str:
    """Timestamp-first ordering for the on-disk leaf dir: keeps directory
    listings sorted chronologically and preserves the historical
    ``{timestamp}{label}{tags}`` suffix that resume auto-detection matches
    against (see ``DistillTrainer._resolve_resume_path``)."""
    return f"{c.ts}{c.label_suffix}{c.tag_suffix}"


def _dedup_tags(tags: List) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for t in tags:
        t = str(t)
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _auto_tags(cfg: DictConfig, job_type: str, datasets: str) -> List[str]:
    tags: List = [job_type, _method_label(cfg, job_type)]
    tags += [d for d in str(datasets).split("+") if d]
    nc = cfg.get("data", {}).get("num_classes")
    if nc is not None:
        tags.append("binary" if int(nc) == 1 else f"nc{int(nc)}")
    if job_type == "distill":
        if teacher := cfg.get("teacher"):
            tags.append(f"teacher:{teacher}")
        if student := cfg.get("student"):
            tags.append(f"student:{student}")
    tags += get_experiment_tags(cfg)
    return _dedup_tags(tags)


def resolve_wandb_identity(
    cfg: DictConfig, default_job_type: str
) -> Dict[str, Any]:
    """Compose the W&B init kwargs (project/entity/group/name/job_type/tags/mode)
    plus a ``dir_name`` for the on-disk experiment directory.

    Explicit ``cfg.wandb.{group,name,tags}`` win over the auto scheme. In a W&B
    sweep, ``name`` is left to W&B (returned as ``None``) but ``dir_name`` is
    always a real string since a directory needs one regardless.
    """
    wcfg = cfg.get("wandb", {})
    job_type = wcfg.get("job_type") or default_job_type

    group = wcfg.get("group") or _auto_group(cfg, job_type)

    components = _run_components(cfg, job_type)
    auto_name = _compose_name(components)

    explicit_name = wcfg.get("name")
    dir_name = explicit_name or _compose_dir_name(components)
    name = None if _is_sweep() else (explicit_name or auto_name)

    explicit_tags = wcfg.get("tags")
    if explicit_tags is not None:
        explicit_tags = OmegaConf.to_container(explicit_tags, resolve=True) or []
    dedup = _dedup_tags(list(explicit_tags or []) + _auto_tags(cfg, job_type, components.datasets))

    mode = wcfg.get("mode", None)
    if wcfg.get("disabled", False):
        mode = "disabled"

    return {
        "entity": wcfg.get("entity", "hheo"),
        # One shared project for every job_type (config/wandb/*.yaml always
        # sets this explicitly; the default here only covers a config group
        # that omits it).
        "project": wcfg.get("project", "medfm"),
        "job_type": job_type,
        "group": group,
        "name": name,
        "tags": dedup,
        "mode": mode,
        "dir_name": dir_name,
    }


def build_experiment_dir(
    cfg: DictConfig, root_segment: str, identity: Dict[str, Any]
) -> Path:
    """Compose the on-disk experiment directory from an already-resolved identity.

    Structure: ``{output.dir}/{job_type}/{root_segment}/[{explicit_group}/]{dir_name}``.

    ``root_segment`` is the stable bucket other tooling keys off of —
    ``{model_name}`` for training, ``{teacher}_to_{student}`` for
    distillation — and is never touched by grouping so resume auto-detection
    (which globs immediate children of this bucket) keeps working. The
    explicit ``wandb.group`` override (e.g. a sweep's "{topic}/{key_param}"
    study label) nests one level below it *only* when a sweep actually sets
    one — ad hoc single runs stay flat.
    """
    output_dir = cfg.get("output", {}).get("dir", "logs")
    explicit_group = cfg.get("wandb", {}).get("group")
    parts = [identity["job_type"], root_segment]
    if explicit_group:
        parts.append(str(explicit_group))
    parts.append(identity["dir_name"])
    return Path(output_dir).joinpath(*parts)
