"""Hydra structured-config schemas for the stable, shared config groups.

These dataclasses give the most-used config groups (``data`` and ``training``)
a typed schema so Hydra/OmegaConf can validate them at compose time — catching
typos, wrong types, and unknown keys before a run starts.

Scope is intentionally limited to the groups whose structure is stable across
experiments. Heterogeneous, instantiation-driven groups (``method``, ``model``,
``trainer``) keep their plain-yaml form: they vary per distiller/model and carry
``_target_`` / ``_partial_`` keys that are better validated by Hydra's
instantiate machinery than by a single rigid schema.

Registration is opt-in: importing this module and calling
:func:`register_schemas` adds the schemas to Hydra's ConfigStore under the
``data``/``training`` groups (name ``base_schema``). Runtime entrypoints are
left untouched, so this does not change how any experiment composes today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore


@dataclass
class SamplingConfig:
    """Balanced multi-dataset / class-level sampling (``data.sampling``)."""

    enabled: bool = False
    alpha: float = 0.5
    # image-level class -> weight (0=normal, 1=benign, 2=malignant)
    class_weights: Dict[int, float] = field(
        default_factory=lambda: {0: 1.0, 1: 1.0, 2: 1.0}
    )
    normal_cap: Optional[float] = None
    decouple_dataset_class: bool = True
    num_samples: Optional[int] = None
    replacement: bool = True
    seed: int = 42


@dataclass
class DataConfig:
    """Dataset selection + global image/sampling settings (``data`` group)."""

    name: str = "dynamic"
    type: str = "Dynamic"
    train: List[str] = field(default_factory=lambda: ["BUSBRA", "BUSI", "B"])
    val: List[str] = field(default_factory=list)
    test: List[str] = field(
        default_factory=lambda: ["BUID", "BUS_UCLM", "BUS_UCLM_filtered"]
    )
    num_classes: int = 3
    img_size: int = 224
    auto_img_size_by_sam_type: bool = True
    sam_img_size_map: Dict[str, int] = field(
        default_factory=lambda: {"vit_b": 224, "vit_l": 1024, "vit_h": 1024}
    )
    normalization: str = "imagenet"
    augment_train: bool = True
    sampling: SamplingConfig = field(default_factory=SamplingConfig)


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 20
    min_delta: float = 1e-6


@dataclass
class TrainingConfig:
    """Optimisation schedule (``training`` group, distillation variant)."""

    num_epochs: int = 1000
    lr: float = 1e-4
    teacher_lr: float = 1e-5
    warmup_epochs: int = 5
    batch_size: int = 8
    num_workers: int = 8
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass
class WandbConfig:
    """W&B run identity + logging behaviour (``wandb`` group).

    ``group``/``name`` are ``None`` by default; the trainer fills them in from
    the run's hyper-parameters (see ``utils/wandb_utils.py``) unless overridden.
    """

    entity: str = "hheo"
    project: str = "medfm"
    job_type: str = "train"
    group: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    mode: Optional[str] = None  # None | online | offline | disabled
    disabled: bool = False
    log_model: bool = True


def register_schemas(cs: Optional[ConfigStore] = None) -> ConfigStore:
    """Register the schemas under their config groups (name ``base_schema``).

    Idempotent: safe to call more than once. Returns the ConfigStore used.
    """
    cs = cs or ConfigStore.instance()
    cs.store(group="data", name="base_schema", node=DataConfig)
    cs.store(group="training", name="base_schema", node=TrainingConfig)
    cs.store(group="wandb", name="base_schema", node=WandbConfig)
    return cs
