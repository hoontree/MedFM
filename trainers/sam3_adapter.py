"""
SAM3 Integration Module

This module provides two ways to integrate SAM3 into the unified training framework:

1. SAM3Orchestrator: Uses SAM3's native Trainer with full capabilities (DDP, AMP, etc.)
   - Recommended for production training
   - Supports all SAM3 features including multi-node training

2. SAM3TrainerAdapter: Inherits from BaseTrainer for simple single-GPU training
   - Useful for quick experiments and debugging
   - Integrates with the unified framework's logging and checkpointing

Usage:
    # Via main.py (uses SAM3TrainerAdapter)
    python main.py model=sam3

    # Via orchestrator for advanced usage
    python -m trainers.sam3_adapter --config your_config.yaml
"""

import os
import sys
import logging
import random
import warnings
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from .base_trainer import BaseTrainer
from utils.wandb_utils import resolve_wandb_identity, build_experiment_dir
from utils.checkpoint import load_model_weights

# The vendored SAM3 package (model/sam3_vendor, an unmodified fork of
# facebookresearch/sam3) imports `pkg_resources` and `timm.models.layers`,
# both deprecated. Every spawned DDP rank *and* every DataLoader worker
# process re-triggers these on import (each is a fresh interpreter that may
# never import trainers.sam3_adapter at all, e.g. workers only need to
# unpickle the Dataset), so a plain warnings.filterwarnings() call here only
# covers whichever process actually executes this module. PYTHONWARNINGS is
# read by CPython at interpreter *startup*, before any user code runs, and is
# inherited as a normal env var by every child process (spawn or fork) — the
# only filter that reliably reaches all of them. Also set filterwarnings for
# *this* process, since exporting the env var doesn't retroactively affect an
# already-running interpreter.
_PKG_RESOURCES_FILTER = "ignore:pkg_resources is deprecated:UserWarning"
_TIMM_LAYERS_FILTER = "ignore:Importing from timm.models.layers is deprecated:FutureWarning"
_extra_filters = ",".join([_PKG_RESOURCES_FILTER, _TIMM_LAYERS_FILTER])
os.environ["PYTHONWARNINGS"] = (
    f"{os.environ['PYTHONWARNINGS']},{_extra_filters}"
    if os.environ.get("PYTHONWARNINGS")
    else _extra_filters
)
warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    message=r"Importing from timm\.models\.layers is deprecated",
    category=FutureWarning,
)

# The vendored SAM3 package (model/sam3, symlinked to the model/sam3_vendor
# submodule's sam3/ dir) imports itself internally as a top-level `sam3`
# package (`from sam3.model.decoder import ...`), not `model.sam3....`. Put
# the submodule root on sys.path so both `model.sam3.X` (used below) and the
# package's own internal `sam3.X` imports resolve to the same files.
_SAM3_VENDOR_ROOT = Path(__file__).resolve().parent.parent / "model" / "sam3_vendor"
if str(_SAM3_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAM3_VENDOR_ROOT))

# SAM3's ViT-Det MLP calls an inference-only fused kernel that raises whenever
# autograd is on, so every path that back-propagates through the vision backbone
# must patch it. The patch is shared with model/sam3_student.py (SAM3 as a KD
# student) so the two cannot drift — see model/sam3_patches.py.
from model.sam3_patches import patch_fused_mlp_for_training as _patch_fused_mlp_for_training


class Sam3WandbWriter:
    """Rank-0-only W&B writer for SAM3's native DDP ``Trainer`` — mirrors the
    rank-gating pattern of ``sam3.train.utils.logger.TensorBoardWriterWrapper``
    (same env-var-based rank check, same ``log``/``log_dict`` interface) so it
    slots into ``Logger`` via ``_patch_logger_for_wandb`` below.

    ``identity`` is the *exact* dict ``SAM3Orchestrator`` resolved once via
    ``resolve_wandb_identity`` for the run's directory — reusing it here
    (instead of recomputing) is what keeps the W&B run and the on-disk
    ``exp_dir`` in sync. If ``wandb.disabled``/``mode=disabled`` was set,
    ``identity["mode"]=="disabled"`` and ``wandb.init`` returns a no-op run,
    so nothing extra is needed to respect that flag here.
    """

    def __init__(self, identity, run_config=None):
        from sam3.train.utils.train_utils import get_machine_local_and_dist_rank
        import wandb

        # Instantiated with _recursive_=False (see _patch_logger_for_wandb),
        # so both args may still be raw OmegaConf nodes rather than plain
        # dicts/None.
        if isinstance(identity, DictConfig):
            identity = OmegaConf.to_container(identity, resolve=True)
        if isinstance(run_config, DictConfig):
            run_config = OmegaConf.to_container(run_config, resolve=True)

        _, self._rank = get_machine_local_and_dist_rank()
        self._run = None
        if self._rank == 0:
            self._run = wandb.init(config=run_config or {}, **identity)

    def log(self, name, data, step):
        # Don't force `step` as the wandb step: the native Trainer logs several
        # independent, non-aligned counters through the same Logger — e.g. a
        # global per-iteration `self.steps[phase]` for Losses/Optim/Step_Stats,
        # but the *epoch* number (0, 1, 2, ...) for the end-of-epoch summary
        # dict — and wandb (unlike TensorBoard) enforces one strictly
        # monotonic step per run, silently dropping anything that goes
        # "backward". Let wandb auto-increment its own step per call instead
        # (never non-monotonic, so nothing is ever dropped) and keep the
        # caller's own counter as an ordinary logged field.
        if self._run is not None:
            self._run.log({name: data, "trainer/step": step})

    def log_dict(self, payload, step):
        if self._run is not None:
            self._run.log({**payload, "trainer/step": step})

    def log_hparams(self, hparams, meters):
        if self._run is not None:
            self._run.config.update(hparams, allow_val_change=True)

    def close(self):
        if self._run is not None:
            self._run.finish()
            self._run = None


_logger_patched_for_wandb = False


def _patch_logger_for_wandb():
    """SAM3's native ``Logger`` (sam3.train.utils.logger.Logger) only ever
    wires up TensorBoard — it silently ignores ``logging_conf.wandb_writer``
    entirely (see the class docstring: "supports tensorboard only for
    simplicity"). Patch it to also instantiate + forward to a wandb writer
    when one is configured (see ``build_sam3_config``'s "wandb_writer" entry),
    so the native DDP trainer gets the same W&B tracking every other trainer
    in this project has. Idempotent — safe to call more than once.

    Patches ``trainers.sam3_trainer.Logger`` specifically — not
    ``sam3.train.utils.logger.Logger`` — because this codebase has two live
    import paths to the same vendored files (``sam3.*`` via the
    ``model/sam3_vendor`` sys.path entry, and ``model.sam3.*`` via the
    ``model/sam3`` symlink); something along that chain ends up loading
    ``sam3.train.utils.logger`` a second time as a distinct module object; the
    name ``trainers.sam3_trainer.Logger`` (bound at that module's own import
    time) is the *only* reference guaranteed to be the one it actually calls.
    """
    global _logger_patched_for_wandb
    if _logger_patched_for_wandb:
        return
    from hydra.utils import instantiate
    import trainers.sam3_trainer as _trainer_mod

    LoggerClass = _trainer_mod.Logger
    original_init = LoggerClass.__init__

    def patched_init(self, logging_conf):
        original_init(self, logging_conf)
        wb_conf = getattr(logging_conf, "wandb_writer", None)
        # _recursive_=False: wb_conf.run_config embeds the *entire* unified
        # cfg (for wandb.init(config=...)) so the logged run config matches
        # what every other trainer logs — but that cfg has its own
        # trainer._target_ (SAM3TrainerAdapter/SAM3Orchestrator) buried
        # inside it, and hydra's default recursive instantiate walks into
        # *any* nested `_target_` it finds, including that one. Keep
        # run_config a plain nested structure; only Sam3WandbWriter itself
        # (the top-level target) should actually get constructed.
        self.wandb_logger = instantiate(wb_conf, _recursive_=False) if wb_conf else None

    def patched_log_dict(self, payload, step):
        if self.tb_logger:
            self.tb_logger.log_dict(payload, step)
        if getattr(self, "wandb_logger", None):
            self.wandb_logger.log_dict(payload, step)

    def patched_log(self, name, data, step):
        if self.tb_logger:
            self.tb_logger.log(name, data, step)
        if getattr(self, "wandb_logger", None):
            self.wandb_logger.log(name, data, step)

    def patched_log_hparams(self, hparams, meters):
        if self.tb_logger:
            self.tb_logger.log_hparams(hparams, meters)
        if getattr(self, "wandb_logger", None):
            self.wandb_logger.log_hparams(hparams, meters)

    LoggerClass.__init__ = patched_init
    LoggerClass.log_dict = patched_log_dict
    LoggerClass.log = patched_log
    LoggerClass.log_hparams = patched_log_hparams
    _logger_patched_for_wandb = True


@dataclass
class SAM3Config:
    """Configuration dataclass for SAM3 training."""

    # Model
    checkpoint_path: Optional[str] = None
    bpe_path: str = "model/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    enable_segmentation: bool = True

    # Training
    num_epochs: int = 20
    batch_size: int = 1
    num_workers: int = 4
    gradient_accumulation_steps: int = 1

    # Learning rate
    lr_transformer: float = 8e-5
    lr_vision_backbone: float = 2.5e-5
    lr_language_backbone: float = 5e-6
    weight_decay: float = 0.1

    # AMP
    amp_enabled: bool = True
    amp_dtype: str = "bfloat16"

    # Distributed
    backend: str = "nccl"
    find_unused_parameters: bool = True

    # Resolution
    resolution: int = 1008

    # Logging
    log_freq: int = 10
    val_epoch_freq: int = 1


class SAM3Orchestrator:
    """
    Orchestrator for running SAM3's native training infrastructure.

    This class bridges the unified framework's configuration system with
    SAM3's native Hydra-based configuration and Trainer class.

    Features:
    - Converts unified config to SAM3's config format
    - Supports both single-node and multi-node training
    - Preserves all SAM3 capabilities (DDP, AMP, gradient accumulation)
    - Integrates with the framework's output directory structure

    Example:
        orchestrator = SAM3Orchestrator(cfg)
        orchestrator.run()
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize SAM3 Orchestrator.

        Args:
            cfg: Unified framework configuration
        """
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)

        # Single source of truth for both the on-disk directory and the W&B
        # run identity — the exact same call every other trainer's
        # BaseTrainer._setup_directories()/_setup_wandb() makes (see
        # utils/wandb_utils.py). Resolved once here (one timestamp, one
        # hparam-tag computation) and reused unchanged for the W&B run
        # initialized inside the spawned rank-0 process — see
        # build_sam3_config()'s "wandb_writer" entry and Sam3WandbWriter below.
        self.identity = resolve_wandb_identity(self.cfg, default_job_type="train")

        # Setup paths
        self._setup_paths()

    def _setup_paths(self):
        """Setup experiment directories — same layout every other model gets:
        logs/train/{model_name}/[{group}/]{run_name}. See
        utils.wandb_utils.build_experiment_dir."""
        model_name = self.cfg.model.get("name", "sam3")
        self.exp_dir = build_experiment_dir(
            self.cfg, root_segment=model_name, identity=self.identity
        )
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = self.exp_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def build_sam3_config(self) -> DictConfig:
        """
        Convert unified framework config to SAM3's native config format.

        Returns:
            OmegaConf DictConfig compatible with SAM3's Trainer
        """
        sam3_cfg = self.cfg.get("sam3", {})
        training_cfg = self.cfg.get("training", {})

        # Build SAM3-compatible config
        config = {
            "paths": {
                "checkpoint_path": sam3_cfg.get("checkpoint_path"),
                "bpe_path": sam3_cfg.get(
                    "bpe_path", "model/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
                ),
                "experiment_log_dir": str(self.exp_dir),
            },
            "scratch": {
                "enable_segmentation": sam3_cfg.get("enable_segmentation", True),
                "resolution": sam3_cfg.get("resolution", 1008),
                "train_batch_size": training_cfg.get("batch_size", 1),
                "val_batch_size": 1,
                "num_train_workers": training_cfg.get("num_workers", 4),
                "num_val_workers": 2,
                "max_data_epochs": training_cfg.get("max_epochs", 20),
                "gradient_accumulation_steps": training_cfg.get(
                    "gradient_accumulation_steps", 1
                ),
                "lr_scale": sam3_cfg.get("lr_scale", 0.1),
                "train_norm_mean": [0.5, 0.5, 0.5],
                "train_norm_std": [0.5, 0.5, 0.5],
                "val_norm_mean": [0.5, 0.5, 0.5],
                "val_norm_std": [0.5, 0.5, 0.5],
            },
            "trainer": {
                "_target_": "trainers.sam3_trainer.Trainer",
                "max_epochs": training_cfg.get("max_epochs", 20),
                "accelerator": "cuda",
                # Aligned to the project-wide policy: seed 42 (matching every other
                # model) and cudnn determinism driven by hardware.deterministic
                # (default True) instead of the native trainer's hardcoded
                # non-deterministic default — SAM3 runs are now seed-comparable with
                # the rest. Set hardware.deterministic=false to restore SAM3's faster
                # non-deterministic kernels.
                "seed_value": self.cfg.get("hardware", {}).get("seed", 42),
                "cuda": {
                    "cudnn_deterministic": self.cfg.get("hardware", {}).get("deterministic", True),
                    "cudnn_benchmark": not self.cfg.get("hardware", {}).get("deterministic", True),
                },
                "val_epoch_freq": sam3_cfg.get("val_epoch_freq", 1),
                # "train_only" by default: the native trainer's own eval is
                # detection-style, and this project reports metrics by loading
                # the saved checkpoint into SAM3TrainerAdapter instead. Override
                # with sam3.native_mode=train to also run the native val loop.
                "mode": sam3_cfg.get("native_mode", "train_only"),
                "gradient_accumulation_steps": training_cfg.get(
                    "gradient_accumulation_steps", 1
                ),
                "distributed": {
                    "backend": "nccl",
                    # SAM3's ViT-Det backbone uses activation checkpointing,
                    # which re-enters backward and confuses DDP's per-iteration
                    # "which params got a gradient" bucket bookkeeping unless
                    # the graph structure is declared static (PyTorch's own
                    # recommendation for DDP + activation checkpointing).
                    # static_graph=True auto-detects unused parameters itself
                    # (that's exactly what fixed the earlier "did not receive
                    # grad" crash) — find_unused_parameters=True is redundant
                    # once it's on, and PyTorch warns loudly about the
                    # combination.
                    "gradient_as_bucket_view": True,
                    "static_graph": True,
                },
                "optim": {
                    "amp": {
                        "enabled": sam3_cfg.get("amp_enabled", True),
                        "amp_dtype": sam3_cfg.get("amp_dtype", "bfloat16"),
                    },
                    "optimizer": {
                        "_target_": "torch.optim.AdamW",
                    },
                    # A single constant-LR / constant-WD param group. The native
                    # Trainer's construct_optimizer requires `options` (it builds
                    # per-group schedulers and the train loop indexes
                    # self.optim.schedulers[j] when logging); a plain optimizer
                    # with no options would leave schedulers=None and crash.
                    # (Simpler than the reference's per-backbone InvSqrt schedule
                    # — enough to make multi-GPU training run.)
                    "options": {
                        "lr": [
                            {
                                "scheduler": {
                                    "_target_": "fvcore.common.param_scheduler.ConstantParamScheduler",
                                    "value": training_cfg.get("base_lr", 1e-4)
                                    * sam3_cfg.get("lr_scale", 0.1),
                                }
                            }
                        ],
                        "weight_decay": [
                            {
                                "scheduler": {
                                    "_target_": "fvcore.common.param_scheduler.ConstantParamScheduler",
                                    "value": training_cfg.get("weight_decay", 0.1),
                                }
                            }
                        ],
                    },
                    "gradient_clip": {
                        "_target_": "sam3.train.optim.optimizer.GradientClipper",
                        "max_norm": 0.1,
                        "norm_type": 2,
                    },
                },
                "checkpoint": {
                    "save_dir": str(self.ckpt_dir),
                    "save_freq": training_cfg.get("save_interval", 5),
                },
                "logging": {
                    "log_dir": str(self.log_dir),
                    "log_freq": sam3_cfg.get("log_freq", 10),
                    "tensorboard_writer": {
                        "_target_": "sam3.train.utils.logger.make_tensorboard_logger",
                        "log_dir": str(self.exp_dir / "tensorboard"),
                        "flush_secs": 120,
                        "should_log": True,
                    },
                    # Same identity as the exp_dir above (self.identity —
                    # resolve_wandb_identity, computed once in __init__), so
                    # the W&B run name/group/tags match the on-disk path
                    # exactly. Instantiated inside the spawned rank-0 process
                    # (_patch_logger_for_wandb makes the native Logger class
                    # forward to it) — dir_name isn't a wandb.init() kwarg.
                    "wandb_writer": {
                        "_target_": "trainers.sam3_adapter.Sam3WandbWriter",
                        "identity": {k: v for k, v in self.identity.items() if k != "dir_name"},
                        "run_config": OmegaConf.to_container(self.cfg, resolve=True),
                    },
                },
                "model": {
                    "_target_": "sam3.model_builder.build_sam3_image_model",
                    "bpe_path": sam3_cfg.get(
                        "bpe_path", "model/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
                    ),
                    "device": "cpu",
                    "eval_mode": False,
                    "enable_segmentation": sam3_cfg.get("enable_segmentation", True),
                    "checkpoint_path": sam3_cfg.get("checkpoint_path"),
                },
                "data": self._build_data_config(),
                "loss": self._build_loss_config(),
                "meters": self._build_meters_config(),
            },
            "launcher": {
                "num_nodes": 1,
                "gpus_per_node": len(self.cfg.get("hardware", {}).get("gpu_ids", [0])),
                "experiment_log_dir": str(self.exp_dir),
            },
            "submitit": {
                "use_cluster": False,
                "port_range": [10000, 65000],
            },
        }

        return OmegaConf.create(config)

    def _sam3_transforms_cfg(self):
        """Hydra `_target_` config for the transform pipeline (mirrors
        SAM3TrainerAdapter._build_sam3_transforms): decode lazy RLE, to tensor,
        normalize with SAM3's [0.5]*3. Images are already pre-resized to
        `resolution` by tools/generate_sam3_coco_annotations.py, so no
        resize/pad step is needed."""
        sam3_cfg = self.cfg.get("sam3", {})
        mean = list(sam3_cfg.get("norm_mean", [0.5, 0.5, 0.5]))
        std = list(sam3_cfg.get("norm_std", [0.5, 0.5, 0.5]))
        return [
            {
                "_target_": "sam3.train.transforms.basic_for_api.ComposeAPI",
                "transforms": [
                    {"_target_": "sam3.train.transforms.segmentation.DecodeRle"},
                    {"_target_": "sam3.train.transforms.basic_for_api.ToTensorAPI"},
                    {
                        "_target_": "sam3.train.transforms.basic_for_api.NormalizeAPI",
                        "mean": mean,
                        "std": std,
                    },
                ],
            }
        ]

    def _dataset_cfg(self, keys, training: bool):
        """Build a (possibly Concat) Sam3ImageDataset config from COCO-manifest
        keys — same manifest/keys the single-GPU adapter consumes."""
        import json

        sam3_cfg = self.cfg.get("sam3", {})
        manifest = json.loads(Path(sam3_cfg["coco_manifest"]).read_text())

        def one(key):
            return {
                "_target_": "sam3.train.data.sam3_image_dataset.Sam3ImageDataset",
                "img_folder": manifest[key]["img_folder"],
                "ann_file": manifest[key]["ann_file"],
                "transforms": self._sam3_transforms_cfg(),
                "max_ann_per_img": sam3_cfg.get("max_ann_per_img", 100),
                "multiplier": 1,
                "training": training,
                "load_segmentation": sam3_cfg.get("enable_segmentation", True),
                "max_train_queries": sam3_cfg.get("max_train_queries", 50),
                "max_val_queries": sam3_cfg.get("max_val_queries", 50),
            }

        keys = list(keys)
        if len(keys) == 1:
            return one(keys[0])
        return {
            "_target_": "torch.utils.data.ConcatDataset",
            "datasets": [one(k) for k in keys],
        }

    def _collate_cfg(self, dict_key: str):
        """Partial collate_fn_api bound to a dict_key. The native Trainer's
        `_step` does `key, batch = batch.popitem()` and routes to the loss
        registered under `key` (falling back to 'default'=DummyLoss). Train
        uses 'all' → real Sam3LossWrapper; val uses 'val' → DummyLoss, which
        sidesteps the eval-mode missing-`indices` issue (SAM3 only computes
        matcher indices when training). Real val metrics come from the adapter
        loading the saved checkpoint, per the module/train.py notes."""
        return {
            "_target_": "sam3.train.data.collator.collate_fn_api",
            "_partial_": True,
            "dict_key": dict_key,
            "with_seg_masks": self.cfg.get("sam3", {}).get("enable_segmentation", True),
        }

    def _build_data_config(self) -> Dict:
        """Build SAM3 native data config from the COCO manifest (same manifest
        and datasets/splits the single-GPU adapter uses — see
        tools/generate_sam3_coco_annotations.py). Each split is a TorchDataset
        (which internally builds a DistributedSampler, so DDP shards the data
        across GPUs)."""
        sam3_cfg = self.cfg.get("sam3", {})
        training_cfg = self.cfg.get("training", {})

        return {
            "train": {
                "_target_": "sam3.train.data.torch_dataset.TorchDataset",
                "dataset": self._dataset_cfg(sam3_cfg["coco_train_datasets"], training=True),
                "shuffle": True,
                "batch_size": training_cfg.get("batch_size", 1),
                "num_workers": training_cfg.get("num_workers", 4),
                "pin_memory": True,
                "drop_last": True,
                "collate_fn": self._collate_cfg("all"),
            },
            "val": {
                "_target_": "sam3.train.data.torch_dataset.TorchDataset",
                "dataset": self._dataset_cfg(sam3_cfg["coco_val_datasets"], training=False),
                "shuffle": False,
                "batch_size": 1,
                "num_workers": min(2, training_cfg.get("num_workers", 4)),
                "pin_memory": True,
                "drop_last": False,
                "collate_fn": self._collate_cfg("val"),
            },
        }

    def _build_loss_config(self) -> Dict:
        """Build SAM3 loss configuration."""
        sam3_cfg = self.cfg.get("sam3", {})
        enable_segmentation = sam3_cfg.get("enable_segmentation", True)

        if enable_segmentation:
            return {
                "all": {
                    "_target_": "sam3.train.loss.sam3_loss.Sam3LossWrapper",
                    "matcher": {
                        "_target_": "sam3.train.matcher.BinaryHungarianMatcherV2",
                        "focal": True,
                        "cost_class": 2.0,
                        "cost_bbox": 5.0,
                        "cost_giou": 2.0,
                        "alpha": 0.25,
                        "gamma": 2,
                    },
                    # The decoder always emits a one-to-many aux head; the loss
                    # wrapper needs its own matcher for it (same values as
                    # SAM3's reference fine-tuning recipe). Without it the loss
                    # crashes on 'NoneType is not callable'.
                    "o2m_matcher": {
                        "_target_": "sam3.train.matcher.BinaryOneToManyMatcher",
                        "alpha": 0.3,
                        "threshold": 0.4,
                        "topk": 4,
                    },
                    "o2m_weight": 2.0,
                    # single-process DDP has torch.distributed initialized, but
                    # keep loss normalization local to the batch for stability.
                    "normalization": "local",
                    "loss_fns_find": [
                        {
                            "_target_": "sam3.train.loss.loss_fns.Boxes",
                            "weight_dict": {
                                "loss_bbox": 5.0,
                                "loss_giou": 2.0,
                            },
                        },
                        {
                            "_target_": "sam3.train.loss.loss_fns.IABCEMdetr",
                            "weight_dict": {
                                "loss_ce": 20.0,
                                "presence_loss": 20.0,
                            },
                            "pos_weight": 10.0,
                            "alpha": 0.25,
                            "gamma": 2,
                            "use_presence": True,
                        },
                        {
                            "_target_": "sam3.train.loss.loss_fns.Masks",
                            "weight_dict": {
                                "loss_mask": 200.0,
                                "loss_dice": 10.0,
                            },
                        },
                    ],
                },
                "default": {
                    "_target_": "sam3.train.loss.sam3_loss.DummyLoss",
                },
            }
        else:
            return {
                "all": {
                    "_target_": "sam3.train.loss.sam3_loss.DummyLoss",
                },
                "default": {
                    "_target_": "sam3.train.loss.sam3_loss.DummyLoss",
                },
            }

    def _build_meters_config(self) -> Dict:
        """Build SAM3 meters configuration.

        `_check_val_key_match` asserts ``set(val_keys) == set(meters_conf["val"].keys())``,
        where `val_keys` comes from the val collate's `dict_key` ("val") — so
        this must be a dict keyed by "val", not None (`None.keys()` crashes).
        No actual meters are registered (empty inner dict); real metrics come
        from loading the checkpoint into SAM3TrainerAdapter after training —
        see the module docstring and train.py's native-trainer routing.
        """
        return {
            "val": {"val": {}},
        }

    def run(self):
        """
        Run SAM3 training using native Trainer.

        This method sets up the distributed environment and launches
        SAM3's native training infrastructure.
        """
        from model.sam3.train.utils.train_utils import register_omegaconf_resolvers

        try:
            register_omegaconf_resolvers()
        except Exception as e:
            self.logger.warning(f"OmegaConf resolvers already registered: {e}")

        sam3_config = self.build_sam3_config()

        # Save config
        config_path = self.exp_dir / "sam3_config.yaml"
        with open(config_path, "w") as f:
            OmegaConf.save(sam3_config, f)

        self.logger.info(f"SAM3 config saved to {config_path}")
        self.logger.info(f"Experiment directory: {self.exp_dir}")

        # Run training
        self._run_single_node(sam3_config)

    def _run_single_node(self, cfg: DictConfig):
        """Run single-node training."""
        num_gpus = cfg.launcher.gpus_per_node
        main_port = random.randint(
            cfg.submitit.port_range[0], cfg.submitit.port_range[1]
        )

        if num_gpus == 1:
            self._single_proc_run(0, main_port, cfg, 1)
        else:
            torch.multiprocessing.set_start_method("spawn", force=True)
            torch.multiprocessing.start_processes(
                self._single_proc_run,
                args=(main_port, cfg, num_gpus),
                nprocs=num_gpus,
                start_method="spawn",
            )

    def _single_proc_run(
        self, local_rank: int, main_port: int, cfg: DictConfig, world_size: int
    ):
        """Single GPU process."""
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(main_port)
        os.environ["RANK"] = str(local_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        # Same inference-only fused-MLP kernel that blocks autograd in the
        # adapter path (see _patch_fused_mlp_for_training) is used by the ViT
        # backbone here too — patch it per spawned process before the model is
        # built so backbone fine-tuning works under DDP.
        _patch_fused_mlp_for_training()
        # Native Logger ignores wandb_writer entirely by default — see
        # _patch_logger_for_wandb. Must run before `instantiate(cfg.trainer)`
        # below constructs the Logger.
        _patch_logger_for_wandb()

        from hydra.utils import instantiate

        trainer = instantiate(cfg.trainer, _recursive_=False)
        trainer.run()


class SAM3TrainerAdapter(BaseTrainer):
    """
    Adapter that wraps SAM3 model for use with the unified BaseTrainer interface.

    This adapter provides a simplified interface for training SAM3 models
    within the unified framework. It's suitable for:
    - Quick experiments
    - Single-GPU training
    - Debugging and development

    For production training with full SAM3 capabilities (multi-GPU, DDP, etc.),
    use SAM3Orchestrator instead.

    Example:
        trainer = SAM3TrainerAdapter(cfg)
        trainer.setup('train')
        trainer.train()
    """

    def __init__(self, cfg: DictConfig, model: Optional[nn.Module] = None):
        """Initialize SAM3 trainer adapter."""
        super().__init__(cfg, model=model)

        self.sam3_cfg = cfg.get("sam3", {})
        self.resolution = self.sam3_cfg.get("resolution", 1008)
        self.enable_segmentation = self.sam3_cfg.get("enable_segmentation", True)

        # Loss components
        self.criterion = None
        self.scaler = None

    def _create_model(self):
        """Create or use provided SAM3 model."""
        if self.model is None:
            try:
                from model.sam3.model_builder import build_sam3_image_model
            except ImportError:
                self.logger.error(
                    "Failed to import SAM3. Make sure model/sam3 is properly set up."
                )
                raise

            _patch_fused_mlp_for_training()

            bpe_path = self.sam3_cfg.get(
                "bpe_path", "model/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
            )
            checkpoint_path = self.sam3_cfg.get("checkpoint_path")

            self.model = build_sam3_image_model(
                bpe_path=bpe_path,
                device="cpu",  # Will be moved to GPU later
                eval_mode=False,
                enable_segmentation=self.enable_segmentation,
                checkpoint_path=checkpoint_path,
            )

        self.model = self.model.to(self.device)

        # Setup AMP
        amp_enabled = self.sam3_cfg.get("amp_enabled", True)
        if amp_enabled:
            self.scaler = torch.amp.GradScaler(self.device)

        # Setup criterion (simplified for adapter)
        self._setup_criterion()

        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.logger.info(f"SAM3 Model loaded")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

    def _setup_criterion(self):
        """Setup loss criterion."""
        try:
            from model.sam3.train.loss.sam3_loss import Sam3LossWrapper, DummyLoss
            from model.sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
            from model.sam3.train.loss.loss_fns import Boxes, IABCEMdetr, Masks

            if self.enable_segmentation:
                matcher = BinaryHungarianMatcherV2(
                    focal=True,
                    cost_class=2.0,
                    cost_bbox=5.0,
                    cost_giou=2.0,
                    alpha=0.25,
                    gamma=2,
                )
                # The decoder always emits a one-to-many (_o2m) auxiliary head;
                # Sam3LossWrapper needs its own matcher for it regardless of
                # `enable_segmentation` — values match SAM3's own reference
                # fine-tuning recipe (roboflow_v100_full_ft_100_images.yaml).
                o2m_matcher = BinaryOneToManyMatcher(alpha=0.3, threshold=0.4, topk=4)

                self.criterion = Sam3LossWrapper(
                    matcher=matcher,
                    o2m_matcher=o2m_matcher,
                    o2m_weight=2.0,
                    # "global" (the default) calls torch.distributed.all_reduce,
                    # which requires an initialized process group — this adapter
                    # runs single-process, so use the non-distributed variant.
                    normalization="local",
                    loss_fns_find=[
                        Boxes(weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0}),
                        IABCEMdetr(
                            weight_dict={"loss_ce": 20.0, "presence_loss": 20.0},
                            pos_weight=10.0,
                            alpha=0.25,
                            gamma=2,
                            use_presence=True,
                        ),
                        Masks(weight_dict={"loss_mask": 200.0, "loss_dice": 10.0}),
                    ],
                )
            else:
                self.criterion = DummyLoss()

        except ImportError as e:
            self.logger.warning(f"Could not import SAM3 loss components: {e}")
            self.logger.warning("Using simple BCE + Dice loss instead")
            self._setup_simple_criterion()

    def _setup_simple_criterion(self):
        """Setup simple criterion as fallback."""
        from utils.criterion import DiceLoss

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(self.cfg.data.num_classes)
        self.criterion = None  # Will use manual loss computation

    def _create_dataloaders(self):
        """Create data loaders.

        SAM3's model (``Sam3Image.forward``) requires a ``BatchedDatapoint``
        (image + text/box find-queries), not a plain image tensor — so the
        unified framework's ``SegDatasetProcessor`` output only works here via
        the ``sam3.coco_manifest`` path below, which loads COCO-JSON manifests
        (see ``tools/generate_sam3_coco_annotations.py``, which derives them
        from the exact same datasets/splits every other model trains on).
        """
        if "data_config" in self.sam3_cfg:
            self._create_sam3_dataloaders()
        elif self.sam3_cfg.get("coco_manifest"):
            self._create_sam3_coco_dataloaders()
        else:
            # Fallback: unified framework's data loading. NOTE this produces
            # plain image/mask tensors that Sam3Image.forward() cannot consume
            # directly — kept only so _create_dataloaders never hard-fails.
            from utils.data_processing import SegDatasetProcessor

            self.train_loader, self.val_loader, self.test_loader = (
                SegDatasetProcessor.build_data_loaders(self.cfg)
            )

            self.logger.info(f"Train set size: {len(self.train_loader.dataset)}")
            self.logger.info(f"Val set size: {len(self.val_loader.dataset)}")

    def _build_sam3_transforms(self, training: bool):
        """Minimal transform pipeline for pre-resized (square, fixed
        resolution) COCO images: decode the lazy RLE segmentation, convert to
        tensor, normalize to SAM3's own [0.5]*3 mean/std. No resize/pad step
        needed since tools/generate_sam3_coco_annotations.py already bakes
        images at sam3.resolution."""
        from sam3.train.transforms.segmentation import DecodeRle
        from sam3.train.transforms.basic_for_api import ComposeAPI, NormalizeAPI, ToTensorAPI

        mean = self.sam3_cfg.get("norm_mean", [0.5, 0.5, 0.5])
        std = self.sam3_cfg.get("norm_std", [0.5, 0.5, 0.5])
        return [
            ComposeAPI(
                transforms=[
                    DecodeRle(),
                    ToTensorAPI(),
                    NormalizeAPI(mean=mean, std=std),
                ]
            )
        ]

    def _create_sam3_coco_dataloaders(self):
        """Build DataLoaders from COCO-JSON manifests (SAM3's real, native
        data contract) using SAM3's own Dataset/collate implementation —
        see tools/generate_sam3_coco_annotations.py for how the manifest is
        produced from medfm's shared ultrasound datasets/splits.
        """
        import json
        from functools import partial
        from torch.utils.data import ConcatDataset, DataLoader
        from sam3.train.data.sam3_image_dataset import Sam3ImageDataset
        from sam3.train.data.collator import collate_fn_api

        manifest = json.loads(Path(self.sam3_cfg["coco_manifest"]).read_text())

        def _build(keys, training: bool):
            datasets = [
                Sam3ImageDataset(
                    img_folder=manifest[key]["img_folder"],
                    ann_file=manifest[key]["ann_file"],
                    transforms=self._build_sam3_transforms(training),
                    max_ann_per_img=self.sam3_cfg.get("max_ann_per_img", 100),
                    multiplier=1,
                    training=training,
                    load_segmentation=self.enable_segmentation,
                    max_train_queries=self.sam3_cfg.get("max_train_queries", 50),
                    max_val_queries=self.sam3_cfg.get("max_val_queries", 50),
                )
                for key in keys
            ]
            return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

        # collate_fn_api wraps the real BatchedDatapoint in {dict_key: ...} —
        # an arbitrary label used upstream to multiplex several simultaneous
        # data sources. We only have one, so the key is just a fixed name.
        self._sam3_dict_key = "medfm"
        collate = partial(
            collate_fn_api, dict_key=self._sam3_dict_key, with_seg_masks=self.enable_segmentation
        )

        train_dataset = _build(self.sam3_cfg["coco_train_datasets"], training=True)
        val_dataset = _build(self.sam3_cfg["coco_val_datasets"], training=False)

        batch_size = self.cfg.training.get("batch_size", 1)
        num_workers = self.cfg.training.get("num_workers", 4)
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=collate, drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=min(2, num_workers), collate_fn=collate, drop_last=False,
        )

        self.logger.info(f"SAM3 COCO train set size: {len(train_dataset)}")
        self.logger.info(f"SAM3 COCO val set size: {len(val_dataset)}")

    def _create_sam3_dataloaders(self):
        """Create SAM3 native data loaders."""
        from hydra.utils import instantiate

        data_config = self.sam3_cfg.get("data_config", {})

        if "train" in data_config:
            train_dataset = instantiate(data_config["train"])
            self.train_loader = train_dataset.get_loader(epoch=0)

        if "val" in data_config:
            val_dataset = instantiate(data_config["val"])
            self.val_loader = val_dataset.get_loader(epoch=0)

    def _create_optimizer(self):
        """Create optimizer with parameter groups."""
        base_lr = self.cfg.training.get("base_lr", 1e-4)
        weight_decay = self.cfg.training.get("weight_decay", 0.1)

        # Create parameter groups for different learning rates
        vision_params = []
        transformer_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if "vision_backbone" in name:
                vision_params.append(param)
            elif "language_backbone" in name:
                # Skip language backbone for medical imaging
                continue
            else:
                transformer_params.append(param)

        lr_scale = self.sam3_cfg.get("lr_scale", 0.1)

        param_groups = [
            {"params": transformer_params, "lr": base_lr * lr_scale},
            {
                "params": vision_params,
                "lr": base_lr * lr_scale * 0.3,
            },  # Lower for backbone
        ]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=weight_decay,
        )

        self.logger.info(f"Optimizer: AdamW")
        self.logger.info(
            f"Base LR: {base_lr * lr_scale}, Vision LR: {base_lr * lr_scale * 0.3}"
        )

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        # Use warmup + inverse square root decay (similar to SAM3)
        total_steps = self.cfg.training.get("max_epochs", 20) * len(self.train_loader)
        warmup_steps = min(500, total_steps // 10)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            else:
                return max(0.1, (warmup_steps / (step + 1)) ** 0.5)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _sam3_forward(self, batch):
        """Move a SAM3-native ``{dict_key: BatchedDatapoint}`` batch (from
        ``_create_sam3_coco_dataloaders``) to device and run model + criterion.

        ``Sam3LossWrapper.forward(find_stages, find_targets)`` takes the
        model's raw ``SAM3Output`` and the batch's ``find_targets`` — but
        each is a ``BatchedFindTarget`` dataclass, not the dict-like object
        the loss functions index with ``targets["num_boxes"]``. The model
        itself owns the conversion (``Sam3Image.back_convert``, the same
        call ``sam3.train.trainer.Trainer._step`` makes) — not a separate
        labels tensor.
        """
        from sam3.model.utils.misc import copy_data_to_device

        batch = copy_data_to_device(batch, self.device)
        input_ = batch[self._sam3_dict_key]
        outputs = self.model(input_)
        find_targets = [self.model.back_convert(t) for t in input_.find_targets]
        loss_dict = self.criterion(outputs, find_targets)
        return loss_dict["core_loss"], loss_dict, outputs, find_targets

    def _extract_pred_gt_masks(self, model_output, find_targets):
        """Pull a (pred_logits, gt_mask) pair — both ``[1, 1, H, W]`` — out of
        SAM3's raw output, so it can be scored with the exact same
        ``utils.evaluate.Evaluator_seg`` every other trainer in this project
        uses (Dice/IoU/HD95/Sensitivity/Specificity/PixelAcc/BFScore/ECE).

        Uses the same matched-pair indexing as the Masks loss
        (``model/sam3_vendor/sam3/train/loss/loss_fns.py::Masks.get_loss``):
        the Hungarian matcher's ``indices`` (already computed inside the
        model's forward — see ``Sam3Image._compute_matching``, gated on
        ``self.training``) tell us which predicted query, if any, was
        assigned to the image's GT object.

        Assumes ``val_loader``/``test_loader`` batch_size=1 and our COCO
        manifest's single-object-per-image convention (every foreground
        pixel collapsed into one "lesion" instance per image — see
        ``tools/generate_sam3_coco_annotations.py``), so there is at most
        one GT object to match per call.
        """
        import torch.nn.functional as F
        from sam3.model.model_misc import SAM3Output

        with SAM3Output.iteration_mode(
            model_output, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
        ) as stages:
            step_out = stages[0][-1]  # only find-stage; last (only, in eval) step
        target = find_targets[0]

        device = step_out["pred_masks"].device
        batch_idx, src_idx, tgt_idx = step_out["indices"]

        if target["masks"] is None or batch_idx.numel() == 0:
            # No GT object in this image (negative/"normal" sample) — the
            # correct prediction is an empty mask, so score against one.
            empty = torch.zeros(1, 1, self.resolution, self.resolution, device=device)
            return empty, empty.clone()

        # batch_size=1 and >=0 GT object per image, so the first matched pair
        # (if any) is the only one that can exist. `tgt_idx` (indices[2]) is
        # None whenever the matcher didn't need to reorder targets — same
        # convention Masks.get_loss follows: use target position 0 directly.
        b, q = batch_idx[0].item(), src_idx[0].item()
        t = tgt_idx[0].item() if tgt_idx is not None else 0
        pred_logits = step_out["pred_masks"][b, q][None, None].float()
        gt_mask = target["masks"][t][None, None].float().to(device)
        pred_logits = F.interpolate(
            pred_logits, size=gt_mask.shape[-2:], mode="bilinear", align_corners=False
        )
        return pred_logits, gt_mask

    def _legacy_tensor_forward(self, batch):
        """Fallback when _create_dataloaders used the plain SegDatasetProcessor
        tensors instead of the SAM3-native COCO path. Kept only so callers
        never hard-crash on an unexpected config — Sam3Image.forward() does
        NOT actually accept a plain image tensor (see module docstring on
        _create_sam3_coco_dataloaders for the real path)."""
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            images = batch[0].to(self.device)
            labels = batch[1].to(self.device)
        elif isinstance(batch, dict):
            images = batch.get("image", batch.get("img_batch")).to(self.device)
            labels = batch.get("label", batch.get("find_targets"))
        else:
            self.logger.warning(f"Unknown batch format: {type(batch)}")
            return None

        outputs = self.model(images)
        if self.criterion is not None:
            loss = self.criterion(outputs, labels)
            if isinstance(loss, dict):
                loss = loss.get("core_loss", sum(loss.values()))
        else:
            loss = self._compute_simple_loss(outputs, labels)
        return loss

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        amp_enabled = self.sam3_cfg.get("amp_enabled", True)
        amp_dtype = getattr(torch, self.sam3_cfg.get("amp_dtype", "bfloat16"))
        using_sam3_native = hasattr(self, "_sam3_dict_key")

        train_pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch + 1}/{self.cfg.training.get("max_epochs", 20)}',
        )

        for batch in train_pbar:
            self.optimizer.zero_grad()

            # Forward pass with AMP
            with torch.amp.autocast(
                device_type="cuda", enabled=amp_enabled, dtype=amp_dtype
            ):
                if using_sam3_native:
                    loss, _, _, _ = self._sam3_forward(batch)
                else:
                    loss = self._legacy_tensor_forward(batch)
                    if loss is None:
                        continue

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            current_lr = self.optimizer.param_groups[0]["lr"]
            train_pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.6f}"}
            )

            self.global_step += 1

        return {
            "loss": total_loss / max(num_batches, 1),
        }

    def _compute_simple_loss(self, outputs, labels):
        """Compute simple loss when SAM3 criterion is not available."""
        if hasattr(outputs, "masks") or (
            isinstance(outputs, dict) and "masks" in outputs
        ):
            pred_masks = (
                outputs["masks"] if isinstance(outputs, dict) else outputs.masks
            )
        else:
            pred_masks = outputs

        # Handle different label formats
        if isinstance(labels, torch.Tensor):
            target = labels.float()
        else:
            target = labels

        bce = self.bce_loss(pred_masks, target)
        dice = self.dice_loss(pred_masks, target)

        return 0.2 * bce + 0.8 * dice

    def _visualize_validation(self, epoch: int, predictions_cache=None):
        """No-op: SAM3 dataloaders yield dict-structured batches
        ({dict_key: BatchedDatapoint}) that the generic tuple-batch
        visualization path (BaseTrainer._run_vis_inference does ``batch[0]``)
        cannot consume — it raises ``KeyError: 0``. SAM3 validate() also never
        returns a predictions cache, so BaseTrainer.train() would re-run that
        incompatible inference on every 5th (visualization) epoch and crash.
        Qualitative SAM3 rendering lives in the native orchestrator trainer.
        """
        _ = (epoch, predictions_cache)
        return

    def validate(self, epoch: int, return_predictions: bool = False) -> Dict[str, float]:  # noqa: ARG002
        """Validate model."""
        _ = epoch  # Required by BaseTrainer interface but not used here
        _ = return_predictions  # not implemented for SAM3 yet; always returns just metrics
        self.model.eval()

        total_loss = 0.0
        num_batches = 0
        using_sam3_native = hasattr(self, "_sam3_dict_key")

        if using_sam3_native:
            # Sam3Image.forward_grounding only computes the matcher `indices`
            # the loss needs when `self.training` is truthy (or interactive
            # steps are configured) — see model/sam3_vendor/sam3/model/sam3_image.py.
            # Flip *only* the top-level module's flag (not the recursive
            # .train() call) so children stay in eval() (dropout off, etc.)
            # while satisfying that gate.
            self.model.training = True

        # Use evaluator if available (legacy tensor path only — the SAM-style
        # evaluator expects a segmentation-model interface incompatible with
        # SAM3's find-query BatchedDatapoint forward).
        if not using_sam3_native and hasattr(self, "evaluator") and self.evaluator is not None:
            try:
                val_metrics = self.evaluator.evaluate_model_sam(
                    self.model,
                    self.val_loader,
                    self.device,
                    self.cfg.data.get("num_classes", 2),
                    img_size=self.resolution,
                )
                return val_metrics
            except Exception as e:
                self.logger.warning(f"SAM-style evaluation failed: {e}")

        amp_enabled = self.sam3_cfg.get("amp_enabled", True)
        amp_dtype = getattr(torch, self.sam3_cfg.get("amp_dtype", "bfloat16"))

        from utils.evaluate import Evaluator_seg

        num_classes = self.cfg.data.get("num_classes", 1)
        batch_metrics = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Same autocast context as train_epoch — under plain
                # torch.no_grad() (no autocast), the fused inference-only
                # addmm_act kernel (see _patch_fused_mlp_for_training) returns
                # bf16 while the rest of the model runs fp32, causing a dtype
                # mismatch in the very next nn.Linear.
                with torch.amp.autocast(
                    device_type="cuda", enabled=amp_enabled, dtype=amp_dtype
                ):
                    if using_sam3_native:
                        loss, _, outputs, find_targets = self._sam3_forward(batch)
                        pred_logits, gt_mask = self._extract_pred_gt_masks(outputs, find_targets)
                    else:
                        loss = self._legacy_tensor_forward(batch)
                        if loss is None:
                            continue

                total_loss += loss.item()
                num_batches += 1
                if using_sam3_native:
                    batch_metrics.append(
                        Evaluator_seg.evaluate_batch(pred_logits, gt_mask, num_classes=num_classes)
                    )

        metrics = {"loss": total_loss / max(num_batches, 1)}
        if using_sam3_native and batch_metrics:
            # Same aggregated keys every other trainer reports: Dice/IoU/HD95/
            # Sensitivity/Specificity/PixelAcc/BFScore(+_std)/ECE.
            metrics.update(Evaluator_seg.aggregate_metrics(batch_metrics, num_classes=num_classes))
        else:
            metrics["Dice"] = 0.0
        return metrics

    def test(self) -> Dict[str, float]:
        """Test model."""
        self.model.eval()

        # Similar to validate. NOTE: this scores against val_loader, not a
        # held-out test set — _create_sam3_coco_dataloaders doesn't build a
        # separate test_loader yet (matches the framework-level fallback
        # elsewhere; not something today's metrics work touched).
        return self.validate(self.current_epoch)

    def _save_model(self, path: Path):
        """Save model checkpoint."""
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
        }

        if self.scaler is not None:
            checkpoint["scaler"] = self.scaler.state_dict()

        torch.save(checkpoint, str(path))

    def _load_checkpoint(self, path: Path):
        """Load model checkpoint (any project schema)."""
        self.logger.info(f"Loading checkpoint: {path}")
        load_model_weights(self.model, path, map_location=self.device, strict=True)


def run_sam3_orchestrator(cfg: DictConfig):
    """
    Convenience function to run SAM3 training via orchestrator.

    Args:
        cfg: Unified framework configuration
    """
    orchestrator = SAM3Orchestrator(cfg)
    orchestrator.run()


if __name__ == "__main__":
    """
    Direct execution for advanced SAM3 training.

    Usage:
        python -m trainers.sam3_adapter --config path/to/config.yaml
    """
    import argparse

    parser = argparse.ArgumentParser(description="SAM3 Training Orchestrator")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "val", "test"]
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.mode = args.mode

    orchestrator = SAM3Orchestrator(cfg)
    orchestrator.run()
