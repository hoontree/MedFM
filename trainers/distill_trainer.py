import os
import logging as _logging
from collections import defaultdict
from hydra.utils import instantiate
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional, Tuple, override

from utils.data_processing import SegDatasetProcessor
from utils.logger import setup_logger
from utils.schedule import build_scheduler
from distillers import create_distiller
from utils.distill_utils import (
    get_dataset_short_name,
    get_experiment_tags,
    save_experiment_summary,
    load_model_cfg,
)
from utils.visualize import visualize_segmentation
from utils.utils import set_seed
from utils.wandb_utils import resolve_wandb_identity, build_experiment_dir
from trainers.base_trainer import BaseTrainer
from omegaconf import OmegaConf, DictConfig


class DistillTrainer(BaseTrainer):
    """
    Trainer for Knowledge Distillation.
    Inherits common infrastructure from BaseTrainer and overrides distillation-specific logic.
    """
    
    # Construction
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)  # initializes cfg, device, evaluator, base attrs

        # Distill-specific best metric tracking (best_metric/best_iou/best_hd95 from base)
        self.best_dice = 0.0   # alias used in distill logging (best_metric tracks the same)
        self.best_biou = 0.0
        self.best_iou_path = None
        self.best_biou_path = None
        self.best_hd95_path = None
        self.final_metrics: Dict[str, float] = {}


        set_seed(cfg.hardware.seed)

        # Resume bookkeeping (set before directory setup so a resumed run can
        # reuse its previous exp_dir instead of spawning a fresh timestamp).
        self.start_epoch = 0
        self._resume_ckpt_path = None

        # Resolve teacher/student model configs from config/model/{name}.yaml.
        # The selected binary/multiclass checkpoint is folded into `checkpoint`.
        self._resolve_model_cfgs()

        # Locate a resume checkpoint (cfg.resume = "auto" | <path> | null) before
        # creating directories — "auto" reuses the latest prior run dir whose
        # signature (run_name + hparam tags) matches and that holds a last.pth.
        self._resume_ckpt_path = self._resolve_resume_path()

        # Resolve run identity once: the W&B name/group/tags and the on-disk
        # directory name are derived from this single call (one timestamp, one
        # hparam-tag computation) so they never drift apart.
        self._identity = resolve_wandb_identity(cfg, default_job_type="distill")

        self._setup_directories()
        save_experiment_summary(self.cfg, self.exp_dir)
        self._setup_logger()

        self.dataset_name = get_dataset_short_name(cfg)

        self.logger.info(f"Starting Distillation: {cfg.method.name}")
        self.logger.info(
            f"Teacher: {self.teacher_name} -> Student: {self.student_name}"
        )
        self.logger.info(f"Dataset: {self.dataset_name}")
        self.logger.info(f"Experiment directory: {self.exp_dir}")

        self._setup_wandb()
        self._create_dataloaders()
        self._create_model()
        self._create_optimizer()
        self._create_scheduler()
        self._setup_early_stopping()  # sets self.early_stopping from base

        # Restore full training state (weights + optimiser + scheduler + best
        # trackers + early-stopping + epoch/step) when resuming.
        if self._resume_ckpt_path is not None:
            self._load_resume_state(self._resume_ckpt_path)

    def _resolve_model_cfgs(self):
        """Load teacher/student model configs from config/model/{name}.yaml.

        Reads ``cfg.teacher`` / ``cfg.student`` (each a string model name),
        loads the corresponding yaml under ``config/model/``, picks the
        binary or multiclass checkpoint based on ``data.num_classes``, and
        exposes the result as ``cfg.teacher_cfg`` / ``cfg.student_cfg`` plus
        the convenience attrs ``self.teacher_name`` / ``self.student_name``.
        """
        num_classes = self.cfg.data.num_classes
        self.teacher_name = self.cfg.teacher
        self.student_name = self.cfg.student
        if not isinstance(self.teacher_name, str) or not isinstance(self.student_name, str):
            raise ValueError(
                "cfg.teacher and cfg.student must be model names (strings) "
                "referring to config/model/{name}.yaml."
            )

        teacher_cfg = load_model_cfg(self.teacher_name, num_classes)
        student_cfg = load_model_cfg(self.student_name, num_classes)

        # Explicit checkpoint override (e.g. picking a specific trained
        # artifact for the teacher without a dedicated model yaml).
        if (ckpt := self.cfg.get("teacher_checkpoint")) is not None:
            teacher_cfg.checkpoint = ckpt

        # Student should not inherit the teacher's fine-tuned checkpoint by
        # default — students typically train from a pretrained backbone or
        # from scratch. The user can re-enable via student_cfg.checkpoint=...
        if self.cfg.get("use_student_finetuned_ckpt", False) is False:
            student_cfg.checkpoint = None
        if (ckpt := self.cfg.get("student_checkpoint")) is not None:
            student_cfg.checkpoint = ckpt

        OmegaConf.set_struct(self.cfg, False)
        self.cfg.teacher_cfg = teacher_cfg
        self.cfg.student_cfg = student_cfg

        self.teacher_ckpt = teacher_cfg.get("checkpoint")

    # Setup (directories / logging / wandb)
    @override
    def _setup_directories(self):
        """Setup experiment directories.

        Structure: ``logs/distill/{teacher}_to_{student}/[{group}/]{run_name}``,
        built by the shared ``utils.wandb_utils.build_experiment_dir`` from
        ``self._identity`` so the folder name matches the W&B run name exactly
        (same timestamp, same hparam tags — resolved once in ``__init__``).

        When resuming, reuse the previous run's directory (the parent of the
        located ``checkpoints/last.pth``) so logs/checkpoints continue in place
        instead of spawning a fresh timestamped dir.
        """
        if self._resume_ckpt_path is not None:
            self.exp_dir = self._resume_ckpt_path.parent.parent
        else:
            root_segment = f"{self.teacher_name}_to_{self.student_name}"
            self.exp_dir = build_experiment_dir(
                self.cfg, root_segment=root_segment, identity=self._identity
            )
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir = self.exp_dir / "visualizations"
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = self.exp_dir

        config_file = self.exp_dir / "config.yaml"
        with open(config_file, "w") as f:
            OmegaConf.save(self.cfg, f)

    @override
    def _setup_logger(self):
        """Override: use distill.log and medfm.distill logger name."""
        log_file = self.exp_dir / "distill.log"
        self.logger = setup_logger(str(log_file), logger_name="medfm.distill")

    @override
    def _setup_wandb(self):
        """Override: initialize a single wandb run for the entire distillation process.

        Run identity (project/group/name/tags) comes from ``self._identity``
        (resolved once in ``__init__``) so distillation and supervised runs
        use one scheme and the run name matches the exp_dir already created by
        ``_setup_directories``; group defaults to ``distill/{method}/{datasets}``
        so related runs cluster.
        """
        identity = {k: v for k, v in self._identity.items() if k != "dir_name"}
        if self.cfg.get("debug", False):
            identity["mode"] = "disabled"

        # Persist a stable run id in the exp_dir so a resumed run (which reuses
        # the same exp_dir) re-attaches to the *same* W&B run and continues its
        # history instead of starting a fresh one.
        run_id_file = self.exp_dir / "wandb_run_id"
        resuming = self._resume_ckpt_path is not None and run_id_file.exists()
        if resuming:
            run_id = run_id_file.read_text().strip()
        else:
            run_id = wandb.util.generate_id()
            run_id_file.write_text(run_id)

        self.wandb_run = wandb.init(
            config=OmegaConf.to_container(self.cfg, resolve=True),
            id=run_id,
            resume="allow" if resuming else "never",
            **identity,
        )
        self._define_wandb_metrics()

    def _wandb_log(self, data: dict) -> None:
        """Log metrics to wandb"""
        if self.wandb_run is None:
            return
        data = self._round_log_values(data)
        self.wandb_run.log(data)
        self.logger.debug(f"WandB log: {data}")

    def _wandb_summary_update(self, data: dict) -> None:
        """Update wandb summary metrics."""
        if self.wandb_run is None:
            return
        self.wandb_run.summary.update(self._round_log_values(data))

    # Model / optimizer construction
    @override
    def _create_dataloaders(self):
        self.train_loader, self.val_loader, self.test_loader = (
            SegDatasetProcessor.build_data_loaders(self.cfg)
        )

    @property
    def _is_online(self) -> bool:
        """Return True when online distillation is active.

        Triggered automatically when method.name == 'online'
        """
        return self.cfg.get("method", {}).get("name") == "online"

    @override
    def _create_model(self):
        """Create teacher, student, and distiller."""
        if self.teacher_ckpt:
            self.logger.info(f"Using teacher checkpoint: {self.teacher_ckpt}")

        self.teacher = instantiate(self.cfg.teacher_cfg)
        self.teacher = self.teacher.to(self.device)

        if self._is_online:
            self.teacher.train()
            self.logger.info(
                "[Online Distillation] Teacher is trainable (jointly updated with student)."
            )
        else:
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False

        OmegaConf.set_struct(self.cfg.method, False)

        self.student = instantiate(self.cfg.student_cfg)
        self.student = self.student.to(self.device)
        self.model = self.student  # base-class compatibility

        self.distiller = create_distiller(self.cfg).to(self.device)
        self.distiller.prepare(self.student, self.teacher)

        self._log_model_info()

    @override
    def _create_optimizer(self):
        param_groups = [
            {"params": self.student.parameters(), "lr": self.cfg.training.lr}
        ]
        if self._is_online:
            teacher_trainable = [
                p for p in self.teacher.parameters() if p.requires_grad
            ]
            if teacher_trainable:
                teacher_lr = self.cfg.training.get(
                    "teacher_lr", self.cfg.training.lr
                )
                param_groups.append(
                    {"params": teacher_trainable, "lr": teacher_lr}
                )
                self.logger.info(
                    f"[Online Distillation] Teacher trainable params: "
                    f"{sum(p.numel() for p in teacher_trainable):,}  lr={teacher_lr}"
                )
        if list(self.distiller.parameters()):
            param_groups.append(
                {"params": self.distiller.parameters(), "lr": self.cfg.training.lr}
            )
        self.optimizer = optim.AdamW(
            param_groups, weight_decay=self.cfg.optimizer.weight_decay
        )

    @override
    def _create_scheduler(self):
        self.scheduler = build_scheduler(self.optimizer, self.cfg)

    @override
    def _log_model_info(self):
        """Override: log teacher + student parameter counts separately."""
        t_total = sum(p.numel() for p in self.teacher.parameters())
        s_total = sum(p.numel() for p in self.student.parameters())
        s_trainable = sum(
            p.numel() for p in self.student.parameters() if p.requires_grad
        )
        self.logger.info(f"Teacher total parameters: {t_total:,}")
        self.logger.info(f"Student total parameters: {s_total:,}")
        self.logger.info(f"Student trainable parameters: {s_trainable:,}")
        self._wandb_summary_update(
            {
                "model/teacher_total_params": t_total,
                "model/student_total_params": s_total,
                "model/student_trainable_params": s_trainable,
            }
        )

    # Forward helpers (teacher / student calls)
    @staticmethod
    def _is_sam_model(model) -> bool:
        """Return True if *model* is a SAM-based model (LoRA_Sam)."""
        try:
            from model.sam_hybrid_adapter import LoRA_Sam
            return isinstance(model, LoRA_Sam)
        except ImportError:
            return False

    def _call_teacher(self, images):
        """Call teacher model with the appropriate forward signature."""
        if self._is_sam_model(self.teacher):
            img_size = self.cfg.teacher_cfg.get("img_size", self.cfg.data.img_size)
            multimask = self.cfg.data.num_classes > 1
            return self.teacher(images, multimask, img_size)
        else:
            raw = self.teacher(images)
            if isinstance(raw, dict):
                return raw
            return {"masks": raw}

    def _call_student(self, images):
        """Call student model and normalise its output to a dict."""
        if self._is_sam_model(self.student):
            img_size = self.cfg.student_cfg.get("img_size", self.cfg.data.img_size)
            multimask = self.cfg.data.num_classes > 1
            raw = self.student(images, multimask, img_size)
            if isinstance(raw, dict):
                return raw
            return {"masks": raw}
        else:
            raw = self.student(images, return_features=True)
            if isinstance(raw, (list, tuple)):
                return {"masks": raw[0], "features": raw[1]}
            return {"masks": raw}

    # Training step
    @override
    def train_epoch(self, epoch) -> Dict[str, float]:
        self.student.train()
        self.distiller.train()
        if self._is_online:
            self.teacher.train()

        running_losses = {}
        pbar = tqdm(
            self.train_loader, desc=f"Epoch {epoch + 1}/{self.cfg.training.num_epochs}"
        )

        for i, (images, masks, *_) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.distiller.on_step_begin()

            if self._is_online:
                teacher_outputs = self._call_teacher(images)
            else:
                with torch.no_grad():
                    teacher_outputs = self._call_teacher(images)

            student_outputs = self._call_student(images)

            loss_dict = self.distiller(student_outputs, teacher_outputs, masks)
            loss = loss_dict["loss"]

            self.optimizer.zero_grad()
            loss.backward()

            max_norm = self.cfg.optimizer.gradient_clip.get("max_norm", 1.0)
            all_params = [p for pg in self.optimizer.param_groups for p in pg["params"]]
            nn.utils.clip_grad_norm_(all_params, max_norm=max_norm)

            self.optimizer.step()

            for k, v in loss_dict.items():
                val = v.item() if hasattr(v, "item") else v
                running_losses[k] = running_losses.get(k, 0.0) + val

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if self.global_step % 10 == 0:
                step_log = {"global_step": self.global_step}
                for k, v in loss_dict.items():
                    val = v.item() if hasattr(v, "item") else v
                    step_log[f"step/{k}"] = val
                step_log["step/lr"] = self.optimizer.param_groups[0]["lr"]
                self._wandb_log(step_log)
            self.global_step += 1

        return {k: v / (i + 1) for k, v in running_losses.items()}

    # Evaluation
    @property
    def _first_test_loader(self):
        """Get the first test loader (handles both dict and single loader)."""
        if isinstance(self.test_loader, dict):
            return next(iter(self.test_loader.values()))
        return self.test_loader

    def _evaluate_model(self, model, loader, return_predictions=False):
        """Call the appropriate evaluate_model variant depending on model type."""
        num_classes = self.cfg.data.num_classes
        if self._is_sam_model(model):
            img_size = self.cfg.data.img_size
            return self.evaluator.evaluate_model_sam(
                model, loader, self.device, num_classes,
                img_size=img_size, return_predictions=return_predictions,
            )
        return self.evaluator.evaluate_model(
            model, loader, self.device, num_classes,
            return_predictions=return_predictions,
        )

    @override
    def validate(self, epoch, return_predictions: bool = False):
        self.student.eval()
        if self._is_online:
            self.teacher.eval()
        result = self._evaluate_model(
            self.student, self.val_loader, return_predictions=return_predictions
        )
        if return_predictions:
            val_metrics, images_l, preds_l, masks_l, fnames_l, _ = result
        else:
            val_metrics = result
        self.evaluator.print_metrics(val_metrics, phase="validation")
        if return_predictions:
            return val_metrics, {"__val__": (images_l, preds_l, masks_l, fnames_l)}
        return val_metrics

    @staticmethod
    def _group_summary(dicts: list) -> Dict[str, float]:
        """Aggregate per-dataset metric dicts into a group mean + overall std.

        Datasets are weighted equally. For a base metric `X`, the group mean is
        the mean of per-dataset means; the overall std combines within-dataset
        variance (`X_std`) and between-dataset variance via the law of total
        variance: std = sqrt(mean(std_i^2) + var(mean_i)).
        """
        import numpy as np

        base_metrics = [k for k in dicts[0] if not k.endswith("_std")]
        out: Dict[str, float] = {}
        for k in base_metrics:
            means = np.array([d[k] for d in dicts if k in d], dtype=float)
            if means.size == 0:
                continue
            out[k] = float(means.mean())
            std_key = f"{k}_std"
            within = np.array(
                [d[std_key] for d in dicts if std_key in d], dtype=float
            )
            if within.size == means.size:
                total_var = float((within ** 2).mean() + means.var())
                out[std_key] = float(np.sqrt(total_var))
        return out

    @override
    def test(self, phase="test") -> Dict[str, float]:
        self.student.eval()
        all_metrics = {}
        predictions_cache = {}
        is_multi = isinstance(self.test_loader, dict)

        # Collect per-dataset metric dicts grouped by internal (held-out *_test
        # splits) vs external validation sets, so we can report a per-group mean
        # and an overall std across the group.
        group_metrics = {"internal": [], "external": []}

        for ds_name, loader in self._iter_test_loaders():
            result = self._evaluate_model(self.student, loader, return_predictions=True)
            metrics, images_list, preds_list, masks_list, fnames_list, per_sample = result

            if is_multi:
                self.logger.info(f"--- {phase} ({ds_name}) ---")
                self.evaluator.print_metrics(metrics, phase=f"{phase}_{ds_name}")
                numeric = self._numeric_items(metrics)
                for k, v in numeric.items():
                    all_metrics[f"{ds_name}/{k}"] = v
                group = "internal" if ds_name.endswith("_test") else "external"
                group_metrics[group].append(numeric)
            else:
                self.logger.info(f"--- {phase} ---")
                self.evaluator.print_metrics(metrics, phase=phase)
                all_metrics.update(self._numeric_items(metrics))

            predictions_cache[ds_name] = (images_list, preds_list, masks_list, fnames_list, per_sample)

        if is_multi:
            for group, dicts in group_metrics.items():
                if not dicts:
                    continue
                summary = self._group_summary(dicts)
                self.logger.info(f"--- {phase} ({group}_mean) ---")
                self.evaluator.print_metrics(summary, phase=f"{phase}_{group}_mean")
                for k, v in summary.items():
                    all_metrics[f"{group}_mean/{k}"] = v

        if phase == "final_test":
            return all_metrics, predictions_cache

        return all_metrics

    # Metrics logging & checkpointing

    @override
    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict,
        test_metrics: Dict = None,
    ):
        num_epochs = self.cfg.training.num_epochs
        self.logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        self.logger.info(
            "Train:\n    " + ", ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items())
        )
        val_items = self._numeric_items(val_metrics)

        log_data = {
            "epoch": epoch + 1,
            "train/lr": self.optimizer.param_groups[0]["lr"],
        }
        for k, v in train_metrics.items():
            log_data[f"train/{k}"] = v
        for k, v in val_items.items():
            log_data[f"val/{k}"] = v
        self._wandb_log(log_data)

    @override
    def _save_checkpoint(self, epoch: int, val_metrics: dict):
        """Override: save best checkpoints for the configured metrics.

        ``cfg.checkpoint.save_best_metrics`` (default all of
        ``[Dice, IoU, BIoU, HD95]``) controls which best-of checkpoints are
        kept; e.g. set it to ``[Dice]`` to keep only the selection-metric
        checkpoint and cut per-run disk ~4×. Best-Dice is always saved (it is
        the model used for final evaluation) regardless of the setting.
        """
        save_metrics = set(
            self.cfg.get("checkpoint", {}).get(
                "save_best_metrics", ["Dice", "IoU", "BIoU", "HD95"]
            )
        )
        dice = val_metrics.get("Dice", 0.0)
        iou = val_metrics.get("IoU", 0.0)
        biou = val_metrics.get("BIoU", 0.0)
        hd95 = val_metrics.get("HD95", float("inf"))

        # --- Best Dice (primary – used for final evaluation) ---
        if dice > self.best_dice:
            if self.best_model_path and self.best_model_path.exists():
                try:
                    self.best_model_path.unlink()
                except Exception as e:
                    self.logger.warning(f"Could not delete old best-Dice model: {e}")
            self.best_dice = dice
            self.best_model_path = (
                self.ckpt_dir / f"best_epoch_{epoch+1}_dice{dice:.4f}.pth"
            )
            self._save_distill_model(self.best_model_path, epoch, val_metrics)
            self.logger.info(f"Saved best-Dice model: {self.best_model_path}")
            self._wandb_summary_update(
                {"best_dice": dice, "checkpoint_path": str(self.best_model_path)}
            )

        # --- Best IoU ---
        if "IoU" in save_metrics and iou > self.best_iou:
            if self.best_iou_path and self.best_iou_path.exists():
                try:
                    self.best_iou_path.unlink()
                except Exception as e:
                    self.logger.warning(f"Could not delete old best-IoU model: {e}")
            self.best_iou = iou
            self.best_iou_path = (
                self.ckpt_dir / f"best_epoch_{epoch+1}_iou{iou:.4f}.pth"
            )
            self._save_distill_model(self.best_iou_path, epoch, val_metrics)
            self.logger.info(f"Saved best-IoU model: {self.best_iou_path}")
            self._wandb_summary_update(
                {"best_iou": iou, "best_iou_checkpoint": str(self.best_iou_path)}
            )

        # --- Best BIoU ---
        if "BIoU" in save_metrics and biou > self.best_biou:
            if self.best_biou_path and self.best_biou_path.exists():
                try:
                    self.best_biou_path.unlink()
                except Exception as e:
                    self.logger.warning(f"Could not delete old best-BIoU model: {e}")
            self.best_biou = biou
            self.best_biou_path = (
                self.ckpt_dir / f"best_epoch_{epoch+1}_biou{biou:.4f}.pth"
            )
            self._save_distill_model(self.best_biou_path, epoch, val_metrics)
            self.logger.info(f"Saved best-BIoU model: {self.best_biou_path}")
            self._wandb_summary_update(
                {"best_biou": biou, "best_biou_checkpoint": str(self.best_biou_path)}
            )

        # --- Best HD95 (lower is better) ---
        if "HD95" in save_metrics and hd95 < self.best_hd95:
            if self.best_hd95_path and self.best_hd95_path.exists():
                try:
                    self.best_hd95_path.unlink()
                except Exception as e:
                    self.logger.warning(f"Could not delete old best-HD95 model: {e}")
            self.best_hd95 = hd95
            self.best_hd95_path = (
                self.ckpt_dir / f"best_epoch_{epoch+1}_hd95{hd95:.4f}.pth"
            )
            self._save_distill_model(self.best_hd95_path, epoch, val_metrics)
            self.logger.info(f"Saved best-HD95 model: {self.best_hd95_path}")
            self._wandb_summary_update(
                {"best_hd95": hd95, "best_hd95_checkpoint": str(self.best_hd95_path)}
            )

    def _save_distill_model(self, path: Path, epoch: int, metrics: dict):
        """Save student + distiller (+ teacher if online) state dict to path."""
        payload = {
            "epoch": epoch + 1,
            "model_state_dict": self.student.state_dict(),
            "distiller_state_dict": self.distiller.state_dict(),
            **{k: v for k, v in metrics.items() if isinstance(v, (float, int))},
        }
        if self._is_online:
            payload["teacher_state_dict"] = self.teacher.state_dict()
        torch.save(payload, path)

    # Resume (full training-state checkpointing)
    def _resolve_resume_path(self) -> Optional[Path]:
        """Resolve the resume checkpoint from ``cfg.resume``.

        ``cfg.resume`` may be:
            * ``null`` / unset / ``false`` → no resume (default).
            * ``"auto"`` → reuse the most recent prior run dir under
              ``logs/distill/<teacher>_to_<student>/[<group>/]`` whose name ends
              with this run's signature (``_<run_name>`` + hparam tags) and
              contains ``checkpoints/last.pth``. The optional ``<group>``
              segment mirrors ``build_experiment_dir`` — a re-launched sweep
              passes the same explicit ``wandb.group`` it used originally, so
              the search stays scoped to the same directory it wrote to.
            * an explicit path to a ``last.pth`` (or a run dir holding one).

        Returns the checkpoint path, or ``None`` when nothing resumable is found.
        """
        resume = self.cfg.get("resume", None)
        if resume in (None, False, "", "false", "none", "null"):
            return None

        if resume not in ("auto", True, "true"):
            p = Path(str(resume))
            if p.is_dir():
                p = p / "checkpoints" / "last.pth"
            return p if p.exists() else None

        base = (
            Path(self.cfg.output.dir)
            / "distill"
            / f"{self.teacher_name}_to_{self.student_name}"
        )
        explicit_group = self.cfg.get("wandb", {}).get("group")
        if explicit_group:
            base = base / str(explicit_group)
        if not base.exists():
            return None
        label = f"_{self.cfg.get('run_name')}" if self.cfg.get("run_name") else ""
        tags = get_experiment_tags(self.cfg)
        signature = f"{label}{'_' + '_'.join(tags) if tags else ''}"
        cands = [
            d for d in base.iterdir()
            if d.is_dir() and d.name.endswith(signature)
            and (d / "checkpoints" / "last.pth").exists()
        ]
        if not cands:
            return None
        latest = max(
            cands, key=lambda d: (d / "checkpoints" / "last.pth").stat().st_mtime
        )
        return latest / "checkpoints" / "last.pth"

    def _resume_payload_extra(self) -> dict:
        """Hook: extra state to persist for resume (subclasses extend)."""
        return {}

    def _load_resume_extra(self, payload: dict) -> None:
        """Hook: restore extra state saved by ``_resume_payload_extra``."""
        return None

    def _save_resume_checkpoint(self, epoch: int) -> None:
        """Persist the full training state to ``checkpoints/last.pth`` (atomic).

        Written every epoch so an interrupted run can continue from ``epoch+1``
        with optimiser/scheduler/best-tracker/early-stopping state intact.
        """
        es = self.early_stopping
        payload = {
            "epoch": epoch,  # last completed epoch; resume starts at epoch+1
            "global_step": self.global_step,
            "model_state_dict": self.student.state_dict(),
            "distiller_state_dict": self.distiller.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "best_trackers": {
                "best_dice": self.best_dice,
                "best_metric": self.best_metric,
                "best_iou": self.best_iou,
                "best_biou": self.best_biou,
                "best_hd95": self.best_hd95,
                "best_model_path": str(self.best_model_path) if self.best_model_path else None,
                "best_iou_path": str(self.best_iou_path) if self.best_iou_path else None,
                "best_biou_path": str(self.best_biou_path) if self.best_biou_path else None,
                "best_hd95_path": str(self.best_hd95_path) if self.best_hd95_path else None,
            },
            "early_stopping": (
                {"counter": es.counter, "best_score": es.best_score, "early_stop": es.early_stop}
                if es is not None else None
            ),
            **self._resume_payload_extra(),
        }
        if self._is_online:
            payload["teacher_state_dict"] = self.teacher.state_dict()

        tmp = self.ckpt_dir / "last.pth.tmp"
        torch.save(payload, tmp)
        tmp.replace(self.ckpt_dir / "last.pth")

    def _load_resume_state(self, path: Path) -> None:
        """Restore full training state saved by ``_save_resume_checkpoint``."""
        self.logger.info(f"Resuming from {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        self.student.load_state_dict(ckpt["model_state_dict"])
        self.student.to(self.device)
        if "distiller_state_dict" in ckpt:
            self.distiller.load_state_dict(ckpt["distiller_state_dict"], strict=False)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if self._is_online and "teacher_state_dict" in ckpt:
            self.teacher.load_state_dict(ckpt["teacher_state_dict"])

        bt = ckpt.get("best_trackers", {})
        self.best_dice = bt.get("best_dice", self.best_dice)
        self.best_metric = bt.get("best_metric", self.best_metric)
        self.best_iou = bt.get("best_iou", self.best_iou)
        self.best_biou = bt.get("best_biou", self.best_biou)
        self.best_hd95 = bt.get("best_hd95", self.best_hd95)
        self.best_model_path = Path(bt["best_model_path"]) if bt.get("best_model_path") else None
        self.best_iou_path = Path(bt["best_iou_path"]) if bt.get("best_iou_path") else None
        self.best_biou_path = Path(bt["best_biou_path"]) if bt.get("best_biou_path") else None
        self.best_hd95_path = Path(bt["best_hd95_path"]) if bt.get("best_hd95_path") else None

        es_state = ckpt.get("early_stopping")
        if es_state is not None and self.early_stopping is not None:
            self.early_stopping.counter = es_state["counter"]
            self.early_stopping.best_score = es_state["best_score"]
            self.early_stopping.early_stop = es_state["early_stop"]

        self._load_resume_extra(ckpt)

        self.global_step = ckpt.get("global_step", 0)
        self.start_epoch = int(ckpt.get("epoch", -1)) + 1
        self.logger.info(
            f"Resumed at epoch {self.start_epoch} (global_step={self.global_step}, "
            f"best_dice={self.best_dice:.4f})"
        )

    # Visualization helpers
    def _run_distill_vis_inference(self, loader, num_samples: Optional[int] = None):
        """Run teacher+student forward passes and collect tensors for visualization.

        Returns ``(images_list, preds_list, masks_list, filenames_list)`` where
        each ``preds_list`` entry is ``{"teacher": ..., "student": ...}``.
        """
        from tqdm import tqdm

        self.teacher.eval()
        self.student.eval()
        num_classes = self.cfg.data.num_classes

        images_list, preds_list, masks_list, fnames_list = [], [], [], []
        collected = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc="Distill visualization inference"):
                images = batch[0].to(self.device)
                masks = batch[1].to(self.device)
                last = batch[-1]
                fnames = (
                    list(last)
                    if isinstance(last, (list, tuple)) and last and isinstance(last[0], str)
                    else None
                )

                t_out = self._call_teacher(images)
                s_out = self._call_student(images)
                t_logits = t_out["masks"] if isinstance(t_out, dict) else t_out
                s_logits = s_out["masks"] if isinstance(s_out, dict) else s_out

                if num_classes == 1:
                    t_preds = (torch.sigmoid(t_logits) > 0.5).float()
                    s_preds = (torch.sigmoid(s_logits) > 0.5).float()
                else:
                    t_preds = torch.argmax(t_logits, dim=1, keepdim=True).float()
                    s_preds = torch.argmax(s_logits, dim=1, keepdim=True).float()

                images_list.append(images.cpu())
                preds_list.append({"teacher": t_preds.cpu(), "student": s_preds.cpu()})
                masks_list.append(masks.cpu())
                fnames_list.append(fnames)

                collected += images.size(0)
                if num_samples is not None and collected >= num_samples:
                    break

        if not any(fnames_list):
            fnames_list = None
        return images_list, preds_list, masks_list, fnames_list

    def _save_per_sample_visualizations(self, predictions_cache: dict) -> None:
        """Save per-sample visualizations for every test dataset.

        Test-only path: this is invoked from ``run_test_only`` so that per-sample
        figures are not regenerated during the post-training final_test pass.
        """
        for ds_name, cached in predictions_cache.items():
            images_list, preds_list, masks_list, fnames_list = (
                cached[0], cached[1], cached[2], cached[3],
            )
            vis_dir = self.vis_dir / "per_sample" / ds_name
            visualize_segmentation(
                images_list,
                [{"pred": p} for p in preds_list],
                masks_list,
                num_classes=self.cfg.data.num_classes,
                save_dir=vis_dir,
                filenames_list=fnames_list,
            )

    # Entry points (main loops)

    @override
    def train(self):
        """Main distillation training loop."""
        if self.start_epoch >= self.cfg.training.num_epochs:
            self.logger.info(
                f"Resumed run already reached num_epochs "
                f"({self.start_epoch}/{self.cfg.training.num_epochs}); skipping to final eval."
            )
        for epoch in range(self.start_epoch, self.cfg.training.num_epochs):
            self.current_epoch = epoch
            train_losses = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            self._log_metrics(epoch, train_losses, val_metrics)

            if (epoch + 1) % 5 == 0:
                vis_loader = self._first_test_loader
                self.logger.info(
                    f"Generating validation visualizations for epoch {epoch + 1}..."
                )
                images_l, preds_l, masks_l, fnames_l = self._run_distill_vis_inference(
                    vis_loader, num_samples=self.cfg.visualization.num_samples
                )
                visualize_segmentation(
                    images_l, preds_l, masks_l,
                    num_classes=self.cfg.data.num_classes,
                    save_dir=self.vis_dir,
                    num_samples=self.cfg.visualization.num_samples,
                    phase_name="distillation",
                    filenames_list=fnames_l,
                    log_to_wandb=True,
                    epoch=epoch,
                )

            self._save_checkpoint(epoch, val_metrics)

            if self.early_stopping is not None:
                self.early_stopping(val_metrics.get("Dice", val_metrics.get("dice", 0.0)))
                if self.early_stopping.should_stop():
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            self.scheduler.step()
            # Persist full training state so an interrupted run resumes here.
            self._save_resume_checkpoint(epoch)

        self._final_evaluation()
        wandb.finish()
        return {
            "best_model_path": (
                str(self.best_model_path) if self.best_model_path else None
            ),
            "exp_dir": str(self.exp_dir),
            "final_metrics": dict(self.final_metrics),
        }

    def _final_evaluation(self):
        if self.best_model_path and self.best_model_path.exists():
            self.logger.info("Clearing memory before final evaluation...")
            del self.optimizer
            del self.distiller
            torch.cuda.empty_cache()

            self.logger.info(f"Loading best model for final evaluation: {self.best_model_path}")
            checkpoint = torch.load(self.best_model_path, map_location="cpu", weights_only=False)
            self.student.load_state_dict(checkpoint["model_state_dict"])
            self.student.to(self.device)

            test_metrics, student_cache = self.test(phase="final_test")
            self.final_metrics = {f"final_test/{k}": v for k, v in test_metrics.items()}
            self._wandb_summary_update(self.final_metrics)
            # final_test/* is bound to step_metric="epoch" in _define_wandb_metrics.
            # Pass an explicit "epoch" so these points land on the run's final
            # epoch on the x-axis instead of W&B's last-seen internal step.
            self._wandb_log({**self.final_metrics, "epoch": self.cfg.training.num_epochs})
            self._save_latex_metrics_table_from_metrics(test_metrics)
            self._save_final_metrics_json(test_metrics)
            self._save_per_sample_metrics(student_cache)

            first_ds = next(iter(student_cache))
            student_images_list, student_preds_list, masks_list, fnames_list, _ = student_cache[first_ds]
            first_loader = (
                self._first_test_loader
                if not isinstance(self.test_loader, dict)
                else list(self.test_loader.values())[0]
            )
            teacher_result = self._evaluate_model(self.teacher, first_loader, return_predictions=True)
            _, teacher_images_list, teacher_preds_list, _, _, _ = teacher_result

            preds_dict_list = [
                {"teacher": t, "student": s}
                for t, s in zip(teacher_preds_list, student_preds_list)
            ]
            visualize_segmentation(
                student_images_list,
                preds_dict_list,
                masks_list,
                num_classes=self.cfg.data.num_classes,
                save_dir=self.vis_dir,
                num_samples=self.cfg.visualization.num_samples,
                phase_name="distillation",
                filenames_list=fnames_list,
                epoch=None,
            )
        else:
            self.logger.warning(
                "Best model checkpoint not found. Final metrics are unavailable."
            )

    @override
    def run_test_only(self, checkpoint_path: str):
        """Override: distill test-only also writes per-sample visualizations.

        Loads the student weights from ``checkpoint_path``, runs the multi-dataset
        test pass, persists per-sample CSVs (via base ``_save_test_results``), and
        additionally renders per-sample visualizations under
        ``vis_dir/per_sample/<dataset>``.
        """
        from datetime import datetime

        self.logger.info("TEST-ONLY MODE (distillation student)")
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        self.logger.info(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.student.load_state_dict(state_dict)
        self.student.to(self.device)
        self.student.eval()
        self.best_model_path = ckpt_path

        if wandb.run is None:
            identity = resolve_wandb_identity(self.cfg, default_job_type="distill")
            identity["tags"] = ["test-only", *identity["tags"]]
            identity.pop("dir_name", None)
            if self.cfg.get("debug", False):
                identity["mode"] = "disabled"
            self.wandb_run = wandb.init(
                config=OmegaConf.to_container(self.cfg, resolve=True),
                **identity,
            )

        test_metrics, predictions_cache = self.test(phase="final_test")
        self._save_test_results(test_metrics, predictions_cache=predictions_cache)
        self._save_per_sample_visualizations(predictions_cache)

        wandb.finish()
        self.logger.info("Test-only evaluation completed!")

    # Small utilities
    @staticmethod
    def _numeric_items(d: dict) -> dict:
        """Filter dict to only numeric (float/int) values."""
        return {k: v for k, v in d.items() if isinstance(v, (float, int))}

    @staticmethod
    def _make_limited_loader(loader, limit_batches, batch_size):
        """Create a small DataLoader subset for dry-run testing."""
        from torch.utils.data import Subset, DataLoader

        n = min(len(loader.dataset), limit_batches * batch_size)
        return DataLoader(
            Subset(loader.dataset, list(range(n))), batch_size=batch_size, num_workers=0
        )
