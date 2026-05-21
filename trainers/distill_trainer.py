import os
import logging as _logging
from hydra.utils import instantiate
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional, Tuple, override

from utils.data_processing_seg import SegDatasetProcessor
from utils.evaluate import Evaluator_seg
from utils.logger import setup_logger
from utils.schedule import build_scheduler
from distillers import create_distiller
from utils.distill_utils import (
    get_dataset_short_name,
    create_log_dir,
    save_experiment_summary,
    load_model_cfg,
)
from utils.visualize import visualize_segmentation
from utils.utils import set_seed
from trainers.base_trainer import BaseTrainer
from omegaconf import OmegaConf, DictConfig


class DistillTrainer(BaseTrainer):
    """
    Trainer for Knowledge Distillation.
    Inherits common infrastructure from BaseTrainer and overrides distillation-specific logic.
    """

    # =========================================================================
    # 1. Construction
    # =========================================================================

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)  # initializes cfg, device, evaluator, base attrs

        # Distill-specific best metric tracking (best_metric/best_iou/best_hd95 from base)
        self.best_dice = 0.0   # alias used in distill logging (best_metric tracks the same)
        self.best_biou = 0.0
        self.best_iou_path = None
        self.best_biou_path = None
        self.best_hd95_path = None
        self.final_metrics: Dict[str, float] = {}
        self._wandb_metric_prefix = ""

        set_seed(cfg.hardware.seed)

        # Resolve teacher/student model configs from config/model/{name}.yaml.
        # The selected binary/multiclass checkpoint is folded into `checkpoint`.
        self._resolve_model_cfgs()

        self._setup_directories()
        save_experiment_summary(self.cfg, self.exp_dir)
        self._setup_logger()
        self._setup_pipeline_log()

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

        # Pipeline mode: a freshly-trained teacher checkpoint can be injected
        # via pipeline.teacher_ckpt_override and takes precedence over the
        # binary/multiclass default.
        pipeline_cfg = self.cfg.get("pipeline", {})
        if (ckpt_override := pipeline_cfg.get("teacher_ckpt_override")):
            teacher_cfg.checkpoint = ckpt_override

        # Student should not inherit the teacher's fine-tuned checkpoint by
        # default — students typically train from a pretrained backbone or
        # from scratch. The user can re-enable via student_cfg.checkpoint=...
        if self.cfg.get("use_student_finetuned_ckpt", False) is False:
            student_cfg.checkpoint = None

        OmegaConf.set_struct(self.cfg, False)
        self.cfg.teacher_cfg = teacher_cfg
        self.cfg.student_cfg = student_cfg

        self.teacher_ckpt = teacher_cfg.get("checkpoint")

    # =========================================================================
    # 2. Setup (directories / logging / wandb)
    # =========================================================================

    @override
    def _setup_directories(self):
        """Setup experiment directories using the standardized create_log_dir structure."""
        self.exp_dir = create_log_dir(self.cfg)
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

    def _setup_pipeline_log(self):
        """Attach an extra file handler for pipeline logging if in pipeline mode."""
        pipeline_cfg = self.cfg.get("pipeline", {})
        teacher_log_dir = pipeline_cfg.get("teacher_log_dir", None)
        if teacher_log_dir:
            pipeline_log_path = Path(teacher_log_dir) / "pipeline.log"
            pipeline_handler = _logging.FileHandler(
                str(pipeline_log_path), mode="a", encoding="utf-8"
            )
            pipeline_handler.setLevel(_logging.INFO)
            fmt = "[%(asctime)s %(name)s] (%(filename)s:%(lineno)d): %(levelname)s %(message)s"
            pipeline_handler.setFormatter(_logging.Formatter(fmt))
            self.logger.addHandler(pipeline_handler)

    @override
    def _setup_wandb(self):
        """Override: support pipeline mode by reusing the teacher wandb run."""
        pipeline_cfg = self.cfg.get("pipeline", {})
        teacher_run_id = pipeline_cfg.get("teacher_run_id", None)
        is_sweep = os.environ.get("WANDB_SWEEP_ID") is not None
        wandb_mode = self.cfg.get("wandb", {}).get("mode", None)
        if self.cfg.get("debug", False) or self.cfg.get("wandb", {}).get("disabled", False):
            wandb_mode = "disabled"

        if teacher_run_id is not None and wandb.run is not None:
            # Pipeline: reuse open teacher run, prefix metrics with "distill/"
            self.wandb_run = wandb.run
            self._wandb_metric_prefix = "distill/"
            self.logger.info(
                f"[Pipeline] Reusing teacher WandB run '{self.wandb_run.id}'. "
                "Distillation metrics will be prefixed with 'distill/'"
            )
        else:
            exp_name = (
                None
                if is_sweep
                else f"{self.teacher_name}_{self.student_name}_{self.cfg.method.name}"
            )
            self.wandb_run = wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                name=exp_name,
                config=OmegaConf.to_container(self.cfg, resolve=True),
                mode=wandb_mode,
            )
            self._define_wandb_metrics()

    def _wandb_log(self, data: dict) -> None:
        """Log metrics to wandb, applying pipeline prefix when sharing a run."""
        if self.wandb_run is None:
            return
        if self._wandb_metric_prefix:
            data = {f"{self._wandb_metric_prefix}{k}": v for k, v in data.items()}
        self.wandb_run.log(data)

    def _wandb_summary_update(self, data: dict) -> None:
        """Update wandb summary, applying pipeline prefix when sharing a run."""
        if self.wandb_run is None:
            return
        if self._wandb_metric_prefix:
            data = {f"{self._wandb_metric_prefix}{k}": v for k, v in data.items()}
        self.wandb_run.summary.update(data)

    # =========================================================================
    # 3. Model / optimizer construction
    # =========================================================================

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

    # =========================================================================
    # 4. Forward helpers (teacher / student calls)
    # =========================================================================

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

    # =========================================================================
    # 5. Training step
    # =========================================================================

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

    # =========================================================================
    # 6. Evaluation
    # =========================================================================

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

    @override
    def test(self, phase="test") -> Dict[str, float]:
        self.student.eval()
        all_metrics = {}
        predictions_cache = {}
        is_multi = isinstance(self.test_loader, dict)

        for ds_name, loader in self._iter_test_loaders():
            result = self._evaluate_model(self.student, loader, return_predictions=True)
            metrics, images_list, preds_list, masks_list, fnames_list, per_sample = result

            if is_multi:
                self.logger.info(f"--- {phase} ({ds_name}) ---")
                self.evaluator.print_metrics(metrics, phase=f"{phase}_{ds_name}")
                for k, v in self._numeric_items(metrics).items():
                    all_metrics[f"{ds_name}/{k}"] = v
            else:
                self.logger.info(f"--- {phase} ---")
                self.evaluator.print_metrics(metrics, phase=phase)
                all_metrics.update(self._numeric_items(metrics))

            predictions_cache[ds_name] = (images_list, preds_list, masks_list, fnames_list, per_sample)

        if phase == "final_test":
            return all_metrics, predictions_cache

        return all_metrics

    # =========================================================================
    # 7. Metrics logging & checkpointing
    # =========================================================================

    @override
    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict,
        test_metrics: Dict = None,
    ):
        """Override: use pipeline-aware _wandb_log and include lr."""
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
        """Override: save best checkpoints for Dice, IoU, BIoU, and HD95."""
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
        if iou > self.best_iou:
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
        if biou > self.best_biou:
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
        if hd95 < self.best_hd95:
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

    # =========================================================================
    # 8. Visualization helpers
    # =========================================================================

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

    # =========================================================================
    # 9. Entry points (main loops)
    # =========================================================================

    @override
    def train(self):
        """Main distillation training loop."""
        for epoch in range(self.cfg.training.num_epochs):
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

            self.logger.info(
                f"Loading best model for final evaluation: {self.best_model_path}"
            )
            checkpoint = torch.load(
                self.best_model_path, map_location="cpu", weights_only=False
            )
            self.student.load_state_dict(checkpoint["model_state_dict"])
            self.student.to(self.device)
            self.student.eval()

            test_metrics, student_cache = self.test(phase="final_test")
            self.final_metrics = {f"final_test/{k}": v for k, v in test_metrics.items()}
            self._wandb_summary_update(self.final_metrics)
            self._wandb_log(self.final_metrics)
            self._save_latex_metrics_table_from_metrics(test_metrics)
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
            self.wandb_run = wandb.init(
                entity=self.cfg.get("wandb", {}).get("entity", "hheo"),
                project=self.cfg.get("wandb", {}).get("project", "TinyUSFM"),
                name=(
                    f"{self.teacher_name}_{self.student_name}_"
                    f"{self.cfg.method.name}_test_only_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                ),
                config=OmegaConf.to_container(self.cfg, resolve=True),
                tags=["test-only", "distillation"],
                mode=(
                    "disabled"
                    if self.cfg.get("wandb", {}).get("disabled", False)
                    else self.cfg.get("wandb", {}).get("mode", None)
                ),
            )

        test_metrics, predictions_cache = self.test(phase="final_test")
        self._save_test_results(test_metrics, predictions_cache=predictions_cache)
        self._save_per_sample_visualizations(predictions_cache)

        wandb.finish()
        self.logger.info("Test-only evaluation completed!")

    def dry_run(self):
        """Perform a quick end-to-end test of the training pipeline."""
        self.logger.info("Starting Dry Run (Pipeline Validation)...")

        limit_batches = 2
        batch_size = self.cfg.training.get("batch_size", 1)

        try:
            self.logger.info(f"Testing training step with {limit_batches} batches...")
            self.train_epoch(0)

            torch.cuda.empty_cache()

            self.logger.info("Testing validation step...")
            self._evaluate_model(
                self.student,
                self._make_limited_loader(self.val_loader, limit_batches, batch_size),
            )

            torch.cuda.empty_cache()

            self.logger.info("Testing evaluation step...")
            self._evaluate_model(
                self.student,
                self._make_limited_loader(self._first_test_loader, limit_batches, batch_size),
            )

            self.logger.info(
                "Dry Run completed successfully. Your experiment setup is valid."
            )

        except Exception as e:
            self.logger.error(f"Dry Run failed: {str(e)}")
            raise e
        finally:
            torch.cuda.empty_cache()

    # =========================================================================
    # 10. Small utilities
    # =========================================================================

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
