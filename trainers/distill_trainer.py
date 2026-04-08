import os
import logging as _logging
from hydra.utils import instantiate
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple

from utils.data_processing_seg import SegDatasetProcessor
from utils.evaluate import Evaluator_seg
from utils.logger import setup_logger
from utils.schedule import build_scheduler
from distillers import create_distiller
from utils.distill_utils import (
    get_teacher_short_name,
    get_student_short_name,
    get_dataset_short_name,
    create_log_dir,
    save_experiment_summary,
)
from utils.visualize import visualize_distillation, visualize_distillation_precomputed
from utils.utils import set_seed
from trainers.base_trainer import BaseTrainer
from omegaconf import OmegaConf, DictConfig


class DistillTrainer(BaseTrainer):
    """
    Trainer for Knowledge Distillation.
    Inherits common infrastructure from BaseTrainer and overrides distillation-specific logic.
    """

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

        from utils.distill_utils import resolve_teacher_checkpoint
        self.teacher_ckpt = resolve_teacher_checkpoint(self.cfg)
        if self.teacher_ckpt:
            self.cfg.teacher.checkpoint = self.teacher_ckpt

        self._setup_directories()
        save_experiment_summary(cfg, self.exp_dir)
        self._setup_logger()
        self._setup_pipeline_log()

        self.teacher_short = get_teacher_short_name(cfg)
        self.student_short = get_student_short_name(cfg)
        self.dataset_name = get_dataset_short_name(cfg)

        self.logger.info(f"Starting Distillation: {cfg.method.name}")
        self.logger.info(
            f"Teacher: {self.teacher_short} -> Student: {self.student_short}"
        )
        self.logger.info(f"Dataset: {self.dataset_name}")
        self.logger.info(f"Experiment directory: {self.exp_dir}")

        self._setup_wandb()
        self._create_dataloaders()
        self._create_model()
        self._create_optimizer()
        self._create_scheduler()
        self._setup_early_stopping()  # sets self.early_stopping from base

    # -------------------------------------------------------------------------
    # Overrides: directories / logger / wandb
    # -------------------------------------------------------------------------

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
                else f"{self.teacher_short}_{self.student_short}_{self.cfg.method.name}_{self.dataset_name}"
            )
            self.wandb_run = wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                name=exp_name,
                config=OmegaConf.to_container(self.cfg, resolve=True),
                mode=wandb_mode,
            )

    # -------------------------------------------------------------------------
    # Pipeline-prefix wandb helpers
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Abstract method implementations (BaseTrainer interface)
    # -------------------------------------------------------------------------

    def _create_dataloaders(self):
        self.train_loader, self.val_loader, self.test_loader = (
            SegDatasetProcessor.build_data_loaders(self.cfg)
        )

    def _create_model(self):
        """Create teacher, student, and distiller."""
        if self.teacher_ckpt:
            self.logger.info(f"Using teacher checkpoint: {self.teacher_ckpt}")

        self.teacher = instantiate(self.cfg.teacher)
        self.teacher = self.teacher.to(self.device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        OmegaConf.set_struct(self.cfg.student, False)
        OmegaConf.set_struct(self.cfg.method, False)

        teacher_uses_alignment = getattr(self.teacher, "use_alignment", False)

        if teacher_uses_alignment:
            t_align_channels = getattr(self.teacher, "alignment_hidden_channels", 256)
            self.logger.info(
                f"Teacher uses alignment layer with {t_align_channels} channels. "
                "Enabling student alignment."
            )
            self.cfg.student.use_alignment = True
            self.cfg.student.alignment_out_channels = t_align_channels
            self.cfg.student.student_channels = t_align_channels
            self.cfg.method.teacher_alignment_channels = t_align_channels

            if (
                self.cfg.method.get("use_gradnorm")
                and self.cfg.method.get("gamma_align", 0) == 0
            ):
                self.logger.info(
                    "Enabling gamma_align=1.0 for GradNorm balancing as alignment is active."
                )
                self.cfg.method.gamma_align = 1.0
        else:
            self.logger.info(
                "Teacher does not use alignment layer. Disabling student alignment layer and align_loss."
            )
            self.cfg.student.use_alignment = False
            self.cfg.method.gamma_align = 0.0

            if "student_channels" not in self.cfg.student:
                self.cfg.student.student_channels = 48

        self.student = instantiate(self.cfg.student)
        self.student = self.student.to(self.device)
        self.model = self.student  # base-class compatibility

        self.distiller = create_distiller(self.cfg).to(self.device)
        self.distiller.prepare(self.student, self.teacher)

        self._log_model_info()

    def _create_optimizer(self):
        param_groups = [
            {"params": self.student.parameters(), "lr": self.cfg.training.lr}
        ]
        if list(self.distiller.parameters()):
            param_groups.append(
                {"params": self.distiller.parameters(), "lr": self.cfg.training.lr}
            )
        self.optimizer = optim.AdamW(
            param_groups, weight_decay=self.cfg.optimizer.weight_decay
        )

    def _create_scheduler(self):
        self.scheduler = build_scheduler(self.optimizer, self.cfg)

    # -------------------------------------------------------------------------
    # Overrides: model info / metrics logging / checkpointing / per-sample output
    # -------------------------------------------------------------------------

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
        self.logger.info(
            "Val:\n    " + ", ".join(f"{k}: {v:.4f}" for k, v in val_items.items())
        )

        log_data = {
            "epoch": epoch + 1,
            "train/lr": self.optimizer.param_groups[0]["lr"],
        }
        for k, v in train_metrics.items():
            log_data[f"train/{k}"] = v
        for k, v in val_items.items():
            log_data[f"val/{k}"] = v
        self._wandb_log(log_data)

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

    def _save_per_sample_metrics(self, predictions_cache: dict) -> None:
        """Override: save per-sample metrics CSV + per-sample visualizations."""
        import pandas as pd
        from utils.visualize import visualize_precomputed_predictions

        out_dir = self.exp_dir / "test_results"
        out_dir.mkdir(parents=True, exist_ok=True)

        all_dfs = []
        for ds_name, (images_list, preds_list, masks_list, fnames_list) in predictions_cache.items():
            vis_dir = self.vis_dir / "per_sample" / ds_name
            visualize_precomputed_predictions(
                images_list,
                preds_list,
                masks_list,
                num_classes=self.cfg.data.num_classes,
                save_dir=vis_dir,
                filenames_list=fnames_list,
            )

            df = Evaluator_seg.build_per_sample_df(
                preds_list,
                masks_list,
                fnames_list,
                num_classes=self.cfg.data.num_classes,
            )
            df.insert(0, "dataset", ds_name)
            csv_path = out_dir / f"metrics_{ds_name}.csv"
            df.to_csv(csv_path, index=False, float_format="%.6f")
            self.logger.info(f"Per-sample metrics saved → {csv_path}")
            all_dfs.append(df)

        if len(all_dfs) > 1:
            combined_path = out_dir / "metrics_all.csv"
            pd.concat(all_dfs, ignore_index=True).to_csv(
                combined_path, index=False, float_format="%.6f"
            )
            self.logger.info(f"Combined per-sample metrics saved → {combined_path}")

    # -------------------------------------------------------------------------
    # Core training / validation / test
    # -------------------------------------------------------------------------

    def train_epoch(self, epoch) -> Dict[str, float]:
        self.student.train()
        self.distiller.train()

        running_losses = {}
        pbar = tqdm(
            self.train_loader, desc=f"Epoch {epoch + 1}/{self.cfg.training.num_epochs}"
        )
        limit_batches = self.cfg.training.get("limit_train_batches")

        for i, (images, masks, *_) in enumerate(pbar):
            if limit_batches is not None and i >= limit_batches:
                break
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.distiller.on_step_begin()

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

    def validate(self, epoch) -> Dict[str, float]:
        self.student.eval()
        val_metrics = self._evaluate_model(self.student, self.val_loader)
        self.logger.info(f"Epoch {epoch+1} Val Dice: {val_metrics['Dice']:.4f}")
        return val_metrics

    def test(self, phase="test") -> Dict[str, float]:
        self.student.eval()
        all_metrics = {}
        predictions_cache = {}
        is_multi = isinstance(self.test_loader, dict)

        for ds_name, loader in self._iter_test_loaders():
            result = self._evaluate_model(self.student, loader, return_predictions=True)
            metrics, images_list, preds_list, masks_list, fnames_list = result

            if is_multi:
                self.logger.info(f"--- {phase} ({ds_name}) ---")
                self.evaluator.print_metrics(metrics, phase=f"{phase}_{ds_name}")
                for k, v in self._numeric_items(metrics).items():
                    all_metrics[f"{ds_name}/{k}"] = v
            else:
                self.logger.info(f"--- {phase} ---")
                self.evaluator.print_metrics(metrics, phase=phase)
                all_metrics.update(self._numeric_items(metrics))

            predictions_cache[ds_name] = (images_list, preds_list, masks_list, fnames_list)

        if phase == "final_test":
            return all_metrics, predictions_cache

        return all_metrics

    # -------------------------------------------------------------------------
    # Main training loop
    # -------------------------------------------------------------------------

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
                visualize_distillation(
                    self.teacher,
                    self.student,
                    vis_loader,
                    self.device,
                    self.cfg.data.num_classes,
                    self.cfg.data.img_size,
                    self.vis_dir,
                    num_samples=self.cfg.visualization.num_samples,
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

    # -------------------------------------------------------------------------
    # Distill-specific helpers
    # -------------------------------------------------------------------------

    @property
    def _first_test_loader(self):
        """Get the first test loader (handles both dict and single loader)."""
        if isinstance(self.test_loader, dict):
            return next(iter(self.test_loader.values()))
        return self.test_loader

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
            img_size = self.cfg.teacher.get("img_size", self.cfg.data.img_size)
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
            img_size = self.cfg.student.get("img_size", self.cfg.data.img_size)
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

    def _save_distill_model(self, path: Path, epoch: int, metrics: dict):
        """Save student + distiller state dict to path."""
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": self.student.state_dict(),
                "distiller_state_dict": self.distiller.state_dict(),
                **{k: v for k, v in metrics.items() if isinstance(v, (float, int))},
            },
            path,
        )

    @staticmethod
    def _numeric_items(d: dict) -> dict:
        """Filter dict to only numeric (float/int) values."""
        return {k: v for k, v in d.items() if isinstance(v, (float, int))}

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
            student_images_list, student_preds_list, masks_list, fnames_list = student_cache[first_ds]
            first_loader = (
                self._first_test_loader
                if not isinstance(self.test_loader, dict)
                else list(self.test_loader.values())[0]
            )
            teacher_result = self._evaluate_model(self.teacher, first_loader, return_predictions=True)
            _, teacher_images_list, teacher_preds_list, _, _ = teacher_result

            visualize_distillation_precomputed(
                student_images_list,
                teacher_preds_list,
                student_preds_list,
                masks_list,
                num_classes=self.cfg.data.num_classes,
                save_dir=self.vis_dir,
                num_samples=self.cfg.visualization.num_samples,
                filenames_list=fnames_list,
                epoch=None,
            )
        else:
            self.logger.warning(
                "Best model checkpoint not found. Final metrics are unavailable."
            )

    @staticmethod
    def _make_limited_loader(loader, limit_batches, batch_size):
        """Create a small DataLoader subset for dry-run testing."""
        from torch.utils.data import Subset, DataLoader

        n = min(len(loader.dataset), limit_batches * batch_size)
        return DataLoader(
            Subset(loader.dataset, list(range(n))), batch_size=batch_size, num_workers=0
        )

    def dry_run(self):
        """Perform a quick end-to-end test of the training pipeline."""
        self.logger.info("Starting Dry Run (Pipeline Validation)...")

        limit_batches = 2
        batch_size = self.cfg.training.get("batch_size", 1)

        try:
            self.logger.info(f"Testing training step with {limit_batches} batches...")
            old_limit = self.cfg.training.get("limit_train_batches")
            self.cfg.training.limit_train_batches = limit_batches
            self.train_epoch(0)

            if old_limit is not None:
                self.cfg.training.limit_train_batches = old_limit
            else:
                self.cfg.training.limit_train_batches = None

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
