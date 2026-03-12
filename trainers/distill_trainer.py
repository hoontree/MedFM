import os
from hydra.utils import instantiate
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional

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
    visualize_distillation,
)
from utils.utils import set_seed
from trainers.base_trainer import EarlyStopping
from omegaconf import OmegaConf, DictConfig


class DistillTrainer:
    """
    Trainer for Knowledge Distillation.
    Encapsulates setup, training, validation, and testing logic.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(cfg.hardware.seed)

        # Setup directories
        self.log_dir = create_log_dir(cfg)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.log_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir = self.log_dir / "visualizations"
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        # Save experiment summary
        save_experiment_summary(cfg, self.log_dir)

        # Setup logger
        self.logger = setup_logger(
            str(self.log_dir / "distill.log"), logger_name="medfm.distill"
        )

        # In pipeline mode, also append distillation logs to the teacher's log directory
        pipeline_cfg = cfg.get("pipeline", {})
        teacher_log_dir = pipeline_cfg.get("teacher_log_dir", None)
        if teacher_log_dir:
            import logging as _logging
            pipeline_log_path = Path(teacher_log_dir) / "pipeline.log"
            pipeline_handler = _logging.FileHandler(
                str(pipeline_log_path), mode="a", encoding="utf-8"
            )
            pipeline_handler.setLevel(_logging.INFO)
            fmt = "[%(asctime)s %(name)s] (%(filename)s:%(lineno)d): %(levelname)s %(message)s"
            pipeline_handler.setFormatter(_logging.Formatter(fmt))
            self.logger.addHandler(pipeline_handler)

        self.teacher_short = get_teacher_short_name(cfg)
        self.student_short = get_student_short_name(cfg)
        self.dataset_name = get_dataset_short_name(cfg)

        self.logger.info(f"Starting Distillation: {cfg.method.name}")
        self.logger.info(
            f"Teacher: {self.teacher_short} -> Student: {self.student_short}"
        )
        self.logger.info(f"Dataset: {self.dataset_name}")
        self.logger.info(f"Log directory: {self.log_dir}")

        # Initialize wandb — reuse the teacher run when running inside a pipeline
        is_sweep = os.environ.get("WANDB_SWEEP_ID") is not None
        teacher_run_id = pipeline_cfg.get("teacher_run_id", None)
        wandb_mode = cfg.get("wandb", {}).get("mode", None)
        if cfg.get("debug", False) or cfg.get("wandb", {}).get("disabled", False):
            wandb_mode = "disabled"

        if teacher_run_id is not None and wandb.run is not None:
            # Pipeline: teacher wandb run is still open — attach to it directly
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
                else f"{self.teacher_short}_{self.student_short}_{cfg.method.name}_{self.dataset_name}"
            )
            self.wandb_run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=exp_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                mode=wandb_mode,
            )
            self._wandb_metric_prefix = ""

        # Build Data Loaders
        self._setup_data()

        # Build Models
        self._setup_models()

        # Build Optimizer & Scheduler
        self._setup_optimizer()

        self.evaluator = Evaluator_seg()
        self.best_dice = 0.0
        self.best_model_path = None
        self.global_step = 0
        self.final_metrics: Dict[str, float] = {}

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

    def _setup_data(self):
        self.train_loader, self.val_loader, self.test_loader = (
            SegDatasetProcessor.build_data_loaders(self.cfg)
        )

    def _setup_models(self):
        self.teacher = instantiate(self.cfg.teacher)
        self.teacher = self.teacher.to(self.device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        OmegaConf.set_struct(self.cfg.student, False)
        OmegaConf.set_struct(self.cfg.method, False)

        if self.teacher.use_alignment:
            t_align_channels = getattr(self.teacher, "alignment_hidden_channels", 256)
            self.logger.info(
                f"Teacher uses alignment layer with {t_align_channels} channels. Enabling student alignment."
            )
            self.cfg.student.use_alignment = True
            self.cfg.student.alignment_out_channels = t_align_channels
            self.cfg.student.student_channels = t_align_channels
            self.cfg.method.teacher_alignment_channels = t_align_channels

            # Proactive GradNorm adjustment: If aligning, ensure align loss is initialized reasonably
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
                "Teacher does not use alignment layer. Disabling student alignment layer."
            )
            self.cfg.student.use_alignment = False
            # If no alignment, student_channels should be the output of the neck (48 for TinyUSFM)
            # Default to 48 if not specified.
            if "student_channels" not in self.cfg.student:
                self.cfg.student.student_channels = 48

        # Create student model with potentially updated config
        self.student = instantiate(self.cfg.student)
        self.student = self.student.to(self.device)

        # Create distiller
        self.distiller = create_distiller(self.cfg).to(self.device)
        self.distiller.prepare(self.student, self.teacher)

    def _setup_optimizer(self):
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

        self.scheduler = build_scheduler(self.optimizer, self.cfg)

    @property
    def _first_test_loader(self):
        """Get the first test loader (handles both dict and single loader)."""
        if isinstance(self.test_loader, dict):
            return next(iter(self.test_loader.values()))
        return self.test_loader

    def train_epoch(self, epoch):
        self.student.train()
        self.distiller.train()

        running_losses = {}
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.cfg.training.num_epochs}")
        limit_batches = self.cfg.training.get("limit_train_batches")

        for i, (images, masks, *_) in enumerate(pbar):
            if limit_batches is not None and i >= limit_batches:
                break
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.distiller.on_step_begin()

            with torch.no_grad():
                teacher_outputs = self.teacher(
                    images, False, self.cfg.teacher.img_size
                )

            student_raw = self.student(images, return_features=True)
            
            student_outputs = {"masks": student_raw[0], "features": student_raw[1]}

            loss_dict = self.distiller(student_outputs, teacher_outputs, masks)
            loss = loss_dict["loss"]

            self.optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping
            max_norm = self.cfg.optimizer.gradient_clip.get("max_norm", 1.0)
            for pg in self.optimizer.param_groups:
                nn.utils.clip_grad_norm_(pg["params"], max_norm=max_norm)

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

    def validate(self, epoch):
        self.student.eval()
        val_metrics = self.evaluator.evaluate_model(
            self.student, self.val_loader, self.device, self.cfg.data.num_classes
        )
        self.logger.info(f"Epoch {epoch+1} Val Dice: {val_metrics['Dice']:.4f}")
        return val_metrics

    def test(self, phase="test"):
        self.student.eval()
        all_metrics = {}
        if isinstance(self.test_loader, dict):
            for ds_name, loader in self.test_loader.items():
                metrics = self.evaluator.evaluate_model(
                    self.student, loader, self.device, self.cfg.data.num_classes
                )
                self.logger.info(f"--- {phase} ({ds_name}) ---")
                self.evaluator.print_metrics(metrics, phase=f"{phase}_{ds_name}")
                for k, v in self._numeric_items(metrics).items():
                    all_metrics[f"{ds_name}/{k}"] = v
        else:
            metrics = self.evaluator.evaluate_model(
                self.student, self.test_loader, self.device, self.cfg.data.num_classes
            )
            self.logger.info(f"--- {phase} ---")
            self.evaluator.print_metrics(metrics, phase=phase)
            all_metrics.update(self._numeric_items(metrics))
        return all_metrics

    def _save_checkpoint(self, epoch, val_dice):
        if val_dice > self.best_dice:
            # Delete previous best
            if self.best_model_path and self.best_model_path.exists():
                try:
                    self.best_model_path.unlink()
                except Exception as e:
                    self.logger.warning(f"Could not delete old best model: {e}")

            self.best_dice = val_dice
            self.best_model_path = (
                self.ckpt_dir / f"best_epoch_{epoch+1}_dice{self.best_dice:.4f}.pth"
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": self.student.state_dict(),
                    "distiller_state_dict": self.distiller.state_dict(),
                    "dice": self.best_dice,
                },
                self.best_model_path,
            )
            self.logger.info(f"Saved best model to {self.best_model_path}")
            return True
        return False

    @staticmethod
    def _numeric_items(d: dict) -> dict:
        """Filter dict to only numeric (float/int) values."""
        return {k: v for k, v in d.items() if isinstance(v, (float, int))}

    def _log_epoch_metrics(self, epoch, train_losses, val_metrics):
        """Log epoch metrics to console and wandb in unified format."""
        num_epochs = self.cfg.training.num_epochs
        self.logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        self.logger.info(
            "Train:\n    "
            + ", ".join(f"{k}: {v:.4f}" for k, v in train_losses.items())
        )
        val_items = self._numeric_items(val_metrics)
        self.logger.info(
            "Val:\n    "
            + ", ".join(f"{k}: {v:.4f}" for k, v in val_items.items())
        )

        # wandb
        log_data = {
            "epoch": epoch + 1,
            "train/lr": self.optimizer.param_groups[0]["lr"],
        }
        for k, v in train_losses.items():
            log_data[f"train/{k}"] = v
        for k, v in val_items.items():
            log_data[f"val/{k}"] = v
        self._wandb_log(log_data)

    def train(self):
        early_stopping_cfg = self.cfg.training.get("early_stopping", {})
        early_stopping = None
        if early_stopping_cfg.get("enabled", False):
            early_stopping = EarlyStopping(
                patience=early_stopping_cfg.get("patience", 15),
                min_delta=early_stopping_cfg.get("min_delta", 0.001),
                mode="max",
            )
            self.logger.info(
                f"Early stopping enabled: patience={early_stopping.patience}, "
                f"min_delta={early_stopping.min_delta}"
            )

        for epoch in range(self.cfg.training.num_epochs):
            train_losses = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            # Unified logging
            self._log_epoch_metrics(epoch, train_losses, val_metrics)

            # Periodic Visualization every 5 epochs
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
                    self.cfg.teacher.img_size,
                    self.vis_dir,
                    num_samples=self.cfg.visualization.num_samples,
                    epoch=epoch,
                )

            # Checkpointing
            self._save_checkpoint(epoch, val_metrics["Dice"])

            # Early stopping
            if early_stopping is not None:
                early_stopping(val_metrics.get("Dice", 0.0))
                if early_stopping.should_stop():
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            self.scheduler.step()

        # Final Evaluation
        self._final_evaluation()
        wandb.finish()
        return {
            "best_model_path": (
                str(self.best_model_path) if self.best_model_path else None
            ),
            "log_dir": str(self.log_dir),
            "final_metrics": dict(self.final_metrics),
        }

    def _final_evaluation(self):
        if self.best_model_path and self.best_model_path.exists():
            # Clear memory before final evaluation
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
            # Ensure student is on device and in eval mode
            self.student.to(self.device)
            self.student.eval()

            test_metrics = self.test(phase="final_test")
            # Prefix with final_test/ for wandb summary
            self.final_metrics = {f"final_test/{k}": v for k, v in test_metrics.items()}
            self._wandb_summary_update(self.final_metrics)

            visualize_distillation(
                self.teacher,
                self.student,
                self._first_test_loader,
                self.device,
                self.cfg.data.num_classes,
                self.cfg.teacher.img_size,
                self.vis_dir,
                num_samples=self.cfg.visualization.num_samples,
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
        return DataLoader(Subset(loader.dataset, list(range(n))), batch_size=batch_size, num_workers=0)

    def dry_run(self):
        """Perform a quick end-to-end test of the training pipeline."""
        self.logger.info("🚀 Starting Dry Run (Pipeline Validation)...")

        limit_batches = 2
        batch_size = self.cfg.training.get("batch_size", 1)

        try:
            # 1. Forward/Backward Test
            self.logger.info(f"Testing training step with {limit_batches} batches...")
            old_limit = self.cfg.training.get("limit_train_batches")
            self.cfg.training.limit_train_batches = limit_batches
            self.train_epoch(0)

            # Restore limit
            if old_limit is not None:
                self.cfg.training.limit_train_batches = old_limit
            else:
                self.cfg.training.limit_train_batches = None

            torch.cuda.empty_cache()

            # 2. Validation Test
            self.logger.info("Testing validation step...")
            self.evaluator.evaluate_model(
                self.student,
                self._make_limited_loader(self.val_loader, limit_batches, batch_size),
                self.device,
                self.cfg.data.num_classes,
            )

            torch.cuda.empty_cache()

            # 3. Test/Evaluation Test
            self.logger.info("Testing evaluation step...")
            self.evaluator.evaluate_model(
                self.student,
                self._make_limited_loader(self._first_test_loader, limit_batches, batch_size),
                self.device,
                self.cfg.data.num_classes,
            )

            self.logger.info(
                "✅ Dry Run completed successfully! Your experiment setup is valid."
            )

        except Exception as e:
            self.logger.error(f"❌ Dry Run failed: {str(e)}")
            raise e
        finally:
            torch.cuda.empty_cache()
