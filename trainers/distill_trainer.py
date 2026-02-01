import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, Optional

from utils.data_processing_seg import SegDatasetProcessor
from utils.evaluate import Evaluator_seg
from utils.logger import setup_logger
from utils.schedule import build_scheduler
from trainers.model_builder import ModelBuilder
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
from omegaconf import OmegaConf, DictConfig


class DistillTrainer:
    """
    Trainer for Knowledge Distillation.
    Encapsulates setup, training, validation, and testing logic.
    """

    def __init__(self, cfg: DictConfig, model: Optional[nn.Module] = None):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(cfg.hardware.seed)

        # Setup directories
        self.log_dir = create_log_dir(cfg)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.log_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir = self.log_dir / "visualizations"
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        # Save experiment summary
        save_experiment_summary(cfg, self.log_dir)

        # Setup logger
        self.logger = setup_logger(str(self.log_dir / "distill.log"))
        self.teacher_short = get_teacher_short_name(cfg)
        self.student_short = get_student_short_name(cfg)
        self.dataset_name = get_dataset_short_name(cfg)

        self.logger.info(f"Starting Distillation: {cfg.method.name}")
        self.logger.info(
            f"Teacher: {self.teacher_short} -> Student: {self.student_short}"
        )
        self.logger.info(f"Dataset: {self.dataset_name}")
        self.logger.info(f"Log directory: {self.log_dir}")

        # Initialize wandb
        exp_name = f"{self.teacher_short}_{self.student_short}_{cfg.method.name}_{self.dataset_name}"
        wandb_mode = "disabled" if cfg.get("debug", False) else None
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=exp_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=wandb_mode,
        )

        # Early validation
        self.validate_config()

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

    def _setup_data(self):
        distill_cfg = self.cfg.get("distillation", {})
        distill_enabled = distill_cfg.get("enabled", False)

        if distill_enabled:
            adaptation_ratio = distill_cfg.get("adaptation_ratio", 0.3)
            split_seed = distill_cfg.get("split_seed", 42)
            split_file = distill_cfg.get("split_file", None)

            self.logger.info(f"=== Distillation Split Mode ===")
            loaders = SegDatasetProcessor.build_distillation_data_loaders(
                self.cfg,
                adaptation_ratio=adaptation_ratio,
                seed=split_seed,
                split_file=split_file,
            )
            self.train_loader = loaders["distillation_train"]
            self.val_loader = loaders["distillation_val"]
            self.test_loader = loaders["test"]
        else:
            self.train_loader, self.val_loader, self.test_loader = (
                SegDatasetProcessor.build_data_loaders(self.cfg)
            )

    def _setup_models(self):
        # Create teacher model (num_classes/img_size from ${data.*} interpolation)
        self.teacher = ModelBuilder.create_model(self.cfg.teacher)
        self.teacher = self.teacher.to(self.device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Create student model
        self.student = ModelBuilder.create_model(self.cfg.student)
        self.student = self.student.to(self.device)

        # Create distiller
        self.distiller = create_distiller(self.cfg).to(self.device)
        self.distiller.prepare(self.student, self.teacher)

    def _setup_optimizer(self):
        cfg = self.cfg
        if cfg.optimizer.name == "AdamW":
            param_groups = [
                {"params": self.student.parameters(), "lr": cfg.training.lr}
            ]
            if list(self.distiller.parameters()):
                param_groups.append(
                    {"params": self.distiller.parameters(), "lr": cfg.training.lr}
                )
            self.optimizer = optim.AdamW(
                param_groups, weight_decay=cfg.optimizer.weight_decay
            )
        else:
            params = list(self.student.parameters()) + list(self.distiller.parameters())
            self.optimizer = optim.Adam(params, lr=cfg.training.lr)

        self.scheduler = build_scheduler(self.optimizer, cfg)

    def train_epoch(self, epoch):
        self.student.train()
        self.distiller.train()

        running_losses = {}
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        limit_batches = self.cfg.training.get("limit_train_batches")

        for i, (images, masks, _) in enumerate(pbar):
            if limit_batches is not None and i >= limit_batches:
                break
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.distiller.on_step_begin()

            with torch.no_grad():
                if hasattr(self.teacher, "image_encoder") or hasattr(
                    self.teacher, "sam"
                ):
                    teacher_outputs = self.teacher(
                        images, False, self.cfg.teacher.img_size
                    )
                else:
                    teacher_outputs = {"masks": self.teacher(images)}

            s_model = (
                self.student.module if hasattr(self.student, "module") else self.student
            )
            student_raw = (
                s_model(images, return_features=True)
                if hasattr(s_model, "backbone")
                else self.student(images)
            )
            if isinstance(student_raw, tuple):
                student_outputs = {"masks": student_raw[0], "features": student_raw[1]}
            else:
                student_outputs = {"masks": student_raw}

            loss_dict = self.distiller(student_outputs, teacher_outputs, masks)
            loss = loss_dict["loss"]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            for k, v in loss_dict.items():
                running_losses[k] = running_losses.get(k, 0.0) + v.item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if self.global_step % 10 == 0:
                step_log = {"global_step": self.global_step}
                for k, v in loss_dict.items():
                    step_log[f"train_step/{k}"] = v.item()
                step_log["train_step/lr"] = self.optimizer.param_groups[0]["lr"]
                wandb.log(step_log)
            self.global_step += 1

        return {k: v / (i + 1) for k, v in running_losses.items()}

    def validate(self, epoch):
        self.student.eval()
        val_metrics = self.evaluator.evaluate_model(
            self.student, self.val_loader, self.device, self.cfg.data.num_classes
        )
        self.logger.info(f"Epoch {epoch+1} Val Dice: {val_metrics['Dice']:.4f}")
        return val_metrics

    def test(self, phase="Test"):
        self.student.eval()
        all_metrics = {}
        if isinstance(self.test_loader, dict):
            for ds_name, loader in self.test_loader.items():
                metrics = self.evaluator.evaluate_model(
                    self.student, loader, self.device, self.cfg.data.num_classes
                )
                self.logger.info(f"--- {phase} ({ds_name}) ---")
                self.evaluator.print_metrics(metrics, phase=f"{phase}_{ds_name}")
                for k, v in metrics.items():
                    if isinstance(v, (float, int)):
                        all_metrics[f"{phase.lower()}/{ds_name}/{k.lower()}"] = v
        else:
            metrics = self.evaluator.evaluate_model(
                self.student, self.test_loader, self.device, self.cfg.data.num_classes
            )
            self.logger.info(f"--- {phase} ---")
            self.evaluator.print_metrics(metrics, phase=phase)
            for k, v in metrics.items():
                if isinstance(v, (float, int)):
                    all_metrics[f"{phase.lower()}/{k.lower()}"] = v
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
                self.models_dir / f"best_epoch_{epoch+1}_dice{self.best_dice:.4f}.pth"
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

    def train(self):
        early_stopping_cfg = self.cfg.training.get("early_stopping", {})
        es_enabled = early_stopping_cfg.get("enabled", False)
        if es_enabled:
            patience = early_stopping_cfg.patience
            min_delta = early_stopping_cfg.min_delta
            es_counter = 0

        for epoch in range(self.cfg.training.num_epochs):
            train_losses = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            # Log to wandb
            log_data = {
                "epoch": epoch + 1,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            for k, v in train_losses.items():
                log_data[f"train/{k}"] = v
            for k, v in val_metrics.items():
                if isinstance(v, (float, int)):
                    log_data[f"val/{k.lower()}"] = v

            # Test & Visualize every epoch
            test_metrics = self.test(phase="Test")
            log_data.update(test_metrics)

            vis_loader = (
                list(self.test_loader.values())[0]
                if isinstance(self.test_loader, dict)
                else self.test_loader
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

            wandb.log(log_data)

            # Checkpointing
            improved = self._save_checkpoint(epoch, val_metrics["Dice"])

            if improved and es_enabled:
                if (
                    val_metrics["Dice"] - self.best_dice
                ) > min_delta:  # Error in logic above, fixed here
                    es_counter = 0
                else:
                    # If improved but not by min_delta, should we reset es_counter?
                    # Usually improvement resets it.
                    es_counter = 0
            elif es_enabled:
                es_counter += 1
                if es_counter >= patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

            self.scheduler.step()

        # Final Evaluation
        self._final_evaluation()
        wandb.finish()

    def _final_evaluation(self):
        if self.best_model_path and self.best_model_path.exists():
            checkpoint = torch.load(self.best_model_path, weights_only=False)
            self.student.load_state_dict(checkpoint["model_state_dict"])

            test_metrics = self.test(phase="Final_Test")
            wandb.log(test_metrics)

            vis_loader = (
                list(self.test_loader.values())[0]
                if isinstance(self.test_loader, dict)
                else self.test_loader
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
                epoch=None,
            )

    def validate_config(self):
        """Perform early validation of configuration."""
        self.logger.info("Checking configuration health...")

        # Check for essential components
        required_keys = ["teacher", "student", "method", "data", "training"]
        for key in required_keys:
            if key not in self.cfg:
                raise ValueError(f"Missing required configuration section: {key}")

        # Check for teacher/student names
        if "name" not in self.cfg.teacher:
            raise ValueError("Teacher name is not specified.")
        if "name" not in self.cfg.student:
            raise ValueError("Student name is not specified.")

        self.logger.info("Configuration validated successfully.")

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
            from torch.utils.data import Subset

            val_subset_size = min(
                len(self.val_loader.dataset), limit_batches * batch_size
            )
            val_indices = list(range(val_subset_size))
            limited_val_loader = torch.utils.data.DataLoader(
                Subset(self.val_loader.dataset, val_indices),
                batch_size=batch_size,
                num_workers=0,
            )
            self.evaluator.evaluate_model(
                self.student, limited_val_loader, self.device, self.cfg.data.num_classes
            )

            torch.cuda.empty_cache()

            # 3. Test/Evaluation Test
            self.logger.info("Testing evaluation step...")
            if isinstance(self.test_loader, dict):
                first_name = list(self.test_loader.keys())[0]
                first_loader = self.test_loader[first_name]
                test_subset_size = min(
                    len(first_loader.dataset), limit_batches * batch_size
                )
                test_indices = list(range(test_subset_size))
                limited_test_loader = torch.utils.data.DataLoader(
                    Subset(first_loader.dataset, test_indices),
                    batch_size=batch_size,
                    num_workers=0,
                )
                self.evaluator.evaluate_model(
                    self.student,
                    limited_test_loader,
                    self.device,
                    self.cfg.data.num_classes,
                )
            else:
                test_subset_size = min(
                    len(self.test_loader.dataset), limit_batches * batch_size
                )
                test_indices = list(range(test_subset_size))
                limited_test_loader = torch.utils.data.DataLoader(
                    Subset(self.test_loader.dataset, test_indices),
                    batch_size=batch_size,
                    num_workers=0,
                )
                self.evaluator.evaluate_model(
                    self.student,
                    limited_test_loader,
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
