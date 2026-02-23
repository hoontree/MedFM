import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional, Tuple

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
from omegaconf import OmegaConf, DictConfig, ListConfig


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
        is_sweep = os.environ.get("WANDB_SWEEP_ID") is not None
        exp_name = (
            None
            if is_sweep
            else f"{self.teacher_short}_{self.student_short}_{cfg.method.name}_{self.dataset_name}"
        )
        wandb_mode = cfg.get("wandb", {}).get("mode", None)
        if cfg.get("debug", False) or cfg.get("wandb", {}).get("disabled", False):
            wandb_mode = "disabled"

        self.wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=exp_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=wandb_mode,
        )

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
        self.pipeline_metric: Optional[float] = None
        self.pipeline_metric_dataset: Optional[str] = None
        self.pipeline_metric_key: str = (
            cfg.get("pipeline", {})
            .get("distill", {})
            .get("metric_key", "pipeline/distill_final_dice")
        )

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

    def _validate_config(self):
        """Proactively validate configuration before model building."""
        t_cfg = self.cfg.get("teacher", {})
        adaptation_mode = t_cfg.get("adaptation_mode", "")

        # Validation for alignment parameters
        has_align = "alignment" in adaptation_mode
        if not has_align:
            if t_cfg.get("alignment_num_blocks") or t_cfg.get(
                "alignment_hidden_channels"
            ):
                self.logger.warning(
                    f"Structural alignment parameters found but adaptation_mode '{adaptation_mode}' does not use it. These will be ignored."
                )

        # Validation for LoRA parameters
        if "lora" not in adaptation_mode and "dual_lora" not in adaptation_mode:
            if t_cfg.get("r_e") or t_cfg.get("r_d"):
                self.logger.warning(
                    f"LoRA rank parameters found but adaptation_mode '{adaptation_mode}' is not LoRA-based."
                )

    def _setup_models(self):
        self._validate_config()
        # Create teacher model (num_classes/img_size from ${data.*} interpolation)
        self.teacher = ModelBuilder.create_model(self.cfg.teacher)
        self.teacher = self.teacher.to(self.device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Allow adding new keys to student config
        from omegaconf import OmegaConf

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
        self.student = ModelBuilder.create_model(self.cfg.student)
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

            # Gradient Clipping
            grad_clip_cfg = self.cfg.optimizer.get("gradient_clip", {})
            if grad_clip_cfg.get("enabled", False):
                nn.utils.clip_grad_norm_(
                    self.optimizer.param_groups[0]["params"],
                    max_norm=grad_clip_cfg.get("max_norm", 1.0),
                )
                if len(self.optimizer.param_groups) > 1:
                    nn.utils.clip_grad_norm_(
                        self.optimizer.param_groups[1]["params"],
                        max_norm=grad_clip_cfg.get("max_norm", 1.0),
                    )

            self.optimizer.step()

            for k, v in loss_dict.items():
                val = v.item() if hasattr(v, "item") else v
                running_losses[k] = running_losses.get(k, 0.0) + val

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if self.global_step % 10 == 0:
                step_log = {"global_step": self.global_step}
                for k, v in loss_dict.items():
                    val = v.item() if hasattr(v, "item") else v
                    step_log[f"train_step/{k}"] = val
                step_log["train_step/lr"] = self.optimizer.param_groups[0]["lr"]
                self.wandb_run.log(step_log)
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

            # Periodic Visualization every 5 epochs
            if (epoch + 1) % 5 == 0:
                vis_loader = (
                    list(self.test_loader.values())[0]
                    if isinstance(self.test_loader, dict)
                    else self.test_loader
                )
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

            self.wandb_run.log(log_data)

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
        return {
            "best_model_path": str(self.best_model_path) if self.best_model_path else None,
            "log_dir": str(self.log_dir),
            "final_metrics": dict(self.final_metrics),
            "pipeline_metric_key": self.pipeline_metric_key,
            "pipeline_metric": self.pipeline_metric,
            "pipeline_metric_dataset": self.pipeline_metric_dataset,
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

            test_metrics = self.test(phase="Final_Test")
            self.final_metrics = test_metrics
            self.wandb_run.summary.update(test_metrics)
            metric_dataset, metric_value = self._resolve_pipeline_metric(test_metrics)
            if metric_value is not None:
                self.pipeline_metric = float(metric_value)
                self.pipeline_metric_dataset = metric_dataset
                self.wandb_run.log({self.pipeline_metric_key: self.pipeline_metric})
                self.wandb_run.summary[self.pipeline_metric_key] = self.pipeline_metric
                self.wandb_run.summary[
                    f"{self.pipeline_metric_key}_dataset"
                ] = self.pipeline_metric_dataset
                self.logger.info(
                    "Pipeline metric recorded: %s=%.6f (dataset=%s)",
                    self.pipeline_metric_key,
                    self.pipeline_metric,
                    self.pipeline_metric_dataset,
                )
            else:
                self.logger.warning(
                    "Pipeline metric key '%s' could not be resolved from final test metrics.",
                    self.pipeline_metric_key,
                )

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
        else:
            self.logger.warning(
                "Best model checkpoint not found. Final metrics and pipeline metric are unavailable."
            )

    def _resolve_pipeline_metric(
        self, test_metrics: Dict[str, float]
    ) -> Tuple[Optional[str], Optional[float]]:
        """Resolve pipeline metric from final test metrics.

        Priority:
        1) final_test/{metric_dataset}/dice
        2) final_test/{first data.test dataset}/dice when fallback policy is enabled
        """
        pipeline_cfg = self.cfg.get("pipeline", {}).get("distill", {})
        metric_dataset = pipeline_cfg.get("metric_dataset", "BUID")
        primary_key = f"final_test/{metric_dataset}/dice"

        if primary_key in test_metrics:
            return str(metric_dataset), float(test_metrics[primary_key])

        fallback_policy = pipeline_cfg.get("metric_fallback", "first_test_dataset")
        if fallback_policy != "first_test_dataset":
            return None, None

        test_datasets = self.cfg.get("data", {}).get("test", [])
        fallback_dataset = None
        if isinstance(test_datasets, str):
            fallback_dataset = test_datasets
        elif isinstance(test_datasets, (list, ListConfig)) and len(test_datasets) > 0:
            fallback_dataset = str(test_datasets[0])

        if fallback_dataset is None:
            return None, None

        fallback_key = f"final_test/{fallback_dataset}/dice"
        if fallback_key not in test_metrics:
            return None, None

        self.logger.warning(
            "Preferred dataset '%s' metric missing. Falling back to dataset '%s'.",
            metric_dataset,
            fallback_dataset,
        )
        return fallback_dataset, float(test_metrics[fallback_key])

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
