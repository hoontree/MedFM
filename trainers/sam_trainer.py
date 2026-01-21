"""
SAM (Segment Anything Model) Trainer

This module provides a trainer for SAM models with LoRA adaptation.
"""

from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm
from importlib import import_module
import wandb

from .base_trainer import BaseTrainer
from model.segment_anything import sam_model_registry
from utils.data_processing_seg import SegDatasetProcessor
from utils.sam_utils import DiceLoss


class SAMTrainer(BaseTrainer):
    """Trainer for SAM models."""

    def __init__(self, cfg):
        """Initialize SAM trainer."""
        super().__init__(cfg)

        # SAM-specific attributes
        self.ce_loss = None
        self.bce_loss = None
        self.dice_loss = None
        self.multimask_output = False
        self.img_size = cfg.model.img_size

    def _get_config_value(self, section: str, keys: list, default, cast_type=float):
        """Get config value with multiple key fallbacks."""
        cfg_section = self.cfg.get(section, {})
        for key in keys:
            if (value := cfg_section.get(key)) is not None:
                return cast_type(value)
        return default

    def _get_num_epochs(self) -> int:
        """Get number of training epochs."""
        return self._get_config_value('training', ['num_epochs', 'max_epochs'], 100, int)

    def _get_base_lr(self) -> float:
        """Get base learning rate."""
        return self._get_config_value('training', ['base_lr', 'lr'], 1e-4, float)

    def _get_dice_weight(self) -> float:
        """Get dice loss weight."""
        return self._get_config_value('training', ['dice_param', 'dice_weight'], 0.8, float)

    def _get_grad_clip_config(self) -> Dict:
        # Support both optimizer.grad_clip (legacy) and optimizer.gradient_clip (new)
        opt_cfg = self.cfg.get('optimizer', {})
        return opt_cfg.get('grad_clip', opt_cfg.get('gradient_clip', {}))

    def _get_n_gpus(self) -> int:
        """Get number of GPUs from config."""
        hw = self.cfg.get('hardware', {})
        # Try direct keys first
        for key in ['n_gpu', 'n_gpus']:
            if (value := hw.get(key)) is not None:
                return int(value)
        # Try gpu_ids list
        if (gpu_ids := hw.get('gpu_ids')) and isinstance(gpu_ids, (list, tuple)):
            return len(gpu_ids)
        return 1

    def _get_warmup_config(self) -> Dict[str, int | bool]:
        """Get warmup configuration with clear priority order."""
        training = self.cfg.get('training', {})
        scheduler = self.cfg.get('scheduler', {})
        
        # Priority 1: warmup_epochs (most specific)
        if (warmup_epochs := training.get('warmup_epochs')) and int(warmup_epochs) > 0:
            return {"enabled": True, "steps": int(warmup_epochs) * len(self.train_loader)}
        
        # Priority 2: warmup_period or warmup_iters
        warmup_steps = training.get('warmup_period') or scheduler.get('warmup_iters')
        if warmup_steps and int(warmup_steps) > 0:
            return {"enabled": True, "steps": int(warmup_steps)}
        
        # Priority 3: warmup flag only (no steps configured)
        if training.get('warmup', False):
            self.logger.warning("Warmup enabled but no steps configured. Disabling warmup.")
        
        return {"enabled": False, "steps": 0}

    def _create_model(self):
        """Create SAM model with LoRA or hybrid adapter."""
        # Get SAM type (vit_b, vit_l, vit_h)
        sam_type = self.cfg.model.get('sam_type', 'vit_b')
        if sam_type not in sam_model_registry:
            # Fallback: check if model name itself is a valid sam type
            if self.cfg.model.name in sam_model_registry:
                sam_type = self.cfg.model.name
            else:
                sam_type = 'vit_b'

        # Get checkpoint path
        checkpoint = self.cfg.model.get('sam_checkpoint', self.cfg.model.get('ckpt'))

        # Register SAM model
        sam, _ = sam_model_registry[sam_type](
            image_size=self.cfg.model.img_size,
            num_classes=self.cfg.data.num_classes,
            checkpoint=checkpoint,
            pixel_mean=[0, 0, 0],
            pixel_std=[1, 1, 1],
        )

        # Load LoRA/Adapter module
        module_path = self.cfg.model.get('module', 'model.sam_lora_image_encoder_mask_decoder')
        pkg = import_module(module_path)

        # Build kwargs for LoRA_Sam (supports both old and new config styles)
        model_kwargs = {}
        if self.cfg.model.get('adaptation_mode'):
            model_kwargs['adaptation_mode'] = self.cfg.model.adaptation_mode

        self.model = pkg.LoRA_Sam(sam, self.cfg.model.get('rank', 4), **model_kwargs).to(self.device)

        # Load LoRA checkpoint if provided (support both config key names)
        lora_ckpt = self.cfg.model.get('lora_checkpoint', self.cfg.model.get('lora_ckpt'))
        if lora_ckpt is not None:
            self.model.load_lora_parameters(lora_ckpt)
            self.logger.info(f"Loaded LoRA checkpoint: {lora_ckpt}")

        # Setup DataParallel
        if self._get_n_gpus() > 1:
            self.model = nn.DataParallel(self.model)
            
        self._setup_loss_functions()

        # Set multimask output
        self.multimask_output = self.cfg.get('multimask_output', False)

        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

    def _create_dataloaders(self):
        """Create data loaders."""
        self.train_loader, self.val_loader, self.test_loader = SegDatasetProcessor.build_data_loaders(self.cfg)
        self.logger.info(f"Train set size: {len(self.train_loader.dataset)}")
        self.logger.info(f"Val set size: {len(self.val_loader.dataset)}")
        
        # Log test set sizes
        if isinstance(self.test_loader, dict):
            total_test = sum(len(loader.dataset) for loader in self.test_loader.values())
            self.logger.info(f"Test set size (Total): {total_test}")
        for name, loader in self._iter_test_loaders():
            prefix = "  - " if isinstance(self.test_loader, dict) else ""
            self.logger.info(f"{prefix}{name}: {len(loader.dataset)}")

    def _create_optimizer(self):
        """Create optimizer."""
        base_lr = self._get_base_lr()

        optimizer_name = self.cfg.optimizer.get('name', 'SGD')

        if optimizer_name == "AdamW":
            self.optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=base_lr,
                betas=(0.9, 0.999),
                weight_decay=self.cfg.optimizer.get('weight_decay', 0.1)
            )
        else:
            self.optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=base_lr,
                momentum=0.9,
                weight_decay=0.0001
            )

        self.logger.info(f"Optimizer: {optimizer_name}, Base LR: {base_lr}")

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup and polynomial decay."""
        warmup_cfg = self._get_warmup_config()
        warmup_steps = warmup_cfg['steps']
        warmup_enabled = warmup_cfg['enabled']

        max_epochs = self._get_num_epochs()
        total_iters = max_epochs * len(self.train_loader)
        power = float(self.cfg.get('scheduler', {}).get('power', 0.9))

        # Minimum learning rate ratio (prevents LR from reaching 0)
        base_lr = self._get_base_lr()
        min_lr = float(self.cfg.get('scheduler', {}).get('min_lr', 1e-6))
        min_lr_ratio = min_lr / base_lr

        def lr_lambda(current_step: int) -> float:
            """Calculate learning rate multiplier for given step."""
            # Warmup phase: linear increase
            if warmup_enabled and warmup_steps > 0 and current_step < warmup_steps:
                return (current_step + 1) / warmup_steps

            # Polynomial decay phase
            shift_iter = current_step - warmup_steps if (warmup_enabled and warmup_steps > 0) else current_step
            shift_iter = min(max(0, shift_iter), total_iters)  # Clamp to valid range
            decay = (1.0 - shift_iter / total_iters) ** power
            return max(min_lr_ratio, decay)

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        self.logger.info(f"Scheduler: LambdaLR with warmup={warmup_enabled}, warmup_steps={warmup_steps}, max_iterations={total_iters}, power={power}, min_lr={min_lr}")
    def _get_current_lr(self) -> float:
        """Get current learning rate from optimizer."""
        return self.optimizer.param_groups[0]['lr']

    def _get_base_model(self):
        """Get base model (handle DataParallel wrapper)."""
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _iter_test_loaders(self):
        """Iterate over test loaders with names."""
        if isinstance(self.test_loader, dict):
            for name, loader in self.test_loader.items():
                yield name, loader
        else:
            yield "test", self.test_loader
            
    def _setup_loss_functions(self):
        """Setup loss functions based on number of classes."""
        # Setup loss functions
        self.ce_loss = CrossEntropyLoss()

        pos_weight = torch.tensor([5.0], device=self.device)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.dice_loss = DiceLoss(self.cfg.data.num_classes)

    def _calc_loss(self, outputs, label_batch, low_res_label_batch):
        """Calculate loss based on number of classes."""
        dice_weight = self._get_dice_weight()

        if self.cfg.data.num_classes == 2:
            # Binary segmentation
            logits = outputs['masks']
            target = label_batch.unsqueeze(1).float()

            # Ensure target and logits have the same resolution
            if logits.shape[-2:] != target.shape[-2:]:
                target = F.interpolate(target, size=logits.shape[-2:], mode='nearest')

            target = (target > 0.5).float()

            loss_ce = self.bce_loss(logits, target)
            loss_dice = self.dice_loss(logits, target)
        else:
            # Multi-class segmentation
            low_res_logits = outputs['low_res_logits']
            loss_ce = self.ce_loss(low_res_logits, low_res_label_batch.long())
            loss_dice = self.dice_loss(low_res_logits, low_res_label_batch, softmax=True)

        loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice

        return loss, loss_ce, loss_dice

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_dice_loss = 0.0

        train_pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self._get_num_epochs()}')

        for image_batch, label_batch, low_res_label_batch in train_pbar:
            image_batch = image_batch.to(self.device)
            label_batch = label_batch.to(self.device)
            low_res_label_batch = low_res_label_batch.to(self.device)

            # Forward pass
            outputs = self.model(image_batch, self.multimask_output, self.img_size)

            # Calculate loss
            loss, loss_ce, loss_dice = self._calc_loss(outputs, label_batch, low_res_label_batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            grad_clip_config = self._get_grad_clip_config()
            if grad_clip_config.get('enabled', False):
                max_norm = grad_clip_config.get('max_norm', 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

            self.optimizer.step()
            self.scheduler.step()

            # Get current learning rate
            lr = self._get_current_lr()

            # Update metrics
            total_loss += loss.item()
            total_ce_loss += loss_ce.item()
            total_dice_loss += loss_dice.item()

            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{lr:.6f}'
            })

            # Log to wandb (step-level metrics)
            if self.global_step % 10 == 0:
                wandb.log({
                    'step_train/loss': loss.item(),
                    'step_train/loss_ce': loss_ce.item(),
                    'step_train/loss_dice': loss_dice.item(),
                    'step_train/learning_rate': lr,
                    'global_step': self.global_step
                })

            self.global_step += 1

        # Calculate average losses
        num_batches = len(self.train_loader)
        metrics = {
            'loss': total_loss / num_batches,
            'loss_ce': total_ce_loss / num_batches,
            'loss_dice': total_dice_loss / num_batches,
        }

        return metrics

    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()

        val_metrics = self.evaluator.evaluate_model_sam(
            self.model,
            self.val_loader,
            self.device,
            self.cfg.data.num_classes,
            img_size=self.img_size
        )

        self.evaluator.print_metrics(val_metrics, phase='validation')

        return val_metrics

    def test(self) -> Dict[str, float]:
        """Test model and visualize predictions using cached inference results."""
        self.model.eval()

        test_metrics = {}
        # Store predictions for visualization (avoid redundant forward pass)
        predictions_cache = {}

        for name, loader in self._iter_test_loaders():
            if isinstance(self.test_loader, dict):
                self.logger.info(f"Testing on dataset: {name}")

            # Get metrics and predictions in one forward pass
            result = self.evaluator.evaluate_model_sam(
                self.model,
                loader,
                self.device,
                self.cfg.data.num_classes,
                img_size=self.img_size,
                return_predictions=True
            )
            metrics, images_list, preds_list, masks_list = result

            self.evaluator.print_metrics(metrics, phase=f'test_{name}' if isinstance(self.test_loader, dict) else 'test')

            # Store metrics with appropriate keys
            if isinstance(self.test_loader, dict):
                for k, v in metrics.items():
                    test_metrics[f"{name}/{k}"] = v
            else:
                test_metrics = metrics

            # Cache predictions for visualization
            predictions_cache[name] = (images_list, preds_list, masks_list)

        # Visualize predictions using cached results
        self._visualize_predictions(predictions_cache)

        return test_metrics

    def _visualize_predictions(self, predictions_cache: Dict = None):
        """Visualize test predictions using pre-computed results."""
        from utils.visualize import visualize_from_predictions

        # Create epoch-specific visualization directory
        vis_dir = self.exp_dir / "visualizations" / f"epoch_{self.current_epoch + 1}"
        num_vis_samples = self.cfg.get('visualization', {}).get('num_samples', None)

        sample_msg = "all" if num_vis_samples is None else f"{num_vis_samples}"
        self.logger.info(f"Generating {sample_msg} visualizations for epoch {self.current_epoch + 1}...")

        for name, loader in self._iter_test_loaders():
            save_dir = vis_dir / name if isinstance(self.test_loader, dict) else vis_dir
            phase_name = f"test_{name}" if isinstance(self.test_loader, dict) else "test"

            if predictions_cache and name in predictions_cache:
                # Use cached predictions (no additional forward pass)
                images_list, preds_list, masks_list = predictions_cache[name]
                visualize_from_predictions(
                    images_list,
                    preds_list,
                    masks_list,
                    self.cfg.data.num_classes,
                    save_dir,
                    num_samples=num_vis_samples,
                    phase_name=phase_name
                )
            else:
                # Fallback to original method if no cache
                from utils.visualize import visualize_predictions
                visualize_predictions(
                    self.model,
                    loader,
                    self.device,
                    self.cfg.data.num_classes,
                    save_dir,
                    num_samples=num_vis_samples,
                    model_type='sam',
                    img_size=self.img_size,
                    phase_name=phase_name
                )

        self.logger.info(f"Visualizations saved to {vis_dir}")

    def _save_model(self, path: Path):
        """Save SAM model (LoRA parameters)."""
        self._get_base_model().save_lora_parameters(str(path))

    def _load_checkpoint(self, path: Path):
        """Load SAM model checkpoint."""
        self.logger.info(f"Loading checkpoint: {path}")
        self._get_base_model().load_lora_parameters(str(path))
