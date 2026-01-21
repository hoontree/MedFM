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
import logging
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from .base_trainer import BaseTrainer


@dataclass
class SAM3Config:
    """Configuration dataclass for SAM3 training."""
    # Model
    checkpoint_path: Optional[str] = None
    bpe_path: str = "model/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    enable_segmentation: bool = True

    # Training
    max_epochs: int = 20
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

        # Setup paths
        self._setup_paths()

    def _setup_paths(self):
        """Setup experiment directories."""
        model_name = self.cfg.model.get('name', 'sam3')
        dataset_name = self.cfg.data.get('name', 'custom')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        logs_root = Path(self.cfg.get('output', {}).get('dir', 'logs'))
        self.exp_dir = logs_root / model_name / dataset_name / timestamp
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
        sam3_cfg = self.cfg.get('sam3', {})
        training_cfg = self.cfg.get('training', {})

        # Build SAM3-compatible config
        config = {
            'paths': {
                'checkpoint_path': sam3_cfg.get('checkpoint_path'),
                'bpe_path': sam3_cfg.get('bpe_path', 'model/sam3/assets/bpe_simple_vocab_16e6.txt.gz'),
                'experiment_log_dir': str(self.exp_dir),
            },
            'scratch': {
                'enable_segmentation': sam3_cfg.get('enable_segmentation', True),
                'resolution': sam3_cfg.get('resolution', 1008),
                'train_batch_size': training_cfg.get('batch_size', 1),
                'val_batch_size': 1,
                'num_train_workers': training_cfg.get('num_workers', 4),
                'num_val_workers': 2,
                'max_data_epochs': training_cfg.get('max_epochs', 20),
                'gradient_accumulation_steps': training_cfg.get('gradient_accumulation_steps', 1),
                'lr_scale': sam3_cfg.get('lr_scale', 0.1),
                'train_norm_mean': [0.5, 0.5, 0.5],
                'train_norm_std': [0.5, 0.5, 0.5],
                'val_norm_mean': [0.5, 0.5, 0.5],
                'val_norm_std': [0.5, 0.5, 0.5],
            },
            'trainer': {
                '_target_': 'sam3.train.trainer.Trainer',
                'max_epochs': training_cfg.get('max_epochs', 20),
                'accelerator': 'cuda',
                'seed_value': self.cfg.get('hardware', {}).get('seed', 123),
                'val_epoch_freq': sam3_cfg.get('val_epoch_freq', 1),
                'mode': self.cfg.get('mode', 'train'),
                'gradient_accumulation_steps': training_cfg.get('gradient_accumulation_steps', 1),
                'distributed': {
                    'backend': 'nccl',
                    'find_unused_parameters': True,
                    'gradient_as_bucket_view': True,
                },
                'optim': {
                    'amp': {
                        'enabled': sam3_cfg.get('amp_enabled', True),
                        'amp_dtype': sam3_cfg.get('amp_dtype', 'bfloat16'),
                    },
                    'optimizer': {
                        '_target_': 'torch.optim.AdamW',
                    },
                    'gradient_clip': {
                        '_target_': 'sam3.train.optim.optimizer.GradientClipper',
                        'max_norm': 0.1,
                        'norm_type': 2,
                    },
                },
                'checkpoint': {
                    'save_dir': str(self.ckpt_dir),
                    'save_freq': training_cfg.get('save_interval', 5),
                },
                'logging': {
                    'log_dir': str(self.log_dir),
                    'log_freq': sam3_cfg.get('log_freq', 10),
                    'tensorboard_writer': {
                        '_target_': 'sam3.train.utils.logger.make_tensorboard_logger',
                        'log_dir': str(self.exp_dir / 'tensorboard'),
                        'flush_secs': 120,
                        'should_log': True,
                    },
                    'wandb_writer': None,
                },
                'model': {
                    '_target_': 'sam3.model_builder.build_sam3_image_model',
                    'bpe_path': sam3_cfg.get('bpe_path', 'model/sam3/assets/bpe_simple_vocab_16e6.txt.gz'),
                    'device': 'cpus',
                    'eval_mode': False,
                    'enable_segmentation': sam3_cfg.get('enable_segmentation', True),
                    'checkpoint_path': sam3_cfg.get('checkpoint_path'),
                },
                'data': self._build_data_config(),
                'loss': self._build_loss_config(),
                'meters': self._build_meters_config(),
            },
            'launcher': {
                'num_nodes': 1,
                'gpus_per_node': len(self.cfg.get('hardware', {}).get('gpu_ids', [0])),
                'experiment_log_dir': str(self.exp_dir),
            },
            'submitit': {
                'use_cluster': False,
                'port_range': [10000, 65000],
            },
        }

        return OmegaConf.create(config)

    def _build_data_config(self) -> Dict:
        """Build SAM3 data configuration from unified config."""
        data_cfg = self.cfg.get('data', {})
        sam3_cfg = self.cfg.get('sam3', {})

        # Check if using custom data config
        if 'train_dataset' in sam3_cfg:
            return sam3_cfg.get('data_config', {})

        # Build default data config for medical image segmentation
        return {
            'train': {
                '_target_': 'sam3.train.data.torch_dataset.TorchDataset',
                'dataset': {
                    '_target_': 'sam3.train.data.sam3_image_dataset.Sam3ImageDataset',
                    'img_folder': data_cfg.get('train_path', ''),
                    'ann_file': data_cfg.get('train_annotation', ''),
                    'load_segmentation': sam3_cfg.get('enable_segmentation', True),
                    'training': True,
                },
                'shuffle': True,
                'batch_size': self.cfg.get('training', {}).get('batch_size', 1),
                'num_workers': self.cfg.get('training', {}).get('num_workers', 4),
                'pin_memory': True,
                'drop_last': True,
            },
            'val': {
                '_target_': 'sam3.train.data.torch_dataset.TorchDataset',
                'dataset': {
                    '_target_': 'sam3.train.data.sam3_image_dataset.Sam3ImageDataset',
                    'img_folder': data_cfg.get('val_path', ''),
                    'ann_file': data_cfg.get('val_annotation', ''),
                    'load_segmentation': sam3_cfg.get('enable_segmentation', True),
                    'training': False,
                },
                'shuffle': False,
                'batch_size': 1,
                'num_workers': 2,
                'pin_memory': True,
                'drop_last': False,
            },
        }

    def _build_loss_config(self) -> Dict:
        """Build SAM3 loss configuration."""
        sam3_cfg = self.cfg.get('sam3', {})
        enable_segmentation = sam3_cfg.get('enable_segmentation', True)

        if enable_segmentation:
            return {
                'all': {
                    '_target_': 'sam3.train.loss.sam3_loss.Sam3LossWrapper',
                    'matcher': {
                        '_target_': 'sam3.train.matcher.BinaryHungarianMatcherV2',
                        'focal': True,
                        'cost_class': 2.0,
                        'cost_bbox': 5.0,
                        'cost_giou': 2.0,
                        'alpha': 0.25,
                        'gamma': 2,
                    },
                    'loss_fns_find': [
                        {
                            '_target_': 'sam3.train.loss.loss_fns.Boxes',
                            'weight_dict': {
                                'loss_bbox': 5.0,
                                'loss_giou': 2.0,
                            },
                        },
                        {
                            '_target_': 'sam3.train.loss.loss_fns.IABCEMdetr',
                            'weight_dict': {
                                'loss_ce': 20.0,
                                'presence_loss': 20.0,
                            },
                            'pos_weight': 10.0,
                            'alpha': 0.25,
                            'gamma': 2,
                            'use_presence': True,
                        },
                        {
                            '_target_': 'sam3.train.loss.loss_fns.Masks',
                            'weight_dict': {
                                'loss_mask': 200.0,
                                'loss_dice': 10.0,
                            },
                        },
                    ],
                },
                'default': {
                    '_target_': 'sam3.train.loss.sam3_loss.DummyLoss',
                },
            }
        else:
            return {
                'all': {
                    '_target_': 'sam3.train.loss.sam3_loss.DummyLoss',
                },
                'default': {
                    '_target_': 'sam3.train.loss.sam3_loss.DummyLoss',
                },
            }

    def _build_meters_config(self) -> Dict:
        """Build SAM3 meters configuration."""
        return {
            'val': None,  # Can be customized for specific evaluation metrics
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
        config_path = self.exp_dir / 'sam3_config.yaml'
        with open(config_path, 'w') as f:
            OmegaConf.save(sam3_config, f)

        self.logger.info(f"SAM3 config saved to {config_path}")
        self.logger.info(f"Experiment directory: {self.exp_dir}")

        # Run training
        self._run_single_node(sam3_config)

    def _run_single_node(self, cfg: DictConfig):
        """Run single-node training."""
        num_gpus = cfg.launcher.gpus_per_node
        main_port = random.randint(
            cfg.submitit.port_range[0],
            cfg.submitit.port_range[1]
        )

        if num_gpus == 1:
            self._single_proc_run(0, main_port, cfg, 1)
        else:
            torch.multiprocessing.set_start_method("spawn", force=True)
            torch.multiprocessing.start_processes(
                self._single_proc_run,
                args=(main_port, cfg, num_gpus),
                nprocs=num_gpus,
                start_method="spawn"
            )

    def _single_proc_run(self, local_rank: int, main_port: int, cfg: DictConfig, world_size: int):
        """Single GPU process."""
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(main_port)
        os.environ["RANK"] = str(local_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)

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

    def __init__(self, cfg: DictConfig):
        """Initialize SAM3 trainer adapter."""
        super().__init__(cfg)

        self.sam3_cfg = cfg.get('sam3', {})
        self.resolution = self.sam3_cfg.get('resolution', 1008)
        self.enable_segmentation = self.sam3_cfg.get('enable_segmentation', True)

        # Loss components
        self.criterion = None
        self.scaler = None

    def _create_model(self):
        """Create SAM3 model."""
        try:
            from model.sam3.model_builder import build_sam3_image_model
        except ImportError:
            self.logger.error("Failed to import SAM3. Make sure model/sam3 is properly set up.")
            raise

        bpe_path = self.sam3_cfg.get('bpe_path', 'model/sam3/assets/bpe_simple_vocab_16e6.txt.gz')
        checkpoint_path = self.sam3_cfg.get('checkpoint_path')

        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            device='cpus',  # Will be moved to GPU later
            eval_mode=False,
            enable_segmentation=self.enable_segmentation,
            checkpoint_path=checkpoint_path,
        )

        self.model = self.model.to(self.device)

        # Setup AMP
        amp_enabled = self.sam3_cfg.get('amp_enabled', True)
        if amp_enabled:
            self.scaler = torch.amp.GradScaler(self.device)

        # Setup criterion (simplified for adapter)
        self._setup_criterion()

        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"SAM3 Model loaded")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

    def _setup_criterion(self):
        """Setup loss criterion."""
        try:
            from model.sam3.train.loss.sam3_loss import Sam3LossWrapper, DummyLoss
            from model.sam3.train.matcher import BinaryHungarianMatcherV2
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

                self.criterion = Sam3LossWrapper(
                    matcher=matcher,
                    loss_fns_find=[
                        Boxes(weight_dict={'loss_bbox': 5.0, 'loss_giou': 2.0}),
                        IABCEMdetr(
                            weight_dict={'loss_ce': 20.0, 'presence_loss': 20.0},
                            pos_weight=10.0,
                            alpha=0.25,
                            gamma=2,
                            use_presence=True,
                        ),
                        Masks(weight_dict={'loss_mask': 200.0, 'loss_dice': 10.0}),
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
        from utils.sam_utils import DiceLoss

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(self.cfg.data.num_classes)
        self.criterion = None  # Will use manual loss computation

    def _create_dataloaders(self):
        """Create data loaders."""
        # Check if using SAM3's native data loading
        if 'data_config' in self.sam3_cfg:
            self._create_sam3_dataloaders()
        else:
            # Use unified framework's data loading
            from utils.data_processing_seg import SegDatasetProcessor

            self.train_loader, self.val_loader, self.test_loader = \
                SegDatasetProcessor.build_data_loaders(self.cfg)

            self.logger.info(f"Train set size: {len(self.train_loader.dataset)}")
            self.logger.info(f"Val set size: {len(self.val_loader.dataset)}")

    def _create_sam3_dataloaders(self):
        """Create SAM3 native data loaders."""
        from hydra.utils import instantiate

        data_config = self.sam3_cfg.get('data_config', {})

        if 'train' in data_config:
            train_dataset = instantiate(data_config['train'])
            self.train_loader = train_dataset.get_loader(epoch=0)

        if 'val' in data_config:
            val_dataset = instantiate(data_config['val'])
            self.val_loader = val_dataset.get_loader(epoch=0)

    def _create_optimizer(self):
        """Create optimizer with parameter groups."""
        base_lr = self.cfg.training.get('base_lr', 1e-4)
        weight_decay = self.cfg.training.get('weight_decay', 0.1)

        # Create parameter groups for different learning rates
        vision_params = []
        transformer_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if 'vision_backbone' in name:
                vision_params.append(param)
            elif 'language_backbone' in name:
                # Skip language backbone for medical imaging
                continue
            else:
                transformer_params.append(param)

        lr_scale = self.sam3_cfg.get('lr_scale', 0.1)

        param_groups = [
            {'params': transformer_params, 'lr': base_lr * lr_scale},
            {'params': vision_params, 'lr': base_lr * lr_scale * 0.3},  # Lower for backbone
        ]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        self.logger.info(f"Optimizer: AdamW")
        self.logger.info(f"Base LR: {base_lr * lr_scale}, Vision LR: {base_lr * lr_scale * 0.3}")

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        # Use warmup + inverse square root decay (similar to SAM3)
        total_steps = self.cfg.training.get('max_epochs', 20) * len(self.train_loader)
        warmup_steps = min(500, total_steps // 10)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            else:
                return max(0.1, (warmup_steps / (step + 1)) ** 0.5)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        amp_enabled = self.sam3_cfg.get('amp_enabled', True)
        amp_dtype = getattr(torch, self.sam3_cfg.get('amp_dtype', 'bfloat16'))

        train_pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch + 1}/{self.cfg.training.get("max_epochs", 20)}'
        )

        for batch in train_pbar:
            # Handle different batch formats
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                # Standard format: (images, labels, ...)
                images = batch[0].to(self.device)
                labels = batch[1].to(self.device)
            elif isinstance(batch, dict):
                # SAM3 native format
                images = batch.get('image', batch.get('img_batch')).to(self.device)
                labels = batch.get('label', batch.get('find_targets'))
            else:
                self.logger.warning(f"Unknown batch format: {type(batch)}")
                continue

            self.optimizer.zero_grad()

            # Forward pass with AMP
            with torch.amp.autocast(device_type='cuda', enabled=amp_enabled, dtype=amp_dtype):
                outputs = self.model(images)

                if self.criterion is not None:
                    loss = self.criterion(outputs, labels)
                    if isinstance(loss, dict):
                        loss = loss.get('core_loss', sum(loss.values()))
                else:
                    # Simple loss computation
                    loss = self._compute_simple_loss(outputs, labels)

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
            current_lr = self.optimizer.param_groups[0]['lr']
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.6f}'
            })

            self.global_step += 1

        return {
            'loss': total_loss / max(num_batches, 1),
        }

    def _compute_simple_loss(self, outputs, labels):
        """Compute simple loss when SAM3 criterion is not available."""
        if hasattr(outputs, 'masks') or (isinstance(outputs, dict) and 'masks' in outputs):
            pred_masks = outputs['masks'] if isinstance(outputs, dict) else outputs.masks
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

    def validate(self, epoch: int) -> Dict[str, float]:  # noqa: ARG002
        """Validate model."""
        _ = epoch  # Required by BaseTrainer interface but not used here
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        # Use evaluator if available
        if hasattr(self, 'evaluator') and self.evaluator is not None:
            try:
                # Try SAM-style evaluation
                val_metrics = self.evaluator.evaluate_model_sam(
                    self.model,
                    self.val_loader,
                    self.device,
                    self.cfg.data.get('num_classes', 2),
                    img_size=self.resolution
                )
                return val_metrics
            except Exception as e:
                self.logger.warning(f"SAM-style evaluation failed: {e}")

        # Fallback to simple validation
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    images = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                elif isinstance(batch, dict):
                    images = batch.get('image', batch.get('img_batch')).to(self.device)
                    labels = batch.get('label', batch.get('find_targets'))
                else:
                    continue

                outputs = self.model(images)

                if self.criterion is not None:
                    loss = self.criterion(outputs, labels)
                    if isinstance(loss, dict):
                        loss = loss.get('core_loss', sum(loss.values()))
                else:
                    loss = self._compute_simple_loss(outputs, labels)

                total_loss += loss.item()
                num_batches += 1

        return {
            'loss': total_loss / max(num_batches, 1),
            'Dice': 0.0,  # Placeholder - implement proper metric computation
        }

    def test(self) -> Dict[str, float]:
        """Test model."""
        self.model.eval()

        # Similar to validate
        return self.validate(self.current_epoch)

    def _save_model(self, path: Path):
        """Save model checkpoint."""
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'epoch': self.current_epoch,
            'global_step': self.global_step,
        }

        if self.scaler is not None:
            checkpoint['scaler'] = self.scaler.state_dict()

        torch.save(checkpoint, str(path))

    def _load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        self.logger.info(f"Loading checkpoint: {path}")
        checkpoint = torch.load(str(path), map_location=self.device)

        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)


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
    parser.add_argument("--mode", type=str, default="train", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.mode = args.mode

    orchestrator = SAM3Orchestrator(cfg)
    orchestrator.run()
