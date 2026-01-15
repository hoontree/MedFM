import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from tensorboardX import SummaryWriter
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import wandb

from utils.data_processing_seg import SegDatasetProcessor
from utils.evaluate import Evaluator_seg
from utils.logger import setup_logger
from utils.schedule import build_scheduler, get_lr_decay_param_groups
from utils.sam_utils import DiceLoss

# SAM model imports
from model.sam_lora_image_encoder import LoRA_Sam as LoRA_Sam_ImageEncoder
from model.sam_lora_image_encoder_mask_decoder import LoRA_Sam as LoRA_Sam_Full
from model.segment_anything import sam_model_registry
from importlib import import_module

# TinyUSFM model imports
from model.tinyusfm_seg import SegmentationModel as TinyUSFM_Seg



import json

class FeatureAdapter(nn.Module):
    """Adapter to match student feature dimension to teacher."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) # Optional non-linearity
        )

    def forward(self, x):
        return self.conv(x)


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_exp_tags(cfg: DictConfig) -> list:
    """Create experiment tags based on hyperparameters."""
    exp_tags = []

    # Basic tags
    if hasattr(cfg.training, 'batch_size'):
        exp_tags.append(f"bs{cfg.training.batch_size}")

    if hasattr(cfg.training, 'lr'):
        if cfg.training.lr != 0.01:
            exp_tags.append(f"lr{cfg.training.lr}")

    # Distillation specific tags
    if hasattr(cfg, 'distillation'):
        if hasattr(cfg.distillation, 'temperature'):
            exp_tags.append(f"T{cfg.distillation.temperature}")
        if hasattr(cfg.distillation, 'alpha'):
            exp_tags.append(f"a{cfg.distillation.alpha}")
        if hasattr(cfg.distillation, 'beta'):
            exp_tags.append(f"b{cfg.distillation.beta}")

    return exp_tags


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining:
    1. Task loss (segmentation loss with ground truth)
    2. Distillation loss (KL divergence between teacher and student logits)
    3. Feature distillation loss (MSE between intermediate features)
    """

    def __init__(self, temperature=4.0, alpha=0.7, beta=0.3, gamma=0.0, num_classes=2):
        """
        Args:
            temperature: Temperature for softening probability distributions
            alpha: Weight for task loss (ground truth)
            beta: Weight for distillation loss (soft targets from teacher)
            gamma: Weight for feature distillation loss
            num_classes: Number of segmentation classes
        """
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes

        # Task losses
        if num_classes == 2:
            self.task_criterion = nn.BCEWithLogitsLoss()
        else:
            self.task_criterion = nn.CrossEntropyLoss()

        self.dice_loss = DiceLoss(num_classes)
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()

    def forward(self, student_logits, teacher_logits, targets,
                student_features=None, teacher_features=None):
        """
        Args:
            student_logits: Student model output logits [B, C, H, W]
            teacher_logits: Teacher model output logits [B, C, H, W]
            targets: Ground truth labels [B, H, W] or [B, 1, H, W] for binary
            student_features: Optional intermediate features from student
            teacher_features: Optional intermediate features from teacher
        """
        # 1. Task loss (hard targets)
        if self.num_classes == 2:
            task_loss = self.task_criterion(student_logits, targets.unsqueeze(1).float())
            dice_loss = self.dice_loss(student_logits, targets.unsqueeze(1).float())
        else:
            task_loss = self.task_criterion(student_logits, targets.long())
            dice_loss = self.dice_loss(student_logits, targets, softmax=True)

        total_task_loss = 0.5 * task_loss + 0.5 * dice_loss

        # 2. Distillation loss (soft targets)
        # Resize teacher logits to match student if needed
        if teacher_logits.shape != student_logits.shape:
            teacher_logits = F.interpolate(
                teacher_logits,
                size=student_logits.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        # Apply temperature scaling and compute KL divergence
        if student_logits.shape[1] == 1:
            # Binary segmentation case: expand to 2 channels for KL divergence
            # softmax([0, x]) is equivalent to [1-sigmoid(x), sigmoid(x)]
            student_logits_expanded = torch.cat([torch.zeros_like(student_logits), student_logits], dim=1)
            teacher_logits_expanded = torch.cat([torch.zeros_like(teacher_logits), teacher_logits], dim=1)
            
            student_soft = F.log_softmax(student_logits_expanded / self.temperature, dim=1)
            teacher_soft = F.softmax(teacher_logits_expanded / self.temperature, dim=1)
        else:
            student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
            teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)

        distill_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)

        # Normalize by spatial dimensions to match task loss scale (per-pixel average)
        # batchmean reduction sums over H and W, so we divide by number of pixels
        num_pixels = student_logits.shape[-2] * student_logits.shape[-1]
        distill_loss = distill_loss / num_pixels

        # 3. Feature distillation loss (if features provided)
        feature_loss = 0.0
        if student_features is not None and teacher_features is not None and self.gamma > 0:
            # Match feature dimensions if needed
            if student_features.shape != teacher_features.shape:
                teacher_features = F.interpolate(
                    teacher_features,
                    size=student_features.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            feature_loss = self.mse_loss(student_features, teacher_features)

        # Total loss
        total_loss = (
            self.alpha * total_task_loss +
            self.beta * distill_loss +
            self.gamma * feature_loss
        )

        return total_loss, total_task_loss, distill_loss, feature_loss


def load_teacher_model(cfg, device):
    """Load finetuned SAM teacher model."""
    # Initialize SAM architecture
    sam, img_embedding_size = sam_model_registry[cfg.teacher.model_name](
        image_size=cfg.teacher.img_size,
        num_classes=cfg.data.num_classes,
        checkpoint=cfg.teacher.sam_checkpoint,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
    )

    # Wrap with LoRA
    pkg = import_module(cfg.teacher.module)
    model = pkg.LoRA_Sam(sam, cfg.teacher.rank).to(device)

    # Load finetuned LoRA weights
    if cfg.teacher.lora_checkpoint is None:
        raise ValueError("Teacher LoRA checkpoint must be provided!")

    model.load_lora_parameters(cfg.teacher.lora_checkpoint)

    # Set to evaluation mode
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


def load_student_model(cfg, device):
    """Load pretrained TinyUSFM student model."""
    model = TinyUSFM_Seg(cfg.data.num_classes)

    # Load pretrained weights if provided
    if cfg.student.pretrained and cfg.student.checkpoint:
        checkpoint = torch.load(cfg.student.checkpoint, map_location="cpu")
        new_state_dict = {
            k.replace('model.', 'backbone.'): v
            for k, v in checkpoint.items()
            if k.startswith('model.')
        }
        load_info = model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded student checkpoint from {cfg.student.checkpoint}")
        if load_info.missing_keys:
            print(f"Missing keys: {load_info.missing_keys}")

    model = model.to(device)
    return model


def visualize_distillation(teacher_model, student_model, test_loader, device,
                          num_classes, teacher_img_size, save_dir, num_samples=10):
    """Visualize teacher vs student predictions by overlaying predicted masks on input images."""
    teacher_model.eval()
    student_model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    sample_count = 0
    wandb_images = []

    with torch.no_grad():
        for batch_idx, (images, masks, low_res_masks) in enumerate(tqdm(test_loader, desc="Visualizing distillation")):
            images, masks = images.to(device), masks.to(device)

            # Get predictions from both models
            teacher_outputs = teacher_model(images, False, teacher_img_size)
            teacher_logits = teacher_outputs['masks']
            student_logits = student_model(images)

            # Convert to predictions
            if num_classes == 2:
                teacher_preds = (torch.sigmoid(teacher_logits) > 0.5).float()
                student_preds = (torch.sigmoid(student_logits) > 0.5).float()
            else:
                teacher_preds = torch.argmax(torch.softmax(teacher_logits, dim=1), dim=1, keepdim=True)
                student_preds = torch.argmax(torch.softmax(student_logits, dim=1), dim=1, keepdim=True)

            # Process each image in the batch
            for i in range(images.size(0)):
                if sample_count >= num_samples:
                    if wandb_images:
                        wandb.log({"distillation/predictions": wandb_images})
                    return

                # Convert tensors to numpy
                img = images[i].cpu().numpy()
                teacher_pred = teacher_preds[i].cpu().numpy()
                student_pred = student_preds[i].cpu().numpy()
                gt_mask = masks[i].cpu().numpy()

                # Handle image channels
                if img.shape[0] == 3:  # RGB
                    img = img.transpose(1, 2, 0)
                    img = (img - img.min()) / (img.max() - img.min())
                elif img.shape[0] == 1:  # Grayscale
                    img = img[0]
                    img = (img - img.min()) / (img.max() - img.min())

                # Handle mask dimensions
                if num_classes == 2:
                    teacher_pred = teacher_pred[0] if teacher_pred.ndim == 3 else teacher_pred
                    student_pred = student_pred[0] if student_pred.ndim == 3 else student_pred

                # Create visualization
                fig, axes = plt.subplots(2, 4, figsize=(20, 10))

                # Row 1: Original components
                axes[0, 0].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
                axes[0, 0].set_title('Input Image', fontsize=14)
                axes[0, 0].axis('off')

                axes[0, 1].imshow(gt_mask, cmap='jet', alpha=0.7)
                axes[0, 1].set_title('Ground Truth', fontsize=14)
                axes[0, 1].axis('off')

                axes[0, 2].imshow(teacher_pred, cmap='jet', alpha=0.7)
                axes[0, 2].set_title('Teacher (SAM)', fontsize=14)
                axes[0, 2].axis('off')

                axes[0, 3].imshow(student_pred, cmap='jet', alpha=0.7)
                axes[0, 3].set_title('Student (TinyUSFM)', fontsize=14)
                axes[0, 3].axis('off')

                # Row 2: Overlays and difference
                axes[1, 0].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
                axes[1, 0].imshow(gt_mask, cmap='jet', alpha=0.4)
                axes[1, 0].set_title('Input + GT', fontsize=14)
                axes[1, 0].axis('off')

                axes[1, 1].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
                axes[1, 1].imshow(teacher_pred, cmap='jet', alpha=0.4)
                axes[1, 1].set_title('Input + Teacher', fontsize=14)
                axes[1, 1].axis('off')

                axes[1, 2].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
                axes[1, 2].imshow(student_pred, cmap='jet', alpha=0.4)
                axes[1, 2].set_title('Input + Student', fontsize=14)
                axes[1, 2].axis('off')

                # Difference map
                diff = np.abs(teacher_pred - student_pred)
                axes[1, 3].imshow(diff, cmap='hot')
                axes[1, 3].set_title('Teacher-Student Difference', fontsize=14)
                axes[1, 3].axis('off')

                plt.tight_layout()

                # Save figure
                save_path = save_dir / f"sample_{sample_count:03d}.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                wandb_images.append(wandb.Image(str(save_path), caption=f"Sample {sample_count}"))
                plt.close()

                sample_count += 1

    # Log remaining images to wandb
    if wandb_images:
        wandb.log({"distillation/predictions": wandb_images})


@hydra.main(version_base=None, config_path="config", config_name="distill")
def main(cfg: DictConfig):
    """Main knowledge distillation training function."""
    # Set environment
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cfg.hardware.gpu_ids))
    set_seed(cfg.hardware.seed)

    # Setup directories
    logs_root = Path(cfg.get('output', {}).get('dir', 'logs'))
    model_name = cfg.student.model_name
    dataset_name = cfg.data.name
    train_mode = "distill"

    exp_tags = create_exp_tags(cfg)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir_name = timestamp + ("_" + "_".join(exp_tags) if exp_tags else "")

    # Final experiment directory
    log_dir = logs_root / model_name / dataset_name / train_mode / exp_dir_name
    log_dir.mkdir(parents=True, exist_ok=True)

    models_dir = log_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = log_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger(str(log_dir / "distill.log"))
    logger.info("KNOWLEDGE DISTILLATION: SAM (Teacher) -> TinyUSFM (Student)")
    logger.info(f"Teacher: {cfg.teacher.model_name} with LoRA")
    logger.info(f"Student: TinyUSFM")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Log directory: {log_dir}")

    # Initialize wandb
    wandb.init(
        entity=cfg.get('wandb', {}).get('entity', 'hheo'),
        project=cfg.get('wandb', {}).get('project', 'TinyUSFM').lower(),
        name=f"distill_{cfg.teacher.model_name}_to_tinyusfm_{timestamp}",
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=str(log_dir)
    )

    # Save config
    config_file = log_dir / 'config.yaml'
    with open(config_file, 'w') as f:
        OmegaConf.save(cfg, f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader = SegDatasetProcessor.build_data_loaders(cfg)
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    if isinstance(test_loader, dict):
        total_test = sum(len(loader.dataset) for loader in test_loader.values())
        logger.info(f"Test set size (Total): {total_test}")
        for name, loader in test_loader.items():
            logger.info(f"  - {name}: {len(loader.dataset)}")
    else:
        logger.info(f"Test set size: {len(test_loader.dataset)}")

    # Load teacher model
    logger.info(f"Loading teacher model ({cfg.teacher.model_name}) from {cfg.teacher.lora_checkpoint}...")
    teacher = load_teacher_model(cfg, device)
    teacher_params = sum(p.numel() for p in teacher.parameters())

    # Load student model
    logger.info(f"Loading student model ({cfg.student.model_name})...")
    student = load_student_model(cfg, device)
    student_params = sum(p.numel() for p in student.parameters())
    
    logger.info("-" * 40)
    logger.info(f"Teacher params: {teacher_params:,}")
    logger.info(f"Student params: {student_params:,}")
    logger.info(f"Compression:    {teacher_params / student_params:.2f}x")
    logger.info("-" * 40)

    # Setup Feature Adapter if distillation is enabled
    adapter = None
    if cfg.distillation.get('gamma', 0.0) > 0:
        # Student: 48 channels (from neck), Teacher: 256 channels (SAM)
        adapter = FeatureAdapter(in_channels=48, out_channels=256).to(device)
        logger.info("Feature Adapter initialized: 48 -> 256 channels")
        
    # Setup optimizer
    if cfg.optimizer.name == 'AdamW':
        param_groups = get_lr_decay_param_groups(
            model=student,
            base_lr=cfg.training.lr,
            weight_decay=cfg.optimizer.weight_decay,
            num_layers=12,
            layer_decay=0.8
        )
        if adapter is not None:
             param_groups.append({
                "params": adapter.parameters(),
                "lr": cfg.training.lr,
                "weight_decay": cfg.optimizer.weight_decay
            })
        optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999))
    elif cfg.optimizer.name == 'Adam':
        params = list(student.parameters())
        if adapter is not None:
            params += list(adapter.parameters())
        optimizer = optim.Adam(
            params,
            lr=cfg.training.lr,
            weight_decay=cfg.optimizer.weight_decay
        )
    else:
        params = list(student.parameters())
        if adapter is not None:
            params += list(adapter.parameters())
        optimizer = optim.SGD(
            params,
            lr=cfg.training.lr,
            momentum=0.9,
            weight_decay=cfg.optimizer.weight_decay
        )

    # Setup scheduler
    use_reduce_on_plateau = cfg.get('scheduler', {}).get('use_reduce_on_plateau', False)
    if use_reduce_on_plateau:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max',
            factor=cfg.scheduler.get('factor', 0.5),
            patience=cfg.scheduler.get('patience', 5),
            min_lr=cfg.scheduler.get('min_lr', 1e-7),
            verbose=True
        )
    else:
        scheduler = build_scheduler(optimizer, cfg)

    # Setup distillation loss
    kd_loss_fn = KnowledgeDistillationLoss(
        temperature=cfg.distillation.temperature,
        alpha=cfg.distillation.alpha,
        beta=cfg.distillation.beta,
        gamma=cfg.distillation.get('gamma', 0.0),
        num_classes=cfg.data.num_classes
    )

    logger.info(f"Distillation loss weights: alpha={cfg.distillation.alpha}, "
                f"beta={cfg.distillation.beta}, temperature={cfg.distillation.temperature}")

    # Setup tensorboard
    writer = SummaryWriter(str(log_dir / "tensorboard"))
    evaluator = Evaluator_seg()

    # Training loop
    best_dice = 0.0
    best_model_path = None
    early_stopping_counter = 0
    patience = cfg.training.get('early_stopping', {}).get('patience', 30)

    logger.info("Starting knowledge distillation training...")

    for epoch in range(cfg.training.num_epochs):
        # ============ Training Phase ============
        student.train()
        if adapter is not None:
            adapter.train()
            
        running_loss = 0.0
        running_task_loss = 0.0
        running_distill_loss = 0.0
        running_feature_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.num_epochs} [Train]")
        limit_train_batches = cfg.training.get('limit_train_batches', None)

        for batch_idx, (images, masks, low_res_masks) in enumerate(train_pbar):
            if limit_train_batches is not None and batch_idx >= limit_train_batches:
                break
            images = images.to(device)
            masks = masks.to(device).float()
            low_res_masks = low_res_masks.to(device)

            # Teacher forward (no gradient)
            with torch.no_grad():
                teacher_outputs = teacher(images, False, cfg.teacher.img_size)
                teacher_logits = teacher_outputs['masks']
                teacher_features = teacher_outputs.get('image_embeddings', None)

            # Student forward
            if adapter is not None:
                student_logits, student_features_raw = student(images, return_features=True)
                student_features = adapter(student_features_raw)
            else:
                student_logits = student(images)
                student_features = None

            # Compute distillation loss
            loss, task_loss, distill_loss, feature_loss = kd_loss_fn(
                student_logits, teacher_logits, masks,
                student_features=student_features,
                teacher_features=teacher_features
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if cfg.optimizer.get('gradient_clip', {}).get('enabled', False):
                max_norm = cfg.optimizer.gradient_clip.get('max_norm', 1.0)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm)

            optimizer.step()

            # Update metrics
            running_loss += loss.item() * images.size(0)
            running_task_loss += task_loss.item() * images.size(0)
            running_distill_loss += distill_loss.item() * images.size(0)
            if feature_loss != 0.0:
                running_feature_loss += feature_loss.item() * images.size(0)

            # Update progress bar
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'task': f"{task_loss.item():.4f}",
                'distill': f"{distill_loss.item():.4f}"
            })

        # Epoch metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_task_loss = running_task_loss / len(train_loader.dataset)
        epoch_distill_loss = running_distill_loss / len(train_loader.dataset)
        epoch_feature_loss = running_feature_loss / len(train_loader.dataset) if running_feature_loss > 0 else 0.0

        # ============ Validation Phase ============
        student.eval()
        if adapter is not None:
            adapter.eval()
            
        val_metrics = evaluator.evaluate_model(student, val_loader, device, cfg.data.num_classes)

        current_lr = optimizer.param_groups[0]['lr']

        # Consolidated logging
        status_msg = (
            f"Epoch {epoch+1}/{cfg.training.num_epochs} | "
            f"Loss: {epoch_loss:.4f} (Task: {epoch_task_loss:.4f}, Distill: {epoch_distill_loss:.4f}) | "
            f"Dice: {val_metrics['Dice']:.4f} | "
            f"HD95: {val_metrics['HD95']:.2f} | "
            f"LR: {current_lr:.6e}"
        )
        
        # Highlight if best
        if val_metrics['Dice'] > best_dice:
            logger.info(f"★ {status_msg}")
        else:
            logger.info(f"  {status_msg}")

        # Tensorboard logging
        writer.add_scalar('Loss/train_total', epoch_loss, epoch + 1)
        writer.add_scalar('Loss/train_task', epoch_task_loss, epoch + 1)
        writer.add_scalar('Loss/train_distill', epoch_distill_loss, epoch + 1)
        if epoch_feature_loss > 0:
            writer.add_scalar('Loss/train_feature', epoch_feature_loss, epoch + 1)
        writer.add_scalar('Dice/val', val_metrics['Dice'], epoch + 1)
        writer.add_scalar('HD95/val', val_metrics['HD95'], epoch + 1)
        writer.add_scalar('LearningRate', current_lr, epoch + 1)

        # Wandb logging
        wandb.log({
            'epoch': epoch + 1,
            'train/loss_total': epoch_loss,
            'train/loss_task': epoch_task_loss,
            'train/loss_distill': epoch_distill_loss,
            'train/loss_feature': epoch_feature_loss,
            'val/dice': val_metrics['Dice'],
            'val/hd95': val_metrics['HD95'],
            'val/pixel_acc': val_metrics['PixelAcc'],
            'learning_rate': current_lr
        }, step=epoch + 1)

        # Visualize predictions periodically
        if (epoch + 1) % 10 == 0:
            epoch_vis_dir = vis_dir / f"epoch_{epoch+1:03d}"
            logger.info(f"Generating visualizations for epoch {epoch+1}...")
            visualize_distillation(
                teacher, student, val_loader, device,
                cfg.data.num_classes, cfg.teacher.img_size,
                epoch_vis_dir, num_samples=5
            )

        # Learning rate scheduling
        if use_reduce_on_plateau:
            scheduler.step(val_metrics['Dice'])
        elif epoch >= cfg.training.get('warmup_epochs', 0):
            scheduler.step()

        # Save best model
        if val_metrics['Dice'] > best_dice:
            best_dice = val_metrics['Dice']
            model_path = models_dir / f"best_epoch{epoch+1}_dice{best_dice:.4f}.pth"
            
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': student.state_dict(),
                'dice': best_dice,
            }
            if adapter is not None:
                save_dict['adapter_state_dict'] = adapter.state_dict()
                
            torch.save(save_dict, str(model_path))
            best_model_path = model_path
            logger.info(f"Saved best model and adapter to {model_path}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # Early stopping
        if early_stopping_counter >= patience:
            logger.info(f"\nEarly stopping triggered at epoch {epoch + 1}")
            logger.info(f"Best validation Dice: {best_dice:.4f}")
            break

        # Periodic checkpoint
        if (epoch + 1) % 20 == 0:
            checkpoint_path = models_dir / f"checkpoint_epoch{epoch+1}.pth"
            save_dict = {'model_state_dict': student.state_dict()}
            if adapter is not None:
                save_dict['adapter_state_dict'] = adapter.state_dict()
            torch.save(save_dict, str(checkpoint_path))

    writer.close()

    # Summary dictionary for automation
    summary = {
        'best_val_dice': float(best_dice),
        'best_epoch': int(best_model_path.stem.split('_')[1].replace('epoch', '')) if best_model_path else 0,
        'train_mode': train_mode,
        'teacher': cfg.teacher.model_name,
        'student': cfg.student.model_name
    }

    # ============ Final Testing ============
    if best_model_path is not None:
        logger.info("\n" + "#" * 40)
        logger.info(f"FINAL TESTING: {best_model_path.name}")
        logger.info("#" * 40)

        student.load_state_dict(torch.load(str(best_model_path)))
        student.eval()
        if adapter is not None: 
            # Note: Adapter state is not currently saved/loaded with model checkpoint
            # If adapter is needed for inference (it's not for segmentation), we would need to save it.
            # But feature distillation is only for training, so we don't need adapter for evaluation.
            adapter.eval()

        # Handle multiple test loaders
        if isinstance(test_loader, dict):
            for name, loader in test_loader.items():
                logger.info(f"\nTesting on {name}...")
                test_metrics = evaluator.evaluate_model(student, loader, device, cfg.data.num_classes)
                evaluator.print_metrics(test_metrics, phase=f'test_{name}')

                # Log to wandb
                wandb.log({
                    f'test_{name}/dice': test_metrics['Dice'],
                    f'test_{name}/hd95': test_metrics['HD95'],
                    f'test_{name}/pixel_acc': test_metrics['PixelAcc']
                })
                
                # Update summary
                summary[f'test_{name}_dice'] = float(test_metrics['Dice'])
                summary[f'test_{name}_hd95'] = float(test_metrics['HD95'])
                
                # Update visual results for each
                num_vis_samples = cfg.get('visualization', {}).get('num_samples', 10)
                test_vis_dir = vis_dir / "test" / name
                visualize_distillation(
                    teacher, student, loader, device,
                    cfg.data.num_classes, cfg.teacher.img_size,
                    test_vis_dir, num_samples=num_vis_samples
                )
                logger.info(f"Visualizations for {name} saved to {test_vis_dir}")

        else:
            test_metrics = evaluator.evaluate_model(student, test_loader, device, cfg.data.num_classes)
            evaluator.print_metrics(test_metrics, phase='test')

            # Log test results
            wandb.log({
                'test/dice': test_metrics['Dice'],
                'test/hd95': test_metrics['HD95'],
                'test/pixel_acc': test_metrics['PixelAcc']
            })

            # Update summary
            summary['test_dice'] = float(test_metrics['Dice'])
            summary['test_hd95'] = float(test_metrics['HD95'])

            # Final visualizations
            logger.info("Generating final test visualizations...")
            num_vis_samples = cfg.get('visualization', {}).get('num_samples', 10)
            test_vis_dir = vis_dir / "test"
            visualize_distillation(
                teacher, student, test_loader, device,
                cfg.data.num_classes, cfg.teacher.img_size,
                test_vis_dir, num_samples=num_vis_samples
            )
            logger.info(f"Visualizations saved to {test_vis_dir}")

    # Save summary to JSON
    summary_path = log_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Summary saved to {summary_path}")

    wandb.finish()
    logger.info("\n" + "=" * 80)
    logger.info("Knowledge Distillation Training Completed!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
