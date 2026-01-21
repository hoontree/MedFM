"""
Single-Task Training for CA-SAM

각 task에 대해 독립적으로 Alignment Layer를 학습
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from typing import Optional

from .ca_sam import CASAM
from .losses import BCEDiceLoss, compute_iou, compute_boundary_iou, MetricsTracker


class SingleTaskTrainer:
    """
    Single-task trainer for CA-SAM
    
    각 task에 대해 Alignment Layer를 학습하고
    VAE를 학습하여 task-specific distribution 모델링
    """
    
    def __init__(
        self,
        model: CASAM,
        task_id: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        num_epochs: int = 24,
        device: str = 'cuda',
        save_dir: str = './checkpoints'
    ):
        self.model = model
        self.task_id = task_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = device
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Set training task
        self.model.set_training_task(task_id)
        
        # Loss function
        self.criterion = BCEDiceLoss().to(device)
        
        # Optimizer (only for current task's Alignment Layer)
        trainable_params = self.model.alignment_layers[task_id].parameters()
        self.optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
        
        # Metrics trackers
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        # Best validation IoU
        self.best_val_iou = 0.0
        
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        self.model.freeze_sam()  # Ensure SAM stays frozen
        self.train_metrics.reset()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Train]')
        
        for batch_idx, (images, masks, prompts) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            pred_logits = self.model(images, task_id=self.task_id, prompts=prompts)
            
            # Compute loss
            loss = self.criterion(pred_logits, masks)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                pred_probs = torch.sigmoid(pred_logits)
                iou = compute_iou(pred_probs, masks)
                biou = compute_boundary_iou(pred_probs, masks)
            
            # Update metrics
            self.train_metrics.update(loss.item(), iou, biou)
            
            # Update progress bar
            avg_loss, avg_iou, avg_biou = self.train_metrics.get_average()
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'iou': f'{avg_iou:.4f}',
                'biou': f'{avg_biou:.4f}'
            })
        
        return self.train_metrics.get_average()
    
    @torch.no_grad()
    def validate(self, epoch: int):
        """Validate on validation set"""
        if self.val_loader is None:
            return None, None, None
        
        self.model.eval()
        self.val_metrics.reset()
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Val]')
        
        for images, masks, prompts in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            pred_logits = self.model(images, task_id=self.task_id, prompts=prompts)
            
            # Compute loss
            loss = self.criterion(pred_logits, masks)
            
            # Compute metrics
            pred_probs = torch.sigmoid(pred_logits)
            iou = compute_iou(pred_probs, masks)
            biou = compute_boundary_iou(pred_probs, masks)
            
            # Update metrics
            self.val_metrics.update(loss.item(), iou, biou)
            
            # Update progress bar
            avg_loss, avg_iou, avg_biou = self.val_metrics.get_average()
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'iou': f'{avg_iou:.4f}',
                'biou': f'{avg_biou:.4f}'
            })
        
        return self.val_metrics.get_average()
    
    def train(self):
        """Full training loop"""
        print(f"\nTraining Task {self.task_id}")
        print("=" * 60)
        
        num_params = self.model.get_num_trainable_params(self.task_id)
        print(f"Trainable parameters: {num_params:,}")
        print("=" * 60)
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss, train_iou, train_biou = self.train_epoch(epoch)
            
            # Validate
            if self.val_loader is not None:
                val_loss, val_iou, val_biou = self.validate(epoch)
                
                print(f"\nEpoch {epoch+1}/{self.num_epochs}")
                print(f"  Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, BIoU: {train_biou:.4f}")
                print(f"  Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, BIoU: {val_biou:.4f}")
                
                # Save best model
                if val_iou > self.best_val_iou:
                    self.best_val_iou = val_iou
                    self.save_checkpoint(epoch, 'best')
                    print(f"  ✓ Best model saved (IoU: {val_iou:.4f})")
            else:
                print(f"\nEpoch {epoch+1}/{self.num_epochs}")
                print(f"  Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, BIoU: {train_biou:.4f}")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, f'epoch_{epoch+1}')
        
        # Save final model
        self.save_checkpoint(self.num_epochs - 1, 'final')
        print("\n" + "=" * 60)
        print(f"Training completed for Task {self.task_id}")
        print(f"Best validation IoU: {self.best_val_iou:.4f}")
        
    def save_checkpoint(self, epoch: int, tag: str):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.save_dir, 
            f'task_{self.task_id}_{tag}.pth'
        )
        
        torch.save({
            'epoch': epoch,
            'task_id': self.task_id,
            'alignment_layer_state': self.model.alignment_layers[self.task_id].state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_val_iou': self.best_val_iou
        }, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.alignment_layers[self.task_id].load_state_dict(
            checkpoint['alignment_layer_state']
        )
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.best_val_iou = checkpoint['best_val_iou']
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Best IoU: {self.best_val_iou:.4f}")
    
    def collect_encoder_features(self):
        """
        Collect encoder features for VAE training
        
        Returns:
            features: [N, C, H, W]
        """
        print(f"\nCollecting encoder features for Task {self.task_id}...")
        
        self.model.eval()
        all_features = []
        
        with torch.no_grad():
            for images, _, _ in tqdm(self.train_loader, desc='Collecting features'):
                images = images.to(self.device)
                
                # Get encoder output
                encoder_output = self.model.forward_encoder(images)
                all_features.append(encoder_output.cpu())
        
        all_features = torch.cat(all_features, dim=0)
        print(f"  Collected {len(all_features)} features with shape {all_features.shape}")
        
        return all_features


if __name__ == "__main__":
    print("=" * 60)
    print("Single-Task Trainer Test")
    print("=" * 60)
    
    # This would require actual SAM model and dataset
    # Just showing the interface here
    
    print("\nTrainer interface:")
    print("  1. Create CA-SAM model")
    print("  2. Add task: task_id = model.add_new_task()")
    print("  3. Create trainer: trainer = SingleTaskTrainer(model, task_id, ...)")
    print("  4. Train: trainer.train()")
    print("  5. Collect features: features = trainer.collect_encoder_features()")
    print("  6. Train VAE: model.train_vae_for_task(task_id, features)")
    print("  7. Calibrate threshold: model.calibrate_task_threshold(task_id, features)")
