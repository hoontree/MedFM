"""
Loss Functions for CA-SAM Training

1. Segmentation losses (Dice + BCE)
2. Boundary-aware losses
3. Metric computations (IoU, BIoU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted masks [B, 1, H, W] (after sigmoid)
            target: Ground truth masks [B, 1, H, W]
        """
        pred = pred.flatten(1)
        target = target.flatten(1)
        
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    Combined BCE and Dice Loss
    
    논문에서 사용하는 segmentation loss
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_logits: Raw predictions [B, 1, H, W] (before sigmoid)
            target: Ground truth [B, 1, H, W]
        """
        bce_loss = self.bce(pred_logits, target)
        
        pred_probs = torch.sigmoid(pred_logits)
        dice_loss = self.dice(pred_probs, target)
        
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return total_loss


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute Intersection over Union (IoU)
    
    Args:
        pred: Predicted masks [B, 1, H, W] (probabilities)
        target: Ground truth [B, 1, H, W]
        threshold: Binarization threshold
        
    Returns:
        IoU score
    """
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return iou.item()


def compute_boundary_iou(pred: torch.Tensor, target: torch.Tensor, 
                         threshold: float = 0.5, dilation: int = 5) -> float:
    """
    Compute Boundary IoU (BIoU)
    
    경계 영역에 대한 IoU를 계산하여 boundary quality 평가
    
    Args:
        pred: Predicted masks [B, 1, H, W]
        target: Ground truth [B, 1, H, W]
        threshold: Binarization threshold
        dilation: Dilation size for boundary extraction
        
    Returns:
        BIoU score
    """
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    
    # Extract boundaries using morphological operations
    kernel = torch.ones(1, 1, dilation, dilation, device=pred.device)
    
    # Dilate
    pred_dilated = F.conv2d(pred, kernel, padding=dilation//2)
    target_dilated = F.conv2d(target, kernel, padding=dilation//2)
    
    pred_dilated = (pred_dilated > 0).float()
    target_dilated = (target_dilated > 0).float()
    
    # Erode (by dilating the inverse)
    pred_eroded = 1 - F.conv2d(1 - pred, kernel, padding=dilation//2)
    target_eroded = 1 - F.conv2d(1 - target, kernel, padding=dilation//2)
    
    pred_eroded = (pred_eroded > 0).float()
    target_eroded = (target_eroded > 0).float()
    
    # Boundary = dilated - eroded
    pred_boundary = pred_dilated - pred_eroded
    target_boundary = target_dilated - target_eroded
    
    # Compute IoU on boundaries
    intersection = (pred_boundary * target_boundary).sum()
    union = pred_boundary.sum() + target_boundary.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    biou = intersection / union
    return biou.item()


class MetricsTracker:
    """
    Track training and evaluation metrics
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics"""
        self.total_loss = 0.0
        self.total_iou = 0.0
        self.total_biou = 0.0
        self.count = 0
        
    def update(self, loss: float, iou: float, biou: float):
        """Update metrics"""
        self.total_loss += loss
        self.total_iou += iou
        self.total_biou += biou
        self.count += 1
        
    def get_average(self) -> Tuple[float, float, float]:
        """Get average metrics"""
        if self.count == 0:
            return 0.0, 0.0, 0.0
        
        avg_loss = self.total_loss / self.count
        avg_iou = self.total_iou / self.count
        avg_biou = self.total_biou / self.count
        
        return avg_loss, avg_iou, avg_biou
    
    def __str__(self):
        avg_loss, avg_iou, avg_biou = self.get_average()
        return f"Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, BIoU: {avg_biou:.4f}"


if __name__ == "__main__":
    print("=" * 60)
    print("Loss Functions Test")
    print("=" * 60)
    
    # Create dummy data
    batch_size = 4
    height, width = 256, 256
    
    pred_logits = torch.randn(batch_size, 1, height, width)
    target = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    
    # Test BCE-Dice Loss
    print("\nTesting BCE-Dice Loss...")
    loss_fn = BCEDiceLoss()
    loss = loss_fn(pred_logits, target)
    print(f"  Loss: {loss.item():.4f}")
    
    # Test IoU
    print("\nTesting IoU...")
    pred_probs = torch.sigmoid(pred_logits)
    iou = compute_iou(pred_probs, target)
    print(f"  IoU: {iou:.4f}")
    
    # Test BIoU
    print("\nTesting BIoU...")
    biou = compute_boundary_iou(pred_probs, target)
    print(f"  BIoU: {biou:.4f}")
    
    # Test Metrics Tracker
    print("\nTesting Metrics Tracker...")
    tracker = MetricsTracker()
    for i in range(5):
        tracker.update(
            loss=torch.rand(1).item(),
            iou=torch.rand(1).item(),
            biou=torch.rand(1).item()
        )
    print(f"  {tracker}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
