"""
CA-SAM: Continual Alignment for SAM

Paper: "Continual Alignment for SAM: Rethinking Foundation Models for
        Medical Image Segmentation in Continual Learning"

Core components:
- AlignmentLayer: Lightweight feature distribution alignment between encoder and decoder
- VAERouter: Task-specific VAE for task discrimination and OOD detection
- CASAM: Complete continual learning framework for SAM

Usage:
    from model.ca_sam import CASAM, AlignmentLayer

    # Create CA-SAM model
    model = CASAM(sam_encoder, sam_decoder)

    # Add new task
    task_id = model.add_new_task()
    model.set_training_task(task_id)
"""

from .alignment_layer import AlignmentLayer, IdentityAlignmentLayer, CAResBlock
from .vae_router import VAERouter, TaskVAE, AttentionPooling, calibrate_threshold
from .ca_sam import CASAM
from .losses import (
    DiceLoss,
    BCEDiceLoss,
    compute_iou,
    compute_boundary_iou,
    MetricsTracker
)

__all__ = [
    # Core modules
    'CASAM',
    'AlignmentLayer',
    'IdentityAlignmentLayer',
    'CAResBlock',

    # VAE Router
    'VAERouter',
    'TaskVAE',
    'AttentionPooling',
    'calibrate_threshold',

    # Losses and metrics
    'DiceLoss',
    'BCEDiceLoss',
    'compute_iou',
    'compute_boundary_iou',
    'MetricsTracker',
]
