"""
CA-SAM: Continual Alignment for SAM

전체 프레임워크를 통합하는 메인 모델:
1. Frozen SAM (encoder + decoder)
2. Task-specific Alignment Layers
3. VAE Router for task discrimination
4. OOD fallback mechanism
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import numpy as np

from .alignment_layer import AlignmentLayer, IdentityAlignmentLayer
from .vae_router import VAERouter, TaskVAE, calibrate_threshold


class CASAM(nn.Module):
    """
    Continual Alignment for SAM
    
    Main components:
    - Frozen SAM encoder & decoder
    - Multiple task-specific Alignment Layers
    - VAE Router for automatic task identification
    - Identity layer for OOD fallback
    """
    
    def __init__(
        self,
        sam_encoder: nn.Module,
        sam_decoder: nn.Module,
        encoder_output_dim: int = 256,
        alignment_hidden_dim: int = 256,
        alignment_num_blocks: int = 4,
        vae_latent_dim: int = 64,
        vae_beta: float = 16.5,
        attention_temperature: float = 1.0
    ):
        super().__init__()
        
        # Frozen SAM components
        self.sam_encoder = sam_encoder
        self.sam_decoder = sam_decoder
        
        # Freeze SAM
        for param in self.sam_encoder.parameters():
            param.requires_grad = False
        for param in self.sam_decoder.parameters():
            param.requires_grad = False
            
        self.sam_encoder.eval()
        self.sam_decoder.eval()
        
        # Alignment Layer configuration
        self.encoder_output_dim = encoder_output_dim
        self.alignment_hidden_dim = alignment_hidden_dim
        self.alignment_num_blocks = alignment_num_blocks
        
        # Task-specific Alignment Layers
        self.alignment_layers = nn.ModuleList()
        
        # Identity layer for OOD fallback
        self.identity_layer = IdentityAlignmentLayer()
        
        # VAE Router
        self.vae_router = VAERouter(
            input_dim=encoder_output_dim,
            latent_dim=vae_latent_dim,
            temperature=attention_temperature,
            beta=vae_beta
        )
        
        # Current task index (for training)
        self.current_task_id = -1
        
        # Training mode flag
        self.is_training_mode = False
        
    def freeze_sam(self):
        """Ensure SAM remains frozen"""
        self.sam_encoder.eval()
        self.sam_decoder.eval()
        for param in self.sam_encoder.parameters():
            param.requires_grad = False
        for param in self.sam_decoder.parameters():
            param.requires_grad = False
    
    def add_new_task(self):
        """
        Add new task with dedicated Alignment Layer and VAE
        
        Returns:
            task_id: New task ID
        """
        # Create new Alignment Layer
        alignment_layer = AlignmentLayer(
            in_channels=self.encoder_output_dim,
            hidden_channels=self.alignment_hidden_dim,
            num_blocks=self.alignment_num_blocks
        )
        self.alignment_layers.append(alignment_layer)
        
        # Create new VAE
        vae = self.vae_router.add_task_vae()
        
        task_id = len(self.alignment_layers) - 1
        self.current_task_id = task_id
        
        return task_id
    
    def set_training_task(self, task_id: int):
        """
        Set current task for training
        Only the specified task's Alignment Layer will be trainable
        """
        self.current_task_id = task_id
        self.is_training_mode = True
        
        # Freeze all alignment layers
        for layer in self.alignment_layers:
            for param in layer.parameters():
                param.requires_grad = False
                
        # Unfreeze current task's alignment layer
        if 0 <= task_id < len(self.alignment_layers):
            for param in self.alignment_layers[task_id].parameters():
                param.requires_grad = True
                
        # Ensure SAM stays frozen
        self.freeze_sam()
    
    def set_inference_mode(self):
        """Set model to inference mode"""
        self.is_training_mode = False
        self.eval()
        
    def forward_encoder(self, image: torch.Tensor, **kwargs):
        """
        Forward through frozen SAM encoder
        
        Args:
            image: Input image [B, 3, H, W]
            **kwargs: Additional arguments for SAM encoder
            
        Returns:
            encoder_output: [B, C, H', W']
        """
        with torch.no_grad():
            encoder_output = self.sam_encoder(image, **kwargs)
        return encoder_output
    
    def forward_alignment(self, encoder_output: torch.Tensor, task_id: Optional[int] = None):
        """
        Forward through Alignment Layer
        
        Args:
            encoder_output: SAM encoder output [B, C, H, W]
            task_id: Specific task ID (if None, use router)
            
        Returns:
            aligned_features: [B, C, H, W]
            selected_task_id: Used task ID
        """
        if task_id is None:
            # Use router for task discrimination
            if self.is_training_mode:
                task_id = self.current_task_id
            else:
                task_id, _ = self.vae_router.route_task(encoder_output)
        
        # Select alignment layer
        if task_id == -1:
            # OOD: use identity layer (frozen SAM)
            aligned_features = self.identity_layer(encoder_output)
        else:
            aligned_features = self.alignment_layers[task_id](encoder_output)
            
        return aligned_features, task_id
    
    def forward_decoder(self, aligned_features: torch.Tensor, **kwargs):
        """
        Forward through frozen SAM decoder
        
        Args:
            aligned_features: Aligned features [B, C, H, W]
            **kwargs: Additional arguments for SAM decoder (prompts, etc.)
            
        Returns:
            masks: Predicted masks
        """
        with torch.no_grad():
            masks = self.sam_decoder(aligned_features, **kwargs)
        return masks
    
    def forward(
        self, 
        image: torch.Tensor, 
        task_id: Optional[int] = None,
        return_task_id: bool = False,
        **decoder_kwargs
    ):
        """
        Full forward pass: image -> encoder -> alignment -> decoder -> masks
        
        Args:
            image: Input image [B, 3, H, W]
            task_id: Specific task ID (None for automatic routing)
            return_task_id: Whether to return selected task ID
            **decoder_kwargs: Arguments for decoder (prompts, etc.)
            
        Returns:
            masks: Predicted segmentation masks
            (optional) selected_task_id
        """
        # Encoder
        encoder_output = self.forward_encoder(image)
        
        # Alignment
        aligned_features, selected_task_id = self.forward_alignment(
            encoder_output, task_id=task_id
        )
        
        # Decoder
        masks = self.forward_decoder(aligned_features, **decoder_kwargs)
        
        if return_task_id:
            return masks, selected_task_id
        else:
            return masks
    
    def train_vae_for_task(
        self,
        task_id: int,
        train_features: torch.Tensor,
        num_epochs: int = 10,
        learning_rate: float = 5e-4,
        batch_size: int = 32
    ):
        """
        Train VAE for specific task
        
        Args:
            task_id: Task ID
            train_features: Training features [N, C, H, W]
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
        """
        vae = self.vae_router.task_vaes[task_id]
        vae.train()
        
        optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
        
        num_samples = len(train_features)
        
        for epoch in range(num_epochs):
            total_loss = 0
            total_recon = 0
            total_kl = 0
            num_batches = 0
            
            # Shuffle data
            indices = torch.randperm(num_samples)
            
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_features = train_features[batch_indices]
                
                # Extract global features
                global_features = self.vae_router.extract_global_feature(batch_features)
                
                # Compute loss
                elbo_loss, recon_loss, kl_loss = vae.compute_elbo(
                    global_features, 
                    beta=self.vae_router.beta
                )
                
                # Backward
                optimizer.zero_grad()
                elbo_loss.backward()
                optimizer.step()
                
                total_loss += elbo_loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            avg_recon = total_recon / num_batches
            avg_kl = total_kl / num_batches
            
            if (epoch + 1) % 2 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Loss: {avg_loss:.4f} "
                      f"(Recon: {avg_recon:.4f}, KL: {avg_kl:.4f})")
        
        vae.eval()
    
    def calibrate_task_threshold(
        self,
        task_id: int,
        train_features: torch.Tensor,
        percentile: float = 97
    ):
        """
        Calibrate threshold for task using K-fold CV
        
        Args:
            task_id: Task ID
            train_features: Training features [N, C, H, W]
            percentile: Threshold percentile (default: p97)
        """
        vae = self.vae_router.task_vaes[task_id]
        
        # Extract global features
        global_features = []
        for i in range(0, len(train_features), 32):
            batch = train_features[i:i+32]
            gf = self.vae_router.extract_global_feature(batch)
            global_features.append(gf)
        global_features = torch.cat(global_features, dim=0)
        
        # Calibrate threshold
        threshold = calibrate_threshold(
            vae, 
            global_features, 
            percentile=percentile,
            beta=self.vae_router.beta
        )
        
        self.vae_router.set_task_threshold(task_id, threshold)
        print(f"Task {task_id} threshold (p{percentile}): {threshold:.4f}")
        
        return threshold
    
    def get_num_trainable_params(self, task_id: Optional[int] = None):
        """
        Get number of trainable parameters
        
        Args:
            task_id: Specific task (None for all alignment layers)
        """
        if task_id is not None:
            return self.alignment_layers[task_id].get_num_params()
        else:
            return sum(layer.get_num_params() for layer in self.alignment_layers)


if __name__ == "__main__":
    print("=" * 60)
    print("CA-SAM Integration Test")
    print("=" * 60)
    
    # Mock SAM components
    class MockSAMEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 256, 1)
            
        def forward(self, x):
            # Simulate ViT-B output: [B, 256, 64, 64]
            x = nn.functional.interpolate(x, size=(64, 64))
            return self.conv(x)
    
    class MockSAMDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(256, 1, 1)
            
        def forward(self, x):
            # Simulate mask output: [B, 1, H, W]
            x = nn.functional.interpolate(x, size=(256, 256))
            return torch.sigmoid(self.conv(x))
    
    # Create CA-SAM model
    print("\nCreating CA-SAM model...")
    sam_encoder = MockSAMEncoder()
    sam_decoder = MockSAMDecoder()
    
    model = CASAM(
        sam_encoder=sam_encoder,
        sam_decoder=sam_decoder,
        encoder_output_dim=256,
        alignment_num_blocks=4,
        vae_latent_dim=64
    )
    
    print(f"✓ Model created")
    
    # Add 3 tasks
    print("\nAdding 3 tasks...")
    for i in range(3):
        task_id = model.add_new_task()
        num_params = model.get_num_trainable_params(task_id)
        print(f"  Task {task_id}: {num_params:,} parameters")
    
    # Test training mode
    print("\nTraining Task 0...")
    model.set_training_task(0)
    
    # Create dummy data
    images = torch.randn(2, 3, 1024, 1024)
    
    # Forward pass
    masks = model(images)
    print(f"  Input shape: {images.shape}")
    print(f"  Output shape: {masks.shape}")
    
    # Test inference with routing
    print("\nInference with automatic routing...")
    model.set_inference_mode()
    
    masks, task_id = model(images, return_task_id=True)
    print(f"  Selected task: {task_id}")
    print(f"  Output shape: {masks.shape}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
