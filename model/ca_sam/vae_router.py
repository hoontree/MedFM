"""
VAE Router Implementation for CA-SAM

Task-specific VAE를 학습하여:
1. 각 task의 feature distribution 모델링
2. ELBO 기반 task discrimination
3. OOD detection 및 fallback
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class AttentionPooling(nn.Module):
    """
    Parameter-free attention pooling mechanism
    
    공간적으로 중요한 feature에 가중치를 부여하여 global feature 추출
    L2 norm 기반으로 attention weight 계산
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, x):
        """
        Args:
            x: Feature map [B, C, H, W]
            
        Returns:
            Global feature vector [B, C]
        """
        B, C, H, W = x.shape
        
        # Compute L2 norm for each spatial location
        l2_norm = torch.norm(x, p=2, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Flatten spatial dimensions
        l2_norm = l2_norm.view(B, 1, H * W)  # [B, 1, H*W]
        
        # Softmax with temperature
        attention_weights = F.softmax(l2_norm / (C * self.temperature), dim=-1)  # [B, 1, H*W]
        
        # Reshape features
        x_flat = x.view(B, C, H * W)  # [B, C, H*W]
        
        # Weighted sum
        global_feature = torch.bmm(x_flat, attention_weights.transpose(1, 2))  # [B, C, 1]
        global_feature = global_feature.squeeze(-1)  # [B, C]
        
        return global_feature


class TaskVAE(nn.Module):
    """
    Task-specific Variational Autoencoder
    
    각 task의 feature distribution을 학습하고
    ELBO를 통해 task similarity 측정
    """
    def __init__(
        self,
        input_dim: int = 256,
        latent_dim: int = 64,
        hidden_dims: list = [128, 96]
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder: input -> hidden -> latent (mu, logvar)
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(inplace=True)
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder: latent -> hidden -> output
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(inplace=True)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        """
        Encode input to latent distribution parameters
        
        Returns:
            mu, logvar
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        Decode latent variable to reconstruction
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        Forward pass through VAE
        
        Returns:
            reconstruction, mu, logvar
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def compute_elbo(self, x, beta: float = 16.5):
        """
        Compute ELBO (Evidence Lower Bound) loss
        
        ELBO = Reconstruction Loss + β * KL Divergence
        
        Args:
            x: Input feature [B, D]
            beta: KL divergence weight (논문: 16.5)
            
        Returns:
            elbo_loss, recon_loss, kl_loss
        """
        reconstruction, mu, logvar = self.forward(x)
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x, reduction='mean')
        
        # KL divergence: KL(q(z|x) || p(z))
        # where p(z) = N(0, I) and q(z|x) = N(mu, sigma^2)
        # KL = 0.5 * sum(mu^2 + sigma^2 - 1 - log(sigma^2))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)  # Average over batch
        
        # Total ELBO loss
        elbo_loss = recon_loss + beta * kl_loss
        
        return elbo_loss, recon_loss, kl_loss


class VAERouter(nn.Module):
    """
    VAE-based Task Router
    
    여러 task-specific VAE를 관리하고
    ELBO 기반으로 task discrimination 수행
    """
    def __init__(
        self,
        input_dim: int = 256,
        latent_dim: int = 64,
        temperature: float = 1.0,
        beta: float = 16.5
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Attention pooling for feature extraction
        self.attention_pooling = AttentionPooling(temperature=temperature)
        
        # Task-specific VAEs (동적으로 추가됨)
        self.task_vaes = nn.ModuleList()
        
        # Task thresholds for OOD detection (calibrated per task)
        self.task_thresholds = []
        
    def add_task_vae(self, vae: Optional[TaskVAE] = None):
        """새로운 task VAE 추가"""
        if vae is None:
            vae = TaskVAE(
                input_dim=self.input_dim,
                latent_dim=self.latent_dim
            )
        self.task_vaes.append(vae)
        return vae
    
    def set_task_threshold(self, task_idx: int, threshold: float):
        """특정 task의 threshold 설정"""
        while len(self.task_thresholds) <= task_idx:
            self.task_thresholds.append(None)
        self.task_thresholds[task_idx] = threshold
    
    def extract_global_feature(self, encoder_output):
        """
        SAM encoder output에서 global feature 추출
        
        Args:
            encoder_output: [B, C, H, W]
            
        Returns:
            global_feature: [B, C]
        """
        return self.attention_pooling(encoder_output)
    
    def compute_task_scores(self, global_feature):
        """
        모든 task에 대한 ELBO score 계산
        
        Returns:
            task_scores: [num_tasks]
        """
        scores = []
        for vae in self.task_vaes:
            vae.eval()
            with torch.no_grad():
                elbo_loss, _, _ = vae.compute_elbo(global_feature, beta=self.beta)
                scores.append(elbo_loss.item())
        return scores
    
    def route_task(self, encoder_output, return_scores: bool = False):
        """
        Task routing with OOD fallback
        
        Args:
            encoder_output: SAM encoder output [B, C, H, W]
            return_scores: ELBO score들을 반환할지 여부
            
        Returns:
            task_id: 선택된 task ID (OOD인 경우 -1)
            confidence: task에 대한 confidence (낮을수록 높음)
        """
        # Extract global feature
        global_feature = self.extract_global_feature(encoder_output)
        
        # Compute ELBO scores for all tasks
        task_scores = self.compute_task_scores(global_feature)
        
        # Find task with minimum ELBO (highest likelihood)
        best_task_id = int(np.argmin(task_scores))
        best_score = task_scores[best_task_id]
        
        # Check if OOD using threshold
        if self.task_thresholds[best_task_id] is not None:
            threshold = self.task_thresholds[best_task_id]
            if best_score > threshold:
                # OOD detected -> fallback to frozen SAM
                best_task_id = -1
        
        if return_scores:
            return best_task_id, best_score, task_scores
        else:
            return best_task_id, best_score


def calibrate_threshold(vae: TaskVAE, features: torch.Tensor, 
                       percentile: float = 97, beta: float = 16.5):
    """
    K-fold cross-validation으로 task threshold 계산
    
    Args:
        vae: Task-specific VAE
        features: Training features [N, D]
        percentile: Threshold percentile (논문: p97)
        beta: KL weight
        
    Returns:
        threshold: ELBO threshold
    """
    vae.eval()
    elbo_scores = []
    
    with torch.no_grad():
        for i in range(0, len(features), 32):  # Batch processing
            batch = features[i:i+32]
            elbo, _, _ = vae.compute_elbo(batch, beta=beta)
            elbo_scores.append(elbo.item())
    
    threshold = np.percentile(elbo_scores, percentile)
    return threshold


if __name__ == "__main__":
    print("=" * 60)
    print("VAE Router Test")
    print("=" * 60)
    
    # Test configuration
    batch_size = 4
    channels = 256
    height, width = 64, 64
    
    # Simulate SAM encoder output
    encoder_output = torch.randn(batch_size, channels, height, width)
    
    # Create VAE Router
    router = VAERouter(input_dim=channels, latent_dim=64, beta=16.5)
    
    # Add 3 tasks
    print("\nAdding 3 tasks...")
    for i in range(3):
        vae = router.add_task_vae()
        print(f"  Task {i+1}: {sum(p.numel() for p in vae.parameters()):,} parameters")
    
    # Extract global feature
    print("\nExtracting global features...")
    global_feature = router.extract_global_feature(encoder_output)
    print(f"  Encoder output shape: {encoder_output.shape}")
    print(f"  Global feature shape: {global_feature.shape}")
    
    # Test routing (without thresholds)
    print("\nTask routing (without thresholds)...")
    task_id, score, all_scores = router.route_task(encoder_output, return_scores=True)
    print(f"  Selected task: {task_id}")
    print(f"  ELBO score: {score:.4f}")
    print(f"  All scores: {[f'{s:.4f}' for s in all_scores]}")
    
    # Set thresholds
    print("\nSetting thresholds...")
    for i in range(3):
        threshold = 0.5 + i * 0.1  # Dummy thresholds
        router.set_task_threshold(i, threshold)
        print(f"  Task {i}: threshold = {threshold:.4f}")
    
    # Test routing with thresholds (simulate OOD)
    print("\nTask routing with thresholds (high ELBO -> OOD)...")
    # Simulate high ELBO by using random features
    ood_output = torch.randn(batch_size, channels, height, width) * 10
    task_id, score = router.route_task(ood_output)
    print(f"  Selected task: {task_id} {'(OOD fallback)' if task_id == -1 else ''}")
    print(f"  ELBO score: {score:.4f}")
