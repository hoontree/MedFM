import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from pathlib import Path
from omegaconf import OmegaConf
import hydra
import random
from tqdm import tqdm

from utils.logger import setup_logger
from utils.data_processing_seg import SegDatasetProcessor
from model.tinyusfm_seg import SegmentationModel
from model.sam_lora_image_encoder_mask_decoder import LoRA_Sam
from model.segment_anything import sam_model_registry
import torch.nn as nn

class FeatureAdapter(nn.Module):
    """Adapter to match student feature dimension to teacher."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
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

def load_student_model(cfg, device):
    """Load TinyUSFM student model and optional adapter."""
    model = SegmentationModel(num_classes=cfg.data.num_classes)
    adapter = FeatureAdapter(48, 256)
    
    if cfg.get('student_checkpoint'):
        checkpoint = torch.load(cfg.student_checkpoint, map_location='cpu')
        
        # Load student model
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
            
        # Load adapter if available
        if 'adapter_state_dict' in checkpoint:
            adapter.load_state_dict(checkpoint['adapter_state_dict'])
            print("Loaded Feature Adapter weights.")
        else:
            print("Warning: Feature Adapter weights not found in checkpoint. Using random initialization.")
            
    model = model.to(device)
    adapter = adapter.to(device)
    model.eval()
    adapter.eval()
    return model, adapter

def load_teacher_model(cfg, device):
    """Load finetuned SAM teacher model."""
    sam, _ = sam_model_registry[cfg.teacher.model_name](
        image_size=cfg.teacher.img_size,
        num_classes=cfg.data.num_classes,
        checkpoint=cfg.teacher.sam_checkpoint,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
    )
    lora_rank = cfg.teacher.get('rank', 4)
    model = LoRA_Sam(sam, lora_rank).to(device)
    if cfg.teacher.get('lora_checkpoint'):
        model.load_lora_parameters(cfg.teacher.lora_checkpoint)
    model.eval()
    return model

def run_tsne(student_feats, teacher_feats, save_path, n_samples=1000):
    """
    Run t-SNE on joint features after dimension alignment.
    """
    print("Running t-SNE...")
    
    # Randomly sample patches
    if len(student_feats) > n_samples:
        indices = np.random.choice(len(student_feats), n_samples, replace=False)
        student_feats = student_feats[indices]
        teacher_feats = teacher_feats[indices]
        
    combined = np.concatenate([student_feats, teacher_feats], axis=0)
    labels = ["Student"] * len(student_feats) + ["Teacher"] * len(teacher_feats)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(combined)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embedded[:,0], y=embedded[:,1], hue=labels, alpha=0.6)
    plt.title("Joint Feature Space t-SNE (Aligned)")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved joint t-SNE plot to {save_path}")

def visualize_attention(image, student_attn, teacher_attn, save_path):
    """
    Visualize attention maps.
    image: [3, H, W] tensor
    student_attn: [Head, N, N]
    teacher_attn: [Head, N, N]
    """
    # Student: Last layer attention of CLS token (index 0) to all patches (1:)
    # shape: [Head, N+1, N+1] where N+1 is token count
    # Average over heads
    s_attn = student_attn.mean(0) # [N+1, N+1]
    # CLS token is at 0. Attention FROM CLS TO patches -> s_attn[0, 1:]
    # Or patches TO CLS -> s_attn[1:, 0]? Usually CLS gathers info, so CLS to others.
    # In ViT, attention matrix is usually softmax(Q @ K^T). Row i are attention weights from Query i to Keys.
    # So row 0 is CLS token query attending to all keys.
    s_map = s_attn[0, 1:] # [N]
    
    # Reshape to grid
    grid_size = int(np.sqrt(len(s_map)))
    s_map = s_map.reshape(grid_size, grid_size).cpu().numpy()
    
    # Teacher: No CLS token. Visualize attention of CENTER patch.
    t_attn = teacher_attn.mean(0) # [N, N]
    # Center index
    center_idx = len(t_attn) // 2 + int(np.sqrt(len(t_attn))) // 2
    # Attention FROM center patch TO all patches
    t_map = t_attn[center_idx].cpu().numpy()
    grid_size_t = int(np.sqrt(len(t_map)))
    t_map = t_map.reshape(grid_size_t, grid_size_t)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original Image
    img_np = image.permute(1, 2, 0).cpu().numpy()
    # Denormalize? (Assume ImageNet norm)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    
    # Student Heatmap
    # Resize map to image size
    s_map_resized = F.interpolate(torch.tensor(s_map).unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear').squeeze().numpy()
    axes[1].imshow(img_np)
    axes[1].imshow(s_map_resized, cmap='jet', alpha=0.5)
    axes[1].set_title("Student Attention (CLS)")
    
    # Teacher Heatmap
    t_map_resized = F.interpolate(torch.tensor(t_map).unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear').squeeze().numpy()
    axes[2].imshow(img_np)
    axes[2].imshow(t_map_resized, cmap='jet', alpha=0.5)
    axes[2].set_title("Teacher Attention (Center)")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


@hydra.main(version_base=None, config_path="config", config_name="distill")
def main(cfg):
    # Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cfg.hardware.gpu_ids))
    set_seed(cfg.hardware.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    output_dir = Path("logs/visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Data
    _, val_loader, _ = SegDatasetProcessor.build_data_loaders(cfg)
    
    # Models
    student, adapter = load_student_model(cfg, device)
    teacher = load_teacher_model(cfg, device)
    
    # Collect Features
    student_feats_all = []
    teacher_feats_all = []
    
    # Visualize Attention for n samples
    viz_indices = list(range(10)) # First 10
    
    print("Extracting features and visualizing attention...")
    with torch.no_grad():
        for i, (images, masks, _) in enumerate(tqdm(val_loader)):
            if i * images.shape[0] > 100: # Limit to 100 images for t-SNE features
                break
            
            images = images.to(device)
            
            # Forward Student
            _, s_feat_raw = student(images, return_features=True)
            s_feat = adapter(s_feat_raw)
            
            # Get Attention (Last layer)
            # MAEBackbone -> blocks -> Attention
            s_attn = student.backbone.blocks[-1].attn.last_attn # [B, Heads, N, N]
            
            # Forward Teacher
            # SAM image encoder
            t_feat = teacher.sam.image_encoder(images) # [B, 256, 64, 64] -> Wait, input 224? 
            # If input 224, patch size 16 -> 14x14.
            # Let's verify shape.
            
            # Get Attention
            t_attn_flat = teacher.sam.image_encoder.blocks[-1].attn.last_attn # [B*Heads, N, N]
            num_heads = teacher.sam.image_encoder.blocks[-1].attn.num_heads
            t_attn = t_attn_flat.view(images.shape[0], num_heads, t_attn_flat.shape[-1], t_attn_flat.shape[-1])
                        
            # Collect patch features (Center crop or random? or All?)
            # Flatten spatial: [B, C, H, W] -> [B, C, N] -> [B*N, C]
            B, C, H, W = s_feat.shape
            s_f_flat = s_feat.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
            
            B, C, H, W = t_feat.shape
            t_f_flat = t_feat.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
            
            student_feats_all.append(s_f_flat)
            teacher_feats_all.append(t_f_flat)
            
            # Visualize Attention for selected samples
            if i < 1: # Just first batch
                for b in range(min(len(images), 5)):
                    save_path = output_dir / f"attn_viz_{i}_{b}.png"
                    visualize_attention(images[b], s_attn[b], t_attn[b], save_path)
                    
    # t-SNE
    s_all = np.concatenate(student_feats_all, axis=0)
    t_all = np.concatenate(teacher_feats_all, axis=0)
    
    run_tsne(s_all, t_all, output_dir / "tsne_B.png")
    
if __name__ == "__main__":
    main()
