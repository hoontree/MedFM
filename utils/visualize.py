import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import wandb
from tqdm import tqdm
from typing import Optional, Union, Dict, List

def visualize_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    save_dir: Union[Path, str],
    num_samples: Optional[int] = None,
    model_type: str = 'default',
    img_size: Optional[int] = None,
    phase_name: str = "test"
):
    """
    Unified visualization function for various segmentation models.

    Args:
        model: The torch model to evaluate.
        dataloader: DataLoader for images and masks.
        device: Device to run the model on.
        num_classes: Number of segmentation classes.
        save_dir: Directory to save visualization images.
        num_samples: Number of samples to visualize. If None, visualize all samples.
        model_type: Type of model ('default', 'sam', 'segformer').
        img_size: Image size for SAM model.
        phase_name: Name of the phase (e.g., 'test', 'val') for WandB logging.
    """
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    sample_count = 0
    wandb_images = []

    # Denormalization parameters (assuming ImageNet standard)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Visualizing {phase_name} predictions"):
            # Handle different batch structures: (images, masks, low_res_masks/info)
            # We assume first is images, second is high-res masks
            images = batch[0].to(device)
            masks = batch[1].to(device) 
            
            # Forward pass base on model type
            if model_type == 'sam':
                outputs = model(images, False, img_size)
                # Use high-res masks if available, otherwise low-res logits
                logits = outputs.get('masks', outputs.get('low_res_logits'))
                
                if logits is None:
                    # Fallback for unexpected SAM output structure
                    logits = list(outputs.values())[0]

                # Resize logits to match ground truth masks
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                    
                if num_classes == 2:
                    preds = (torch.sigmoid(logits) > 0.5).float()
                else:
                    # For multi-class SAM, logits might be [B, C, H, W]
                    preds = torch.argmax(logits, dim=1, keepdim=True)
            
            elif model_type == 'segformer':
                outputs = model(images)
                logits = outputs.logits
                
                # Resize logits to match ground truth masks
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                
                if num_classes == 2:
                    preds = (torch.sigmoid(logits) > 0.5).float()
                else:
                    preds = torch.argmax(logits, dim=1, keepdim=True)
            
            else: # default (TinyUSFM, etc.)
                outputs = model(images)
                # Ensure outputs has a channel dimension
                if outputs.dim() == 3:
                    outputs = outputs.unsqueeze(1)
                
                if num_classes == 2:
                    if outputs.shape[1] == 1:
                        preds = (torch.sigmoid(outputs) > 0.5).float()
                    else:
                        # Multi-channel but binary task? (Shouldn't happen with our config but handled)
                        preds = torch.argmax(outputs, dim=1, keepdim=True)
                else:
                    preds = torch.argmax(outputs, dim=1, keepdim=True)

            # Move to CPU for plotting
            images_np = images.cpu().numpy()
            masks_np = masks.cpu().numpy()
            preds_np = preds.cpu().numpy()

            for i in range(images_np.shape[0]):
                img = images_np[i]
                mask = masks_np[i]
                pred = preds_np[i]

                # Denormalize image
                if img.shape[0] == 3:  # RGB
                    img = img.transpose(1, 2, 0)
                    img = std * img + mean
                    img = np.clip(img, 0, 1)
                elif img.shape[0] == 1:  # Grayscale
                    img = img[0]
                    img = std[0] * img + mean[0]
                    img = np.clip(img, 0, 1)

                # Fix mask/pred dimensions for plotting (squeeze channel if 1)
                if mask.ndim == 3 and mask.shape[0] == 1: 
                    mask = mask[0]
                if pred.ndim == 3 and pred.shape[0] == 1: 
                    pred = pred[0]
                
                # Create visualization figure
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                
                # Input Image
                axes[0].imshow(img, cmap='gray' if img.ndim == 2 else None)
                axes[0].set_title('Input Image', fontsize=14)
                axes[0].axis('off')

                # Ground Truth
                axes[1].imshow(mask, cmap='jet')
                axes[1].set_title('Ground Truth', fontsize=14)
                axes[1].axis('off')

                # Prediction
                axes[2].imshow(pred, cmap='jet')
                axes[2].set_title('Prediction', fontsize=14)
                axes[2].axis('off')

                # Overlay (Input + Prediction)
                axes[3].imshow(img, cmap='gray' if img.ndim == 2 else None)
                # Overlay mask with some transparency
                overlay = np.zeros((*pred.shape, 4))
                # Map prediction values to colors if multi-class, or just use jet
                # For simplicity and consistency, we use the same jet mapping with alpha
                cmap = plt.get_cmap('jet')
                colored_pred = cmap(pred / (num_classes - 1) if num_classes > 1 else pred)
                colored_pred[..., 3] = 0.4  # Set alpha
                # Mask out background (pred == 0) for cleaner overlay if desired
                # But sometimes background prediction is also interesting.
                # Here we overlay everything but with alpha.
                axes[3].imshow(colored_pred)
                axes[3].set_title('Overlay', fontsize=14)
                axes[3].axis('off')

                plt.tight_layout()
                save_path = save_dir / f"sample_{sample_count:03d}.png"
                plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
                
                # Collect for WandB
                wandb_images.append(wandb.Image(str(save_path), caption=f"Sample {sample_count}"))
                plt.close(fig)

                sample_count += 1

                # Check if we've reached the limit
                if num_samples is not None and sample_count >= num_samples:
                    break

            # Break outer loop if limit reached
            if num_samples is not None and sample_count >= num_samples:
                break

    # Log to WandB if it's initialized
    if wandb_images and wandb.run is not None:
        wandb.log({f"{phase_name}/visualizations": wandb_images})


def visualize_from_predictions(
    images_list: List[torch.Tensor],
    preds_list: List[torch.Tensor],
    masks_list: List[torch.Tensor],
    num_classes: int,
    save_dir: Union[Path, str],
    num_samples: Optional[int] = None,
    phase_name: str = "test"
):
    """
    Visualize pre-computed predictions without running model inference.

    Args:
        images_list: List of image tensors [B, C, H, W] from evaluation
        preds_list: List of prediction tensors [B, 1, H, W] from evaluation
        masks_list: List of mask tensors [B, H, W] from evaluation
        num_classes: Number of segmentation classes
        save_dir: Directory to save visualization images
        num_samples: Number of samples to visualize. If None, visualize all samples.
        phase_name: Name of the phase for WandB logging.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    sample_count = 0
    wandb_images = []

    # Denormalization parameters (assuming ImageNet standard)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for images, preds, masks in zip(images_list, preds_list, masks_list):
        images_np = images.numpy()
        preds_np = preds.numpy()
        masks_np = masks.numpy()

        for i in range(images_np.shape[0]):
            img = images_np[i]
            pred = preds_np[i]
            mask = masks_np[i]

            # Denormalize image
            if img.shape[0] == 3:  # RGB
                img = img.transpose(1, 2, 0)
                img = std * img + mean
                img = np.clip(img, 0, 1)
            elif img.shape[0] == 1:  # Grayscale
                img = img[0]
                img = std[0] * img + mean[0]
                img = np.clip(img, 0, 1)

            # Fix mask/pred dimensions for plotting (squeeze channel if 1)
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            if pred.ndim == 3 and pred.shape[0] == 1:
                pred = pred[0]

            # Create visualization figure
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            # Input Image
            axes[0].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axes[0].set_title('Input Image', fontsize=14)
            axes[0].axis('off')

            # Ground Truth
            axes[1].imshow(mask, cmap='jet')
            axes[1].set_title('Ground Truth', fontsize=14)
            axes[1].axis('off')

            # Prediction
            axes[2].imshow(pred, cmap='jet')
            axes[2].set_title('Prediction', fontsize=14)
            axes[2].axis('off')

            # Overlay (Input + Prediction)
            axes[3].imshow(img, cmap='gray' if img.ndim == 2 else None)
            cmap = plt.get_cmap('jet')
            colored_pred = cmap(pred / (num_classes - 1) if num_classes > 1 else pred)
            colored_pred[..., 3] = 0.4  # Set alpha
            axes[3].imshow(colored_pred)
            axes[3].set_title('Overlay', fontsize=14)
            axes[3].axis('off')

            plt.tight_layout()
            save_path = save_dir / f"sample_{sample_count:03d}.png"
            plt.savefig(str(save_path), dpi=150, bbox_inches='tight')

            # Collect for WandB
            wandb_images.append(wandb.Image(str(save_path), caption=f"Sample {sample_count}"))
            plt.close(fig)

            sample_count += 1

            # Check if we've reached the limit
            if num_samples is not None and sample_count >= num_samples:
                break

        # Break outer loop if limit reached
        if num_samples is not None and sample_count >= num_samples:
            break

    # Log to WandB if it's initialized
    if wandb_images and wandb.run is not None:
        wandb.log({f"{phase_name}/visualizations": wandb_images})
