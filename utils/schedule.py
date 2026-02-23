import torch
import math


class CosineAnnealingWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine Annealing with Warmup scheduler.
    Widely used in knowledge distillation for smooth learning rate decay.

    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup epochs
        max_epochs: Total number of training epochs
        base_lr: Base learning rate (max LR after warmup)
        min_lr: Minimum learning rate at the end of training
        last_epoch: The index of last epoch
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        max_epochs: int,
        base_lr: float,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1

        # Warmup phase: linear increase
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / self.warmup_epochs
            return [self.base_lr * factor for _ in self.optimizer.param_groups]

        # Cosine annealing phase
        cosine_epoch = epoch - self.warmup_epochs
        cosine_total = self.max_epochs - self.warmup_epochs

        # Cosine decay from base_lr to min_lr
        cosine_factor = 0.5 * (1 + math.cos(math.pi * cosine_epoch / cosine_total))
        return [
            self.min_lr + (self.base_lr - self.min_lr) * cosine_factor
            for _ in self.optimizer.param_groups
        ]


class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        max_epochs: int,
        base_lr: float,
        end_lr: float = 0.0,
        power: float = 1.0,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.end_lr = end_lr
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1

        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / self.warmup_epochs
            return [self.base_lr * factor for _ in self.optimizer.param_groups]

        poly_epoch = epoch - self.warmup_epochs
        poly_total = self.max_epochs - self.warmup_epochs
        factor = (1 - poly_epoch / poly_total) ** self.power
        return [
            self.end_lr + (self.base_lr - self.end_lr) * factor
            for _ in self.optimizer.param_groups
        ]


def build_scheduler(optimizer, cfg):
    """
    Build learning rate scheduler for training.
    Uses Hydra instantiate to create scheduler from config.

    Args:
        optimizer: PyTorch optimizer
        cfg: Configuration object with scheduler settings

    Returns:
        Learning rate scheduler instance
    """
    from hydra.utils import instantiate

    # If scheduler config exists, use instantiate
    if hasattr(cfg, "scheduler") and hasattr(cfg.scheduler, "_target_"):
        return instantiate(cfg.scheduler, optimizer=optimizer)

    # Fallback to legacy behavior for backward compatibility
    return CosineAnnealingWarmupLR(
        optimizer=optimizer,
        warmup_epochs=cfg.training.warmup_epochs,
        max_epochs=cfg.training.num_epochs,
        base_lr=cfg.training.lr,
        min_lr=1e-6,
    )


def get_lr_decay_param_groups(
    model, base_lr, weight_decay, num_layers=12, layer_decay=0.8
):
    def get_layer_id(param_name):
        if param_name.startswith("backbone"):
            if "blocks." in param_name:
                block_id = int(param_name.split("blocks.")[1].split(".")[0])
                return block_id
            elif "patch_embed" in param_name:
                return 0
            else:
                return num_layers - 1
        else:
            return num_layers

    param_groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_id = get_layer_id(name)
        group_name = f"layer_{layer_id}"

        if group_name not in param_groups:
            scale = layer_decay ** (num_layers - layer_id)
            param_groups[group_name] = {
                "params": [],
                "lr": base_lr * scale,
                "weight_decay": weight_decay,
            }

        param_groups[group_name]["params"].append(param)

    return list(param_groups.values())
