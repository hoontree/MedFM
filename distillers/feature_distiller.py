import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
from distillers.base_distiller import BaseDistiller
from distillers.registry import DistillerRegistry
from distillers.logit_distiller import LogitDistiller


class FeatureAdapter(nn.Module):
    """Adapter to match student feature dimension to teacher."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


@DistillerRegistry.register("feature")
class FeatureDistiller(LogitDistiller):
    """
    Feature-based distillation.
    Inherits from LogitDistiller to keep task and logit losses.
    Adds MSE loss between intermediate features.
    """

    def __init__(self, cfg: Any):
        super().__init__(cfg)
        self.mse_loss = nn.MSELoss()

        # Feature adapter: Student (48) -> Teacher (256)
        # These channels should ideally be configurable
        student_channels = cfg.method.get("student_channels", 48)
        teacher_channels = cfg.method.get("teacher_channels", 256)
        self.adapter = FeatureAdapter(student_channels, teacher_channels)

    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        # Get baseline logit/task losses
        results = super().forward(student_outputs, teacher_outputs, targets)

        student_features_raw = student_outputs.get("features")
        teacher_features = teacher_outputs.get("image_embeddings")

        if student_features_raw is None or teacher_features is None:
            results["feature_loss"] = torch.tensor(0.0).to(targets.device)
            return results

        # Adapt student features
        student_features = self.adapter(student_features_raw)

        # Match spatial dimensions if needed
        if student_features.shape != teacher_features.shape:
            teacher_features = F.interpolate(
                teacher_features,
                size=student_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        feature_loss = self.mse_loss(student_features, teacher_features)

        # Update total loss
        results["loss"] = results["loss"] + self.gamma * feature_loss
        results["feature_loss"] = feature_loss

        return results
