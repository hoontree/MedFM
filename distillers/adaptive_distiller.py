import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
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


@DistillerRegistry.register("adaptive_layer")
class AdaptiveDistiller(LogitDistiller):
    """
    Adaptive layer-wise distillation.
    Supports multiple layer pairs and can learn weights for each layer.
    """

    def __init__(self, cfg: Any):
        super().__init__(cfg)
        self.mse_loss = nn.MSELoss()

        self.layer_mapping = cfg.method.get("layer_mapping", {})
        self.adaptive_weights = cfg.method.get("adaptive_weights", False)

        # Initialize adapters for each layer pair
        self.adapters = nn.ModuleDict()

        # We need to know the channels for each layer.
        # For a truly generic system, these might need to be in the config.
        # Here we assume some default mappings if not provided.
        layer_channels = cfg.method.get("layer_channels", {})

        for student_layer, teacher_layer in self.layer_mapping.items():
            s_ch = layer_channels.get(student_layer, 48)  # Default
            t_ch = layer_channels.get(teacher_layer, 256)  # Default
            self.adapters[student_layer.replace(".", "_")] = FeatureAdapter(s_ch, t_ch)

        # Optional learnable weights for each layer's loss
        if self.adaptive_weights:
            self.loss_weights = nn.Parameter(torch.ones(len(self.layer_mapping)))
        else:
            self.loss_weights = None

    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        # Get baseline logit/task losses
        results = super().forward(student_outputs, teacher_outputs, targets)

        student_features_dict = student_outputs.get("layer_features", {})
        teacher_features_dict = teacher_outputs.get("layer_features", {})

        if not student_features_dict or not teacher_features_dict:
            # Fallback to standard feature keys if layer_features not provided
            if "features" in student_outputs and "image_embeddings" in teacher_outputs:
                student_features_dict = {"default": student_outputs["features"]}
                teacher_features_dict = {"default": teacher_outputs["image_embeddings"]}
                # We'll just use one adapter if it exists
            else:
                results["feature_loss"] = torch.tensor(0.0).to(targets.device)
                return results

        total_feature_loss = 0.0
        idx = 0

        for student_layer, teacher_layer in self.layer_mapping.items():
            s_key = student_layer.split(".")[-1]  # Simplification
            t_key = teacher_layer.split(".")[-1]

            s_feat = student_features_dict.get(s_key)
            t_feat = teacher_features_dict.get(t_key)

            if s_feat is not None and t_feat is not None:
                adapter_key = student_layer.replace(".", "_")
                if adapter_key in self.adapters:
                    s_feat_adapted = self.adapters[adapter_key](s_feat)

                    if s_feat_adapted.shape != t_feat.shape:
                        t_feat = F.interpolate(
                            t_feat,
                            size=s_feat_adapted.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )

                    layer_loss = self.mse_loss(s_feat_adapted, t_feat)

                    if self.loss_weights is not None:
                        weight = torch.softmax(self.loss_weights, dim=0)[idx]
                        total_feature_loss += weight * layer_loss
                    else:
                        total_feature_loss += layer_loss

                    idx += 1

        # Update total loss
        results["loss"] = results["loss"] + self.gamma * total_feature_loss
        results["feature_loss"] = total_feature_loss

        return results
