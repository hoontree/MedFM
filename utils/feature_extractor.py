import torch
import torch.nn as nn
from typing import Dict, List, Optional


class FeatureExtractor:
    """
    Utility to extract intermediate features from any PyTorch model using hooks.
    """

    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self.features: Dict[str, torch.Tensor] = {}
        self.hooks = []
        self._setup_hooks()

    def _get_activation(self, name):
        def hook(model, input, output):
            # If output is a tuple (some models), take the first element (usually the tensor)
            if isinstance(output, tuple):
                self.features[name] = output[0]
            else:
                self.features[name] = output

        return hook

    def _setup_hooks(self):
        for name in self.layer_names:
            try:
                # Handle nested attributes like 'backbone.blocks.2'
                layer = self.model
                for attr in name.split("."):
                    if attr.isdigit():
                        layer = layer[int(attr)]
                    else:
                        layer = getattr(layer, attr)
                self.hooks.append(
                    layer.register_forward_hook(self._get_activation(name))
                )
            except (AttributeError, IndexError, KeyError):
                print(f"Warning: Could not find layer '{name}' in model.")

    def get_features(self) -> Dict[str, torch.Tensor]:
        return self.features

    def clear(self):
        self.features.clear()

    def remove(self):
        for hook in self.hooks:
            hook.remove()
