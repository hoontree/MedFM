"""Reusable adapter modules for Seg-TransMiter-lite.

- ResidualConvAdapter: residual class-logit map on top of a frozen backbone.
- FeatureProjector: aligns backbone features to the other model's embedding.
- ConvLoRAAdapterStub: thin re-export of the existing Conv-LoRA implementation.
"""

from .adapters import (
    ResidualConvAdapter,
    FeatureProjector,
    ConvLoRAAdapterStub,
)

__all__ = [
    "ResidualConvAdapter",
    "FeatureProjector",
    "ConvLoRAAdapterStub",
]
