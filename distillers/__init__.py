from distillers.registry import DistillerRegistry
from distillers.logit_distiller import LogitDistiller
from distillers.feature_distiller import FeatureDistiller
from distillers.adaptive_distiller import AdaptiveDistiller

# Make sure all distillers are registered when the package is imported
__all__ = [
    "DistillerRegistry",
    "LogitDistiller",
    "FeatureDistiller",
    "AdaptiveDistiller",
]
