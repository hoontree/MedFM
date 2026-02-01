from typing import Any
from distillers.unified_distiller import UnifiedDistiller


def create_distiller(cfg: Any) -> Any:
    """Create a distiller instance based on config."""
    # 1. Try Hydra instantiate
    if hasattr(cfg, "method") and "_target_" in cfg.method:
        from hydra.utils import instantiate

        print(f"Instantiating distiller via Hydra: {cfg.method._target_}")
        return instantiate(cfg.method, cfg)

    return UnifiedDistiller(cfg)


# Make sure all distillers are registered when the package is imported
__all__ = [
    "create_distiller",
    "UnifiedDistiller",
]
