from typing import Any
from distillers.unified_distiller import UnifiedDistiller


def create_distiller(cfg: Any) -> Any:
    """Create a distiller instance based on config."""
    if hasattr(cfg, "method") and "_target_" in cfg.method:
        from hydra.utils import instantiate
        
        return instantiate(cfg.method, cfg)

    return UnifiedDistiller(cfg)

__all__ = [
    "create_distiller",
    "UnifiedDistiller",
]
