from typing import Any
from hydra.utils import instantiate
from distillers.unified_distiller import UnifiedDistiller
from distillers.online_distiller import OnlineDistiller
from distillers.meta_reliability import MetaReliabilityDistiller


def create_distiller(cfg: Any) -> Any:
    """Instantiate the distiller named by ``cfg.method._target_``.

    The distiller constructors take the root ``cfg`` (they read both
    ``cfg.method`` and ``cfg.data.num_classes``), so the root config is
    passed positionally to the ``_target_`` named inside ``cfg.method``.
    """
    if not (hasattr(cfg, "method") and "_target_" in cfg.method):
        raise ValueError(
            "cfg.method must define a `_target_` (e.g. distillers.UnifiedDistiller). "
            f"Got keys: {list(cfg.method.keys()) if hasattr(cfg, 'method') else None}"
        )
    return instantiate(cfg.method, cfg)

__all__ = [
    "create_distiller",
    "UnifiedDistiller",
    "OnlineDistiller",
    "MetaReliabilityDistiller",
]
