from typing import Dict, Type, Any
from distillers.base_distiller import BaseDistiller


class DistillerRegistry:
    """Registry to manage and instantiate different distillation methods."""

    _DISTILLERS: Dict[str, Type[BaseDistiller]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a new distiller."""

        def wrapper(distiller_cls: Type[BaseDistiller]):
            cls._DISTILLERS[name.lower()] = distiller_cls
            return distiller_cls

        return wrapper

    @classmethod
    def get(cls, name: str) -> Type[BaseDistiller]:
        """Get a distiller class by name."""
        name = name.lower()
        if name not in cls._DISTILLERS:
            raise ValueError(
                f"Distiller '{name}' not found. Available: {list(cls._DISTILLERS.keys())}"
            )
        return cls._DISTILLERS[name]

    @classmethod
    def create(cls, cfg: Any) -> BaseDistiller:
        """Create a distiller instance based on config."""
        method_name = cfg.method.name
        distiller_cls = cls.get(method_name)
        return distiller_cls(cfg)
