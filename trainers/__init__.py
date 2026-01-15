from .model_builder import ModelBuilder

__all__ = ['ModelBuilder']


def __getattr__(name):
    """Lazy import for heavy trainer classes."""
    if name == 'BaseTrainer':
        from .base_trainer import BaseTrainer
        return BaseTrainer
    elif name == 'SAMTrainer':
        from .sam_trainer import SAMTrainer
        return SAMTrainer
    elif name == 'TinyUSFMTrainer':
        from .tinyusfm_trainer import TinyUSFMTrainer
        return TinyUSFMTrainer
    elif name == 'SegformerTrainer':
        from .segformer_trainer import SegformerTrainer
        return SegformerTrainer
    raise AttributeError(f"module 'trainers' has no attribute '{name}'")
