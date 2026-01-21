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
    elif name == 'SAM3TrainerAdapter':
        from .sam3_adapter import SAM3TrainerAdapter
        return SAM3TrainerAdapter
    elif name == 'SAM3Orchestrator':
        from .sam3_adapter import SAM3Orchestrator
        return SAM3Orchestrator
    raise AttributeError(f"module 'trainers' has no attribute '{name}'")
