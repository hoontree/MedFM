"""
SAM (Segment Anything Model) checkpoint evaluation entry point.
"""
import logging
from pathlib import Path
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from utils.hardware import set_gpu

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="test_sam")
def main(cfg: DictConfig):
    """Run test-only mode for SAM model."""
    set_gpu(cfg)
    
    # Instantiate trainer
    trainer = instantiate(cfg.trainer, cfg)
    
    # Get checkpoint path from config
    checkpoint_path = cfg.get("test_only", {}).get("checkpoint_path")
    if not checkpoint_path:
        log.error("No checkpoint_path provided in config or command line.")
        log.info("Usage: python test_sam.py model=sam test_only.checkpoint_path=/path/to/checkpoint.pth")
        return

    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        log.error(f"Checkpoint not found: {checkpoint_path}")
        return

    # Initialize and run test-only
    # BaseTrainer has run_test_only method
    trainer.setup(mode="test")
    trainer.run_test_only(checkpoint_path)

if __name__ == "__main__":
    main()
