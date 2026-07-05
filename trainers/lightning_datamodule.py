"""
PyTorch Lightning DataModule for Medical Image Segmentation.

Wraps ``SegDatasetProcessor.build_data_loaders`` so that the same
dataset loading / splitting logic is used for Lightning training.
"""

import logging
from typing import Dict, Optional

import lightning as L
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from utils.data_processing import SegDatasetProcessor

log = logging.getLogger(__name__)


class SegDataModule(L.LightningDataModule):
    """Lightning DataModule backed by ``SegDatasetProcessor``.

    Produces the same train / val / test DataLoaders that the original
    ``SAMTrainer._create_dataloaders`` creates.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.batch_size: int = cfg.training.batch_size
        self.num_workers: int = cfg.training.num_workers

        # Will be set in ``setup()``
        self.train_ds = None
        self.val_ds = None
        self.test_ds_dict: Dict = {}

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None):
        """Build datasets via the existing ``SegDatasetProcessor``."""
        self.train_ds, self.val_ds, self.test_ds_dict = (
            SegDatasetProcessor.build_dataset(self.cfg)
        )
        log.info(
            "DataModule setup: train=%d, val=%d, test datasets=%d",
            len(self.train_ds),
            len(self.val_ds),
            len(self.test_ds_dict),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Return test loaders.

        If there are multiple test datasets, return a list of DataLoaders
        (Lightning will call ``test_step`` with ``dataloader_idx``).
        """
        loaders = [
            DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            for ds in self.test_ds_dict.values()
        ]
        if len(loaders) == 1:
            return loaders[0]
        return loaders

    @property
    def test_dataset_names(self):
        """Ordered list of test-dataset names (matches dataloader order)."""
        return list(self.test_ds_dict.keys())
