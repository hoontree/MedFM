import random
from typing import Optional, Dict, Union, List

import numpy as np
import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset

from utils.data_processing import SegDatasetProcessor


class SegmentationDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for segmentation tasks.

    Wraps SegDatasetProcessor to provide a unified interface for training,
    validation, and testing with automatic DDP support.

    Args:
        cfg: Hydra configuration object
        seed: Random seed for reproducibility
    """

    def __init__(self, cfg, seed: int = 42):
        super().__init__()
        self.cfg = cfg
        self.seed = seed
        self.batch_size = cfg.training.batch_size
        self.num_workers = cfg.training.num_workers

        # Datasets (initialized in setup)
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_datasets: Dict[str, Dataset] = {}

        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=["cfg"])

    def prepare_data(self):
        """Download data if needed. Called only on 1 GPU in DDP."""
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for each stage.
        Called on every GPU in DDP.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        if stage == "fit" or stage is None:
            self.train_dataset, self.val_dataset, self.test_datasets = (
                SegDatasetProcessor.build_dataset(self.cfg)
            )

        if stage == "validate":
            if self.val_dataset is None:
                _, self.val_dataset, _ = SegDatasetProcessor.build_dataset(self.cfg)

        if stage == "test":
            if not self.test_datasets:
                _, _, self.test_datasets = SegDatasetProcessor.build_dataset(self.cfg)

    def _worker_init_fn(self, worker_id: int):
        """Initialize worker with deterministic seed."""
        worker_seed = self.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=self._worker_init_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """
        Returns test dataloaders.

        If multiple test datasets exist, returns a list of DataLoaders.
        Lightning will call test_step with dataloader_idx parameter.
        """
        if not self.test_datasets:
            raise ValueError("No test datasets available. Call setup('test') first.")

        if len(self.test_datasets) == 1:
            ds = list(self.test_datasets.values())[0]
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )

        # Multiple test datasets
        return [
            DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            for ds in self.test_datasets.values()
        ]

    @property
    def test_dataset_names(self) -> List[str]:
        """Return names of test datasets for logging purposes."""
        return list(self.test_datasets.keys())

    def on_exception(self, exception: Exception):
        """Handle exceptions during data loading."""
        pass

    def teardown(self, stage: Optional[str] = None):
        """Clean up after training/testing."""
        pass


class DistillationDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for knowledge distillation workflows.

    Provides shared train/val datasets for teacher and student stages.

    Args:
        cfg: Hydra configuration object
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        cfg,
        seed: int = 42,
    ):
        super().__init__()
        self.cfg = cfg
        self.seed = seed

        self.batch_size = cfg.training.batch_size
        self.num_workers = cfg.training.num_workers

        # Datasets
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_datasets: Dict[str, Dataset] = {}

        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=["cfg"])

    def prepare_data(self):
        """Download data if needed. Called only on 1 GPU in DDP."""
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Setup shared datasets for distillation workflow.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        if stage == "fit" or stage is None:
            datasets = SegDatasetProcessor.build_distillation_datasets(self.cfg)

            self.train_dataset = datasets["train"]
            self.val_dataset = datasets["val"]
            self.test_datasets = datasets.get("test", {})

        if stage == "test":
            if not self.test_datasets:
                datasets = SegDatasetProcessor.build_distillation_datasets(self.cfg)
                self.test_datasets = datasets.get("test", {})

    def _worker_init_fn(self, worker_id: int):
        """Initialize worker with deterministic seed."""
        worker_seed = self.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def train_dataloader(self) -> DataLoader:
        """Return shared training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=self._worker_init_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Return shared validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """
        Returns test dataloaders.
        Test datasets are shared across both phases.
        """
        if not self.test_datasets:
            raise ValueError("No test datasets available. Call setup('test') first.")

        if len(self.test_datasets) == 1:
            ds = list(self.test_datasets.values())[0]
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )

        return [
            DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            for ds in self.test_datasets.values()
        ]

    @property
    def test_dataset_names(self) -> List[str]:
        """Return names of test datasets for logging purposes."""
        return list(self.test_datasets.keys())

    @property
    def train_size(self) -> int:
        """Number of samples in shared training set."""
        return len(self.train_dataset) if self.train_dataset else 0

    def on_exception(self, exception: Exception):
        """Handle exceptions during data loading."""
        pass

    def teardown(self, stage: Optional[str] = None):
        """Clean up after training/testing."""
        pass
