import random
from typing import Optional, List, Dict, Union, Type
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from omegaconf import OmegaConf, ListConfig

from utils.ultrasound_datasets import (
    BUID,
    BUS_UCLM,
    BUS_UCLM_filtered,
    BUSI,
    BUSBRA,
    B,
)
# Dataset class registry
DATASET_REGISTRY: Dict[str, Type[Dataset]] = {
    "BUID": BUID,
    "BUS_UCLM": BUS_UCLM,
    "BUS_UCLM_filtered": BUS_UCLM_filtered,
    "BUSI": BUSI,
    "BUSBRA": BUSBRA,
    "B": B,
}

DEFAULT_SAM_IMG_SIZE_BY_TYPE = {
    "vit_b": 224,
    "vit_l": 256,
    "vit_h": 256,
}


def get_dataset_class(name: str) -> Type[Dataset]:
    """Get dataset class from registry by name."""
    if name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset class: {name}. Available: {available}")
    return DATASET_REGISTRY[name]


class SegDatasetProcessor:
    @staticmethod
    def _sync_img_size_with_sam_type(cfg):
        """Sync data/model img_size from sam_type when enabled.

        Applies to:
        - train path: cfg.model.sam_type
        - distill path: cfg.teacher.sam_type
        """
        data_cfg = cfg.get("data", {})
        if not data_cfg.get("auto_img_size_by_sam_type", True):
            return

        sam_cfg = None
        if "model" in cfg and cfg.model.get("sam_type") is not None:
            sam_cfg = cfg.model
        elif "teacher" in cfg and cfg.teacher.get("sam_type") is not None:
            sam_cfg = cfg.teacher
        if sam_cfg is None:
            return

        sam_type = str(sam_cfg.get("sam_type")).lower()
        size_map = data_cfg.get("sam_img_size_map", DEFAULT_SAM_IMG_SIZE_BY_TYPE)
        target_size = size_map.get(sam_type)
        if target_size is None:
            return

        target_size = int(target_size)
        cfg.data.img_size = target_size
        sam_cfg.img_size = target_size

    @staticmethod
    def load_dataset_from_config(cfg, name, split, force_external=False):
        """Helper to load dataset config and instantiate.

        Args:
            force_external: If True, set usage="external" (for external validation sets).
                            If False, keep the config's default usage (for internal splits).
        """
        config_path = Path(f"config/data/{name}.yaml")
        if not config_path.exists():
            raise ValueError(f"Config for {name} not found at {config_path}")

        data_cfg = OmegaConf.load(config_path)

        # Override global settings
        for attr in ["img_size", "num_classes", "normalization", "multiclass"]:
            if hasattr(cfg.data, attr):
                setattr(data_cfg, attr, getattr(cfg.data, attr))

        if force_external:
            data_cfg.usage = "external"

        dataset_class = get_dataset_class(data_cfg.name)
        return dataset_class(data_cfg, split=split)

    @staticmethod
    def build_dataset(cfg):
        """Build train, val, and test datasets according to config."""
        SegDatasetProcessor._sync_img_size_with_sam_type(cfg)

        # 1. Train/Val sets (Combine into ConcatDataset if multiple)
        train_list = (
            cfg.data.train
            if isinstance(cfg.data.train, (list, ListConfig))
            else [cfg.data.name]
        )
        val_list = getattr(cfg.data, "val", None) or train_list
        if not isinstance(val_list, (list, ListConfig)):
            val_list = [val_list]

        train_datasets = [
            SegDatasetProcessor.load_dataset_from_config(cfg, n, "train")
            for n in train_list
        ]
        val_datasets = [
            SegDatasetProcessor.load_dataset_from_config(cfg, n, "val")
            for n in val_list
        ]

        print(
            f"Loaded Train: {', '.join(train_list)} ({sum(len(d) for d in train_datasets)} samples)"
        )
        print(
            f"Loaded Val: {', '.join(val_list)} ({sum(len(d) for d in val_datasets)} samples)"
        )

        # 2. Test sets (Always return a dictionary for separate evaluation)
        test_datasets = {}

        # 2a. Internal test sets: held-out 15% from each train dataset (usage stays "train")
        for name in train_list:
            print(f"Loading internal test: {name} → {name}_test")
            ds = SegDatasetProcessor.load_dataset_from_config(
                cfg, name, "test", force_external=False
            )
            test_datasets[f"{name}_test"] = ds

        # 2b. External validation sets (usage="external")
        test_list = getattr(cfg.data, "test", [])
        if isinstance(test_list, str):
            test_list = [test_list]
        elif not test_list and not isinstance(cfg.data.train, (list, ListConfig)):
            test_list = [cfg.data.name]

        for name in test_list:
            SegDatasetProcessor._add_test_dataset_with_unfiltered(
                cfg, name, test_datasets
            )

        return (
            (
                ConcatDataset(train_datasets)
                if len(train_datasets) > 1
                else train_datasets[0]
            ),
            ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0],
            test_datasets,
        )

    @staticmethod
    def build_data_loaders(cfg):
        """Standard trainer data loader builder."""
        train_ds, val_ds, test_ds_dict = SegDatasetProcessor.build_dataset(cfg)

        batch_size = cfg.training.batch_size
        num_workers = cfg.training.num_workers

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        test_loaders = {
            name: DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            for name, ds in test_ds_dict.items()
        }

        return train_loader, val_loader, test_loaders

    @staticmethod
    def build_distillation_datasets(cfg):
        """Build datasets for distillation workflow (shared train/val)."""
        train_ds, val_ds, test_ds_dict = SegDatasetProcessor.build_dataset(cfg)

        return {
            "train": train_ds,
            "val": val_ds,
            "test": test_ds_dict,
        }

    @staticmethod
    def build_distillation_data_loaders(cfg):
        """Distillation-specific data loader builder using shared train/val."""
        datasets = SegDatasetProcessor.build_distillation_datasets(cfg)

        batch_size = cfg.training.batch_size
        num_workers = cfg.training.num_workers

        def _get_loader(ds, shuffle):
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
            )

        return {
            "train": _get_loader(datasets["train"], True),
            "val": _get_loader(datasets["val"], False),
            "test": {
                name: _get_loader(ds, False) for name, ds in datasets["test"].items()
            },
        }

    @staticmethod
    def _add_test_dataset_with_unfiltered(cfg, name, test_datasets_dict):
        """Load external test dataset and add to dict."""
        print(f"Loading external test: {name}")
        ds = SegDatasetProcessor.load_dataset_from_config(
            cfg, name, split="test", force_external=True
        )
        test_datasets_dict[name] = ds
