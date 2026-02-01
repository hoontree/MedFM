import random
import numpy as np
from typing import Optional, List, Dict, Union, Type
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from omegaconf import OmegaConf, ListConfig
import json
import copy

from utils.ultrasound_datasets import (
    BUID,
    BUS_UCLM,
    BUSI,
    BUSBRA,
    UltrasoundSegmentationDataset,
    B,
)

# Dataset class registry
DATASET_REGISTRY: Dict[str, Type[Dataset]] = {
    "BUID": BUID,
    "BUS_UCLM": BUS_UCLM,
    "BUSI": BUSI,
    "BUSBRA": BUSBRA,
    "UltrasoundSegmentationDataset": UltrasoundSegmentationDataset,
    "B": B,
}


def get_dataset_class(name: str) -> Type[Dataset]:
    """Get dataset class from registry by name."""
    if name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset class: {name}. Available: {available}")
    return DATASET_REGISTRY[name]


class SegDatasetProcessor:
    @staticmethod
    def load_dataset_from_config(cfg, name, split):
        """Helper to load dataset config and instantiate."""
        config_path = Path(f"config/data/{name}.yaml")
        if not config_path.exists():
            raise ValueError(f"Config for {name} not found at {config_path}")

        data_cfg = OmegaConf.load(config_path)

        # Override global settings
        for attr in ["img_size", "num_classes", "normalization"]:
            if hasattr(cfg.data, attr):
                setattr(data_cfg, attr, getattr(cfg.data, attr))

        if split == "test":
            data_cfg.usage = "external"

        dataset_class = get_dataset_class(data_cfg.name)
        return dataset_class(data_cfg, split=split)

    @staticmethod
    def build_dataset(cfg):
        """Build train, val, and test datasets according to config."""
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
    def build_distillation_datasets(
        cfg, adaptation_ratio=0.5, seed=42, split_file=None, save_split=True
    ):
        """Build non-overlapping datasets for distillation workflow."""
        train_ds, val_ds, test_ds_dict = SegDatasetProcessor.build_dataset(cfg)

        # Determine split path
        if split_file:
            split_path = Path(split_file)
        else:
            name = (
                "_".join(cfg.data.train)
                if isinstance(cfg.data.train, (list, ListConfig))
                else cfg.data.name
            )
            split_path = Path(f"splits/distill_{name}_s{seed}.json")

        indices = SegDatasetProcessor._get_split_indices(
            train_ds, val_ds, adaptation_ratio, seed, split_path, save_split
        )

        return {
            "adaptation_train": Subset(train_ds, indices["train_adapt"]),
            "distillation_train": Subset(train_ds, indices["train_distill"]),
            "adaptation_val": Subset(val_ds, indices["val_adapt"]),
            "distillation_val": Subset(val_ds, indices["val_distill"]),
            "test": test_ds_dict,
        }

    @staticmethod
    def build_distillation_data_loaders(
        cfg, adaptation_ratio=0.5, seed=42, split_file=None, save_split=True
    ):
        """Distillation-specific data loader builder with adaptation/distillation splitting."""
        datasets = SegDatasetProcessor.build_distillation_datasets(
            cfg, adaptation_ratio, seed, split_file, save_split
        )

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
            "adaptation_train": _get_loader(datasets["adaptation_train"], True),
            "adaptation_val": _get_loader(datasets["adaptation_val"], False),
            "distillation_train": _get_loader(datasets["distillation_train"], True),
            "distillation_val": _get_loader(datasets["distillation_val"], False),
            "test": {
                name: _get_loader(ds, False) for name, ds in datasets["test"].items()
            },
        }

    @staticmethod
    def _get_split_indices(train_ds, val_ds, ratio, seed, path, save):
        """Unified index splitting logic (Stratified-safe)."""
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                # Structural check
                if all(
                    k in data
                    for k in [
                        "train_adapt",
                        "train_distill",
                        "val_adapt",
                        "val_distill",
                    ]
                ):
                    if (
                        data.get("n_train") == len(train_ds)
                        and data.get("seed") == seed
                    ):
                        print(f"Loaded valid split indices from {path}")
                        return data
            except Exception:
                pass
            print(f"Split file {path} invalid or size mismatch. Regenerating.")

        # Generate new split
        rng = np.random.default_rng(seed)

        def _split_idx(n):
            idx = rng.permutation(n)
            k = int(n * ratio)
            return idx[:k].tolist(), idx[k:].tolist()

        train_a, train_d = _split_idx(len(train_ds))
        val_a, val_d = _split_idx(len(val_ds))

        result = {
            "seed": seed,
            "ratio": ratio,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "train_adapt": train_a,
            "train_distill": train_d,
            "val_adapt": val_a,
            "val_distill": val_d,
        }

        if save:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Saved new split indices to {path}")

        return result

    @staticmethod
    def _add_test_dataset_with_unfiltered(cfg, name, test_datasets_dict):
        """Load test dataset and optionally add its unfiltered version."""
        print(f"Loading Test dataset: {name}")
        ds = SegDatasetProcessor.load_dataset_from_config(cfg, name, split="test")
        test_datasets_dict[name] = ds

        if getattr(ds, "filter_empty_masks", False):
            ds_un = copy.copy(ds)
            ds_un.image_files = list(ds.image_files_unfiltered)
            ds_un.mask_files = list(ds.mask_files_unfiltered)
            test_datasets_dict[f"{name}_unfiltered"] = ds_un
            print(f"  -> Added {name}_unfiltered ({len(ds_un)} samples)")
