import random
import numpy as np
from typing import Optional, List, Dict, Union, Type
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from omegaconf import OmegaConf, ListConfig
import json
import copy

from utils.ultrasound_datasets import BUID, BUS_UCLM, BUSI, BUSBRA, UltrasoundSegmentationDataset, B


# Dataset class registry - safer than eval()
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
        raise ValueError(
            f"Unknown dataset class: {name}. Available: {available}"
        )
    return DATASET_REGISTRY[name]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SegDatasetProcessor:
    @staticmethod
    def load_dataset_from_config(cfg, name, split):
        """Helper to load dataset config and instantiate."""
        from pathlib import Path

        config_path = Path(f"config/data/{name}.yaml")
        if not config_path.exists():
            raise ValueError(f"Config for {name} not found at {config_path}")

        data_cfg = OmegaConf.load(config_path)

        # Override specific global settings if needed
        if hasattr(cfg.data, "img_size"):
            data_cfg.img_size = cfg.data.img_size
        if hasattr(cfg.data, "num_classes"):
            data_cfg.num_classes = cfg.data.num_classes
        if hasattr(cfg.data, "normalization"):
            data_cfg.normalization = cfg.data.normalization

        # Test datasets should use full data (external validation)
        if split == "test":
            data_cfg.usage = "external"

        # Override usage if provided in cfg.data (e.g. for dynamic usage override)
        if hasattr(cfg.data, "usage") and cfg.data.usage:
            # Only override if not test logic above?
            # Existing logic didn't do this, but for internal val splitting consistency it might be useful.
            # Strict adherence to legacy logic:
            pass

        dataset_class = get_dataset_class(data_cfg.name)
        return dataset_class(data_cfg, split=split)

    @staticmethod
    def build_dataset(cfg): 
        # Case 1: Dynamic/Combined Dataset (Multi-dataset support)
        # Check if train/test are lists rather than just looking at type
        is_list_train = hasattr(cfg.data, "train") and isinstance(
            cfg.data.train, (list, ListConfig)
        )
        is_combined_type = hasattr(cfg.data, "type") and cfg.data.type in [
            "Combined",
            "Dynamic",
        ]

        if is_list_train or is_combined_type:
            # 1. Train Sets
            train_datasets = []
            train_list = cfg.data.train if is_list_train else []
            for name in train_list:
                train_datasets.append(
                    SegDatasetProcessor.load_dataset_from_config(cfg, name, split="train")
                )

            if not train_datasets:
                raise ValueError("No training datasets specified in config.")
            train_dataset = ConcatDataset(train_datasets)
            print(f"Using combined training dataset consisting of:{', '.join(train_list)}")
            print(f"Total training samples: {len(train_dataset)}")
            
            # 2. Validation Sets (Internal)
            val_datasets = []
            val_list = getattr(cfg.data, "val", None)
            if (
                val_list is None
                or not isinstance(val_list, (list, ListConfig))
                or len(val_list) == 0
            ):
                # Default to same as train if not specified
                val_list = train_list

            for name in val_list:
                val_datasets.append(
                    SegDatasetProcessor.load_dataset_from_config(cfg, name, split="val")
                )
            val_dataset = ConcatDataset(val_datasets)
            print(f"Using combined validation dataset consisting of:{', '.join(val_list)}")
            print(f"Total validation samples: {len(val_dataset)}")
            # 3. Test Sets (External Validation) - Keep Separate
            test_datasets = {}
            test_list = getattr(cfg.data, "test", [])

            if isinstance(test_list, (list, ListConfig)):
                for name in test_list:
                    SegDatasetProcessor._add_test_dataset_with_unfiltered(cfg, name, test_datasets)
            elif isinstance(test_list, str):
                SegDatasetProcessor._add_test_dataset_with_unfiltered(cfg, test_list, test_datasets)

            return train_dataset, val_dataset, test_datasets

        else:
            dataset_class = get_dataset_class(cfg.data.name)
            print(f"{dataset_class.__name__} is used for segmentation task.")
            train_dataset = dataset_class(cfg.data, split="train")
            val_dataset = dataset_class(cfg.data, split="val")

            # Single dataset case for test should also follow the dictionary format with unfiltered support
            test_datasets = {}
            SegDatasetProcessor._add_test_dataset_with_unfiltered(
                cfg, cfg.data.name, test_datasets
            )

            return train_dataset, val_dataset, test_datasets

    @staticmethod
    def build_data_loaders(cfg):
        train_dataset, val_dataset, test_dataset = SegDatasetProcessor.build_dataset(
            cfg
        )

        def worker_init_fn(worker_id):
            seed = 42 + worker_id
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
        )

        # Handle Test Loader (Single vs Multiple)
        if isinstance(test_dataset, dict):
            test_loader = {}
            total_test_samples = 0
            for name, ds in test_dataset.items():
                loader = DataLoader(
                    ds,
                    batch_size=cfg.training.batch_size,
                    shuffle=False,
                    num_workers=cfg.training.num_workers,
                    pin_memory=True,
                )
                test_loader[name] = loader
                total_test_samples += len(ds)
            print(
                f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples (Total): {total_test_samples}"
            )
        else:
            test_loader = DataLoader(
                test_dataset,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
            )
            print(
                f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}"
            )

        return train_loader, val_loader, test_loader

    @staticmethod
    def _get_split_indices_path(cfg, seed: int) -> Path:
        """Get the path for storing/loading split indices."""
        # Create a unique identifier based on dataset config
        dataset_id = getattr(cfg.data, "name", "unknown")
        if hasattr(cfg.data, "train") and isinstance(
            cfg.data.train, (list, ListConfig)
        ):
            dataset_id = "_".join(cfg.data.train)
        return Path(f"splits/distillation_split_{dataset_id}_seed{seed}.json")

    @staticmethod
    def _save_split_indices(
        path: Path,
        adaptation_train_indices: List[int],
        distillation_train_indices: List[int],
        adaptation_val_indices: List[int],
        distillation_val_indices: List[int],
        adaptation_ratio: float,
        seed: int,
    ):
        """Save split indices to JSON file for reproducibility across runs."""
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "seed": seed,
            "adaptation_ratio": adaptation_ratio,
            "adaptation_train_indices": adaptation_train_indices,
            "distillation_train_indices": distillation_train_indices,
            "adaptation_val_indices": adaptation_val_indices,
            "distillation_val_indices": distillation_val_indices,
            "n_train_total": len(adaptation_train_indices)
            + len(distillation_train_indices),
            "n_val_total": len(adaptation_val_indices) + len(distillation_val_indices),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Split indices saved to: {path}")

    @staticmethod
    def _load_split_indices(path: Path) -> Optional[Dict]:
        """Load split indices from JSON file if exists."""
        if not path.exists():
            return None

        with open(path, "r") as f:
            data = json.load(f)

        print(f"Split indices loaded from: {path}")
        return data

    @staticmethod
    def build_distillation_datasets(
        cfg,
        adaptation_ratio: float = 0.5,
        seed: int = 42,
        split_file: Optional[str] = None,
        save_split: bool = True,
    ) -> Dict[str, Union[Dataset, Dict[str, Dataset]]]:
        """
        Build non-overlapping datasets for knowledge distillation workflow.

        Splits the training data into two disjoint sets:
        - Adaptation set: Used for teacher fine-tuning (e.g., LoRA adaptation)
        - Distillation set: Used for student training with teacher guidance

        The split indices are saved to a JSON file to ensure consistency across
        separate training and distillation runs.

        Args:
            cfg: Hydra configuration object
            adaptation_ratio: Ratio of training data for adaptation (default: 0.5)
            seed: Random seed for reproducible splitting
            split_file: Optional path to split file. If None, auto-generated path is used.
            save_split: Whether to save split indices to file (default: True)

        Returns:
            Dictionary containing:
                - 'adaptation_train': Dataset for teacher adaptation training
                - 'adaptation_val': Dataset for teacher adaptation validation
                - 'distillation_train': Dataset for distillation training
                - 'distillation_val': Dataset for distillation validation
                - 'test': Test dataset(s) for final evaluation
        """

        # Build full dataset first
        train_dataset, val_dataset, test_datasets = SegDatasetProcessor.build_dataset(
            cfg
        )

        # Determine split file path
        if split_file:
            split_path = Path(split_file)
        else:
            split_path = SegDatasetProcessor._get_split_indices_path(cfg, seed)

        # Try to load existing split
        existing_split = SegDatasetProcessor._load_split_indices(split_path)

        n_total = len(train_dataset)
        n_val_total = len(val_dataset)

        if existing_split is not None:
            # Validate that the loaded split matches current dataset
            if existing_split["n_train_total"] != n_total:
                print(
                    f"WARNING: Split file train size ({existing_split['n_train_total']}) "
                    f"doesn't match current dataset ({n_total}). Regenerating split."
                )
                existing_split = None
            elif existing_split["n_val_total"] != n_val_total:
                print(
                    f"WARNING: Split file val size ({existing_split['n_val_total']}) "
                    f"doesn't match current dataset ({n_val_total}). Regenerating split."
                )
                existing_split = None
            elif existing_split["seed"] != seed:
                print(
                    f"WARNING: Split file seed ({existing_split['seed']}) "
                    f"doesn't match requested seed ({seed}). Regenerating split."
                )
                existing_split = None

        if existing_split is not None:
            # Use loaded indices
            adaptation_train_indices = existing_split["adaptation_train_indices"]
            distillation_train_indices = existing_split["distillation_train_indices"]
            adaptation_val_indices = existing_split["adaptation_val_indices"]
            distillation_val_indices = existing_split["distillation_val_indices"]
            print(f"Using existing split from: {split_path}")
        else:
            # Generate new split
            n_adaptation = int(n_total * adaptation_ratio)

            # Create reproducible random indices
            rng = np.random.default_rng(seed)
            indices = rng.permutation(n_total)

            adaptation_train_indices = indices[:n_adaptation].tolist()
            distillation_train_indices = indices[n_adaptation:].tolist()

            # Split validation set similarly
            n_val_adaptation = int(n_val_total * adaptation_ratio)

            val_indices = rng.permutation(n_val_total)
            adaptation_val_indices = val_indices[:n_val_adaptation].tolist()
            distillation_val_indices = val_indices[n_val_adaptation:].tolist()

            # Save split for future runs
            if save_split:
                SegDatasetProcessor._save_split_indices(
                    split_path,
                    adaptation_train_indices,
                    distillation_train_indices,
                    adaptation_val_indices,
                    distillation_val_indices,
                    adaptation_ratio,
                    seed,
                )

        # Create subset datasets
        adaptation_train = Subset(train_dataset, adaptation_train_indices)
        distillation_train = Subset(train_dataset, distillation_train_indices)
        adaptation_val = Subset(val_dataset, adaptation_val_indices)
        distillation_val = Subset(val_dataset, distillation_val_indices)

        print(f"\n=== Distillation Dataset Split (seed={seed}) ===")
        print(f"Total training samples: {n_total}")
        print(
            f"  Adaptation set: {len(adaptation_train)} ({len(adaptation_train)/n_total*100:.1f}%)"
        )
        print(
            f"  Distillation set: {len(distillation_train)} ({len(distillation_train)/n_total*100:.1f}%)"
        )
        print(f"Total validation samples: {n_val_total}")
        print(f"  Adaptation val: {len(adaptation_val)}")
        print(f"  Distillation val: {len(distillation_val)}")

        return {
            "adaptation_train": adaptation_train,
            "adaptation_val": adaptation_val,
            "distillation_train": distillation_train,
            "distillation_val": distillation_val,
            "test": test_datasets,
        }

    @staticmethod
    def build_distillation_data_loaders(
        cfg,
        adaptation_ratio: float = 0.5,
        seed: int = 42,
        split_file: Optional[str] = None,
        save_split: bool = True,
    ) -> Dict[str, Union[DataLoader, Dict[str, DataLoader]]]:
        """
        Build DataLoaders for knowledge distillation workflow.

        Returns separate loaders for adaptation and distillation phases,
        ensuring no data overlap between the two stages.

        Args:
            cfg: Hydra configuration object
            adaptation_ratio: Ratio of training data for adaptation (default: 0.5)
            seed: Random seed for reproducible splitting
            split_file: Optional path to split file. If None, auto-generated path is used.
            save_split: Whether to save split indices to file (default: True)

        Returns:
            Dictionary containing DataLoaders:
                - 'adaptation_train': Loader for teacher adaptation training
                - 'adaptation_val': Loader for teacher adaptation validation
                - 'distillation_train': Loader for distillation training
                - 'distillation_val': Loader for distillation validation
                - 'test': Test loader(s) for final evaluation
        """
        datasets = SegDatasetProcessor.build_distillation_datasets_stratified(
            cfg, adaptation_ratio, seed, split_file, save_split
        )

        def worker_init_fn(worker_id):
            worker_seed = seed + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        batch_size = cfg.training.batch_size
        num_workers = cfg.training.num_workers

        # Create loaders for adaptation phase
        adaptation_train_loader = DataLoader(
            datasets["adaptation_train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

        adaptation_val_loader = DataLoader(
            datasets["adaptation_val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Create loaders for distillation phase
        distillation_train_loader = DataLoader(
            datasets["distillation_train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

        distillation_val_loader = DataLoader(
            datasets["distillation_val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Handle test loaders (can be dict or single dataset)
        test_datasets = datasets["test"]
        if isinstance(test_datasets, dict):
            test_loaders = {}
            for name, ds in test_datasets.items():
                test_loaders[name] = DataLoader(
                    ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
        else:
            test_loaders = DataLoader(
                test_datasets,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

        return {
            "adaptation_train": adaptation_train_loader,
            "adaptation_val": adaptation_val_loader,
            "distillation_train": distillation_train_loader,
            "distillation_val": distillation_val_loader,
            "test": test_loaders,
        }

    @staticmethod
    def _get_stratified_split_indices_path(cfg, seed: int) -> Path:
        """Get the path for storing/loading stratified split indices."""
        train_list = cfg.data.train if hasattr(cfg.data, "train") else []
        if isinstance(train_list, (list, ListConfig)):
            dataset_id = "_".join(train_list)
        else:
            dataset_id = getattr(cfg.data, "name", "unknown")
        return Path(
            f"splits/distillation_stratified_split_{dataset_id}_seed{seed}.json"
        )

    @staticmethod
    def _save_stratified_split_indices(
        path: Path,
        train_splits: Dict[str, Dict[str, List[int]]],
        val_splits: Dict[str, Dict[str, List[int]]],
        adaptation_ratio: float,
        seed: int,
    ):
        """Save stratified split indices to JSON file."""
        

        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "seed": seed,
            "adaptation_ratio": adaptation_ratio,
            "train_splits": train_splits,  # {dataset_name: {adapt_idx: [...], distill_idx: [...]}}
            "val_splits": val_splits,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Stratified split indices saved to: {path}")

    @staticmethod
    def _load_stratified_split_indices(path: Path) -> Optional[Dict]:
        """Load stratified split indices from JSON file if exists."""
        

        if not path.exists():
            return None

        with open(path, "r") as f:
            data = json.load(f)

        print(f"Stratified split indices loaded from: {path}")
        return data

    @staticmethod
    def build_distillation_datasets_stratified(
        cfg,
        adaptation_ratio: float = 0.5,
        seed: int = 42,
        split_file: Optional[str] = None,
        save_split: bool = True,
    ) -> Dict[str, Union[Dataset, Dict[str, Dataset]]]:
        """
        Build non-overlapping datasets with stratified splitting by dataset source.

        When using ConcatDataset (multiple datasets), this ensures each source
        dataset is split proportionally, maintaining dataset distribution balance.

        The split indices are saved to a JSON file to ensure consistency across
        separate training and distillation runs.

        Args:
            cfg: Hydra configuration object
            adaptation_ratio: Ratio of training data for adaptation (default: 0.5)
            seed: Random seed for reproducible splitting
            split_file: Optional path to split file. If None, auto-generated path is used.
            save_split: Whether to save split indices to file (default: True)

        Returns:
            Same structure as build_distillation_datasets()
        """

        # Check if using combined/dynamic datasets
        is_list_train = hasattr(cfg.data, "train") and isinstance(
            cfg.data.train, (list, ListConfig)
        )
        is_combined_type = hasattr(cfg.data, "type") and cfg.data.type in [
            "Combined",
            "Dynamic",
        ]

        if not (is_list_train or is_combined_type):
            # Fall back to simple splitting for single dataset
            return SegDatasetProcessor.build_distillation_datasets(
                cfg, adaptation_ratio, seed, split_file, save_split
            )

        # Determine split file path
        if split_file:
            split_path = Path(split_file)
        else:
            split_path = SegDatasetProcessor._get_stratified_split_indices_path(
                cfg, seed
            )

        # Try to load existing split
        existing_split = SegDatasetProcessor._load_stratified_split_indices(split_path)

        # Load individual datasets
        train_list = cfg.data.train if is_list_train else []
        val_list = getattr(cfg.data, "val", None)
        if (
            val_list is None
            or not isinstance(val_list, (list, ListConfig))
            or len(val_list) == 0
        ):
            val_list = train_list

        # Load datasets first to validate against existing split
        train_datasets = {}
        val_datasets = {}

        for name in train_list:
            train_datasets[name] = SegDatasetProcessor.load_dataset_from_config(
                cfg, name, split="train"
            )

        for name in val_list:
            val_datasets[name] = SegDatasetProcessor.load_dataset_from_config(
                cfg, name, split="val"
            )

        # Validate existing split if found
        if existing_split is not None:
            valid = True
            # Check train datasets
            for name in train_list:
                if name not in existing_split["train_splits"]:
                    print(
                        f"WARNING: Dataset {name} not found in split file. Regenerating split."
                    )
                    valid = False
                    break
                expected_size = len(
                    existing_split["train_splits"][name]["adapt_idx"]
                ) + len(existing_split["train_splits"][name]["distill_idx"])
                if expected_size != len(train_datasets[name]):
                    print(
                        f"WARNING: Dataset {name} size mismatch ({expected_size} vs {len(train_datasets[name])}). Regenerating split."
                    )
                    valid = False
                    break

            if valid and existing_split["seed"] != seed:
                print(
                    f"WARNING: Split file seed ({existing_split['seed']}) doesn't match requested seed ({seed}). Regenerating split."
                )
                valid = False

            if not valid:
                existing_split = None

        adaptation_train_datasets = []
        distillation_train_datasets = []
        adaptation_val_datasets = []
        distillation_val_datasets = []

        print(f"\n=== Stratified Distillation Split (seed={seed}) ===")

        if existing_split is not None:
            # Use loaded indices
            print("Using existing stratified split")

            for name in train_list:
                ds = train_datasets[name]
                adapt_idx = existing_split["train_splits"][name]["adapt_idx"]
                distill_idx = existing_split["train_splits"][name]["distill_idx"]

                adaptation_train_datasets.append(Subset(ds, adapt_idx))
                distillation_train_datasets.append(Subset(ds, distill_idx))

                print(
                    f"  {name} train: {len(ds)} -> adapt={len(adapt_idx)}, distill={len(distill_idx)}"
                )

            for name in val_list:
                ds = val_datasets[name]
                adapt_idx = existing_split["val_splits"][name]["adapt_idx"]
                distill_idx = existing_split["val_splits"][name]["distill_idx"]

                adaptation_val_datasets.append(Subset(ds, adapt_idx))
                distillation_val_datasets.append(Subset(ds, distill_idx))

                print(
                    f"  {name} val: {len(ds)} -> adapt={len(adapt_idx)}, distill={len(distill_idx)}"
                )
        else:
            # Generate new split
            rng = np.random.default_rng(seed)

            train_splits = {}
            val_splits = {}

            # Split each training dataset
            for name in train_list:
                ds = train_datasets[name]
                n = len(ds)
                n_adapt = int(n * adaptation_ratio)

                indices = rng.permutation(n)
                adapt_idx = indices[:n_adapt].tolist()
                distill_idx = indices[n_adapt:].tolist()

                train_splits[name] = {
                    "adapt_idx": adapt_idx,
                    "distill_idx": distill_idx,
                }

                adaptation_train_datasets.append(Subset(ds, adapt_idx))
                distillation_train_datasets.append(Subset(ds, distill_idx))

                print(
                    f"  {name} train: {n} -> adapt={len(adapt_idx)}, distill={len(distill_idx)}"
                )

            # Split each validation dataset
            for name in val_list:
                ds = val_datasets[name]
                n = len(ds)
                n_adapt = int(n * adaptation_ratio)

                indices = rng.permutation(n)
                adapt_idx = indices[:n_adapt].tolist()
                distill_idx = indices[n_adapt:].tolist()

                val_splits[name] = {"adapt_idx": adapt_idx, "distill_idx": distill_idx}

                adaptation_val_datasets.append(Subset(ds, adapt_idx))
                distillation_val_datasets.append(Subset(ds, distill_idx))

                print(
                    f"  {name} val: {n} -> adapt={len(adapt_idx)}, distill={len(distill_idx)}"
                )

            # Save split for future runs
            if save_split:
                SegDatasetProcessor._save_stratified_split_indices(
                    split_path,
                    train_splits,
                    val_splits,
                    adaptation_ratio,
                    seed,
                )

        # Combine into ConcatDatasets
        adaptation_train = ConcatDataset(adaptation_train_datasets)
        distillation_train = ConcatDataset(distillation_train_datasets)
        adaptation_val = ConcatDataset(adaptation_val_datasets)
        distillation_val = ConcatDataset(distillation_val_datasets)

        # Load test datasets (unchanged)
        test_datasets = {}
        test_list = getattr(cfg.data, "test", [])

        if isinstance(test_list, (list, ListConfig)):
            for name in test_list:
                SegDatasetProcessor._add_test_dataset_with_unfiltered(
                    cfg, name, test_datasets
                )
        elif isinstance(test_list, str):
            SegDatasetProcessor._add_test_dataset_with_unfiltered(
                cfg, test_list, test_datasets
            )

        print(
            f"\nTotal: adapt_train={len(adaptation_train)}, distill_train={len(distillation_train)}"
        )
        print(
            f"       adapt_val={len(adaptation_val)}, distill_val={len(distillation_val)}"
        )

        return {
            "adaptation_train": adaptation_train,
            "adaptation_val": adaptation_val,
            "distillation_train": distillation_train,
            "distillation_val": distillation_val,
            "test": test_datasets,  # Always return dict (empty if no test datasets)
        }

    @staticmethod
    def _add_test_dataset_with_unfiltered(cfg, name, test_datasets_dict):
        """Load test dataset and add unfiltered version if filter_empty_masks is enabled."""
        print(f"Loading Test dataset: {name}")
        ds: Dataset = SegDatasetProcessor.load_dataset_from_config(cfg, name, split="test")
        test_datasets_dict[name] = ds

        # If dataset has filter_empty_masks enabled, also add unfiltered version
        if hasattr(ds, "filter_empty_masks") and ds.filter_empty_masks:
            
            # Use copy to avoid reloading and re-filtering
            ds_unfiltered = copy.copy(ds)
            
            # Restore original unfiltered file lists
            ds_unfiltered.image_files = list(ds.image_files_unfiltered)
            ds_unfiltered.mask_files = list(ds.mask_files_unfiltered)
            test_datasets_dict[f"{name}_unfiltered"] = ds_unfiltered
            print(
                f"  -> Also added {name}_unfiltered (original: {len(ds_unfiltered)} samples)"
            )