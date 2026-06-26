import random
from typing import Optional, List, Dict, Union, Type, Sequence

import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from omegaconf import OmegaConf, ListConfig, DictConfig

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


# Image-level class label conventions used across datasets:
#   0 -> normal (image-level, no lesion present)
#   1 -> benign
#   2 -> malignant
# `class_labels` attribute on each dataset stores these per-sample.
_DEFAULT_CLASS_WEIGHTS = {0: 1.0, 1: 1.0, 2: 1.0}


def _extract_class_labels(ds: Dataset) -> List[int]:
    """Best-effort extraction of per-sample image-level class labels.

    Each ultrasound dataset stores them under `class_labels`. Datasets that do
    not expose them (e.g. BUS_UCLM in binary mode) fall back to label=1.
    """
    labels = getattr(ds, "class_labels", None)
    if labels is None:
        return [1] * len(ds)
    return list(labels)


def _build_balanced_sampler(
    train_datasets: Sequence[Dataset],
    train_names: Sequence[str],
    sampling_cfg: DictConfig,
) -> Optional[WeightedRandomSampler]:
    """Build a WeightedRandomSampler combining dataset- and class-level reweighting.

    Per-sample weight is the product of:
      * dataset weight w_d ∝ N_d^alpha / N_d   (so sampling prob p_d ∝ N_d^alpha)
      * class weight   w_c  (image-level normal/benign/malignant balance)

    When `augment_train=True`, each dataset is loaded twice (original + augmented),
    so `train_datasets` may have 2 entries per name. They share the same name/N_d
    and contribute jointly to N_d.
    """
    if not sampling_cfg.get("enabled", False):
        return None
    if not train_datasets:
        return None

    alpha = float(sampling_cfg.get("alpha", 0.5))
    class_weights_cfg = sampling_cfg.get("class_weights", None)
    if class_weights_cfg is None:
        class_weights = dict(_DEFAULT_CLASS_WEIGHTS)
    else:
        # OmegaConf DictConfig -> {str: float}; normalize keys to int
        raw = OmegaConf.to_container(class_weights_cfg, resolve=True) \
            if isinstance(class_weights_cfg, DictConfig) else dict(class_weights_cfg)
        class_weights = {int(k): float(v) for k, v in raw.items()}
        for k, v in _DEFAULT_CLASS_WEIGHTS.items():
            class_weights.setdefault(k, v)

    normal_cap = sampling_cfg.get("normal_cap", None)
    num_samples_cfg = sampling_cfg.get("num_samples", None)
    replacement = bool(sampling_cfg.get("replacement", True))
    # Decouple the two effects: when True, each dataset's total sampling mass is
    # renormalized to exactly N_d^alpha *after* class reweighting, so `alpha`
    # controls dataset balance and `class_weights` control the class mix *within*
    # each dataset independently. When False, the legacy multiplicative behaviour
    # is used (class_weights leak into the dataset draw probability). This is a
    # no-op when class_weights are uniform, so it is safe to leave on.
    decouple = bool(sampling_cfg.get("decouple_dataset_class", True))

    # Group dataset entries by source name (original + augmented share weight per N_d)
    name_to_total_n: Dict[str, int] = {}
    per_ds_meta = []  # list of (name, n, labels)
    for ds, name in zip(train_datasets, train_names):
        labels = _extract_class_labels(ds)
        n = len(ds)
        per_ds_meta.append((name, n, labels))
        name_to_total_n[name] = name_to_total_n.get(name, 0) + n

    # Step 1: per-sample class weight only. The dataset factor is applied in
    # step 3 so that `normal_cap` (step 2) and the optional decoupling can reason
    # about within-dataset class fractions cleanly.
    total_samples = sum(n for _, n, _ in per_ds_meta)
    weights_t = torch.zeros(total_samples, dtype=torch.double)
    block_ranges: Dict[str, List[tuple]] = {}  # name -> [(start, end), ...]
    class_count_by_name: Dict[str, Dict[int, int]] = {}
    missing_label_names: List[str] = []
    idx = 0
    for ds, (name, n, labels) in zip(train_datasets, per_ds_meta):
        if getattr(ds, "class_labels", None) is None:
            missing_label_names.append(name)
        ccount = class_count_by_name.setdefault(name, {})
        cur = torch.empty(n, dtype=torch.double)
        for j, lbl in enumerate(labels):
            ccount[int(lbl)] = ccount.get(int(lbl), 0) + 1
            cur[j] = class_weights.get(int(lbl), 1.0)
        weights_t[idx: idx + n] = cur
        block_ranges.setdefault(name, []).append((idx, idx + n))
        idx += n

    # Step 2: optional cap on the *expected fraction* of normal-class draws per
    # dataset. If a dataset has too much normal share relative to `normal_cap`,
    # scale its normal-class weights down so the expected normal fraction within
    # that dataset equals `normal_cap`. Operates on class weights only; invariant
    # to the uniform per-dataset factor applied in step 3.
    if normal_cap is not None:
        cap = float(normal_cap)
        idx = 0
        for name, n, labels in per_ds_meta:
            cur_w = weights_t[idx: idx + n]
            normal_mask = torch.tensor([int(l) == 0 for l in labels])
            total = cur_w.sum().item()
            normal_sum = cur_w[normal_mask].sum().item()
            if total > 0 and normal_sum > 0:
                cur_frac = normal_sum / total
                if cur_frac > cap:
                    # scale normal weights so normal_sum' / (total - normal_sum + normal_sum') = cap
                    other = total - normal_sum
                    target_normal_sum = cap * other / max(1.0 - cap, 1e-9)
                    scale = target_normal_sum / normal_sum
                    cur_w[normal_mask] = cur_w[normal_mask] * scale
                    weights_t[idx: idx + n] = cur_w
            idx += n

    # Step 3: apply the dataset factor so dataset draw probability ∝ N_d^alpha.
    #   decouple=True  -> renormalize each dataset's mass to exactly N_d^alpha,
    #                     keeping the within-dataset class mix from steps 1-2.
    #                     => p_d ∝ N_d^alpha regardless of class composition.
    #   decouple=False -> legacy: multiply every sample by N_d^alpha / N_d, so a
    #                     skewed class mix leaks into the dataset draw probability.
    # Both are identical when class_weights are uniform and normal_cap is null.
    for name, ranges in block_ranges.items():
        N_d = max(name_to_total_n[name], 1)
        if decouple:
            s_d = sum(weights_t[a:b].sum().item() for a, b in ranges)
            if s_d <= 0:
                continue
            scale = (N_d ** alpha) / s_d
        else:
            scale = (N_d ** alpha) / N_d
        for a, b in ranges:
            weights_t[a:b] *= scale

    if missing_label_names and class_weights != _DEFAULT_CLASS_WEIGHTS:
        print(f"[sampler]   WARNING: no class_labels for {sorted(set(missing_label_names))}; "
              f"class_weights treated all their samples as benign (label=1).")

    num_samples = int(num_samples_cfg) if num_samples_cfg else int(weights_t.numel())
    generator = None
    seed = sampling_cfg.get("seed", None)
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    # Diagnostics
    print("[sampler] BalancedSampler enabled")
    print(f"[sampler]   alpha={alpha} class_weights={class_weights} "
          f"normal_cap={normal_cap} decouple={decouple} "
          f"num_samples={num_samples} replacement={replacement}")
    all_w_sum = weights_t.sum().item()
    for name, total_n in name_to_total_n.items():
        ccount = class_count_by_name.get(name, {})
        ds_w_sum = 0.0
        running = 0
        for nm, n2, _ in per_ds_meta:
            if nm == name:
                ds_w_sum += weights_t[running: running + n2].sum().item()
            running += n2
        p_d = ds_w_sum / all_w_sum if all_w_sum > 0 else 0.0
        print(f"[sampler]   {name}: N={total_n} class_counts={ccount} p_draw={p_d:.3f}")

    return WeightedRandomSampler(
        weights=weights_t, num_samples=num_samples, replacement=replacement,
        generator=generator,
    )


class SegDatasetProcessor:
    @staticmethod
    def _sync_img_size_with_sam_type(cfg):
        """Sync data/model img_size from sam_type when enabled.

        Applies to:
        - train path: cfg.model.sam_type
        - distill path: cfg.teacher_cfg.sam_type (then cfg.student_cfg.sam_type)
        """
        data_cfg = cfg.get("data", {})
        if not data_cfg.get("auto_img_size_by_sam_type", True):
            return

        sam_cfg = None
        if "model" in cfg and isinstance(cfg.model, DictConfig) and cfg.model.get("sam_type") is not None:
            sam_cfg = cfg.model
        elif "teacher_cfg" in cfg and cfg.teacher_cfg.get("sam_type") is not None:
            sam_cfg = cfg.teacher_cfg
        elif "student_cfg" in cfg and cfg.student_cfg.get("sam_type") is not None:
            sam_cfg = cfg.student_cfg
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
    def load_dataset_from_config(cfg, name, split, force_external=False, transform=False):
        """Helper to load dataset config and instantiate.

        Args:
            force_external: If True, set usage="external" (for external validation sets).
                            If False, keep the config's default usage (for internal splits).
            transform: Whether to apply data augmentation.
        """
        config_path = Path(f"config/data/{name}.yaml")
        if not config_path.exists():
            raise ValueError(f"Config for {name} not found at {config_path}")

        data_cfg = OmegaConf.load(config_path)

        # Override global settings
        for attr in ["img_size", "num_classes", "normalization"]:
            if hasattr(cfg.data, attr):
                setattr(data_cfg, attr, getattr(cfg.data, attr))

        if force_external:
            data_cfg.usage = "external"

        dataset_class = get_dataset_class(data_cfg.name)
        return dataset_class(data_cfg, split=split, transform=transform)

    @staticmethod
    def build_dataset(cfg, return_components: bool = False):
        """Build train, val, and test datasets according to config.

        Args:
            return_components: If True, returns a dict including raw per-dataset
                lists (`train_components`, `train_component_names`) so callers
                can build a balanced sampler aware of dataset boundaries.
        """
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

        augment_train = getattr(cfg.data, "augment_train", True)
        train_datasets = []
        train_component_names: List[str] = []
        for n in train_list:
            original = SegDatasetProcessor.load_dataset_from_config(cfg, n, "train", transform=False)
            train_datasets.append(original)
            train_component_names.append(n)
            if augment_train:
                augmented = SegDatasetProcessor.load_dataset_from_config(cfg, n, "train", transform=True)
                train_datasets.append(augmented)
                train_component_names.append(n)
        val_datasets = [
            SegDatasetProcessor.load_dataset_from_config(cfg, n, "val")
            for n in val_list
        ]

        aug_note = " (original + augmented)" if augment_train else ""
        print(
            f"Loaded Train: {', '.join(train_list)} ({sum(len(d) for d in train_datasets)} samples{aug_note})"
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

        train_concat = (
            ConcatDataset(train_datasets)
            if len(train_datasets) > 1
            else train_datasets[0]
        )
        val_concat = (
            ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
        )

        if return_components:
            return {
                "train": train_concat,
                "val": val_concat,
                "test": test_datasets,
                "train_components": train_datasets,
                "train_component_names": train_component_names,
            }

        return (train_concat, val_concat, test_datasets)

    @staticmethod
    def build_data_loaders(cfg):
        """Standard trainer data loader builder."""
        components = SegDatasetProcessor.build_dataset(cfg, return_components=True)
        train_ds = components["train"]
        val_ds = components["val"]
        test_ds_dict = components["test"]

        batch_size = cfg.training.batch_size
        num_workers = cfg.training.num_workers

        sampler = _build_balanced_sampler(
            components["train_components"],
            components["train_component_names"],
            cfg.data.get("sampling", OmegaConf.create({})),
        )
        shuffle = sampler is None
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
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
        return SegDatasetProcessor.build_dataset(cfg, return_components=True)

    @staticmethod
    def build_distillation_data_loaders(cfg):
        """Distillation-specific data loader builder using shared train/val."""
        datasets = SegDatasetProcessor.build_distillation_datasets(cfg)

        batch_size = cfg.training.batch_size
        num_workers = cfg.training.num_workers

        sampler = _build_balanced_sampler(
            datasets["train_components"],
            datasets["train_component_names"],
            cfg.data.get("sampling", OmegaConf.create({})),
        )

        def _get_loader(ds, shuffle, sampler=None):
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle and sampler is None,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
            )

        return {
            "train": _get_loader(datasets["train"], True, sampler=sampler),
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
