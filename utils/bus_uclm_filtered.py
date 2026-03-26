import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from utils.ultrasound_datasets import BaseUltrasoundDataset


class BUS_UCLM_filtered(BaseUltrasoundDataset):
    """
    BUS-UCLM dataset loader with precomputed empty-mask filtering.

    This class avoids re-running mask filtering every time by caching filtered
    file pairs into a JSON index and reusing it on subsequent runs.

    Config options (all optional):
    - partition_dir: dataset partition directory (default: "partitions")
    - extensions: allowed image extensions
    - filter_empty_masks: whether to exclude background-only masks (default: True)
    - filtered_index_dir: directory name for cache files (default: ".filtered_indices")
    - rebuild_filtered_index: force rebuilding cached index (default: False)
    """

    def __init__(self, cfg, split, transform: Optional[bool] = False):
        super().__init__(cfg, split, transform)

        self.root = Path(cfg.path.root)
        self.partition_dir = getattr(cfg, "partition_dir", "partitions")
        self.extensions = tuple(
            getattr(cfg, "extensions", (".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
        )
        self.filter_empty_masks = bool(getattr(cfg, "filter_empty_masks", True))
        self.rebuild_filtered_index = bool(getattr(cfg, "rebuild_filtered_index", False))

        filtered_index_dir_name = getattr(
            cfg, "filtered_index_dir", ".filtered_indices"
        )
        self.filtered_index_dir = self.root / self.partition_dir / filtered_index_dir_name
        self.filtered_index_dir.mkdir(parents=True, exist_ok=True)

        target_split = "val" if split in ["val", "valid"] else split
        self.target_split = target_split

        self.image_files, self.mask_files = self._load_or_build_pairs()

    def _load_or_build_pairs(self) -> Tuple[List[Path], List[Path]]:
        index_path = self._index_path()
        if index_path.exists() and not self.rebuild_filtered_index:
            return self._load_pairs_from_index(index_path)

        image_files, mask_files = self._build_pairs_from_source()
        self._save_pairs_to_index(index_path, image_files, mask_files)
        return image_files, mask_files

    def _index_path(self) -> Path:
        usage = getattr(self.cfg, "usage", "external")
        safe_usage = str(usage).replace("/", "_")
        safe_split = str(self.target_split).replace("/", "_")
        return self.filtered_index_dir / (
            f"bus_uclm_filtered_usage-{safe_usage}_split-{safe_split}_seed-{self.seed}.json"
        )

    def _build_pairs_from_source(self) -> Tuple[List[Path], List[Path]]:
        usage = getattr(self.cfg, "usage", "external")

        if usage == "train":
            if self.target_split in ["train", "val"]:
                image_dir = self.root / self.partition_dir / "train" / "images"
                mask_dir = self.root / self.partition_dir / "train" / "masks"

                self._validate_dir(image_dir, "Image")
                self._validate_dir(mask_dir, "Mask")

                all_image_files, all_mask_files = self._get_paired_files(
                    image_dir, mask_dir, self.extensions
                )
                image_files, mask_files = self._split_train_val(
                    all_image_files, all_mask_files, self.target_split
                )
            else:
                image_dir = self.root / self.partition_dir / "test" / "images"
                mask_dir = self.root / self.partition_dir / "test" / "masks"

                self._validate_dir(image_dir, "Image")
                self._validate_dir(mask_dir, "Mask")

                image_files, mask_files = self._get_paired_files(
                    image_dir, mask_dir, self.extensions
                )
        else:
            image_dir = self.root / "data" / "images"
            mask_dir = self.root / "data" / "masks"

            self._validate_dir(image_dir, "Image")
            self._validate_dir(mask_dir, "Mask")

            image_files, mask_files = self._get_paired_files(
                image_dir, mask_dir, self.extensions
            )

        if self.filter_empty_masks:
            image_files, mask_files = self._filter_empty_masks(image_files, mask_files)

        return image_files, mask_files

    def _validate_dir(self, directory: Path, kind: str) -> None:
        if not directory.exists():
            raise ValueError(f"{kind} directory does not exist: {directory}")

    def _save_pairs_to_index(
        self, index_path: Path, image_files: List[Path], mask_files: List[Path]
    ) -> None:
        payload = {
            "usage": getattr(self.cfg, "usage", "external"),
            "split": self.target_split,
            "seed": self.seed,
            "filter_empty_masks": self.filter_empty_masks,
            "num_pairs": len(image_files),
            "pairs": [],
        }

        for img_path, mask_path in zip(image_files, mask_files):
            payload["pairs"].append(
                {
                    "image": self._to_serializable_path(Path(img_path)),
                    "mask": self._to_serializable_path(Path(mask_path)),
                }
            )

        with index_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _load_pairs_from_index(self, index_path: Path) -> Tuple[List[Path], List[Path]]:
        with index_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        pairs = payload.get("pairs", [])
        image_files: List[Path] = []
        mask_files: List[Path] = []

        for pair in pairs:
            img_path = self._from_serializable_path(pair["image"])
            mask_path = self._from_serializable_path(pair["mask"])

            if img_path.exists() and mask_path.exists():
                image_files.append(img_path)
                mask_files.append(mask_path)

        if len(image_files) != len(pairs):
            print(
                "Warning: Some cached paths do not exist anymore. "
                "Rebuilding index is recommended (set rebuild_filtered_index=true)."
            )

        return image_files, mask_files

    def _to_serializable_path(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            return str(p)

    def _from_serializable_path(self, value: str) -> Path:
        p = Path(value)
        if p.is_absolute():
            return p
        return self.root / p

    def _get_paired_files(
        self, image_dir: Path, mask_dir: Path, extensions: Tuple[str, ...]
    ) -> Tuple[List[Path], List[Path]]:
        pairs = []
        for ext in extensions:
            for img_path in image_dir.glob(f"*{ext}"):
                mask_path = mask_dir / img_path.name
                if not mask_path.exists():
                    mask_path_upper = mask_dir / f"{img_path.stem}{img_path.suffix.upper()}"
                    if mask_path_upper.exists():
                        mask_path = mask_path_upper
                    else:
                        print(f"Warning: Mask not found for {img_path.name} in {mask_dir}")
                        continue
                pairs.append((img_path, mask_path))

            for img_path in image_dir.glob(f"*{ext.upper()}"):
                if any(str(p[0]) == str(img_path) for p in pairs):
                    continue

                mask_path = mask_dir / img_path.name
                if not mask_path.exists():
                    mask_path_lower = mask_dir / f"{img_path.stem}{img_path.suffix.lower()}"
                    if mask_path_lower.exists():
                        mask_path = mask_path_lower
                    else:
                        print(f"Warning: Mask not found for {img_path.name} in {mask_dir}")
                        continue

                pairs.append((img_path, mask_path))

        pairs.sort(key=lambda x: str(x[0]))

        if not pairs:
            return [], []

        images, masks = zip(*pairs)
        return list(images), list(masks)

    def _split_train_val(self, image_files, mask_files, split_type):
        random.seed(self.seed)

        train_imgs, val_imgs, train_masks, val_masks = train_test_split(
            image_files, mask_files, test_size=0.2, random_state=self.seed
        )

        if split_type == "train":
            return train_imgs, train_masks
        return val_imgs, val_masks

    def _filter_empty_masks(self, image_files, mask_files):
        print(f"Filtering empty masks... (Total before: {len(image_files)})")
        filtered_images = []
        filtered_masks = []

        try:
            from tqdm import tqdm

            iterator = tqdm(
                zip(image_files, mask_files),
                total=len(image_files),
                desc="Filtering masks",
            )
        except ImportError:
            iterator = zip(image_files, mask_files)

        for img_path, mask_path in iterator:
            try:
                mask = Image.open(mask_path).convert("L")
                if np.array(mask).max() > 0:
                    filtered_images.append(img_path)
                    filtered_masks.append(mask_path)
            except Exception as e:
                print(f"Warning: Error reading mask {mask_path} during filtering: {e}")

        print(
            f"Filtering complete. Kept {len(filtered_images)} pairs "
            f"(Removed {len(image_files) - len(filtered_images)})"
        )
        return filtered_images, filtered_masks

    def _convert_rgb_mask_to_classes(self, mask_rgb: np.ndarray) -> np.ndarray:
        mask = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)

        red_mask = (
            (mask_rgb[:, :, 0] == 255)
            & (mask_rgb[:, :, 1] == 0)
            & (mask_rgb[:, :, 2] == 0)
        )
        mask[red_mask] = 2

        green_mask = (
            (mask_rgb[:, :, 0] == 0)
            & (mask_rgb[:, :, 1] == 255)
            & (mask_rgb[:, :, 2] == 0)
        )
        mask[green_mask] = 1

        return mask

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        image = self._load_image(image_path)

        mask_rgb = self._load_mask(mask_path, mode="RGB")
        mask_rgb_array = np.array(mask_rgb)
        mask_array = self._convert_rgb_mask_to_classes(mask_rgb_array)
        mask = Image.fromarray(mask_array)

        image, mask = self._resize_images(image, mask)

        if self.transform:
            image, mask = self._joint_transform(image, mask)

        return self._create_tensors(image, mask, Path(image_path).stem)
