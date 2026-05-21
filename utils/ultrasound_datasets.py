from pathlib import Path
from typing import List, Tuple, Optional
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from transformers import SegformerImageProcessor
import json


def _format_mask(mask, num_classes):
    """
    Format mask into [C, H, W] float 0/1 tensor.

    Args:
        mask: PIL Image or [H, W] array
        num_classes: Number of classes
    Returns:
        torch.Tensor: [C, H, W] float. For num_classes=1, shape is [1, H, W].
    """
    mask_np = np.array(mask)
    if num_classes == 1:
        # Binary case: foreground is any non-zero pixel.
        # Works for both class-index masks (0/1) and intensity masks (0/255).
        mask_np = (mask_np > 0).astype(np.float32)
        return torch.from_numpy(mask_np).unsqueeze(0).float()
    else:
        # Multi-class case: convert to one-hot [C, H, W]
        # In this project, multi-class labels are usually 0, 1, 2...
        mask_long = torch.from_numpy(mask_np.astype(np.int64)).long()
        # Handle cases where mask_np might have values >= num_classes due to noise if any
        mask_long = torch.clamp(mask_long, 0, num_classes - 1)
        mask_one_hot = torch.nn.functional.one_hot(mask_long, num_classes=num_classes)
        return mask_one_hot.permute(2, 0, 1).float()


class BaseUltrasoundDataset(Dataset):
    """
    Base class for ultrasound segmentation datasets.
    Provides shared functionality for image/mask loading, transformations, and tensor creation.
    """

    # Subclasses that support multiclass should set this to True
    SUPPORTS_MULTICLASS = False

    def __init__(self, cfg, split, transform: Optional[bool] = False):
        """
        Initialize base dataset with common configuration.

        Args:
            cfg: Configuration object with dataset parameters
            split: Dataset split ('train', 'val', 'test')
            transform: Whether to apply data augmentation
        """
        self.cfg = cfg
        self.num_classes = cfg.num_classes
        self.transform = transform
        self.split = split

        # Multiclass mode is determined solely by num_classes (>1 → multiclass).
        self.multiclass = self.num_classes > 1

        # Image and mask sizes
        img_size = int(cfg.img_size)
        self.image_size = (img_size, img_size)
        self.low_res_size = img_size // 4, img_size // 4

        # Common configuration
        self.seed = getattr(cfg, "seed", 42)
        self.normalization = getattr(cfg, "normalization", "imagenet")
        self.task_type = getattr(
            cfg, "task_type", "tumor"
        )  # For conditional transforms

    def _load_image(self, path: Path) -> Image.Image:
        """
        Load an image from path with error handling.

        Args:
            path: Path to image file

        Returns:
            PIL Image in RGB mode

        Raises:
            ValueError: If image cannot be loaded
        """
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Error loading image {path}: {e}")

    def _load_mask(self, path: Path, mode: str = "L") -> Image.Image:
        """
        Load a mask from path with error handling.

        Args:
            path: Path to mask file
            mode: PIL image mode ('L' for grayscale, 'RGB' for color)

        Returns:
            PIL Image in specified mode

        Raises:
            ValueError: If mask cannot be loaded
        """
        try:
            return Image.open(path).convert(mode)
        except Exception as e:
            raise ValueError(f"Error loading mask {path}: {e}")

    def _resize_images(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Resize image and mask to target size.

        Args:
            image: PIL Image
            mask: PIL Image

        Returns:
            Tuple of (resized_image, resized_mask)
        """
        image = TF.resize(image, self.image_size, interpolation=Image.BILINEAR)
        mask = TF.resize(mask, self.image_size, interpolation=Image.NEAREST)
        return image, mask

    def _apply_normalization(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization to image tensor.

        Args:
            image_tensor: Image tensor [C, H, W]

        Returns:
            Normalized image tensor
        """
        if self.normalization == "imagenet":
            return T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
                image_tensor
            )
        return image_tensor

    def _create_tensors(
        self, image: Image.Image, mask: Image.Image, filename: str = "",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Convert PIL images to tensors with proper formatting.

        Args:
            image: PIL Image (RGB)
            mask: PIL Image (grayscale or class labels)
            filename: Source image filename (stem) for visualization.

        Returns:
            Tuple of (image_tensor, mask_tensor, low_res_mask_tensor, filename)
        """
        # Image tensorization and normalization
        image_tensor = TF.to_tensor(image)
        image_tensor = self._apply_normalization(image_tensor)

        # Mask tensor creation
        mask_tensor = _format_mask(mask, self.num_classes)

        # Low-res mask
        low_res_mask_img = mask.resize(self.low_res_size, Image.NEAREST)
        low_res_tensor = _format_mask(low_res_mask_img, self.num_classes)

        return image_tensor, mask_tensor, low_res_tensor, filename

    def _apply_class_label(self, mask: Image.Image, class_label: int) -> Image.Image:
        """
        In multiclass mode, remap foreground pixels in a binary mask to class_label.

        Args:
            mask: Grayscale PIL Image (binary or 0/255 values)
            class_label: Integer class label to assign to foreground pixels

        Returns:
            PIL Image with foreground pixels set to class_label
        """
        mask_np = np.array(mask, dtype=np.uint8)
        # Normalize: treat any non-zero value as foreground
        if mask_np.max() > 2:
            foreground = mask_np > 127
        else:
            foreground = mask_np > 0
        result = np.zeros_like(mask_np)
        result[foreground] = class_label
        return Image.fromarray(result)

    def _joint_transform(
        self, image: Image.Image, label: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Apply joint augmentation transformations to image and label.

        Transformations include:
        - Horizontal/vertical flips (conditional on task_type for tumor tasks)
        - Random rotation (-30 to 30 degrees)
        - Gamma correction
        - Random scaling and cropping
        - Contrast adjustment

        Args:
            image: PIL Image
            label: PIL Image (mask)

        Returns:
            Tuple of (transformed_image, transformed_label)
        """
        # Horizontal and vertical flips (conditional for tumor task)
        if self.task_type == "tumor":
            if random.random() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)

            if random.random() > 0.5:
                image = TF.vflip(image)
                label = TF.vflip(label)

        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-30, 30)
            image = TF.rotate(image, angle)
            label = TF.rotate(label, angle)

        # Gamma correction
        if random.random() > 0.5:
            g = np.random.randint(10, 25) / 10.0
            image_np = np.array(image)
            image_np = (np.power(image_np / 255, 1.0 / g)) * 255
            image_np = image_np.astype(np.uint8)
            image = Image.fromarray(image_np)

        # Random scaling and cropping
        if random.random() > 0.5:
            scale = np.random.uniform(1, 1.3)
            h, w = self.image_size
            new_h, new_w = int(h * scale), int(w * scale)
            image = TF.resize(image, (new_h, new_w), interpolation=Image.BILINEAR)
            label = TF.resize(label, (new_h, new_w), interpolation=Image.NEAREST)
            i, j, crop_h, crop_w = T.RandomCrop.get_params(image, self.image_size)
            image = TF.crop(image, i, j, crop_h, crop_w)
            label = TF.crop(label, i, j, crop_h, crop_w)

        # Contrast adjustment
        if random.random() > 0.5:
            contr_tf = T.ColorJitter(contrast=(0.8, 2.0))
            image = contr_tf(image)

        return image, label


class BUID(BaseUltrasoundDataset):
    """
    BUID Dataset (Breast Ultrasound Images Dataset)
    - Total: 232 cases (109 Benign, 123 Malignant)
    - Binary segmentation: background (0), lesion (1)
    - Multiclass: Benign=1, Malignant=2
    - RGB images with sizes 590x590 or 600x600
    - File structure:
        Benign/
            XXX Benign Image.bmp
            XXX Benign Mask.tif (binary mask)
        Malignant/
            XXX Malignant Image.bmp
            XXX Malignant Mask.tif (binary mask)
    - Random split by class: 70% train, 15% val, 15% test
    """

    SUPPORTS_MULTICLASS = True
    # Map directory name -> class label for multiclass mode
    CLASS_LABEL_MAP = {"Benign": 1, "Malignant": 2}

    def __init__(self, cfg, split, transform: Optional[bool] = False):
        super().__init__(cfg, split, transform)

        self.root = Path(cfg.path.root)
        self.classes = cfg.classes
        self.usage = getattr(cfg, "usage", "external")

        self.images, self.masks, self.class_labels = self._unzip_triples(
            self._collect_paired_files()
        )
        if self.usage != "external":
            self.images, self.masks, self.class_labels = self._split_dataset(
                self.images, self.masks, self.class_labels
            )

    def _split_dataset(self, images, masks, class_labels):
        # Group by class
        benign_triples = []
        malignant_triples = []

        for img, msk, lbl in zip(images, masks, class_labels):
            if "Benign" in str(img):
                benign_triples.append((img, msk, lbl))
            elif "Malignant" in str(img):
                malignant_triples.append((img, msk, lbl))
            else:
                benign_triples.append((img, msk, lbl))

        # 70% train, 15% val, 15% test
        def split_class(triples, seed):
            if not triples:
                return [], [], []
            train, temp = train_test_split(triples, test_size=0.3, random_state=seed)
            val, test = train_test_split(temp, test_size=0.5, random_state=seed)
            return train, val, test

        b_train, b_val, b_test = split_class(benign_triples, self.seed)
        m_train, m_val, m_test = split_class(malignant_triples, self.seed)

        train_triples = b_train + m_train
        val_triples = b_val + m_val
        test_triples = b_test + m_test

        target_split = "val" if self.split in ["val", "valid"] else self.split

        if target_split == "train":
            selected = train_triples
        elif target_split == "val":
            selected = val_triples
        elif target_split == "test":
            selected = test_triples
        else:
            raise ValueError(f"Invalid split: {self.split}")

        selected.sort(key=lambda x: str(x[0]))
        return self._unzip_triples(selected)

    def _collect_paired_files(self) -> List[Tuple[Path, Path, int]]:
        pairs = []
        for class_dir_name in self.classes.values():
            class_dir = self.root / class_dir_name
            if not class_dir.exists():
                print(f"Warning: Class directory does not exist: {class_dir}")
                continue

            class_label = self.CLASS_LABEL_MAP.get(class_dir_name, 1)
            image_pattern = f"* {class_dir_name} Image.bmp"
            for img_path in class_dir.glob(image_pattern):
                base_name = img_path.stem.replace(" Image", " Mask")
                mask_path = class_dir / f"{base_name}.tif"

                if mask_path.exists():
                    pairs.append((img_path, mask_path, class_label))
                else:
                    print(f"Warning: Mask file not found: {mask_path}")

        pairs.sort(key=lambda x: str(x[0]))
        return pairs

    def _unzip_triples(self, triples):
        if not triples:
            return [], [], []
        images, masks, labels = zip(*triples)
        return list(images), list(masks), list(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        class_label = self.class_labels[idx]

        image = self._load_image(image_path)
        mask = self._load_mask(mask_path, mode="L")

        if self.multiclass:
            mask = self._apply_class_label(mask, class_label)

        image, mask = self._resize_images(image, mask)

        if self.transform:
            image, mask = self._joint_transform(image, mask)

        return self._create_tensors(image, mask, Path(image_path).stem)


class BUS_UCLM(BaseUltrasoundDataset):
    """
    BUS-UCLM Dataset
    - Total: 683 images from 38 patients
    - Classes: Normal (419), Benign (174), Malignant (90)
    - Image size: 768x1024 (resized to config.img_size)
    - Mask encoding:
        Red (255, 0, 0) -> Malignant (class 2)
        Green (0, 255, 0) -> Benign (class 1)
        Black (0, 0, 0) -> Background/Normal (class 0)
    - Multiclass: uses RGB color to determine class label (Benign=1, Malignant=2)
    - Binary: any colored region is foreground (1)
    - Split: Pre-defined train/test split in partitions folder
    """

    SUPPORTS_MULTICLASS = True

    def __init__(self, cfg, split, transform: Optional[bool] = False):
        super().__init__(cfg, split, transform)

        self.root = Path(cfg.path.root)
        self.partition_dir = getattr(cfg, "partition_dir", "partitions")
        self.extensions = getattr(
            cfg, "extensions", (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        )
        self.filter_empty_masks = getattr(cfg, "filter_empty_masks", False)

        if self.cfg.usage == "train":
            # Support both 'val' and 'valid'
            target_split = "val" if split in ["val", "valid"] else split

            if target_split in ["train", "val"]:
                self.image_dir = self.root / self.partition_dir / "train" / "images"
                self.mask_dir = self.root / self.partition_dir / "train" / "masks"

                if not self.image_dir.exists():
                    raise ValueError(
                        f"Image directory does not exist: {self.image_dir}"
                    )
                if not self.mask_dir.exists():
                    raise ValueError(f"Mask directory does not exist: {self.mask_dir}")

                all_image_files, all_mask_files = self._get_paired_files(
                    self.image_dir, self.mask_dir, self.extensions
                )

                self.image_files, self.mask_files = self._split_train_val(
                    all_image_files, all_mask_files, target_split
                )
            else:  # test
                self.image_dir = self.root / self.partition_dir / "test" / "images"
                self.mask_dir = self.root / self.partition_dir / "test" / "masks"

                if not self.image_dir.exists():
                    raise ValueError(
                        f"Image directory does not exist: {self.image_dir}"
                    )
                if not self.mask_dir.exists():
                    raise ValueError(f"Mask directory does not exist: {self.mask_dir}")

                self.image_files, self.mask_files = self._get_paired_files(
                    self.image_dir, self.mask_dir, self.extensions
                )
        else:  # external validation - use all data
            self.image_dir = self.root / "data" / "images"
            self.mask_dir = self.root / "data" / "masks"

            if not self.image_dir.exists():
                raise ValueError(f"Image directory does not exist: {self.image_dir}")
            if not self.mask_dir.exists():
                raise ValueError(f"Mask directory does not exist: {self.mask_dir}")

            self.image_files, self.mask_files = self._get_paired_files(
                self.image_dir, self.mask_dir, self.extensions
            )

        # Apply filtering if requested (only for train split usually, but logic allows any)
        # Always store unfiltered version first
        self.image_files_unfiltered = list(self.image_files)
        self.mask_files_unfiltered = list(self.mask_files)

        if self.filter_empty_masks:
            self.image_files, self.mask_files = self._filter_empty_masks(
                self.image_files, self.mask_files
            )

    def _get_paired_files(
        self, image_dir: Path, mask_dir: Path, extensions: Tuple[str, ...]
    ) -> Tuple[List[Path], List[Path]]:
        pairs = []
        for ext in extensions:
            # Find all images
            for img_path in image_dir.glob(f"*{ext}"):
                # Assumes mask has SAME filename as image
                mask_path = mask_dir / img_path.name
                if not mask_path.exists():
                    # Try uppercase extension just in case
                    mask_path_upper = (
                        mask_dir / f"{img_path.stem}{img_path.suffix.upper()}"
                    )
                    if mask_path_upper.exists():
                        mask_path = mask_path_upper
                    else:
                        print(
                            f"Warning: Mask not found for {img_path.name} in {mask_dir}"
                        )
                        continue

                pairs.append((img_path, mask_path))

            # Also check uppercase extension for images
            for img_path in image_dir.glob(f"*{ext.upper()}"):
                # Avoid duplicates if case-insensitive filesystem or extensions overlap
                if any(str(p[0]) == str(img_path) for p in pairs):
                    continue

                mask_path = mask_dir / img_path.name
                if not mask_path.exists():
                    mask_path_lower = (
                        mask_dir / f"{img_path.stem}{img_path.suffix.lower()}"
                    )
                    if mask_path_lower.exists():
                        mask_path = mask_path_lower
                    else:
                        print(
                            f"Warning: Mask not found for {img_path.name} in {mask_dir}"
                        )
                        continue
                pairs.append((img_path, mask_path))

        # Sort pairs by image path
        pairs.sort(key=lambda x: str(x[0]))

        if not pairs:
            return [], []

        images, masks = zip(*pairs)
        return list(images), list(masks)

    def _split_train_val(self, image_files, mask_files, split_type):
        """Split train partition into train/val"""
        random.seed(self.seed)

        # 80% train, 20% val
        train_imgs, val_imgs, train_masks, val_masks = train_test_split(
            image_files, mask_files, test_size=0.2, random_state=self.seed
        )

        if split_type == "train":
            return train_imgs, train_masks
        else:  # val
            return val_imgs, val_masks

    def _filter_empty_masks(self, image_files, mask_files):
        """Filter out pairs where the mask is completely empty (background only)"""
        print(f"Filtering empty masks... (Total before: {len(image_files)})")
        filtered_images = []
        filtered_masks = []

        # This might be slow if dataset is huge, but for ~600 images it's fine.
        # Use tqdm if available or just print progress
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
                # We need to check if the mask has any non-zero values
                # Reading as grayscale/RGB is fine as long as we detect non-black
                mask = Image.open(mask_path).convert("L")
                if np.array(mask).max() > 0:
                    filtered_images.append(img_path)
                    filtered_masks.append(mask_path)
            except Exception as e:
                print(f"Warning: Error reading mask {mask_path} during filtering: {e}")

        print(
            f"Filtering complete. Kept {len(filtered_images)} pairs (Removed {len(image_files) - len(filtered_images)})"
        )
        return filtered_images, filtered_masks

    def _convert_rgb_mask_to_classes(self, mask_rgb: np.ndarray) -> np.ndarray:
        """
        Convert RGB mask to class labels
        Red (255, 0, 0) -> 2 (Malignant)
        Green (0, 255, 0) -> 1 (Benign)
        Black (0, 0, 0) -> 0 (Background/Normal)
        """
        mask = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)

        # Check for malignant (red)
        red_mask = (
            (mask_rgb[:, :, 0] == 255)
            & (mask_rgb[:, :, 1] == 0)
            & (mask_rgb[:, :, 2] == 0)
        )
        mask[red_mask] = 2

        # Check for benign (green)
        green_mask = (
            (mask_rgb[:, :, 0] == 0)
            & (mask_rgb[:, :, 1] == 255)
            & (mask_rgb[:, :, 2] == 0)
        )
        mask[green_mask] = 1

        # Background (black) is already 0

        return mask

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        image = self._load_image(image_path)

        # Load RGB mask and convert to class labels (Benign=1, Malignant=2)
        mask_rgb = self._load_mask(mask_path, mode="RGB")
        mask_rgb_array = np.array(mask_rgb)
        mask_array = self._convert_rgb_mask_to_classes(mask_rgb_array)

        if not self.multiclass:
            # Binary mode: collapse all foreground to 1
            mask_array = (mask_array > 0).astype(np.uint8)

        mask = Image.fromarray(mask_array)

        image, mask = self._resize_images(image, mask)

        if self.transform:
            image, mask = self._joint_transform(image, mask)

        return self._create_tensors(image, mask, Path(image_path).stem)
    
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

        if not self.multiclass:
            mask_array = (mask_array > 0).astype(np.uint8)

        mask = Image.fromarray(mask_array)

        image, mask = self._resize_images(image, mask)

        if self.transform:
            image, mask = self._joint_transform(image, mask)

        return self._create_tensors(image, mask, Path(image_path).stem)



class BUSI(BaseUltrasoundDataset):
    """
    BUSI Dataset (Breast Ultrasound Images)
    - Total: 780 images (437 Benign, 210 Malignant, 133 Normal)
    - Binary segmentation: background (0), lesion (1)
    - Multiclass: benign=1, malignant=2, normal=0 (treated as background)
    - RGB images with varying sizes
    - Class directories: benign, malignant, normal
    - Some images have multiple masks (_mask.png, _mask_1.png) that need to be combined
    - Random split by class: 70% train, 15% val, 15% test
    """

    SUPPORTS_MULTICLASS = True
    CLASS_LABEL_MAP = {"benign": 1, "malignant": 2, "normal": 0}

    def __init__(self, cfg, split, transform: Optional[bool] = False):
        super().__init__(cfg, split, transform)

        self.root = Path(cfg.path.root)
        self.classes = cfg.classes
        self.image_suffix = getattr(cfg, "image_suffix", "")
        self.mask_suffix = getattr(cfg, "mask_suffix", "_mask")
        self.extensions = getattr(cfg, "extensions", [".png"])
        self.combine_multiple_masks = getattr(cfg, "combine_multiple_masks", True)

        # BUSI dataset has class-based subdirectory structure
        all_image_files, all_mask_files, all_class_labels = self._get_busi_files()

        # Split data by class (random split)
        self.image_files, self.mask_files, self.class_labels = self._split_data_by_class(
            all_image_files, all_mask_files, all_class_labels, self.split
        )

    def _split_data_by_class(self, image_files, mask_files, class_labels, split_type):
        """클래스별로 데이터를 분할 (70% train, 15% val, 15% test)"""
        random.seed(self.seed)

        # 클래스별로 파일들을 그룹화
        class_groups = {}
        for img_path, mask_path, lbl in zip(image_files, mask_files, class_labels):
            class_name = img_path.parent.name
            if class_name not in class_groups:
                class_groups[class_name] = {"images": [], "masks": [], "labels": []}
            class_groups[class_name]["images"].append(img_path)
            class_groups[class_name]["masks"].append(mask_path)
            class_groups[class_name]["labels"].append(lbl)

        split_images, split_masks, split_labels = [], [], []
        target_split = "val" if split_type in ["val", "valid"] else split_type

        for class_name, files in class_groups.items():
            images = files["images"]
            masks = files["masks"]
            labels = files["labels"]

            indices = list(range(len(images)))
            train_idx, temp_idx = train_test_split(
                indices, test_size=0.3, random_state=self.seed
            )
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.5, random_state=self.seed
            )

            if target_split == "train":
                sel = train_idx
            elif target_split == "val":
                sel = val_idx
            else:  # test
                sel = test_idx

            for i in sel:
                split_images.append(images[i])
                split_masks.append(masks[i])
                split_labels.append(labels[i])

        # Sort by image path for determinism
        sorted_triples = sorted(
            zip(split_images, split_masks, split_labels), key=lambda x: str(x[0])
        )
        if not sorted_triples:
            return [], [], []
        split_images, split_masks, split_labels = zip(*sorted_triples)
        return list(split_images), list(split_masks), list(split_labels)

    def _get_busi_files(self) -> Tuple[List[Path], List[List[Path]], List[int]]:
        """
        BUSI 데이터셋의 클래스별 디렉토리에서 이미지와 마스크 파일들을 찾아 반환
        일부 이미지는 여러 마스크를 가지므로 mask_files는 리스트의 리스트
        Strict pairing: image -> multiple masks
        """
        pairs = []

        for class_name in self.classes.values():
            class_dir = self.root / class_name

            if not class_dir.exists():
                print(f"Warning: Class directory does not exist: {class_dir}")
                continue

            class_label = self.CLASS_LABEL_MAP.get(class_name.lower(), 1)

            for ext in self.extensions:
                pattern = f"*{self.image_suffix}{ext}"
                for file_path in class_dir.glob(pattern):
                    if self.mask_suffix not in file_path.stem:
                        base_name = file_path.stem
                        masks_for_image = []

                        mask_name = f"{base_name}{self.mask_suffix}{ext}"
                        mask_path = class_dir / mask_name
                        if mask_path.exists():
                            masks_for_image.append(mask_path)

                        if self.combine_multiple_masks:
                            for i in range(1, 10):
                                mask_name_i = f"{base_name}{self.mask_suffix}_{i}{ext}"
                                mask_path_i = class_dir / mask_name_i
                                if mask_path_i.exists():
                                    masks_for_image.append(mask_path_i)

                        if masks_for_image:
                            pairs.append((file_path, masks_for_image, class_label))
                        else:
                            print(f"Warning: No mask file found for {file_path}")

        pairs.sort(key=lambda x: str(x[0]))

        if not pairs:
            return [], [], []

        images, masks, labels = zip(*pairs)
        return list(images), list(masks), list(labels)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        mask_paths = self.mask_files[index]  # List of mask paths
        class_label = self.class_labels[index]

        image = self._load_image(image_path)

        # Load and combine multiple masks via OR
        combined_mask = None
        for mask_path in mask_paths:
            mask = self._load_mask(mask_path, mode="L")
            mask_array = np.array(mask, dtype=np.uint8)
            if combined_mask is None:
                combined_mask = mask_array
            else:
                combined_mask = combined_mask | mask_array

        if combined_mask is None:
            raise ValueError(f"No masks found for image {image_path}")

        mask = Image.fromarray(combined_mask)

        if self.multiclass:
            mask = self._apply_class_label(mask, class_label)

        image, mask = self._resize_images(image, mask)

        if self.transform:
            image, mask = self._joint_transform(image, mask)

        return self._create_tensors(image, mask, Path(image_path).stem)


class BUSBRA(BaseUltrasoundDataset):
    """
    BUSBRA Dataset (Breast Ultrasound Brazil)
    - Total: 1875 images (left/right breast images)
    - Binary segmentation: background (0), lesion (1)
    - Multiclass: benign=1, malignant=2 (from bus_data.csv Pathology column)
    - Grayscale images with varying sizes
    - Naming convention:
        Images: bus_XXXX-{l|r}.png (l=left, r=right)
        Masks: mask_XXXX-{l|r}.png
    - Split: Pre-defined split using busbra_{split}.txt files
    """

    SUPPORTS_MULTICLASS = True
    PATHOLOGY_LABEL_MAP = {"benign": 1, "malignant": 2}

    def __init__(self, cfg, split, transform: Optional[bool] = False):
        super().__init__(cfg, split, transform)

        self.root = Path(cfg.path.root)
        self.image_prefix = getattr(cfg, "image_prefix", "bus_")
        self.mask_prefix = getattr(cfg, "mask_prefix", "mask_")
        self.extensions = getattr(cfg, "extensions", [".png"])

        image_dir_name = getattr(cfg, "image_dir", "Images")
        mask_dir_name = getattr(cfg, "mask_dir", "Masks")

        self.image_dir = self.root / image_dir_name
        self.mask_dir = self.root / mask_dir_name

        # Always load CSV for class-stratified splitting; also used for multiclass labels
        csv_label_map = {}  # image_id -> class_label
        csv_patient_class = {}  # patient_id -> class_label (for stratified split)
        csv_path = self.root / "bus_data.csv"
        if csv_path.exists():
            import csv
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_id = row["ID"].strip()
                    pathology = row["Pathology"].strip().lower()
                    label = self.PATHOLOGY_LABEL_MAP.get(pathology, 1)
                    csv_label_map[img_id] = label
                    # patient_id = numeric part of "bus_XXXX-l/r"
                    parts = img_id.replace(self.image_prefix, "").split("-")
                    if parts:
                        csv_patient_class[parts[0]] = label
        else:
            print(f"Warning: BUSBRA CSV not found at {csv_path}, defaulting to label=1")

        # Find all images and pair with masks
        pairs = []
        for ext in self.extensions:
            image_candidates = sorted(self.image_dir.glob(f"*{ext}"))

            for img_path in image_candidates:
                filename = img_path.stem
                if filename.startswith(self.image_prefix):
                    base_content = filename[len(self.image_prefix):]
                    mask_filename = f"{self.mask_prefix}{base_content}{img_path.suffix}"
                    mask_path = self.mask_dir / mask_filename

                    if mask_path.exists():
                        # ID in CSV is the full filename stem (e.g. "bus_1234-l")
                        class_label = csv_label_map.get(filename, 1)
                        pairs.append((img_path, mask_path, class_label))
                    else:
                        print(
                            f"Warning: Mask not found for {img_path}: expected {mask_path}"
                        )
                else:
                    print(
                        f"Warning: Image {filename} does not start with prefix {self.image_prefix}"
                    )

        pairs.sort(key=lambda x: str(x[0]))

        if pairs:
            all_image_files, all_mask_files, all_class_labels = zip(*pairs)
            all_image_files = list(all_image_files)
            all_mask_files = list(all_mask_files)
            all_class_labels = list(all_class_labels)
        else:
            all_image_files, all_mask_files, all_class_labels = [], [], []

        # Split by patient ID, stratified by class (70/15/15)
        self.image_list, self.mask_list, self.class_labels = self._split_by_patient(
            all_image_files, all_mask_files, all_class_labels, self.split, csv_patient_class
        )

    def _split_by_patient(self, image_files, mask_files, class_labels, split_type, patient_class_map=None):
        random.seed(self.seed)

        # Build patient_id -> image indices map
        patient_map = {}
        for idx, img_path in enumerate(image_files):
            filename = img_path.stem
            if filename.startswith(self.image_prefix):
                core_name = filename[len(self.image_prefix):]
                parts = core_name.split("-")
                if parts:
                    patient_id = parts[0]
                    if patient_id not in patient_map:
                        patient_map[patient_id] = []
                    patient_map[patient_id].append(idx)

        # Group patient IDs by class for stratified split
        if patient_class_map:
            class_to_pids = {}
            for pid in sorted(patient_map.keys()):
                cls = patient_class_map.get(pid, 1)
                class_to_pids.setdefault(cls, []).append(pid)
        else:
            class_to_pids = {1: sorted(patient_map.keys())}

        # 70/15/15 split per class, then merge
        train_pids, val_pids, test_pids = [], [], []
        for pids in class_to_pids.values():
            tr, temp = train_test_split(pids, test_size=0.3, random_state=self.seed)
            va, te = train_test_split(temp, test_size=0.5, random_state=self.seed)
            train_pids.extend(tr)
            val_pids.extend(va)
            test_pids.extend(te)

        target_split = "val" if split_type in ["val", "valid"] else split_type

        selected_indices = []
        pid_set = {"train": train_pids, "val": val_pids, "test": test_pids}[target_split]
        for pid in pid_set:
            selected_indices.extend(patient_map[pid])

        sorted_indices = sorted(selected_indices)
        final_images = [image_files[i] for i in sorted_indices]
        final_masks = [mask_files[i] for i in sorted_indices]
        final_labels = [class_labels[i] for i in sorted_indices]

        return final_images, final_masks, final_labels

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        mask_path = self.mask_list[index]
        class_label = self.class_labels[index]

        image = self._load_image(img_path)
        mask = self._load_mask(mask_path, mode="L")

        if self.multiclass:
            mask = self._apply_class_label(mask, class_label)

        image, mask = self._resize_images(image, mask)

        if self.transform:
            image, mask = self._joint_transform(image, mask)

        return self._create_tensors(image, mask, Path(img_path).stem)


class BUSBRA_SegFormer(BUSBRA):
    def __init__(self, cfg, split, transform: Optional[bool] = False):
        super().__init__(cfg, split, transform)
        self.image_processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512"
        )

    def __getitem__(self, index):
        img_path = self.image_list[index]
        mask_path = self.mask_list[index]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        inputs = self.image_processor(
            images=image, segmentation_maps=mask, return_tensors="np"
        )
        for k in inputs:
            inputs[k] = inputs[k].squeeze(0)

        return inputs["pixel_values"], inputs["labels"], Path(img_path).stem


class B(BaseUltrasoundDataset):
    """
    Dataset B
    - Total: 163 images
    - Binary segmentation: background (0), lesion (1)
    - Multiclass: benign=1, malignant=2 (from DatasetB.csv Diagnosis column)
    - File structure:
        original/ (Images, filename = 6-digit zero-padded Image number)
        GT/ (Masks)
        DatasetB.csv: Image,Type,Diagnosis
    - Class-stratified split: 70% train, 15% val, 15% test
    """

    SUPPORTS_MULTICLASS = True
    DIAGNOSIS_LABEL_MAP = {"benign": 1, "malignant": 2}

    def __init__(self, cfg, split, transform: Optional[bool] = False):
        super().__init__(cfg, split, transform)

        self.root = Path(cfg.path.root)
        self.extensions = getattr(cfg, "extensions", [".png"])

        self.image_dir_name = getattr(cfg, "image_dir", "original")
        self.mask_dir_name = getattr(cfg, "mask_dir", "GT")

        self.image_dir = self.root / self.image_dir_name
        self.mask_dir = self.root / self.mask_dir_name

        if not self.image_dir.exists():
            raise ValueError(f"Image directory does not exist: {self.image_dir}")
        if not self.mask_dir.exists():
            raise ValueError(f"Mask directory does not exist: {self.mask_dir}")

        # Always load CSV for class-stratified splitting; also used for multiclass labels
        self._csv_label_map = {}
        csv_path = self.root / "DatasetB.csv"
        if csv_path.exists():
            import csv
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_num = row["Image"].strip()
                    type_str = row["Type"].strip().lower()
                    filename_stem = str(int(img_num)).zfill(6)
                    self._csv_label_map[filename_stem] = self.DIAGNOSIS_LABEL_MAP.get(
                        type_str, 1
                    )
        else:
            print(f"Warning: Dataset B CSV not found at {csv_path}, defaulting to label=1")

        images, masks, class_labels = self._collect_paired_files()
        self.images, self.masks, self.class_labels = self._split_dataset(
            images, masks, class_labels
        )

    def _collect_paired_files(self):
        pairs = []
        for ext in self.extensions:
            for img_path in self.image_dir.glob(f"*{ext}"):
                mask_path = self.mask_dir / img_path.name

                if mask_path.exists():
                    class_label = self._csv_label_map.get(img_path.stem, 1)
                    pairs.append((img_path, mask_path, class_label))
                else:
                    print(f"Warning: Mask not found for {img_path.name}")

        pairs.sort(key=lambda x: str(x[0]))

        if not pairs:
            return [], [], []

        images, masks, labels = zip(*pairs)
        return list(images), list(masks), list(labels)

    def _split_dataset(self, images, masks, class_labels):
        if not images:
            return [], [], []

        # Group indices by class label for stratified 70/15/15 split
        class_to_indices = {}
        for i, lbl in enumerate(class_labels):
            class_to_indices.setdefault(lbl, []).append(i)

        train_idx, val_idx, test_idx = [], [], []
        for indices in class_to_indices.values():
            tr, temp = train_test_split(indices, test_size=0.3, random_state=self.seed)
            va, te = train_test_split(temp, test_size=0.5, random_state=self.seed)
            train_idx.extend(tr)
            val_idx.extend(va)
            test_idx.extend(te)

        target_split = "val" if self.split in ["val", "valid"] else self.split
        sel = {"train": train_idx, "val": val_idx, "test": test_idx}[target_split]

        return (
            [images[i] for i in sel],
            [masks[i] for i in sel],
            [class_labels[i] for i in sel],
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        class_label = self.class_labels[idx]

        image = self._load_image(image_path)
        mask = self._load_mask(mask_path, mode="L")

        if self.multiclass:
            mask = self._apply_class_label(mask, class_label)

        image, mask = self._resize_images(image, mask)

        if self.transform:
            image, mask = self._joint_transform(image, mask)

        return self._create_tensors(image, mask, Path(image_path).stem)


class DDTI(BaseUltrasoundDataset):
    """DDTI Dataset (DDTI: Deep Learning for Tumor Identification)"""

    def __init__(self, cfg, split, transform: Optional[bool] = False):
        super().__init__(cfg, split, transform)

        self.root = Path(cfg.path.root)
        self.extensions = getattr(cfg, "extensions", [".png"])

        # Directory structure from yaml or default
        self.image_dir_name = getattr(cfg, "image_dir", "original")
        self.mask_dir_name = getattr(cfg, "mask_dir", "GT")

        self.image_dir = self.root / self.image_dir_name
        self.mask_dir = self.root / self.mask_dir_name

        if not self.image_dir.exists():
            raise ValueError(f"Image directory does not exist: {self.image_dir}")
        if not self.mask_dir.exists():
            raise ValueError(f"Mask directory does not exist: {self.mask_dir}")

        # Collect paired files
        images, masks = self._collect_paired_files()

        # Split dataset
        self.images, self.masks = self._split_dataset(images, masks)
