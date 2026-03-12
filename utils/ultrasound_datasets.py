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
        # Binary case: assume > 0 or > 127 is foreground depending on value range
        if mask_np.max() > 1:
            mask_np = (mask_np > 127).astype(np.float32)
        else:
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
    - RGB images with sizes 590x590 or 600x600
    - File structure:
        Benign/
            XXX Benign Image.bmp
            XXX Benign Mask.tif (binary mask)
            XXX Benign Lesion.bmp (not used)
        Malignant/
            XXX Malignant Image.bmp
            XXX Malignant Mask.tif (binary mask)
            XXX Malignant Lesion.bmp (not used)
    - Random split by class: 70% train, 15% val, 15% test
    """

    def __init__(self, cfg, split, transform: Optional[bool] = False):
        super().__init__(cfg, split, transform)

        self.root = Path(cfg.path.root)
        self.classes = cfg.classes
        self.usage = getattr(cfg, "usage", "external")

        if self.usage == "external":
            # Get all image and mask files from both classes
            self.images, self.masks = self._unzip_pairs(self._collect_paired_files())
        else:
            # split into train/val/test: 70/15/15
            self.images, self.masks = self._unzip_pairs(self._collect_paired_files())
            self.images, self.masks = self._split_dataset(self.images, self.masks)

    def _split_dataset(self, images, masks):
        # Group by class
        benign_pairs = []
        malignant_pairs = []

        for img, msk in zip(images, masks):
            if "Benign" in str(img):
                benign_pairs.append((img, msk))
            elif "Malignant" in str(img):
                malignant_pairs.append((img, msk))
            else:
                # Fallback if naming convention is different
                benign_pairs.append((img, msk))

        # Split each class separately to ensure balance
        # 70% train, 15% val, 15% test
        def split_class(pairs, seed):
            if not pairs:
                return [], [], []

            # First split: 70% train, 30% temp
            train, temp = train_test_split(pairs, test_size=0.3, random_state=seed)
            # Second split: 50% val, 50% test (from 30% temp -> 15% each)
            val, test = train_test_split(temp, test_size=0.5, random_state=seed)
            return train, val, test

        b_train, b_val, b_test = split_class(benign_pairs, self.seed)
        m_train, m_val, m_test = split_class(malignant_pairs, self.seed)

        # Combine
        train_pairs = b_train + m_train
        val_pairs = b_val + m_val
        test_pairs = b_test + m_test

        # Support both 'val' and 'valid'
        target_split = "val" if self.split in ["val", "valid"] else self.split

        if target_split == "train":
            selected = train_pairs
        elif target_split == "val":
            selected = val_pairs
        elif target_split == "test":
            selected = test_pairs
        else:
            raise ValueError(f"Invalid split: {self.split}")

        # Sort for consistency
        selected.sort(key=lambda x: str(x[0]))

        return self._unzip_pairs(selected)

    def _collect_paired_files(self) -> List[Tuple[Path, Path]]:
        pairs = []
        for class_dir_name in self.classes.values():
            class_dir = self.root / class_dir_name
            if not class_dir.exists():
                print(f"Warning: Class directory does not exist: {class_dir}")
                continue

            image_pattern = f"* {class_dir_name} Image.bmp"
            for img_path in class_dir.glob(image_pattern):
                base_name = img_path.stem.replace(" Image", " Mask")
                mask_path = class_dir / f"{base_name}.tif"

                if mask_path.exists():
                    pairs.append((img_path, mask_path))
                else:
                    print(f"Warning: Mask file not found: {mask_path}")

        # Sort pairs by image path to ensure consistent ordering
        pairs.sort(key=lambda x: str(x[0]))
        return pairs

    def _unzip_pairs(self, pairs):
        if not pairs:
            return [], []
        images, masks = zip(*pairs)
        return list(images), list(masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]

        # Load image and mask using base class methods
        image = self._load_image(image_path)
        mask = self._load_mask(mask_path, mode="L")

        # Resize to target size
        image, mask = self._resize_images(image, mask)

        # Apply transformations if enabled
        if self.transform:
            image, mask = self._joint_transform(image, mask)

        # Create tensors using base class method
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
    - Split: Pre-defined train/test split in partitions folder
    """

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

                if target_split == "train":
                    self.image_files, self.mask_files = self._split_train_val(
                        all_image_files, all_mask_files, self.split
                    )
                else:  # val
                    self.image_files, self.mask_files = self._split_train_val(
                        all_image_files, all_mask_files, self.split
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

        # Load image using base class method
        image = self._load_image(image_path)

        # Load RGB mask and convert to class labels
        mask_rgb = self._load_mask(mask_path, mode="RGB")
        mask_rgb_array = np.array(mask_rgb)
        mask_array = self._convert_rgb_mask_to_classes(mask_rgb_array)
        mask = Image.fromarray(mask_array)

        # Resize to target size
        image, mask = self._resize_images(image, mask)

        # Apply transformations if enabled
        if self.transform:
            image, mask = self._joint_transform(image, mask)

        # Create tensors using base class method
        return self._create_tensors(image, mask, Path(image_path).stem)


class BUSI(BaseUltrasoundDataset):
    """
    BUSI Dataset (Breast Ultrasound Images)
    - Total: 780 images (437 Benign, 210 Malignant, 133 Normal)
    - Binary segmentation: background (0), lesion (1)
    - RGB images with varying sizes
    - Class directories: benign, malignant, normal
    - Some images have multiple masks (_mask.png, _mask_1.png) that need to be combined
    - Random split by class: 70% train, 15% val, 15% test
    """

    def __init__(self, cfg, split, transform: Optional[bool] = False):
        super().__init__(cfg, split, transform)

        self.root = Path(cfg.path.root)
        self.classes = cfg.classes
        self.image_suffix = getattr(cfg, "image_suffix", "")
        self.mask_suffix = getattr(cfg, "mask_suffix", "_mask")
        self.extensions = getattr(cfg, "extensions", [".png"])
        self.combine_multiple_masks = getattr(cfg, "combine_multiple_masks", True)

        # BUSI dataset has class-based subdirectory structure
        all_image_files, all_mask_files = self._get_busi_files()

        # Split data by class (random split)
        self.image_files, self.mask_files = self._split_data_by_class(
            all_image_files, all_mask_files, self.split
        )

    def _split_data_by_class(self, image_files, mask_files, split_type):
        """클래스별로 데이터를 분할"""
        random.seed(self.seed)

        # 클래스별로 파일들을 그룹화
        class_groups = {}
        for img_path, mask_path in zip(image_files, mask_files):
            class_name = img_path.parent.name
            if class_name not in class_groups:
                class_groups[class_name] = {"images": [], "masks": []}
            class_groups[class_name]["images"].append(img_path)
            class_groups[class_name]["masks"].append(mask_path)

        split_images, split_masks = [], []
        # Support both 'val' and 'valid'
        target_split = "val" if split_type in ["val", "valid"] else split_type

        for class_name, files in class_groups.items():
            images = files["images"]
            masks = files["masks"]

            # 80% train, 20% val (Internal Validation)
            train_imgs, val_imgs, train_masks, val_masks = train_test_split(
                images, masks, test_size=0.2, random_state=self.seed
            )

            if target_split == "train":
                split_images.extend(train_imgs)
                split_masks.extend(train_masks)
            elif target_split == "val":
                split_images.extend(val_imgs)
                split_masks.extend(val_masks)
            elif target_split == "test":
                # BUSI doesn't have a separate test set in this 80/20 split config
                # Returning val set as test set, or could raise error
                split_images.extend(val_imgs)
                split_masks.extend(val_masks)

        return sorted(split_images), sorted(split_masks)

    def _get_busi_files(self) -> Tuple[List[Path], List[List[Path]]]:
        """
        BUSI 데이터셋의 클래스별 디렉토리에서 이미지와 마스크 파일들을 찾아 반환
        일부 이미지는 여러 마스크를 가지므로 mask_files는 리스트의 리스트
        Strict pairing: image -> multiple masks
        """
        pairs = []

        # 각 클래스 디렉토리를 순회
        for class_name in self.classes.values():
            class_dir = self.root / class_name

            if not class_dir.exists():
                print(f"Warning: Class directory does not exist: {class_dir}")
                continue

            # 해당 클래스 디렉토리에서 파일들 찾기
            for ext in self.extensions:
                # 이미지 파일들 찾기 (마스크가 아닌 것들)
                pattern = f"*{self.image_suffix}{ext}"
                for file_path in class_dir.glob(pattern):
                    # 마스크 파일이 아닌 경우만 이미지로 간주
                    if self.mask_suffix not in file_path.stem:

                        # 대응되는 모든 마스크 파일 찾기
                        # _mask.png, _mask_1.png, _mask_2.png 등
                        base_name = file_path.stem
                        masks_for_image = []

                        # 기본 마스크
                        mask_name = f"{base_name}{self.mask_suffix}{ext}"
                        mask_path = class_dir / mask_name
                        if mask_path.exists():
                            masks_for_image.append(mask_path)

                        # 추가 마스크 (_mask_1, _mask_2, ...)
                        if self.combine_multiple_masks:
                            # 1부터 9까지 체크
                            for i in range(1, 10):
                                mask_name_i = f"{base_name}{self.mask_suffix}_{i}{ext}"
                                mask_path_i = class_dir / mask_name_i
                                if mask_path_i.exists():
                                    masks_for_image.append(mask_path_i)
                                # Skip finding if not strictly consecutive? Implementation was breaking before.
                                # To be safe, we check them all or stop at first missing?
                                # Original logic stopped at break. We keep "break".
                                else:
                                    break

                        if masks_for_image:
                            pairs.append((file_path, masks_for_image))
                        else:
                            print(f"Warning: No mask file found for {file_path}")

        # Sort pairs by image path
        pairs.sort(key=lambda x: str(x[0]))

        if not pairs:
            return [], []

        images, masks = zip(*pairs)
        return list(images), list(masks)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        mask_paths = self.mask_files[index]  # List of mask paths

        # Load image using base class method
        image = self._load_image(image_path)

        # Load and combine multiple masks
        combined_mask = None
        for mask_path in mask_paths:
            mask = self._load_mask(mask_path, mode="L")
            mask_array = np.array(mask, dtype=np.uint8)

            if combined_mask is None:
                combined_mask = mask_array
            else:
                # Combine masks using OR operation
                combined_mask = combined_mask | mask_array

        if combined_mask is None:
            raise ValueError(f"No masks found for image {image_path}")

        # Convert to PIL Image
        mask = Image.fromarray(combined_mask)

        # Resize to target size
        image, mask = self._resize_images(image, mask)

        # Apply transformations if enabled
        if self.transform:
            image, mask = self._joint_transform(image, mask)

        # Create tensors using base class method
        return self._create_tensors(image, mask, Path(image_path).stem)


class BUSBRA(BaseUltrasoundDataset):
    """
    BUSBRA Dataset (Breast Ultrasound Brazil)
    - Total: 1875 images (left/right breast images)
    - Binary segmentation: background (0), lesion (1)
    - Grayscale images with varying sizes
    - Naming convention:
        Images: bus_XXXX-{l|r}.png (l=left, r=right)
        Masks: mask_XXXX-{l|r}.png
    - Split: Pre-defined split using busbra_{split}.txt files
    """

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

        # 1. Scan directory for all images
        # 1. Scan directory for all images and pair with masks
        # Find all images first, then find corresponding mask
        pairs = []
        for ext in self.extensions:
            # Find all images in sorted order first
            image_candidates = sorted(self.image_dir.glob(f"*{ext}"))

            for img_path in image_candidates:
                filename = img_path.stem
                if filename.startswith(self.image_prefix):
                    base_content = filename[len(self.image_prefix) :]
                    mask_filename = f"{self.mask_prefix}{base_content}{img_path.suffix}"
                    mask_path = self.mask_dir / mask_filename

                    if mask_path.exists():
                        pairs.append((img_path, mask_path))
                    else:
                        print(
                            f"Warning: Mask not found for {img_path}: expected {mask_path}"
                        )
                else:
                    print(
                        f"Warning: Image {filename} does not start with prefix {self.image_prefix}"
                    )

        # Sort pairs by image path
        pairs.sort(key=lambda x: str(x[0]))

        if pairs:
            all_image_files, all_mask_files = zip(*pairs)
            all_image_files, all_mask_files = list(all_image_files), list(
                all_mask_files
            )
        else:
            all_image_files, all_mask_files = [], []

        # 2. Split by Patient ID (80/20)
        # Naming convention: bus_XXXX-{l|r} -> Patient ID is XXXX
        self.image_list, self.mask_list = self._split_by_patient(
            all_image_files, all_mask_files, self.split
        )

    def _split_by_patient(self, image_files, mask_files, split_type):
        random.seed(self.seed)

        # Extract patient IDs
        # bus_1234-l.png -> 1234
        patient_map = {}  # patient_id -> list of indices

        for idx, img_path in enumerate(image_files):
            filename = img_path.stem  # bus_1234-l
            # Remove prefix
            if filename.startswith(self.image_prefix):
                core_name = filename[len(self.image_prefix) :]  # 1234-l
                # Split by '-' to get ID
                parts = core_name.split("-")
                if len(parts) >= 1:
                    patient_id = parts[0]  # 1234

                    if patient_id not in patient_map:
                        patient_map[patient_id] = []
                    patient_map[patient_id].append(idx)

        patient_ids = sorted(list(patient_map.keys()))

        # Split patients 80/20
        train_pids, val_pids = train_test_split(
            patient_ids, test_size=0.2, random_state=self.seed
        )

        # Support both 'val' and 'valid'
        target_split = "val" if split_type in ["val", "valid"] else split_type

        selected_indices = []
        if target_split == "train":
            for pid in train_pids:
                selected_indices.extend(patient_map[pid])
        elif target_split in ["val", "test"]:
            # Use val set for test as well since we only have 80/20 split
            for pid in val_pids:
                selected_indices.extend(patient_map[pid])

        final_images = [str(image_files[i]) for i in sorted(selected_indices)]
        final_masks = [str(mask_files[i]) for i in sorted(selected_indices)]

        return final_images, final_masks

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        mask_path = self.mask_list[index]

        # Load image and mask using base class methods
        image = self._load_image(img_path)
        mask = self._load_mask(mask_path, mode="L")

        # Resize to target size
        image, mask = self._resize_images(image, mask)

        # Apply transformations if enabled
        if self.transform:
            image, mask = self._joint_transform(image, mask)

        # Create tensors using base class method
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


class UltrasoundSegmentationDataset(BaseUltrasoundDataset):
    """Simple ultrasound segmentation dataset without low-res tensor output."""

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        num_classes: int,
        transform: Optional[bool] = False,
        image_size: Tuple[int, int] = (512, 512),
        task_type: str = "tumor",
    ):
        # Create a minimal config object for base class
        from types import SimpleNamespace

        cfg = SimpleNamespace(
            num_classes=num_classes,
            img_size=image_size[0],
            seed=42,
            normalization="imagenet",
            task_type=task_type,
        )
        super().__init__(cfg, split="train", transform=transform)

        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        self.image_files = sorted(
            [
                f.name
                for f in self.image_dir.iterdir()
                if f.suffix.lower() in (".png", ".jpg", ".jpeg")
            ]
        )

        assert len(self.image_files) > 0, "No image files found"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_dir / self.image_files[idx]
        label_path = self.label_dir / self.image_files[idx]

        # Load image and mask using base class methods
        image = self._load_image(img_path)
        label = self._load_mask(label_path, mode="L")

        # Resize to target size
        image, label = self._resize_images(image, label)

        # Apply transformations if enabled
        if self.transform:
            image, label = self._joint_transform(image, label)

        # Tensorize and normalize image
        image_tensor = TF.to_tensor(image)
        image_tensor = self._apply_normalization(image_tensor)

        # Format mask (without low-res tensor)
        label_tensor = _format_mask(label, self.num_classes)

        return image_tensor, label_tensor, Path(img_path).stem


class B(BaseUltrasoundDataset):
    """
    Dataset B
    - Total: 163 images
    - Binary segmentation: background (0), lesion (1)
    - File structure:
        original/ (Images)
        GT/ (Masks)
    - Random split: 80% train, 20% val
    """

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

    def _collect_paired_files(self):
        pairs = []
        for ext in self.extensions:
            # Find all images
            for img_path in self.image_dir.glob(f"*{ext}"):
                # Assumes mask has SAME filename as image
                mask_path = self.mask_dir / img_path.name

                if mask_path.exists():
                    pairs.append((img_path, mask_path))
                else:
                    # Check for case sensitivity issues if needed, but assuming strict for now
                    print(f"Warning: Mask not found for {img_path.name}")

        # Sort for deterministic split
        pairs.sort(key=lambda x: str(x[0]))

        if not pairs:
            return [], []

        images, masks = zip(*pairs)
        return list(images), list(masks)

    def _split_dataset(self, images, masks):
        if not images:
            return [], []

        # 80% train, 20% val
        train_imgs, val_imgs, train_masks, val_masks = train_test_split(
            images, masks, test_size=0.2, random_state=self.seed
        )

        # Support both 'val' and 'valid'
        target_split = "val" if self.split in ["val", "valid"] else self.split

        if target_split == "train":
            return train_imgs, train_masks
        elif target_split == "val":
            return val_imgs, val_masks
        else:
            return val_imgs, val_masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]

        # Load image and mask using base class methods
        image = self._load_image(image_path)
        mask = self._load_mask(mask_path, mode="L")

        # Resize to target size
        image, mask = self._resize_images(image, mask)

        # Apply transformations if enabled
        if self.transform:
            image, mask = self._joint_transform(image, mask)

        # Create tensors using base class method
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
