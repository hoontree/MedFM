#!/usr/bin/env python3
"""
Convert TinyUSFM datasets to COCO format for SAM3 training
Supports: BUSBRA, BUSI, BUID, BUS_UCLM, B, and combined datasets
"""

import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm
import pycocotools.mask as mask_utils
from omegaconf import OmegaConf


class COCOConverter:
    """Convert segmentation datasets to COCO format"""

    def __init__(self, dataset_config_path: str, output_dir: str):
        """
        Args:
            dataset_config_path: Path to dataset YAML config
            output_dir: Output directory for COCO format data
        """
        self.cfg = OmegaConf.load(dataset_config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_name = self.cfg.name
        self.root = Path(self.cfg.path.root)

        print(f"Converting {self.dataset_name} to COCO format")
        print(f"Source: {self.root}")
        print(f"Output: {self.output_dir}")

    def convert_all_splits(self):
        """Convert train, val, and test splits"""
        for split in ['train', 'val', 'test']:
            print(f"\n{'='*60}")
            print(f"Converting {split} split...")
            print(f"{'='*60}")

            try:
                image_mask_pairs = self.get_image_mask_pairs(split)

                if not image_mask_pairs:
                    print(f"Warning: No data found for {split} split")
                    continue

                coco_data = self.create_coco_annotations(image_mask_pairs, split)

                output_file = self.output_dir / f"{split}_annotations.json"
                with open(output_file, 'w') as f:
                    json.dump(coco_data, f, indent=2)

                print(f"✓ Saved {split} annotations to {output_file}")
                print(f"  - {len(coco_data['images'])} images")
                print(f"  - {len(coco_data['annotations'])} annotations")

            except Exception as e:
                print(f"Error converting {split} split: {e}")
                import traceback
                traceback.print_exc()

    def get_image_mask_pairs(self, split: str) -> List[Tuple[Path, Path]]:
        """Get image-mask pairs based on dataset type"""
        dataset_name = self.dataset_name

        if dataset_name == "BUSBRA":
            return self._get_busbra_pairs(split)
        elif dataset_name == "BUSI":
            return self._get_busi_pairs(split)
        elif dataset_name == "BUID":
            return self._get_buid_pairs(split)
        elif dataset_name == "BUS_UCLM":
            return self._get_bus_uclm_pairs(split)
        elif dataset_name == "B":
            return self._get_b_pairs(split)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def _get_busbra_pairs(self, split: str) -> List[Tuple[Path, Path]]:
        """Get BUSBRA image-mask pairs"""
        image_dir = self.root / self.cfg.image_dir
        mask_dir = self.root / self.cfg.mask_dir

        image_prefix = self.cfg.get('image_prefix', 'bus_')
        mask_prefix = self.cfg.get('mask_prefix', 'mask_')
        extensions = self.cfg.get('extensions', ['.png'])

        pairs = []
        for ext in extensions:
            for img_path in sorted(image_dir.glob(f"{image_prefix}*{ext}")):
                # Extract base name: bus_1234-l.png -> 1234-l
                filename = img_path.stem
                if filename.startswith(image_prefix):
                    base_content = filename[len(image_prefix):]
                    mask_filename = f"{mask_prefix}{base_content}{ext}"
                    mask_path = mask_dir / mask_filename

                    if mask_path.exists():
                        pairs.append((img_path, mask_path))

        # Split by patient ID (same logic as BUSBRA dataset class)
        return self._split_busbra_by_patient(pairs, split)

    def _split_busbra_by_patient(self, pairs: List[Tuple[Path, Path]], split: str) -> List[Tuple[Path, Path]]:
        """Split BUSBRA data by patient ID"""
        from sklearn.model_selection import train_test_split
        import random

        seed = self.cfg.get('seed', 42)
        random.seed(seed)

        # Extract patient IDs
        patient_map = {}
        for idx, (img_path, mask_path) in enumerate(pairs):
            filename = img_path.stem
            image_prefix = self.cfg.get('image_prefix', 'bus_')

            if filename.startswith(image_prefix):
                core_name = filename[len(image_prefix):]
                parts = core_name.split('-')
                if len(parts) >= 1:
                    patient_id = parts[0]

                    if patient_id not in patient_map:
                        patient_map[patient_id] = []
                    patient_map[patient_id].append(idx)

        patient_ids = sorted(list(patient_map.keys()))

        # Split patients 80/20
        train_pids, val_pids = train_test_split(
            patient_ids, test_size=0.2, random_state=seed
        )

        selected_indices = []
        if split == 'train':
            for pid in train_pids:
                selected_indices.extend(patient_map[pid])
        else:  # val or test
            for pid in val_pids:
                selected_indices.extend(patient_map[pid])

        return [pairs[i] for i in sorted(selected_indices)]

    def _get_busi_pairs(self, split: str) -> List[Tuple[Path, Path]]:
        """Get BUSI image-mask pairs"""
        from sklearn.model_selection import train_test_split
        import random

        seed = self.cfg.get('seed', 42)
        random.seed(seed)

        classes = self.cfg.get('classes', {'benign': 'benign', 'malignant': 'malignant', 'normal': 'normal'})
        mask_suffix = self.cfg.get('mask_suffix', '_mask')
        extensions = self.cfg.get('extensions', ['.png'])

        all_pairs = []

        for class_name in classes.values():
            class_dir = self.root / class_name

            if not class_dir.exists():
                continue

            for ext in extensions:
                for file_path in class_dir.glob(f"*{ext}"):
                    if mask_suffix not in file_path.stem:
                        base_name = file_path.stem
                        masks_for_image = []

                        # Basic mask
                        mask_path = class_dir / f"{base_name}{mask_suffix}{ext}"
                        if mask_path.exists():
                            masks_for_image.append(mask_path)

                        # Additional masks (_mask_1, _mask_2, ...)
                        for i in range(1, 10):
                            mask_path_i = class_dir / f"{base_name}{mask_suffix}_{i}{ext}"
                            if mask_path_i.exists():
                                masks_for_image.append(mask_path_i)
                            else:
                                break

                        if masks_for_image:
                            all_pairs.append((file_path, masks_for_image))

        # Split 80/20 by class
        train_pairs, val_pairs = train_test_split(
            all_pairs, test_size=0.2, random_state=seed
        )

        if split == 'train':
            return train_pairs
        else:  # val or test
            return val_pairs

    def _get_buid_pairs(self, split: str) -> List[Tuple[Path, Path]]:
        """Get BUID image-mask pairs"""
        from sklearn.model_selection import train_test_split
        import random

        seed = self.cfg.get('seed', 42)
        random.seed(seed)

        classes = self.cfg.get('classes', {'benign': 'Benign', 'malignant': 'Malignant'})

        pairs = []
        for class_dir_name in classes.values():
            class_dir = self.root / class_dir_name

            if not class_dir.exists():
                continue

            image_pattern = f"* {class_dir_name} Image.bmp"
            for img_path in class_dir.glob(image_pattern):
                base_name = img_path.stem.replace(" Image", " Mask")
                mask_path = class_dir / f"{base_name}.tif"

                if mask_path.exists():
                    pairs.append((img_path, mask_path))

        # Split 70/15/15
        train, temp = train_test_split(pairs, test_size=0.3, random_state=seed)
        val, test = train_test_split(temp, test_size=0.5, random_state=seed)

        if split == 'train':
            return train
        elif split == 'val':
            return val
        else:
            return test

    def _get_bus_uclm_pairs(self, split: str) -> List[Tuple[Path, Path]]:
        """Get BUS-UCLM image-mask pairs"""
        from sklearn.model_selection import train_test_split
        import random

        seed = self.cfg.get('seed', 42)
        random.seed(seed)

        partition_dir = self.cfg.get('partition_dir', 'partitions')
        extensions = self.cfg.get('extensions', ('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))

        if split in ['train', 'val']:
            image_dir = self.root / partition_dir / 'train' / 'images'
            mask_dir = self.root / partition_dir / 'train' / 'masks'

            all_pairs = self._get_paired_files(image_dir, mask_dir, extensions)

            # Split 80/20
            train_pairs, val_pairs = train_test_split(
                all_pairs, test_size=0.2, random_state=seed
            )

            return train_pairs if split == 'train' else val_pairs
        else:  # test
            image_dir = self.root / partition_dir / 'test' / 'images'
            mask_dir = self.root / partition_dir / 'test' / 'masks'

            return self._get_paired_files(image_dir, mask_dir, extensions)

    def _get_b_pairs(self, split: str) -> List[Tuple[Path, Path]]:
        """Get Dataset B image-mask pairs"""
        from sklearn.model_selection import train_test_split
        import random

        seed = self.cfg.get('seed', 42)
        random.seed(seed)

        image_dir = self.root / self.cfg.get('image_dir', 'original')
        mask_dir = self.root / self.cfg.get('mask_dir', 'GT')
        extensions = self.cfg.get('extensions', ['.png'])

        all_pairs = self._get_paired_files(image_dir, mask_dir, extensions)

        # Split 80/20
        train_pairs, val_pairs = train_test_split(
            all_pairs, test_size=0.2, random_state=seed
        )

        if split == 'train':
            return train_pairs
        else:  # val or test
            return val_pairs

    def _get_paired_files(self, image_dir: Path, mask_dir: Path, extensions: Tuple[str, ...]) -> List[Tuple[Path, Path]]:
        """Get paired image and mask files"""
        pairs = []
        for ext in extensions:
            for img_path in image_dir.glob(f'*{ext}'):
                mask_path = mask_dir / img_path.name

                if mask_path.exists():
                    pairs.append((img_path, mask_path))

        pairs.sort(key=lambda x: str(x[0]))
        return pairs

    def create_coco_annotations(self, image_mask_pairs: List, split: str) -> Dict:
        """Create COCO format annotations from image-mask pairs"""
        coco_data = {
            "info": {
                "description": f"{self.dataset_name} Dataset - {split} split",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().isoformat()
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "lesion", "supercategory": "object"}
            ]
        }

        annotation_id = 1

        for image_id, pair in enumerate(tqdm(image_mask_pairs, desc=f"Processing {split}")):
            # Handle both single mask and multiple masks (BUSI case)
            if isinstance(pair[1], list):
                img_path, mask_paths = pair
                mask_list = mask_paths
            else:
                img_path, mask_path = pair
                mask_list = [mask_path]

            try:
                # Load image to get dimensions
                with Image.open(img_path) as img:
                    img_width, img_height = img.size

                # Add image info
                coco_data["images"].append({
                    "id": image_id,
                    "file_name": str(img_path.relative_to(self.root)),
                    "width": img_width,
                    "height": img_height
                })

                # Load and combine masks
                combined_mask = None
                for mask_path in mask_list:
                    mask = np.array(Image.open(mask_path).convert('L'))

                    if combined_mask is None:
                        combined_mask = mask
                    else:
                        combined_mask = combined_mask | mask

                # Convert mask to binary (0/1)
                mask_binary = (combined_mask > 127).astype(np.uint8)

                # Find all instances in the mask
                instances = self._extract_instances(mask_binary)

                for instance_mask in instances:
                    # Calculate bounding box
                    rows = np.any(instance_mask, axis=1)
                    cols = np.any(instance_mask, axis=0)

                    if not rows.any() or not cols.any():
                        continue

                    ymin, ymax = np.where(rows)[0][[0, -1]]
                    xmin, xmax = np.where(cols)[0][[0, -1]]

                    # COCO format: [x, y, width, height] normalized
                    bbox = [
                        float(xmin) / img_width,
                        float(ymin) / img_height,
                        float(xmax - xmin + 1) / img_width,
                        float(ymax - ymin + 1) / img_height
                    ]

                    area = float(instance_mask.sum())

                    # Encode mask to RLE
                    instance_mask_fortran = np.asfortranarray(instance_mask)
                    rle = mask_utils.encode(instance_mask_fortran)
                    if isinstance(rle['counts'], bytes):
                        rle['counts'] = rle['counts'].decode('utf-8')

                    # Add annotation
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": bbox,
                        "area": area,
                        "segmentation": rle,
                        "iscrowd": 0
                    })

                    annotation_id += 1

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        return coco_data

    def _extract_instances(self, mask_binary: np.ndarray) -> List[np.ndarray]:
        """Extract individual instances from a binary mask"""
        from scipy import ndimage

        # Label connected components
        labeled_mask, num_features = ndimage.label(mask_binary)

        instances = []
        for i in range(1, num_features + 1):
            instance_mask = (labeled_mask == i).astype(np.uint8)
            instances.append(instance_mask)

        return instances


def main():
    parser = argparse.ArgumentParser(description="Convert TinyUSFM datasets to COCO format")
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to dataset config YAML file (e.g., config/data/BUSBRA.yaml)'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        required=True,
        help='Output directory for COCO format data'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='Splits to convert (default: train val test)'
    )

    args = parser.parse_args()

    # Convert dataset
    converter = COCOConverter(args.config, args.output_dir)
    converter.convert_all_splits()

    print(f"\n{'='*60}")
    print("✓ Conversion complete!")
    print(f"{'='*60}")
    print(f"COCO annotations saved to: {args.output_dir}")
    print("\nNext steps:")
    print("1. Copy/symlink your images to the output directory")
    print("2. Use with SAM3 training script:")
    print(f"   python examples/train_simple_segmentation.py \\")
    print(f"       --data_root {args.output_dir} \\")
    print(f"       --checkpoint ./checkpoints/sam3_hiera_b+.pt \\")
    print(f"       --output_dir ./experiments/my_experiment")


if __name__ == "__main__":
    main()
