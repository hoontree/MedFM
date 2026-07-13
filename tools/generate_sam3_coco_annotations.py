"""Generate COCO-format image folders + annotation JSONs for SAM3 fine-tuning
from medfm's existing ultrasound datasets.

SAM3's native data pipeline (``sam3.train.data.sam3_image_dataset.Sam3ImageDataset``
/ ``COCO_FROM_JSON``) requires a real COCO JSON file + image folder on disk —
there is no way to feed it in-memory tensors directly. See the reference
fine-tuning recipe at
``model/sam3_vendor/sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml``,
which expects exactly this ``img_folder`` + ``ann_file`` (``_annotations.coco.json``)
layout.

This script does **not** reimplement any dataset-selection/split logic. It
composes the exact same Hydra config every other model trains on (``data:
dynamic``), then calls ``SegDatasetProcessor.load_dataset_from_config`` —
the same call ``build_data_loaders`` makes internally — so the resulting
COCO split is byte-for-byte the same train/val/test partition (same 70/15/15
stratified split, same seed) used everywhere else in the project. It only
adds a translation step: each dataset's mask tensor is converted into
COCO-style bbox + RLE-segmentation annotations, one instance per distinct
foreground class value in the mask (same technique as
``model/sam3_vendor/sam3/train/data/simple_segmentation_dataset.py``'s
``_mask_to_annotations``, but pre-baked to disk instead of computed lazily
at dataloader time — SAM3's own ``COCO_FROM_JSON`` loader needs a category
list up front, which a lazy per-sample conversion can't provide).

Usage:
    # Default: BUSBRA+BUSI+B, binary (num_classes=1), train+val splits.
    uv run tools/generate_sam3_coco_annotations.py

    # Include the internal test split and a specific external eval set.
    uv run tools/generate_sam3_coco_annotations.py --splits train,val,test \
        --external BUID

    # Multiclass (benign/malignant).
    uv run tools/generate_sam3_coco_annotations.py --num-classes 3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from pycocotools import mask as mask_util

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from hydra import compose, initialize_config_dir  # noqa: E402
from config.schema import register_schemas  # noqa: E402
from model import sam3_prompts  # noqa: E402
from utils.data_processing import SegDatasetProcessor  # noqa: E402

DEFAULT_OUT = PROJECT_DIR / "datasets" / "sam3_coco"
DEFAULT_INTERNAL_DATASETS = ["BUSBRA", "BUSI", "B"]


def _mask_tensor_to_class_map(mask_tensor, num_classes: int) -> np.ndarray:
    """Collapse medfm's mask tensor format into a single [H,W] class map
    (0=background). Binary: `_format_mask` gives [1,H,W] 0/1. Multiclass:
    [C,H,W] one-hot, channel 0 is background by convention."""
    if num_classes == 1:
        return (mask_tensor[0].numpy() > 0.5).astype(np.uint8)
    return mask_tensor.argmax(dim=0).numpy().astype(np.uint8)


def _mask_to_coco_annotations(mask_np: np.ndarray, image_id: int, ann_id_start: int) -> list:
    """One instance per distinct foreground class value; bbox from the
    instance's pixel extent, segmentation as COCO RLE."""
    annotations = []
    next_id = ann_id_start
    for cls_val in (v for v in np.unique(mask_np) if v > 0):
        instance_mask = (mask_np == cls_val).astype(np.uint8)
        if instance_mask.sum() == 0:
            continue
        rows, cols = np.any(instance_mask, axis=1), np.any(instance_mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        bbox = [
            float(xmin), float(ymin),
            float(xmax - xmin + 1), float(ymax - ymin + 1),
        ]
        rle = mask_util.encode(np.asfortranarray(instance_mask))
        rle["counts"] = rle["counts"].decode("utf-8")
        annotations.append({
            "id": next_id,
            "image_id": image_id,
            "category_id": int(cls_val),
            "bbox": bbox,
            "area": float(instance_mask.sum()),
            "segmentation": rle,
            "iscrowd": 0,
        })
        next_id += 1
    return annotations


def generate_split(
    cfg, name: str, split: str, out_root: Path, force_external: bool, num_classes: int
):
    dataset = SegDatasetProcessor.load_dataset_from_config(
        cfg, name, split, force_external=force_external, transform=False
    )
    if len(dataset) == 0:
        print(f"[skip] {name}:{split} -- 0 samples")
        return None

    split_dir = out_root / f"{name}_{split}"
    img_dir = split_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    images, annotations = [], []
    next_ann_id = 1
    n_empty = 0

    for idx in range(len(dataset)):
        image_tensor, mask_tensor, _low_res, filename = dataset[idx]
        img_np = (image_tensor.clamp(0, 1).numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        pil_img = PILImage.fromarray(img_np, mode="RGB")
        file_name = f"{idx:06d}_{filename}.jpg"
        pil_img.save(img_dir / file_name, quality=95)

        h, w = img_np.shape[:2]
        images.append({"id": idx, "file_name": file_name, "height": h, "width": w})

        class_map = _mask_tensor_to_class_map(mask_tensor, num_classes)
        anns = _mask_to_coco_annotations(class_map, image_id=idx, ann_id_start=next_ann_id)
        if not anns:
            n_empty += 1
        annotations.extend(anns)
        next_ann_id += len(anns)

    # Category names double as SAM3's grounding query text, so they come from the
    # shared prompt contract (model/sam3_prompts.py) that Sam3Teacher also reads —
    # the FT text and the KD teacher text cannot drift apart.
    categories = [
        {"id": c, "name": name}
        for c, name in enumerate(sam3_prompts.class_prompts(num_classes), start=1)
    ]

    ann_path = split_dir / "_annotations.coco.json"
    with open(ann_path, "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)

    print(
        f"[ok] {name}:{split} -> {len(images)} images, {len(annotations)} annotations, "
        f"{n_empty} empty-mask images"
    )
    print(f"     img_folder={img_dir}")
    print(f"     ann_file={ann_path}")
    return {"img_folder": str(img_dir), "ann_file": str(ann_path)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", default=",".join(DEFAULT_INTERNAL_DATASETS),
                    help="comma list of internal datasets (medfm's own 70/15/15 split)")
    p.add_argument("--external", default="",
                    help="comma list of external (usage=external, wholesale) datasets, e.g. BUID")
    p.add_argument("--splits", default="train,val",
                    help="comma list of splits to generate for --datasets (external is always 'test')")
    p.add_argument("--num-classes", type=int, default=1,
                    help="1 = binary (lesion), >1 = multiclass (matches data.num_classes)")
    p.add_argument("--img-size", type=int, default=1008, help="matches sam3.yaml's data.img_size override")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    return p.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.out_dir)

    register_schemas()
    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_DIR / "config")):
        cfg = compose(
            config_name="train_sam",
            overrides=[
                "model=sam3",
                f"data.img_size={args.img_size}",
                f"data.num_classes={args.num_classes}",
                "data.normalization=none",  # keep raw [0,1] tensors; SAM3 normalizes separately
            ],
        )

    manifest = {}
    for name in [d for d in args.datasets.split(",") if d]:
        for split in [s for s in args.splits.split(",") if s]:
            result = generate_split(cfg, name, split, out_root, force_external=False, num_classes=args.num_classes)
            if result:
                manifest[f"{name}_{split}"] = result

    for name in [d for d in args.external.split(",") if d]:
        result = generate_split(cfg, name, "test", out_root, force_external=True, num_classes=args.num_classes)
        if result:
            manifest[f"{name}_test"] = result

    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
