"""Dataset wrapper that augments a base segmentation dataset with SAM cache.

For each ``(image, mask, low_res_mask, filename)`` triple returned by the
underlying dataset, this wrapper attaches the corresponding cached SAM
teacher tensors saved by
:mod:`project.seg_transmiter_lite.sam_teacher_cache`:

* ``sam_mask``       – ``[1, H, W]`` float binary
* ``sam_score``      – scalar float
* ``sam_embedding``  – ``[C, h, w]`` (always present; zeros if missing)
* ``has_lesion``     – 0/1 flag for samples with no lesion in GT

The returned tuple is::

    (image, mask, low_res_mask, filename, sam_dict)

so trainers can ``*_, sam = batch`` to ignore the SAM payload when running
the supervised-only baseline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)


class SAMCacheDataset(Dataset):
    """Wrap a base dataset and pair every sample with cached SAM outputs.

    Args:
        base:          Underlying ultrasound dataset (returns 4-tuple).
        cache_dir:     Directory containing ``<image_id>.pt`` files.
        embedding_shape: Fallback ``(C, h, w)`` for missing embeddings,
                       so the collate step yields tensors of consistent shape.
                       Defaults to SAM vit_b @ 224 -> (256, 14, 14).
        strict:        If True, raise on any missing cache file; otherwise
                       fall back to a zero record with ``has_lesion=0``.
    """

    def __init__(
        self,
        base: Dataset,
        cache_dir: str | Path,
        embedding_shape: tuple = (256, 14, 14),
        strict: bool = False,
    ):
        super().__init__()
        self.base = base
        self.cache_dir = Path(cache_dir)
        self.embedding_shape = embedding_shape
        self.strict = strict
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"SAM cache dir not found: {self.cache_dir}")

    def __len__(self):
        return len(self.base)

    def _empty_record(self, image_shape):
        return {
            "sam_mask": torch.zeros((1, image_shape[-2], image_shape[-1])),
            "sam_score": torch.tensor(0.0),
            "sam_embedding": torch.zeros(self.embedding_shape),
            "has_lesion": torch.tensor(0.0),
        }

    def _load_record(self, filename: str, image_shape):
        # Try a few candidate filenames so the wrapper is robust to the
        # caller producing the cache from a different split layout.
        candidates = [
            self.cache_dir / f"{filename}.pt",
            self.cache_dir / "train" / f"{filename}.pt",
            self.cache_dir / "val" / f"{filename}.pt",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            if self.strict:
                raise FileNotFoundError(f"Missing SAM cache for {filename}")
            return self._empty_record(image_shape)

        try:
            data = torch.load(path, map_location="cpu")
        except Exception as e:  # noqa: BLE001
            LOGGER.warning("Failed to load SAM cache %s: %s", path, e)
            return self._empty_record(image_shape)

        emb = data.get("sam_image_embedding")
        if emb is None:
            emb = torch.zeros(self.embedding_shape)
        return {
            "sam_mask": data.get("sam_mask", torch.zeros((1, image_shape[-2], image_shape[-1]))).float(),
            "sam_score": data.get("sam_score", torch.tensor(0.0)).float(),
            "sam_embedding": emb.float(),
            "has_lesion": torch.tensor(1.0 if data.get("has_lesion", True) else 0.0),
        }

    def __getitem__(self, idx):
        image, mask, low_res, filename = self.base[idx]
        sam = self._load_record(filename, image.shape)
        return image, mask, low_res, filename, sam
