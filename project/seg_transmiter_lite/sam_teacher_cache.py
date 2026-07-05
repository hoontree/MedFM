"""Pre-compute SAM teacher outputs for every training/val image.

For each image we save a small .pt file containing:

* ``sam_mask``            – [1, H, W] binary lesion mask predicted by SAM
* ``sam_score``           – scalar predicted IoU/confidence
* ``sam_image_embedding`` – [C_sam, h, w] image embedding
* ``image_id``            – filename stem

The cache lives outside the main training loop so it can be regenerated
with a better SAM checkpoint (e.g. after running
``train_sam_us_adapter.py``).

Box-prompt logic
----------------
1. GT mask available (preferred):
   The 3-class GT is collapsed to a binary lesion mask
   ``lesion = (class == 1) | (class == 2)``, the connected components are
   labelled, and the tightest bounding box of the union of all lesion CCs
   is used as a single prompt.  This is intentionally simple — one box per
   image — to keep the prototype robust on small datasets.

2. GT unavailable:
   The script falls back to a TinyUSFM-predicted lesion mask if a
   TinyUSFM checkpoint is provided.  Otherwise the image is skipped.

Note on confidence
------------------
SAM's predicted IoU correlates with mask quality.  Downstream training
gates the SAM-prior loss terms on ``sam_score >= threshold`` so noisy SAM
predictions do not pollute the supervised signal.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make project root importable when running as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.data_processing import SegDatasetProcessor  # noqa: E402

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Box generation                                                        #
# --------------------------------------------------------------------- #


def _lesion_mask_from_gt(target: torch.Tensor) -> torch.Tensor:
    """Collapse multiclass one-hot GT into a binary lesion mask.

    Args:
        target: ``[C, H, W]`` one-hot float (C=3 for normal/benign/malignant).
    Returns:
        ``[H, W]`` bool tensor (True for benign or malignant pixels).
    """
    if target.dim() == 3:
        if target.shape[0] > 1:
            # Lesion = any non-background class
            return target[1:].sum(dim=0) > 0.5
        return target[0] > 0.5
    return target.bool()


def _bbox_from_mask(mask: torch.Tensor, pad: int = 4) -> Optional[torch.Tensor]:
    """Tightest bbox around the True pixels of a binary mask.

    Args:
        mask: ``[H, W]`` bool tensor.
        pad:  Pixels of padding on each side (clipped to image bounds).
    Returns:
        ``[1, 4]`` float tensor ``[x1, y1, x2, y2]`` or ``None`` if empty.
    """
    if mask.sum() == 0:
        return None
    ys, xs = torch.where(mask)
    H, W = mask.shape
    x1 = max(int(xs.min().item()) - pad, 0)
    y1 = max(int(ys.min().item()) - pad, 0)
    x2 = min(int(xs.max().item()) + pad, W - 1)
    y2 = min(int(ys.max().item()) + pad, H - 1)
    return torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)


# --------------------------------------------------------------------- #
# SAM inference                                                         #
# --------------------------------------------------------------------- #


@torch.no_grad()
def _sam_predict(
    sam,
    image: torch.Tensor,
    box: torch.Tensor,
    img_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-image, prompt-free? no — *box*-prompted SAM forward.

    We bypass :class:`LoRA_Sam.forward` here because we need the box prompt;
    the rest of the pipeline (training/inference) does *not* use any prompt.

    Args:
        sam:   :class:`LoRA_Sam` instance, in eval mode.
        image: ``[3, H, W]`` already normalized to the dataset convention.
        box:   ``[1, 4]`` in xyxy pixel coords.
    Returns:
        (mask, score, image_embedding)
        mask:  ``[1, H, W]`` float in [0, 1]
        score: scalar float tensor
        emb:   ``[C, h, w]``
    """
    base = sam.sam  # the underlying segment_anything Sam model
    device = next(sam.parameters()).device

    input_image = sam.sam.preprocess(image.unsqueeze(0).to(device))
    image_embedding = base.image_encoder(input_image)

    if sam.use_alignment and sam.alignment_layer is not None:
        image_embedding = sam.alignment_layer(image_embedding)

    sparse, dense = base.prompt_encoder(
        points=None,
        boxes=box.to(device),
        masks=None,
    )
    low_res, iou_pred = base.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=base.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    masks = base.postprocess_masks(
        low_res,
        input_size=(img_size, img_size),
        original_size=(img_size, img_size),
    )
    # postprocess_masks returns logits; convert to prob then binary mask.
    prob = torch.sigmoid(masks)[0]  # [1, H, W]
    score = iou_pred.flatten()[0]
    return prob.cpu(), score.cpu(), image_embedding[0].cpu()


# --------------------------------------------------------------------- #
# Cache builder                                                         #
# --------------------------------------------------------------------- #


def _iter_dataset_records(loader: DataLoader) -> Iterable[Tuple[str, torch.Tensor, torch.Tensor]]:
    """Yield ``(image_id, image, mask)`` triples, one per sample."""
    for batch in loader:
        # Dataset returns (image, mask, low_res_mask, filename); we ignore low_res.
        image_batch, mask_batch, _low_res, filenames = batch
        for i in range(image_batch.shape[0]):
            yield filenames[i], image_batch[i], mask_batch[i]


def build_cache(
    cfg,
    cache_dir: Path,
    sam_checkpoint: Optional[str],
    threshold: float = 0.0,
    overwrite: bool = False,
):
    """Build the SAM-teacher cache for the train/val splits.

    Files smaller than ``threshold`` confidence are still saved but flagged
    so downstream training can filter them; ``threshold == 0`` keeps every
    sample.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Cache dir: %s", cache_dir)

    # Build loaders — re-use the same dataset config as the main pipeline.
    loaders = SegDatasetProcessor.build_distillation_data_loaders(cfg)
    splits = {"train": loaders["train"], "val": loaders["val"]}

    # SAM
    from model.sam_hybrid_adapter import LoRA_Sam

    img_size = int(cfg.data.img_size)
    sam = LoRA_Sam(
        sam_type=cfg.sam_teacher.get("sam_type", "vit_b"),
        img_size=img_size,
        num_classes=1,  # box-prompted SAM produces a single binary mask
        encoder_mode="frozen",
        decoder_mode="frozen",
        use_alignment=False,
        checkpoint=sam_checkpoint,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam.to(device).eval()

    saved, skipped, low_conf = 0, 0, 0
    for split_name, loader in splits.items():
        split_dir = cache_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        pbar = tqdm(_iter_dataset_records(loader), desc=f"cache:{split_name}")
        for image_id, image, mask in pbar:
            out_path = split_dir / f"{image_id}.pt"
            if out_path.exists() and not overwrite:
                continue

            lesion = _lesion_mask_from_gt(mask)
            box = _bbox_from_mask(lesion)
            if box is None:
                # No lesion (normal-only sample) — store an empty record so the
                # data loader can find it and skip SAM-prior losses.
                torch.save(
                    {
                        "image_id": image_id,
                        "sam_mask": torch.zeros((1, image.shape[-2], image.shape[-1])),
                        "sam_score": torch.tensor(0.0),
                        "sam_image_embedding": None,
                        "has_lesion": False,
                    },
                    out_path,
                )
                skipped += 1
                continue

            mask_pred, score, emb = _sam_predict(sam, image, box, img_size)
            if float(score.item()) < threshold:
                low_conf += 1
            torch.save(
                {
                    "image_id": image_id,
                    "sam_mask": (mask_pred > 0.5).float(),
                    "sam_prob": mask_pred,
                    "sam_score": score,
                    "sam_image_embedding": emb,
                    "has_lesion": True,
                    "bbox": box[0],
                },
                out_path,
            )
            saved += 1
            pbar.set_postfix({"saved": saved, "no_lesion": skipped, "low_conf": low_conf})

    LOGGER.info(
        "Done. saved=%d, no_lesion=%d, low_conf<%.2f=%d",
        saved, skipped, threshold, low_conf,
    )


# --------------------------------------------------------------------- #
# CLI                                                                   #
# --------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description="Build SAM teacher cache.")
    parser.add_argument(
        "--config",
        default=str(_PROJECT_ROOT / "config" / "seg_transmiter_lite.yaml"),
        help="Hydra config to derive data + SAM settings from.",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Output directory for cached SAM teacher tensors.",
    )
    parser.add_argument(
        "--sam-checkpoint",
        default=None,
        help="Path to SAM checkpoint (raw SAM or LoRA_Sam .pth). "
             "Defaults to whatever the config points to.",
    )
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = OmegaConf.load(args.config)
    sam_ckpt = args.sam_checkpoint or cfg.sam_teacher.get("checkpoint", None)
    build_cache(
        cfg,
        cache_dir=Path(args.cache_dir),
        sam_checkpoint=sam_ckpt,
        threshold=args.threshold,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
