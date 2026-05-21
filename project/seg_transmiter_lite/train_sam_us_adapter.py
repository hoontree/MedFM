"""Optional reverse stage: train a small adapter on top of SAM.

This is the symmetric counterpart of
:mod:`project.seg_transmiter_lite.train_tiny_with_sam_prior` — the SAM
image encoder (frozen, or with LoRA/Conv-LoRA enabled) acts as the
deployable backbone and a small adapter provides residual class logits.

The training loop is shared; this entrypoint just sets ``direction =
sam_student`` and (optionally) flips ``cfg.freeze_base = false`` so the
SAM LoRA params get optimizer updates too.

Why a separate file?
--------------------
1. Default values for ``direction`` / ``freeze_base`` are reversed.
2. Lets ``--help`` advertise SAM-specific options explicitly.
3. Output checkpoints are saved under a different default subdir so
   the two artefacts don't collide.

Loss composition is identical to the forward stage; the SAM teacher cache
is reused if present (in which case TinyUSFM serves only as the GT-aligned
``L_sup`` baseline and SAM-prior signals come from the cache).  When you
want TinyUSFM logits as a soft teacher, set ``cfg.loss.obj=0`` and use the
project's existing ``UnifiedDistiller`` instead — there is no point
duplicating that pipeline here.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from project.seg_transmiter_lite.train_tiny_with_sam_prior import train  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train SAM+adapter (reverse stage).")
    parser.add_argument(
        "--config",
        default=str(_PROJECT_ROOT / "config" / "seg_transmiter_lite.yaml"),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--train-lora",
        action="store_true",
        help="Allow SAM LoRA/Conv-LoRA params to be optimized "
             "(sets freeze_base=False).",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("overrides", nargs="*", help="key=value overrides (dotted)")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    cfg.direction = "sam_student"
    if args.train_lora:
        cfg.freeze_base = False

    output_dir = Path(
        args.output_dir
        or cfg.get("output_dir_sam_student")
        or f"logs/seg_transmiter_lite/sam_student/{int(time.time())}"
    )
    train(cfg, output_dir)


if __name__ == "__main__":
    main()
