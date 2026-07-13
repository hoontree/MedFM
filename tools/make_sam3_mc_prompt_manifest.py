"""Build a SAM3 multiclass COCO manifest that differs from the base one ONLY in
the category *names* — i.e. the grounding text prompts.

Why: SAM3's COCO loader uses each category's ``name`` verbatim as the grounding
query text (``coco_json_loaders.py``: ``query["query_text"] = cat_idx_to_text[cat_id]``),
and SAM3's text encoder is FROZEN during fine-tuning. The first multiclass FT used
the bare diagnostic adjectives "benign"/"malignant" — abstract terms with almost no
visual prior in a grounding model's text space — and the teacher collapsed to
"everything is benign" (per-class Dice ~0.27, benign over-predicted 4.3x).

This script re-points the categories at *visually grounded* noun phrases (BI-RADS-style
ultrasound descriptors), so the frozen text encoder supplies cues the image branch can
actually align to. Images are symlinked, not re-encoded, so this is instant.

  uv run tools/make_sam3_mc_prompt_manifest.py            # -> datasets/sam3_coco_mc_v2
"""
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.sam3_prompts import PROMPT_SETS  # noqa: E402


def _abs(path) -> Path:
    """Manifests store project-root-relative paths; resolve against the root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_DIR / p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/sam3_coco_mc")
    ap.add_argument("--dst", default="datasets/sam3_coco_mc_v2")
    ap.add_argument("--prompt-set", default="v2", choices=sorted(PROMPT_SETS),
                    help="named prompt set from model/sam3_prompts.py to write "
                         "into the category names")
    args = ap.parse_args()

    prompts = PROMPT_SETS[args.prompt_set]
    src = PROJECT_DIR / args.src
    dst = PROJECT_DIR / args.dst
    manifest_in = json.loads((src / "manifest.json").read_text())

    manifest_out = {}
    for key, entry in manifest_in.items():
        split_dst = dst / key
        split_dst.mkdir(parents=True, exist_ok=True)

        # Symlink the image folder (identical pixels; only the prompt text changes).
        # A relative symlink target would resolve against the link's own directory
        # and dangle, so the target is made absolute first.
        img_dst = split_dst / "images"
        if img_dst.is_symlink() or img_dst.exists():
            img_dst.unlink()
        os.symlink(_abs(entry["img_folder"]).resolve(), img_dst)

        ann = json.loads(_abs(entry["ann_file"]).read_text())
        old = {c["id"]: c["name"] for c in ann["categories"]}
        ann["categories"] = [
            {"id": c["id"], "name": prompts.get(c["id"], c["name"])}
            for c in ann["categories"]
        ]
        new = {c["id"]: c["name"] for c in ann["categories"]}
        ann_dst = split_dst / "_annotations.coco.json"
        ann_dst.write_text(json.dumps(ann))

        # Keep manifest paths project-root-relative, matching the source manifest
        # (training runs from the project root).
        manifest_out[key] = {
            "img_folder": str(img_dst.relative_to(PROJECT_DIR)),
            "ann_file": str(ann_dst.relative_to(PROJECT_DIR)),
        }
        print(f"[ok] {key}: {old} -> {new}")

    (dst / "manifest.json").write_text(json.dumps(manifest_out, indent=2))
    print(f"\nWrote manifest: {dst / 'manifest.json'}")
    print(f"Prompts ({args.prompt_set}): {prompts}")


if __name__ == "__main__":
    main()
