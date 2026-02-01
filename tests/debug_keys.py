import torch
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from model.segment_anything.build_sam import build_sam_vit_b


def print_keys(nc):
    print(f"\n--- Keys for num_classes = {nc} ---")
    model, _ = build_sam_vit_b(image_size=1024, num_classes=nc, checkpoint=None)
    keys = [k for k in model.state_dict().keys() if k.startswith("mask_decoder.")]
    for k in keys:
        if "output_hypernetworks" in k or "iou_prediction" in k or "mask_tokens" in k:
            print(k)


if __name__ == "__main__":
    print_keys(1)
