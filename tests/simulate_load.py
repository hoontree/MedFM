import torch
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from model.segment_anything.build_sam import build_sam_vit_b


def simulate_load_from(nc, checkpoint_path):
    print(f"\n--- Simulating load_from for num_classes = {nc} ---")
    model, _ = build_sam_vit_b(image_size=1024, num_classes=nc, checkpoint=None)
    sam_dict = model.state_dict()

    state_dict = torch.load(checkpoint_path, map_location="cpu")

    except_keys = ["mask_tokens", "output_hypernetworks_mlps", "iou_prediction_head"]

    new_state_dict = {}
    skipped_by_except = []
    skipped_by_not_in_keys = []
    included = []

    for k, v in state_dict.items():
        if not k.startswith("mask_decoder."):
            continue

        if k not in sam_dict.keys():
            skipped_by_not_in_keys.append(k)
            continue

        is_except = False
        for ex in except_keys:
            if ex in k:
                is_except = True
                break

        if is_except:
            skipped_by_except.append(k)
        else:
            included.append(k)
            new_state_dict[k] = v

    print(
        f"Total decoder keys in checkpoint: {len([k for k in state_dict.keys() if k.startswith('mask_decoder.')])}"
    )
    print(f"Skipped because not in model: {len(skipped_by_not_in_keys)}")
    print(f"Skipped because in except_keys: {len(skipped_by_except)}")
    print(f"Included in new_state_dict: {len(included)}")

    if len(skipped_by_except) > 0:
        print("\nSample skipped by except_keys:")
        for sk in sorted(skipped_by_except)[:10]:
            print(f"  - {sk}")

    if len(included) > 0:
        print("\nSample included keys:")
        for ik in sorted(included)[:10]:
            print(f"  - {ik}")


if __name__ == "__main__":
    ckpt = "checkpoints/sam_vit_b_01ec64.pth"
    if os.path.exists(ckpt):
        simulate_load_from(1, ckpt)
    else:
        print(f"Checkpoint not found at {ckpt}")
