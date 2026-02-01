import torch
import torch.nn as nn
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from model.segment_anything.build_sam import build_sam_vit_b


def analyze_nc(nc, checkpoint_path):
    print(f"\n--- Analyzing num_classes = {nc} ---")

    # 1. Build model
    model, _ = build_sam_vit_b(image_size=1024, num_classes=nc, checkpoint=None)

    # 2. Record initial state
    state_before = {k: v.clone() for k, v in model.state_dict().items()}

    # 3. Load checkpoint
    with open(checkpoint_path, "rb") as f:
        state_dict = torch.load(f, map_location="cpu")

    try:
        model.load_state_dict(state_dict)
        print("Result: load_state_dict(strict=True) SUCCEEDED")
    except Exception as e:
        print(f"Result: load_state_dict(strict=True) FAILED as expected")
        from model.segment_anything.build_sam import load_from

        new_state_dict = load_from(model, state_dict, 1024, 16)
        model.load_state_dict(new_state_dict)
        print("Result: load_from fallback used.")

    state_after = model.state_dict()

    # 4. Check which keys changed
    loaded_keys = []
    skipped_keys = []

    # Focus on mask_decoder
    for k in state_after.keys():
        if k.startswith("mask_decoder."):
            if not torch.equal(state_before[k], state_after[k]):
                loaded_keys.append(k)
            else:
                skipped_keys.append(k)

    print(f"Total decoder keys: {len(loaded_keys) + len(skipped_keys)}")
    print(f"Loaded decoder keys: {len(loaded_keys)}")
    print(f"Skipped decoder keys: {len(skipped_keys)}")
    if skipped_keys:
        print("Sample skipped keys (first 10):")
        for sk in skipped_keys[:10]:
            print(f"  - {sk}")

    # Calculate param counts
    loaded_params = sum(model.state_dict()[k].numel() for k in loaded_keys)
    total_decoder_params = sum(
        model.state_dict()[k].numel()
        for k in state_after.keys()
        if k.startswith("mask_decoder.")
    )

    print(
        f"Decoder utilization: {loaded_params:,} / {total_decoder_params:,} ({loaded_params/total_decoder_params:.2%})"
    )


if __name__ == "__main__":
    ckpt = "checkpoints/sam_vit_b_01ec64.pth"
    if os.path.exists(ckpt):
        analyze_nc(3, ckpt)  # Original
        analyze_nc(1, ckpt)  # Typical adaptation
    else:
        print(f"Checkpoint not found at {ckpt}")
