import torch
import torch.nn as nn
import os
import sys
import copy

# Add project root to path
sys.path.append(os.getcwd())

from model.segment_anything.build_sam import build_sam_vit_b


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def analyze_utilization(num_classes_list, checkpoint_path):
    print(f"Analyzing utilization for checkpoint: {checkpoint_path}\n")
    print(
        f"{'num_classes':<15} | {'Decoder Total':<15} | {'Decoder Loaded':<15} | {'Decoder %':<10}"
    )
    print("-" * 65)

    for nc in num_classes_list:
        # Build model
        model, _ = build_sam_vit_b(image_size=1024, num_classes=nc, checkpoint=None)

        # Clone state dict before loading
        state_before = {k: v.clone() for k, v in model.state_dict().items()}

        # Manually load the checkpoint into the existing model
        # Using the same logic as build_sam.py's load_from
        with open(checkpoint_path, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")

        try:
            model.load_state_dict(state_dict)
        except:
            from model.segment_anything.build_sam import load_from

            new_state_dict = load_from(model, state_dict, 1024, 16)
            model.load_state_dict(new_state_dict)

        state_after = model.state_dict()

        decoder_total_params = count_parameters(model.mask_decoder)
        decoder_loaded_params = 0

        for name, p in model.mask_decoder.named_parameters():
            # Construct full key name to look up in state_dict
            key = f"mask_decoder.{name}"
            if not torch.equal(state_before[key], state_after[key]):
                decoder_loaded_params += p.numel()

        decoder_percent = (
            (decoder_loaded_params / decoder_total_params) * 100
            if decoder_total_params > 0
            else 0
        )

        print(
            f"{nc:<15} | {decoder_total_params:<15,}| {decoder_loaded_params:<15,} | {decoder_percent:>8.2f}%"
        )


if __name__ == "__main__":
    ckpt = "checkpoints/sam_vit_b_01ec64.pth"
    if os.path.exists(ckpt):
        analyze_utilization([1, 2, 3, 4, 10], ckpt)
    else:
        print(f"Checkpoint not found at {ckpt}")
