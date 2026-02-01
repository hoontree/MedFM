import torch
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from model.segment_anything.build_sam import build_sam_vit_b


def check_checkpoint(checkpoint_path):
    print(f"Checking checkpoint: {checkpoint_path}")

    # 1. Inspect checkpoint keys
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    ckpt_keys = set(state_dict.keys())
    print(f"Total keys in checkpoint: {len(ckpt_keys)}")

    decoder_keys = [k for k in ckpt_keys if "mask_decoder" in k]
    print(f"Decoder keys in checkpoint: {len(decoder_keys)}")
    if len(decoder_keys) > 0:
        print(f"Sample decoder keys: {decoder_keys[:5]}")
    else:
        print("WARNING: No decoder keys found in checkpoint!")

    # 2. Build model without checkpoint first to get random weights
    num_classes = 1  # Default
    image_size = 1024  # Default
    model, _ = build_sam_vit_b(
        image_size=image_size, num_classes=num_classes, checkpoint=None
    )

    # Snapshot of initial weights
    initial_encoder_weight = model.image_encoder.patch_embed.proj.weight.clone()
    initial_decoder_weight = model.mask_decoder.transformer.layers[
        0
    ].self_attn.q_proj.weight.clone()

    # 3. Load checkpoint via the build function
    print("\nLoading checkpoint via build_sam_vit_b...")
    model_loaded, _ = build_sam_vit_b(
        image_size=image_size, num_classes=num_classes, checkpoint=checkpoint_path
    )

    # 4. Compare weights
    encoder_changed = not torch.equal(
        initial_encoder_weight, model_loaded.image_encoder.patch_embed.proj.weight
    )
    decoder_changed = not torch.equal(
        initial_decoder_weight,
        model_loaded.mask_decoder.transformer.layers[0].self_attn.q_proj.weight,
    )

    print("\nResults:")
    print(f"Encoder weights changed: {encoder_changed}")
    print(f"Decoder weights changed: {decoder_changed}")

    if decoder_changed:
        print("SUCCESS: Decoder weights were loaded from the checkpoint.")
    else:
        print(
            "FAILURE: Decoder weights were NOT loaded from the checkpoint (or they randomly matched, which is unlikely)."
        )

    # Check if specifically any transformer weights are in the checkpoint and loaded
    # The build_sam.py has a load_from function that might filter some keys.
    # Let's see if those filtered keys are actually in the checkpoint.
    except_keys = ["mask_tokens", "output_hypernetworks_mlps", "iou_prediction_head"]
    for ek in except_keys:
        found = any(ek in k for k in ckpt_keys)
        print(f"Is '{ek}' in checkpoint? {found}")


if __name__ == "__main__":
    ckpt = "checkpoints/sam_vit_b_01ec64.pth"
    if os.path.exists(ckpt):
        check_checkpoint(ckpt)
    else:
        print(f"Checkpoint not found at {ckpt}")
