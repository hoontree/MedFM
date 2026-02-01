import torch
import copy
from model.segment_anything import sam_model_registry


def check_weights_loaded(
    num_classes=1, checkpoint_path="checkpoints/sam_vit_b_01ec64.pth"
):
    print(f"\n===== Testing num_classes={num_classes} =====")

    # 1. Create model without checkpoint
    model_raw, _ = sam_model_registry["vit_b"](
        image_size=224, num_classes=num_classes, checkpoint=None
    )

    # Keep a copy of initial weights
    initial_state = copy.deepcopy(model_raw.state_dict())

    # 2. Create model with checkpoint
    model_loaded, _ = sam_model_registry["vit_b"](
        image_size=224, num_classes=num_classes, checkpoint=checkpoint_path
    )
    loaded_state = model_loaded.state_dict()

    # 3. Compare specific parts
    components = {
        "Encoder (Patch Embed)": "image_encoder.patch_embed.proj.weight",
        "Decoder (Transformer)": "mask_decoder.transformer.layers.0.self_attn.q_proj.weight",
        "Mask Token (Class-specific)": "mask_decoder.mask_tokens.weight",
        "Output MLP (Class-specific)": "mask_decoder.output_hypernetworks_mlps.0.layers.0.weight",
    }

    for name, key in components.items():
        if key not in initial_state or key not in loaded_state:
            print(f"[{name}] Key {key} not found in state_dict.")
            continue

        initial_w = initial_state[key]
        loaded_w = loaded_state[key]

        # Check if weights are different (meaning they were loaded)
        is_loaded = not torch.equal(initial_w, loaded_w)

        status = "LOADED ✅" if is_loaded else "NOT LOADED ❌ (Remains Initialized)"
        print(f"{name:30}: {status}")


if __name__ == "__main__":
    ckpt = "checkpoints/sam_vit_b_01ec64.pth"
    import os

    if not os.path.exists(ckpt):
        print(f"Error: Checkpoint {ckpt} not found!")
    else:
        # Original SAM is trained with 3 multimask outputs + 1 singlemask = 4 tokens.
        # But our build_sam uses num_classes as num_multimask_outputs.
        # Original SAM has num_classes=3.
        check_weights_loaded(num_classes=3)  # Should load everything
        check_weights_loaded(num_classes=1)  # Should skip class-specific parts
