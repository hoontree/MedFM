"""Grounding prompt text shared by the SAM3 COCO manifests and ``Sam3Teacher``.

SAM3's COCO loader uses each category's ``name`` verbatim as the grounding query
text (``coco_json_loaders.py``: ``query["query_text"] = cat_idx_to_text[cat_id]``)
and its text encoder is *frozen* during fine-tuning. So the prompts a
``Sam3Teacher`` is given at KD time must be the exact strings baked into the
manifest it was fine-tuned on — otherwise the teacher grounds text the
fine-tune never saw and silently distills garbage.

Both sides of that contract import from here, so the strings cannot drift apart.
Deliberately dependency-free (no torch) — the COCO annotation tools import it too.

Prompt sets
-----------
``v1`` — bare diagnostic adjectives. Abstract terms with almost no visual prior
in a grounding model's text space; the first multiclass FT collapsed to
"everything is benign" (per-class Dice ~0.27, benign over-predicted 4.3x).

``v2`` — BI-RADS-style noun phrases. Benign breast lesions are typically
oval/round with a circumscribed margin; malignant ones irregular with
spiculated margins. Keeping the diagnostic word gives the encoder both cues.
"""

from typing import Dict, List

# Binary (num_classes == 1): a single foreground concept.
BINARY_PROMPT = "lesion"

# Multiclass foreground prompts, keyed by class index (0 = background/normal).
# Follows the DIAGNOSIS_LABEL_MAP / dynamic.yaml convention (1=benign, 2=malignant).
PROMPT_SETS: Dict[str, Dict[int, str]] = {
    "v1": {1: "benign", 2: "malignant"},
    "v2": {
        1: "oval circumscribed benign tumor",
        2: "irregular spiculated malignant tumor",
    },
}

DEFAULT_PROMPT_SET = "v1"


def class_prompts(num_classes: int, prompt_set: str = DEFAULT_PROMPT_SET) -> List[str]:
    """Grounding prompts for each foreground channel.

    Binary (``num_classes == 1``) → ``[BINARY_PROMPT]``. Multiclass → one prompt
    per foreground class, ordered by class index (channel 0 is background).
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be >=1, got {num_classes}.")
    if num_classes == 1:
        return [BINARY_PROMPT]
    if prompt_set not in PROMPT_SETS:
        raise ValueError(
            f"Unknown prompt_set '{prompt_set}'. Expected one of {sorted(PROMPT_SETS)}."
        )
    names = PROMPT_SETS[prompt_set]
    return [names.get(c, f"class_{c}") for c in range(1, num_classes)]
