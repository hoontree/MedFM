"""One robust way to read model weights out of this project's checkpoints.

Historically each trainer wrapped the model state_dict under a different key —
``DistillTrainer`` used ``"model_state_dict"``, ``SAM3TrainerAdapter`` used
``"model"``, ``BaseTrainer`` saved a raw state_dict, and some checkpoints carry
``"state_dict"``. Each load site hand-unwrapped only the key it happened to
write, so a checkpoint saved by one path and loaded by another matched *zero*
parameters under ``strict=False`` and silently fell back to base/HF weights (the
exact failure Stage A hit in ``Sam3Teacher._load_finetuned``).

This module centralizes the unwrap so any load site accepts any of the schemas,
and turns "matched (almost) nothing" into a hard error instead of a silent
revert. Save formats are intentionally left as-is here (external consumers such
as ``infer.py`` read them directly); only the read path is unified.

The SAM LoRA path is deliberately *not* covered: ``LoRA_Sam`` persists only its
LoRA tensors via its own ``save_lora_parameters`` / ``load_lora_parameters`` and
is not a plain ``state_dict`` — a genuinely different artifact type.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch

# Envelope keys, most specific first, that may wrap the model state_dict.
_MODEL_KEYS = ("model_state_dict", "model", "state_dict")


def unwrap_state_dict(obj) -> Dict:
    """Return the model parameter dict from any project checkpoint envelope.

    Accepts a raw ``state_dict`` or a dict wrapping it under ``model_state_dict``
    / ``model`` / ``state_dict``. A raw state_dict is returned unchanged.
    """
    if isinstance(obj, dict):
        for k in _MODEL_KEYS:
            v = obj.get(k)
            if isinstance(v, dict):
                return v
    return obj


def load_model_weights(
    model,
    path,
    *,
    map_location="cpu",
    strict: bool = False,
    error_on_empty: bool = True,
) -> Tuple[list, list]:
    """Load ``model``'s weights from any project checkpoint schema at ``path``.

    Args:
        strict: forwarded to ``load_state_dict``. ``False`` tolerates partial
            overlaps (SAM backbones, LoRA-augmented modules); ``True`` demands an
            exact key match.
        error_on_empty: with ``strict=False``, raise if the checkpoint matched no
            model parameters (i.e. every current key is ``missing``) — that means
            the file's schema was unrecognized and the model silently kept its
            initial weights, never what you asked to load.

    Returns ``(missing, unexpected)`` from ``load_state_dict``.
    """
    obj = torch.load(str(path), map_location=map_location, weights_only=False)
    sd = unwrap_state_dict(obj)
    # Zero key overlap == the checkpoint's schema was unrecognized and nothing
    # meaningful would load; a legitimate partial load (SAM backbone at a
    # different resolution, LoRA-augmented modules) still shares many keys.
    # (Can't key off load_state_dict's `missing` count: buffers like
    # BatchNorm.num_batches_tracked get a default and undercount it.)
    if error_on_empty and not strict:
        model_keys = set(model.state_dict().keys())
        if model_keys and not (model_keys & set(sd.keys())):
            raise RuntimeError(
                f"Checkpoint {Path(path).name} shares no keys with the model — its "
                "schema is likely unrecognized (expected a raw state_dict or one wrapped "
                "under 'model_state_dict'/'model'/'state_dict'). Refusing to run on "
                "silently-unloaded weights."
            )
    return model.load_state_dict(sd, strict=strict)
