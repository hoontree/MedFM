"""Runtime patches the vendored SAM3 needs before it can be *trained*.

Shared by every SAM3 training path — ``trainers/sam3_adapter.py`` (native fine-tune)
and ``model/sam3_student.py`` (SAM3 as a KD student) — so the patch cannot drift
between them or be forgotten on one side. Importing it does nothing; call the function.

Deliberately dependency-free apart from torch + the vendored package, so ``model/``
never has to import from ``trainers/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# The vendored SAM3 imports itself as a top-level `sam3` package, so its root must be
# importable before `from sam3.model import ...` resolves.
_SAM3_VENDOR_ROOT = Path(__file__).resolve().parent / "sam3_vendor"
if str(_SAM3_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAM3_VENDOR_ROOT))

_fused_mlp_patched = False


def patch_fused_mlp_for_training() -> None:
    """Make SAM3's ViT-Det MLP differentiable.

    ``sam3/model/vitdet.py::Mlp.forward`` always calls a fused ``addmm_act`` kernel
    (``sam3/perflib/fused.py``) that detaches the linear weights and *raises*
    ``ValueError("Expected grad to be disabled.")`` the moment autograd is on — it is an
    inference-only optimization with no training fallback upstream. Any path that
    back-propagates through the vision backbone therefore has to swap in a plain
    ``F.linear`` + activation equivalent while grad is enabled; the fast fused kernel
    still runs under ``torch.no_grad()`` (eval / the frozen KD teacher).

    Idempotent — safe to call repeatedly (repeated dry runs, per-process under DDP).
    """
    global _fused_mlp_patched
    if _fused_mlp_patched:
        return

    import torch.nn.functional as F
    from sam3.model import vitdet as _vitdet

    original_addmm_act = _vitdet.addmm_act

    def _trainable_addmm_act(activation, linear, mat1):
        if not torch.is_grad_enabled():
            return original_addmm_act(activation, linear, mat1)
        x = linear(mat1)
        if activation in (F.relu, torch.nn.ReLU):
            return F.relu(x)
        if activation in (F.gelu, torch.nn.GELU):
            return F.gelu(x)
        raise ValueError(f"Unexpected activation {activation}")

    _vitdet.addmm_act = _trainable_addmm_act
    _fused_mlp_patched = True
