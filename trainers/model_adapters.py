"""Model adapters: one uniform seam for calling/evaluating a teacher or student.

Distillation has to talk to models with different forward signatures — a
``LoRA_Sam`` wants ``model(images, multimask, img_size)`` and a separate
``evaluate_model_sam`` path, while the dense models (TinyUSFM, SAM3, SegFormer)
take a bare ``model(images)`` and go through ``evaluate_model``. Previously that
difference was a hardcoded ``isinstance(model, LoRA_Sam)`` binary duplicated
across three DistillTrainer methods, so a new model family had to either
masquerade as the bare-tensor "else" case or edit all three branches.

An adapter wraps one model and exposes the two operations the trainer/distiller
need — ``forward_kd(images) -> {"masks", ["features"]}`` and ``evaluate(...)``.
The single ``isinstance`` check now lives in one factory (:func:`make_adapter`);
adding a model family is one new adapter plus one factory line, with no edits to
the training loop. Behaviour is byte-for-byte what the old branches did.
"""

from __future__ import annotations

from typing import Any, Dict


class ModelAdapter:
    """Base adapter. Holds the model plus the context the call sites need.

    Args:
        model: the wrapped teacher/student module.
        cfg: the root run config (for ``data.img_size`` / ``data.num_classes``).
        model_cfg: this model's resolved config (``teacher_cfg`` / ``student_cfg``),
            used for the per-model ``img_size`` override on the SAM path.
        is_student: students request intermediate features for feature-CWD; the
            teacher forward never does. Mirrors the old _call_teacher/_call_student
            asymmetry exactly.
    """

    def __init__(self, model, cfg, model_cfg, is_student: bool):
        self.model = model
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.is_student = is_student

    def forward_args(self, images):  # pragma: no cover - interface
        """Positional+keyword args for a raw forward call: ``(args, kwargs)``.

        Exposed so callers that can't use :meth:`forward_kd` directly — e.g.
        MetaDistillTrainer's ``functional_call`` param-swap — reuse the exact same
        signature decision instead of re-deriving it.
        """
        raise NotImplementedError

    def forward_kd(self, images) -> Dict[str, Any]:
        """Call the model and normalise the output to ``{"masks", ["features"]}``."""
        args, kwargs = self.forward_args(images)
        raw = self.model(*args, **kwargs)
        if isinstance(raw, dict):
            return raw
        if self.is_student and isinstance(raw, (list, tuple)):
            return {"masks": raw[0], "features": raw[1]}
        return {"masks": raw}

    def evaluate(self, evaluator, loader, device, num_classes, return_predictions):  # pragma: no cover
        raise NotImplementedError


class LoRASamAdapter(ModelAdapter):
    """LoRA_Sam: forward takes (images, multimask, img_size); eval is the SAM path."""

    def forward_args(self, images):
        img_size = self.model_cfg.get("img_size", self.cfg.data.img_size)
        multimask = self.cfg.data.num_classes > 1
        return (images, multimask, img_size), {}

    def evaluate(self, evaluator, loader, device, num_classes, return_predictions):
        return evaluator.evaluate_model_sam(
            self.model, loader, device, num_classes,
            img_size=self.cfg.data.img_size,
            return_predictions=return_predictions,
        )


class DenseAdapter(ModelAdapter):
    """Dense segmenter (TinyUSFM / SAM3 / SegFormer): bare ``model(images)``.

    Students additionally pass ``return_features=True`` so feature-CWD can consume
    intermediate maps; a model that ignores the kwarg (e.g. SAM3) simply returns a
    bare tensor and yields no ``features`` key — unchanged from the old path.
    """

    def forward_args(self, images):
        if self.is_student:
            return (images,), {"return_features": True}
        return (images,), {}

    def evaluate(self, evaluator, loader, device, num_classes, return_predictions):
        return evaluator.evaluate_model(
            self.model, loader, device, num_classes,
            return_predictions=return_predictions,
        )


def make_adapter(model, cfg, model_cfg, is_student: bool) -> ModelAdapter:
    """Pick the adapter for *model*. The one place model type is inspected."""
    try:
        from model.sam_hybrid_adapter import LoRA_Sam
        if isinstance(model, LoRA_Sam):
            return LoRASamAdapter(model, cfg, model_cfg, is_student)
    except ImportError:
        pass
    return DenseAdapter(model, cfg, model_cfg, is_student)
