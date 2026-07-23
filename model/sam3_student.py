"""SAM3 as a **trainable** dense-logit KD student.

Why this exists
---------------
The multiclass teacher-strength study turned up a sharp asymmetry in SAM3. As a
*segmenter* it is excellent — binary foreground Dice 0.856, the best teacher on the
binary spectrum. As a *diagnostic classifier* it is the worst model we have: per-class
(benign/malignant) Dice 0.377, far below the SAM backbones (0.72–0.75) and even below
the TinyUSFM student (0.705) on the identical data and labels.

The cause is structural, not a tuning miss. SAM3 has to route class identity through a
**frozen text encoder** — the class *is* the prompt — whereas SAM's free output channels
carry it directly. Two rounds of fixes (score-gated max → top1 aggregation, 0.270→0.354;
bare adjectives → BI-RADS noun phrases, 0.354→0.377) recovered +40% relative and still
left the gap wide open: the model keeps collapsing onto whichever prompt is the more
"attractive" concept rather than discriminating between them.

That sets up the obvious question, and it is the one this class exists to ask:
**can the class knowledge SAM3 cannot learn from text be distilled into it from a
teacher that has it?** Teacher = SAM vit_h (multiclass 0.755); student = SAM3, which
already localizes better than the teacher does. It is a KD setup where the student is
*stronger than its teacher at the pixel task and weaker at the label task* — exactly the
regime where an unweighted logit-KD should be actively harmful and reliability-aware KD
should pay for itself.

Implementation
--------------
Everything that makes SAM3 look like a dense segmenter — the imagenet-224 → SAM3-1008
bridge, one grounding prompt per foreground class, the instance→dense aggregation, the
[bg, c1, …] log-prob assembly — is the *same* delicate contract the teacher uses, so it
is not duplicated here: ``Sam3Teacher`` owns it, gated on a ``trainable`` flag, and this
subclass only flips that flag and decides what is frozen.

Frozen: the text encoder (SAM3's own FT recipe keeps it frozen, and the prompt strings
are fixed by the manifest contract in ``model/sam3_prompts.py``). Trainable: everything
else.

Aggregation: a trainable student MUST use ``soft`` (score-weighted max) — ``Sam3Teacher``
resolves ``aggregate="auto"`` to ``soft`` when ``trainable=True`` for exactly this reason.
``top1``/``max`` feed the instance score in only through an argmax / a ``> threshold``
boolean (both non-differentiable), so gradient reaches ``pred_masks`` and nothing else —
and SAM3's *class* decision **is** the instance score (which prompt wins), so under those
aggregations the classification head gets exactly zero gradient and the student could
never learn the benign/malignant distinction, with a loss that still goes down. ``soft``
multiplies mask prob by score, keeping that whole path live. ``tools/smoke_sam3_student.py``
asserts the classification head actually receives gradient.
"""

from __future__ import annotations

import logging

import torch.nn as nn

from model.sam3_patches import patch_fused_mlp_for_training
from model.sam3_teacher import Sam3Teacher

LOGGER = logging.getLogger(__name__)


class Sam3Student(Sam3Teacher):
    """Trainable SAM3, exposed as a ``[B, num_classes, H, W]`` dense-logit student.

    Args:
        freeze_text_encoder: Keep SAM3's language tower frozen (default, and what
            SAM3's own fine-tune recipe does). The grounding prompts are fixed by the
            manifest contract anyway, so there is nothing for it to learn.
        See ``Sam3Teacher`` for the rest — ``checkpoint`` should normally point at the
        SAM3 multiclass FT artifact so the student starts from a model that already
        segments well and only has to acquire the class distinction.
    """

    def __init__(self, *args, freeze_text_encoder: bool = True, **kwargs):
        # SAM3's ViT-Det MLP uses an inference-only fused kernel that *raises* as soon
        # as autograd is enabled. Must be patched before the first forward, or every
        # training step dies with "Expected grad to be disabled."
        patch_fused_mlp_for_training()

        kwargs["trainable"] = True
        super().__init__(*args, **kwargs)

        for p in self.model.parameters():
            p.requires_grad_(True)
        self._set_train_flags(True)

        self.freeze_text_encoder = freeze_text_encoder
        if freeze_text_encoder:
            self._freeze_text_tower()

        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.model.parameters())
        LOGGER.info(
            "[Sam3Student] trainable=%d/%d params (text_encoder frozen=%s) "
            "num_classes=%d prompts=%r aggregate=%s",
            n_train, n_all, freeze_text_encoder, self.num_classes,
            self.class_prompts, self.aggregate,
        )

    def _set_train_flags(self, mode: bool) -> None:
        """Put SAM3's submodules in ``mode`` but force the *top-level* flag off.

        Two different things hang off ``.training`` inside SAM3, and we want opposite
        answers for each:

        * ``Sam3Image.forward_grounding`` runs its Hungarian matcher when
          ``self.training`` is set — ``if self.training: self._compute_matching(out,
          self.back_convert(find_target))`` — and ``back_convert`` dereferences
          ``find_target.boxes``. We deliberately pass ``find_target=None``: the distiller
          consumes dense logits, not SAM3's native set-matching loss, so the matcher is
          both unwanted and un-feedable. Leaving the top-level flag on crashes with
          ``'NoneType' object has no attribute 'boxes'``.
        * the vision backbone gates its activation checkpointing on *its* ``.training``
          (``act_ckpt_whole_vision_backbone and self.training``), and that is what makes
          back-prop through a 1008² ViT fit in memory at all.

        So: recurse ``mode`` into the children (checkpointing + dropout follow it), then
        clear the parent's flag alone. Same trick ``SAM3TrainerAdapter.validate`` uses in
        reverse — it sets ``self.model.training = True`` without a recursive ``.train()``
        to make the matcher produce indices while the children stay in eval.

        The upshot is that the student computes exactly what the frozen teacher computes,
        with gradients attached.
        """
        self.model.train(mode)
        self.model.training = False

    # Dotted-path *segments* identifying SAM3's language tower. Matched against
    # whole `.`-separated name components (not raw substrings) so a bare "text"
    # does not also catch unrelated params like "...context...", "...next..." or
    # "...pretext...". The frozen-text-encoder invariant is load-bearing for the
    # KD premise, so a zero-match is a hard error, not a warning.
    _TEXT_TOWER_SEGMENTS = frozenset(
        {"text", "text_encoder", "language", "language_model", "token_embed", "token_embedding"}
    )

    def _freeze_text_tower(self) -> None:
        """Freeze SAM3's language tower by segment-exact name match.

        Raises ``RuntimeError`` if nothing matches: silently training the text
        encoder would violate the invariant the whole class rests on, and the old
        warn-and-continue path let that happen invisibly whenever the vendored
        module names drifted.
        """
        frozen = 0
        matched_names = []
        for name, p in self.model.named_parameters():
            segments = {s.lower() for s in name.split(".")}
            if segments & self._TEXT_TOWER_SEGMENTS:
                p.requires_grad_(False)
                frozen += p.numel()
                matched_names.append(name)
        if frozen == 0:
            raise RuntimeError(
                "[Sam3Student] freeze_text_encoder=True but no language-tower params "
                f"matched segments {sorted(self._TEXT_TOWER_SEGMENTS)}. The vendored "
                "SAM3 parameter names likely changed; refusing to run rather than "
                "silently train the text encoder. Update _TEXT_TOWER_SEGMENTS."
            )
        LOGGER.info(
            "[Sam3Student] froze %d text-tower params across %d tensors",
            frozen, len(matched_names),
        )

    def train(self, mode: bool = True):  # noqa: D401
        """Keep the matcher-off / checkpointing-on invariant across every .train()/.eval()
        the trainer makes (it flips modes each epoch for validation)."""
        nn.Module.train(self, mode)
        self._set_train_flags(mode)
        return self

    def trainable_parameters(self):
        """The parameters an optimizer should actually see."""
        return (p for p in self.model.parameters() if p.requires_grad)
