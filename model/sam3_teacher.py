"""SAM3 wrapped as a frozen dense-logit KD teacher.

SAM3 is a promptable, open-vocabulary detection+segmentation model (DETR-style):
its native output is a set of instance masks selected by a Hungarian matcher, not
the dense ``[B, C, H, W]`` per-pixel logit map the distillation pipeline consumes
(``distillers.unified_distiller`` reads ``teacher_outputs["masks"]`` and
interpolates it to the student resolution).

``Sam3Teacher`` bridges that gap. Given the student's imagenet-normalized input
batch it:
  1. de-normalizes back to [0,1] RGB and re-normalizes to SAM3's own 1008-res
     [0.5]-mean pipeline (the resolution/normalization bridge),
  2. runs SAM3 grounding with a fixed text prompt **per foreground class**
     ("lesion" for binary; "benign"/"malignant" for the 3-class task — the same
     category names baked into the COCO manifest by
     ``tools/generate_sam3_coco_annotations.py`` so the FT text and the teacher
     text match),
  3. aggregates the predicted instances of each prompt (score-gated max in
     probability space) into a per-class foreground **probability** map, then
     assembles them into a dense ``[B, num_classes, H, W]`` **logit** map
     (binary → single fg log-odds channel; multiclass → ``[bg, c1, …]`` with
     ``bg = 1 - max_c fg_c``, encoded as log-probs so the downstream
     channel-softmax / CWD recovers a proper categorical), and
  4. returns a bare ``[B, num_classes, H, W]`` tensor so the existing distiller /
     reliability code can treat it exactly like the SAM teachers.

The model is frozen and run under ``torch.no_grad`` (not the processor's
``inference_mode``, whose "inference tensors" cannot be mixed into the student's
autograd graph). It is an online teacher: the KD loop feeds it the *augmented*
image each step, so its outputs cannot be pre-cached per file.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import sam3_prompts
from model.sam3_prompts import DEFAULT_PROMPT_SET

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The vendored SAM3 package imports itself internally as a top-level `sam3`
# package (`from sam3.model import ...`), so the submodule root must be on
# sys.path before importing model.sam3.* (mirrors trainers/sam3_adapter.py).
_SAM3_VENDOR_ROOT = PROJECT_ROOT / "model" / "sam3_vendor"
if str(_SAM3_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAM3_VENDOR_ROOT))

# Imagenet stats used by the seg data pipeline (utils/data_processing.py,
# normalization="imagenet"); needed to invert the student's normalization.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_EPS = 1e-6


def _project_path(path: Optional[str]) -> Optional[str]:
    """Resolve a project-relative path against the repo root when it exists."""
    if path is None or Path(path).is_absolute():
        return path
    cand = PROJECT_ROOT / path
    return str(cand) if cand.exists() else path


class Sam3Teacher(nn.Module):
    """Frozen SAM3 image model exposed as a dense per-class-logit teacher.

    Args:
        checkpoint: Fine-tuned SAM3 state_dict path (saved by
            ``SAM3TrainerAdapter._save_model``). ``None`` → use the HuggingFace
            pretrained weights (useful for smoke-testing before FT completes).
        num_classes: Output channels. ``1`` → binary foreground (single
            "lesion" grounding). ``>=2`` → multiclass: ``num_classes-1``
            foreground prompts are grounded independently and assembled into a
            ``[bg, c1, …]`` logit map.
        class_prompts: Explicit list of open-vocabulary prompts, one per
            foreground class (length ``1`` for binary, ``num_classes-1`` for
            multiclass). ``None`` → ``sam3_prompts.class_prompts(num_classes,
            prompt_set)``. Must match the FT manifest category names for the
            grounding to be meaningful — see ``model/sam3_prompts.py``.
        prompt_set: Which named prompt set to fall back on when ``class_prompts``
            is unset ("v1" bare adjectives | "v2" BI-RADS noun phrases).
        resolution: SAM3 native input resolution (1008).
        score_threshold: Instances with confidence below this are dropped before
            aggregation (keeps spurious low-score queries out of the map).
        bpe_path: BPE vocab for the text encoder.
        load_from_hf: Whether to pull the base SAM3 weights from HF before
            optionally overlaying ``checkpoint`` (kept True so the architecture
            is always correctly populated).
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        num_classes: int = 1,
        class_prompts: Optional[Sequence[str]] = None,
        prompt_set: str = DEFAULT_PROMPT_SET,
        resolution: int = 1008,
        score_threshold: float = 0.2,
        aggregate: str = "auto",
        bpe_path: str = "model/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        load_from_hf: bool = True,
        **kwargs,
    ):
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"Sam3Teacher num_classes must be >=1, got {num_classes}.")
        self.num_classes = int(num_classes)
        # Instance→dense aggregation.
        #   "max"  — score-gated pixel-max over ALL kept instances. Fine for the single
        #            "lesion" concept (binary), which is what the §8 binary spectrum used.
        #   "top1" — only the highest-score instance's mask, mirroring the SAM3 trainer's
        #            Hungarian-matched single-instance eval. A broad multiclass prompt
        #            ("benign") spawns many spurious instances that "max" unions into a
        #            flooded map: measured per-class Dice 0.270 (max) → 0.354 (top1), with
        #            benign over-prediction dropping 4.3x → 1.75x.
        #   "auto" — top1 when multiclass, max when binary (keeps binary results intact).
        if aggregate == "auto":
            aggregate = "top1" if self.num_classes >= 2 else "max"
        self.aggregate = aggregate
        self.score_threshold = float(score_threshold)

        # One grounding prompt per foreground channel: binary has a single
        # foreground channel; multiclass has (num_classes-1), channel 0 = bg.
        self.class_prompts = self._resolve_class_prompts(class_prompts, prompt_set)

        bpe_resolved = _project_path(bpe_path)

        from model.sam3.model_builder import build_sam3_image_model
        from model.sam3.model.sam3_image_processor import Sam3Processor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Always build a correctly-populated base model; the fine-tuned weights
        # (if any) are overlaid afterwards so we never depend on the base
        # loader's HF-specific key remapping for our own checkpoints.
        self.model = build_sam3_image_model(
            bpe_path=bpe_resolved,
            device=device,
            eval_mode=True,
            checkpoint_path=None,
            load_from_HF=load_from_hf,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )
        if checkpoint:
            self._load_finetuned(checkpoint)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Processor supplies the exact SAM3 input transform + the single-image
        # FindStage; we reuse those but drive grounding ourselves under no_grad.
        self.processor = Sam3Processor(
            self.model, resolution=resolution, device=device, confidence_threshold=self.score_threshold
        )

        self.register_buffer(
            "inet_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "inet_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )
        # Per-prompt text-embedding cache (prompts are constant across steps).
        self._text_cache: Dict[str, Dict[str, torch.Tensor]] = {}

        n_params = sum(p.numel() for p in self.model.parameters())
        LOGGER.info(
            "[Sam3Teacher] built (frozen) params=%d num_classes=%d prompts=%r score_thr=%.2f ckpt=%s",
            n_params, self.num_classes, self.class_prompts, self.score_threshold, checkpoint,
        )

    def _resolve_class_prompts(
        self, class_prompts: Optional[Sequence[str]], prompt_set: str
    ) -> List[str]:
        """One prompt per foreground channel (binary: 1; multiclass: K-1)."""
        default = sam3_prompts.class_prompts(self.num_classes, prompt_set)
        prompts = [str(p) for p in class_prompts] if class_prompts is not None else default
        if len(prompts) != len(default):
            raise ValueError(
                f"Sam3Teacher: expected {len(default)} class_prompts for num_classes="
                f"{self.num_classes}, got {len(prompts)}: {prompts}"
            )
        return prompts

    def _load_finetuned(self, checkpoint: str):
        path = _project_path(checkpoint)
        if not Path(path).exists():
            raise FileNotFoundError(f"[Sam3Teacher] fine-tuned checkpoint not found: {path}")
        # Load to CPU: the file also carries AdamW optimizer state (~2x the model),
        # which map_location=device would needlessly materialize in VRAM before
        # being discarded. load_state_dict copies into the CUDA params regardless.
        sd = torch.load(path, map_location="cpu")
        # SAM3TrainerAdapter._save_model may wrap under a key; normalize.
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        LOGGER.info(
            "[Sam3Teacher] loaded fine-tuned weights from %s (missing=%d, unexpected=%d)",
            path, len(missing), len(unexpected),
        )

    # keep the frozen SAM3 in eval mode regardless of parent .train() calls
    def train(self, mode: bool = True):  # noqa: D401
        super().train(mode)
        self.model.eval()
        return self

    def _forward_text(self, prompt: str):
        """Encode a (constant) text prompt once and reuse it."""
        if prompt not in self._text_cache:
            self._text_cache[prompt] = self.model.backbone.forward_text(
                [prompt], device=self.processor.device
            )
        # shallow copy so per-image state updates don't mutate the cache
        return dict(self._text_cache[prompt])

    def _aggregate_instances(self, outputs) -> torch.Tensor:
        """Instance masks + scores → single [1,1,h,w] foreground probability."""
        mask_logits = outputs["pred_masks"].float()        # [1, Q, h, w] (logits)
        cls_logits = outputs["pred_logits"].float()        # [1, Q, 1]
        presence = outputs["presence_logit_dec"].float()   # [1, ...]

        # Per-instance confidence = sigmoid(cls) * sigmoid(presence) (matches
        # Sam3Processor._forward_grounding).
        probs_cls = cls_logits.sigmoid()                       # [1, Q, 1]
        presence_score = presence.sigmoid().unsqueeze(1)       # [1, 1, ...]
        score = (probs_cls * presence_score).squeeze(-1)       # [1, Q]
        mask_probs = mask_logits.sigmoid()                     # [1, Q, h, w]

        if self.aggregate == "top1":
            # Single highest-score instance, gated by threshold. Mirrors the
            # trainer's single matched-instance eval and avoids unioning many
            # spurious instances into one flooded map. Kept on-device: an
            # .item() here would sync the GPU once per prompt per image.
            q = score.argmax(dim=1, keepdim=True)                        # [1, 1]
            idx = q[..., None, None].expand(-1, -1, *mask_probs.shape[-2:])
            fg = mask_probs.gather(1, idx)                               # [1, 1, h, w]
            keep = (score.gather(1, q) > self.score_threshold).to(fg.dtype)
            return fg * keep[..., None, None]

        # Default "max": score-gated pixel-max over all kept instances.
        keep = (score > self.score_threshold).to(mask_probs.dtype)  # [1, Q]
        mask_probs = mask_probs * keep[..., None, None]
        fg_prob, _ = mask_probs.max(dim=1, keepdim=True)       # [1, 1, h, w]
        return fg_prob

    @torch.no_grad()
    def _dense_logit_one(self, img01_chw: torch.Tensor, out_hw) -> torch.Tensor:
        """Ground a single [3,H,W] (in [0,1]) image → [1, num_classes, out_h, out_w]
        logit map. The 1008-res image backbone runs once; each foreground prompt
        reuses those features with only its text swapped."""
        proc = self.processor
        model = self.model

        # SAM3 input transform (uint8→resize 1008→float→[0.5] normalize).
        img = proc.transform(img01_chw).unsqueeze(0).to(proc.device)  # [1,3,R,R]
        # SAM3's ViT-Det uses a fused MLP kernel that emits bf16; run the whole
        # forward under bf16 autocast (as the SAM3 trainer does) so linear
        # layers see a consistent dtype instead of Float/BFloat16 mismatches.
        fg_probs: List[torch.Tensor] = []
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(proc.device != "cpu")):
            backbone_out = model.backbone.forward_image(img)  # image features (prompt-independent)
            # Constant across prompts (a pair of empty box tensors) — build once.
            geometric_prompt = model._get_dummy_prompt()
            for prompt in self.class_prompts:
                bo = dict(backbone_out)
                bo.update(self._forward_text(prompt))
                outputs = model.forward_grounding(
                    backbone_out=bo,
                    find_input=proc.find_stage,
                    geometric_prompt=geometric_prompt,
                    find_target=None,
                )
                fg_probs.append(self._aggregate_instances(outputs))

        eps = _EPS
        if self.num_classes == 1:
            # Binary: single foreground channel as calibrated log-odds (unchanged
            # contract — Evaluator/reliability read teacher confidence = sigmoid).
            fg_prob = fg_probs[0].clamp(eps, 1.0 - eps)
            logits = torch.log(fg_prob / (1.0 - fg_prob))            # [1, 1, h, w]
        else:
            # Multiclass: [1, K-1, h, w] per-class fg prob, plus a background
            # channel = 1 - max_c fg_c (prob that no foreground prompt fired).
            fg = torch.cat(fg_probs, dim=1)                          # [1, K-1, h, w]
            bg = 1.0 - fg.max(dim=1, keepdim=True).values           # [1, 1, h, w]
            probs = torch.cat([bg, fg], dim=1).clamp(eps, 1.0)      # [1, K, h, w]
            # Encode as log-probs: downstream softmax(dim=1) (CWD / meta) then
            # recovers a normalized categorical over {bg, c1, …}.
            logits = torch.log(probs)                                # [1, K, h, w]

        logits = F.interpolate(
            logits.float(), size=out_hw, mode="bilinear", align_corners=False
        )
        return logits

    @torch.no_grad()
    def forward(self, images: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """images: [B,3,H,W] imagenet-normalized → [B,num_classes,H,W] logits.

        Returns a bare tensor (like the TinyUSFM student), not a dict. The
        distiller's ``_call_teacher`` wraps a non-dict return in
        ``{"masks": ...}``, and ``Evaluator_seg`` (used to log teacher quality in
        ``_final_evaluation``) calls ``model(images)`` and expects a tensor — a
        dict return breaks it (``'dict' has no attribute 'shape'``).
        """
        b, _, h, w = images.shape
        x01 = (images * self.inet_std + self.inet_mean).clamp(0.0, 1.0)
        logits = [self._dense_logit_one(x01[i], (h, w)) for i in range(b)]
        return torch.cat(logits, dim=0)
