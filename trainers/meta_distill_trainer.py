"""Trainer for Meta-Reliability Distillation.

Extends :class:`DistillTrainer` with the bilevel optimisation that trains the
reliability predictor ``Rθ`` (in :class:`MetaReliabilityDistiller`) against the
student's *post-update* supervised performance on a held-out meta set.

Per training step (``mode="meta"``):

    1. forward student/teacher on the **train** batch; build ``r = Rθ(z)``.
    2. inner virtual update (one step, second-order graph kept so ∇θ can flow):
           φ' = φ - η ∇φ [L_sup_train + λ · r · L_KD_train]
    3. evaluate ``Sφ'`` on the **meta** batch → ``L_meta = L_sup_meta + reg``;
       update θ via ``meta_optimizer`` (the only thing trained in this phase).
    4. real student update with the *now-updated* θ (``r`` detached on this
       step) via the main optimiser — the standard ``DistillTrainer`` forward.

For ``mode="learned"`` / ``"fixed"`` there is no bilevel step: the base
``DistillTrainer.train_epoch`` is used (in ``learned`` mode the predictor is
trained jointly by the prior/sparsity folded into the distiller loss).

The inner update is restricted to the student's *trainable* parameters
(``requires_grad``) — for SAM that is the LoRA params, for TinyUSFM the whole
model — so the second-order graph stays affordable.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterator, Optional, override

import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import functional_call
from tqdm import tqdm

from trainers.distill_trainer import DistillTrainer

logger = logging.getLogger(__name__)


def _infinite(loader) -> Iterator:
    """Cycle a dataloader forever (re-shuffles each pass when the loader does)."""
    while True:
        for batch in loader:
            yield batch


class MetaDistillTrainer(DistillTrainer):
    """DistillTrainer with a meta-learned reliability predictor."""

    # Meta learning computes the task loss inside its own inner/outer loops via the
    # distiller, so it must NOT have the task loss moved to the trainer.
    _task_loss_in_trainer = False

    def __init__(self, cfg):
        # Parse meta scalars *before* super().__init__ — the base constructor
        # calls our overridden _create_optimizer(), which needs these.
        mr = cfg.method.get("meta_reliability", {}) or {}
        self.mr_mode = str(mr.get("mode", "meta"))
        self.meta_inner_lr = float(mr.get("inner_lr", cfg.training.lr))
        self.meta_lr = float(mr.get("meta_lr", 1e-3))
        self.meta_weight_decay = float(mr.get("meta_weight_decay", 0.0))
        self.meta_grad_clip = float(mr.get("meta_grad_clip", 1.0))
        self.meta_split = str(mr.get("meta_split", "train"))  # "train" | "val"

        super().__init__(cfg)

        if not hasattr(self.distiller, "reliability_predictor"):
            raise ValueError(
                "MetaDistillTrainer requires a distiller exposing `reliability_predictor` "
                "(use method=meta_reliability)."
            )

        # Cache the student's trainable parameter names once — the inner virtual
        # update and the functional meta forward both key off this set.
        self._student_param_names = [
            n for n, p in self.student.named_parameters() if p.requires_grad
        ]
        self._meta_iter: Optional[Iterator] = None
        if self.mr_mode == "meta":
            self._meta_iter = _infinite(
                self.val_loader if self.meta_split == "val" else self.train_loader
            )
        elif self.mr_mode == "fixed":
            for p in self.distiller.reliability_predictor.parameters():
                p.requires_grad = False

    # --- optimisers ---------------------------------------------------------
    @override
    def _create_optimizer(self):
        """Split predictor params off into a separate meta-optimiser (meta mode).

        In ``learned`` / ``fixed`` mode the base optimiser is used unchanged
        (the predictor rides the main optimiser in ``learned`` mode and is
        frozen in ``fixed`` mode).
        """
        if getattr(self, "mr_mode", "meta") != "meta":
            super()._create_optimizer()
            self.meta_optimizer = None
            return

        predictor = self.distiller.reliability_predictor
        predictor_ids = {id(p) for p in predictor.parameters()}

        param_groups = [
            {"params": self.student.parameters(), "lr": self.cfg.training.lr}
        ]
        other_distiller = [
            p for p in self.distiller.parameters() if id(p) not in predictor_ids
        ]
        if other_distiller:
            param_groups.append({"params": other_distiller, "lr": self.cfg.training.lr})

        self.optimizer = optim.AdamW(
            param_groups, weight_decay=self.cfg.optimizer.weight_decay
        )
        self.meta_optimizer = optim.AdamW(
            predictor.parameters(),
            lr=self.meta_lr,
            weight_decay=self.meta_weight_decay,
        )
        logger.info(
            "[Meta] main optimiser over student (+%d adapter params); "
            "meta optimiser over %d predictor params (lr=%.1e)",
            len(other_distiller),
            sum(p.numel() for p in predictor.parameters()),
            self.meta_lr,
        )

    # --- resume hooks (persist the separate meta optimiser) ----------------
    @override
    def _resume_payload_extra(self) -> dict:
        if getattr(self, "meta_optimizer", None) is not None:
            return {"meta_optimizer_state_dict": self.meta_optimizer.state_dict()}
        return {}

    @override
    def _load_resume_extra(self, payload: dict) -> None:
        if getattr(self, "meta_optimizer", None) is not None and payload.get(
            "meta_optimizer_state_dict"
        ):
            self.meta_optimizer.load_state_dict(payload["meta_optimizer_state_dict"])

    # --- functional student forward ----------------------------------------
    def _student_logits_functional(self, params: Dict[str, torch.Tensor], images):
        """Forward the student with parameters swapped via ``functional_call``.

        Names absent from ``params`` fall back to the module's own
        parameters/buffers, so passing only the (fast-updated) trainable subset
        merged over the originals is sufficient.
        """
        # Reuse the student adapter's exact forward signature (SAM: (images,
        # multimask, img_size); dense student: (images,), return_features=True)
        # so the meta param-swap path can't drift from the normal forward.
        args, kwargs = self.student_adapter.forward_args(images)
        out = functional_call(self.student, params, args, kwargs)
        if isinstance(out, dict):
            return out["masks"]
        if isinstance(out, (list, tuple)):
            return out[0]
        return out

    @staticmethod
    def _to_index_mask(masks: torch.Tensor) -> torch.Tensor:
        gt = masks
        if gt.dim() == 4:
            gt = gt.argmax(dim=1) if gt.shape[1] > 1 else gt[:, 0]
        return gt.long()

    # --- the bilevel step ---------------------------------------------------
    def _meta_update(self, images, masks) -> Dict[str, float]:
        """One inner virtual update + meta gradient step on θ. Returns diagnostics."""
        d = self.distiller
        gt = self._to_index_mask(masks)

        # 1. train-batch forward; reliability with grad to θ.
        with torch.no_grad():
            teacher_outputs = self._call_teacher(images)
        teacher_logits = teacher_outputs["masks"]
        student_outputs = self._call_student(images)
        student_logits = student_outputs["masks"]
        if teacher_logits.shape != student_logits.shape:
            teacher_logits = nn.functional.interpolate(
                teacher_logits, size=student_logits.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        r, info = d.compute_reliability(student_logits, teacher_logits, gt)

        # inner train loss L_train(φ; θ): task + λ·r·KD (r attached to θ).
        inner_kd = d.reliability_kd_term(
            student_logits, teacher_logits, gt, r, detach_reliability=False
        )
        inner_loss = d.w_reliability_kd * inner_kd
        if d.w_task > 0:
            inner_loss = inner_loss + d.w_task * d._compute_task_loss(student_logits, masks)

        # 2. one-step virtual update of the trainable student params (keep graph).
        named = dict(self.student.named_parameters())
        trainable = [named[n] for n in self._student_param_names]
        grads = torch.autograd.grad(
            inner_loss, trainable, create_graph=True, allow_unused=True
        )
        fast = {}
        for n, p, g in zip(self._student_param_names, trainable, grads):
            fast[n] = p if g is None else p - self.meta_inner_lr * g
        merged = {**named, **fast}

        # 3. meta-batch forward through φ'; θ-objective = L_sup_meta + reg.
        meta_images, meta_masks, *_ = next(self._meta_iter)
        meta_images = meta_images.to(self.device)
        meta_masks = meta_masks.to(self.device)
        meta_logits = self._student_logits_functional(merged, meta_images)
        meta_sup = d._compute_task_loss(meta_logits, meta_masks)

        reg, reg_diag = d.reliability_regularization(r, teacher_logits, gt, info)
        meta_loss = meta_sup + reg

        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        nn.utils.clip_grad_norm_(
            d.reliability_predictor.parameters(), max_norm=self.meta_grad_clip
        )
        self.meta_optimizer.step()

        diag = {
            "meta/meta_sup_loss": meta_sup.item(),
            "meta/meta_total_loss": meta_loss.item(),
            "meta/inner_train_loss": inner_loss.item(),
        }
        diag.update(reg_diag)
        return diag

    # --- epoch loop ---------------------------------------------------------
    @override
    def train_epoch(self, epoch) -> Dict[str, float]:
        # Tell the distiller which epoch we're in (warm-up / prior annealing).
        if hasattr(self.distiller, "set_epoch"):
            self.distiller.set_epoch(epoch, self.cfg.training.num_epochs)

        if self.mr_mode != "meta":
            # learned / fixed: predictor trained (or frozen) inside the standard
            # distiller forward — reuse the base loop verbatim.
            return super().train_epoch(epoch)

        self.student.train()
        self.distiller.train()
        running: Dict[str, float] = {}
        pbar = tqdm(
            self.train_loader, desc=f"Epoch {epoch + 1}/{self.cfg.training.num_epochs}"
        )

        for i, (images, masks, *_) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.distiller.on_step_begin()

            # --- (a) meta phase: update θ from the virtual student update ---
            meta_diag = self._meta_update(images, masks)

            # --- (b) real student update with the updated θ (r detached) ---
            student_outputs = self._call_student(images)
            with torch.no_grad():
                teacher_outputs = self._call_teacher(images)
            loss_dict = self.distiller(student_outputs, teacher_outputs, masks)
            loss = loss_dict["loss"]

            self.optimizer.zero_grad()
            loss.backward()
            max_norm = self.cfg.optimizer.gradient_clip.get("max_norm", 1.0)
            nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=max_norm)
            self.optimizer.step()

            step_vals = {**{k: (v.item() if hasattr(v, "item") else v)
                            for k, v in loss_dict.items()}, **meta_diag}
            for k, v in step_vals.items():
                running[k] = running.get(k, 0.0) + v

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "meta": f"{meta_diag['meta/meta_sup_loss']:.4f}",
            })

            if self.global_step % 10 == 0:
                step_log = {"global_step": self.global_step}
                for k, v in step_vals.items():
                    step_log[f"step/{k}"] = v
                step_log["step/lr"] = self.optimizer.param_groups[0]["lr"]
                self._wandb_log(step_log)
            self.global_step += 1

        return {k: v / (i + 1) for k, v in running.items()}
