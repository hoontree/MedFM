# SAM3-as-student distillation — experiment plan (NOT yet run)

Status: **planned, technically de-risked, not launched.** All the wiring exists and is
smoke-verified; this document is the spec to run it later. Kept separate from the
multiclass teacher-strength study (`meta_reliability_kd_study.md` §9) on purpose — it is
a distinct research direction, not a point on that spectrum.

## Question

The multiclass study found a sharp asymmetry in SAM3:

| | foreground (localization) | per-class (benign/malignant) |
|---|---|---|
| SAM3 | **0.856** (best of all models) | **0.377** (worst — below the TinyUSFM student) |
| SAM vit_h | — | 0.755 |
| TinyUSFM | — | 0.705 |

SAM3 localizes better than anything we have but classifies worse than everything,
because class identity has to pass through its **frozen text encoder** — the class *is*
the prompt. Two rounds of fixes (max→top1 aggregation 0.270→0.354; bare adjectives →
BI-RADS noun phrases 0.354→0.377) recovered +40% relative and did not close the gap.

**Question: can the class knowledge SAM3 cannot learn from text be distilled into it from
a teacher that has it?**

## Design (per the owner's decisions)

- **Student**: SAM3 (`model/sam3_student.py`, `config/model/sam3_student.yaml`), started
  from the SAM3 multiclass FT checkpoint (foreground 0.842, per-class 0.377) — it already
  localizes; only the label distinction is missing.
- **Teacher**: **TinyUSFM** (multiclass 0.705). A single controlled contrast, **not** a
  teacher-strength spectrum. (TinyUSFM, not SAM vit_h: it is the project's own student
  model and the natural reference; keeps the comparison to the model family the study is
  built around.)
- **Methods** (same 5-cell set as every other sweep, via `_teacher_strength_base.yaml`):
  task_only (SAM3 fine-tuned alone, the floor), logit_kd, reliability, learned_pseudo,
  meta_scalar.
- **Framing**: this is a KD setup where the **student is stronger than its teacher at the
  pixel task and weaker at the label task** — the regime where unweighted logit-KD should
  be actively harmful and reliability/meta KD should pay for itself. It mirrors, from the
  student side, what the SAM3-*teacher* row already showed from the teacher side
  (vanilla KD −28.8pt collapse; reliability recovers it).

## What already exists (smoke-verified)

- `model/sam3_student.py` — `Sam3Student(Sam3Teacher)`: flips the shared `trainable`
  flag; the 224→1008 bridge / per-class prompts / aggregation live once in `Sam3Teacher`.
- **Batched backbone** (`Sam3Teacher.forward`): the 1008² vision backbone runs **once per
  batch** instead of once per image; verified numerically identical to the old per-image
  loop (per-class Dice 0.3768 unchanged). This is what makes back-prop affordable.
- **Two SAM3 "inference-only" blocks made trainable**:
  - fused ViT-Det MLP kernel that raises when autograd is on → `model/sam3_patches.py`
    (`patch_fused_mlp_for_training`, shared with `trainers/sam3_adapter.py` so they can't
    drift).
  - Hungarian matcher gated on `self.training` (dereferences `find_target.boxes`, which
    we pass as `None`) → `Sam3Student._set_train_flags` puts children in train mode
    (keeps activation-checkpointing on for memory) but forces the **top-level** flag off
    (matcher off).
- **Differentiable aggregation** `aggregate="soft"` (score-weighted max), the default for
  a trainable SAM3. **This is essential, not cosmetic**: top1/max feed the instance score
  in only through an argmax / `>threshold` boolean, so gradient reaches `pred_masks` and
  **nothing else** — the classification head (`pred_logits`×presence, which *is* SAM3's
  class decision) gets zero gradient and the student can never learn to classify, while
  the loss still falls. Smoke on real images confirmed: top1 → 2/10 cls-head tensors get
  gradient; soft → 10/10.
- `tools/smoke_sam3_student.py` — measures cls-head gradient reach, peak memory, speed on
  real val images (noise grounds nothing, so gradient is legitimately 0 and can't test).
- Optimizer now filters `requires_grad` (`trainers/distill_trainer.py::_create_optimizer`)
  so AdamW doesn't allocate state for SAM3's 355M frozen text tower.

## Measured budget (RTX PRO 6000, 98 GB)

| batch | peak mem | speed |
|---|---|---|
| 1 | 11.4 GB | 2.6 img/s |
| 4 | 15.8 GB | 2.2 img/s |
| 8 | 22.9 GB | 2.1 img/s |

~2 img/s → **~30 min/epoch** (train 3934) → convergence in **hours to ~1 day per cell**;
5 cells occupy a gpu6 GPU for ~a day. This slowness is the reason it is not launched by
default — it is a deliberate spend decision, not a technical blocker.

## To launch

1. Write `config/sweeps/sam3_student.yaml`:
   ```yaml
   extends: _teacher_strength_base.yaml
   group_label: sam3_student/tinyusfm
   base_overrides:
     - student=sam3_student
     - teacher=tinyusfm
     - data.num_classes=3
   ```
   (Note: `student=` / `teacher=` roles are swapped vs the teacher-strength sweeps.
   Confirm `run_reliability_ablation.py` + `distill_trainer` resolve a non-tinyusfm
   student — `load_model_cfg` is already defaults-aware, but the student path has only
   ever been exercised with TinyUSFM. Dry-run + a `--smoke` cell first.)
2. Set SAM3-appropriate optimization: `training.lr≈1e-5` (not the distill default 1e-4 —
   848M pretrained weights), `training.batch_size=8`, tight grad-clip (SAM3 FT uses 0.1),
   `training.num_epochs` modest (it starts near-converged on localization).
3. `uv run tools/run_reliability_ablation.py --manifest config/sweeps/sam3_student.yaml \
   --smoke --workers gpu6:0` → then the full run on `gpu6:0,gpu6:1`.
4. Report per-class Dice (not just foreground): the whole question is whether the class
   gap (0.377 → ?) closes under each method.

## Success criterion

Does distillation lift SAM3's per-class Dice from 0.377 toward the teacher's 0.705 —
and does reliability/meta KD beat vanilla logit-KD in doing so (the no-harm story from
the student side)? If even reliability KD cannot move it, that is itself the finding:
the frozen-text-encoder class bottleneck is not a distillation-fixable deficiency.
