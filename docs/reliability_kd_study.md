# Reliability-aware KD: Systematic Study Plan

Status: active · Owner: hheo · Created: 2026-06-19

This document defines a systematic experiment, visualization, and analysis plan
for **Reliability-weighted Knowledge Distillation** (`w_reliability_kd`) in the
TinyUSFM↔SAM distillation framework.

Code under study:
- `utils/reliability_kd.py` — `build_reliability_map` and its per-pixel factors.
- `utils/criterion.py::ReliabilityWeightedKDLoss` — the loss the map reweights.
- `distillers/unified_distiller.py` — wiring + per-step diagnostics.
- `config/method/unified.yaml` — the tunable surface.

## 1. Motivation & hypothesis

Plain logit-KD distills *every* teacher pixel with equal trust. In medical
segmentation the teacher (SAM) is confidently wrong near boundaries and on
ambiguous regions, so uniform KD injects those errors into the student
(TinyUSFM). The reliability map down-weights KD where the teacher signal is
untrustworthy. It is a product of (up to) four per-pixel factors:

```
r = confidence
    * entropy_penalty            (use_entropy_penalty)
    * teacher_correctness_gate   (use_teacher_correctness_gate)   # GT-conditioned
    * student_bypass_gate        (use_student_bypass)             # GT + student
```

An optional post-processing step, `use_reliability_smoothing`, then applies
prediction-aware (bilateral) spatial smoothing to the composed map.

**H1 (additivity):** each factor improves student test Dice over `confidence`-only,
with the GT-conditioned gates (teacher_correctness, student_bypass) contributing most.

**H2 (mechanism):** reliability is systematically *lower* on pixels where the
teacher disagrees with GT — i.e. the map suppresses confidently-wrong teacher
pixels, not just high-entropy ones.

**H3 (sensitivity):** performance is robust to the gate weights within a band, with
`teacher_correctness_wrong_weight=0` (hard gate) competitive with soft values, and
`student_bypass_weight` controlling a precision/recall-style trade-off.

## 2. Fixed experimental setup

| Item | Value |
|------|-------|
| Teacher | SAM `vit_b`, ft, multiclass ckpt (`experiment_results/SAM_multiclass_ft/...`) |
| Student | TinyUSFM (FPN decoder), trained from pretrained backbone |
| Data | `data=dynamic`: train `[BUSBRA, BUSI, B]` (70/15/15), test `[BUID, BUS_UCLM, BUS_UCLM_filtered]` (external) |
| num_classes | 3 (multiclass) |
| temperature | 4.0 |
| Loss base | `w_task=1.0`, `w_reliability_kd=1.0`, all other `w_*=0` |
| Fixed factors | `use_entropy_penalty=true`, `confidence_mode=max_prob` (not ablated) |
| Optim | AdamW, lr 1e-4, cosine + 5ep warmup, batch 8, grad-clip 1.0 |
| Selection | best val Dice |
| Seed | 42 (single seed for screening; top-2 configs re-run with seeds {42, 1, 7}) |
| Report | per-dataset + mean test Dice / IoU / HD95 / BIoU |

All runs are launched through `distill.py` with Hydra overrides; nothing in the
base config files is mutated per-experiment.

## 3. Experiment matrix

Experiments are declared in `config/sweeps/reliability_ablation.yaml` (a manifest
of `name → hydra overrides`) and dispatched by
`tools/run_reliability_ablation.py`. Groups:

### A. Baselines (context, not reliability variants)
- `base_task_only` — `w_reliability_kd=0` (student trained on GT only).
- `base_logit_kd` — plain Hinton KD (`w_logit_kd=1`, reliability off).
- `base_uncertainty_kd` — entropy-weighted KD (`use_uncertainty_weighted_kd=true`).

### B. Factor on/off — additive build-up (primary axis)
`entropy_penalty` is **fixed ON** for every run; the build-up starts from
`confidence × entropy`, switches the GT-conditioned gates on one at a time, and
finally adds prediction-aware spatial smoothing of the composed map:

| name | entropy | teacher_gate | student_bypass | smoothing |
|------|:--:|:--:|:--:|:--:|
| `b0_base`             | on  | off | off | off |
| `b1_teacher_gate`     | on  | on  | off | off |
| `b2_student_bypass`   | on  | on  | on  (= default) | off |
| `b3_smoothing`        | on  | on  | on  | on |

`use_reliability_smoothing` runs `prediction_aware_reliability_smoothing` on the
final map (bilateral-style averaging guided by teacher-prediction similarity, so
reliability is shared within consistent regions but not across edges).

### B'. Leave-one-out from the full map (isolates each factor's marginal value)
- `lo_no_teacher_gate`, `lo_no_student_bypass`.

### C. confidence_mode
Fixed to `max_prob` (unified.yaml default) for every run — not ablated.

### D. Gate weight sensitivity (primary axis)
Compact probe of deviations from the full-map defaults (teacher
`wrong_weight=0.0`, student `bypass_weight=0.1`):
- teacher-correctness gate: `tg_wrong_0.1`, `tg_wrong_0.5` (soft vs hard);
- student-bypass gate: `sb_weight_0.0`, `sb_weight_0.3`.

### E. Temperature (secondary)
Softens both the KD target and the reliability map (sets `method.temperature`
and `method.reliability_kd.temperature` together):
- `temp_2.0`, `temp_4.0` (default), `temp_6.0`, `temp_8.0`.

### F. Batch size (secondary)
- `bs_4`, `bs_8` (default), `bs_16`.

## 3b. LoRA-teacher experiment (secondary study)

**Question:** does reliability-KD still help when the teacher is a cheaper
*LoRA*-fine-tuned SAM instead of a fully fine-tuned one? A weaker teacher makes
more confident mistakes, so the GT-conditioned gates should matter *more*.

Only the headline methods are run (no full ablation), teacher = SAM with LoRA
(rank 4, encoder + decoder), everything else identical to §2.

**Phase 1 — train the LoRA teacher** (multiclass, from the pretrained SAM
backbone; none exists yet — multiclass was only ever fully fine-tuned):
```bash
uv run train.py model=sam model.encoder_mode=lora model.decoder_mode=lora \
    model.r_e=4 model.r_d=4 hardware.gpu_ids=[<g>]
```
Then copy the best checkpoint to the stable path the teacher config expects:
```bash
mkdir -p experiment_results/SAM_multiclass_lora/checkpoints
cp logs/sam/.../checkpoints/best_epoch_*_dice*.pth \
   experiment_results/SAM_multiclass_lora/checkpoints/best.pth
```

**Phase 2 — reliability-KD with the LoRA teacher**
(`config/sweeps/reliability_teacher_lora.yaml`, manifest base swaps in
`teacher=sam_lora`):

| name | description | ft-teacher counterpart |
|------|-------------|------------------------|
| `task_only`      | GT-only student (lower bound)        | `base_task_only` |
| `logit_kd`       | plain Hinton KD                      | `base_logit_kd` |
| `reliability`    | full reliability map (default)       | `b2_student_bypass` |
| `reliability_sm` | full map + prediction-aware smoothing| `b3_smoothing` |

```bash
uv run tools/run_reliability_ablation.py \
    --manifest config/sweeps/reliability_teacher_lora.yaml \
    --group reliability_teacher_lora
```

**Read-out:** compare each LoRA-teacher run to its ft-teacher counterpart from
the main sweep (same metric keys, different W&B group). Expectation: the
reliability gain (`reliability` − `logit_kd`) is **larger** for the LoRA teacher
than for the ft teacher (H4).

**Hyperparameter comparison (lean):** the LoRA manifest also adds a small
reliability/KD knob comparison around the `reliability` centre point (T=4,
wrong=0.0, bypass=0.1) — `temp_2.0`, `temp_8.0`, `tg_wrong_0.25`,
`sb_weight_0.3`. No factor on/off ablation (that lives only in the §3 main
sweep).

## 3c. TinyUSFM-teacher experiment (reverse / headline direction)

Direction **TinyUSFM → SAM** (teacher = TinyUSFM, student = SAM) — the project's
stated headline direction, mirrored against the SAM→TinyUSFM sweeps. Kept
deliberately lean: **target-only vs reliability + a hyperparameter comparison**,
no factor ablation.

- Teacher: TinyUSFM multiclass ft ckpt (dice 0.7053); frozen.
- Student: SAM (`vit_b`, ft) trained from the pretrained backbone.
- Manifest: `config/sweeps/reliability_teacher_tinyusfm.yaml`
  (`base_overrides` swaps `teacher=tinyusfm student=sam`).

| name | description |
|------|-------------|
| `task_only`     | GT-only SAM student (lower bound) |
| `reliability`   | full reliability map (T=4 default) |
| `temp_2.0` / `temp_8.0` | temperature comparison |
| `tg_wrong_0.25` | teacher-correctness gate weight |
| `sb_weight_0.3` | student-bypass weight |

```bash
uv run tools/run_reliability_ablation.py \
    --manifest config/sweeps/reliability_teacher_tinyusfm.yaml \
    --group reliability_teacher_tinyusfm
```

## 4. Visualization & analysis

Tool: `tools/analyze_reliability.py` (Hydra, reuses `DistillTrainer` model/data
build so it is faithful to training). Outputs to `logs/reliability_analysis/<ts>/`.

**Qualitative** — per-sample panels:
`image | GT | teacher pred | student pred | confidence | entropy_penalty |
teacher_correctness_gate | student_bypass_gate | final r`,
each factor as a `[0,1]` heatmap, so the contribution of each factor is visible.

**Quantitative** (the H2/H3 evidence), aggregated over N batches per loader:
- mean of each component (mirrors the `reliability/*_mean` W&B diagnostics).
- mean reliability split by **teacher-correct vs teacher-wrong** pixels — H2.
- reliability histograms (correct vs wrong) and a single-number gap.
- fraction of pixels with `r < 0.1` (effectively gated) overall and on wrong pixels.
- Pearson corr(reliability, teacher_correctness).
Saved as `stats.json` + `reliability_hist.png` + per-sample PNGs.

**During training:** `UnifiedDistiller` already logs `reliability/<factor>_mean`
and `mean_reliability` every step to W&B — track these per run to confirm the map
behaves as designed and to compare factor scales across the ablation.

## 5. Success criteria

- H1: `b2_student_bypass` (full) > `b0_base` on mean test Dice; each LOO drop
  identifies which factor carries the gain.
- H2: mean reliability on teacher-wrong pixels < 0.5 × mean on teacher-correct
  pixels (analysis script).
- H3: a stable plateau of test Dice across the gate-weight grid (D).

## 6. Compute & execution

Available GPUs: local host (GPU 1, 2 free) + `ssh gpu4` (GPU 0–3 free, shared FS,
`uv run` available) → up to 6 parallel runs. Each ablation cell is one `distill.py`
run; the dispatcher pins `hardware.gpu_ids` and a unique run name and can target
local or gpu4 workers.

Smoke verification (this iteration): analysis script on a few batches + a 1-epoch
tiny distill (`data.train=[B] data.test=[B] training.num_epochs=1`) to confirm the
reliability path runs end-to-end before committing GPU-hours.
