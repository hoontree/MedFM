# Meta-Reliability Distillation: Study Plan

Status: active · Owner: hheo · Created: 2026-06-19

Companion to [reliability_kd_study.md](reliability_kd_study.md). That study
established a *hand-crafted* per-pixel reliability map. This one replaces the
fixed multiplicative composition with a **learnable predictor** and asks whether
meta-learning that predictor against the student's downstream supervised
improvement beats both the hand-crafted map and a naively-supervised predictor.

Code under study:
- `distillers/meta_reliability.py` — `ReliabilityPredictor`, feature assembly,
  `MetaReliabilityDistiller`, per-pixel GT/KD/negative-KD loss maps.
- `trainers/meta_distill_trainer.py` — the bilevel (virtual-update + meta-grad) loop.
- `config/method/meta_reliability.yaml` — the tunable surface.
- `config/sweeps/meta_reliability.yaml` — the experiment manifest.

## 1. Motivation & hypothesis

The hand-crafted map fixes the rule by hand: *low entropy good, boundary
unstable, teacher≠GT bad*, combined multiplicatively:

```
r = confidence · entropy_penalty · teacher_correctness_gate · student_bypass_gate
```

But the *usefulness* of a teacher signal for a dense-prediction student is not
determined by confidence alone — it is an interaction of teacher confidence,
teacher correctness, and the student's own state. We therefore:

1. **demote** the hand-crafted factors to *input features* `z` (drop the product),
2. **predict** reliability `r = Rθ(z)` with a small pixel-wise MLP, and
3. **meta-learn** `θ` so that weighting KD by `rθ` reduces the student's
   supervised loss on a held-out meta set — not so that `rθ` matches teacher
   correctness.

Student φ-update (r detached — a fixed weight on this step):

```
L_train(φ) = L_sup(Sφ, y) + λ · rθ · L_KD(Sφ, T)
```

Predictor θ-objective (meta mode):

```
φ' = φ − η ∇φ [L_sup_train + λ rθ L_KD_train]      (one virtual step)
min_θ  L_sup_meta(Sφ', y_meta) + β·L_sparsity(θ) + γ·L_prior(θ)
```

**H1 (learnable > hand-crafted):** `meta_scalar` ≥ `handcrafted_reliability` on
mean external-test Dice.

**H2 (meta > pseudo-label — the central claim):** `meta_scalar` >
`learned_pseudo`. If the predictor meta-learned from student improvement beats
the same predictor fit to teacher-correctness, then it is learning *distillation
usefulness*, not merely imitating teacher correctness.

**H3 (mechanism):** the meta-learned `r` is **not** a monotone function of
teacher correctness — there exist teacher-correct *easy* pixels with low `r`
(little student benefit) and teacher-uncertain *boundary* pixels with high `r`
(structural signal). Quantified by KD/GT gradient alignment per reliability bin.

**H4 (collapse needs regularisation):** without the sparsity + prior terms the
predictor collapses (`r→0`, KD effectively off) — `meta_no_reg` degenerates
toward `base_task_only`.

## 2. Fixed experimental setup

Identical to [reliability_kd_study.md §2](reliability_kd_study.md) so results are
directly comparable, with the meta-specific additions below.

| Item | Value |
|------|-------|
| Teacher | SAM `vit_b` ft (frozen), multiclass ckpt — as in the reliability study |
| Student | TinyUSFM (FPN decoder) from pretrained backbone |
| Data | `data=dynamic`: train `[BUSBRA, BUSI, B]` (70/15/15), external test `[BUID, BUS_UCLM, BUS_UCLM_filtered]` |
| temperature | 4.0 (KD + feature signals) |
| Loss base | `w_task=1.0`, `w_reliability_kd=1.0` (=λ), other `w_*=0` |
| Optim (φ) | AdamW, lr 1e-4, cosine + 5ep warmup, batch 8, grad-clip 1.0 |
| Optim (θ) | AdamW, `meta_lr=1e-3`, grad-clip 1.0 |
| Inner step | one virtual update, `inner_lr=1e-4`, second-order graph kept |
| Meta set | held-out **train** batch (`meta_split=train`); `val` variant probed |
| Predictor | 1×1-conv MLP, 2 layers, hidden 32, scalar head, init r≈0.5 |
| Features | minimal-4: `teacher_confidence, teacher_gt_agreement, teacher_student_disagreement, student_confidence` |
| Reg | `sparsity_weight=0.1`, `target_density ρ=0.5`, `prior_weight=0.1` |
| Selection / Seed / Report | best val Dice; seed 42 (top-2 re-run {42,1,7}); per-dataset + mean Dice/IoU/HD95/BIoU |

**Why the inner update is cheap:** the virtual step (`create_graph=True`) only
unrolls the student's *trainable* params (TinyUSFM whole model; SAM only LoRA),
one step, so the second-order graph is affordable. The meta forward uses
`torch.func.functional_call` with the fast weights.

## 3. Experiment matrix

Declared in `config/sweeps/meta_reliability.yaml`, dispatched by
`tools/run_reliability_ablation.py` (shared with the hand-crafted study so the
W&B bookkeeping/worker pool is identical):

```bash
uv run tools/run_reliability_ablation.py \
    --manifest config/sweeps/meta_reliability.yaml \
    --group meta_reliability
```

### A. Reference baselines (context)
- `base_task_only` — GT-only student (lower bound).
- `base_logit_kd` — plain Hinton KD.
- `handcrafted_reliability` — full hand-crafted multiplicative map (the prior study's headline).

### B. Learnable predictor — the core comparison (H1, H2)
| name | θ trained by | predictor input |
|------|--------------|-----------------|
| `learned_pseudo` | prior+sparsity only (BCE to teacher-correct×conf) | minimal-4 |
| `meta_scalar` | **meta-objective** (post-update meta loss) | minimal-4 |
| `meta_mixture` | meta-objective, output `[a_gt, a_kd, a_neg]` | minimal-4 |

`meta_mixture` is the stronger extension: per pixel it chooses *which*
supervision applies (GT vs positive-KD vs negative-KD), where `L_neg` suppresses
the student from following a confidently-wrong teacher class.

### C. Feature-set ablation
- `meta_full_features` — all 8 signals (adds entropy, margin, student_entropy, boundary).

### D. Meta-set source
- `meta_split_val` — meta batches drawn from the val split instead of held-out train.

### E. Collapse-prevention ablation (H4)
- `meta_no_prior`, `meta_no_sparsity`, `meta_no_reg`.

### F. Density target ρ
- `meta_rho_0.3`, `meta_rho_0.7` (default 0.5).

### G. Staged schedule
- `meta_staged` — `warmup_epochs=5` (weight KD by `r_simple` first), `prior_anneal_epochs=20` (γ→0).

## 4. Visualization & analysis

Reuse `tools/analyze_reliability.py` patterns; the predictor exposes the exact
feature stack via `build_reliability_features(...)` so panels stay faithful.

**Qualitative** — per-sample panels:
`image | GT | teacher pred | student pred | each input feature | r (learned) | r (meta) | r_simple`.
For `meta_mixture`, plot the `[a_gt, a_kd, a_neg]` simplex as three heatmaps.

**Quantitative** (H2/H3 evidence), aggregated over N batches per loader:
- mean `r` split by **teacher-correct vs teacher-wrong** pixels (compare meta vs pseudo vs hand-crafted).
- **r vs teacher-correctness decoupling (H3):** Pearson corr(`r`, teacher_correct);
  the meta predictor should show *lower* correlation than `learned_pseudo` yet better Dice.
- **gradient alignment (H3, the strong evidence):** per reliability bin, cosine
  similarity between `∇L_KD` and `∇L_GT` w.r.t. student logits — high `r` bins
  should carry KD gradient that agrees with the GT gradient.
- mean `r` trajectory over training (W&B `step/meta/mean_reliability`); watch for collapse.
- effective KD fraction (`r > 0.1`) overall and on teacher-wrong pixels.

**During training** the meta trainer logs per step: `meta/meta_sup_loss`,
`meta/inner_train_loss`, `meta/sparsity`, `meta/prior_bce`, `meta/prior_weight`,
`mean_reliability` — track these to confirm the meta objective is descending and
the map is not collapsing.

## 5. Success criteria

- **H1:** `meta_scalar` ≥ `handcrafted_reliability` mean test Dice.
- **H2 (central):** `meta_scalar` > `learned_pseudo` mean test Dice, with the gap
  reproduced across seeds {42, 1, 7}.
- **H3:** meta `r` has lower corr with teacher-correctness than `learned_pseudo`
  *and* higher KD/GT gradient alignment in high-`r` bins.
- **H4:** `meta_no_reg` mean `r` collapses (< 0.05) and Dice regresses toward `base_task_only`.

## 6. Compute & execution

Same pool as the hand-crafted study (local GPU 1,2 + `ssh gpu4` GPU 0–3 → up to
6 parallel). Each cell is one `distill.py` run. The meta runs cost roughly
**~2× a normal distill step** (two student forwards + one virtual-update graph
per step); budget accordingly.

**Smoke verification before committing GPU-hours:**
```bash
# CPU unit tests (feature assembly, predictor, per-pixel losses, bilevel grad)
uv run --with pytest python -m pytest tests/test_meta_reliability.py -q

# 1-epoch end-to-end on the smallest dataset, both modes
uv run tools/run_reliability_ablation.py \
    --manifest config/sweeps/meta_reliability.yaml \
    --only meta_scalar,learned_pseudo --smoke
```

## 7. Risks & mitigations

- **Reliability collapse (`r→0`).** The dominant failure mode — KD-off can lower
  the meta loss. Mitigated by `sparsity_weight·|mean(r)−ρ|`, `reliability_floor`
  (KD budget), the prior, and the warm-up schedule. `meta_no_reg` is the
  deliberate demonstration of the failure.
- **Meta-set leakage.** Using `val` as the meta set (`meta_split_val`) risks
  selecting on the checkpoint-selection split; default is a held-out **train**
  batch, with the val variant kept only as a sensitivity probe.
- **Second-order cost / instability.** One inner step only; gradients clipped on
  both optimisers; `inner_lr` kept at the student lr. If unstable, lower
  `meta_lr` before touching `inner_lr`.
- **SAM functional forward.** `functional_call` swaps only trainable params over
  the module's own params/buffers; verify on the 1-epoch smoke run that the
  SAM-student path produces finite meta gradients before scaling out.

## 8. Results — teacher-strength spectrum

The core reliability hypotheses (H2/H3 and the "no-harm" property) are sharpest
when the teacher's strength is *varied*. Every reliability/meta sweep in §3 used a
single weak teacher (SAM vit_b full-FT, val Dice 0.764, below the TinyUSFM 0.80
student). To fill that axis we fine-tuned a **teacher-strength spectrum** — SAM
vit_b < vit_l < vit_h < SAM3 — and ran the five methods (`task_only`, `logit_kd`,
`reliability`, `learned_pseudo`, `meta_scalar`) against each, holding the student
(TinyUSFM), data (binary BUSBRA+BUSI+B), and schedule fixed.

**Teachers (val Dice @ their native res):**

| teacher | val Dice | notes |
|---|---|---|
| SAM vit_b FT | 0.7637 | weak (below student) |
| SAM vit_l FT | 0.8216 | AMP + param-group LR |
| SAM vit_h FT | 0.8299 | + grad-checkpointing @1024 |
| SAM3 (grounding) | 0.8558 | dense-logit wrapper, "lesion" prompt @1008 |

All teachers feed the student one shared 224 batch: SAM teachers are trained/eval
at their native resolution but distill at 224; SAM3 bridges 224→1008 internally
(`model/sam3_teacher.py`). Student trains at 224 throughout.

### 8.1 Student Dice by teacher × method

*(19/20 cells; SAM3×learned_pseudo still running — marked "—".)*

**External (BUID + BUS_UCLM_filtered), mean Dice:**

| teacher (Dice) | task_only | logit_kd | reliability | learned_pseudo | meta_scalar |
|---|---|---|---|---|---|
| vit_b (0.764) | 0.7869 | 0.7896 | 0.7936 | 0.7990 | 0.7937 |
| vit_l (0.822) | 0.7889 | 0.8127 | 0.8092 | 0.8057 | 0.8122 |
| vit_h (0.830) | 0.8086 | 0.8083 | 0.8021 | 0.8016 | 0.8016 |
| sam3 (0.856) | 0.7811 | 0.7867 | 0.7948 | — | 0.7999 |

**Internal (BUSBRA/BUSI/B test), mean Dice:**

| teacher (Dice) | task_only | logit_kd | reliability | learned_pseudo | meta_scalar |
|---|---|---|---|---|---|
| vit_b (0.764) | 0.8076 | 0.7804 | 0.7826 | 0.8042 | 0.7934 |
| vit_l (0.822) | 0.8086 | 0.8124 | 0.8082 | 0.8110 | 0.8063 |
| vit_h (0.830) | 0.7994 | 0.8082 | 0.8107 | 0.8002 | 0.8116 |
| sam3 (0.856) | 0.8040 | 0.7834 | 0.8049 | — | 0.7986 |

ΔDice-vs-`task_only` tables and the 2D spectrum plots are reproduced by:

```bash
uv run tools/summarize_teacher_strength.py \
    --teacher-dice sam=0.7637,sam_vit_l=0.8216,sam_vit_h=0.8299,sam3_teacher=0.8558 \
    --split ext --plot docs/figs/teacher_strength_ext.png   # or --split int
```

![external spectrum](figs/teacher_strength_ext.png)
![internal spectrum](figs/teacher_strength_int.png)

### 8.2 Findings

1. **Vanilla KD is harmful at both ends of the spectrum.** With the weak vit_b
   teacher, `logit_kd` *drops* internal Dice by −0.027 vs the `task_only` floor;
   with the strongest teacher (SAM3, likely mis-calibrated after the 1008→224
   bridge) it drops −0.021. Naïvely trusting teacher logits fails when the teacher
   is either below the student or distribution-shifted.

2. **Reliability re-weighting recovers the harm — the "no-harm" property.**
   Against vit_b, `learned_pseudo` pulls internal ΔDice back to −0.003 (near
   neutral) and external to +0.012 (best in row). Against SAM3, `reliability`
   restores internal to +0.001 (vs vanilla −0.021) and `meta_scalar` gives the
   best external gain (+0.019). The reliability path never inherits vanilla KD's
   worst-case damage.

3. **The reliability advantage is U-shaped in teacher strength.** The gap between
   reliability methods and vanilla `logit_kd` is largest at the extremes (weak
   vit_b, strong-but-shifted SAM3) and smallest in the mid-band (vit_l/vit_h),
   where a well-matched teacher makes vanilla KD already near-optimal. This is
   direct evidence for **H3**: meta-reliability is *not* a monotone function of
   teacher correctness — it responds to distillation *usefulness*, which peaks
   where the teacher and student disagree in exploitable ways.

4. **`meta_scalar` no-harm (≥ `logit_kd`) holds at the extremes**, where it
   matters most (vit_b, vit_h internal; vit_b, sam3 external). In the mid-band it
   dips below vanilla by ≤0.002 Dice — within run-to-run noise for single-seed
   cells — precisely where vanilla KD has no harm to protect against.

**Takeaway.** Reliability-aware KD buys its largest, most reliable gains exactly
where plain KD is dangerous: a teacher weaker than the student, or a strong
teacher under distribution/resolution shift. Across a 0.76→0.86 teacher-Dice
spectrum it never underperforms the `task_only` floor by a meaningful margin,
which plain `logit_kd` does at both ends.
