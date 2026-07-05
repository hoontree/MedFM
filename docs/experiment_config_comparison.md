# 실험 설정 비교 — Teacher 학습 & Distillation 학습

신뢰도 기반 KD 연구를 위해 진행된 **teacher/baseline 모델 학습**과 **distillation 학습**
실험의 하이퍼파라미터·핵심 config를 집계·비교한 문서입니다. 결과 분석은
[rel_ko.md](rel_ko.md), 설계는 [reliability_kd_study.md](reliability_kd_study.md) 참고.

출처 config: `config/training/{sam,tinyusfm,distillation}_training.yaml`,
`config/model/{sam,sam_lora,tinyusfm}.yaml`, `config/method/unified.yaml`,
`config/data/dynamic.yaml`, `config/sweeps/*`.

공통 데이터: `dynamic` (num_classes=3, img 224), 학습 BUSBRA/BUSI/B,
외부 테스트 BUID/BUS_UCLM/BUS_UCLM_filtered, seed 42, 1000 epoch cap,
best val Dice로 체크포인트 선택.

---

## 1. Teacher / Baseline 모델 학습 (`train.py`)

balanced sampling on/off만 통제변수로 분리 (`config/sweeps/baseline_finetune.yaml`).

| 항목 | **SAM-FT** (`sam.yaml`) | **SAM-LoRA** (`sam_lora.yaml`) | **TinyUSFM** (`tinyusfm.yaml`) |
|---|---|---|---|
| trainer | SAMTrainer | SAMTrainer | TinyUSFMTrainer |
| encoder/decoder mode | `ft` / `ft` | `lora` / `lora` (r_e=r_d=4) | FPN decoder (`decoder_type=fpn`) |
| backbone | SAM vit_b | SAM vit_b | MAE vit (embed 192) |
| **lr** | 1e-4 | 1e-4 | 1e-4 |
| **batch_size** | 32 | 32 | 16 |
| warmup_epochs | 2 (+ warmup steps 200) | 2 | 5 |
| optimizer | AdamW, wd=0.01 | AdamW, wd=0.01 | AdamW, **wd=0**, betas(0.9,0.999) |
| layer decay | — | — | **0.8 (12 layers)** |
| scheduler | PolyLR power=0.9, min_lr 1e-6 | 동일 | WarmupPolyLR power=0.9, end_lr 1e-6 |
| grad clip max_norm | 1.0 | 1.0 | 1.0 |
| dice_weight | 0.8 | 0.8 | 0.8 |
| early stopping | patience **15**, min_δ 1e-6 | 15 | patience **20**, min_δ 1e-6 |
| wandb project | sam_train | sam_train | TinyUSFM_Seg |

### 학습 결과 (val Dice, rel_ko §6 통일 recipe)

| 모델 | sampling off | sampling on |
|---|---:|---:|
| TinyUSFM | 0.725 (ext best 0.644) | 0.714 (int best 0.765) |
| SAM-FT | 0.674 | 0.682 |
| SAM-LoRA | 0.710 | **0.720 (ext best 0.662)** |

> 옛 FT 교사(val 0.57)는 미튜닝 아티팩트였고, sampler 켠 통일 recipe 재학습으로
> 0.682까지 회복. KD 교사로는 sampling-on SAM-LoRA가 최선.

### balanced sampler 설정 (sampling on일 때)

alpha=0.5 (sqrt tempering), class_weights 1/1/1 균일, normal_cap=null,
`decouple_dataset_class=true` (p_d ∝ N_d^0.5), seed 42.

---

## 2. Distillation 학습 (`distill.py`, `method/unified.yaml`)

모든 distillation은 공통 학습 recipe(`training/distillation.yaml`)를 공유하고,
**loss weight / reliability knob**만 sweep마다 바뀝니다.

### 공통 학습 설정 (전 distillation 동일)

| 항목 | 값 |
|---|---|
| lr (student) | 1e-4 |
| batch_size | 8 |
| num_epochs | 1000 (early stop patience **20**, min_δ 1e-6) |
| warmup_epochs | 5 |
| optimizer | AdamW, wd=0.01, grad clip 1.0 |
| scheduler | **CosineAnnealingWarmupLR**, min_lr 1e-6 |
| temperature (기본) | 4.0 |
| task loss | Dice + CE (w_task=1.0, pos_weight 5 — binary 전용) |

**Reliability map 기본 knob (`unified.yaml`):** confidence=max_prob,
entropy_penalty on, teacher_correctness_gate on (wrong=0.0),
student_bypass on (weight=0.1), smoothing **off**.

### Sweep별 차이 (teacher / 방향 / sampling 구성)

| Sweep (manifest) | Teacher | Student | sampling | 주요 method 변형 |
|---|---|---|---|---|
| `reliability_ablation` (§3,5) | SAM-FT (옛 0.57) | TinyUSFM | off | 20-run factor ablation (T-gate/S-byp/smooth on·off, T=2/6/8, wrong=0/0.25) |
| `reliability_teacher_lora` (§1) | SAM-LoRA | TinyUSFM | off | task_only / logit_kd / uncertainty_kd / **reliability** / +smoothing / T=2,8 / wrong=0.25 / sb=0.3 |
| `reliability_teacher_lora_sampled` (§1.4) | SAM-LoRA | TinyUSFM | **on** (α=0.5) | `smpl_` task_only / logit / reliability / +sm / T / knobs |
| `reliability_teacher_*sampled` (FT) (§1.4) | SAM-FT (재학습 0.682) | TinyUSFM | on | logit_kd / reliability |
| `reliability_teacher_tinyusfm` (§3 3행) | TinyUSFM | SAM | off | task_only / reliability / T=2,8 / knobs (역방향) |

### 방법별 method base_overrides

공통: `w_task=1.0`, logit_cwd/feature_cwd = 0.

| 방법 | 핵심 override |
|---|---|
| task-only | 모든 KD weight = 0 |
| reliability-KD | `w_reliability_kd=1.0`, `w_logit_kd=0` |
| logit-KD | `w_logit_kd=1.0`, `w_reliability_kd=0` (순수 Hinton KL) |
| uncertainty-KD | `w_uncertainty_kd=1.0` + `use_uncertainty_weighted_kd=true` (linear, β=1.0) |

### Hyperparameter 비교 축 (모든 KD sweep 공통 4개)

| Knob | 중심값 | 비교값 |
|---|---|---|
| temperature | 4.0 | 2.0 / 8.0 |
| teacher_correctness_wrong_weight | 0.0 (하드 차단) | 0.25 |
| student_bypass_weight | 0.1 | 0.3 |
| reliability_smoothing | off | on |

---

## 3. 핵심 비교 요점

- **Teacher 학습**: SAM-FT/LoRA는 동일 recipe(bs32, lr1e-4, warmup2),
  TinyUSFM만 bs16·wd0·layer-decay0.8·warmup5로 다름. LoRA만 encoder/decoder를
  rank-4 LoRA로 고정.
- **Distillation 학습**: teacher 학습과 달리 **bs8, warmup5, Cosine 스케줄,
  patience20**. 단일 `unified.yaml`에서 loss weight만 토글해 method를 정의 —
  task/logit/uncertainty/reliability가 전부 같은 학습 recipe 위에서 비교됨.
- **통제 변수**: sweep 간 차이는 (1) teacher 종류(FT/LoRA/TinyUSFM),
  (2) sampling on/off, (3) 4개 reliability knob 뿐. 나머지는 전부 고정.

---

## 4. 공정성 평가 (비교가 통제됐는가?)

### 공정했던 부분

- **Distillation 공통 recipe**: 모든 KD 변형이 동일한 bs8/lr1e-4/warmup5/Cosine/
  patience20 위에서, `unified.yaml`의 loss weight 토글만으로 정의됨 → 학습 조건
  자체는 깨끗하게 통제됨.
- **FT 교사 아티팩트 교정**: 옛 FT 교사(val 0.57)는 학습 부족 아티팩트였고,
  이것으로 한 §3 비교는 불공정. §1.4·§6에서 동일 recipe + sampler로 재학습(0.682)해
  교사 품질을 분리한 건 올바른 보정.
- **LoRA vs FT 분리**: 재학습 교사끼리는 adaptation mode(`lora` vs `ft`)만 다르고
  나머지 동일 → "LoRA 우위 = adaptation 효과" 결론은 공정하게 뒷받침됨.
- **Sampling 통제**: on/off를 seed·recipe 고정하 단일 변수로 분리.

### 불공정/취약했던 부분

1. **방법별 하이퍼파라미터 예산 불균형 (가장 큰 문제).** sweep manifest상
   temperature(2/8)·wrong_weight(0.25)·sb_weight(0.3) 4개 knob 튜닝이
   **reliability-KD에만** 적용됨. `logit_kd`·`uncertainty_kd`는 각각 T=4 단일
   실행뿐. 그런데 대표 실행을 "스위프 최고 BUID Dice"로 선택 → reliability-KD는
   5+개 후보 중 최고, logit-KD는 1개뿐. **multiple-comparisons 우위**가 reliability에
   구조적으로 유리. "reliability ≈/≥ logit" 결론은 reliability 쪽으로 과대평가됐을
   수 있음. logit-KD도 동일하게 T를 sweep해야 공정.
2. **Test set 선택 편향 (selection-on-test).** 대표 실행을 BUID Dice 최고로 선택한
   뒤 그 BUID를 다시 외부 평균에 포함해 보고 → 선택 기준과 평가 지표가 겹쳐 외부
   점수가 낙관적으로 부풀려짐. 선택은 val Dice로, 보고는 test로 분리해야 함.
3. **Teacher 아키텍처 비교는 본질적 비통제.** SAM(bs32, wd0.01, warmup2, PolyLR)
   vs TinyUSFM(bs16, wd0, layer-decay0.8, warmup5)는 recipe가 다름
   (`baseline_finetune.yaml`도 의도적 비통제로 명시). 순수 아키텍처 비교로 인용하면
   불공정.
4. **3-way KD 비교의 방향 혼합.** §3 표는 student가 TinyUSFM/SAM으로 섞여 student
   용량·recipe가 행마다 달라 같은 축 비교가 어려움(저자도 용량 차이를 결정요인으로 인지).

### 종합 판정

| 비교 축 | 공정성 |
|---|---|
| Distillation 학습 recipe | 높음 |
| LoRA vs FT 교사 (재학습) | 높음 |
| Sampling on/off | 높음 |
| **방법 간 (reliability vs logit/uncertainty)** | **낮음** — 튜닝 예산·선택 편향 |
| Teacher 아키텍처 (SAM vs TinyUSFM) | 비통제(의도적) |

**우선 보완 권장:** (1) logit-KD/uncertainty-KD에도 동일한 temperature sweep 추가,
(2) 대표 실행 선택을 BUID→**val Dice** 기준으로 변경 후 외부 점수 재집계.
이 둘을 고치면 핵심 주장("reliability-KD가 약한 교사 영역에서 유리")의 신뢰도가 크게 향상.
