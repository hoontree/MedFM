# TinyUSFM 코드 실행 구조 분석

이 문서는 프로젝트의 전체 실행 흐름을 정리하고, 발견된 코드 이슈를 기록합니다.

---

## 목차

1. [진입점 (Entry Points)](#1-진입점-entry-points)
2. [설정 시스템 (Configuration System)](#2-설정-시스템-configuration-system)
3. [Teacher 학습 흐름 (train.py)](#3-teacher-학습-흐름-trainpy)
4. [Knowledge Distillation 흐름 (distill.py)](#4-knowledge-distillation-흐름-distillpy)
5. [Pipeline 모드 (Teacher → Distillation)](#5-pipeline-모드-teacher--distillation)
6. [모델별 Trainer 구조](#6-모델별-trainer-구조)
7. [데이터 로딩 흐름](#7-데이터-로딩-흐름)
8. [발견된 코드 이슈](#8-발견된-코드-이슈)

---

## 1. 진입점 (Entry Points)

프로젝트에는 3개의 주요 진입점이 있습니다:

| 진입점 | 설명 | Hydra config_name |
|--------|------|-------------------|
| `train.py` | Teacher 모델 학습 (+ 선택적 파이프라인 증류) | `config/train.yaml` |
| `distill.py` | Knowledge Distillation 전용 | `config/distill.yaml` |
| `eval.py` | 평가 전용 (체크포인트 로드 후 추론) | `config/train.yaml` |

---

## 2. 설정 시스템 (Configuration System)

### 2.1 Hydra 설정 계층 구조

```
config/
├── train.yaml              ← train.py 메인 설정
├── distill.yaml             ← distill.py 메인 설정
├── model/                   ← 모델별 설정 (train.py에서 사용)
│   ├── sam.yaml             → trainer: SAMTrainer, training: sam_training
│   ├── tinyusfm.yaml        → trainer: TinyUSFMTrainer, training: tinyusfm (⚠️ 이슈)
│   ├── segformer.yaml       → trainer: SegformerTrainer
│   ├── sam3.yaml            → trainer: SAM3TrainerAdapter
│   ├── usfm.yaml            → tinyusfm 상속
│   ├── ca_sam.yaml           → CA-SAM 설정
│   └── encoder/
│       ├── vit_b.yaml
│       ├── vit_l.yaml
│       └── vit_h.yaml
├── data/                    ← 데이터셋 설정
│   ├── dynamic.yaml         ← 기본 동적 데이터 설정
│   ├── B.yaml, BUSI.yaml, BUSBRA.yaml, BUID.yaml, ...
│   └── BUSBRA_SegFormer.yaml
├── training/                ← 학습 하이퍼파라미터
│   ├── sam_training.yaml
│   ├── tinyusfm_training.yaml
│   └── distillation.yaml
├── teacher/                 ← 증류용 Teacher 설정
│   └── sam.yaml
├── student/                 ← 증류용 Student 설정
│   ├── tinyusfm.yaml
│   └── usfm.yaml
├── method/                  ← 증류 방법 설정
│   └── unified.yaml
└── log/
    └── wandb.yaml
```

### 2.2 train.yaml 구성

```yaml
defaults:
  - _self_
  - data: dynamic          # config/data/dynamic.yaml
  - model: sam             # config/model/sam.yaml (기본 모델)

pipeline:
  enabled: false            # True면 Teacher → Distill 파이프라인 실행
  distill:
    enabled: true
```

`model: sam`을 선택하면 `config/model/sam.yaml`이 로드되고, 이 안에서:
```yaml
# @package _global_
defaults:
  - _self_
  - /training: sam_training    # config/training/sam_training.yaml 추가 로드

trainer:
  _target_: trainers.sam_trainer.SAMTrainer    # Hydra instantiate 대상
```

### 2.3 distill.yaml 구성

```yaml
defaults:
  - data: dynamic
  - teacher: sam              # config/teacher/sam.yaml
  - student: tinyusfm         # config/student/tinyusfm.yaml
  - method: unified           # config/method/unified.yaml
  - training: distillation    # config/training/distillation.yaml
  - _self_

trainer:
  _target_: trainers.distill_trainer.DistillTrainer
```

### 2.4 설정 합성 순서

**train.py의 경우:**
1. `config/train.yaml` 로드
2. `defaults` 순서대로 합성:
   - `data: dynamic` → `config/data/dynamic.yaml`
   - `model: sam` → `config/model/sam.yaml`
     - sam.yaml 내부의 `/training: sam_training` → `config/training/sam_training.yaml`
3. CLI 오버라이드 적용 (`model.encoder_mode=frozen` 등)

**distill.py의 경우:**
1. `config/distill.yaml` 로드
2. `defaults` 순서대로 합성:
   - `data: dynamic` → `config/data/dynamic.yaml`
   - `teacher: sam` → `config/teacher/sam.yaml`
   - `student: tinyusfm` → `config/student/tinyusfm.yaml`
   - `method: unified` → `config/method/unified.yaml`
   - `training: distillation` → `config/training/distillation.yaml`
3. CLI 오버라이드 적용

---

## 3. Teacher 학습 흐름 (train.py)

### 3.1 실행 흐름 다이어그램

```
python train.py model=sam
│
├─ @hydra.main(config_name="train")
│   └─ main(cfg)
│       │
│       ├─ set_gpu(cfg)                          # CUDA_VISIBLE_DEVICES 설정
│       ├─ suppress_teacher_wandb_in_sweep(cfg)   # Sweep 시 Teacher WandB 비활성화
│       │
│       ├─ instantiate(cfg.trainer, cfg)          # Hydra가 SAMTrainer(cfg) 생성
│       │   └─ SAMTrainer.__init__(cfg)
│       │       └─ BaseTrainer.__init__(cfg)      # 기본 속성 초기화
│       │
│       ├─ trainer.setup(mode="train")
│       │   ├─ _set_seed()                        # 시드 설정
│       │   ├─ _setup_directories(mode)           # 로그/체크포인트 디렉토리 생성
│       │   ├─ _setup_logger()                    # 로거 설정
│       │   ├─ _setup_wandb()                     # WandB 초기화
│       │   ├─ _create_dataloaders()              # 데이터로더 생성 [모델별 구현]
│       │   ├─ _create_model()                    # 모델 생성 [모델별 구현]
│       │   ├─ _create_optimizer()                # 옵티마이저 생성 [모델별 구현]
│       │   ├─ _create_scheduler()                # 스케줄러 생성 [모델별 구현]
│       │   └─ _setup_early_stopping()            # Early Stopping 설정
│       │
│       └─ trainer.train()                        # BaseTrainer.train()
│           │
│           └─ for epoch in range(num_epochs):
│               ├─ train_epoch(epoch)             # [모델별 구현]
│               │   └─ forward → loss → backward → optimizer.step → scheduler.step
│               │
│               ├─ validate(epoch)                # [모델별 구현]
│               │   └─ evaluator.evaluate_model_sam()
│               │
│               ├─ _visualize_validation(epoch)   # 5 epoch마다
│               ├─ _log_metrics(epoch, ...)       # WandB + 콘솔 로깅
│               ├─ _save_checkpoint(epoch, ...)   # Best/Periodic 저장
│               └─ early_stopping 체크
│
│           ├─ _load_checkpoint(best_model_path)  # Best 모델 로드
│           ├─ test()                             # [모델별 구현] 테스트
│           ├─ _save_test_results()               # final_test/ 메트릭 저장
│           └─ wandb.finish()
```

### 3.2 SAMTrainer 구체적 흐름

1. **모델 생성** (`_create_model`):
   - `instantiate(cfg.model)` → `LoRA_Sam(...)` 생성
   - encoder_mode/decoder_mode에 따라 LoRA, Conv-LoRA, Frozen, Full FT 선택
   - SAM 사전훈련 가중치 자동 로드 (sam_type에 따라 `checkpoints/sam_vit_b_*.pth` 등)

2. **데이터 로더** (`_create_dataloaders`):
   - `SegDatasetProcessor.build_data_loaders(cfg)` 호출
   - 반환: `(train_loader, val_loader, test_loaders_dict)`
   - 각 배치는 4-tuple: `(image, label, low_res_label, filename)`

3. **학습 루프** (`train_epoch`):
   - 각 배치: `image_batch, label_batch, low_res_label_batch, *_`
   - Forward: `model(image_batch, False, img_size)` → `outputs` dict
   - Loss: BCE + Dice (가중 합산) + 선택적 MoE Loss
   - Gradient Clipping → Optimizer Step → Scheduler Step (iter 기반)

4. **검증** (`validate`):
   - `evaluator.evaluate_model_sam(model, val_loader, ...)` → Dice, HD95, IoU 등

5. **테스트** (`test`):
   - 각 test 데이터셋에 대해 개별 평가
   - 시각화 생성 (predictions + ground truth 비교)

---

## 4. Knowledge Distillation 흐름 (distill.py)

### 4.1 실행 흐름 다이어그램

```
python distill.py
│
├─ @hydra.main(config_name="distill")
│   └─ main(cfg)
│       │
│       ├─ set_gpu(cfg)
│       │
│       ├─ instantiate(cfg.trainer, cfg)
│       │   └─ DistillTrainer.__init__(cfg)
│       │       ├─ set_seed()
│       │       ├─ create_log_dir()               # 로그 디렉토리 생성
│       │       ├─ save_experiment_summary()
│       │       ├─ setup_logger()
│       │       ├─ wandb.init()
│       │       │
│       │       ├─ _setup_data()
│       │       │   └─ SegDatasetProcessor.build_data_loaders(cfg)
│       │       │
│       │       ├─ _setup_models()
│       │       │   ├─ instantiate(cfg.teacher)    # SAM Teacher 생성 + 로드
│       │       │   │   → teacher.eval(), requires_grad=False
│       │       │   ├─ instantiate(cfg.student)    # TinyUSFM Student 생성
│       │       │   └─ create_distiller(cfg)       # UnifiedDistiller 생성
│       │       │       └─ distiller.prepare(student, teacher)
│       │       │           → Feature Hook 등록, Adapter 초기화
│       │       │
│       │       └─ _setup_optimizer()
│       │           ├─ AdamW(student.params + distiller.params)
│       │           └─ build_scheduler(optimizer, cfg)
│       │               → CosineAnnealingWarmupLR 또는 Hydra instantiate
│       │
│       └─ trainer.train()
│           │
│           └─ for epoch in range(num_epochs):
│               ├─ train_epoch(epoch)
│               │   └─ for batch in train_loader:
│               │       ├─ Teacher Forward (no_grad):
│               │       │   teacher(images, False, teacher.img_size)
│               │       │   → {"masks", "low_res_logits", "image_embeddings", ...}
│               │       │
│               │       ├─ Student Forward:
│               │       │   student(images, return_features=True)
│               │       │   → (seg_logits, features)
│               │       │   → {"masks": seg_logits, "features": features}
│               │       │
│               │       ├─ Distiller Loss:
│               │       │   distiller(student_outputs, teacher_outputs, masks)
│               │       │   → {"loss", "task_loss", "distill_loss", ...}
│               │       │
│               │       └─ Backward → Clip Grad → Optimizer Step
│               │
│               ├─ validate(epoch)
│               │   └─ evaluator.evaluate_model(student, val_loader, ...)
│               │
│               ├─ visualize_distillation()        # 5 epoch마다
│               ├─ _save_checkpoint(epoch, val_dice)
│               ├─ early_stopping 체크
│               └─ scheduler.step()                # epoch 기반
│
│           └─ _final_evaluation()
│               ├─ 메모리 해제 (optimizer, distiller 삭제)
│               ├─ Best 모델 로드
│               ├─ test(phase="final_test")
│               ├─ WandB summary 업데이트
│               └─ 최종 시각화
```

### 4.2 UnifiedDistiller 손실 함수 구성

```
Total Loss = α × Task Loss
           + β × Distill Loss (KL Divergence)
           + γ × Feature Loss (MSE, 다중 레이어)
           + γ_attn × Attention Map Loss (MSE)
           + γ_align × Alignment Layer Loss (MSE)
           + λ_boundary × Boundary KD (Sobel edge map)
           + λ_shape × Shape KD (Signed Distance Transform)
           + λ_uncertainty × Uncertainty KD (Entropy map)
```

각 계수를 0으로 설정하면 해당 손실 비활성화. GradNorm으로 자동 가중치 조정 가능.

---

## 5. Pipeline 모드 (Teacher → Distillation)

```
python train.py pipeline.enabled=true model=sam
```

### 5.1 실행 흐름

```
main(cfg)
│
├─ Stage 1: Teacher Training
│   ├─ suppress_teacher_wandb_in_sweep()   # Sweep 시 Teacher WandB off
│   ├─ trainer = instantiate(cfg.trainer, cfg)
│   ├─ trainer.setup(mode="train")
│   └─ trainer.train()                     # 정상 학습 (wandb.finish() 스킵)
│
├─ _run_distillation(cfg, trainer)
│   ├─ teacher_ckpt = trainer.best_model_path
│   ├─ release_trainer(trainer)             # GPU 메모리 해제
│   │   → model/optimizer/scheduler → cpu → None → gc + cuda.empty_cache
│   │
│   ├─ build_distill_cfg(cfg, teacher_ckpt, ...)
│   │   ├─ compose(config_name="distill", overrides=[f"teacher={name}"])
│   │   ├─ distill_cfg.teacher = merge(distill teacher, train model)
│   │   ├─ teacher.checkpoint = best_ckpt_path
│   │   ├─ distill_cfg.data = train_cfg.data     # 동일 데이터 공유
│   │   └─ distill_cfg.hardware = merged hardware
│   │
│   ├─ instantiate(distill_cfg.trainer, distill_cfg)
│   └─ distill_trainer.train()
│
└─ Pipeline Summary (최종 메트릭 출력)
```

### 5.2 WandB 연동

- Pipeline 모드에서 Teacher WandB를 비활성화
- Distillation은 Teacher의 WandB run을 재사용 (같은 run에 `distill/` prefix로 기록)
- Sweep 시 최종 메트릭은 `final_test/BUID/dice` 등으로 기록

---

## 6. 모델별 Trainer 구조

### 6.1 클래스 계층

```
BaseTrainer (ABC)            ← trainers/base_trainer.py
├── SAMTrainer               ← trainers/sam_trainer.py
├── TinyUSFMTrainer          ← trainers/tinyusfm_trainer.py
├── SegformerTrainer         ← trainers/segformer_trainer.py
└── SAM3TrainerAdapter       ← trainers/sam3_adapter.py

DistillTrainer               ← trainers/distill_trainer.py (독립 클래스, BaseTrainer 미상속)
```

### 6.2 BaseTrainer 추상 메서드

| 메서드 | 역할 |
|--------|------|
| `_create_model()` | 모델 인스턴스 생성 |
| `_create_dataloaders()` | Train/Val/Test DataLoader 생성 |
| `_create_optimizer()` | 옵티마이저 생성 |
| `_create_scheduler()` | LR 스케줄러 생성 |
| `train_epoch(epoch)` | 1 epoch 학습 로직 |
| `validate(epoch)` | 검증 로직 |
| `test()` | 테스트 로직 |

### 6.3 `setup()` → `train()` 호출 흐름

```python
# train.py에서:
trainer = instantiate(cfg.trainer, cfg)  # __init__ 호출
trainer.setup(mode="train")              # 전체 초기화
trainer.train()                          # 학습 루프 시작
```

`setup()` 내부 호출 순서:
1. `_set_seed()` — 재현성 보장
2. `_setup_directories()` — `logs/{phase}/{model_name}/{timestamp}/` 생성
3. `_setup_logger()` — 파일 + 콘솔 로거
4. `_setup_wandb()` — W&B 초기화
5. `_create_dataloaders()` — **모델별 구현** (보통 `SegDatasetProcessor.build_data_loaders`)
6. `_create_model()` — **모델별 구현** (Hydra instantiate 또는 직접 생성)
7. `_create_optimizer()` — **모델별 구현** (보통 AdamW)
8. `_create_scheduler()` — **모델별 구현** (PolyLR, CosineAnnealing 등)
9. `_setup_early_stopping()` — patience/min_delta 기반

### 6.4 SAMTrainer 스케줄러 vs DistillTrainer 스케줄러

| 항목 | SAMTrainer | DistillTrainer |
|------|-----------|----------------|
| 스케줄러 | LambdaLR (Poly decay, iter 기반) | CosineAnnealingWarmupLR (epoch 기반) |
| step 시점 | 매 iteration | 매 epoch 끝 |
| warmup | iter 기반 linear warmup | epoch 기반 linear warmup |

---

## 7. 데이터 로딩 흐름

### 7.1 데이터셋 빌드

```
SegDatasetProcessor.build_data_loaders(cfg)
│
├─ _sync_img_size_with_sam_type(cfg)       # sam_type에 따라 img_size 자동 조정
│   └─ vit_b→224, vit_l→1024, vit_h→1024
│
├─ build_dataset(cfg)
│   ├─ Train: cfg.data.train 리스트에서 각 데이터셋 로드 → ConcatDataset
│   ├─ Val: cfg.data.val (없으면 train과 동일한 데이터셋의 val split)
│   └─ Test: cfg.data.test 리스트 → {name: dataset} 딕셔너리
│
├─ DataLoader 생성 (train: shuffle=True, val/test: shuffle=False)
└─ 반환: (train_loader, val_loader, {name: test_loader} dict)
```

### 7.2 데이터셋 반환값

대부분의 데이터셋 (`BUSBRA`, `BUSI`, `BUS_UCLM`, `B` 등):
```python
# BaseUltrasoundDataset._create_tensors() → 4-tuple:
(image_tensor, mask_tensor, low_res_mask_tensor, filename)
```

예외 — `UltrasoundSegmentationDataset`:
```python
# 3-tuple (low_res_mask 없음):
(image_tensor, label_tensor, filename)
```

### 7.3 multi-dataset 평가 구조

```
test_loader = {
    "BUID": DataLoader(...),
    "BUID_unfiltered": DataLoader(...),     # filter_empty_masks=True인 경우
    "BUS_UCLM": DataLoader(...),
    "BUS_UCLM_unfiltered": DataLoader(...),
}
```

테스트 시 각 데이터셋에 대해 개별 평가 → 메트릭에 `"BUID/Dice"`, `"BUS_UCLM/Dice"` 형태로 기록.

---

## 8. 발견된 코드 이슈

### 🔴 CRITICAL — 즉시 수정 필요

#### Issue 1: GPU 감지 로직 오류
**파일:** `trainers/sam_trainer.py` Line 53
```python
if len(self.cfg.get("hardware", {"gpu_ids": [0]})) > 1:
    self.model = nn.DataParallel(self.model)
```
**문제:** `len()`이 `hardware` dict의 key 수를 세는 것. `hardware` dict에 `gpu_ids`와 `seed` 2개의 key가 있으면 항상 `True`가 되어 단일 GPU에서도 DataParallel이 래핑됨.

**올바른 코드:**
```python
if len(self.cfg.get("hardware", {}).get("gpu_ids", [0])) > 1:
```

---

#### Issue 2: Hydra config 파일명 불일치
**파일:** `config/model/tinyusfm.yaml` Line 6
```yaml
defaults:
  - /training: tinyusfm
```
**문제:** Hydra가 `config/training/tinyusfm.yaml`을 찾지만, 실제 파일명은 `tinyusfm_training.yaml`. `python train.py model=tinyusfm` 실행 시 Hydra ConfigCompositionException 발생.

**SAM은 정상:** `- /training: sam_training` → `sam_training.yaml` ✅

**수정 방법 둘 중 하나:**
- 파일명을 `tinyusfm.yaml`로 변경, 또는
- config를 `- /training: tinyusfm_training`으로 수정

---

#### Issue 3: Pipeline 모드에서 상대 경로 문제
**파일:** `utils/pipeline.py` Lines 37-39
```python
preset = Path("config/teacher") / f"{teacher_name}.yaml"
if not preset.exists():
    raise FileNotFoundError(f"Teacher preset not found: {preset}")
```
**문제:** Hydra는 실행 시 CWD를 `outputs/YYYY-MM-DD/HH-MM-SS/`로 변경함. 상대 경로 `config/teacher/sam.yaml`은 프로젝트 루트 기준이므로 Hydra 실행 환경에서 파일을 찾지 못함.

**수정 방향:** 절대 경로 사용 또는 `hydra.utils.get_original_cwd()` 활용.

---

### 🟠 MODERATE — 기능에 영향을 줄 수 있음

#### Issue 4: BaseTrainer에서 이전 Best 체크포인트 미삭제
**파일:** `trainers/base_trainer.py` `_save_checkpoint()` (Line ~545)

**문제:** 새로운 Best 모델을 저장할 때 이전 Best 체크포인트 파일을 삭제하지 않아 디스크 공간 낭비.

**비교:** `DistillTrainer._save_checkpoint()`은 이전 best를 정상적으로 삭제:
```python
if self.best_model_path and self.best_model_path.exists():
    self.best_model_path.unlink()
```

---

#### Issue 5: DistillTrainer의 데이터 unpacking 불일치
**파일:** `trainers/distill_trainer.py` Line 222
```python
for i, (images, masks, *_) in enumerate(pbar):
```
**문제:** 대부분의 데이터셋은 4-tuple `(image, mask, low_res_mask, filename)`을 반환. `masks`에 실제로 `mask_tensor`가 할당되고 `low_res_mask`는 `*_`에 들어감. 기능상으로는 작동하지만:
- `masks` 변수에 full-resolution mask가 할당되어 정상 동작함
- 하지만 `low_res_mask`를 아예 사용하지 않는 것이 의도인지 불명확
- `UltrasoundSegmentationDataset`은 3-tuple만 반환하여, 같은 DataLoader에서 혼용 시 문제 가능

---

#### Issue 6: schedule.py fallback에서 config 키 검증 없음
**파일:** `utils/schedule.py` `build_scheduler()` (Line ~100)
```python
# Fallback to legacy behavior
return CosineAnnealingWarmupLR(
    optimizer=optimizer,
    warmup_epochs=cfg.training.warmup_epochs,  # AttributeError 가능
    ...
)
```
**문제:** `cfg.training.warmup_epochs`가 없는 config에서 fallback에 도달하면 `AttributeError` 발생. `sam_training.yaml`에는 `warmup_epochs`가 없으나, 해당 Trainer는 자체 스케줄러를 만들어 이 함수를 호출하지 않으므로 현재는 발생하지 않음. 하지만 새 모델 추가 시 발생 가능.

---

### 🟡 MINOR — 개선 권장

#### Issue 7: DistillTrainer가 BaseTrainer를 상속하지 않음
`DistillTrainer`는 `BaseTrainer`와 독립적으로 구현되어 setup, 로깅, 체크포인팅 등의 로직이 중복됨. `EarlyStopping` 클래스도 `BaseTrainer`에서 import하여 사용.

#### Issue 8: wandb.run 의존성
`BaseTrainer._log_model_info()`, `SAMTrainer._log_model_configuration()` 등에서 `wandb.run is not None` 체크 후 `wandb` 글로벌 변수 사용. WandB가 disabled 모드일 때 `wandb.run`이 `None`이 아닌 Mock 객체일 수 있음 (W&B 버전에 따라 다름).

---

## 부록: 설정 해석 순서 요약

### `python train.py model=sam` 실행 시:

```
1. Hydra 초기화
   → config/train.yaml 로드
   → defaults 처리:
     - data: dynamic → config/data/dynamic.yaml
     - model: sam   → config/model/sam.yaml
       - sam.yaml 내부: /training: sam_training → config/training/sam_training.yaml
   → CLI 오버라이드 적용

2. main(cfg) 호출
   → cfg.trainer._target_ = "trainers.sam_trainer.SAMTrainer"
   → instantiate(cfg.trainer, cfg) → SAMTrainer(cfg)

3. SAMTrainer.__init__(cfg)
   → BaseTrainer.__init__(cfg) 호출
   → SAM 고유 속성 초기화 (img_size, base_lr, warmup 등)

4. trainer.setup(mode="train")
   → 시드, 디렉토리, 로거, WandB 설정
   → SegDatasetProcessor.build_data_loaders(cfg)
     → img_size 자동 조정 (vit_b → 224)
     → dynamic.yaml의 train/test 리스트에서 데이터셋 로드
   → instantiate(cfg.model) → LoRA_Sam(sam_type, encoder_mode, decoder_mode, ...)
   → AdamW 옵티마이저 + LambdaLR 스케줄러

5. trainer.train()
   → BaseTrainer.train() 호출
   → 에폭 루프: train_epoch → validate → checkpoint → early_stopping
   → 학습 완료 후: best 모델 로드 → test → 결과 저장
```

### `python distill.py` 실행 시:

```
1. Hydra 초기화
   → config/distill.yaml 로드
   → defaults 처리:
     - data: dynamic        → config/data/dynamic.yaml
     - teacher: sam         → config/teacher/sam.yaml
     - student: tinyusfm    → config/student/tinyusfm.yaml
     - method: unified      → config/method/unified.yaml
     - training: distillation → config/training/distillation.yaml
   → CLI 오버라이드 적용

2. main(cfg) 호출
   → cfg.trainer._target_ = "trainers.distill_trainer.DistillTrainer"
   → instantiate(cfg.trainer, cfg) → DistillTrainer(cfg)

3. DistillTrainer.__init__(cfg)
   → 전체 초기화 (setup 메서드 분리 없이 __init__에서 모두 처리)
   → SegDatasetProcessor.build_data_loaders(cfg)
   → Teacher: instantiate(cfg.teacher) → LoRA_Sam (frozen)
   → Student: instantiate(cfg.student) → SegmentationModel (TinyUSFM)
   → Distiller: create_distiller(cfg) → UnifiedDistiller
   → AdamW + CosineAnnealingWarmupLR

4. trainer.train()
   → 에폭 루프: train_epoch → validate → checkpoint → early_stopping
   → _final_evaluation(): best 모델 로드 → test → 시각화
   → wandb.finish()
```
