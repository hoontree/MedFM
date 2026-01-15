# Knowledge Distillation: SAM → TinyUSFM

이 문서는 finetuned SAM을 teacher로, pretrained TinyUSFM을 student로 하는 knowledge distillation 학습 방법을 설명합니다.

## 개요

- **Teacher**: Finetuned SAM with LoRA
- **Student**: Pretrained TinyUSFM
- **목적**: SAM의 성능을 경량 모델인 TinyUSFM으로 전이

## 파일 구조

```
TinyUSFM/
├── distill.py                  # Knowledge distillation 학습 스크립트
├── config/
│   └── distill.yaml           # Distillation 설정 파일
├── sam_trainer.py             # SAM 학습 스크립트 (참고)
├── train_sam.py               # SAM 학습 실행 파일 (참고)
├── train_tinyusfm.py          # TinyUSFM 학습 스크립트 (참고)
└── model/
    ├── sam_lora_image_encoder.py
    ├── sam_lora_image_encoder_mask_decoder.py
    └── tinyusfm_seg.py
```

## 사전 준비

### 1. Teacher 모델 준비 (Finetuned SAM)

먼저 SAM을 학습시켜 teacher 모델을 준비합니다:

```bash
python train_sam.py
```

학습된 LoRA 체크포인트 경로를 기록해둡니다:
```
logs/vit_b/BUSBRA/pretrain/20241203_123456/checkpoints/best_epoch_10_dice0.8500.pth
```

### 2. Student 모델 준비 (Pretrained TinyUSFM)

TinyUSFM은 사전학습된 체크포인트를 사용하거나, scratch부터 시작할 수 있습니다.
사전학습 체크포인트가 있다면 경로를 기록해둡니다.

## 설정 파일 수정

`config/distill.yaml` 파일을 프로젝트에 맞게 수정합니다:

### 필수 수정 사항

```yaml
# Teacher 모델 설정
teacher:
  model_name: vit_b  # SAM 모델 크기 (vit_b, vit_l, vit_h)
  module: model.sam_lora_image_encoder  # LoRA 모듈
  img_size: 256
  rank: 4
  # SAM ImageNet pretrained weights
  sam_checkpoint: work_dir/SAM/sam_vit_b_01ec64.pth
  # ★ 중요: Finetuned LoRA checkpoint 경로 (필수!)
  lora_checkpoint: logs/vit_b/BUSBRA/pretrain/20241203_123456/checkpoints/best_epoch_10_dice0.8500.pth

# Student 모델 설정
student:
  model_name: TinyUSFM_Seg
  pretrained: true
  # Optional: TinyUSFM pretrained checkpoint
  checkpoint: null  # 또는 pretrained_tinyusfm.pth 경로
```

### Distillation 하이퍼파라미터

```yaml
distillation:
  temperature: 4.0    # 높을수록 softer probability (일반적으로 1-20)
  alpha: 0.5         # Task loss weight (ground truth)
  beta: 0.5          # Distillation loss weight (teacher)
  gamma: 0.0         # Feature distillation weight (선택적)
```

**권장 설정:**
- `alpha + beta = 1.0` 유지 권장
- `temperature`: 4.0-8.0 사이 시작, 실험을 통해 조정
- `gamma > 0`: 중간 feature도 distillation하려면 활성화

### 학습 파라미터

```yaml
training:
  num_epochs: 150
  lr: 0.0001        # Distillation에서는 낮은 learning rate 권장
  warmup_epochs: 5
  batch_size: 16
  num_workers: 8
```

## 실행 방법

### 기본 실행

```bash
python distill.py
```

### Hydra 오버라이드를 통한 실행

```bash
# Teacher checkpoint 지정
python distill.py teacher.lora_checkpoint=path/to/sam_lora.pth

# Distillation 파라미터 조정
python distill.py distillation.temperature=6.0 distillation.alpha=0.3 distillation.beta=0.7

# Learning rate 조정
python distill.py training.lr=0.0005 training.batch_size=32

# GPU 설정
python distill.py hardware.gpu_ids=[0,1]
```

## Knowledge Distillation 상세 설명

### 1. 손실 함수 구성

전체 손실은 다음과 같이 구성됩니다:

```
Total Loss = α × Task Loss + β × Distillation Loss + γ × Feature Loss
```

#### Task Loss (Ground Truth 기반)
- Binary segmentation: BCE Loss + Dice Loss
- Multi-class segmentation: Cross-Entropy Loss + Dice Loss
- Student가 정답 레이블을 직접 학습

#### Distillation Loss (Teacher 기반)
- KL Divergence between teacher and student logits
- Temperature scaling으로 soft probability 생성
- Teacher의 지식(dark knowledge) 전달

#### Feature Loss (선택적)
- Teacher와 student의 중간 feature map 사이의 MSE
- `gamma > 0`일 때만 활성화

### 2. Temperature Scaling

Temperature T로 logits를 나누어 softer probability를 생성:

```python
student_soft = softmax(student_logits / T)
teacher_soft = softmax(teacher_logits / T)
```

- T=1: 일반적인 softmax
- T>1: 더 smooth한 확률 분포 (작은 확률값도 의미있게)
- 높은 T: 클래스 간 관계 정보를 더 많이 전달

### 3. 학습 과정

1. **Teacher Forward**: Frozen teacher로 soft targets 생성
2. **Student Forward**: Student 모델로 예측
3. **Loss 계산**:
   - Task loss: Student vs Ground Truth
   - Distillation loss: Student vs Teacher (soft targets)
4. **Backpropagation**: Student만 업데이트 (teacher는 frozen)

## 출력 구조

학습 결과는 다음과 같이 저장됩니다:

```
logs/distillation/{dataset}/{timestamp}/
├── config.yaml                      # 학습 설정
├── distill.log                      # 학습 로그
├── models/
│   ├── best_epoch10_dice0.8234.pth # Best model
│   └── checkpoint_epoch20.pth       # Periodic checkpoints
├── visualizations/
│   ├── epoch_010.png               # Periodic visualization
│   ├── epoch_020.png
│   └── test_sample_000.png         # Test visualizations
├── tensorboard/                     # TensorBoard logs
├── test_results.txt                # 최종 테스트 결과
└── wandb/                          # Wandb logs
```

## 시각화

학습 중 다음을 시각화합니다:

1. **Input Image**: 원본 입력 이미지
2. **Ground Truth**: 정답 마스크
3. **Teacher Prediction**: SAM의 예측
4. **Student Prediction**: TinyUSFM의 예측
5. **Overlays**: 입력 이미지에 마스크 오버레이
6. **Difference Map**: Teacher-Student 차이

## 성능 모니터링

### TensorBoard

```bash
tensorboard --logdir logs/distillation/{dataset}/{timestamp}/tensorboard
```

### Wandb

- Train loss (total, task, distillation, feature)
- Validation metrics (Dice, HD95, Pixel Accuracy)
- Learning rate
- Visualizations

## 실험 팁

### 1. Temperature 튜닝
- 낮은 temperature (1-4): Hard targets에 가까움
- 중간 temperature (4-8): 일반적으로 좋은 성능
- 높은 temperature (8-20): 더 soft, 작은 데이터셋에 유리

### 2. Alpha/Beta 비율
- `alpha=0.7, beta=0.3`: Task loss 중심 (정확도 우선)
- `alpha=0.5, beta=0.5`: 균형잡힌 학습
- `alpha=0.3, beta=0.7`: Distillation 중심 (teacher 지식 우선)

### 3. Learning Rate
- Distillation에서는 일반적으로 낮은 LR 사용
- Pretrained student: 0.0001 - 0.0005
- Scratch student: 0.001 - 0.005

### 4. Feature Distillation
- `gamma=0.1-0.3`: 중간 feature도 distillation
- Feature dimension 맞추기 필요할 수 있음

## 문제 해결

### OOM (Out of Memory)
```yaml
training:
  batch_size: 8  # Batch size 줄이기
hardware:
  gpu_ids: [0]   # 단일 GPU 사용
```

### Teacher와 Student 해상도 불일치
코드는 자동으로 interpolation으로 해결합니다.

### Teacher checkpoint 로딩 실패
- `teacher.lora_checkpoint` 경로 확인
- LoRA 파라미터만 저장되어 있는지 확인
- `teacher.module` 설정이 올바른지 확인

## 참고 자료

- Knowledge Distillation 원리: Hinton et al., "Distilling the Knowledge in a Neural Network"
- SAM 논문: Kirillov et al., "Segment Anything"
- LoRA: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"

## 문의

문제가 발생하거나 질문이 있으면 이슈를 등록해주세요.
