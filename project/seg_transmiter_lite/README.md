# Seg-TransMiter-lite — 실험 정리

작성일: 2026-05-15
위치: `project/seg_transmiter_lite/`

## 1. 한 줄 요약

**Frozen TinyUSFM + 작은 residual adapter**를 학습하되, SAM을 **box-prompt teacher prior**로만 활용하여 3-class breast ultrasound segmentation
(`0=normal, 1=benign, 2=malignant`) 을 푼다.
최종 deploy 시점에는 TinyUSFM + adapter만 필요하고, **SAM은 inference에 들어가지 않는다.**

## 2. 기존 distillation 실험과의 차이

| 항목 | 기존 `UnifiedDistiller` (`TinyUSFM → SAM`) | Seg-TransMiter-lite |
|---|---|---|
| 학습 대상 | SAM (LoRA / Conv-LoRA, fine-tune) | 작은 adapter (수K~수만 params) |
| Frozen 모델 | TinyUSFM (teacher) | TinyUSFM과 SAM 둘 다 (default) |
| Teacher 신호 | Dense logit/feature KD | Box-prompted SAM mask + score + embedding (cache) |
| Deploy 모델 | SAM + LoRA | TinyUSFM + adapter |
| Inference에 SAM 필요? | 예 (SAM이 student) | **아니오** |
| Task | binary or multiclass | **3-class semantic** |

기존 distillation은 SAM을 학습 가능한 student로 취급하는 dense KD 방식이고, 이 실험은 SAM의 **prior knowledge만 뽑아 caching**해서 TinyUSFM 위에 얹는 방식이다.

## 3. 양방향 지원 (`direction` 플래그)

같은 학습 루프가 두 방향을 모두 지원한다.

| direction | Frozen base | 학습 대상 | Deploy 모델 |
|---|---|---|---|
| `tiny_student` (default) | TinyUSFM | adapter + projector | TinyUSFM + adapter |
| `sam_student` | SAM (LoRA optional) | adapter + projector (+ LoRA) | SAM + adapter |

`sam_student`는 SAM 쪽이 ultrasound 도메인에 적응이 약한 가정에서, **TinyUSFM이 캐싱한 SAM 출력**으로 SAM 위의 adapter를 학습한다.
`train_sam_us_adapter.py` 가 `direction=sam_student`을 강제하는 thin shim이다.

SAM 학습/추론은 **항상 prompt-free**(`prompt_encoder(None, None, None)`). Box prompt는 오직 cache 생성 단계에만 사용한다.

## 4. 데이터셋과 라벨

- 데이터: `BUS_UCLM` (학습), `BUS_UCLM_filtered` (평가)
  → 빨강=malignant, 초록=benign 으로 RGB mask가 인코딩된 데이터셋
- 다른 데이터셋(BUSI, BUSBRA, B)은 lesion class를 구분하지 않는 binary mask라 3-class 학습에 직접 쓰기 어렵다.
- `data.num_classes=3`. mask는 one-hot `[B, 3, H, W]` float로 들어온다.

## 5. 손실 구성

총 손실:
```
L_total = w_sup * L_sup + w_obj * L_obj + w_boundary * L_b + w_feat * L_feat
```

| 구성 | 정의 | 기본 weight | Gate |
|---|---|---|---|
| `L_sup` | DiceCE(final_logits, GT one-hot) | 1.0 | always on |
| `L_obj` | BCE+Dice(lesion_prob, SAM_mask) | 0.5 | `sam_score >= 0.75` & `has_lesion` |
| `L_b` | Sobel-edge MSE(lesion_prob, SAM_mask) | 0.2 | 위와 동일 |
| `L_feat` | Cosine align(projector(feat), SAM_emb) | 0.1 | `has_lesion` |

여기서 `lesion_prob = softmax(final_logits)[:, 1] + softmax(final_logits)[:, 2]` (benign+malignant 합).

**핵심 안전장치 — confidence gate**: SAM의 predicted IoU(`sam_score`)가 threshold 미만이면 그 샘플의 `L_obj/L_b`를 0으로 만들고, batch 평균은 통과한 샘플 수로만 나눈다(`gated_mean`). 노이즈가 큰 SAM 출력이 학습을 오염시키지 않는다.

`final_logits = base_logits + adapter(feat)`. Adapter의 마지막 1x1 conv는 **zero-init**이라 학습 0 step에서는 final == base가 보장된다.

## 6. 모델 아키텍처 요약

### 6.1 ResidualConvAdapter (`model/seg_transmiter/adapters.py`)
```
[B, C_in, h, w]
 → 1x1 Conv (C_in → r)
 → GELU
 → DW 3x3 Conv (r → r)
 → GELU
 → 1x1 Conv (r → n_cls)   # zero-init
 → bilinear upsample to (H, W)
 = residual class logits [B, n_cls, H, W]
```
기본 `r = max(C_in // 2, 16)`. 채널 mixing은 1x1로, spatial은 depthwise 3x3로 분리해서 가볍게.

`tiny_student`일 때 `C_in=48` (FPN neck @ scale=1.0), `n_cls=3` → **약 1.5K params**.
`sam_student`일 때 `C_in=256` → 약 35K params.

### 6.2 FeatureProjector
```
[B, C_src, h_src, w_src]
 → 1x1 Conv (C_src → C_dst)
 → bilinear resize to (h_dst, w_dst)
 → GroupNorm + GELU
 = [B, C_dst, h_dst, w_dst]
```

- `tiny_student`: TinyUSFM neck (48ch, 14×14) → SAM embedding (256ch, 14×14)
- `sam_student`: SAM embedding (256ch, 14×14) → TinyUSFM neck shape (48ch, 14×14)

### 6.3 차원이 맞아 떨어지는 이유
- TinyUSFM `out_indices=(3,5,7,11)`, ViT 224/16=14 → neck scale=1.0 feature는 `[B, 48, 14, 14]`.
- SAM vit_b을 `img_size=224`로 빌드하면 image embedding도 `[B, 256, 14, 14]`.
- 두 공간 모두 14×14 token grid라서 spatial은 resize 없이도 그대로 맞는다.

## 7. SAM Teacher Cache

`sam_teacher_cache.py`가 train/val 전체에 대해 한 번만 돌면 된다.

흐름:
1. Dataset에서 `(image, multiclass GT, low_res, filename)` 로드.
2. GT를 binary lesion mask로 collapse: `lesion = class1 ∪ class2`.
3. lesion이 비어있지 않으면 padded tightest bbox 생성.
4. SAM 본체에 박스 prompt로 forward → `(mask_prob, iou_score, image_embedding)`.
5. `cache_dir/<split>/<filename>.pt`로 저장.
   - lesion이 없으면 `has_lesion=False`인 빈 record만 저장 (정상 샘플도 supervised loss에 사용됨).

저장 필드:
```python
{
    "image_id":  str,
    "sam_mask":  [1, H, W] float,   # threshold > 0.5
    "sam_prob":  [1, H, W] float,
    "sam_score": scalar (predicted IoU),
    "sam_image_embedding": [C_sam, h, w],
    "has_lesion": bool,
    "bbox": [4],
}
```

캐시를 분리해 둔 이유:
- 학습 속도가 빨라진다 (SAM forward를 epoch마다 하지 않음).
- `train_sam_us_adapter.py`로 SAM을 한번 더 도메인 adaptation 한 뒤 **더 좋은 cache로 재생성**할 수 있다 (`adapter_sam2` variant).

## 8. 평가 variant

`eval.py --variant {base, adapter_sup, adapter_sam, adapter_sam2}`

| Variant | base model | adapter checkpoint | 의도 |
|---|---|---|---|
| `base` | frozen TinyUSFM | (없음, zero-init residual) | TinyUSFM 단독 성능 (lower bound) |
| `adapter_sup` | frozen TinyUSFM | `loss.obj=0 loss.boundary=0 loss.feat=0`으로 학습한 adapter | GT만 가지고 adapter가 얼마나 보태주는지 |
| `adapter_sam` | frozen TinyUSFM | 기본 손실 전부 켜고 학습 | SAM prior 기여도 |
| `adapter_sam2` | frozen TinyUSFM | adapted SAM으로 재캐싱 후 학습 | 2-stage prior 개선 효과 |

Metric은 `Evaluator_seg` 그대로 사용 — class별 Dice / IoU / HD95, foreground(class 1, 2) mean.

## 9. 디렉토리 구조

재사용 가능한 building block은 기존 `model/`, `utils/` 아래에 두고, 학습/캐시 entry point는 `project/seg_transmiter_lite/`에 모아 둔다.

```
medfm/
├── config/
│   └── seg_transmiter_lite.yaml         # plain OmegaConf (no Hydra defaults)
├── model/
│   └── seg_transmiter/
│       ├── __init__.py
│       └── adapters.py                  # ResidualConvAdapter, FeatureProjector
├── utils/
│   └── seg_transmiter_losses.py         # dice_ce, bce_dice, boundary, feature_align, confidence_gate
└── project/
    └── seg_transmiter_lite/
        ├── __init__.py
        ├── README.md                    # ← 이 문서
        ├── sam_cache_dataset.py         # base dataset + SAM cache 페어링
        ├── sam_teacher_cache.py         # SAM teacher cache builder (CLI)
        ├── train_tiny_with_sam_prior.py # 메인 학습 (양방향 지원)
        ├── train_sam_us_adapter.py      # sam_student direction shim
        └── eval.py                      # variant별 평가 (CLI)
```

## 10. 실행 순서

### 0) 환경
프로젝트 표준대로 `uv` 사용.

### 1) SAM teacher cache 생성 (한 번만)
```bash
uv run python -m project.seg_transmiter_lite.sam_teacher_cache \
    --config config/seg_transmiter_lite.yaml \
    --cache-dir cache/seg_transmiter_lite/sam_teacher
```
결과: `cache/seg_transmiter_lite/sam_teacher/{train,val}/<image_id>.pt`

### 2) Adapter 학습 (tiny_student, default)
```bash
uv run python -m project.seg_transmiter_lite.train_tiny_with_sam_prior \
    --config config/seg_transmiter_lite.yaml \
    --output-dir logs/seg_transmiter_lite/tiny_student/run_01
```
`best.pth`에는 **adapter + projector state만** 저장한다 (~수십 KB). base 모델은 별도 보관.

### 3) 평가
```bash
uv run python -m project.seg_transmiter_lite.eval \
    --config config/seg_transmiter_lite.yaml \
    --variant adapter_sam \
    --checkpoint logs/seg_transmiter_lite/tiny_student/run_01/best.pth \
    --visualize
```

### 4) (선택) Reverse 단계로 SAM 적응 후 cache 재생성
```bash
# 4a. SAM 위에 adapter 학습 (LoRA까지 켜고 싶으면 --train-lora)
uv run python -m project.seg_transmiter_lite.train_sam_us_adapter \
    --train-lora \
    --output-dir logs/seg_transmiter_lite/sam_student/run_01

# 4b. 적응된 SAM으로 cache 재생성
uv run python -m project.seg_transmiter_lite.sam_teacher_cache \
    --cache-dir cache/seg_transmiter_lite/sam_teacher_v2 \
    --sam-checkpoint logs/seg_transmiter_lite/sam_student/run_01/best.pth \
    --overwrite

# 4c. 새 cache로 tiny adapter 재학습 (adapter_sam2)
uv run python -m project.seg_transmiter_lite.train_tiny_with_sam_prior \
    sam_teacher.cache_dir=cache/seg_transmiter_lite/sam_teacher_v2 \
    --output-dir logs/seg_transmiter_lite/tiny_student/run_02
```

## 11. 기존 코드 재사용 표

| 컴포넌트 | 재사용 | 비고 |
|---|---|---|
| Data loader | `utils.data_processing_seg.SegDatasetProcessor.build_dataset` | `multiclass=true, num_classes=3`로 옵션만 변경 |
| Multiclass dataset | `BUS_UCLM`, `BUS_UCLM_filtered` | 라벨이 이미 3-class 인코딩 |
| Supervised loss | `utils.criterion.TaskLoss` 호환 (래퍼 `supervised_task_loss`) | dice_ce_loss는 동일 결과를 명시적으로 분해해 weight 노출 |
| Metric | `utils.evaluate.Evaluator_seg.evaluate_batch` | multiclass branch 자동 디스패치 |
| TinyUSFM base | `model.tinyusfm_seg.SegmentationModel(..., return_features=True)` | wrapper 클래스 불필요, return_features API 그대로 사용 |
| SAM base | `model.sam_hybrid_adapter.LoRA_Sam` | prompt-free forward 그대로 사용 |
| SAM 박스 prompt | `Sam.prompt_encoder` 직접 호출 (cache 단계만) | 학습/추론 path는 box 사용 안 함 |
| Visualization | `utils.visualize.visualize_segmentation` | batch-list 입력 형식 그대로 |
| Conv-LoRA | `model.adaptation_layers.ConvLoRALinear` re-export (`ConvLoRAAdapterStub`) | SAM 인코더 q/v 주입 시 활용 |

## 12. 가정과 한계

- **3-class 학습 데이터의 절대량이 작다.** BUS_UCLM의 benign/malignant 라벨이 있는 샘플 수가 한정적이라, supervised baseline 자체가 noisy할 수 있다. 따라서 adapter는 일부러 작게(<= 수만 params) 잡았다.
- **lesion 합치기**: cache에서 benign과 malignant를 합쳐 단일 binary box를 만든다. 한 영상에 두 종류 병변이 공존하는 케이스는 드물어 단순화했다.
- **SAM 입력 정규화 불일치**: dataset은 ImageNet 0-1 정규화를 적용하고, SAM의 `preprocess`는 0-255 mean/std를 기대한다. 이는 프로젝트 전체에서 받아들인 trade-off로, cache 단계도 동일하게 처리했다 (정확도가 약간 손해일 수는 있어도 학습 코드와 캐시 코드 간 일관성이 더 중요).
- **Hydra 미사용**: 이 prototype의 config는 plain `OmegaConf.load`로 읽는다. `defaults:` block과 `${now:...}` resolver는 의도적으로 사용하지 않는다 (`output_dir`은 CLI fallback에서 `int(time.time())`으로 채움).

## 13. 다음 ablation 후보

연구적으로 다음 비교가 의미 있을 것:

1. **Adapter capacity sweep**: bottleneck `{16, 32, 64}`.
2. **Confidence threshold sweep**: `sam_threshold ∈ {0.5, 0.65, 0.75, 0.85}`.
3. **Boundary loss 유무**: `w_boundary=0` vs 0.2.
4. **Feature alignment 유무**: `w_feat=0` vs 0.1.
5. **Adapted SAM의 prior 품질 검증**: `adapter_sam` vs `adapter_sam2`.
6. **TinyUSFM base 강도**: pretrained-only vs binary fine-tuned checkpoint.

각 결과는 `eval.py`의 `metrics.json`로 떨어지므로 batch sweep이 용이하다.
