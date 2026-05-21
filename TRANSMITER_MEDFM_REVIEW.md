# TransMiter 방법론의 TinyUSFM -> SAM 적용 검토

작성일: 2026-05-06

## 요약

현재 프로젝트의 방향은 기존 SAM -> TinyUSFM distillation이 아니라, 실험상 더 강한 TinyUSFM을 teacher로 두고 SAM을 student로 학습하는 TinyUSFM -> SAM distillation이다. TransMiter의 핵심 아이디어는 이 방향과 일부 맞지만, 원본 방법을 그대로 옮기는 것은 권장하기 어렵다.

결론적으로는 다음 순서가 가장 타당하다.

1. 현재 코드에 이미 구현된 TinyUSFM -> SAM dense distillation을 먼저 사용한다.
2. TransMiter의 logit proxy transfer는 2차 ablation으로만 검토한다.
3. TransMiter의 Procrustes alignment 아이디어는 logit보다 feature/token 공간에 적용하는 편이 더 적합하다.

## TransMiter 핵심 아이디어

TransMiter는 단순한 teacher-student KD가 아니라, fine-tuned source model의 adaptation knowledge를 별도 proxy model에 추출한 뒤, target model의 표현 공간에 맞춰 전이하는 구조다.

원본 구현 기준으로는 다음 단계로 구성된다.

- Knowledge extraction: source zero-shot logits를 입력으로 받아 source fine-tuned logits를 모사하는 `ProxyModel`을 KL loss로 학습한다.
- Knowledge transfer: 학습된 source proxy를 target proxy로 복사한다.
- Space alignment: source model과 target model의 latent/logit 표현을 모아 Orthogonal Procrustes로 basis 차이를 보정한다.
- Evaluation: target model 출력 위에 proxy 변환을 얹어 source의 adaptation 효과를 target에도 재사용한다.

즉, TransMiter의 본질은 "큰 teacher가 작은 student를 직접 가르친다"가 아니라, "한 모델에서 얻은 fine-tuning 변화량 또는 logit-space 변환을 다른 모델 공간으로 옮긴다"에 가깝다.

## 현재 프로젝트와 맞는 점

### 1. Teacher가 더 작고 더 강한 상황과 잘 맞는다

현재 프로젝트에서는 TinyUSFM이 SAM보다 lightweight이면서 segmentation 성능이 더 높다. TransMiter도 "강한 모델 -> 약한 모델"의 일반 KD보다는, source에서 얻은 adaptation knowledge를 target으로 옮기는 문제를 다룬다. 따라서 TinyUSFM이 SAM보다 작다는 사실은 TransMiter식 접근을 막는 요인이 아니다.

오히려 다음 가설을 세울 수 있다.

- TinyUSFM은 ultrasound segmentation domain에 더 잘 적응한 source model이다.
- SAM은 범용 vision prior는 강하지만 ultrasound segmentation target task에서는 충분히 적응되지 않았다.
- TinyUSFM의 task/domain adaptation signal을 SAM의 LoRA 또는 decoder adaptation으로 옮길 수 있다.

### 2. 현재 코드에 dense distillation 기반이 이미 있다

`distillers/unified_distiller.py`에는 다음 손실들이 이미 구현되어 있다.

- supervised task loss: BCE/CE + Dice
- logit KD: teacher/student mask logits 간 KL
- feature KD: intermediate feature MSE
- CWD: channel-wise distillation for dense prediction
- uncertainty-weighted KD

특히 CWD는 dense segmentation에 잘 맞는다. TransMiter의 원본 proxy가 classification logit vector에 초점을 둔다면, 현재 프로젝트의 CWD는 pixel/channel/spatial saliency를 직접 맞춘다. segmentation task에서는 이쪽이 더 직접적인 신호다.

### 3. SAM과 TinyUSFM 모두 feature alignment 지점을 제공한다

TinyUSFM은 ViT backbone block feature와 neck feature를 갖고 있고, `return_features=True`일 때 segmentation logits와 중간 feature를 반환할 수 있다. SAM adapter도 image encoder output, low-res logits, mask output을 반환할 수 있다.

따라서 다음 방식의 feature-level transfer가 가능하다.

- TinyUSFM backbone block token -> SAM image encoder block token
- TinyUSFM neck feature -> SAM image embedding
- TinyUSFM segmentation saliency map -> SAM low-res/final mask logits

TransMiter의 Procrustes alignment 아이디어는 이 feature/token 공간에 적용할 때 더 의미가 있다.

### 4. 현재 설정에도 TinyUSFM -> SAM 실험 흔적이 있다

`config/distill_usfm_to_sam_multiclass.yaml`은 이미 `teacher: tinyusfm_multiclass`, `student: sam`, `method: cwd_feature`로 구성되어 있다. 이는 TransMiter 원본을 가져오기 전에 먼저 실행해볼 수 있는 가장 가까운 baseline이다.

## 현재 프로젝트와 맞지 않는 점

### 1. TransMiter 원본은 classification/VLM logit transfer에 맞춰져 있다

TransMiter는 CLIP류 vision-language classification 모델을 대상으로 한다. 원본 proxy는 class logit vector를 입력받아 fine-tuned class logit vector를 재구성한다.

반면 이 프로젝트는 medical image segmentation이고, 핵심 출력은 `[B, C, H, W]` dense mask logits다. 특히 binary segmentation에서는 `C=1`이라 class-logit vector 공간이 거의 없다. 원본 TransMiter의 class-space projection, auxiliary/negative class padding, class embedding 기반 전이는 binary segmentation에서는 의미가 크게 줄어든다.

### 2. Binary segmentation에서는 logit-space Procrustes 자유도가 너무 작다

TransMiter의 Procrustes alignment는 source/target logit 또는 latent vector 공간의 basis 차이를 맞추는 데 강점이 있다. 하지만 binary segmentation의 최종 logit은 foreground 1채널이다. 이를 `[bg, fg] = [-x, x]`로 확장해도 사실상 2차원 구조라, 복잡한 basis transfer가 줄 수 있는 이득이 제한적이다.

즉, 최종 mask logit에 원본 TransMiter proxy를 붙이면 구현은 가능하지만 방법론적 장점은 약하다.

### 3. Dense spatial information을 원본 proxy가 충분히 다루지 못한다

Segmentation 성능은 픽셀 단위 boundary, local context, spatial saliency에 크게 의존한다. TransMiter 원본은 이미지 단위 class logits를 변환하므로, boundary quality나 lesion shape prior를 직접 전달하지 않는다.

현재 프로젝트에는 오히려 CWD가 더 적합하다. CWD는 각 채널의 spatial distribution을 softmax로 정규화하고 teacher/student 간 공간 saliency를 맞춘다. medical segmentation에서는 이 신호가 class-logit proxy보다 더 직접적이다.

### 4. SAM의 prompt-free segmentation 구조와 원본 TransMiter의 가정이 다르다

SAM은 image encoder, prompt encoder, mask decoder로 구성된다. 현재 프로젝트에서는 prompt-free 또는 자동화된 segmentation 학습을 위해 SAM을 LoRA/decoder fine-tuning한다.

TransMiter는 source/target image classifier가 같은 class head 의미 공간을 공유하거나, 적어도 class head를 통해 비교 가능한 logit을 만든다는 가정이 강하다. SAM segmentation에서는 mask decoder token, prompt embedding, image embedding의 역할이 달라서 원본 proxy 삽입 위치가 명확하지 않다.

## 적용 가능성 판단

### 바로 적용하기 좋은 부분

현재 코드의 `UnifiedDistiller`와 `cwd_feature` preset을 활용하는 방식이 가장 적합하다.

추천 baseline:

```bash
uv run distill.py --config-name distill_usfm_to_sam_multiclass
```

binary task라면 별도 config를 만들어 다음 defaults를 쓰는 것이 좋다.

```yaml
defaults:
  - data: dynamic
  - teacher: tinyusfm
  - student: sam
  - method: cwd_feature
  - training: distillation
  - _self_
```

권장 loss ablation:

- `alpha=1, beta=1, gamma=0, delta=0, zeta=0`: supervised + vanilla logit KD
- `alpha=1, beta=0, gamma=0, delta=1, zeta=0`: supervised + logit CWD
- `alpha=1, beta=0, gamma=0, delta=1, zeta=1`: supervised + logit CWD + feature CWD
- `alpha=1, beta=1, gamma=0, delta=0, zeta=0, use_uncertainty_weighted_kd=true`: uncertainty-weighted KD 포함

### 조심해서 적용할 부분

TransMiter의 `ProxyModel`을 segmentation logit에 직접 붙이는 방식은 가능하지만 우선순위가 낮다.

가능한 형태:

- TinyUSFM pretrained or weak model logits -> TinyUSFM fine-tuned logits 변환 proxy 학습
- SAM logits에 동일 proxy를 적용
- SAM logits와 TinyUSFM logits의 통계 차이를 Procrustes 또는 affine mapping으로 보정

하지만 binary segmentation에서는 최종 logit channel 수가 작아 이 접근의 표현력이 제한될 가능성이 높다. multiclass segmentation에서는 조금 더 의미가 있지만, 여전히 spatial structure를 별도로 고려해야 한다.

### 가장 연구적으로 의미 있는 적용

TransMiter의 Procrustes alignment를 최종 logits가 아니라 feature/token 공간에 적용하는 것이 가장 타당하다.

구체적인 방향:

1. Frozen TinyUSFM teacher와 initial SAM student를 같은 train set에 forward한다.
2. TinyUSFM backbone block feature와 SAM image encoder block feature를 수집한다.
3. 각 feature를 `[B, C, H, W]` 또는 token matrix `[N, D]`로 정규화한다.
4. TinyUSFM feature space와 SAM feature space 사이의 linear/orthogonal mapping을 추정한다.
5. 이 mapping을 현재 `FeatureAdapter` 또는 SAM alignment layer 초기값으로 사용한다.
6. 이후 CWD/feature KD로 end-to-end distillation한다.

이 방식은 TransMiter의 "basis alignment 후 adaptation knowledge transfer" 개념을 유지하면서도 segmentation의 dense feature 구조에 맞춘 변형이다.

## 권장 실험 로드맵

### Stage 0: 설정 정리

현재 `config/distill.yaml`은 여전히 `teacher: sam`, `student: tinyusfm` 기본값이다. TinyUSFM -> SAM이 주 방향이라면 기본 config를 바꾸거나, `distill_tinyusfm_to_sam.yaml`처럼 명확한 이름의 config를 추가하는 것이 좋다.

### Stage 1: 기존 distillation baseline 확립

먼저 TransMiter를 이식하지 말고 아래 baseline을 비교한다.

- SAM fine-tuning only
- TinyUSFM teacher supervised performance
- TinyUSFM -> SAM vanilla KD
- TinyUSFM -> SAM CWD logit-only
- TinyUSFM -> SAM CWD logit + feature
- TinyUSFM -> SAM uncertainty-weighted KD

이 단계에서 TinyUSFM -> SAM이 SAM fine-tuning only보다 좋아지는지 확인해야 한다.

### Stage 2: feature-level TransMiter 변형

다음으로 feature/token alignment 초기화를 추가한다.

- TinyUSFM/SAM block pair: `[7, 9, 11]`
- mapping: orthogonal Procrustes 또는 ridge linear projection
- 적용 위치: `FeatureAdapter`, `alignment_layer`, 또는 distiller 내부 adapter
- 비교: random adapter init vs Procrustes init

이 실험은 TransMiter 방법론의 핵심인 "모델 간 표현 공간 정렬"을 segmentation에 맞게 옮기는 실험이다.

### Stage 3: logit-proxy TransMiter ablation

마지막으로 원본 TransMiter에 가장 가까운 logit proxy를 ablation으로 추가한다.

- binary에서는 낮은 우선순위
- multiclass에서는 중간 우선순위
- final mask logit보다 low-res logits 또는 multi-token mask decoder output에 붙이는 편이 더 낫다

## 최종 판단

TransMiter는 현재 프로젝트에 "그대로 적용"하기보다는 "아이디어를 변형해서 적용"하는 것이 맞다.

잘 맞는 부분:

- 작은 source model이 더 강한 상황에서도 transfer를 시도할 수 있다는 관점
- source adaptation knowledge를 target model로 옮기는 문제 설정
- 서로 다른 architecture 사이의 representation alignment
- Procrustes/orthogonal mapping을 이용한 lightweight transfer

잘 맞지 않는 부분:

- 원본의 classification/VLM class-logit 중심 설계
- binary segmentation의 낮은 logit 차원
- spatial boundary와 dense saliency를 직접 다루지 않는 proxy 구조
- SAM prompt/mask decoder 구조와 class head 기반 가정의 차이

따라서 이 프로젝트에서 가장 설득력 있는 방법론은 다음과 같다.

> TinyUSFM을 teacher로 사용하고, SAM을 LoRA/decoder student로 fine-tuning하되, 기본 신호는 CWD 및 feature distillation으로 주고, TransMiter의 Procrustes alignment는 feature adapter 초기화 또는 alignment layer 초기화에 사용하는 방식.

이 접근이 원본 TransMiter의 핵심 정신을 유지하면서도 medical image segmentation과 현재 코드베이스에 가장 잘 맞는다.
