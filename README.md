# medfm

의료 영상(초음파) 분할을 위한 학습/증류 프레임워크입니다.  
Hydra 기반 설정 시스템으로 **Teacher 학습**, **Knowledge Distillation**, **평가(Eval)**를 실행할 수 있습니다.

## 주요 기능

- SAM 계열 Teacher 학습 (LoRA/Hybrid 설정 포함)
- TinyUSFM Student 학습
- Teacher → Student 지식 증류 (`distill.py`)
- Teacher 학습 후 자동 증류 파이프라인 (`train.py pipeline.enabled=true`)
- 다중 테스트 데이터셋 분리 평가 (`eval.py`)

## 프로젝트 구조

- `train.py`: 기본 학습 엔트리포인트 (Hydra `config/train.yaml`)
- `distill.py`: 지식 증류 엔트리포인트 (Hydra `config/distill.yaml`)
- `eval.py`: 체크포인트 평가 엔트리포인트
- `config/`: 모델/데이터/학습 설정
- `trainers/`: 모델별 Trainer 구현
- `utils/`: 데이터 처리, 평가, 로깅, 스케줄러 유틸
- `tests/`: pytest 테스트

## 환경 설정

### 1) 의존성 설치

```bash
uv sync
```

개발용 테스트/도구까지 설치하려면:

```bash
uv sync --extra dev
```

### 2) 가상환경 활성화

```bash
source .venv/bin/activate
```

## 데이터셋 설정

데이터 경로는 각 데이터 설정 파일에서 관리합니다.

- 예: `config/data/BUID.yaml`의 `path.root`
- 예: `config/data/BUSBRA.yaml`의 `path.root`

로컬 환경에 맞게 `path.root`를 수정한 뒤 실행하세요.

## 실행 방법

### 1) 기본 학습

기본값(`config/train.yaml`)으로 학습:

```bash
python train.py
```

모델/데이터 오버라이드 예시:

```bash
python train.py model=E0_DF data=dynamic
python train.py hardware.gpu_ids=[0]
python train.py training.batch_size=8 training.num_epochs=100
```

### 2) Teacher 학습 후 자동 Distillation 파이프라인

Teacher 학습 완료 후, 같은 컨텍스트(데이터/하드웨어/스플릿)로 Distillation 단계까지 자동 실행:

```bash
python train.py pipeline.enabled=true model=E0_AL_DL
```

### 3) Distillation 단독 실행

기본 distillation 설정으로 실행:

```bash
python distill.py
```

설정 오버라이드 예시:

```bash
python distill.py \
  teacher=E0_AL_DL \
  student=tinyusfm \
  distillation.adaptation_ratio=0.3 \
  hardware.gpu_ids=[0]
```

### 4) 평가

체크포인트 경로를 지정해 테스트셋 평가:

```bash
python eval.py checkpoint=/path/to/checkpoint.pth
```

동적 데이터셋 구성으로 평가:

```bash
python eval.py \
  checkpoint=/path/to/checkpoint.pth \
  data=dynamic \
  data.test='[BUID,BUS_UCLM]'
```

평가 결과는 `logs/eval/...` 하위에 저장됩니다.

## 자주 쓰는 Hydra 오버라이드

- GPU 지정: `hardware.gpu_ids=[0]`
- 배치 크기: `training.batch_size=8`
- 학습 epoch: `training.num_epochs=100`
- 데이터셋 동적 지정: `data=dynamic data.train='[BUSBRA,BUSI]' data.test='[BUID]'`

## 테스트

```bash
pytest
```

## 라이선스

`LICENSE` 파일을 참고하세요.
