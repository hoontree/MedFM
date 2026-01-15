# TinyUSFM - Multi-Model Training Framework

통합된 딥러닝 모델 학습/테스트 프레임워크입니다. SAM, TinyUSFM 등 다양한 모델을 하나의 인터페이스로 관리할 수 있습니다.

## ✨ 주요 특징

- 🎯 **통합 인터페이스**: 하나의 명령으로 모든 모델 학습/테스트
- 🔧 **모듈화 설계**: 새로운 모델 추가가 쉬움
- ⚙️ **Hydra Config**: 강력하고 유연한 설정 시스템
- 📊 **자동 로깅**: WandB, TensorBoard 통합
- 🚀 **확장 가능**: 새 모델을 150 lines만으로 추가

## 🚀 빠른 시작

### 사용 가능한 모델 확인

```bash
python main.py list_models=true
```

### 기본 학습

```bash
# SAM 모델 학습 (기본)
python main.py

# TinyUSFM 모델 학습
python main.py model=tinyusfm

# VIT-L 학습
python main.py model=vit_l
```

### 하이퍼파라미터 조정

```bash
python main.py model=sam \
    training.batch_size=64 \
    training.base_lr=0.001 \
    hardware.gpu_ids=[0,1]
```

### 테스트

```bash
python main.py mode=test model=sam \
    checkpoint=/path/to/checkpoint.pth
```

## 📖 문서

- **[QUICKSTART.md](QUICKSTART.md)** ⭐ - 빠른 시작 가이드 (추천!)
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - 상세 사용법
- **[FRAMEWORK_README.md](FRAMEWORK_README.md)** - 프레임워크 구조
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - 마이그레이션 가이드

## 💡 사용 예제

```bash
# SAM 학습
python main.py model=sam training.batch_size=128

# TinyUSFM Pretrained
python main.py model=tinyusfm model.pretrained=true

# 테스트
python main.py mode=test model=sam checkpoint=/path/to/checkpoint.pth

# 스크립트 사용
./scripts/train_sam.sh
./scripts/test.sh sam /path/to/checkpoint.pth
```

## 📁 프로젝트 구조

```
TinyUSFM/
├── main.py                    # 통합 진입점
├── trainers/                  # 모델별 trainer
├── models/                    # Model builder
├── config/                    # Hydra 설정
│   ├── train.yaml
│   ├── model/
│   └── data/
└── scripts/                   # 실행 스크립트
```

상세한 내용은 [QUICKSTART.md](QUICKSTART.md)를 참조하세요.
