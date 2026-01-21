# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TinyUSFM is a multi-model medical image segmentation training framework focusing on knowledge distillation from SAM (Segment Anything Model) to lightweight models. The project supports training, testing, and distillation workflows for medical imaging segmentation tasks.

**Key Models:**
- **SAM (Segment Anything Model)**: Teacher model with LoRA fine-tuning (vit_b, vit_l, vit_h variants)
- **TinyUSFM**: Lightweight student model for efficient inference
- **SegFormer**: Alternative segmentation model
- **SAM3**: Meta's latest SAM with vision-language capabilities (integrated via adapter/orchestrator)

## Essential Commands

### Environment Setup
```bash
# Install dependencies (uses uv package manager)
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Training

```bash
# Train SAM model (default)
python main.py

# Train specific model
python main.py model=sam         # SAM with default config
python main.py model=tinyusfm    # TinyUSFM model
python main.py model=segformer   # SegFormer model
python main.py model=sam3        # SAM3 model (simple mode)

# Override hyperparameters
python main.py model=sam training.batch_size=32 training.base_lr=0.001

# Specify GPU
python main.py hardware.gpu_ids=[0,1]

# List available models
python main.py list_models=true
```

### Testing

```bash
# Test mode
python main.py mode=test model=sam checkpoint=/path/to/checkpoint.pth

# Test-only mode (configured in config)
python main.py test_only.enabled=true test_only.checkpoint_path=/path/to/checkpoint.pth
```

### SAM3 Training

SAM3 supports two training modes:

```bash
# Simple mode - uses SAM3TrainerAdapter (BaseTrainer interface)
python main.py model=sam3

# Native mode - uses SAM3Orchestrator (full SAM3 DDP/AMP capabilities)
python main.py model=sam3 sam3.use_native_trainer=true

# With custom parameters
python main.py model=sam3 \
    sam3.resolution=1024 \
    sam3.enable_segmentation=true \
    sam3.amp_enabled=true \
    training.batch_size=2

# Direct orchestrator usage (advanced)
python -m trainers.sam3_adapter --config path/to/config.yaml
```

**SAM3 Key Configuration (`config/model/sam3.yaml`):**
- `sam3.use_native_trainer`: Use native SAM3 Trainer (default: false)
- `sam3.checkpoint_path`: Path to SAM3 checkpoint (null for HuggingFace download)
- `sam3.bpe_path`: BPE vocabulary path for text encoder
- `sam3.enable_segmentation`: Enable segmentation masks (default: true)
- `sam3.resolution`: Input resolution (default: 1008)
- `sam3.amp_enabled`: Enable AMP (default: true)
- `sam3.amp_dtype`: AMP dtype - bfloat16 or float16

### Knowledge Distillation

```bash
# Basic distillation (SAM → TinyUSFM)
python distill.py

# Override distillation parameters
python distill.py \
    teacher.lora_checkpoint=/path/to/sam_lora.pth \
    distillation.temperature=6.0 \
    distillation.alpha=0.5 \
    distillation.beta=0.5 \
    distillation.gamma=1.0

# Run batch distillation experiments
python run_distill_experiments.py
python run_distill_experiments.py --debug  # Quick debug run
```

### Evaluation

```bash
# Calculate metrics from predictions
python calculate_metrics.py \
    --pred_dir /path/to/predictions \
    --gt_dir /path/to/ground_truth \
    --output metrics.csv

# With Notion upload
python calculate_metrics.py \
    --pred_dir /path/to/predictions \
    --gt_dir /path/to/ground_truth \
    --output metrics.csv \
    --upload-notion \
    --notion-model-name "SAM-LoRA"
```

### Visualization

```bash
# Visualize analysis (t-SNE, feature maps)
python visualize_analysis.py
```

### Testing
```bash
# Run tests (pytest available in dev dependencies)
pytest

# Install dev dependencies if needed
uv sync --extra dev
```

## Architecture

### Configuration System (Hydra)

The project uses Hydra for hierarchical configuration management:

- **config/train.yaml**: Main training configuration entry point
- **config/distill.yaml**: Knowledge distillation configuration
- **config/model/**: Model-specific configs (sam.yaml, TinyUSFM.yaml, segformer.yaml, sam3.yaml)
- **config/data/**: Dataset configurations (BUSBRA.yaml, BUSI.yaml, etc.)
- **config/model/encoder/**: Encoder variants (vit_b.yaml, vit_l.yaml, vit_h.yaml)

Configuration override syntax: `python main.py key.subkey=value`

### Trainer System

The framework uses a factory pattern with model-specific trainers:

- **trainers/base_trainer.py**: Abstract base class with common training infrastructure
  - Handles: setup, data loading, training loop, validation, checkpointing, logging
  - Provides: early stopping, WandB integration, TensorBoard logging

- **trainers/model_builder.py**: Factory for creating trainers
  - Registry: `TRAINER_MAP` maps model names to trainer classes
  - To add new model: register in `TRAINER_MAP` and create corresponding trainer

- **Model-specific trainers**:
  - `sam_trainer.py`: SAM with LoRA fine-tuning (image encoder or full model)
  - `tinyusfm_trainer.py`: TinyUSFM lightweight model
  - `segformer_trainer.py`: SegFormer transformer-based segmentation
  - `sam3_adapter.py`: SAM3 integration module with two components:
    - `SAM3TrainerAdapter`: Inherits BaseTrainer for unified interface
    - `SAM3Orchestrator`: Wraps native SAM3 Trainer for full DDP/AMP support

**Key trainer methods to implement:**
- `create_model()`: Instantiate model architecture
- `create_dataloaders()`: Setup train/val/test data loaders
- `train_epoch()`: Single epoch training logic
- `validate()`: Validation logic

### Model Architecture

Models are in `model/` directory:

- **SAM models**:
  - `sam_lora_image_encoder.py`: LoRA adaptation of image encoder only
  - `sam_lora_image_encoder_mask_decoder.py`: LoRA for encoder + mask decoder
  - `segment_anything/`: Original SAM implementation

- **Lightweight models**:
  - `tinyusfm_seg.py`: TinyUSFM segmentation model
  - `usfm_seg.py`: USFM segmentation variant

- **SAM3**: `model/sam3/` - Meta's SAM3 implementation
  - `model/`: Core model architecture (`sam3_image.py`, `vl_combiner.py`)
  - `train/`: Native training infrastructure with DDP support
    - `trainer.py`: Full-featured Trainer with AMP, gradient accumulation
    - `configs/`: Hydra configs for various tasks
    - `loss/`: Loss functions (`sam3_loss.py`)
  - `eval/`: Evaluation utilities and postprocessors
  - `agent/`: Agent-based features for interactive segmentation

### Knowledge Distillation

Located in `distill.py`:

- **Teacher**: Fine-tuned SAM with LoRA (frozen during distillation)
- **Student**: TinyUSFM (trained)
- **Loss components**:
  - Task loss (α): Ground truth segmentation loss (BCE + Dice)
  - Distillation loss (β): KL divergence between teacher/student logits with temperature scaling
  - Feature loss (γ): Optional MSE between intermediate features

**Key configuration parameters:**
- `temperature`: Softness of probability distributions (typically 4-8)
- `alpha`: Weight for task loss (0-1)
- `beta`: Weight for distillation loss (0-1, typically alpha + beta = 1)
- `gamma`: Weight for feature distillation (0 = disabled)

### Data Processing

- **utils/data_processing_seg.py**: Main data processing for segmentation
  - `SegDatasetProcessor`: Handles dataset loading, augmentation, train/val/test splits
  - Supports multiple medical imaging datasets (BUSBRA, BUSI, BUS_UCLM, etc.)

### Utilities

- **utils/evaluate.py**: `Evaluator_seg` class for computing segmentation metrics (Dice, HD95, IoU, etc.)
- **utils/sam_utils.py**: SAM-specific utilities including DiceLoss
- **utils/schedule.py**: Learning rate schedulers (WarmupPolyLR, ReduceLROnPlateau)
- **utils/logger.py**: Logging setup
- **utils/visualize.py**: Visualization utilities for segmentation results

## Project Structure Notes

### Output Organization

Training outputs are organized as:
```
logs/{model_name}/{dataset}/{train_type}/{timestamp}/
├── checkpoints/
│   ├── best_epoch_N_diceX.XXXX.pth
│   └── checkpoint_epoch_N.pth
├── tensorboard/
├── config.yaml
└── training.log
```

Distillation outputs:
```
logs/distillation/{dataset}/{timestamp}/
├── models/
├── visualizations/
├── test_results.txt
└── summary.json
```

### Checkpoints

- **SAM checkpoints**: ImageNet pretrained weights in `checkpoints/sam_vit_{b,l,h}_*.pth`
- **LoRA checkpoints**: Fine-tuned LoRA parameters (typically much smaller than full model)
- **TinyUSFM checkpoints**: Full model weights in `checkpoints/TinyUSFM.pth`

### WandB Integration

- Project: `TinyUSFM`
- Entity: `hheo`
- Automatically logs: losses, metrics, learning rate, visualizations
- Config stored in `.env` file

### GPU Management

- Configure GPUs via `hardware.gpu_ids` parameter
- Automatically sets `CUDA_VISIBLE_DEVICES` environment variable
- Multi-GPU training supported

## Important Implementation Details

### SAM3 Integration

SAM3 was integrated from the official Meta repository with a dual-mode architecture:

**Simple Mode (SAM3TrainerAdapter):**
- Inherits from `BaseTrainer` for consistent interface with other models
- Suitable for single-GPU training, quick experiments, debugging
- Uses unified data loaders (`SegDatasetProcessor`)
- Registered in `ModelBuilder` as 'sam3'

**Native Mode (SAM3Orchestrator):**
- Preserves all SAM3 native capabilities (multi-GPU DDP, AMP, gradient accumulation)
- Converts unified config to SAM3's Hydra-based format
- Supports SLURM/Submitit for cluster training
- Activated via `sam3.use_native_trainer=true`

**Key Files:**
- `trainers/sam3_adapter.py`: Integration module (SAM3TrainerAdapter + SAM3Orchestrator)
- `config/model/sam3.yaml`: SAM3-specific configuration
- `model/sam3/`: Original SAM3 codebase from Meta

### LoRA Fine-tuning

SAM uses LoRA (Low-Rank Adaptation) for efficient fine-tuning:
- Only LoRA parameters are trained, base SAM weights are frozen
- Rank parameter (`rank: 4` typically) controls adapter capacity
- Two variants: encoder-only or encoder+decoder adaptation

### Segmentation Task Types

Configured via `task.type`:
- `organ`: Organ segmentation (binary or multi-class)
- `tumor`: Tumor segmentation

### Dataset Configuration

Datasets configured in `config/data/`:
- Must specify: `train_dataset`, `val_dataset`, `test_dataset`
- Each dataset config includes: path, image size, num_classes
- Supports combined datasets (e.g., BUSBRA+BUSI)

### Early Stopping

Configured per model:
- `early_stopping.enabled`: Enable/disable
- `early_stopping.patience`: Epochs to wait before stopping
- `early_stopping.min_delta`: Minimum improvement threshold

## Development Workflow

1. **Adding a new model**:
   - Create trainer class inheriting from `BaseTrainer` in `trainers/`
   - Implement abstract methods: `create_model()`, `create_dataloaders()`, `train_epoch()`, `validate()`
   - Register in `trainers/model_builder.py::TRAINER_MAP`
   - Create config file in `config/model/your_model.yaml`

2. **Adding a new dataset**:
   - Create config in `config/data/your_dataset.yaml`
   - Ensure dataset follows expected structure (images/, masks/ directories)
   - Update `utils/data_processing_seg.py` if custom loading logic needed

3. **Modifying distillation**:
   - Core logic in `distill.py`
   - Loss function: `KnowledgeDistillationLoss` class
   - Feature adaptation: `FeatureAdapter` class for dimension matching

## Notes for Claude Code

- This is a research codebase for medical image segmentation with focus on knowledge distillation
- Korean comments may appear in some documentation files (Korean documentation exists in README.md and DISTILLATION_README.md)
- Environment variables for Notion integration are in `.env` (token and database ID)
- The project uses `uv` as package manager (modern alternative to pip)
- Checkpoints and logs can be large - avoid committing them
- WandB logs are stored locally in `wandb/` directory
