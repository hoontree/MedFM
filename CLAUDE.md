# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TinyUSFM is a multi-model medical image segmentation training framework focusing on knowledge distillation for medical imaging segmentation tasks. Currently, TinyUSFM outperforms SAM across all evaluation metrics, so experiments are being conducted in the **TinyUSFM → SAM** distillation direction (TinyUSFM as teacher, SAM as student).

**Key Models:**
- **TinyUSFM**: Lightweight model — currently used as **teacher** (outperforms SAM on all metrics)
- **SAM (Segment Anything Model)**: Now used as **student**, with LoRA fine-tuning (vit_b, vit_l, vit_h variants)
- **SegFormer**: Alternative segmentation model

## Essential Commands

### Environment Setup
```bash
# Install dependencies (uses uv package manager)
uv sync
```

### Training

```bash
# Train SAM model (default)
uv run main.py

# Train specific model
uv run main.py model=sam         # SAM with default config
uv run main.py model=tinyusfm    # TinyUSFM model
uv run main.py model=segformer   # SegFormer model
# Override hyperparameters
uv run main.py model=sam training.batch_size=32 training.base_lr=0.001

# Specify GPU
uv run main.py hardware.gpu_ids=[0,1]

# List available models
uv run main.py list_models=true
```

### Testing

```bash
# Test mode
uv run main.py mode=test model=sam checkpoint=/path/to/checkpoint.pth

# Test-only mode (configured in config)
uv run main.py test_only.enabled=true test_only.checkpoint_path=/path/to/checkpoint.pth
```

### Knowledge Distillation

```bash
# Basic distillation (TinyUSFM → SAM, current experimental direction)
uv run distill.py

# Override distillation parameters
uv run distill.py \
    teacher.lora_checkpoint=/path/to/sam_lora.pth \
    distillation.temperature=6.0 \
    distillation.alpha=0.5 \
    distillation.beta=0.5 \
    distillation.gamma=1.0

# Run batch distillation experiments
uv run run_distill_experiments.py
uv run run_distill_experiments.py --debug  # Quick debug run
```

### Teacher→Distill Pipeline (Single Command)

```bash
# Train teacher, then automatically run distillation with the same data/hardware/split context
uv run train.py pipeline.enabled=true model.encoder_mode=frozen model.decoder_mode=lora

# Example with explicit encoder/decoder modes
uv run train.py \
  pipeline.enabled=true \
  model.encoder_mode=conv_lora \
  model.decoder_mode=lora
```

Pipeline behavior:
- Distillation uses the teacher best checkpoint from the same run.
- Sweep metric is `final_test/BUID/dice`.

## Architecture

### Configuration System (Hydra)

The project uses Hydra for hierarchical configuration management:

- **config/train.yaml**: Main training configuration entry point
- **config/distill.yaml**: Knowledge distillation configuration
- **config/model/**: Model-specific configs (sam.yaml, TinyUSFM.yaml, segformer.yaml)
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


### Knowledge Distillation

Located in `distill.py`, implemented in `distillers/unified_distiller.py`:

- **Teacher**: TinyUSFM (frozen during distillation — outperforms SAM on all metrics)
- **Student**: SAM with LoRA (trained)
- **Loss components** (`UnifiedDistiller`):
  - Task loss (α): Ground truth segmentation loss (BCE + Dice)
  - Distillation loss (β): KL divergence between teacher/student logits with temperature scaling
  - Feature loss (γ): Optional MSE between intermediate feature maps (requires `layer_mapping` config)

**Key configuration parameters:**
- `temperature`: Softness of probability distributions (typically 4-8)
- `alpha`: Weight for task loss (0-1)
- `beta`: Weight for distillation loss (0-1, typically alpha + beta = 1)
- `gamma`: Weight for feature distillation (0 = disabled)

**Note:** `UnifiedDistiller` supports optional GradNorm automatic loss balancing (`use_gradnorm=true`). The `LOSS_WEIGHT_MAP` covers only the three active losses (task/distill/feature).

### Data Processing

- **utils/data_processing_seg.py**: Main data processing for segmentation
  - `SegDatasetProcessor`: Handles dataset loading, augmentation, train/val/test splits
  - Supports multiple medical imaging datasets (BUSBRA, BUSI, BUS_UCLM, etc.)

#### Dataset Split Strategy (`config/data/dynamic.yaml`)

Train datasets (BUSBRA, BUSI, B) are split 70/15/15 into train/val/internal-test.
External validation datasets (BUID, BUS_UCLM, BUS_UCLM_filtered) are loaded with `usage="external"` (all samples).

`build_data_loaders()` returns:
- `train_loader`: 70% of train datasets
- `val_loader`: 15% val split — used for checkpoint selection
- `test_loaders`: dict with both internal and external sets:
  ```
  {
    "BUSBRA_test": internal 15%,
    "BUSI_test":   internal 15%,
    "B_test":      internal 15%,
    "BUID":        external (all),
    "BUS_UCLM":    external (all),
    "BUS_UCLM_filtered": external (all),
  }
  ```

`load_dataset_from_config(cfg, name, split, force_external=False)`:
- `force_external=True` → sets `usage="external"` (used for external validation sets)
- `force_external=False` (default) → keeps config's `usage` value (used for internal train/val/test splits)

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
- Pipeline sweeps should run with teacher W&B disabled (`wandb.disabled=true`) so one sweep trial maps to one distillation run metric (`final_test/BUID/dice`).

### GPU Management

- Configure GPUs via `hardware.gpu_ids` parameter
- Automatically sets `CUDA_VISIBLE_DEVICES` environment variable
- Multi-GPU training supported

## Important Implementation Details

### SAM Adaptation (`model/sam_hybrid_adapter.py`)

`LoRA_Sam` provides a unified SAM wrapper with flexible encoder/decoder adaptation modes, configured via `encoder_mode` / `decoder_mode` in the model config.

**Encoder modes:** `lora`, `conv_lora`, `ft`, `frozen`
**Decoder modes:** `lora`, `ft`, `frozen`

Key combinations:
- `encoder_mode=lora, decoder_mode=lora` — LoRA on both (default efficient fine-tuning)
- `encoder_mode=conv_lora` — Conv-LoRA with MoE experts on encoder qkv layers
- `encoder_mode=frozen, decoder_mode=ft` — freeze encoder, fine-tune decoder only

**Key parameters:**
- `r_e` / `r_d`: LoRA rank for encoder / decoder (default 4)
- `conv_lora_expert_num`: number of MoE experts for Conv-LoRA (default 4)
- `checkpoint`: SAM pretrained path (e.g. `checkpoints/sam_vit_b_*.pth`) or fine-tuned checkpoint

### Dataset Configuration

Datasets configured in `config/data/`:
- Must specify: `train_dataset`, `val_dataset`, `test_dataset`
- Each dataset config includes: path, image size, num_classes
- Supports combined datasets (e.g., BUSBRA+BUSI)

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
   - Loss implementation: `UnifiedDistiller` in `distillers/unified_distiller.py`
   - Active losses: task (α), logit distillation (β), feature distillation (γ)
   - Feature adaptation: `FeatureAdapter` class for channel dimension matching
   - Metric keys from `Evaluator_seg` are always capitalized: `"Dice"`, `"IoU"`, `"HD95"`, `"BIoU"`

## Notes for Claude Code

- This is a research codebase for medical image segmentation with focus on knowledge distillation
- Korean comments may appear in some documentation files (Korean documentation exists in README.md and DISTILLATION_README.md)
- Environment variables for Notion integration are in `.env` (token and database ID)
- The project uses `uv` as package manager (modern alternative to pip)
- Checkpoints and logs can be large - avoid committing them
- WandB logs are stored locally in `wandb/` directory
