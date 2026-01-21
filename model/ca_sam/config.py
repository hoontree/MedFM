"""
Configuration for CA-SAM Training

모든 하이퍼파라미터와 설정을 관리하는 config 클래스
"""

from dataclasses import dataclass, field
from typing import List, Optional
import yaml
import os


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # SAM configuration
    sam_model_type: str = "vit_b"  # vit_b, vit_l, vit_h
    sam_checkpoint: str = "sam_vit_b_01ec64.pth"
    encoder_output_dim: int = 256  # ViT-B: 256, ViT-L: 1024, ViT-H: 1280
    
    # Alignment Layer configuration
    alignment_hidden_dim: int = 256
    alignment_num_blocks: int = 4  # 논문: 3-5 blocks
    
    # VAE Router configuration
    vae_latent_dim: int = 64
    vae_beta: float = 16.5  # KL divergence weight
    attention_temperature: float = 1.0
    
    # Threshold calibration
    threshold_percentile: float = 97.0  # p97


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Optimization
    learning_rate: float = 1e-4
    batch_size: int = 6
    num_epochs: int = 24
    
    # VAE training
    vae_learning_rate: float = 5e-4
    vae_num_epochs: int = 10
    vae_batch_size: int = 32
    
    # Device
    device: str = "cuda"
    num_workers: int = 4
    
    # Checkpointing
    save_dir: str = "./checkpoints"
    save_interval: int = 5  # Save every N epochs
    
    # Logging
    log_interval: int = 10  # Log every N batches


@dataclass
class DataConfig:
    """Dataset configuration"""
    # Image preprocessing
    image_size: int = 1024  # SAM default: 1024x1024
    
    # Task sequence (continual learning order)
    task_sequence: List[str] = field(default_factory=lambda: [
        "ACDC",
        "EBHI-SEG", 
        "56Nx",
        "DN",
        "Polyp",
        "MSD_Prostate",
        "MSD_Spleen",
        "Promise12",
        "STS-2D"
    ])
    
    # Dataset paths
    data_root: str = "./data/medical_datasets"
    
    # Data splits
    train_split: float = 0.8
    val_split: float = 0.2


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    # Metrics
    compute_biou: bool = True
    biou_dilation: int = 5
    
    # Continual learning metrics
    compute_forgetting: bool = True
    
    # OOD evaluation datasets
    ood_datasets: List[str] = field(default_factory=lambda: [
        "DDTI",
        "BUSI_benign",
        "BUSI_malignant", 
        "Brain_Tumor",
        "CAMO"
    ])


@dataclass
class Config:
    """Complete configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Experiment
    experiment_name: str = "ca_sam_experiment"
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
            experiment_name=config_dict.get('experiment_name', 'ca_sam_experiment'),
            seed=config_dict.get('seed', 42)
        )
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'model': {
                'sam_model_type': self.model.sam_model_type,
                'sam_checkpoint': self.model.sam_checkpoint,
                'encoder_output_dim': self.model.encoder_output_dim,
                'alignment_hidden_dim': self.model.alignment_hidden_dim,
                'alignment_num_blocks': self.model.alignment_num_blocks,
                'vae_latent_dim': self.model.vae_latent_dim,
                'vae_beta': self.model.vae_beta,
                'attention_temperature': self.model.attention_temperature,
                'threshold_percentile': self.model.threshold_percentile,
            },
            'training': {
                'learning_rate': self.training.learning_rate,
                'batch_size': self.training.batch_size,
                'num_epochs': self.training.num_epochs,
                'vae_learning_rate': self.training.vae_learning_rate,
                'vae_num_epochs': self.training.vae_num_epochs,
                'vae_batch_size': self.training.vae_batch_size,
                'device': self.training.device,
                'num_workers': self.training.num_workers,
                'save_dir': self.training.save_dir,
                'save_interval': self.training.save_interval,
                'log_interval': self.training.log_interval,
            },
            'data': {
                'image_size': self.data.image_size,
                'task_sequence': self.data.task_sequence,
                'data_root': self.data.data_root,
                'train_split': self.data.train_split,
                'val_split': self.data.val_split,
            },
            'evaluation': {
                'compute_biou': self.evaluation.compute_biou,
                'biou_dilation': self.evaluation.biou_dilation,
                'compute_forgetting': self.evaluation.compute_forgetting,
                'ood_datasets': self.evaluation.ood_datasets,
            },
            'experiment_name': self.experiment_name,
            'seed': self.seed,
        }
        
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def __str__(self):
        """Pretty print configuration"""
        lines = [
            "="*60,
            "CA-SAM Configuration",
            "="*60,
            "",
            "Model:",
            f"  SAM: {self.model.sam_model_type}",
            f"  Encoder dim: {self.model.encoder_output_dim}",
            f"  Alignment blocks: {self.model.alignment_num_blocks}",
            f"  VAE latent dim: {self.model.vae_latent_dim}",
            f"  VAE β: {self.model.vae_beta}",
            "",
            "Training:",
            f"  Learning rate: {self.training.learning_rate}",
            f"  Batch size: {self.training.batch_size}",
            f"  Epochs: {self.training.num_epochs}",
            f"  Device: {self.training.device}",
            "",
            "Data:",
            f"  Image size: {self.data.image_size}",
            f"  Tasks: {len(self.data.task_sequence)}",
            f"  Task sequence: {', '.join(self.data.task_sequence[:3])}...",
            "",
            "="*60
        ]
        return "\n".join(lines)


# Default configuration
DEFAULT_CONFIG = Config()


if __name__ == "__main__":
    print("="*60)
    print("Configuration Test")
    print("="*60)
    
    # Create default config
    config = Config()
    print("\nDefault Configuration:")
    print(config)
    
    # Save to YAML
    yaml_path = "/tmp/ca_sam_config.yaml"
    config.to_yaml(yaml_path)
    print(f"\n✓ Configuration saved to: {yaml_path}")
    
    # Load from YAML
    loaded_config = Config.from_yaml(yaml_path)
    print("\n✓ Configuration loaded from YAML")
    
    # Verify
    assert config.model.sam_model_type == loaded_config.model.sam_model_type
    assert config.training.learning_rate == loaded_config.training.learning_rate
    print("\n✓ Configuration save/load verified")
    
    print("\n" + "="*60)
    print("Configuration ready to use!")
