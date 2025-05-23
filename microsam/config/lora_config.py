"""
LoRA微调配置管理模块 (修复版)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
from pathlib import Path


@dataclass
class LoRAConfig:
    """LoRA适配器配置"""
    
    # LoRA核心参数
    rank: int = 8                    # LoRA rank (通常4-64)
    alpha: float = 16.0              # LoRA alpha (通常等于rank的2倍)
    dropout: float = 0.1             # LoRA dropout
    target_modules: Optional[List[str]] = None  # 目标模块，None表示自动选择
    
    # 训练相关
    bias: str = "none"               # bias处理: "none", "all", "lora_only"
    task_type: str = "FEATURE_EXTRACTION"  # 任务类型
    
    def __post_init__(self):
        if self.target_modules is None:
            # 默认目标模块（针对Vision Transformer）
            self.target_modules = [
                "qkv", "proj", "fc1", "fc2", "mlp.lin1", "mlp.lin2",
                "attention.qkv", "attention.proj", "mlp"
            ]


@dataclass
class TrainingConfig:
    """训练配置"""
    
    # 基本训练参数
    learning_rate: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 10
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 50
    
    # 优化器设置
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # 学习率调度
    lr_scheduler_type: str = "cosine"
    num_warmup_steps: int = 100
    
    # 数据增强
    use_data_augmentation: bool = True
    augmentation_probability: float = 0.5
    
    # 早停和保存
    early_stopping_patience: int = 5
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False


@dataclass
class DataConfig:
    """数据配置 (修复版)"""
    
    # 数据路径
    train_data_dir: str = ""
    val_data_dir: str = ""
    test_data_dir: str = ""
    
    # 数据处理
    image_size: tuple = (1024, 1024)  # SAM默认输入尺寸
    max_objects_per_image: int = 100
    train_split_ratio: float = 0.8
    val_split_ratio: float = 0.1
    
    # 数据加载
    batch_size: int = 8  # 添加批大小
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    # prefetch_factor: Optional[int] = None
    
    # 数据过滤
    min_object_size: int = 10
    max_object_size: Optional[int] = None
    filter_empty_images: bool = True
    
    # 数据增强 - 添加缺少的属性
    use_data_augmentation: bool = True
    augmentation_probability: float = 0.5
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.5
    rotation_prob: float = 0.3
    brightness_contrast_prob: float = 0.3
    noise_prob: float = 0.2
    
    # 归一化参数
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class ModelConfig:
    """模型配置"""
    
    # 基础模型
    base_model_name: str = "vit_b_lm"
    base_model_path: Optional[str] = None
    
    # 模型架构
    freeze_backbone: bool = True
    freeze_prompt_encoder: bool = True
    freeze_mask_decoder: bool = False
    
    # LoRA设置
    apply_lora_to: List[str] = field(default_factory=lambda: ["image_encoder"])
    
    # 输出设置
    num_classes: int = 1
    use_aux_loss: bool = True


@dataclass
class ExperimentConfig:
    """实验配置"""
    
    # 实验信息
    experiment_name: str = "lora_finetune"
    run_name: Optional[str] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # 输出路径
    output_dir: str = "./lora_experiments"
    logging_dir: str = "./logs"
    cache_dir: str = "./cache"
    
    # 实验追踪
    use_wandb: bool = False
    wandb_project: str = "micro_sam_lora"
    wandb_entity: Optional[str] = None
    
    # 硬件设置
    device: str = "auto"  # "auto", "cuda", "cpu"
    mixed_precision: bool = True
    dataloader_num_workers: int = 4
    
    # 调试和监控
    debug_mode: bool = False
    profile_training: bool = False
    save_strategy: str = "steps"
    evaluation_strategy: str = "steps"


@dataclass 
class LoRATrainingSettings:
    """LoRA训练完整设置"""
    
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def __post_init__(self):
        # 自动设置运行名称
        if self.experiment.run_name is None:
            self.experiment.run_name = f"{self.experiment.experiment_name}_{self.model.base_model_name}_r{self.lora.rank}"
        
        # 创建输出目录
        Path(self.experiment.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.experiment.logging_dir).mkdir(parents=True, exist_ok=True)
        Path(self.experiment.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # 同步批大小设置
        if hasattr(self.training, 'batch_size'):
            self.data.batch_size = self.training.batch_size
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        from dataclasses import asdict
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """从字典创建配置"""
        return cls(
            lora=LoRAConfig(**config_dict.get('lora', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {}))
        )
    
    @classmethod
    def from_json(cls, config_path: str):
        """从JSON文件加载配置"""
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def save_to_json(self, config_path: str):
        """保存配置到JSON文件"""
        import json
        config_dict = self.to_dict()
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def validate(self) -> bool:
        """验证配置有效性"""
        errors = []
        
        # 验证LoRA参数
        if self.lora.rank <= 0:
            errors.append("LoRA rank must be positive")
        
        if self.lora.alpha <= 0:
            errors.append("LoRA alpha must be positive")
        
        if not (0 <= self.lora.dropout <= 1):
            errors.append("LoRA dropout must be between 0 and 1")
        
        # 验证训练参数
        if self.training.learning_rate <= 0:
            errors.append("Learning rate must be positive")
        
        if self.training.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        if self.training.num_epochs <= 0:
            errors.append("Number of epochs must be positive")
        
        # 验证数据配置
        if not self.data.train_data_dir:
            errors.append("Training data directory is required")
        
        if not (0 < self.data.train_split_ratio < 1):
            errors.append("Train split ratio must be between 0 and 1")
        
        # 验证路径
        try:
            Path(self.experiment.output_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory: {e}")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    def get_model_output_dir(self) -> Path:
        """获取模型输出目录"""
        return Path(self.experiment.output_dir) / self.experiment.run_name
    
    def get_checkpoint_dir(self) -> Path:
        """获取检查点目录"""
        return self.get_model_output_dir() / "checkpoints"
    
    def get_logs_dir(self) -> Path:
        """获取日志目录"""
        return Path(self.experiment.logging_dir) / self.experiment.run_name


# 预定义的配置模板
LORA_PRESET_CONFIGS = {
    "quick": LoRATrainingSettings(
        lora=LoRAConfig(rank=4, alpha=8),
        training=TrainingConfig(
            batch_size=4,
            num_epochs=3,
            learning_rate=5e-4,
            save_steps=100,
            eval_steps=50
        ),
        experiment=ExperimentConfig(experiment_name="quick_test")
    ),
    
    "standard": LoRATrainingSettings(
        lora=LoRAConfig(rank=8, alpha=16),
        training=TrainingConfig(
            batch_size=8,
            num_epochs=10,
            learning_rate=1e-4,
            save_steps=500,
            eval_steps=100
        ),
        experiment=ExperimentConfig(experiment_name="standard_training")
    ),
    
    "high_rank": LoRATrainingSettings(
        lora=LoRAConfig(rank=32, alpha=64),
        training=TrainingConfig(
            batch_size=4,
            num_epochs=15,
            learning_rate=5e-5,
            save_steps=500,
            eval_steps=100
        ),
        experiment=ExperimentConfig(experiment_name="high_rank_training")
    ),
    
    "debug": LoRATrainingSettings(
        lora=LoRAConfig(rank=2, alpha=4),
        training=TrainingConfig(
            batch_size=2,
            num_epochs=1,
            learning_rate=1e-3,
            save_steps=10,
            eval_steps=5,
            logging_steps=1
        ),
        experiment=ExperimentConfig(
            experiment_name="debug",
            debug_mode=True
        )
    )
}


def get_config_for_model(model_name: str) -> LoRATrainingSettings:
    """为特定模型获取推荐配置"""
    base_config = LORA_PRESET_CONFIGS["standard"].copy() if hasattr(LORA_PRESET_CONFIGS["standard"], 'copy') else LoRATrainingSettings()
    
    # 重新创建配置以避免引用问题
    base_config = LoRATrainingSettings()
    
    if model_name == "vit_t_lm":
        # 小模型可以用更高的学习率和更大的batch size
        base_config.training.learning_rate = 2e-4
        base_config.training.batch_size = 16
        base_config.lora.rank = 8
        
    elif model_name == "vit_b_lm":
        # 中等模型使用标准配置
        base_config.training.learning_rate = 1e-4
        base_config.training.batch_size = 8
        base_config.lora.rank = 8
        
    elif model_name == "vit_l_lm":
        # 大模型需要更小的学习率和batch size
        base_config.training.learning_rate = 5e-5
        base_config.training.batch_size = 4
        base_config.lora.rank = 16
    
    base_config.model.base_model_name = model_name
    return base_config