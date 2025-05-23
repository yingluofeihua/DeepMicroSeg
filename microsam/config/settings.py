"""
配置管理模块
包含所有系统配置和默认设置
"""

from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    device: str = "cuda"
    cache_dir: Optional[str] = None


@dataclass
class EvaluationConfig:
    """评测配置"""
    batch_size: Optional[int] = None  # None表示处理所有图像
    skip_existing: bool = True
    process_timeout: int = 600  # 秒
    save_visualizations: bool = True
    save_detailed_metrics: bool = True
    create_summary_report: bool = True
    generate_unified_csv: bool = True


@dataclass
class HardwareConfig:
    """硬件配置"""
    max_gpu_workers: int = 4
    device: str = "cuda"
    enable_multiprocessing: bool = True


class BatchEvaluationSettings:
    """批量评测系统设置"""
    
    def __init__(self, 
                 output_base_dir: str = None,
                 cache_dir: str = None):
        
        # 基本路径
        self.output_base_dir = output_base_dir or "/tmp/batch_evaluation_results"
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "micro_sam")
        
        # 模型配置
        self.models = [
            ModelConfig("vit_t_lm"),
            ModelConfig("vit_b_lm"), 
            ModelConfig("vit_l_lm")
        ]
        
        # 评测配置
        self.evaluation = EvaluationConfig()
        
        # 硬件配置
        self.hardware = HardwareConfig()
        
        # 支持的文件格式
        self.supported_image_formats = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        self.supported_mask_formats = ['.png', '.tif', '.tiff']
        
        # 评测指标配置
        self.metrics_config = {
            'calculate_ap50': True,
            'calculate_ap75': True,
            'calculate_iou': True,
            'calculate_dice': True,
            'calculate_hd95': True,
            'hd95_percentile': 95
        }
        
    def get_model_names(self) -> List[str]:
        """获取所有模型名称"""
        return [model.name for model in self.models]
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """获取指定模型的配置"""
        for model in self.models:
            if model.name == model_name:
                return model
        return None
    
    def update_paths(self, output_dir: str = None, cache_dir: str = None):
        """更新路径配置"""
        if output_dir:
            self.output_base_dir = output_dir
        if cache_dir:
            self.cache_dir = cache_dir
            
        # 更新所有模型的缓存目录
        for model in self.models:
            model.cache_dir = self.cache_dir
    
    def validate_config(self) -> bool:
        """验证配置的有效性"""
        try:
            # 检查输出目录
            Path(self.output_base_dir).mkdir(parents=True, exist_ok=True)
            
            # 检查缓存目录
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            
            # 检查模型配置
            if not self.models:
                raise ValueError("至少需要配置一个模型")
            
            return True
        except Exception as e:
            print(f"配置验证失败: {e}")
            return False
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'output_base_dir': self.output_base_dir,
            'cache_dir': self.cache_dir,
            'models': [{'name': m.name, 'device': m.device} for m in self.models],
            'evaluation': {
                'batch_size': self.evaluation.batch_size,
                'skip_existing': self.evaluation.skip_existing,
                'process_timeout': self.evaluation.process_timeout
            },
            'hardware': {
                'max_gpu_workers': self.hardware.max_gpu_workers,
                'device': self.hardware.device
            },
            'metrics_config': self.metrics_config
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """从字典创建配置"""
        instance = cls(
            output_base_dir=config_dict.get('output_base_dir'),
            cache_dir=config_dict.get('cache_dir')
        )
        
        # 更新评测配置
        if 'evaluation' in config_dict:
            eval_config = config_dict['evaluation']
            instance.evaluation.batch_size = eval_config.get('batch_size')
            instance.evaluation.skip_existing = eval_config.get('skip_existing', True)
            instance.evaluation.process_timeout = eval_config.get('process_timeout', 600)
        
        # 更新硬件配置
        if 'hardware' in config_dict:
            hw_config = config_dict['hardware']
            instance.hardware.max_gpu_workers = hw_config.get('max_gpu_workers', 4)
            instance.hardware.device = hw_config.get('device', 'cuda')
        
        # 更新指标配置
        if 'metrics_config' in config_dict:
            instance.metrics_config.update(config_dict['metrics_config'])
        
        return instance


# 默认配置实例
DEFAULT_CONFIG = BatchEvaluationSettings()

# 预定义的配置模板
PRESET_CONFIGS = {
    'fast': BatchEvaluationSettings(),
    'comprehensive': BatchEvaluationSettings(),
    'debug': BatchEvaluationSettings()
}

# 为预设配置设置不同的参数
PRESET_CONFIGS['fast'].evaluation.batch_size = 10
PRESET_CONFIGS['fast'].evaluation.save_visualizations = False
PRESET_CONFIGS['fast'].models = [ModelConfig("vit_t_lm")]

PRESET_CONFIGS['debug'].evaluation.batch_size = 2
PRESET_CONFIGS['debug'].evaluation.process_timeout = 60
PRESET_CONFIGS['debug'].hardware.max_gpu_workers = 1