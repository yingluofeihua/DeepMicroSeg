"""
SAM原始模型加载器
专门用于训练，加载SAM的原始PyTorch组件
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import os

# 初始化导入标志
SEGMENT_ANYTHING_AVAILABLE = False
MICRO_SAM_AVAILABLE = False

# 尝试导入segment_anything
try:
    from segment_anything import build_sam_vit_h, build_sam_vit_l, build_sam_vit_b
    from segment_anything.modeling import Sam
    SEGMENT_ANYTHING_AVAILABLE = True
except ImportError:
    pass

# 尝试导入micro_sam
try:
    from micro_sam.models import get_sam_model
    from micro_sam.util import get_sam_model_cls
    MICRO_SAM_AVAILABLE = True
except ImportError:
    pass


class SAMModelLoader:
    """SAM模型加载器 - 专门用于获取可训练的PyTorch模型"""
    
    def __init__(self, model_type: str = "vit_b_lm", device: str = "cuda"):
        self.model_type = model_type
        self.device = device
        self.model = None
        self.image_encoder = None
        self.prompt_encoder = None
        self.mask_decoder = None
        
    def load_model(self) -> bool:
        """加载SAM模型的各个组件"""
        try:
            print(f"正在加载SAM训练模型: {self.model_type}")
            
            # 尝试不同的加载方法
            if MICRO_SAM_AVAILABLE:
                success = self._load_from_micro_sam()
            elif SEGMENT_ANYTHING_AVAILABLE:
                success = self._load_from_segment_anything()
            else:
                print("错误: 无法找到SAM模型库")
                return False
            
            if success:
                self._move_to_device()
                self._print_model_info()
                return True
            else:
                return False
                
        except Exception as e:
            print(f"加载SAM模型失败: {e}")
            return False
    
    def _load_from_micro_sam(self) -> bool:
        """从micro_sam加载模型"""
        try:
            from micro_sam.util import get_sam_model
            
            # micro_sam的模型名称映射
            model_name_map = {
                "vit_t_lm": "vit_t_lm",
                "vit_b_lm": "vit_b_lm", 
                "vit_l_lm": "vit_l_lm"
            }
            
            if self.model_type not in model_name_map:
                print(f"不支持的模型类型: {self.model_type}")
                return False
            
            # 加载完整的SAM模型
            self.model = get_sam_model(
                model_type=model_name_map[self.model_type],
                device=self.device,
                return_sam=True  # 返回完整的SAM模型而不是预测器
            )
            
            # 提取各个组件
            if hasattr(self.model, 'image_encoder'):
                self.image_encoder = self.model.image_encoder
            if hasattr(self.model, 'prompt_encoder'):
                self.prompt_encoder = self.model.prompt_encoder
            if hasattr(self.model, 'mask_decoder'):
                self.mask_decoder = self.model.mask_decoder
                
            return True
            
        except Exception as e:
            print(f"从micro_sam加载失败: {e}")
            return False
    
    def _load_from_segment_anything(self) -> bool:
        """从segment_anything加载模型"""
        try:
            # segment_anything的模型构建函数映射
            model_builders = {
                "vit_b_lm": build_sam_vit_b,
                "vit_l_lm": build_sam_vit_l, 
                "vit_h_lm": build_sam_vit_h
            }
            
            # 如果是微调模型，使用vit_b作为基础
            if self.model_type not in model_builders:
                if "vit_b" in self.model_type:
                    builder = build_sam_vit_b
                elif "vit_l" in self.model_type:
                    builder = build_sam_vit_l
                else:
                    builder = build_sam_vit_b  # 默认
            else:
                builder = model_builders[self.model_type]
            
            # 构建模型
            self.model = builder()
            
            # 提取各个组件
            self.image_encoder = self.model.image_encoder
            self.prompt_encoder = self.model.prompt_encoder
            self.mask_decoder = self.model.mask_decoder
            
            return True
            
        except Exception as e:
            print(f"从segment_anything加载失败: {e}")
            return False
    
    def _move_to_device(self):
        """将模型移动到指定设备"""
        if self.model is not None:
            self.model = self.model.to(self.device)
        
        if self.image_encoder is not None:
            self.image_encoder = self.image_encoder.to(self.device)
        
        if self.prompt_encoder is not None:
            self.prompt_encoder = self.prompt_encoder.to(self.device)
        
        if self.mask_decoder is not None:
            self.mask_decoder = self.mask_decoder.to(self.device)
    
    def _print_model_info(self):
        """打印模型信息"""
        print(f"SAM模型加载成功:")
        print(f"  模型类型: {self.model_type}")
        print(f"  设备: {self.device}")
        
        if self.image_encoder is not None:
            total_params = sum(p.numel() for p in self.image_encoder.parameters())
            trainable_params = sum(p.numel() for p in self.image_encoder.parameters() if p.requires_grad)
            print(f"  图像编码器参数: {total_params:,} (可训练: {trainable_params:,})")
        
        if self.prompt_encoder is not None:
            total_params = sum(p.numel() for p in self.prompt_encoder.parameters())
            print(f"  提示编码器参数: {total_params:,}")
        
        if self.mask_decoder is not None:
            total_params = sum(p.numel() for p in self.mask_decoder.parameters())
            print(f"  掩码解码器参数: {total_params:,}")
    
    def get_trainable_components(self) -> Dict[str, nn.Module]:
        """获取可训练的组件"""
        components = {}
        
        if self.image_encoder is not None:
            components['image_encoder'] = self.image_encoder
        
        if self.prompt_encoder is not None:
            components['prompt_encoder'] = self.prompt_encoder
            
        if self.mask_decoder is not None:
            components['mask_decoder'] = self.mask_decoder
        
        return components
    
    def freeze_components(self, components_to_freeze: list):
        """冻结指定组件"""
        component_map = {
            'image_encoder': self.image_encoder,
            'prompt_encoder': self.prompt_encoder,
            'mask_decoder': self.mask_decoder
        }
        
        for component_name in components_to_freeze:
            if component_name in component_map and component_map[component_name] is not None:
                for param in component_map[component_name].parameters():
                    param.requires_grad = False
                print(f"已冻结组件: {component_name}")
    
    def unfreeze_components(self, components_to_unfreeze: list):
        """解冻指定组件"""
        component_map = {
            'image_encoder': self.image_encoder,
            'prompt_encoder': self.prompt_encoder,
            'mask_decoder': self.mask_decoder
        }
        
        for component_name in components_to_unfreeze:
            if component_name in component_map and component_map[component_name] is not None:
                for param in component_map[component_name].parameters():
                    param.requires_grad = True
                print(f"已解冻组件: {component_name}")
    
    def save_model_state(self, save_path: str):
        """保存模型状态"""
        if self.model is None:
            print("模型未加载，无法保存")
            return False
        
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            state_dict = {
                'model_type': self.model_type,
                'model_state_dict': self.model.state_dict(),
                'image_encoder_state_dict': self.image_encoder.state_dict() if self.image_encoder else None,
                'prompt_encoder_state_dict': self.prompt_encoder.state_dict() if self.prompt_encoder else None,
                'mask_decoder_state_dict': self.mask_decoder.state_dict() if self.mask_decoder else None
            }
            
            torch.save(state_dict, save_path)
            print(f"模型状态已保存到: {save_path}")
            return True
            
        except Exception as e:
            print(f"保存模型状态失败: {e}")
            return False
    
    def load_model_state(self, load_path: str) -> bool:
        """加载模型状态"""
        try:
            load_path = Path(load_path)
            if not load_path.exists():
                print(f"模型文件不存在: {load_path}")
                return False
            
            checkpoint = torch.load(load_path, map_location=self.device)
            
            if self.model is not None and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if self.image_encoder is not None and 'image_encoder_state_dict' in checkpoint:
                self.image_encoder.load_state_dict(checkpoint['image_encoder_state_dict'])
            
            if self.prompt_encoder is not None and 'prompt_encoder_state_dict' in checkpoint:
                self.prompt_encoder.load_state_dict(checkpoint['prompt_encoder_state_dict'])
            
            if self.mask_decoder is not None and 'mask_decoder_state_dict' in checkpoint:
                self.mask_decoder.load_state_dict(checkpoint['mask_decoder_state_dict'])
            
            print(f"模型状态已从 {load_path} 加载")
            return True
            
        except Exception as e:
            print(f"加载模型状态失败: {e}")
            return False


def create_sam_model_loader(model_type: str, device: str = "cuda") -> SAMModelLoader:
    """创建SAM模型加载器的便捷函数"""
    return SAMModelLoader(model_type, device)


def load_sam_for_training(model_type: str, device: str = "cuda", 
                         freeze_image_encoder: bool = True,
                         freeze_prompt_encoder: bool = True,
                         freeze_mask_decoder: bool = False) -> Optional[SAMModelLoader]:
    """为训练加载SAM模型的便捷函数"""
    loader = create_sam_model_loader(model_type, device)
    
    if not loader.load_model():
        return None
    
    # 根据配置冻结组件
    components_to_freeze = []
    if freeze_image_encoder:
        components_to_freeze.append('image_encoder')
    if freeze_prompt_encoder:
        components_to_freeze.append('prompt_encoder')
    if freeze_mask_decoder:
        components_to_freeze.append('mask_decoder')
    
    if components_to_freeze:
        loader.freeze_components(components_to_freeze)
    
    return loader