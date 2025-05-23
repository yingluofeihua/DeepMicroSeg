"""
LoRA适配器实现
支持对SAM模型的各个组件添加LoRA适配器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math
from pathlib import Path
import json


class LoRALayer(nn.Module):
    """LoRA适配器层"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        bias: bool = False
    ):
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA权重
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 初始化
        self.reset_parameters()
    
    def reset_parameters(self):
        """重置参数"""
        # A使用正态分布初始化
        nn.init.normal_(self.lora_A.weight, std=1/self.rank)
        # B初始化为0，确保开始时LoRA输出为0
        nn.init.zeros_(self.lora_B.weight)
        if self.lora_B.bias is not None:
            nn.init.zeros_(self.lora_B.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.lora_B(self.dropout(self.lora_A(x))) * self.scaling


# 在 lora/adapters.py 中修复 LoRALinear 类

class LoRALinear(nn.Module):
    """带LoRA的线性层 - 修复设备一致性"""
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 保存原始层（冻结）
        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # 获取原始层的设备
        device = next(self.original_layer.parameters()).device
        
        # LoRA适配器 - 确保在正确设备上
        self.lora = LoRALayer(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=original_layer.bias is not None
        ).to(device)  # 🔧 确保LoRA在正确设备上
        
        # 是否启用LoRA
        self.enable_lora = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 - 确保设备一致性"""
        # 确保LoRA层在与输入相同的设备上
        if self.lora.lora_A.weight.device != x.device:
            self.lora = self.lora.to(x.device)
        
        # 原始层输出
        output = self.original_layer(x)
        
        # 添加LoRA输出
        if self.enable_lora:
            lora_output = self.lora(x)
            output = output + lora_output
        
        return output
    
    def merge_weights(self):
        """将LoRA权重合并到原始权重中 - 修复设备一致性"""
        if not self.enable_lora:
            return
        
        # 确保所有张量在同一设备上
        device = self.original_layer.weight.device
        
        # 将LoRA权重移动到正确设备
        self.lora = self.lora.to(device)
        
        # 计算LoRA权重
        lora_weight = self.lora.lora_B.weight @ self.lora.lora_A.weight * self.lora.scaling
        
        # 合并到原始权重
        self.original_layer.weight.data += lora_weight
        
        if self.original_layer.bias is not None and self.lora.lora_B.bias is not None:
            self.original_layer.bias.data += self.lora.lora_B.bias.data
        
        # 禁用LoRA
        self.enable_lora = False
    
    def unmerge_weights(self):
        """从原始权重中分离LoRA权重 - 修复设备一致性"""
        if self.enable_lora:
            return
        
        # 确保所有张量在同一设备上
        device = self.original_layer.weight.device
        
        # 将LoRA权重移动到正确设备
        self.lora = self.lora.to(device)
        
        # 计算LoRA权重
        lora_weight = self.lora.lora_B.weight @ self.lora.lora_A.weight * self.lora.scaling
        
        # 从原始权重中减去
        self.original_layer.weight.data -= lora_weight
        
        if self.original_layer.bias is not None and self.lora.lora_B.bias is not None:
            self.original_layer.bias.data -= self.lora.lora_B.bias.data
        
        # 启用LoRA
        self.enable_lora = True


class LoRAMultiheadAttention(nn.Module):
    """带LoRA的多头注意力机制"""
    
    def __init__(
        self,
        original_attention: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        target_modules: List[str] = None
    ):
        super().__init__()
        
        self.original_attention = original_attention
        
        # 默认目标模块
        if target_modules is None:
            target_modules = ["query", "key", "value", "dense"]
        
        # 为指定的模块添加LoRA
        self.lora_modules = {}
        for name, module in original_attention.named_modules():
            if isinstance(module, nn.Linear):
                module_name = name.split('.')[-1]
                if any(target in module_name.lower() for target in target_modules):
                    self.lora_modules[name] = LoRALinear(
                        module, rank=rank, alpha=alpha, dropout=dropout
                    )
        
        # 注册LoRA模块
        for name, lora_module in self.lora_modules.items():
            self.add_module(f"lora_{name.replace('.', '_')}", lora_module)
    
    def forward(self, *args, **kwargs):
        """前向传播 - 这里需要根据具体的注意力机制实现"""
        # 临时替换原始模块
        original_modules = {}
        for name, lora_module in self.lora_modules.items():
            original_modules[name] = self._get_submodule(self.original_attention, name)
            self._set_submodule(self.original_attention, name, lora_module)
        
        try:
            # 执行原始注意力计算
            output = self.original_attention(*args, **kwargs)
        finally:
            # 恢复原始模块
            for name, original_module in original_modules.items():
                self._set_submodule(self.original_attention, name, original_module)
        
        return output
    
    def _get_submodule(self, module: nn.Module, path: str) -> nn.Module:
        """获取子模块"""
        for name in path.split('.'):
            module = getattr(module, name)
        return module
    
    def _set_submodule(self, module: nn.Module, path: str, value: nn.Module):
        """设置子模块"""
        *path_parts, name = path.split('.')
        for part in path_parts:
            module = getattr(module, part)
        setattr(module, name, value)


class LoRAModelWrapper(nn.Module):
    """LoRA模型包装器"""
    
    def __init__(
        self,
        base_model: nn.Module,
        config: Dict[str, Any]
    ):
        super().__init__()
        
        self.base_model = base_model
        self.config = config
        self.lora_modules = {}
        
        # 冻结基础模型
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 添加LoRA适配器
        self._add_lora_adapters()
    
    def _add_lora_adapters(self):
        """添加LoRA适配器到模型"""
        rank = self.config.get('rank', 8)
        alpha = self.config.get('alpha', 16.0)
        dropout = self.config.get('dropout', 0.1)
        target_modules = self.config.get('target_modules', [])
        
        # 遍历模型的所有模块
        for name, module in self.base_model.named_modules():
            if self._should_add_lora(name, module, target_modules):
                if isinstance(module, nn.Linear):
                    lora_module = LoRALinear(
                        module, rank=rank, alpha=alpha, dropout=dropout
                    )
                    self.lora_modules[name] = lora_module
                    self._replace_module(name, lora_module)
                
                elif hasattr(module, 'query') or hasattr(module, 'attention'):
                    # 注意力模块
                    lora_module = LoRAMultiheadAttention(
                        module, rank=rank, alpha=alpha, dropout=dropout
                    )
                    self.lora_modules[name] = lora_module
                    self._replace_module(name, lora_module)
        
        print(f"添加了 {len(self.lora_modules)} 个LoRA适配器")
    
    def _should_add_lora(self, name: str, module: nn.Module, target_modules: List[str]) -> bool:
        """判断是否应该为该模块添加LoRA"""
        if not target_modules:
            # 默认策略：为所有线性层和注意力层添加LoRA
            return isinstance(module, nn.Linear) or 'attention' in name.lower()
        
        # 检查模块名是否匹配目标模块
        return any(target in name.lower() for target in target_modules)
    
    def _replace_module(self, name: str, new_module: nn.Module):
        """替换模块"""
        *path_parts, module_name = name.split('.')
        parent = self.base_model
        
        for part in path_parts:
            parent = getattr(parent, part)
        
        setattr(parent, module_name, new_module)
    
    def forward(self, *args, **kwargs):
        """前向传播"""
        return self.base_model(*args, **kwargs)
    
    def save_lora_weights(self, save_path: str):
        """保存LoRA权重"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存LoRA权重
        lora_state_dict = {}
        for name, module in self.lora_modules.items():
            if hasattr(module, 'lora'):
                lora_state_dict[f"{name}.lora_A.weight"] = module.lora.lora_A.weight
                lora_state_dict[f"{name}.lora_B.weight"] = module.lora.lora_B.weight
                if module.lora.lora_B.bias is not None:
                    lora_state_dict[f"{name}.lora_B.bias"] = module.lora.lora_B.bias
        
        torch.save(lora_state_dict, save_path / "lora_weights.pth")
        
        # 保存配置
        with open(save_path / "lora_config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"LoRA权重已保存到: {save_path}")
    
    def load_lora_weights(self, load_path: str):
        """加载LoRA权重"""
        load_path = Path(load_path)
        
        # 加载权重
        weights_file = load_path / "lora_weights.pth"
        if weights_file.exists():
            lora_state_dict = torch.load(weights_file, map_location='cpu')
            
            # 加载权重到对应模块
            for name, module in self.lora_modules.items():
                if hasattr(module, 'lora'):
                    if f"{name}.lora_A.weight" in lora_state_dict:
                        module.lora.lora_A.weight.data = lora_state_dict[f"{name}.lora_A.weight"]
                    if f"{name}.lora_B.weight" in lora_state_dict:
                        module.lora.lora_B.weight.data = lora_state_dict[f"{name}.lora_B.weight"]
                    if f"{name}.lora_B.bias" in lora_state_dict:
                        module.lora.lora_B.bias.data = lora_state_dict[f"{name}.lora_B.bias"]
            
            print(f"LoRA权重已从 {load_path} 加载")
        else:
            print(f"警告: 未找到LoRA权重文件 {weights_file}")
    
    def merge_and_save(self, save_path: str):
        """合并LoRA权重并保存完整模型"""
        # 合并所有LoRA权重
        for module in self.lora_modules.values():
            if hasattr(module, 'merge_weights'):
                module.merge_weights()
        
        # 保存合并后的模型
        torch.save(self.base_model.state_dict(), save_path)
        print(f"合并的模型已保存到: {save_path}")
        
        # 恢复LoRA状态
        for module in self.lora_modules.values():
            if hasattr(module, 'unmerge_weights'):
                module.unmerge_weights()
    
    def get_trainable_parameters(self) -> int:
        """获取可训练参数数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'trainable_percentage': 100 * trainable_params / total_params
        }
    
    def print_model_info(self):
        """打印模型信息"""
        info = self.get_trainable_parameters()
        print(f"模型参数统计:")
        print(f"  总参数数: {info['total_parameters']:,}")
        print(f"  可训练参数数: {info['trainable_parameters']:,}")
        print(f"  可训练参数比例: {info['trainable_percentage']:.2f}%")
        print(f"  LoRA模块数: {len(self.lora_modules)}")


def create_lora_model(base_model: nn.Module, config: Dict[str, Any]) -> LoRAModelWrapper:
    """创建LoRA模型的便捷函数"""
    return LoRAModelWrapper(base_model, config)


def load_lora_model(base_model: nn.Module, lora_path: str) -> LoRAModelWrapper:
    """加载已训练的LoRA模型"""
    lora_path = Path(lora_path)
    
    # 加载LoRA配置
    config_file = lora_path / "lora_config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        # 使用默认配置
        config = {'rank': 8, 'alpha': 16.0, 'dropout': 0.1}
    
    # 创建LoRA模型
    lora_model = LoRAModelWrapper(base_model, config)
    
    # 加载权重
    lora_model.load_lora_weights(lora_path)
    
    return lora_model