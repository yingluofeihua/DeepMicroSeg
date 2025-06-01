"""
SAM专用LoRA包装器
针对SAM架构优化的LoRA实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path

from .adapters import LoRALayer, LoRALinear
from core.sam_model_loader import SAMModelLoader


class SAMLoRAWrapper(nn.Module):
    """SAM专用LoRA包装器"""
    
    def __init__(
        self,
        sam_model_loader: SAMModelLoader,
        lora_config: Dict[str, Any]
    ):
        super().__init__()
        
        self.sam_loader = sam_model_loader
        self.config = lora_config
        self.lora_modules = {}
        
        # 获取SAM组件
        self.image_encoder = sam_model_loader.image_encoder
        self.prompt_encoder = sam_model_loader.prompt_encoder
        self.mask_decoder = sam_model_loader.mask_decoder
        
        # 添加LoRA适配器
        self._add_lora_to_sam()
        
        # 设置训练模式
        self._setup_training_mode()
    
    def _add_lora_to_sam(self):
        """为SAM的各个组件添加LoRA适配器"""
        apply_lora_to = self.config.get('apply_lora_to', ['image_encoder'])
        
        print(f"将LoRA应用到: {apply_lora_to}")
        
        if 'image_encoder' in apply_lora_to and self.image_encoder is not None:
            self._add_lora_to_image_encoder()
        
        if 'prompt_encoder' in apply_lora_to and self.prompt_encoder is not None:
            self._add_lora_to_prompt_encoder()
        
        if 'mask_decoder' in apply_lora_to and self.mask_decoder is not None:
            self._add_lora_to_mask_decoder()
        
        print(f"总计添加了 {len(self.lora_modules)} 个LoRA模块")
    
    def _add_lora_to_image_encoder(self):
        """为图像编码器添加LoRA"""
        target_modules = self.config.get('target_modules', [
            'qkv', 'proj', 'mlp', 'fc1', 'fc2'  # 更通用的目标模块
        ])
        
        lora_count = 0
        
        print(f"图像编码器中搜索目标模块: {target_modules}")
        
        # 先打印所有线性层的名称用于调试
        linear_modules = []
        for name, module in self.image_encoder.named_modules():
            if isinstance(module, nn.Linear):
                linear_modules.append(name)
        
        print(f"图像编码器中的所有线性层: {linear_modules[:10]}...")  # 只显示前10个
        
        for name, module in self.image_encoder.named_modules():
            if isinstance(module, nn.Linear):
                # 检查是否匹配目标模块
                if self._should_add_lora_to_module(name, target_modules):
                    lora_module = LoRALinear(
                        module,
                        rank=self.config.get('rank', 8),
                        alpha=self.config.get('alpha', 16.0),
                        dropout=self.config.get('dropout', 0.1)
                    )
                    
                    # 替换模块
                    self._replace_module_in_image_encoder(name, lora_module)
                    self.lora_modules[f'image_encoder.{name}'] = lora_module
                    lora_count += 1
                    print(f"  为 {name} 添加LoRA适配器")
        
        print(f"图像编码器添加了 {lora_count} 个LoRA模块")
    
    def _add_lora_to_prompt_encoder(self):
        """为提示编码器添加LoRA"""
        # 提示编码器通常较小，可以为所有线性层添加LoRA
        lora_count = 0
        for name, module in self.prompt_encoder.named_modules():
            if isinstance(module, nn.Linear):
                lora_module = LoRALinear(
                    module,
                    rank=self.config.get('rank', 4),  # 提示编码器使用更小的rank
                    alpha=self.config.get('alpha', 8.0),
                    dropout=self.config.get('dropout', 0.1)
                )
                
                self._replace_module_in_prompt_encoder(name, lora_module)
                self.lora_modules[f'prompt_encoder.{name}'] = lora_module
                lora_count += 1
        
        print(f"提示编码器添加了 {lora_count} 个LoRA模块")
    
    def _add_lora_to_mask_decoder(self):
        """为掩码解码器添加LoRA"""
        target_modules = self.config.get('target_modules', [
            'transformer', 'iou_prediction_head', 'mask_tokens'
        ])
        
        lora_count = 0
        for name, module in self.mask_decoder.named_modules():
            if isinstance(module, nn.Linear):
                if self._should_add_lora_to_module(name, target_modules):
                    lora_module = LoRALinear(
                        module,
                        rank=self.config.get('rank', 8),
                        alpha=self.config.get('alpha', 16.0),
                        dropout=self.config.get('dropout', 0.1)
                    )
                    
                    self._replace_module_in_mask_decoder(name, lora_module)
                    self.lora_modules[f'mask_decoder.{name}'] = lora_module
                    lora_count += 1
        
        print(f"掩码解码器添加了 {lora_count} 个LoRA模块")
    
    def _should_add_lora_to_module(self, module_name: str, target_modules: List[str]) -> bool:
        """判断是否应该为该模块添加LoRA"""
        if not target_modules:
            return True  # 如果没有指定目标模块，则为所有线性层添加
        
        return any(target in module_name for target in target_modules)
    
    def _replace_module_in_image_encoder(self, module_path: str, new_module: nn.Module):
        """在图像编码器中替换模块"""
        self._replace_module(self.image_encoder, module_path, new_module)
    
    def _replace_module_in_prompt_encoder(self, module_path: str, new_module: nn.Module):
        """在提示编码器中替换模块"""
        self._replace_module(self.prompt_encoder, module_path, new_module)
    
    def _replace_module_in_mask_decoder(self, module_path: str, new_module: nn.Module):
        """在掩码解码器中替换模块"""
        self._replace_module(self.mask_decoder, module_path, new_module)
    
    def _replace_module(self, parent_module: nn.Module, module_path: str, new_module: nn.Module):
        """在父模块中替换指定路径的子模块"""
        path_parts = module_path.split('.')
        current_module = parent_module
        
        # 导航到父模块
        for part in path_parts[:-1]:
            current_module = getattr(current_module, part)
        
        # 替换最后一级模块
        setattr(current_module, path_parts[-1], new_module)
    
    def _setup_training_mode(self):
        """设置训练模式"""
        # 根据配置冻结组件
        if self.config.get('freeze_image_encoder', True):
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            # 但是要确保LoRA参数可训练
            for name, module in self.lora_modules.items():
                if 'image_encoder' in name and hasattr(module, 'lora'):
                    for param in module.lora.parameters():
                        param.requires_grad = True
        
        if self.config.get('freeze_prompt_encoder', True):
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False
            for name, module in self.lora_modules.items():
                if 'prompt_encoder' in name and hasattr(module, 'lora'):
                    for param in module.lora.parameters():
                        param.requires_grad = True
        
        if self.config.get('freeze_mask_decoder', False):
            for param in self.mask_decoder.parameters():
                param.requires_grad = False
            for name, module in self.lora_modules.items():
                if 'mask_decoder' in name and hasattr(module, 'lora'):
                    for param in module.lora.parameters():
                        param.requires_grad = True
    
    def forward(self, batch_inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """前向传播 - 512×512版本"""
        try:
            # 提取输入并获取设备
            images = batch_inputs['images']  # ✅ [B, 3, 512, 512]
            if images.dim() == 3:                  # 单样本但没 batch 维
                images = images.unsqueeze(0)       # 变成 [1,3,H,W]
            device = images.device
            # print(f"images: {images.shape}")
            
            # print(f"SAM输入图像尺寸: {images.shape}")  # [B, 3, 512, 512]
            
            # 🔧 关键修复：SAM原本期望1024×1024，但你使用512×512
            # 有两种处理方式：
            
            # 方式1：上采样到1024×1024（推荐）
            if images.shape[-1] != 1024 or images.shape[-2] != 1024:
                # print(f"上采样图像从 {images.shape[-2:]} 到 (1024, 1024)")
                images = F.interpolate(
                    images, 
                    size=(1024, 1024), 
                    mode='bilinear', 
                    align_corners=False
                )
                # print(f"上采样后图像尺寸: {images.shape}")  # [B, 3, 1024, 1024]
            # print(f"images11111: {images.shape}")
            # 🔧 确保所有模型组件都在正确设备上
            self._ensure_models_on_device(device)
            
            # 图像编码
            try:
                image_embeddings = self.image_encoder(images)  # 输入 [B, 3, 1024, 1024]
                # print(f"图像编码输出: {image_embeddings.shape}")  # [B, 256, 64, 64]
            except Exception as e:
                print(f"图像编码失败: {e}")
                raise
            
            # 批量处理每个样本
            batch_outputs = []
            batch_size = images.shape[0]
            
            for i in range(batch_size):
                try:
                    single_image_embedding = image_embeddings[i:i+1]
                    
                    # 提示编码
                    sparse_embeddings, dense_embeddings = self._encode_prompts(
                        batch_inputs.get('point_coords', []), 
                        batch_inputs.get('point_labels', []), 
                        batch_inputs.get('boxes', []), 
                        batch_inputs.get('mask_inputs', None), 
                        i, device
                    )
                    
                    # 获取位置编码
                    try:
                        image_pe = self.prompt_encoder.get_dense_pe().to(device)
                    except:
                        image_pe = torch.zeros(1, 256, 64, 64, device=device)
                    
                    # 掩码解码
                    low_res_masks, iou_predictions = self.mask_decoder(
                        image_embeddings=single_image_embedding,
                        image_pe=image_pe,
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=batch_inputs.get('multimask_output', False)
                    )
                    
                    # print(f"掩码解码输出: {low_res_masks.shape}")  # [1, 1, 256, 256]
                    
                    # 🔧 关键：将256×256的输出调整到512×512
                    if low_res_masks.shape[-1] != 512:
                        # print(f"调整掩码从 {low_res_masks.shape[-2:]} 到 (512, 512)")
                        low_res_masks = F.interpolate(
                            low_res_masks,
                            size=(512, 512),
                            mode='bilinear',
                            align_corners=False
                        )
                        # print(f"调整后掩码尺寸: {low_res_masks.shape}")  # [1, 1, 512, 512]
                    
                    batch_outputs.append({
                        'masks': low_res_masks,      # ✅ [1, 1, 512, 512]
                        'iou_predictions': iou_predictions
                    })
                    
                except Exception as e:
                    print(f"处理样本 {i} 失败: {e}")
                    # 返回512×512的默认输出
                    batch_outputs.append({
                        'masks': torch.zeros(1, 1, 512, 512, device=device),
                        'iou_predictions': torch.zeros(1, 1, device=device)
                    })
            
            # 合并批量输出
            result = self._merge_batch_outputs(batch_outputs)
            # print(f"最终输出掩码尺寸: {result['masks'].shape}")  # [B, 1, 512, 512]
            
            return result
            
        except Exception as e:
            print(f"SAM forward传播异常: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回512×512的默认输出
            device = batch_inputs['images'].device
            batch_size = batch_inputs['images'].shape[0]
            return {
                'masks': torch.zeros(batch_size, 1, 512, 512, device=device),
                'iou_predictions': torch.zeros(batch_size, 1, device=device)
            }
        
    def _ensure_models_on_device(self, device: torch.device):
        """确保所有模型组件都在指定设备上"""
        
        # 移动主要组件
        if self.image_encoder is not None:
            self.image_encoder = self.image_encoder.to(device)
        
        if self.prompt_encoder is not None:
            self.prompt_encoder = self.prompt_encoder.to(device)
        
        if self.mask_decoder is not None:
            self.mask_decoder = self.mask_decoder.to(device)
        
        # 🔧 移动所有LoRA模块
        for name, lora_module in self.lora_modules.items():
            if hasattr(lora_module, 'lora'):
                lora_module.lora = lora_module.lora.to(device)
            lora_module = lora_module.to(device)

    def _encode_prompts(self, point_coords, point_labels, boxes, mask_inputs, batch_idx, device):
        """编码提示信息 - 确保设备一致性"""
        
        # 处理点提示
        points = None
        if (isinstance(point_coords, list) and batch_idx < len(point_coords) and 
            isinstance(point_coords[batch_idx], torch.Tensor) and len(point_coords[batch_idx]) > 0):
            
            batch_point_coords = point_coords[batch_idx].to(device)
            batch_point_labels = (point_labels[batch_idx].to(device) 
                                if isinstance(point_labels, list) and batch_idx < len(point_labels) 
                                else None)
            
            if batch_point_labels is not None and len(batch_point_labels) > 0:
                points = (batch_point_coords.unsqueeze(0), batch_point_labels.unsqueeze(0))
            else:
                labels = torch.ones(len(batch_point_coords), dtype=torch.long, device=device)
                points = (batch_point_coords.unsqueeze(0), labels.unsqueeze(0))
        
        # 处理框提示
        box = None
        if (isinstance(boxes, list) and batch_idx < len(boxes) and 
            isinstance(boxes[batch_idx], torch.Tensor) and len(boxes[batch_idx]) > 0):
            
            batch_boxes = boxes[batch_idx].to(device)
            box = batch_boxes[0].unsqueeze(0)
        
        # 处理掩码提示
        mask = None
        if (isinstance(mask_inputs, list) and batch_idx < len(mask_inputs) and 
            mask_inputs[batch_idx] is not None):
            mask = mask_inputs[batch_idx].to(device)
        
        # 使用提示编码器编码
        try:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=box,
                masks=mask
            )
        except Exception as e:
            print(f"提示编码失败: {e}")
            # 创建默认的空提示编码
            sparse_embeddings = torch.zeros(1, 0, 256, device=device)
            dense_embeddings = torch.zeros(1, 256, 64, 64, device=device)
        
        return sparse_embeddings, dense_embeddings
    
    def _merge_batch_outputs(self, batch_outputs: List[Dict]) -> Dict[str, torch.Tensor]:
        """合并批量输出"""
        if not batch_outputs:
            return {'masks': torch.empty(0), 'iou_predictions': torch.empty(0)}
        
        # 合并掩码
        masks = torch.cat([output['masks'] for output in batch_outputs], dim=0)
        iou_predictions = torch.cat([output['iou_predictions'] for output in batch_outputs], dim=0)
        
        return {
            'masks': masks,
            'iou_predictions': iou_predictions
        }
    
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
        
        torch.save(lora_state_dict, save_path / "sam_lora_weights.pth")
        
        # 保存配置
        config_to_save = self.config.copy()
        config_to_save['model_type'] = self.sam_loader.model_type
        
        with open(save_path / "sam_lora_config.json", 'w') as f:
            json.dump(config_to_save, f, indent=2)
        
        print(f"SAM LoRA权重已保存到: {save_path}")
    
    def load_lora_weights(self, load_path: str):
        """加载LoRA权重"""
        load_path = Path(load_path)
        
        # 加载权重
        weights_file = load_path / "sam_lora_weights.pth"
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
            
            print(f"SAM LoRA权重已从 {load_path} 加载")
        else:
            print(f"警告: 未找到LoRA权重文件 {weights_file}")
    
    def merge_and_save_full_model(self, save_path: str):
        """合并LoRA权重并保存完整SAM模型"""
        # 合并所有LoRA权重
        for module in self.lora_modules.values():
            if hasattr(module, 'merge_weights'):
                module.merge_weights()
        
        # 保存合并后的SAM模型
        self.sam_loader.save_model_state(save_path)
        print(f"合并的SAM模型已保存到: {save_path}")
        
        # 恢复LoRA状态
        for module in self.lora_modules.values():
            if hasattr(module, 'unmerge_weights'):
                module.unmerge_weights()
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        """获取可训练参数统计"""
        total_params = 0
        trainable_params = 0
        lora_params = 0
        
        # 统计所有参数
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                if 'lora' in name.lower():
                    lora_params += param.numel()
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'lora_parameters': lora_params,
            'trainable_percentage': 100 * trainable_params / total_params if total_params > 0 else 0,
            'lora_percentage': 100 * lora_params / total_params if total_params > 0 else 0
        }
    
    def print_model_info(self):
        """打印模型信息"""
        info = self.get_trainable_parameters()
        print(f"\nSAM LoRA模型信息:")
        print(f"  总参数数: {info['total_parameters']:,}")
        print(f"  可训练参数数: {info['trainable_parameters']:,}")
        print(f"  LoRA参数数: {info['lora_parameters']:,}")
        print(f"  可训练参数比例: {info['trainable_percentage']:.2f}%")
        print(f"  LoRA参数比例: {info['lora_percentage']:.2f}%")
        print(f"  LoRA模块数: {len(self.lora_modules)}")
        
        # 按组件分组显示LoRA模块
        component_counts = {}
        for name in self.lora_modules.keys():
            component = name.split('.')[0]
            component_counts[component] = component_counts.get(component, 0) + 1
        
        print(f"  LoRA模块分布:")
        for component, count in component_counts.items():
            print(f"    {component}: {count} 个模块")


def create_sam_lora_model(model_type: str, lora_config: Dict[str, Any], device: str = "cuda") -> Optional[SAMLoRAWrapper]:
    """创建SAM LoRA模型的便捷函数"""
    try:
        # 加载SAM模型
        from core.sam_model_loader import load_sam_for_training
        
        sam_loader = load_sam_for_training(
            model_type=model_type,
            device=device,
            freeze_image_encoder=lora_config.get('freeze_image_encoder', True),
            freeze_prompt_encoder=lora_config.get('freeze_prompt_encoder', True),
            freeze_mask_decoder=lora_config.get('freeze_mask_decoder', False)
        )
        
        if sam_loader is None:
            print("SAM模型加载失败")
            return None
        
        # 创建LoRA包装器
        sam_lora_model = SAMLoRAWrapper(sam_loader, lora_config)
        
        return sam_lora_model
        
    except Exception as e:
        print(f"创建SAM LoRA模型失败: {e}")
        return None


def load_sam_lora_model(model_type: str, lora_path: str, device: str = "cuda") -> Optional[SAMLoRAWrapper]:
    """加载已训练的SAM LoRA模型"""
    try:
        lora_path = Path(lora_path)
        
        # 加载LoRA配置
        config_file = lora_path / "sam_lora_config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                lora_config = json.load(f)
        else:
            # 使用默认配置
            lora_config = {
                'rank': 8, 
                'alpha': 16.0, 
                'dropout': 0.1,
                'apply_lora_to': ['image_encoder']
            }
            print("使用默认LoRA配置")
        
        # 创建SAM LoRA模型
        sam_lora_model = create_sam_lora_model(model_type, lora_config, device)
        
        if sam_lora_model is None:
            return None
        
        # 加载LoRA权重
        sam_lora_model.load_lora_weights(lora_path)
        
        return sam_lora_model
        
    except Exception as e:
        print(f"加载SAM LoRA模型失败: {e}")
        return None