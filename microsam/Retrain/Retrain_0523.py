import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import time
import logging
from datetime import datetime, timedelta
from scipy.spatial import distance
from skimage import io, measure
import pandas as pd
import tifffile
from PIL import Image
import cv2
import imageio.v2 as imageio
import torch.distributed as dist
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="imageio")
import wandb
# 导入官方推荐的模块
import torch_em
from torch_em.data.sampler import MinInstanceSampler
from torch_em.transform.label import PerObjectDistanceTransform
import threading

import micro_sam.training as sam_training
from micro_sam.training import train_sam, default_sam_loader
from micro_sam.instance_segmentation import get_predictor_and_decoder, InstanceSegmentationWithDecoder
from micro_sam.automatic_segmentation import automatic_instance_segmentation
from micro_sam.util import export_custom_sam_model
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import numpy as np
from pathlib import Path
import micro_sam.training as sam_training
from micro_sam.training.trainable_sam import TrainableSAM


class ModelParameterAnalyzer:
    """模型参数分析器，用于计算和显示SAM模型的详细参数信息"""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.param_info = {}
        
    def log_info(self, message):
        """统一的日志输出方法"""
        if self.logger:
            self.logger.log_info(message)
        else:
            print(message)
    
    def analyze_sam_model_parameters(self, model_type="vit_b_lm", checkpoint_path=None, 
                                   freeze=None, peft_kwargs=None, device=None) -> Dict:
        """
        分析SAM模型参数
        
        Args:
            model_type: 模型类型 (vit_b, vit_b_lm, vit_l, vit_h等)
            checkpoint_path: 预训练权重路径
            freeze: 冻结的模块列表 ["image_encoder", "prompt_encoder", "mask_decoder"]
            peft_kwargs: PEFT配置（如LoRA参数）
            device: 设备
            
        Returns:
            包含详细参数信息的字典
        """
        self.log_info("="*80)
        self.log_info("开始分析模型参数...")
        
        try:
            # 获取设备
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 创建模型
            self.log_info(f"创建模型: {model_type}")
            if checkpoint_path:
                self.log_info(f"从检查点加载: {checkpoint_path}")
            
            model = sam_training.get_trainable_sam_model(
                model_type=model_type,
                device=device,
                checkpoint_path=checkpoint_path,
                freeze=freeze,
                peft_kwargs=peft_kwargs
            )
            
            # 分析参数
            param_analysis = self._analyze_parameters(model)
            
            # 分析内存使用
            memory_analysis = self._analyze_memory_usage(model, device)
            
            # 分析模型结构
            structure_analysis = self._analyze_model_structure(model)
            
            # 合并所有分析结果
            full_analysis = {
                **param_analysis,
                **memory_analysis,
                **structure_analysis,
                'model_type': model_type,
                'checkpoint_path': str(checkpoint_path) if checkpoint_path else None,
                'freeze_config': freeze,
                'peft_config': peft_kwargs,
                'device': str(device)
            }
            
            # 打印详细信息
            self._print_analysis_results(full_analysis)
            
            self.param_info = full_analysis
            return full_analysis
            
        except Exception as e:
            self.log_info(f"模型参数分析失败: {e}")
            import traceback
            self.log_info(traceback.format_exc())
            return {}
    
    def _analyze_parameters(self, model) -> Dict:
        """分析模型参数数量和类型"""
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        
        # 按模块分析参数
        module_params = {}
        
        # 如果是TrainableSAM，需要访问内部的sam模型
        if isinstance(model, TrainableSAM):
            actual_model = model.sam
        else:
            actual_model = model
        
        # 分析各个组件
        for name, module in actual_model.named_children():
            module_total = 0
            module_trainable = 0
            
            for param in module.parameters():
                param_count = param.numel()
                module_total += param_count
                total_params += param_count
                
                if param.requires_grad:
                    module_trainable += param_count
                    trainable_params += param_count
                else:
                    frozen_params += param_count
            
            module_params[name] = {
                'total': module_total,
                'trainable': module_trainable,
                'frozen': module_total - module_trainable,
                'trainable_ratio': (module_trainable / module_total * 100) if module_total > 0 else 0
            }
        
        # 检查是否有PEFT层（LoRA等）
        peft_params = self._count_peft_parameters(model)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': frozen_params,
            'trainable_ratio': (trainable_params / total_params * 100) if total_params > 0 else 0,
            'module_params': module_params,
            'peft_params': peft_params
        }
    
    def _count_peft_parameters(self, model) -> Dict:
        """统计PEFT（如LoRA）参数"""
        peft_info = {
            'has_peft': False,
            'peft_params': 0,
            'peft_type': None,
            'peft_details': {}
        }
        
        # 检查是否有LoRA或其他PEFT层
        lora_params = 0
        lora_layers = []
        
        for name, module in model.named_modules():
            # 检查LoRA层（通常包含'lora'关键字）
            if 'lora' in name.lower() or hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                peft_info['has_peft'] = True
                peft_info['peft_type'] = 'LoRA'
                
                layer_params = sum(p.numel() for p in module.parameters())
                lora_params += layer_params
                lora_layers.append({
                    'name': name,
                    'params': layer_params
                })
        
        if peft_info['has_peft']:
            peft_info['peft_params'] = lora_params
            peft_info['peft_details']['lora_layers'] = lora_layers
            peft_info['peft_details']['num_lora_layers'] = len(lora_layers)
        
        return peft_info
    
    def _analyze_memory_usage(self, model, device) -> Dict:
        """分析模型内存使用"""
        # 计算模型参数内存
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        param_size_mb = param_size / (1024 * 1024)
        
        # 计算模型buffer内存
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        buffer_size_mb = buffer_size / (1024 * 1024)
        
        total_model_size_mb = param_size_mb + buffer_size_mb
        
        # GPU内存信息（如果使用GPU）
        gpu_memory_info = {}
        if device.type == 'cuda' and torch.cuda.is_available():
            gpu_memory_info = {
                'gpu_available': True,
                'gpu_name': torch.cuda.get_device_name(device),
                'total_memory_gb': torch.cuda.get_device_properties(device).total_memory / (1024**3),
                'allocated_memory_mb': torch.cuda.memory_allocated(device) / (1024**2),
                'cached_memory_mb': torch.cuda.memory_reserved(device) / (1024**2)
            }
        else:
            gpu_memory_info = {'gpu_available': False}
        
        return {
            'param_memory_mb': param_size_mb,
            'buffer_memory_mb': buffer_size_mb,
            'total_model_memory_mb': total_model_size_mb,
            'gpu_memory_info': gpu_memory_info
        }
    
    def _analyze_model_structure(self, model) -> Dict:
        """分析模型结构"""
        # 如果是TrainableSAM，分析内部结构
        if isinstance(model, TrainableSAM):
            actual_model = model.sam
        else:
            actual_model = model
        
        # 统计不同类型的层
        layer_counts = {}
        for name, module in actual_model.named_modules():
            module_type = type(module).__name__
            layer_counts[module_type] = layer_counts.get(module_type, 0) + 1
        
        # 分析主要组件
        main_components = []
        for name, module in actual_model.named_children():
            main_components.append({
                'name': name,
                'type': type(module).__name__,
                'params': sum(p.numel() for p in module.parameters())
            })
        
        return {
            'layer_counts': layer_counts,
            'main_components': main_components,
            'model_class': type(model).__name__
        }
    
    def _print_analysis_results(self, analysis: Dict):
        """打印分析结果"""
        self.log_info("="*80)
        self.log_info("模型参数分析结果")
        self.log_info("="*80)
        
        # 基本信息
        self.log_info(f"模型类型: {analysis['model_type']}")
        self.log_info(f"模型类: {analysis['model_class']}")
        self.log_info(f"设备: {analysis['device']}")
        if analysis.get('checkpoint_path'):
            self.log_info(f"检查点: {Path(analysis['checkpoint_path']).name}")
        
        self.log_info("-" * 50)
        
        # 参数统计
        self.log_info("参数统计:")
        self.log_info(f"  总参数数: {analysis['total_params']:,}")
        self.log_info(f"  可训练参数数: {analysis['trainable_params']:,}")
        self.log_info(f"  冻结参数数: {analysis['frozen_params']:,}")
        self.log_info(f"  可训练参数比例: {analysis['trainable_ratio']:.2f}%")
        
        # PEFT信息
        peft_info = analysis.get('peft_params', {})
        if peft_info.get('has_peft'):
            self.log_info(f"  PEFT类型: {peft_info['peft_type']}")
            self.log_info(f"  PEFT参数数: {peft_info['peft_params']:,}")
            if 'peft_details' in peft_info:
                details = peft_info['peft_details']
                if 'num_lora_layers' in details:
                    self.log_info(f"  LoRA层数: {details['num_lora_layers']}")
        
        self.log_info("-" * 50)
        
        # 内存使用
        self.log_info("内存使用:")
        self.log_info(f"  模型内存使用: {analysis['total_model_memory_mb']:.2f} MB")
        self.log_info(f"  参数内存: {analysis['param_memory_mb']:.2f} MB")
        self.log_info(f"  缓冲区内存: {analysis['buffer_memory_mb']:.2f} MB")
        
        # GPU信息
        gpu_info = analysis.get('gpu_memory_info', {})
        if gpu_info.get('gpu_available'):
            self.log_info(f"  GPU: {gpu_info['gpu_name']}")
            self.log_info(f"  GPU总内存: {gpu_info['total_memory_gb']:.2f} GB")
            self.log_info(f"  已分配GPU内存: {gpu_info['allocated_memory_mb']:.2f} MB")
            self.log_info(f"  缓存GPU内存: {gpu_info['cached_memory_mb']:.2f} MB")
        
        self.log_info("-" * 50)
        
        # 模块参数分布
        self.log_info("各模块参数分布:")
        module_params = analysis.get('module_params', {})
        for module_name, params in module_params.items():
            self.log_info(f"  {module_name}:")
            self.log_info(f"    总参数: {params['total']:,}")
            self.log_info(f"    可训练: {params['trainable']:,}")
            self.log_info(f"    冻结: {params['frozen']:,}")
            self.log_info(f"    可训练比例: {params['trainable_ratio']:.2f}%")
        
        # 冻结配置
        if analysis.get('freeze_config'):
            self.log_info("-" * 50)
            self.log_info(f"冻结配置: {analysis['freeze_config']}")
        
        # PEFT配置
        if analysis.get('peft_config'):
            self.log_info("-" * 50)
            self.log_info("PEFT配置:")
            for key, value in analysis['peft_config'].items():
                self.log_info(f"  {key}: {value}")
        
        self.log_info("="*80)
    
    def get_parameter_summary(self) -> str:
        """获取参数摘要（单行格式）"""
        if not self.param_info:
            return "未进行参数分析"
        
        total = self.param_info.get('total_params', 0)
        trainable = self.param_info.get('trainable_params', 0)
        frozen = self.param_info.get('frozen_params', 0)
        ratio = self.param_info.get('trainable_ratio', 0)
        memory = self.param_info.get('total_model_memory_mb', 0)
        
        return (f"总参数数: {total:,} | 可训练参数数: {trainable:,} | "
                f"冻结参数数: {frozen:,} | 可训练参数比例: {ratio:.2f}% | "
                f"模型内存使用: {memory:.2f} MB")
    
    def save_analysis_report(self, save_path: str):
        """保存分析报告到文件"""
        if not self.param_info:
            self.log_info("无参数分析数据可保存")
            return
        
        try:
            import json
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.param_info, f, indent=2, ensure_ascii=False, default=str)
            self.log_info(f"参数分析报告已保存到: {save_path}")
        except Exception as e:
            self.log_info(f"保存分析报告失败: {e}")


# 使用示例函数
def analyze_training_model_parameters(model_type="vit_b_lm", checkpoint_path=None, 
                                    freeze=None, peft_kwargs=None, logger=None):
    """
    分析训练模型参数的便捷函数
    
    使用示例:
    # 基本分析
    analyze_training_model_parameters("vit_b_lm", logger=logger)
    
    # 分析冻结image_encoder的模型
    analyze_training_model_parameters(
        model_type="vit_b_lm",
        freeze=["image_encoder"],
        logger=logger
    )
    
    # 分析使用LoRA的模型
    analyze_training_model_parameters(
        model_type="vit_b_lm",
        peft_kwargs={"rank": 4, "attention_layers_to_update": [9, 10, 11]},
        logger=logger
    )
    """
    analyzer = ModelParameterAnalyzer(logger=logger)
    return analyzer.analyze_sam_model_parameters(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        freeze=freeze,
        peft_kwargs=peft_kwargs
    )


class PatchExtractor:
    """提取512x512补丁，10像素重叠"""
    
    def __init__(self, patch_size=512, overlap=10):
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
    
    def extract_patches(self, image, mask, image_path):
        """从图像和掩码中提取补丁"""
        patches = []
        
        # 确保输入是2D或3D numpy数组
        if not isinstance(image, np.ndarray) or not isinstance(mask, np.ndarray):
            return patches
        
        # 获取图像尺寸
        if len(image.shape) >= 2:
            h, w = image.shape[:2]
        else:
            return patches
        
        if len(mask.shape) != 2:
            return patches
        
        # 检查尺寸是否足够
        if h < self.patch_size or w < self.patch_size:
            return patches
        
        # 计算补丁的起始位置
        y_positions = list(range(0, h - self.patch_size + 1, self.stride))
        x_positions = list(range(0, w - self.patch_size + 1, self.stride))
        
        # 如果最后一个位置不能覆盖到边界，添加边界位置
        if len(y_positions) == 0:
            y_positions = [0]
        elif y_positions[-1] + self.patch_size < h:
            y_positions.append(h - self.patch_size)
            
        if len(x_positions) == 0:
            x_positions = [0]
        elif x_positions[-1] + self.patch_size < w:
            x_positions.append(w - self.patch_size)
        
        patch_id = 0
        for y in y_positions:
            for x in x_positions:
                try:
                    # 确保索引不越界
                    y_end = min(y + self.patch_size, h)
                    x_end = min(x + self.patch_size, w)
                    
                    # 如果补丁太小，跳过
                    if (y_end - y) < self.patch_size or (x_end - x) < self.patch_size:
                        continue
                    
                    # 提取补丁
                    img_patch = image[y:y_end, x:x_end]
                    mask_patch = mask[y:y_end, x:x_end]
                    
                    # 检查补丁尺寸
                    if img_patch.shape[:2] != (self.patch_size, self.patch_size):
                        continue
                    if mask_patch.shape != (self.patch_size, self.patch_size):
                        continue
                    
                    # 检查掩码补丁是否包含足够的前景
                    foreground_pixels = np.sum(mask_patch > 0)
                    if foreground_pixels >= 100:  # 至少100个前景像素
                        patches.append({
                            'image_patch': img_patch.copy(),
                            'mask_patch': mask_patch.copy(),
                            'original_image': str(image_path),
                            'patch_id': patch_id,
                            'position': (y, x),
                            'size': (self.patch_size, self.patch_size),
                            'foreground_pixels': foreground_pixels
                        })
                    
                    patch_id += 1
                    
                except Exception as e:
                    # 记录错误但继续处理
                    continue
        
        return patches

class DetailedLogger:
    """详细的可读日志记录器"""
    
    def __init__(self, save_dir, rank=0):
        self.rank = rank
        self.save_dir = Path(save_dir)
        self.log_dir = self.save_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if rank == 0:
            # 主日志文件
            self.main_log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            # 训练详细日志
            self.training_log_file = self.log_dir / f"training_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            # epoch时间日志
            self.epoch_timing_file = self.log_dir / f"epoch_timing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            # 设置主logger
            self.logger = logging.getLogger('MicroSAMTraining')
            self.logger.setLevel(logging.INFO)
            
            # 清除现有handlers
            if self.logger.handlers:
                self.logger.handlers.clear()
            
            # 文件handler
            file_handler = logging.FileHandler(self.main_log_file)
            file_handler.setLevel(logging.INFO)
            
            # 控制台handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            
            # 初始化CSV文件
            self.init_timing_csv()
            
            self.logger.info("="*80)
            self.logger.info("MICRO-SAM TRAINING SESSION STARTED")
            self.logger.info("="*80)
    
    def init_timing_csv(self):
        """初始化epoch时间记录CSV"""
        if self.rank == 0:
            timing_headers = [
                'epoch', 'start_time', 'end_time', 'duration_seconds', 
                'duration_formatted', 'samples_processed', 'avg_time_per_sample',
                'learning_rate', 'train_loss', 'val_loss', 'memory_used_gb'
            ]
            with open(self.epoch_timing_file, 'w') as f:
                f.write(','.join(timing_headers) + '\n')
    
    def log_info(self, message):
        """记录信息"""
        if self.rank == 0:
            self.logger.info(message)
    
    def log_error(self, message):
        """记录错误"""
        if self.rank == 0:
            self.logger.error(message)
    
    def log_warning(self, message):
        """记录警告"""
        if self.rank == 0:
            self.logger.warning(message)
    
    def log_training_details(self, details):
        """记录详细的训练信息"""
        if self.rank == 0:
            with open(self.training_log_file, 'a') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] {details}\n")
    
    def log_epoch_timing(self, epoch_data):
        """记录epoch时间信息"""
        if self.rank == 0:
            with open(self.epoch_timing_file, 'a') as f:
                values = [
                    epoch_data.get('epoch', ''),
                    epoch_data.get('start_time', ''),
                    epoch_data.get('end_time', ''),
                    epoch_data.get('duration_seconds', ''),
                    epoch_data.get('duration_formatted', ''),
                    epoch_data.get('samples_processed', ''),
                    epoch_data.get('avg_time_per_sample', ''),
                    epoch_data.get('learning_rate', ''),
                    epoch_data.get('train_loss', ''),
                    epoch_data.get('val_loss', ''),
                    epoch_data.get('memory_used_gb', '')
                ]
                f.write(','.join(map(str, values)) + '\n')
    
    def log_dataset_summary(self, summary_data):
        """记录数据集摘要"""
        if self.rank == 0:
            self.logger.info("DATASET SUMMARY:")
            self.logger.info(f"  Total patches extracted: {summary_data.get('total_patches', 0)}")
            self.logger.info(f"  Training patches: {summary_data.get('train_patches', 0)}")
            self.logger.info(f"  Validation patches: {summary_data.get('val_patches', 0)}")
            self.logger.info(f"  Original images processed: {summary_data.get('original_images', 0)}")
            self.logger.info(f"  Patch size: {summary_data.get('patch_size', 'N/A')}")
            self.logger.info(f"  Overlap: {summary_data.get('overlap', 'N/A')} pixels")
            
            for dataset_name, info in summary_data.get('dataset_breakdown', {}).items():
                self.logger.info(f"  {dataset_name}: {info['patches']} patches from {info['images']} images")

class MicroSAMDatasetAdapter(Dataset):
    """智能适配micro_sam期望格式的数据集"""
    
    def __init__(self, patch_infos, is_train=True, with_segmentation_decoder=True):
        self.patch_infos = patch_infos
        self.is_train = is_train
        self.with_segmentation_decoder = with_segmentation_decoder
    
    def __len__(self):
        return len(self.patch_infos)
    
    def __getitem__(self, idx):
        patch_info = self.patch_infos[idx]
        
        # 加载图像和掩码
        img_path = patch_info['img_path']
        mask_path = patch_info['mask_path']
        
        try:
            # 使用imageio加载（所有补丁都是2D灰度图）
            image = imageio.imread(img_path).astype(np.float32)
            mask = imageio.imread(mask_path).astype(np.uint8)
            
            # 确保是2D
            if len(image.shape) > 2:
                image = image[:, :, 0] if image.shape[2] == 1 else np.mean(image, axis=2)
            if len(mask.shape) > 2:
                mask = mask[:, :, 0] if mask.shape[2] == 1 else mask[:, :, 0]
            
            # 标准化图像到[0,1]
            if image.max() > 1:
                image = image / 255.0
            
            # 转换为张量
            image_tensor = torch.from_numpy(image).unsqueeze(0).float()  # (1, H, W)
            
            # 根据with_segmentation_decoder调整标签格式
            if self.with_segmentation_decoder:
                # micro_sam期望4通道的目标数据：
                # channel 0: 原始标签（实例分割）
                # channel 1: 前景/背景二值掩码
                # channel 2: 边界掩码
                # channel 3: 距离变换
                
                # 创建4通道目标张量
                target_4ch = torch.zeros((4, mask.shape[0], mask.shape[1]), dtype=torch.float32)
                
                # Channel 0: 实例标签
                target_4ch[0] = torch.from_numpy(mask).float()
                
                # Channel 1: 前景掩码（二值化）
                foreground_mask = (mask > 0).astype(np.float32)
                target_4ch[1] = torch.from_numpy(foreground_mask)
                
                # Channel 2: 边界掩码（简化处理）
                from scipy.ndimage import binary_erosion
                boundary_mask = foreground_mask - binary_erosion(foreground_mask).astype(np.float32)
                target_4ch[2] = torch.from_numpy(boundary_mask)
                
                # Channel 3: 距离变换（简化处理）
                from scipy.ndimage import distance_transform_edt
                if np.any(foreground_mask):
                    dist_transform = distance_transform_edt(foreground_mask)
                    # 归一化到[0,1]
                    if dist_transform.max() > 0:
                        dist_transform = dist_transform / dist_transform.max()
                else:
                    dist_transform = np.zeros_like(foreground_mask)
                target_4ch[3] = torch.from_numpy(dist_transform.astype(np.float32))
                
                return image_tensor, target_4ch
            else:
                # 标准的2D掩码
                mask_tensor = torch.from_numpy(mask).long()  # (H, W)
                return image_tensor, mask_tensor
            
        except Exception as e:
            print(f"Error loading patch {idx}: {e}")
            # 返回零张量作为fallback
            if self.with_segmentation_decoder:
                return (torch.zeros(1, 512, 512, dtype=torch.float32), 
                       torch.zeros(4, 512, 512, dtype=torch.float32))
            else:
                return (torch.zeros(1, 512, 512, dtype=torch.float32), 
                       torch.zeros(512, 512, dtype=torch.long))

class OptimizedDatasetHandler:
    """优化的数据集处理器，支持补丁提取和缓存，使用模型名避免冲突"""
    
    def __init__(self, json_files, train_ratio=0.8, patch_size=512, overlap=10, logger=None, force_regenerate=False, model_name="default"):
        self.json_files = json_files
        self.train_ratio = train_ratio
        self.patch_extractor = PatchExtractor(patch_size, overlap)
        self.logger = logger or DetailedLogger("./logs")
        self.force_regenerate = force_regenerate
        self.model_name = model_name  # 新增：模型名参数
        
        self.all_patches = []
        self.dataset_info = {}
        self.patch_save_dir = None
        
        # 设置补丁保存目录 - 基于模型名和补丁大小
        self.patch_save_dir = Path(f"/LD-FS/home/yunshuchen/micro_sam/patches_{self.model_name}_{self.patch_extractor.patch_size}")
        self.cache_info_file = self.patch_save_dir / f"patch_cache_info_{self.model_name}.json"
        
        self.logger.log_info(f"Starting dataset preprocessing for model: {self.model_name}")
        self.logger.log_info(f"Cache directory: {self.patch_save_dir}")
        
        # 检查是否已有缓存的补丁
        self.debug_cache_mismatch()
        if self.check_existing_patches() and not force_regenerate:
            self.logger.log_info(f"Found existing patches for model {self.model_name}, loading from cache...")
            self.load_cached_patches()
        else:
            if force_regenerate:
                self.logger.log_info(f"Force regenerating patches for model {self.model_name}...")
            else:
                self.logger.log_info(f"No valid cache found for model {self.model_name}, generating patches...")
            self.process_all_datasets()
            self.save_cache_info()
        
        self.split_train_val()

    # 在 OptimizedDatasetHandler 类中替换这两个方法

    def check_existing_patches(self):
        """检查是否已存在有效的补丁缓存 - 修复版本"""
        try:
            if not self.cache_info_file.exists():
                self.logger.log_info(f"Cache file not found: {self.cache_info_file}")
                return False
            
            # 读取缓存信息
            with open(self.cache_info_file, 'r') as f:
                cache_info = json.load(f)
            
            # 检查模型名是否匹配
            cached_model_name = cache_info.get('model_name', '')
            if cached_model_name != self.model_name:
                self.logger.log_info(f"Model name mismatch - cached: {cached_model_name}, current: {self.model_name}")
                return False
            
            # 更宽松的配置检查 - 只检查关键参数
            cached_config = cache_info.get('config', {})
            
            # 检查补丁大小和重叠 - 这些是最重要的
            if (cached_config.get('patch_size') != self.patch_extractor.patch_size or
                cached_config.get('overlap') != self.patch_extractor.overlap):
                self.logger.log_info(f"Core config mismatch - patch_size or overlap different")
                return False
            
            # 检查补丁文件是否还存在
            total_patches = cache_info.get('total_patches', 0)
            if total_patches == 0:
                self.logger.log_info(f"No patches found in cache for model {self.model_name}")
                return False
            
            # 检查缓存目录和文件
            if not self.patch_save_dir.exists():
                self.logger.log_info(f"Cache directory does not exist: {self.patch_save_dir}")
                return False
            
            # 快速文件数量检查
            actual_patch_files = list(self.patch_save_dir.rglob("*_img.png"))
            actual_mask_files = list(self.patch_save_dir.rglob("*_mask.png"))
            
            # 允许一定的文件丢失（95%阈值）
            min_expected = max(1, int(total_patches * 0.95))
            
            if len(actual_patch_files) < min_expected or len(actual_mask_files) < min_expected:
                self.logger.log_info(f"Insufficient patch files - expected: {total_patches}, found: img={len(actual_patch_files)}, mask={len(actual_mask_files)}")
                return False
            
            self.logger.log_info(f"Found valid cache for model {self.model_name} with {len(actual_patch_files)} patch pairs")
            return True
            
        except Exception as e:
            self.logger.log_warning(f"Error checking cache for model {self.model_name}: {e}")
            return False

    def load_cached_patches(self):
        """从缓存加载补丁信息 - 修复版本"""
        try:
            with open(self.cache_info_file, 'r') as f:
                cache_info = json.load(f)
            
            # 验证模型名
            cached_model_name = cache_info.get('model_name', '')
            if cached_model_name != self.model_name:
                raise ValueError(f"Model name mismatch in cache: expected {self.model_name}, got {cached_model_name}")
            
            cached_patches = cache_info.get('patches', [])
            self.dataset_info = cache_info.get('dataset_info', {})
            
            # 验证补丁文件存在性，但更宽松
            valid_patches = []
            
            self.logger.log_info(f"Loading cached patches for model {self.model_name}...")
            
            for patch_info in cached_patches:
                img_path = Path(patch_info['img_path'])
                mask_path = Path(patch_info['mask_path'])
                
                # 只检查文件是否存在，不做复杂验证
                if img_path.exists() and mask_path.exists():
                    valid_patches.append(patch_info)
            
            if len(valid_patches) < len(cached_patches) * 0.9:  # 90%阈值
                self.logger.log_warning(f"Too many missing files ({len(cached_patches)-len(valid_patches)}/{len(cached_patches)}), will regenerate")
                raise ValueError("Cache contains too many missing files")
            
            self.all_patches = valid_patches
            
            self.logger.log_info(f"Successfully loaded {len(self.all_patches)} patches from cache for model {self.model_name}")
            
            # 记录缓存摘要
            summary_data = {
                'total_patches': len(self.all_patches),
                'original_images': sum(info.get('valid_images', 0) for info in self.dataset_info.values()),
                'patch_size': f"{self.patch_extractor.patch_size}x{self.patch_extractor.patch_size}",
                'overlap': self.patch_extractor.overlap,
                'model_name': self.model_name,
                'dataset_breakdown': {
                    name: {'patches': info.get('total_patches', 0), 'images': info.get('valid_images', 0)}
                    for name, info in self.dataset_info.items()
                },
                'source': f'cache-{self.model_name}'
            }
            
            self.logger.log_dataset_summary(summary_data)
            
        except Exception as e:
            self.logger.log_error(f"Failed to load cached patches for model {self.model_name}: {e}")
            # 如果缓存加载失败，重新处理
            self.process_all_datasets()
            self.save_cache_info()

    # 另外，在 OptimizedDatasetHandler.__init__ 方法中，可以添加调试信息
    def debug_cache_mismatch(self):
        """调试缓存不匹配的原因"""
        if not self.cache_info_file.exists():
            self.logger.log_info("No cache file exists - this is normal for first run")
            return
        
        try:
            with open(self.cache_info_file, 'r') as f:
                cache_info = json.load(f)
            
            self.logger.log_info("=== CACHE DEBUG INFO ===")
            self.logger.log_info(f"Cache file: {self.cache_info_file}")
            self.logger.log_info(f"Cached model name: {cache_info.get('model_name', 'N/A')}")
            self.logger.log_info(f"Current model name: {self.model_name}")
            
            cached_config = cache_info.get('config', {})
            self.logger.log_info(f"Cached patch size: {cached_config.get('patch_size', 'N/A')}")
            self.logger.log_info(f"Current patch size: {self.patch_extractor.patch_size}")
            self.logger.log_info(f"Cached overlap: {cached_config.get('overlap', 'N/A')}")
            self.logger.log_info(f"Current overlap: {self.patch_extractor.overlap}")
            
            cached_json_files = cached_config.get('json_files', [])
            current_json_files = [str(f) for f in self.json_files]
            
            self.logger.log_info(f"Cached JSON files count: {len(cached_json_files)}")
            self.logger.log_info(f"Current JSON files count: {len(current_json_files)}")
            
            if len(cached_json_files) != len(current_json_files):
                self.logger.log_info("JSON file count mismatch!")
            
            self.logger.log_info("=== END CACHE DEBUG ===")
            
        except Exception as e:
            self.logger.log_error(f"Error in cache debug: {e}")
    # 使用方法：在 OptimizedDatasetHandler.__init__ 中，在检查缓存之前添加：
    # self.debug_cache_mismatch()
    def save_cache_info(self):
        """保存补丁缓存信息，修复JSON序列化问题"""
        try:
            # 清理数据，移除numpy类型
            clean_patches = []
            for patch_info in self.all_patches:
                clean_patch = {}
                for key, value in patch_info.items():
                    if key == 'patch_info':
                        # 特殊处理patch_info字典
                        clean_patch_info = {}
                        for pk, pv in value.items():
                            if isinstance(pv, (np.integer, np.int64, np.int32)):
                                clean_patch_info[pk] = int(pv)
                            elif isinstance(pv, (np.floating, np.float64, np.float32)):
                                clean_patch_info[pk] = float(pv)
                            elif isinstance(pv, np.ndarray):
                                clean_patch_info[pk] = pv.tolist()
                            elif isinstance(pv, tuple):
                                clean_patch_info[pk] = list(pv)
                            else:
                                clean_patch_info[pk] = pv
                        clean_patch[key] = clean_patch_info
                    elif isinstance(value, (np.integer, np.int64, np.int32)):
                        clean_patch[key] = int(value)
                    elif isinstance(value, (np.floating, np.float64, np.float32)):
                        clean_patch[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        clean_patch[key] = value.tolist()
                    elif isinstance(value, tuple):
                        clean_patch[key] = list(value)
                    else:
                        clean_patch[key] = value
                clean_patches.append(clean_patch)
            
            # 清理dataset_info
            clean_dataset_info = {}
            for name, info in self.dataset_info.items():
                clean_info = {}
                for key, value in info.items():
                    if isinstance(value, (np.integer, np.int64, np.int32)):
                        clean_info[key] = int(value)
                    elif isinstance(value, (np.floating, np.float64, np.float32)):
                        clean_info[key] = float(value)
                    else:
                        clean_info[key] = value
                clean_dataset_info[name] = clean_info
            
            cache_info = {
                'model_name': self.model_name,  # 新增：保存模型名
                'config': {
                    'patch_size': int(self.patch_extractor.patch_size),
                    'overlap': int(self.patch_extractor.overlap),
                    'json_files': [str(f) for f in self.json_files],
                    'model_name': self.model_name
                },
                'total_patches': len(clean_patches),
                'dataset_info': clean_dataset_info,
                'patches': clean_patches,
                'generated_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'cache_version': '2.0'  # 升级版本号
            }
            
            self.patch_save_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.cache_info_file, 'w') as f:
                json.dump(cache_info, f, indent=2)
            
            self.logger.log_info(f"Saved cache info for model {self.model_name} to {self.cache_info_file}")
            
        except Exception as e:
            self.logger.log_error(f"Failed to save cache info for model {self.model_name}: {e}")
            import traceback
            self.logger.log_error(traceback.format_exc())

    def clear_cache(self):
        """清除补丁缓存"""
        try:
            if self.patch_save_dir.exists():
                import shutil
                shutil.rmtree(self.patch_save_dir)
                self.logger.log_info(f"Cleared patch cache for model {self.model_name}: {self.patch_save_dir}")
            else:
                self.logger.log_info(f"No cache to clear for model {self.model_name}")
        except Exception as e:
            self.logger.log_error(f"Failed to clear cache for model {self.model_name}: {e}")
    
    def clear_all_caches(self):
        """清除所有模型的缓存（用于完全重置）"""
        try:
            base_cache_dir = Path("/LD-FS/home/yunshuchen/micro_sam")
            cache_dirs = list(base_cache_dir.glob("patches_*"))
            
            for cache_dir in cache_dirs:
                if cache_dir.is_dir():
                    import shutil
                    shutil.rmtree(cache_dir)
                    self.logger.log_info(f"Removed cache directory: {cache_dir}")
            
            self.logger.log_info(f"Cleared all patch caches ({len(cache_dirs)} directories)")
            
        except Exception as e:
            self.logger.log_error(f"Failed to clear all caches: {e}")

    def process_all_datasets(self):
        """处理所有数据集并提取补丁"""
        total_images = 0
        total_patches = 0
        
        self.logger.log_info(f"Processing datasets for model: {self.model_name}")
        
        for json_file in self.json_files:
            self.logger.log_info(f"Processing dataset: {json_file}")
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                self.logger.log_error(f"Failed to read JSON file {json_file}: {e}")
                continue
            
            dataset_name = data.get('dataset_name', Path(json_file).stem)
            
            # 修复路径问题 - 清理双重路径
            images_path_str = str(data['images_path'])
            masks_path_str = str(data['masks_path'])
            
            # 检查并修复双重路径
            if '/Retrain/Retrain/' in images_path_str:
                images_path_str = images_path_str.replace('/Retrain/Retrain/', '/Retrain/')
                self.logger.log_info(f"Fixed duplicate path in images_path: {images_path_str}")
            
            if '/Retrain/Retrain/' in masks_path_str:
                masks_path_str = masks_path_str.replace('/Retrain/Retrain/', '/Retrain/')
                self.logger.log_info(f"Fixed duplicate path in masks_path: {masks_path_str}")
            
            images_path = Path(images_path_str)
            masks_path = Path(masks_path_str)
            
            # 验证路径存在
            if not images_path.exists():
                self.logger.log_error(f"Images path does not exist: {images_path}")
                continue
            
            if not masks_path.exists():
                self.logger.log_error(f"Masks path does not exist: {masks_path}")
                continue
            
            # 创建补丁保存目录（基于模型名）
            dataset_patch_dir = self.patch_save_dir / dataset_name
            dataset_patch_dir.mkdir(parents=True, exist_ok=True)
            
            dataset_patches = 0
            valid_images = 0
            skipped_images = 0
            corrupted_files = 0
            
            for img_name, info in data['mapping'].items():
                try:
                    # 检查info结构和mask_file
                    if not isinstance(info, dict):
                        self.logger.log_warning(f"Invalid info structure for {img_name}: {info}")
                        skipped_images += 1
                        continue
                    
                    mask_file = info.get('mask_file')
                    if mask_file is None or mask_file == '':
                        self.logger.log_warning(f"No mask file specified for {img_name}")
                        skipped_images += 1
                        continue
                    
                    img_path = images_path / img_name
                    mask_path = masks_path / mask_file
                    
                    if not img_path.exists():
                        self.logger.log_warning(f"Image file not found: {img_path}")
                        skipped_images += 1
                        continue
                    
                    if not mask_path.exists():
                        self.logger.log_warning(f"Mask file not found: {mask_path}")
                        skipped_images += 1
                        continue
                    
                    # 检查文件是否损坏
                    if self.is_file_corrupted(img_path) or self.is_file_corrupted(mask_path):
                        self.logger.log_warning(f"Corrupted file detected: {img_path} or {mask_path}")
                        corrupted_files += 1
                        continue
                    
                    # 加载图像和掩码
                    image = self.load_image_robust(img_path)
                    mask = self.load_image_robust(mask_path)
                    
                    if image is None:
                        self.logger.log_warning(f"Failed to load image: {img_path}")
                        skipped_images += 1
                        continue
                    
                    if mask is None:
                        self.logger.log_warning(f"Failed to load mask: {mask_path}")
                        skipped_images += 1
                        continue
                    
                    # 验证加载的数据质量
                    if not self.validate_loaded_data(image, mask, img_path, mask_path):
                        skipped_images += 1
                        continue
                    
                    # 确保图像和掩码尺寸匹配
                    if image.shape[:2] != mask.shape[:2]:
                        self.logger.log_info(f"Size mismatch detected - Image: {image.shape}, Mask: {mask.shape} for {img_path}")
                        
                        # 智能匹配尺寸
                        image, mask = self.match_image_mask_sizes(image, mask, img_path)
                        
                        if image is None or mask is None:
                            self.logger.log_warning(f"Failed to match image and mask sizes: {img_path}")
                            skipped_images += 1
                            continue
                    
                    # 检查图像尺寸，如果太小则上采样（现在对匹配后的尺寸检查）
                    original_size = image.shape[:2]
                    if image.shape[0] < self.patch_extractor.patch_size or image.shape[1] < self.patch_extractor.patch_size:
                        self.logger.log_info(f"Upsampling small image from {original_size} to minimum {self.patch_extractor.patch_size}: {img_path}")
                        image, mask = self.resize_to_minimum_size(image, mask, self.patch_extractor.patch_size)
                        
                        if image is None or mask is None:
                            self.logger.log_warning(f"Failed to resize image: {img_path}")
                            skipped_images += 1
                            continue
                        
                        self.logger.log_info(f"  Resized to: {image.shape[:2]}")
                    
                    # 提取补丁
                    patches = self.patch_extractor.extract_patches(image, mask, img_path)
                    
                    if not patches:
                        self.logger.log_warning(f"No valid patches extracted from {img_path}")
                        skipped_images += 1
                        continue
                    
                    # 保存补丁（包含模型名在路径中）
                    saved_patches = 0
                    for i, patch in enumerate(patches):
                        patch_name = f"{Path(img_path).stem}_patch_{i}"
                        
                        # 保存图像补丁
                        img_patch_path = dataset_patch_dir / f"{patch_name}_img.png"
                        mask_patch_path = dataset_patch_dir / f"{patch_name}_mask.png"
                        
                        # 安全保存图像补丁
                        if self.save_patch_safe(patch['image_patch'], img_patch_path, is_mask=False):
                            if self.save_patch_safe(patch['mask_patch'], mask_patch_path, is_mask=True):
                                # 添加到补丁列表（包含模型信息）
                                self.all_patches.append({
                                    'img_path': str(img_patch_path),
                                    'mask_path': str(mask_patch_path),
                                    'dataset': dataset_name,
                                    'model_name': self.model_name,  # 新增：记录模型名
                                    'original_image': str(img_path),
                                    'patch_info': {
                                        'patch_id': patch['patch_id'],
                                        'position': patch['position'],
                                        'size': patch['size'],
                                        'foreground_pixels': patch['foreground_pixels']
                                    }
                                })
                                saved_patches += 1
                            else:
                                # 如果掩码保存失败，删除图像文件
                                if img_patch_path.exists():
                                    img_patch_path.unlink()
                        
                    dataset_patches += saved_patches
                    
                    if saved_patches > 0:
                        valid_images += 1
                    else:
                        skipped_images += 1
                    
                    if valid_images % 10 == 0:
                        self.logger.log_info(f"  Processed {valid_images} images, extracted {dataset_patches} patches (skipped {skipped_images})")
                
                except Exception as e:
                    self.logger.log_error(f"Error processing {img_name}: {e}")
                    skipped_images += 1
                    continue
            
            self.dataset_info[dataset_name] = {
                'total_images': len(data['mapping']),
                'valid_images': valid_images,
                'skipped_images': skipped_images,
                'corrupted_files': corrupted_files,
                'total_patches': dataset_patches,
                'model_name': self.model_name  # 新增：记录模型名
            }
            
            total_images += valid_images
            total_patches += dataset_patches
            
            self.logger.log_info(f"Dataset {dataset_name} completed for model {self.model_name}:")
            self.logger.log_info(f"  Total images in dataset: {len(data['mapping'])}")
            self.logger.log_info(f"  Valid images processed: {valid_images}")
            self.logger.log_info(f"  Skipped images: {skipped_images}")
            self.logger.log_info(f"  Corrupted files: {corrupted_files}")
            self.logger.log_info(f"  Extracted patches: {dataset_patches}")
            
            # 如果数据集没有有效图像，记录警告
            if valid_images == 0:
                self.logger.log_warning(f"No valid images found in dataset {dataset_name} for model {self.model_name}!")
                # 如果损坏文件过多，建议跳过整个数据集
                if corrupted_files > len(data['mapping']) * 0.8:  # 80%以上损坏
                    self.logger.log_warning(f"Dataset {dataset_name} has {corrupted_files}/{len(data['mapping'])} corrupted files, consider removing from training")
            else:
                # 计算数据集质量得分
                quality_score = valid_images / len(data['mapping'])
                self.logger.log_info(f"  Dataset quality: {quality_score:.2%} ({valid_images}/{len(data['mapping'])} usable)")
        
        # 检查是否有有效的补丁
        if total_patches == 0:
            raise ValueError(f"No valid patches extracted from any dataset for model {self.model_name}!")
        
        # 记录总体统计
        summary_data = {
            'total_patches': total_patches,
            'original_images': total_images,
            'patch_size': f"{self.patch_extractor.patch_size}x{self.patch_extractor.patch_size}",
            'overlap': self.patch_extractor.overlap,
            'model_name': self.model_name,
            'dataset_breakdown': {
                name: {'patches': info['total_patches'], 'images': info['valid_images']}
                for name, info in self.dataset_info.items()
            }
        }
        
        self.logger.log_dataset_summary(summary_data)

    # 其他方法保持不变...
    def match_image_mask_sizes(self, image, mask, img_path):
        """智能匹配图像和掩码尺寸"""
        img_h, img_w = image.shape[:2]
        mask_h, mask_w = mask.shape[:2]
        
        if img_h == mask_h and img_w == mask_w:
            return image, mask  # 尺寸已匹配
        
        self.logger.log_info(f"Resizing mismatched sizes - Image: {image.shape[:2]}, Mask: {mask.shape[:2]} for {img_path}")
        
        # 选择较大的尺寸作为目标尺寸
        target_h = max(img_h, mask_h)
        target_w = max(img_w, mask_w)
        
        try:
            # 调整图像尺寸
            if img_h != target_h or img_w != target_w:
                if len(image.shape) == 2:
                    resized_image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                else:
                    resized_image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            else:
                resized_image = image
            
            # 调整掩码尺寸
            if mask_h != target_h or mask_w != target_w:
                resized_mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            else:
                resized_mask = mask
            
            self.logger.log_info(f"  Resized to common size: {resized_image.shape[:2]}")
            return resized_image, resized_mask
            
        except Exception as e:
            self.logger.log_error(f"Failed to resize mismatched images: {e}")
            return None, None

    def resize_to_minimum_size(self, image, mask, min_size=512):
        """将图像和掩码调整到最小尺寸"""
        h, w = image.shape[:2]
        
        # 检查是否需要调整大小
        if h >= min_size and w >= min_size:
            return image, mask
        
        # 计算缩放因子，确保最小边达到min_size
        scale_h = min_size / h if h < min_size else 1.0
        scale_w = min_size / w if w < min_size else 1.0
        scale = max(scale_h, scale_w)  # 使用较大的缩放因子确保两边都达到最小尺寸
        
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        try:
            # 调整图像大小
            if len(image.shape) == 2:
                # 灰度图像
                resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                # 多通道图像
                resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # 调整掩码大小（使用最近邻插值保持标签值）
            resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            return resized_image, resized_mask
            
        except Exception as e:
            self.logger.log_error(f"Failed to resize image: {e}")
            return None, None

    def is_file_corrupted(self, file_path):
        """检查文件是否损坏"""
        try:
            file_size = Path(file_path).stat().st_size
            # 检查文件大小是否合理（至少1KB）
            if file_size < 1024:
                return True
            
            # 尝试读取文件头部
            with open(file_path, 'rb') as f:
                header = f.read(10)
                if len(header) < 10:
                    return True
                
                # 检查TIFF文件头部
                if file_path.suffix.lower() in ['.tif', '.tiff']:
                    # TIFF文件应该以II或MM开头
                    if not (header.startswith(b'II') or header.startswith(b'MM')):
                        return True
            
            return False
            
        except Exception:
            return True
    
    def load_image_robust(self, file_path):
        """鲁棒的图像加载"""
        try:
            # 首先检查文件是否损坏
            if self.is_file_corrupted(file_path):
                self.logger.log_error(f"File appears to be corrupted: {file_path}")
                return None
            
            # 根据文件扩展名智能选择加载器
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in ['.tif', '.tiff']:
                # TIFF文件的加载器顺序
                loaders = [
                    ('tifffile', lambda p: tifffile.imread(str(p))),
                    ('imageio', lambda p: imageio.imread(str(p))),
                    ('PIL', lambda p: np.array(Image.open(str(p)))),
                    ('skimage', lambda p: io.imread(str(p))),
                    ('cv2', lambda p: cv2.imread(str(p), cv2.IMREAD_UNCHANGED))
                ]
            else:
                # PNG, JPG等其他格式的加载器顺序
                loaders = [
                    ('imageio', lambda p: imageio.imread(str(p))),
                    ('PIL', lambda p: np.array(Image.open(str(p)))),
                    ('cv2', lambda p: cv2.imread(str(p), cv2.IMREAD_UNCHANGED)),
                    ('cv2_gray', lambda p: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)),
                    ('skimage', lambda p: io.imread(str(p))),
                    ('tifffile', lambda p: tifffile.imread(str(p)))  # 作为最后的备选
                ]
            
            image = None
            successful_loader = None
            
            for loader_name, loader_func in loaders:
                try:
                    image = loader_func(file_path)
                    if image is not None and isinstance(image, np.ndarray) and image.size > 0:
                        successful_loader = loader_name
                        break
                except Exception as e:
                    # 只对前两个加载器记录警告（更可能成功的），其他的跳过
                    if loaders.index((loader_name, loader_func)) < 2:
                        error_msg = str(e)[:100] + "..." if len(str(e)) > 100 else str(e)
                        self.logger.log_warning(f"Loader {loader_name} failed for {file_path}: {error_msg}")
                    continue
            
            if image is None:
                self.logger.log_error(f"All loaders failed for {file_path}")
                return None
            
            # 立即标准化数据范围
            is_mask = 'mask' in str(file_path).lower()
            image = self.normalize_image_data(image, is_mask)
            
            # 处理多通道图像
            original_shape = image.shape
            
            # 如果是多维数组，处理维度
            if len(image.shape) > 3:
                # 可能是时间序列或多波段图像，取第一个
                image = image[0] if image.shape[0] < image.shape[-1] else image[..., 0]
            
            if len(image.shape) == 3:
                if image.shape[2] > 3:
                    # 太多通道，取前3个或第一个
                    if 'mask' in str(file_path).lower():
                        image = image[:, :, 0]  # 掩码取第一个通道
                    else:
                        image = image[:, :, :3]  # 图像取前3个通道
                elif image.shape[2] == 3 and 'mask' in str(file_path).lower():
                    # 掩码应该是单通道，转换RGB为灰度
                    image = np.mean(image, axis=2).astype(image.dtype)
                elif image.shape[2] == 2:
                    # 2通道图像，取第一个通道
                    image = image[:, :, 0]
            
            # 最终确保掩码是2D，图像可以是2D或3D
            if 'mask' in str(file_path).lower() and len(image.shape) != 2:
                self.logger.log_error(f"Mask {file_path} has invalid final shape: {image.shape} (original: {original_shape})")
                return None
            
            # 检查图像尺寸
            if image.shape[0] < 10 or image.shape[1] < 10:
                self.logger.log_error(f"Image {file_path} too small: {image.shape}")
                return None
            
            # 检查图像是否全为零（可能是损坏的文件）
            if np.all(image == 0):
                # 对于掩码，全零可能是合理的（无标注），但对于图像则不合理
                if 'mask' in str(file_path).lower():
                    self.logger.log_warning(f"Mask has no annotations (all zeros): {file_path}")
                    return None  # 跳过无标注的掩码
                else:
                    self.logger.log_warning(f"Image appears to be all zeros: {file_path}")
                    return None
            
            # 最终数据范围验证
            if image.min() < 0 or image.max() > 255:
                self.logger.log_warning(f"Image data range [{image.min()}, {image.max()}] outside [0, 255], re-normalizing: {file_path}")
                image = self.normalize_image_data(image, is_mask)
                
            return image
            
        except Exception as e:
            self.logger.log_error(f"Failed to load {file_path}: {e}")
            return None

    def normalize_image_data(self, image, is_mask=False):
        """标准化图像数据到[0, 255]范围"""
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        if is_mask:
            # 掩码处理：确保标签值合理
            if image.dtype == np.bool_:
                return image.astype(np.uint8) * 255
            elif image.max() <= 1 and image.min() >= 0:
                # 二值掩码 0-1 范围
                return (image * 255).astype(np.uint8)
            elif image.min() >= 0 and image.max() <= 255:
                # 已经在合理范围
                return image.astype(np.uint8)
            else:
                # 实例分割掩码，重新映射标签
                unique_labels = np.unique(image)
                unique_labels = unique_labels[unique_labels >= 0]  # 移除负值
                
                normalized_mask = np.zeros_like(image, dtype=np.uint8)
                for i, label in enumerate(unique_labels):
                    if label == 0:  # 背景保持为0
                        continue
                    normalized_mask[image == label] = min(i + 1, 255)
                
                return normalized_mask
        else:
            # 图像处理：标准化到[0, 255]
            if image.dtype == np.bool_:
                return image.astype(np.uint8) * 255
            
            # 处理负值和异常值
            if image.min() < 0:
                # 有负值，需要重新映射
                image_float = image.astype(np.float64)
                image_float = image_float - image_float.min()  # 将最小值移到0
                if image_float.max() > 0:
                    image_float = image_float / image_float.max() * 255  # 缩放到[0, 255]
                return image_float.astype(np.uint8)
            elif image.max() <= 1.0:
                # 0-1范围的浮点数
                return (image * 255).astype(np.uint8)
            elif image.max() <= 255:
                # 已经在[0, 255]范围
                return image.astype(np.uint8)
            else:
                # 需要缩放的大值
                image_float = image.astype(np.float64)
                image_float = image_float / image_float.max() * 255
                return image_float.astype(np.uint8)

    def save_patch_safe(self, patch_data, save_path, is_mask=False):
        """安全地保存补丁数据，修复形状不匹配问题"""
        try:
            # 首先标准化数据
            normalized_data = self.normalize_image_data(patch_data, is_mask)
            
            # 关键修复：确保图像和掩码都保存为单通道灰度图
            if len(normalized_data.shape) > 2:
                if normalized_data.shape[2] == 3:
                    # RGB转灰度 - 使用标准权重
                    normalized_data = np.dot(normalized_data[...,:3], [0.299, 0.587, 0.114])
                    normalized_data = normalized_data.astype(np.uint8)
                elif normalized_data.shape[2] == 1:
                    # 单通道但是3D，压缩为2D
                    normalized_data = normalized_data[:, :, 0]
                else:
                    # 其他通道数，取第一个通道
                    normalized_data = normalized_data[:, :, 0]
            
            # 确保最终是2D
            if len(normalized_data.shape) != 2:
                self.logger.log_error(f"Failed to convert to 2D: {normalized_data.shape}")
                return False
            
            # 使用PIL保存为PNG格式，确保单通道
            from PIL import Image
            if normalized_data.dtype != np.uint8:
                normalized_data = normalized_data.astype(np.uint8)
            
            pil_image = Image.fromarray(normalized_data, mode='L')  # 'L'模式表示灰度
            pil_image.save(str(save_path))
            
            # 验证保存的文件
            try:
                test_load = np.array(Image.open(str(save_path)))
                if len(test_load.shape) != 2:
                    self.logger.log_warning(f"Saved file has wrong dimensions {test_load.shape}: {save_path}")
                    return False
                if test_load.min() < 0 or test_load.max() > 255:
                    self.logger.log_warning(f"Saved file has invalid range [{test_load.min()}, {test_load.max()}]: {save_path}")
                    return False
            except Exception as e:
                self.logger.log_error(f"Failed to verify saved file {save_path}: {e}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.log_error(f"Failed to save patch to {save_path}: {e}")
            return False

    def validate_loaded_data(self, image, mask, img_path, mask_path):
        """验证加载的数据质量"""
        # 检查数据范围
        if image.min() < 0 or image.max() > 255:
            self.logger.log_warning(f"Image data out of range [{image.min()}, {image.max()}]: {img_path}")
            return False
        
        if mask.min() < 0 or mask.max() > 255:
            self.logger.log_warning(f"Mask data out of range [{mask.min()}, {mask.max()}]: {mask_path}")
            return False
        
        # 检查掩码是否有有效内容
        if np.all(mask == 0):
            self.logger.log_warning(f"Mask has no annotations: {mask_path}")
            return False
        
        # 检查数据类型
        if not isinstance(image, np.ndarray) or not isinstance(mask, np.ndarray):
            self.logger.log_warning(f"Invalid data types - Image: {type(image)}, Mask: {type(mask)}")
            return False
        
        return True
    
    def split_train_val(self):
        """分割训练和验证集"""
        np.random.seed(42)
        indices = np.random.permutation(len(self.all_patches))
        split_idx = int(len(self.all_patches) * self.train_ratio)
        
        self.train_patches = [self.all_patches[i] for i in indices[:split_idx]]
        self.val_patches = [self.all_patches[i] for i in indices[split_idx:]]
        
        # 更新摘要数据
        self.logger.log_info(f"Data split completed for model {self.model_name}:")
        self.logger.log_info(f"  Training patches: {len(self.train_patches)}")
        self.logger.log_info(f"  Validation patches: {len(self.val_patches)}")
    
    def create_dataloaders(self, batch_size=4, num_workers=2):
        """创建鲁棒的micro_sam数据加载器，修复采样问题"""
        self.logger.log_info(f"Creating robust micro_sam dataloaders for model {self.model_name} with {len(self.train_patches)} train and {len(self.val_patches)} val patches")
        
        # 验证数据集中的文件格式和内容
        self.logger.log_info("Validating saved patch formats and content...")
        valid_train_patches = []
        valid_val_patches = []
        
        # 验证训练数据
        for i, patch_info in enumerate(self.train_patches):
            try:
                img_path = patch_info['img_path']
                mask_path = patch_info['mask_path']
                
                # 检查文件是否存在
                if not Path(img_path).exists() or not Path(mask_path).exists():
                    self.logger.log_warning(f"Missing files for patch {i}: {img_path} or {mask_path}")
                    continue
                
                # 尝试加载并验证数据
                img = imageio.imread(img_path)
                mask = imageio.imread(mask_path)
                
                # 验证格式
                if len(img.shape) != 2 or len(mask.shape) != 2:
                    self.logger.log_warning(f"Invalid shape - Image: {img.shape}, Mask: {mask.shape}")
                    continue
                
                if img.shape != mask.shape:
                    self.logger.log_warning(f"Shape mismatch - Image: {img.shape}, Mask: {mask.shape}")
                    continue
                
                # 验证掩码内容 - 这是关键检查
                unique_values = np.unique(mask)
                foreground_pixels = np.sum(mask > 0)
                
                # 确保掩码有足够的前景内容
                if foreground_pixels < 50:  # 降低要求到50个前景像素
                    self.logger.log_warning(f"Insufficient foreground pixels ({foreground_pixels}) in mask: {mask_path}")
                    continue
                
                # 检查标签值是否合理
                if len(unique_values) < 2:  # 至少要有背景和前景
                    self.logger.log_warning(f"Invalid mask labels in {mask_path}: {unique_values}")
                    continue
                
                valid_train_patches.append(patch_info)
                
            except Exception as e:
                self.logger.log_warning(f"Error validating train patch {i}: {e}")
                continue
        
        # 验证验证数据
        for i, patch_info in enumerate(self.val_patches):
            try:
                img_path = patch_info['img_path']
                mask_path = patch_info['mask_path']
                
                if not Path(img_path).exists() or not Path(mask_path).exists():
                    continue
                
                img = imageio.imread(img_path)
                mask = imageio.imread(mask_path)
                
                if (len(img.shape) != 2 or len(mask.shape) != 2 or 
                    img.shape != mask.shape or np.sum(mask > 0) < 50 or
                    len(np.unique(mask)) < 2):
                    continue
                
                valid_val_patches.append(patch_info)
                
            except Exception as e:
                continue
        
        # 更新有效补丁列表
        self.train_patches = valid_train_patches
        self.val_patches = valid_val_patches
        
        self.logger.log_info(f"After validation: {len(self.train_patches)} valid train patches, {len(self.val_patches)} valid val patches")
        
        # 检查是否有足够的有效数据
        if len(self.train_patches) < 20:
            raise ValueError(f"Insufficient valid training data: only {len(self.train_patches)} valid patches")
        
        if len(self.val_patches) < 5:
            raise ValueError(f"Insufficient valid validation data: only {len(self.val_patches)} valid patches")
        
        try:
            # 收集有效补丁路径
            train_imgs = [patch_info['img_path'] for patch_info in self.train_patches]
            train_masks = [patch_info['mask_path'] for patch_info in self.train_patches]
            val_imgs = [patch_info['img_path'] for patch_info in self.val_patches]
            val_masks = [patch_info['mask_path'] for patch_info in self.val_patches]
            
            self.logger.log_info(f"Creating robust micro_sam dataloaders for model {self.model_name}...")
            
            # 移除采样器，使用更简单的数据加载方式
            sampler = None
            
            # 简化标签变换，避免复杂的采样问题
            from torch_em.transform.raw import normalize
            
            def robust_raw_transform(raw):
                """更鲁棒的原始数据变换"""
                try:
                    # 确保数据在[0,1]范围内
                    if raw.max() > 1:
                        raw = raw.astype(np.float32) / 255.0
                    
                    # 再转回[0,255]范围（micro_sam期望的输入）
                    raw = (raw * 255).astype(np.float32)
                    return raw
                except Exception:
                    # 出错时返回原数据
                    return raw.astype(np.float32)
            
            # 简化的标签变换，避免复杂的距离变换导致采样失败
            def robust_label_transform(labels):
                """简化但鲁棒的标签变换"""
                try:
                    # 确保标签是整数类型
                    if labels.dtype != np.uint8:
                        labels = labels.astype(np.uint8)
                    
                    # 检查并修复标签值
                    unique_labels = np.unique(labels)
                    if len(unique_labels) < 2:
                        # 如果只有一个标签值，创建一个简单的前景/背景
                        labels = (labels > 0).astype(np.uint8)
                    
                    # 返回float32类型以兼容PerObjectDistanceTransform
                    return labels.astype(np.float32)
                except Exception:
                    # 出错时返回二值化的标签
                    return (labels > 0).astype(np.float32)
            
            # 直接使用torch_em的default_segmentation_loader，但配置更鲁棒的参数
            self.logger.log_info("Using torch_em.default_segmentation_loader with robust configuration...")
            
            import torch_em
            
            # 训练数据加载器 - 使用更宽松的配置
            train_loader = torch_em.default_segmentation_loader(
                raw_paths=train_imgs,
                raw_key=None,
                label_paths=train_masks,
                label_key=None,
                patch_shape=(512, 512),
                batch_size=batch_size,
                num_workers=0,  # 避免多进程问题
                shuffle=True,
                ndim=2,
                is_seg_dataset=True,
                raw_transform=robust_raw_transform,
                label_transform=robust_label_transform,
                # 关键：移除或简化采样器相关参数
                sampler=None,
                # 增加容错性
                ensure_image_id_consistency=False,
                # 允许更多的采样尝试，但设置合理上限
                n_samples=None,  # 使用所有样本
            )
            
            # 验证数据加载器
            val_loader = torch_em.default_segmentation_loader(
                raw_paths=val_imgs,
                raw_key=None,
                label_paths=val_masks,
                label_key=None,
                patch_shape=(512, 512),
                batch_size=1,
                num_workers=0,
                shuffle=False,
                ndim=2,
                is_seg_dataset=True,
                raw_transform=robust_raw_transform,
                label_transform=robust_label_transform,
                sampler=None,
                ensure_image_id_consistency=False,
                n_samples=None,
            )
            
            self.logger.log_info(f"Created torch_em dataloaders for model {self.model_name} - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
            
            # 测试数据加载器的健壮性
            self.logger.log_info(f"Testing dataloader robustness for model {self.model_name}...")
            test_success = False
            test_attempts = 0
            max_test_attempts = 5  # 减少测试尝试次数
            
            while not test_success and test_attempts < max_test_attempts:
                try:
                    test_attempts += 1
                    self.logger.log_info(f"Dataloader test attempt {test_attempts}/{max_test_attempts}")
                    
                    # 测试训练加载器
                    train_iter = iter(train_loader)
                    sample_batch = next(train_iter)
                    
                    if len(sample_batch) >= 2:
                        img_batch = sample_batch[0]
                        mask_batch = sample_batch[1]
                        
                        self.logger.log_info(f"Test successful - Batch shapes: Images: {img_batch.shape}, Masks: {mask_batch.shape}")
                        self.logger.log_info(f"Image data range: [{img_batch.min():.2f}, {img_batch.max():.2f}]")
                        self.logger.log_info(f"Mask data range: [{mask_batch.min():.2f}, {mask_batch.max():.2f}]")
                        
                        # 检查掩码内容
                        unique_values = torch.unique(mask_batch)
                        self.logger.log_info(f"Mask unique values: {unique_values}")
                        
                        test_success = True
                        
                except Exception as e:
                    self.logger.log_warning(f"Dataloader test attempt {test_attempts} failed: {e}")
                    if test_attempts >= max_test_attempts:
                        # 如果标准加载器失败，尝试创建最简单的数据加载器
                        self.logger.log_info("Standard dataloader failed, trying fallback solution...")
                        return self._create_fallback_dataloaders(train_imgs, train_masks, val_imgs, val_masks, batch_size)
                    
                    # 等待一会再重试
                    import time
                    time.sleep(1)
            
            if not test_success:
                raise Exception("Failed to create working dataloaders after multiple attempts")
            
            return train_loader, val_loader
            
        except Exception as e:
            self.logger.log_error(f"Error creating torch_em dataloaders for model {self.model_name}: {e}")
            import traceback
            self.logger.log_error(traceback.format_exc())
            
            # 尝试备用方案
            self.logger.log_info("Attempting fallback dataloader creation...")
            return self._create_fallback_dataloaders(train_imgs, train_masks, val_imgs, val_masks, batch_size)

    def _create_fallback_dataloaders(self, train_imgs, train_masks, val_imgs, val_masks, batch_size):
        """创建备用数据加载器，绕过torch_em的复杂采样机制"""
        self.logger.log_info("Creating fallback dataloaders using simple torch DataLoader...")
        
        from torch.utils.data import DataLoader, Dataset
        import torch
        
        class SimpleMicroSAMDataset(Dataset):
            """简单的数据集类，避免复杂的采样问题"""
            
            def __init__(self, img_paths, mask_paths, logger):
                self.img_paths = img_paths
                self.mask_paths = mask_paths
                self.logger = logger
                
                # 预验证所有数据
                self.valid_indices = []
                for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
                    try:
                        img = imageio.imread(img_path)
                        mask = imageio.imread(mask_path)
                        
                        if (len(img.shape) == 2 and len(mask.shape) == 2 and 
                            img.shape == mask.shape == (512, 512) and
                            np.sum(mask > 0) >= 50):
                            self.valid_indices.append(i)
                    except:
                        continue
                
                self.logger.log_info(f"SimpleMicroSAMDataset: {len(self.valid_indices)}/{len(img_paths)} valid samples")
            
            def __len__(self):
                return len(self.valid_indices)
            
            def __getitem__(self, idx):
                real_idx = self.valid_indices[idx]
                img_path = self.img_paths[real_idx]
                mask_path = self.mask_paths[real_idx]
                
                try:
                    # 加载图像和掩码
                    img = imageio.imread(img_path).astype(np.float32)
                    mask = imageio.imread(mask_path).astype(np.float32)
                    
                    # 标准化图像到[0,1]
                    if img.max() > 1:
                        img = img / 255.0
                    
                    # 确保掩码是实例分割格式
                    if mask.max() <= 1:
                        mask = mask * 255  # 如果是0-1范围，转换为0-255
                    
                    # 转换为张量
                    img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
                    mask_tensor = torch.from_numpy(mask).long()      # (H, W)
                    
                    return img_tensor, mask_tensor
                    
                except Exception as e:
                    # 返回零张量作为fallback
                    self.logger.log_warning(f"Error loading sample {idx}: {e}")
                    return (torch.zeros(1, 512, 512, dtype=torch.float32), 
                        torch.zeros(512, 512, dtype=torch.long))
        
        # 创建数据集
        train_dataset = SimpleMicroSAMDataset(train_imgs, train_masks, self.logger)
        val_dataset = SimpleMicroSAMDataset(val_imgs, val_masks, self.logger)
        
        if len(train_dataset) < 10 or len(val_dataset) < 2:
            raise ValueError(f"Insufficient valid data after fallback: train={len(train_dataset)}, val={len(val_dataset)}")
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,
            drop_last=True,
            pin_memory=False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=0,
            drop_last=False,
            pin_memory=False
        )
        
        # 测试备用加载器
        try:
            train_iter = iter(train_loader)
            sample_batch = next(train_iter)
            img_batch, mask_batch = sample_batch
            
            self.logger.log_info(f"Fallback dataloader test successful:")
            self.logger.log_info(f"  Image batch shape: {img_batch.shape}")
            self.logger.log_info(f"  Mask batch shape: {mask_batch.shape}")
            self.logger.log_info(f"  Image range: [{img_batch.min():.3f}, {img_batch.max():.3f}]")
            self.logger.log_info(f"  Mask range: [{mask_batch.min()}, {mask_batch.max()}]")
            
        except Exception as e:
            raise Exception(f"Even fallback dataloader failed: {e}")
        
        return train_loader, val_loader



def diagnose_and_fix_data_issues(json_mapping_files, logger):
    """诊断和修复数据问题"""
    logger.log_info("Diagnosing data quality issues...")
    
    total_images = 0
    valid_images = 0
    corrupted_images = 0
    empty_masks = 0
    small_masks = 0
    
    for json_file in json_mapping_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            dataset_name = data.get('dataset_name', Path(json_file).stem)
            logger.log_info(f"Checking dataset: {dataset_name}")
            
            images_path_str = str(data['images_path'])
            masks_path_str = str(data['masks_path'])
            
            # 修复双重路径
            if '/Retrain/Retrain/' in images_path_str:
                images_path_str = images_path_str.replace('/Retrain/Retrain/', '/Retrain/')
            if '/Retrain/Retrain/' in masks_path_str:
                masks_path_str = masks_path_str.replace('/Retrain/Retrain/', '/Retrain/')
            
            images_path = Path(images_path_str)
            masks_path = Path(masks_path_str)
            
            if not images_path.exists() or not masks_path.exists():
                logger.log_error(f"Dataset paths do not exist: {images_path}, {masks_path}")
                continue
            
            dataset_valid = 0
            dataset_total = len(data['mapping'])
            
            for img_name, info in data['mapping'].items():
                total_images += 1
                
                if not isinstance(info, dict) or not info.get('mask_file'):
                    continue
                
                img_path = images_path / img_name
                mask_path = masks_path / info['mask_file']
                
                if not img_path.exists() or not mask_path.exists():
                    continue
                
                try:
                    # 快速检查文件
                    img = imageio.imread(str(img_path))
                    mask = imageio.imread(str(mask_path))
                    
                    if len(img.shape) < 2 or len(mask.shape) < 2:
                        corrupted_images += 1
                        continue
                    
                    # 检查掩码质量
                    mask_2d = mask[:, :, 0] if len(mask.shape) > 2 else mask
                    foreground_pixels = np.sum(mask_2d > 0)
                    
                    if foreground_pixels == 0:
                        empty_masks += 1
                        continue
                    
                    if foreground_pixels < 100:
                        small_masks += 1
                        continue
                    
                    valid_images += 1
                    dataset_valid += 1
                    
                except Exception as e:
                    corrupted_images += 1
                    continue
            
            logger.log_info(f"  {dataset_name}: {dataset_valid}/{dataset_total} valid images")
            
        except Exception as e:
            logger.log_error(f"Error checking {json_file}: {e}")
            continue
    
    logger.log_info("Data Quality Summary:")
    logger.log_info(f"  Total images checked: {total_images}")
    logger.log_info(f"  Valid images: {valid_images}")
    logger.log_info(f"  Corrupted images: {corrupted_images}")
    logger.log_info(f"  Empty masks: {empty_masks}")
    logger.log_info(f"  Small masks (<100 pixels): {small_masks}")
    logger.log_info(f"  Data quality: {valid_images/total_images*100:.1f}%")
    
    if valid_images < 100:
        raise ValueError(f"Insufficient valid data for training: only {valid_images} valid images found")
    
    return valid_images

class EnhancedMicroSAMTrainer:
    """增强的训练器，带详细日志记录和wandb集成，支持实时监控"""
    
    def __init__(self, save_dir, val_patches, logger):
        self.save_dir = Path(save_dir)
        self.checkpoint_dir = self.save_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.val_patches = val_patches
        self.best_dice = 0.0
        self.training_history = []
        
        # 训练统计
        self.epoch_times = []
        self.current_epoch_start = None
        
        # wandb配置
        self.wandb_run = None
        
        # 监控相关
        self.monitoring_active = False
        self.monitoring_thread = None
        self.last_logged_epoch = -1
    
    def init_wandb(self, model_name, epochs, learning_rate, batch_size, n_objects_per_batch, 
                   train_size, val_size, model_type="vit_b_lm"):
        """初始化wandb"""
        try:
            self.wandb_run = wandb.init(
                project="Lead",  # 设置为你要求的项目名
                name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "model_name": model_name,
                    "model_type": model_type,
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "n_objects_per_batch": n_objects_per_batch,
                    "train_size": train_size,
                    "val_size": val_size,
                    "patch_size": 512,
                    "overlap": 10,
                    "architecture": "MicroSAM",
                    "with_segmentation_decoder": True,
                    "early_stopping": 15,
                    "scheduler_patience": 5,
                    "scheduler_factor": 0.5,
                    "min_lr": 1e-7
                }
            )
            self.logger.log_info("Wandb initialized successfully with project 'Lead'")
            return True
        except Exception as e:
            self.logger.log_error(f"Failed to initialize wandb: {e}")
            return False
    
    def log_epoch_start(self, epoch, total_epochs, lr=None):
        """记录epoch开始"""
        self.current_epoch_start = time.time()
        start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.logger.log_info(f"Epoch {epoch}/{total_epochs} started at {start_time_str}")
        if lr:
            self.logger.log_info(f"  Learning rate: {lr}")
        
        self.logger.log_training_details(f"EPOCH {epoch} START - LR: {lr}")
    
    def log_epoch_end(self, epoch, train_loss=None, val_loss=None, samples_processed=None, lr=None):
        """记录epoch结束，并发送到wandb"""
        if self.current_epoch_start is None:
            return
        
        end_time = time.time()
        duration = end_time - self.current_epoch_start
        end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        duration_formatted = f"{int(duration//3600):02d}:{int((duration%3600)//60):02d}:{int(duration%60):02d}"
        
        # GPU内存使用
        memory_used = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        
        self.logger.log_info(f"Epoch {epoch} completed in {duration_formatted}")
        self.logger.log_info(f"  Duration: {duration:.2f} seconds")
        if train_loss:
            self.logger.log_info(f"  Training loss: {train_loss:.6f}")
        if val_loss:
            self.logger.log_info(f"  Validation loss: {val_loss:.6f}")
        if samples_processed:
            avg_time = duration / samples_processed
            self.logger.log_info(f"  Samples processed: {samples_processed}")
            self.logger.log_info(f"  Average time per sample: {avg_time:.4f} seconds")
        self.logger.log_info(f"  GPU memory used: {memory_used:.2f} GB")
        
        # 记录到CSV
        epoch_data = {
            'epoch': epoch,
            'start_time': datetime.fromtimestamp(self.current_epoch_start).strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': end_time_str,
            'duration_seconds': round(duration, 2),
            'duration_formatted': duration_formatted,
            'samples_processed': samples_processed or '',
            'avg_time_per_sample': round(duration / samples_processed, 4) if samples_processed else '',
            'learning_rate': lr or '',
            'train_loss': round(train_loss, 6) if train_loss else '',
            'val_loss': round(val_loss, 6) if val_loss else '',
            'memory_used_gb': round(memory_used, 2)
        }
        
        self.logger.log_epoch_timing(epoch_data)
        self.logger.log_training_details(f"EPOCH {epoch} END - Duration: {duration_formatted}, Loss: {train_loss}")
        
        # 发送到wandb
        if self.wandb_run is not None:
            wandb_metrics = {
                "epoch": epoch,
                "epoch_duration": duration,
                "memory_used_gb": memory_used,
            }
            
            if train_loss is not None:
                wandb_metrics["train_loss"] = train_loss
            if val_loss is not None:
                wandb_metrics["val_loss"] = val_loss
            if lr is not None:
                wandb_metrics["learning_rate"] = lr
            if samples_processed is not None:
                wandb_metrics["samples_processed"] = samples_processed
                wandb_metrics["avg_time_per_sample"] = duration / samples_processed
            
            self.wandb_run.log(wandb_metrics)
        
        self.current_epoch_start = None
    
    def _monitor_training_logs(self, model_name, log_dir):
        """监控训练日志文件并实时上传到wandb"""
        self.logger.log_info("Started training log monitoring thread")
        
        checkpoint_dir = self.checkpoint_dir / model_name
        
        while self.monitoring_active:
            try:
                # 监控tensorboard日志获取训练指标
                if checkpoint_dir.exists():
                    tb_log_dir = checkpoint_dir / "logs"
                    if tb_log_dir.exists():
                        self._parse_tensorboard_logs(tb_log_dir)
                
                time.sleep(60)  # 每60秒检查一次
                
            except Exception as e:
                self.logger.log_error(f"Error in monitoring thread: {e}")
                time.sleep(120)
    
    def _parse_tensorboard_logs(self, tb_log_dir):
        """解析tensorboard日志获取训练指标并上传到wandb"""
        try:
            # 查找最新的event文件
            event_files = list(tb_log_dir.glob("events.out.tfevents.*"))
            if not event_files:
                return
            
            latest_event_file = max(event_files, key=os.path.getmtime)
            
            # 使用简单的二进制读取方式，避免tensorflow依赖
            self._simple_parse_tensorboard(latest_event_file)
                        
        except Exception as e:
            self.logger.log_warning(f"Failed to parse tensorboard logs: {e}")
    
    def _simple_parse_tensorboard(self, event_file):
        """简单解析tensorboard事件文件"""
        try:
            import struct
            
            # 监控关键指标
            target_metrics = [
                b'train/loss', b'train/mask_loss', b'train/iou_loss', b'train/model_iou',
                b'validation/loss', b'validation/mask_loss', b'validation/iou_loss', b'validation/model_iou',
                b'train/learning_rate'
            ]
            
            new_metrics = {}
            
            # 读取文件的最后几KB，寻找最新的数据
            try:
                with open(event_file, 'rb') as f:
                    f.seek(-8192, 2)  # 从文件末尾往前8KB
                    data = f.read()
                    
                    # 寻找指标名称
                    for metric in target_metrics:
                        if metric in data:
                            # 找到指标，尝试提取数值（简化处理）
                            metric_name = metric.decode('utf-8')
                            wandb_name = self._convert_metric_name(metric_name)
                            
                            # 这里可以实现更复杂的数值提取逻辑
                            # 现在先跳过具体数值提取
                            pass
                            
            except Exception:
                pass
                
        except Exception as e:
            pass  # 静默处理解析错误
    
    def _convert_metric_name(self, tb_name):
        """转换tensorboard指标名为wandb格式"""
        mapping = {
            'train/loss': 'train_total_loss',
            'train/mask_loss': 'train_mask_loss',
            'train/iou_loss': 'train_iou_loss',
            'train/model_iou': 'train_predicted_iou',
            'train/learning_rate': 'learning_rate',
            'validation/loss': 'val_total_loss',
            'validation/mask_loss': 'val_mask_loss',
            'validation/iou_loss': 'val_iou_loss',
            'validation/model_iou': 'val_predicted_iou'
        }
        return mapping.get(tb_name, tb_name.replace('/', '_'))
    
    def start_monitoring(self, model_name):
        """启动监控线程"""
        if self.wandb_run is None:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_training_logs,
            args=(model_name, self.checkpoint_dir),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.log_info("Training monitoring started")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        self.logger.log_info("Training monitoring stopped")
    
    def train_model_with_detailed_logging(self, train_loader, val_loader, model_name="microsam_custom", epochs=100):
            """使用官方的micro_sam训练，完全基于官方示例，集成wandb和实时监控"""
            self.logger.log_info("="*60)
            self.logger.log_info(f"OFFICIAL TRAINING STARTED: {model_name}")
            self.logger.log_info(f"Total epochs: {epochs}")
            self.logger.log_info(f"Training samples: {len(train_loader.dataset)}")
            self.logger.log_info(f"Validation samples: {len(val_loader.dataset)}")
            self.logger.log_info(f"Batch size: {train_loader.batch_size}")
            self.logger.log_info("="*60)
            
            # 初始化wandb
            learning_rate = 1e-5
            wandb_initialized = self.init_wandb(
                model_name=model_name,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=train_loader.batch_size,
                n_objects_per_batch=25,
                train_size=len(train_loader.dataset),
                val_size=len(val_loader.dataset),
                model_type="vit_b_lm"
            )
            
            # 启动监控
            if wandb_initialized:
                self.start_monitoring(model_name)
            
            training_start_time = time.time()
            
            try:
                # 记录训练开始时的信息 - 修复版本
                if self.wandb_run is not None:
                    # 使用summary记录状态信息
                    self.wandb_run.summary.update({
                        'training_started': True,
                        'model_name': model_name,
                        'start_time': datetime.fromtimestamp(training_start_time).isoformat()
                    })
                    
                    # 使用log记录数值指标
                    self.wandb_run.log({
                        'timing/start_timestamp': training_start_time,
                        'config/total_epochs': epochs,
                        'timing/estimated_duration_hours': epochs * 0.1,
                        'status/initialization': 1
                    })
                
                # 使用官方的micro_sam训练
                self.logger.log_info("Using official micro_sam training pipeline...")
                
                train_sam(
                    name=model_name,
                    model_type="vit_b_lm",
                    train_loader=train_loader,
                    val_loader=val_loader,
                    n_epochs=epochs,
                    n_objects_per_batch=25,
                    with_segmentation_decoder=True,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    lr=learning_rate,
                    early_stopping=15,
                    save_root=str(self.checkpoint_dir),
                    save_every_kth_epoch=5,
                    scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
                    scheduler_kwargs={'patience': 5, 'factor': 0.5, 'min_lr': 1e-7},
                    overwrite_training=True
                )
                
                # 训练完成后导出模型
                self.export_trained_model(model_name)
                
            except Exception as e:
                self.logger.log_error(f"Training failed: {e}")
                if self.wandb_run is not None:
                    # 修复版本 - 使用summary记录错误状态
                    self.wandb_run.summary.update({
                        'training_failed': True,
                        'error_message': str(e)[:500],  # 限制长度
                        'failure_time': datetime.now().isoformat()
                    })
                    
                    # 使用log记录数值错误指标
                    self.wandb_run.log({
                        'status/error_occurred': 1,
                        'status/training_failed': 1
                    })
                import traceback
                self.logger.log_error(traceback.format_exc())
                raise
            finally:
                # 停止监控
                self.stop_monitoring()
                
                # 记录训练完成信息 - 修复版本
                training_end_time = time.time()
                total_duration = training_end_time - training_start_time
                
                if self.wandb_run is not None:
                    # 使用summary记录最终状态
                    self.wandb_run.summary.update({
                        'training_completed': True,
                        'total_duration_formatted': f"{int(total_duration//3600):02d}:{int((total_duration%3600)//60):02d}:{int(total_duration%60):02d}",
                        'end_time': datetime.fromtimestamp(training_end_time).isoformat(),
                        'final_status': 'completed'
                    })
                    
                    # 使用log记录数值指标
                    self.wandb_run.log({
                        'timing/total_duration_seconds': total_duration,
                        'timing/total_duration_hours': total_duration / 3600,
                        'timing/end_timestamp': training_end_time,
                        'status/completion': 1,
                        'performance/avg_seconds_per_epoch': total_duration / epochs
                    })
                    
                    # 完成wandb run
                    self.wandb_run.finish()
                    self.logger.log_info("Wandb run finished")
            
            total_duration_formatted = f"{int(total_duration//3600):02d}:{int((total_duration%3600)//60):02d}:{int(total_duration%60):02d}"
            
            self.logger.log_info("="*60)
            self.logger.log_info(f"TRAINING COMPLETED: {model_name}")
            self.logger.log_info(f"Total training time: {total_duration_formatted}")
            self.logger.log_info(f"Average time per epoch: {total_duration/epochs:.2f} seconds")
            self.logger.log_info("="*60)
        
           
    def _create_custom_logger(self, model_name):
        """创建自定义logger来监控训练过程"""
        class WandbTrainingLogger:
            def __init__(self, wandb_run, parent_logger):
                self.wandb_run = wandb_run
                self.parent_logger = parent_logger
                self.current_epoch = 0
                
            def log_train(self, step, loss, lr, *args, **kwargs):
                """记录训练指标"""
                if self.wandb_run is not None:
                    self.wandb_run.log({
                        "train_loss": loss,
                        "learning_rate": lr,
                        "step": step
                    })
                    
            def log_validation(self, step, metric, loss, *args, **kwargs):
                """记录验证指标"""
                if self.wandb_run is not None:
                    self.wandb_run.log({
                        "val_loss": loss,
                        "val_metric": metric,
                        "step": step
                    })
        
        return WandbTrainingLogger(self.wandb_run, self.logger)
    
    # def export_trained_model(self, model_name):
    #     """导出训练好的模型，基于官方示例"""
    #     try:
    #         self.logger.log_info("Exporting trained model...")
            
    #         # 设置导出路径
    #         export_path = self.save_dir / f"{model_name}_exported.pth"
    #         checkpoint_path = self.checkpoint_dir / model_name / "best.pt"
            
    #         # 检查checkpoint是否存在
    #         if not checkpoint_path.exists():
    #             self.logger.log_warning(f"Checkpoint not found at {checkpoint_path}, trying latest.pt")
    #             checkpoint_path = self.checkpoint_dir / model_name / "latest.pt"
                
    #         if checkpoint_path.exists():
    #             # 使用官方的导出函数
    #             export_custom_sam_model(
    #                 checkpoint_path=str(checkpoint_path),
    #                 model_type="vit_b_lm",
    #                 save_path=str(export_path),
    #                 with_segmentation_decoder=True
    #             )
                
    #             self.logger.log_info(f"Model exported successfully to: {export_path}")
                
    #             # 记录模型路径到wandb
    #             if self.wandb_run is not None:
    #                 self.wandb_run.log({
    #                     "model_exported": True, 
    #                     "export_path": str(export_path),
    #                     "model_size_mb": export_path.stat().st_size / 1024**2
    #                 })
                
    #         else:
    #             self.logger.log_error(f"No checkpoint found for model {model_name}")
    #             if self.wandb_run is not None:
    #                 self.wandb_run.log({"model_export_failed": True, "reason": "No checkpoint found"})
                
    #     except Exception as e:
    #         self.logger.log_error(f"Failed to export model: {e}")
    #         if self.wandb_run is not None:
    #             self.wandb_run.log({"model_export_failed": True, "reason": str(e)})
    #         import traceback
    #         self.logger.log_error(traceback.format_exc())
    
    def export_trained_model(self, model_name):
        """导出训练好的模型，基于官方示例"""
        try:
            self.logger.log_info("Exporting trained model...")
            
            # 设置导出路径
            export_path = self.save_dir / f"{model_name}_exported.pth"
            checkpoint_path = self.checkpoint_dir / model_name / "best.pt"
            
            # 检查checkpoint是否存在
            if not checkpoint_path.exists():
                self.logger.log_warning(f"Checkpoint not found at {checkpoint_path}, trying latest.pt")
                checkpoint_path = self.checkpoint_dir / model_name / "latest.pt"
                
            if checkpoint_path.exists():
                # 使用官方的导出函数
                export_custom_sam_model(
                    checkpoint_path=str(checkpoint_path),
                    model_type="vit_b_lm",
                    save_path=str(export_path),
                    with_segmentation_decoder=True
                )
                
                self.logger.log_info(f"Model exported successfully to: {export_path}")
                
                # 记录模型路径到wandb - 修复版本
                if self.wandb_run is not None:
                    model_size_mb = export_path.stat().st_size / 1024**2
                    
                    # 使用summary记录模型信息
                    self.wandb_run.summary.update({
                        'model_exported': True,
                        'export_path': str(export_path),
                        'model_size_mb': model_size_mb
                    })
                    
                    # 使用log记录数值指标
                    self.wandb_run.log({
                        'model/size_mb': model_size_mb,
                        'model/export_success': 1
                    })
                
            else:
                self.logger.log_error(f"No checkpoint found for model {model_name}")
                if self.wandb_run is not None:
                    self.wandb_run.summary.update({
                        'model_export_failed': True,
                        'export_error': 'No checkpoint found'
                    })
                    self.wandb_run.log({'model/export_failed': 1})
                
        except Exception as e:
            self.logger.log_error(f"Failed to export model: {e}")
            if self.wandb_run is not None:
                self.wandb_run.summary.update({
                    'model_export_failed': True,
                    'export_error': str(e)[:200]
                })
                self.wandb_run.log({'model/export_failed': 1})
            import traceback
            self.logger.log_error(traceback.format_exc())  
  





def main():
    """优化后的主函数，支持基于模型名的缓存管理和简洁的模型参数分析，仅支持LM模型类型"""
    # 添加命令行参数支持
    import argparse
    parser = argparse.ArgumentParser(description="MicroSAM Training with LM Model Types Support")
    
    # 添加模型类型选择参数 - 仅支持LM模型
    parser.add_argument("--model-type", "-mt", type=str, 
                       choices=["vit_t_lm", "vit_b_lm", "vit_l_lm"],
                       default="vit_b_lm",
                       help="选择SAM LM模型类型 (默认: vit_b_lm)")
    
    parser.add_argument("--force-regenerate", action="store_true", 
                       help="Force regenerate all patches even if cache exists")
    parser.add_argument("--clear-cache", action="store_true",
                       help="Clear existing patch cache before processing")
    parser.add_argument("--clear-all-caches", action="store_true",
                       help="Clear all model caches before processing")
    parser.add_argument("--epochs", type=int, default=80,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Training batch size (auto-adjust based on model type if not specified)")
    parser.add_argument("--n-objects", type=int, default=None,
                       help="Number of objects per batch (auto-adjust based on model type if not specified)")
    parser.add_argument("--learning-rate", type=float, default=None,
                       help="Learning rate (auto-adjust based on model type if not specified)")
    parser.add_argument("--model-name", type=str, default=None,
                       help="Override model name for cache management")
    parser.add_argument("--skip-param-analysis", action="store_true",
                       help="Skip detailed parameter analysis (faster startup)")
    parser.add_argument("--detailed-param-analysis", action="store_true",
                       help="Show detailed parameter analysis for all configurations")
    
    args = parser.parse_args()
    
    # 获取选择的模型类型
    selected_model_type = args.model_type
    
    # 根据模型类型获取推荐配置 - 仅LM模型
    def get_model_config(model_type):
        """根据LM模型类型获取推荐配置"""
        configs = {
            "vit_t_lm": {
                "n_objects_per_batch": 50,
                "batch_size": 8,
                "learning_rate": 2e-5,
                "description": "Tiny LM模型 - 针对显微镜优化的轻量版本",
                "memory_gb_estimate": 8,
                "training_time_estimate": "30-60分钟",
                "accuracy_tier": "良好"
            },
            "vit_b_lm": {
                "n_objects_per_batch": 35,
                "batch_size": 4,
                "learning_rate": 1e-5,
                "description": "Base LM模型 - 针对显微镜优化的标准版本",
                "memory_gb_estimate": 16,
                "training_time_estimate": "1-2小时",
                "accuracy_tier": "优秀"
            },
            "vit_l_lm": {
                "n_objects_per_batch": 25,
                "batch_size": 2,
                "learning_rate": 8e-6,
                "description": "Large LM模型 - 高性能显微镜专用版本",
                "memory_gb_estimate": 32,
                "training_time_estimate": "2-4小时",
                "accuracy_tier": "出色"
            }
        }
        return configs.get(model_type, configs["vit_b_lm"])
    
    # 获取模型配置
    model_config = get_model_config(selected_model_type)
    
    # 应用用户指定的参数，否则使用推荐值
    final_batch_size = args.batch_size if args.batch_size is not None else model_config["batch_size"]
    final_n_objects = args.n_objects if args.n_objects is not None else model_config["n_objects_per_batch"]
    final_learning_rate = args.learning_rate if args.learning_rate is not None else model_config["learning_rate"]
    
    # 设置cache目录
    os.environ["MICROSAM_CACHEDIR"] = "/LD-FS/data/Model/micro_sam"
    
    # 如果提供了自定义模型名，使用它；否则基于模型类型生成
    if args.model_name:
        custom_model_name = args.model_name
    else:
        custom_model_name = f"customize_{selected_model_type}"
    
    # 配置 - 获取今天的日期
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(f"/LD-FS/home/yunshuchen/micro_sam/Retrain/Training_{custom_model_name}_{selected_model_type}_{timestamp}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化详细logger
    logger = DetailedLogger(save_dir)
    
    # JSON映射文件
    json_mapping_files = [
        # "/LD-FS/data/public_dataset/Retrain/mappings/YIM_mapping.json",
        # "/LD-FS/data/public_dataset/Retrain/mappings/Omnipose_mapping.json",
        # "/LD-FS/data/public_dataset/Retrain/mappings/Hoechst_33342-stained_nuclei_mapping.json",
        # "/LD-FS/data/public_dataset/Retrain/mappings/Fluorescent_Neuronal_Cells_1.0_mapping.json",
        # "/LD-FS/data/public_dataset/Retrain/mappings/MDA-MB-231&BT20_cells_mapping.json",
        # "/LD-FS/data/public_dataset/Retrain/mappings/Breast_cancer_cell_dataset_mapping.json",
        # "/LD-FS/data/public_dataset/Retrain/mappings/Stardist_AsPC1_from_TC_mapping.json",
        "/LD-FS/data/public_dataset/Retrain/mappings/BBBC005_mapping.json",
        "/LD-FS/data/public_dataset/Retrain/mappings/BBBC041_mapping.json",
        "/LD-FS/data/public_dataset/Retrain/mappings/Fluorescent_Neuronal_Cells_2.0green_mapping.json",
        "/LD-FS/data/public_dataset/Retrain/mappings/Fluorescent_Neuronal_Cells_2.0red_mapping.json",
        "/LD-FS/data/public_dataset/Retrain/mappings/Fluorescent_Neuronal_Cells_2.0yellow_mapping.json",
        "/LD-FS/data/public_dataset/Retrain/mappings/dynamicnuclearnet_mapping.json",
        "/LD-FS/data/public_dataset/Retrain/mappings/Live_Cell_Fluorescence_Microscopy_mapping.json"
    ]
    
    logger.log_info("Starting optimized MicroSAM training with LM model support...")
    logger.log_info("="*80)
    logger.log_info("TRAINING CONFIGURATION")
    logger.log_info("="*80)
    logger.log_info(f"Selected model type: {selected_model_type}")
    logger.log_info(f"Model description: {model_config['description']}")
    logger.log_info(f"Accuracy tier: {model_config['accuracy_tier']}")
    logger.log_info(f"Estimated training time: {model_config['training_time_estimate']}")
    logger.log_info(f"Estimated memory requirement: {model_config['memory_gb_estimate']} GB")
    logger.log_info(f"Model name: {custom_model_name}")
    logger.log_info(f"Batch size: {final_batch_size} (auto-adjusted)")
    logger.log_info(f"Objects per batch: {final_n_objects} (auto-adjusted)")
    logger.log_info(f"Learning rate: {final_learning_rate} (auto-adjusted)")
    logger.log_info(f"Training epochs: {args.epochs}")
    logger.log_info(f"Save directory: {save_dir}")
    logger.log_info(f"Processing {len(json_mapping_files)} datasets")
    logger.log_info("="*80)

    try:
        # Phase 0: 简洁的模型参数分析
    #     if not args.skip_param_analysis:
    #         logger.log_info("="*80)
    #         logger.log_info("Phase 0: Model Parameter Analysis")
    #         logger.log_info("="*80)
            
    #         # 创建静默分析器（不输出详细信息）
    #         silent_analyzer = ModelParameterAnalyzer(logger=None)
            
    #         # 定义训练配置 - 针对选定的模型类型
    #         training_configs = [
    #             {
    #                 "name": "完全微调",
    #                 "freeze": None,
    #                 "peft_kwargs": None,
    #                 "description": "所有参数可训练"
    #             },
    #             {
    #                 "name": "冻结图像编码器", 
    #                 "freeze": ["image_encoder"],
    #                 "peft_kwargs": None,
    #                 "description": "仅训练提示/掩码解码器"
    #             },
    #             {
    #                 "name": "LoRA微调",
    #                 "freeze": None,
    #                 "peft_kwargs": {"rank": 4, "attention_layers_to_update": [9, 10, 11]},
    #                 "description": "参数高效微调"
    #             },
    #             {
    #                 "name": "LoRA+冻结",
    #                 "freeze": ["prompt_encoder", "mask_decoder"],
    #                 "peft_kwargs": {"rank": 4, "attention_layers_to_update": [9, 10, 11]},
    #                 "description": "最小参数微调"
    #             }
    #         ]
            
    #         # 快速分析各种配置
    #         config_summaries = []
    #         logger.log_info(f"正在分析 {selected_model_type} 模型不同训练配置的参数分布...")
            
    #         for i, config in enumerate(training_configs, 1):
    #             if args.detailed_param_analysis:
    #                 logger.log_info(f"\n{i}. 分析{config['name']}配置...")
    #                 logger.log_info(f"   描述: {config['description']}")
    #                 # 详细分析
    #                 analysis = analyze_training_model_parameters(
    #                     model_type=selected_model_type,  # 使用选定的模型类型
    #                     freeze=config["freeze"],
    #                     peft_kwargs=config["peft_kwargs"],
    #                     logger=logger
    #                 )
    #             else:
    #                 # 静默分析，只获取数据
    #                 analysis = silent_analyzer.analyze_sam_model_parameters(
    #                     model_type=selected_model_type,  # 使用选定的模型类型
    #                     freeze=config["freeze"],
    #                     peft_kwargs=config["peft_kwargs"]
    #                 )
                
    #             if analysis:
    #                 summary = silent_analyzer.get_parameter_summary()
    #                 config_summaries.append({
    #                     "name": config["name"],
    #                     "description": config["description"],
    #                     "summary": summary,
    #                     "trainable_params": analysis.get('trainable_params', 0),
    #                     "trainable_ratio": analysis.get('trainable_ratio', 0),
    #                     "memory_mb": analysis.get('total_model_memory_mb', 0)
    #                 })
            
    #         # 生成简洁的参数对比表
    #         logger.log_info("="*80)
    #         logger.log_info(f"模型配置对比摘要 - {selected_model_type}")
    #         logger.log_info("="*80)
            
    #         # 表格格式输出
    #         logger.log_info("配置名称           | 可训练参数     | 训练比例  | 内存使用   | 描述")
    #         logger.log_info("-" * 75)
            
    #         for config_summary in config_summaries:
    #             name = config_summary['name'][:15].ljust(15)
    #             trainable = f"{config_summary['trainable_params']:,}".rjust(12)
    #             ratio = f"{config_summary['trainable_ratio']:.2f}%".rjust(8)
    #             memory = f"{config_summary['memory_mb']:.0f}MB".rjust(9)
    #             desc = config_summary['description']
                
    #             logger.log_info(f"{name} | {trainable} | {ratio} | {memory} | {desc}")
            
    #         logger.log_info("-" * 75)
            
    #         # 基于GPU内存和模型类型的推荐 - 针对LM模型
    #         gpu_memory_gb = 95.22  # 从日志中看到的H20内存
    #         logger.log_info(f"\n基于当前GPU内存 ({gpu_memory_gb:.0f}GB) 和模型类型 ({selected_model_type}) 的配置推荐:")
            
    #         # 针对LM模型的推荐策略
    #         if selected_model_type == "vit_t_lm":
    #             logger.log_info("  ✓ 推荐: 完全微调配置 (Tiny LM模型，资源充足)")
    #             logger.log_info("  ✓ 备选: LoRA微调配置 (进一步降低内存使用)")
    #             logger.log_info("  • 特点: 训练速度最快，适合快速原型验证")
    #         elif selected_model_type == "vit_b_lm":
    #             logger.log_info("  ✓ 推荐: 完全微调配置 (Base LM模型，内存充足)")
    #             logger.log_info("  ✓ 备选: LoRA微调配置 (更高内存效率)")
    #             logger.log_info("  • 特点: 性能和效率的最佳平衡，推荐用于生产")
    #         elif selected_model_type == "vit_l_lm":
    #             logger.log_info("  ✓ 推荐: 完全微调配置 (Large LM模型，当前内存支持)")
    #             logger.log_info("  ✓ 备选: LoRA微调配置 (推荐用于生产环境)")
    #             logger.log_info("  △ 可选: 冻结图像编码器配置 (如遇内存不足)")
    #             logger.log_info("  • 特点: 最高精度，适合对准确性要求极高的场景")
            
    #         # 保存分析报告
    #         analysis_report_path = save_dir / f"model_parameter_analysis_{selected_model_type}.json"
    #         if config_summaries:
    #             # 使用第一个配置（完全微调）的详细分析保存报告
    #             silent_analyzer.save_analysis_report(str(analysis_report_path))
            
    #         # 保存简洁的配置对比
    #         config_comparison = {
    #             "analysis_time": datetime.now().isoformat(),
    #             "model_type": selected_model_type,
    #             "model_family": "LM (Light Microscopy optimized)",
    #             "gpu_info": f"NVIDIA H20 ({gpu_memory_gb}GB)",
    #             "recommended_config": "完全微调",  # LM模型都推荐完全微调
    #             "training_params": {
    #                 "batch_size": final_batch_size,
    #                 "n_objects_per_batch": final_n_objects,
    #                 "learning_rate": final_learning_rate,
    #                 "epochs": args.epochs
    #             },
    #             "model_characteristics": {
    #                 "accuracy_tier": model_config["accuracy_tier"],
    #                 "training_time_estimate": model_config["training_time_estimate"],
    #                 "memory_requirement": f"{model_config['memory_gb_estimate']}GB",
    #                 "optimized_for": "显微镜图像分割"
    #             },
    #             "configurations": [
    #                 {
    #                     "name": cs["name"],
    #                     "description": cs["description"],
    #                     "trainable_params": cs["trainable_params"],
    #                     "trainable_ratio": round(cs["trainable_ratio"], 2),
    #                     "memory_mb": round(cs["memory_mb"], 1)
    #                 }
    #                 for cs in config_summaries
    #             ]
    #         }
            
    #         comparison_report_path = save_dir / f"training_configurations_comparison_{selected_model_type}.json"
    #         try:
    #             with open(comparison_report_path, 'w', encoding='utf-8') as f:
    #                 json.dump(config_comparison, f, indent=2, ensure_ascii=False, default=str)
    #             logger.log_info(f"\n报告已保存:")
    #             logger.log_info(f"  • 详细参数分析: {analysis_report_path.name}")
    #             logger.log_info(f"  • 配置对比摘要: {comparison_report_path.name}")
    #         except Exception as e:
    #             logger.log_error(f"保存配置对比报告失败: {e}")
            
    #         logger.log_info("="*80)
    #         logger.log_info("参数分析完成，开始数据处理...")
    #         logger.log_info("="*80)
    #     else:
    #         logger.log_info("跳过参数分析，直接开始数据处理...")
    # # Phase 0.5: 数据质量诊断（新增）
        # logger.log_info("="*80)
        # logger.log_info("Phase 0.5: Data Quality Diagnosis")
        # logger.log_info("="*80)
        
        # valid_image_count = diagnose_and_fix_data_issues(json_mapping_files, logger)
        # logger.log_info(f"Data diagnosis completed. Found {valid_image_count} valid images.")
        
        # if valid_image_count < 500:
        #     logger.log_warning(f"Limited data available ({valid_image_count} images). Consider adding more datasets for better training.")
        
        # Phase 1: 数据预处理和补丁提取
        logger.log_info("Phase 1: Dataset preprocessing and patch extraction")
        
        # 创建数据集处理器，传入模型名
        dataset_handler = OptimizedDatasetHandler(
            json_files=json_mapping_files,
            train_ratio=0.8,
            patch_size=512,
            overlap=10,
            logger=logger,
            force_regenerate=args.force_regenerate,
            model_name=custom_model_name
        )
        
        # 清除缓存选项
        if args.clear_cache:
            dataset_handler.clear_cache()
            logger.log_info(f"Cache cleared for model {custom_model_name}")
            return
        
        # 清除所有缓存选项
        if args.clear_all_caches:
            dataset_handler.clear_all_caches()
            logger.log_info("All model caches cleared")
            return
        
        # 检查数据量是否足够
        if len(dataset_handler.all_patches) < 100:
            logger.log_error(f"Insufficient data: only {len(dataset_handler.all_patches)} patches found for model {custom_model_name}")
            logger.log_error("Need at least 100 patches for training. Please check your datasets.")
            return

        # Phase 2: 创建数据加载器
        logger.log_info("Phase 2: Creating official micro_sam data loaders")
        train_loader, val_loader = dataset_handler.create_dataloaders(
            batch_size=final_batch_size,
            num_workers=0
        )

        # Phase 3: 模型训练
        logger.log_info("Phase 3: Model training with official micro_sam pipeline")
        
        # 在训练开始前显示选定的模型配置
        if not args.skip_param_analysis:
            logger.log_info("="*60)
            logger.log_info("即将开始训练的模型配置:")
            logger.log_info(f"  模型类型: {selected_model_type}")
            logger.log_info(f"  模型描述: {model_config['description']}")
            logger.log_info(f"  模型家族: LM (针对显微镜优化)")
            logger.log_info("  训练策略: 完全微调 (所有参数可训练)")
            logger.log_info("  分割解码器: 启用")
            logger.log_info("  实例分割: 支持")
            logger.log_info(f"  批次大小: {final_batch_size}")
            logger.log_info(f"  每批对象数: {final_n_objects}")
            logger.log_info(f"  学习率: {final_learning_rate}")
            logger.log_info(f"  预计训练时间: {model_config['training_time_estimate']}")
            logger.log_info(f"  预计内存需求: {model_config['memory_gb_estimate']} GB")
            
            # # 显示训练配置的简要参数统计
            # if 'config_summaries' in locals() and config_summaries:
            #     training_config = config_summaries[0]  # 完全微调配置
            #     logger.log_info(f"  参数统计: 总参数 {training_config['trainable_params']:,} | "
            #                   f"可训练 {training_config['trainable_ratio']:.1f}% | "
            #                   f"内存 {training_config['memory_mb']:.0f}MB")
            logger.log_info("="*60)
        
        # 创建增强的训练器，传入模型类型和配置
        class EnhancedMicroSAMTrainerWithModelType(EnhancedMicroSAMTrainer):
            def __init__(self, save_dir, val_patches, logger, model_type, model_config):
                super().__init__(save_dir, val_patches, logger)
                self.model_type = model_type
                self.model_config = model_config
                self.logger.log_info(f"训练器初始化 - 模型类型: {model_type}")
                self.logger.log_info(f"模型配置: {model_config}")
            
            def train_model_with_detailed_logging(self, train_loader, val_loader, model_name="microsam_custom", epochs=100):
                """使用指定的LM模型类型进行训练"""
                self.logger.log_info("="*60)
                self.logger.log_info(f"OFFICIAL TRAINING STARTED: {model_name}")
                self.logger.log_info(f"Model type: {self.model_type}")
                self.logger.log_info(f"Model family: LM (Light Microscopy optimized)")
                self.logger.log_info(f"Total epochs: {epochs}")
                self.logger.log_info(f"Training samples: {len(train_loader.dataset)}")
                self.logger.log_info(f"Validation samples: {len(val_loader.dataset)}")
                self.logger.log_info(f"Batch size: {train_loader.batch_size}")
                self.logger.log_info(f"Objects per batch: {final_n_objects}")
                self.logger.log_info(f"Learning rate: {final_learning_rate}")
                self.logger.log_info("="*60)
                
                # 初始化wandb
                wandb_initialized = self.init_wandb(
                    model_name=model_name,
                    epochs=epochs,
                    learning_rate=final_learning_rate,
                    batch_size=train_loader.batch_size,
                    n_objects_per_batch=final_n_objects,
                    train_size=len(train_loader.dataset),
                    val_size=len(val_loader.dataset),
                    model_type=self.model_type
                )
                
                # 启动监控
                if wandb_initialized:
                    self.start_monitoring(model_name)
                
                training_start_time = time.time()
                
                try:
                    # 记录训练开始时的信息
                    if self.wandb_run is not None:
                        self.wandb_run.summary.update({
                            'training_started': True,
                            'model_name': model_name,
                            'model_type': self.model_type,
                            'model_family': 'LM',
                            'start_time': datetime.fromtimestamp(training_start_time).isoformat()
                        })
                        
                        self.wandb_run.log({
                            'timing/start_timestamp': training_start_time,
                            'config/total_epochs': epochs,
                            'config/model_type': self.model_type,
                            'config/model_family': 'LM',
                            'timing/estimated_duration_hours': epochs * 0.1,
                            'status/initialization': 1
                        })
                    
                    # 使用官方的micro_sam训练，传入动态模型类型
                    self.logger.log_info(f"Using official micro_sam training pipeline with {self.model_type}...")
                    
                    train_sam(
                        name=model_name,
                        model_type=self.model_type,  # 使用动态模型类型
                        train_loader=train_loader,
                        val_loader=val_loader,
                        n_epochs=epochs,
                        n_objects_per_batch=final_n_objects,
                        with_segmentation_decoder=True,
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        lr=final_learning_rate,
                        early_stopping=15,
                        save_root=str(self.checkpoint_dir),
                        save_every_kth_epoch=5,
                        scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
                        scheduler_kwargs={'patience': 5, 'factor': 0.5, 'min_lr': 1e-7},
                        overwrite_training=True
                    )
                    
                    # 训练完成后导出模型
                    self.export_trained_model(model_name)
                    
                except Exception as e:
                    self.logger.log_error(f"Training failed: {e}")
                    if self.wandb_run is not None:
                        self.wandb_run.summary.update({
                            'training_failed': True,
                            'error_message': str(e)[:500],
                            'failure_time': datetime.now().isoformat()
                        })
                        self.wandb_run.log({
                            'status/error_occurred': 1,
                            'status/training_failed': 1
                        })
                    import traceback
                    self.logger.log_error(traceback.format_exc())
                    raise
                finally:
                    # 停止监控
                    self.stop_monitoring()
                    
                    # 记录训练完成信息
                    training_end_time = time.time()
                    total_duration = training_end_time - training_start_time
                    
                    if self.wandb_run is not None:
                        self.wandb_run.summary.update({
                            'training_completed': True,
                            'total_duration_formatted': f"{int(total_duration//3600):02d}:{int((total_duration%3600)//60):02d}:{int(total_duration%60):02d}",
                            'end_time': datetime.fromtimestamp(training_end_time).isoformat(),
                            'final_status': 'completed'
                        })
                        
                        self.wandb_run.log({
                            'timing/total_duration_seconds': total_duration,
                            'timing/total_duration_hours': total_duration / 3600,
                            'timing/end_timestamp': training_end_time,
                            'status/completion': 1,
                            'performance/avg_seconds_per_epoch': total_duration / epochs
                        })
                        
                        self.wandb_run.finish()
                        self.logger.log_info("Wandb run finished")
                
                total_duration_formatted = f"{int(total_duration//3600):02d}:{int((total_duration%3600)//60):02d}:{int(total_duration%60):02d}"
                
                self.logger.log_info("="*60)
                self.logger.log_info(f"TRAINING COMPLETED: {model_name}")
                self.logger.log_info(f"Model type: {self.model_type}")
                self.logger.log_info(f"Total training time: {total_duration_formatted}")
                self.logger.log_info(f"Average time per epoch: {total_duration/epochs:.2f} seconds")
                self.logger.log_info("="*60)
            
            def export_trained_model(self, model_name):
                """导出训练好的LM模型，使用动态模型类型"""
                try:
                    self.logger.log_info("Exporting trained LM model...")
                    
                    export_path = self.save_dir / f"{model_name}_exported.pth"
                    checkpoint_path = self.checkpoint_dir / model_name / "best.pt"
                    
                    if not checkpoint_path.exists():
                        self.logger.log_warning(f"Checkpoint not found at {checkpoint_path}, trying latest.pt")
                        checkpoint_path = self.checkpoint_dir / model_name / "latest.pt"
                        
                    if checkpoint_path.exists():
                        # 使用动态模型类型导出
                        export_custom_sam_model(
                            checkpoint_path=str(checkpoint_path),
                            model_type=self.model_type,  # 使用动态模型类型
                            save_path=str(export_path),
                            with_segmentation_decoder=True
                        )
                        
                        self.logger.log_info(f"LM Model exported successfully to: {export_path}")
                        self.logger.log_info(f"Model type: {self.model_type}")
                        self.logger.log_info(f"Optimized for: Light Microscopy")
                        
                        # 记录到wandb
                        if self.wandb_run is not None:
                            model_size_mb = export_path.stat().st_size / 1024**2
                            
                            self.wandb_run.summary.update({
                                'model_exported': True,
                                'export_path': str(export_path),
                                'model_size_mb': model_size_mb,
                                'model_type': self.model_type,
                                'model_family': 'LM'
                            })
                            
                            self.wandb_run.log({
                                'model/size_mb': model_size_mb,
                                'model/export_success': 1
                            })
                        
                    else:
                        self.logger.log_error(f"No checkpoint found for model {model_name}")
                        if self.wandb_run is not None:
                            self.wandb_run.summary.update({
                                'model_export_failed': True,
                                'export_error': 'No checkpoint found'
                            })
                            self.wandb_run.log({'model/export_failed': 1})
                        
                except Exception as e:
                    self.logger.log_error(f"Failed to export model: {e}")
                    if self.wandb_run is not None:
                        self.wandb_run.summary.update({
                            'model_export_failed': True,
                            'export_error': str(e)[:200]
                        })
                        self.wandb_run.log({'model/export_failed': 1})
                    import traceback
                    self.logger.log_error(traceback.format_exc())
        
        trainer = EnhancedMicroSAMTrainerWithModelType(
            save_dir=save_dir,
            val_patches=dataset_handler.val_patches,
            logger=logger,
            model_type=selected_model_type,
            model_config=model_config
        )
        
        trainer.train_model_with_detailed_logging(
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=custom_model_name,
            epochs=args.epochs
        )
        
        logger.log_info("Training completed successfully!")
        logger.log_info(f"All results saved to: {save_dir}")
        logger.log_info(f"Logs available in: {save_dir / 'logs'}")
        logger.log_info(f"Patch cache saved for model {custom_model_name}")
        
        # 记录缓存位置信息
        logger.log_info("="*60)
        logger.log_info("CACHE MANAGEMENT INFO:")
        logger.log_info(f"Model-specific cache directory: {dataset_handler.patch_save_dir}")
        logger.log_info(f"Cache info file: {dataset_handler.cache_info_file}")
        logger.log_info("Cache management commands:")
        logger.log_info(f"  Clear this model's cache: --clear-cache --model-name {custom_model_name}")
        logger.log_info("  Clear all caches: --clear-all-caches")
        logger.log_info("="*60)
        
        # 提供模型使用指南
        logger.log_info("="*60)
        logger.log_info("MODEL USAGE GUIDE:")
        logger.log_info(f"Model name: {custom_model_name}")
        logger.log_info(f"Model type: {selected_model_type}")
        logger.log_info(f"Model family: LM (Light Microscopy optimized)")
        logger.log_info(f"Checkpoint saved in: {save_dir / 'checkpoints'}")
        logger.log_info("To use the trained model:")
        logger.log_info("1. Load with get_predictor_and_segmenter()")
        logger.log_info("2. Use checkpoint path from the checkpoints directory")
        logger.log_info(f"3. Model type: '{selected_model_type}'")
        logger.log_info("4. Use with_segmentation_decoder=True for instance segmentation")
        logger.log_info("5. Optimized for fluorescence, brightfield, phase contrast microscopy")
        logger.log_info("="*60)
        
        # 训练完成后的最终报告
        if not args.skip_param_analysis:
            logger.log_info("="*60)
            logger.log_info("TRAINING SESSION SUMMARY:")
            logger.log_info(f"Session ID: {custom_model_name}_{timestamp}")
            logger.log_info(f"Model type: {selected_model_type}")
            logger.log_info(f"Model family: LM (Light Microscopy optimized)")
            logger.log_info(f"Model description: {model_config['description']}")
            logger.log_info(f"Accuracy tier: {model_config['accuracy_tier']}")
            logger.log_info(f"Training epochs: {args.epochs}")
            logger.log_info(f"Final batch size: {final_batch_size}")
            logger.log_info(f"Objects per batch: {final_n_objects}")
            logger.log_info(f"Learning rate: {final_learning_rate}")
            logger.log_info(f"Total training patches: {len(dataset_handler.train_patches)}")
            logger.log_info(f"Total validation patches: {len(dataset_handler.val_patches)}")
            # if 'analysis_report_path' in locals():
            #     logger.log_info("Reports generated:")
            #     logger.log_info(f"  • Parameter analysis: {analysis_report_path.name}")
            #     logger.log_info(f"  • Configuration comparison: {comparison_report_path.name}")
            #     logger.log_info(f"  • Training logs: logs/")
            logger.log_info("="*60)
        
        # LM模型性能和资源使用建议
        logger.log_info("="*60)
        logger.log_info("LM MODEL PERFORMANCE & RESOURCE RECOMMENDATIONS:")
        logger.log_info("LM模型性能排名 (针对显微镜图像精度):")
        logger.log_info("  1. vit_l_lm (出色) > 2. vit_b_lm (优秀) > 3. vit_t_lm (良好)")
        logger.log_info(f"\n您选择的模型 ({selected_model_type}) 特点:")
        
        if selected_model_type == "vit_t_lm":
            logger.log_info("  • 最适合: 快速原型开发，资源受限环境")
            logger.log_info("  • 训练时间: ~30-60分钟")
            logger.log_info("  • 内存使用: 低 (8-12GB)")
            logger.log_info("  • 推理速度: 最快")
            logger.log_info("  • 准确性: 良好，满足大多数应用需求")
        elif selected_model_type == "vit_b_lm":
            logger.log_info("  • 最适合: 通用使用，速度和精度的最佳平衡")
            logger.log_info("  • 训练时间: ~1-2小时")
            logger.log_info("  • 内存使用: 中等 (16-24GB)")
            logger.log_info("  • 推理速度: 快")
            logger.log_info("  • 准确性: 优秀，推荐用于生产环境")
        elif selected_model_type == "vit_l_lm":
            logger.log_info("  • 最适合: 高精度要求，生产应用")
            logger.log_info("  • 训练时间: ~2-4小时")
            logger.log_info("  • 内存使用: 高 (32-48GB)")
            logger.log_info("  • 推理速度: 中等")
            logger.log_info("  • 准确性: 出色，适合科研和高精度应用")
        
        logger.log_info(f"\nLM模型优势 (vs 标准SAM模型):")
        logger.log_info("  ✓ 专为显微镜图像优化")
        logger.log_info("  ✓ 在荧光、明场、相差显微镜图像上表现更佳")
        logger.log_info("  ✓ 针对细胞、组织、生物样本预训练")
        logger.log_info("  ✓ 更好的小目标检测能力")
        logger.log_info("  ✓ 对噪声和低对比度图像更鲁棒")
        
        logger.log_info(f"\n推荐使用场景:")
        logger.log_info("  • 细胞计数和分割")
        logger.log_info("  • 组织病理学分析")
        logger.log_info("  • 荧光显微镜图像分析")
        logger.log_info("  • 活细胞成像分析")
        logger.log_info("  • 神经元追踪和分析")
        
        # 与其他LM模型的对比
        logger.log_info(f"\n模型选择建议:")
        if selected_model_type == "vit_t_lm":
            logger.log_info("  • 当前选择适合: 快速验证、概念验证、资源受限")
            logger.log_info("  • 考虑升级到 vit_b_lm: 如需更高精度")
            logger.log_info("  • 考虑升级到 vit_l_lm: 如需最高精度且有充足资源")
        elif selected_model_type == "vit_b_lm":
            logger.log_info("  • 当前选择: 最佳平衡点，推荐用于大多数应用")
            logger.log_info("  • 降级到 vit_t_lm: 如需更快推理速度")
            logger.log_info("  • 升级到 vit_l_lm: 如对精度有极高要求")
        elif selected_model_type == "vit_l_lm":
            logger.log_info("  • 当前选择: 最高精度，适合科研和生产环境")
            logger.log_info("  • 降级到 vit_b_lm: 如需平衡精度和速度")
            logger.log_info("  • 降级到 vit_t_lm: 如需最快推理速度")
        
        logger.log_info("="*60)
        
    except Exception as e:
        logger.log_error(f"Training failed: {e}")
        import traceback
        logger.log_error(traceback.format_exc())
        
        # # 在错误情况下也保存已完成的分析
        # if not args.skip_param_analysis and 'silent_analyzer' in locals():
        #     try:
        #         error_report_path = save_dir / f"error_parameter_analysis_{selected_model_type}.json"
        #         silent_analyzer.save_analysis_report(str(error_report_path))
        #         logger.log_info(f"参数分析报告已保存到: {error_report_path}")
        #     except:
        #         pass
        
        raise

# 添加一个独立的缓存管理工具函数
def cache_management():
    """独立的缓存管理工具"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MicroSAM Cache Management Tool")
    parser.add_argument("--list", action="store_true", help="List all cached models")
    parser.add_argument("--clear-model", type=str, help="Clear cache for specific model")
    parser.add_argument("--clear-all", action="store_true", help="Clear all model caches")
    parser.add_argument("--info", type=str, help="Show detailed info for specific model cache")
    parser.add_argument("--size", action="store_true", help="Show cache sizes")
    args = parser.parse_args()
    
    base_cache_dir = Path("/LD-FS/home/yunshuchen/micro_sam")
    
    if args.list or args.size:
        print("Available cached models:")
        print("-" * 50)
        
        cache_dirs = list(base_cache_dir.glob("patches_*"))
        total_size = 0
        
        for cache_dir in sorted(cache_dirs):
            if cache_dir.is_dir():
                # 提取模型名
                dir_name = cache_dir.name
                if "_" in dir_name:
                    parts = dir_name.split("_")
                    if len(parts) >= 3:
                        model_name = "_".join(parts[1:-1])  # 去掉"patches"前缀和大小后缀
                        patch_size = parts[-1]
                    else:
                        model_name = parts[1] if len(parts) > 1 else "unknown"
                        patch_size = "unknown"
                else:
                    model_name = "unknown"
                    patch_size = "unknown"
                
                # 计算缓存大小
                cache_size = 0
                patch_count = 0
                try:
                    for file_path in cache_dir.rglob("*"):
                        if file_path.is_file():
                            cache_size += file_path.stat().st_size
                            if file_path.suffix in ['.png', '.jpg', '.tif']:
                                patch_count += 1
                    
                    cache_size_mb = cache_size / (1024 * 1024)
                    total_size += cache_size_mb
                    
                    # 读取缓存信息
                    info_file = cache_dir / f"patch_cache_info_{model_name}.json"
                    info_status = "✓" if info_file.exists() else "✗"
                    
                    if args.size:
                        print(f"  {model_name:25} | {patch_size:4}px | {cache_size_mb:8.1f}MB | {patch_count//2:6} patches | {info_status}")
                    else:
                        print(f"  {model_name:25} | {patch_size:4}px | {patch_count//2:6} patches | {info_status}")
                        
                except Exception as e:
                    print(f"  {model_name:25} | {patch_size:4}px | Error: {e}")
        
        if args.size:
            print("-" * 50)
            print(f"Total cache size: {total_size:.1f}MB")
        
    elif args.info:
        model_name = args.info
        cache_dirs = list(base_cache_dir.glob(f"patches_{model_name}_*"))
        
        if not cache_dirs:
            print(f"No cache found for model: {model_name}")
            return
        
        for cache_dir in cache_dirs:
            info_file = cache_dir / f"patch_cache_info_{model_name}.json"
            
            if info_file.exists():
                try:
                    with open(info_file, 'r') as f:
                        cache_info = json.load(f)
                    
                    print(f"Cache info for model: {model_name}")
                    print("-" * 50)
                    print(f"Cache directory: {cache_dir}")
                    print(f"Total patches: {cache_info.get('total_patches', 'Unknown')}")
                    print(f"Generated time: {cache_info.get('generated_time', 'Unknown')}")
                    print(f"Cache version: {cache_info.get('cache_version', 'Unknown')}")
                    print(f"Patch size: {cache_info.get('config', {}).get('patch_size', 'Unknown')}")
                    print(f"Overlap: {cache_info.get('config', {}).get('overlap', 'Unknown')}")
                    
                    dataset_info = cache_info.get('dataset_info', {})
                    if dataset_info:
                        print("\nDataset breakdown:")
                        for dataset_name, info in dataset_info.items():
                            print(f"  {dataset_name}: {info.get('total_patches', 0)} patches from {info.get('valid_images', 0)} images")
                    
                except Exception as e:
                    print(f"Error reading cache info: {e}")
            else:
                print(f"No cache info file found for model: {model_name}")
    
    elif args.clear_model:
        model_name = args.clear_model
        cache_dirs = list(base_cache_dir.glob(f"patches_{model_name}_*"))
        
        if not cache_dirs:
            print(f"No cache found for model: {model_name}")
            return
        
        for cache_dir in cache_dirs:
            try:
                import shutil
                shutil.rmtree(cache_dir)
                print(f"Cleared cache for model {model_name}: {cache_dir}")
            except Exception as e:
                print(f"Error clearing cache for {model_name}: {e}")
    
    elif args.clear_all:
        cache_dirs = list(base_cache_dir.glob("patches_*"))
        
        if not cache_dirs:
            print("No caches found to clear")
            return
        
        print(f"Found {len(cache_dirs)} cache directories to clear:")
        for cache_dir in cache_dirs:
            print(f"  {cache_dir}")
        
        confirm = input("Are you sure you want to clear ALL caches? (yes/no): ")
        if confirm.lower() == 'yes':
            for cache_dir in cache_dirs:
                try:
                    import shutil
                    shutil.rmtree(cache_dir)
                    print(f"Cleared: {cache_dir}")
                except Exception as e:
                    print(f"Error clearing {cache_dir}: {e}")
            print("All caches cleared")
        else:
            print("Operation cancelled")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    import sys
    
    # 检查是否是缓存管理模式
    if len(sys.argv) > 1 and sys.argv[1] == "cache":
        # 移除 "cache" 参数，让缓存管理工具处理剩余参数
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        cache_management()
    else:
        main()
