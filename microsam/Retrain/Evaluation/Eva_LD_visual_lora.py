"""
LoRA实验批量评测系统
- 支持多个测试集和多个checkpoint的交叉评测
- 保留原有的所有功能和可视化逻辑
- 更新数据读取方式以适配LoRA实验结构
"""

import os
import time
import json
import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io, measure
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from multiprocessing import Manager
import gc
from glob import glob
from scipy.spatial import distance
import random
from matplotlib.colors import ListedColormap

# Import micro_sam modules
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation
from micro_sam.util import get_sam_model
from micro_sam.instance_segmentation import get_predictor_and_decoder, InstanceSegmentationWithDecoder
import math

# 设置matplotlib使用非交互式后端
import matplotlib
matplotlib.use('Agg')
plt.ioff()

class ResultVisualizer:
    """可视化分割结果"""
    
    def __init__(self, output_dir: Path):
        """Initialize visualizer with output directory"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_comparison(
        self, 
        image: np.ndarray, 
        gt_mask: np.ndarray, 
        prediction: np.ndarray, 
        metrics: Dict[str, Any], 
        img_id: str
    ):
        """Create visualization with the following layout:
        - Top-left: Original image
        - Top-right: Ground truth mask overlay
        - Bottom-left: Prediction overlay
        - Bottom-right: Prediction heatmap
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            
            # Original image (Top-left)
            axes[0, 0].imshow(image, cmap='gray')
            axes[0, 0].set_title("Original Image")
            axes[0, 0].axis("off")
            
            # Ground truth mask overlay (Top-right)
            axes[0, 1].imshow(image, cmap='gray')
            gt_unique_labels = len(np.unique(gt_mask)) - 1
            if gt_unique_labels > 0:
                cmap_gt = create_random_colormap(gt_unique_labels)
                axes[0, 1].imshow(gt_mask, cmap=cmap_gt, alpha=0.5, interpolation="nearest")
            axes[0, 1].set_title(f"Ground Truth ({metrics.get('gt_instances', 0)} instances)")
            axes[0, 1].axis("off")
            
            # Prediction overlay (Bottom-left)
            axes[1, 0].imshow(image, cmap='gray')
            pred_unique_labels = len(np.unique(prediction)) - 1
            if pred_unique_labels > 0:
                cmap_pred = create_random_colormap(pred_unique_labels)
                axes[1, 0].imshow(prediction, cmap=cmap_pred, alpha=0.5, interpolation="nearest")
            axes[1, 0].set_title(f"Prediction ({metrics.get('pred_instances', 0)} instances)")
            axes[1, 0].axis("off")
            
            # Prediction heatmap (Bottom-right)
            axes[1, 1].imshow(image, cmap='gray')
            mask_smooth = np.float32(prediction > 0)
            if np.any(mask_smooth):
                axes[1, 1].imshow(mask_smooth, cmap='hot', alpha=0.7)
            axes[1, 1].set_title("Prediction Heatmap")
            axes[1, 1].axis("off")
            
            # Add metrics text - safely handle None values
            iou_val = metrics.get('iou_score', 0)
            dice_val = metrics.get('dice_score', 0)
            hd95_val = metrics.get('hd95', 0)
            ap50_val = metrics.get('ap50', 0)
            ap75_val = metrics.get('ap75', 0)
            
            # Format string with safely extracted values
            def safe_format(val):
                if val is None:
                    return "N/A"
                if isinstance(val, (int, float)):
                    if math.isnan(val) or math.isinf(val):
                        return "N/A"
                    return f"{val:.3f}"
                return "N/A"
            
            metrics_txt = (
                f"IoU: {safe_format(iou_val)}, Dice: {safe_format(dice_val)}\n"
                f"HD95: {safe_format(hd95_val)}, AP50: {safe_format(ap50_val)}, AP75: {safe_format(ap75_val)}"
            )
            
            fig.suptitle(metrics_txt, fontsize=14)
            
            plt.tight_layout()
            
            # 使用绝对路径保存，确保路径正确
            output_path = self.output_dir / f"comparison_{img_id}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")
            
            plt.close('all')
            
            # 强制垃圾回收
            gc.collect()
            
        except Exception as e:
            print(f"Visualization error for {img_id}: {e}")
            import traceback
            traceback.print_exc()
            plt.close('all')

def create_random_colormap(n_labels):
    """为每个标签创建随机颜色图，用于可视化分割结果"""
    if n_labels <= 0:
        return ListedColormap([(0, 0, 0)])
    
    np.random.seed(42)
    colors = [(np.random.random(), np.random.random(), np.random.random()) for _ in range(n_labels)]
    colors.insert(0, (0, 0, 0))
    return ListedColormap(colors)

def draw_grid_on_image(image, grid_size=50):
    """在图像上绘制网格线"""
    img_with_grid = image.copy()
    h, w = image.shape[:2]
    
    # 绘制垂直线
    for x in range(0, w, grid_size):
        if len(img_with_grid.shape) == 3:
            img_with_grid[:, x:x+1, :] = [255, 255, 255]
        else:
            img_with_grid[:, x:x+1] = 255
    
    # 绘制水平线
    for y in range(0, h, grid_size):
        if len(img_with_grid.shape) == 3:
            img_with_grid[y:y+1, :, :] = [255, 255, 255]
        else:
            img_with_grid[y:y+1, :] = 255
    
    return img_with_grid

def save_overlay_image(image, mask, output_path, alpha=0.5, draw_grid=False, grid_size=50):
    """保存单张叠加图像"""
    try:
        # 确保图像为3通道
        if len(image.shape) == 2:
            img_rgb = np.stack([image] * 3, axis=2)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            img_rgb = np.stack([image[:,:,0]] * 3, axis=2)
        else:
            img_rgb = image.copy()
        
        # 规范化图像值到0-1
        if img_rgb.max() > 1:
            img_rgb = img_rgb / 255.0
        
        # 如果需要绘制网格
        if draw_grid:
            img_rgb = draw_grid_on_image(img_rgb, grid_size)
            if img_rgb.max() > 1:
                img_rgb = img_rgb / 255.0
        
        # 创建图形
        plt.figure(figsize=(8, 8))
        plt.imshow(img_rgb)
        
        # 计算唯一标签数（不包括背景0）
        unique_labels = np.unique(mask)
        unique_labels = unique_labels[unique_labels > 0]
        n_labels = len(unique_labels)
        
        if n_labels > 0:
            # 创建随机颜色映射
            np.random.seed(42)
            colors = [(np.random.random(), np.random.random(), np.random.random()) 
                      for _ in range(n_labels)]
            colors.insert(0, (0, 0, 0))
            cmap = ListedColormap(colors)
            
            # 叠加掩码
            plt.imshow(mask, cmap=cmap, alpha=alpha)
        
        # 设置标题和关闭轴
        grid_info = " (with grid)" if draw_grid else ""
        plt.title(f"Overlay ({n_labels} instances){grid_info}")
        plt.axis('off')
        plt.tight_layout()
        
        # 保存和关闭
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close('all')
        
        # 强制垃圾回收
        gc.collect()
        
        return True
    except Exception as e:
        print(f"Error saving overlay to {output_path}: {e}")
        plt.close('all')
        return False

class TimeoutHandler:
    """超时处理器"""
    def __init__(self, timeout_seconds=300):
        self.timeout_seconds = timeout_seconds
        self.timer = None
    
    def timeout_handler(self):
        print(f"Process timeout after {self.timeout_seconds} seconds!")
        os._exit(1)
    
    def start_timer(self):
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(self.timeout_seconds, self.timeout_handler)
        self.timer.start()
    
    def stop_timer(self):
        if self.timer:
            self.timer.cancel()

class ComprehensiveMetrics:
    """完整的评测指标计算类，包含AP50、AP75、IoU、Dice、HD95"""
    
    @staticmethod
    def calculate_hausdorff_distance_95(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
        """计算HD95指标"""
        try:
            gt_binary = (gt_mask > 0).astype(np.uint8)
            pred_binary = (pred_mask > 0).astype(np.uint8)
            
            gt_contours = measure.find_contours(gt_binary, 0.5)
            pred_contours = measure.find_contours(pred_binary, 0.5)
            
            if not gt_contours or not pred_contours:
                return float('inf')
            
            gt_contour = max(gt_contours, key=len) if len(gt_contours) > 1 else gt_contours[0]
            pred_contour = max(pred_contours, key=len) if len(pred_contours) > 1 else pred_contours[0]
            
            if len(gt_contour) < 2 or len(pred_contour) < 2:
                return float('inf')
            
            distances_gt_to_pred = distance.cdist(gt_contour, pred_contour, 'euclidean')
            distances_pred_to_gt = distance.cdist(pred_contour, gt_contour, 'euclidean')
            
            min_distances_gt_to_pred = np.min(distances_gt_to_pred, axis=1)
            min_distances_pred_to_gt = np.min(distances_pred_to_gt, axis=1)
            
            hd_gt_to_pred = np.percentile(min_distances_gt_to_pred, 95)
            hd_pred_to_gt = np.percentile(min_distances_pred_to_gt, 95)
            
            hd95 = max(hd_gt_to_pred, hd_pred_to_gt)
            
            return float(hd95)
            
        except Exception as e:
            print(f"HD95 calculation error: {e}")
            return float('inf')
    
    @staticmethod
    def calculate_ap_at_threshold(gt_mask: np.ndarray, pred_mask: np.ndarray, iou_threshold: float) -> float:
        """计算指定IoU阈值下的平均精度"""
        try:
            gt_labels = np.unique(gt_mask)[1:]
            pred_labels = np.unique(pred_mask)[1:]
            
            if len(gt_labels) == 0:
                return 1.0 if len(pred_labels) == 0 else 0.0
            if len(pred_labels) == 0:
                return 0.0
            
            iou_matrix = np.zeros((len(pred_labels), len(gt_labels)))
            
            for i, pred_id in enumerate(pred_labels):
                pred_region = (pred_mask == pred_id)
                for j, gt_id in enumerate(gt_labels):
                    gt_region = (gt_mask == gt_id)
                    
                    intersection = np.sum(pred_region & gt_region)
                    union = np.sum(pred_region | gt_region)
                    
                    if union > 0:
                        iou_matrix[i, j] = intersection / union
            
            pred_areas = [np.sum(pred_mask == pred_id) for pred_id in pred_labels]
            sorted_indices = np.argsort(pred_areas)[::-1]
            
            gt_matched = np.zeros(len(gt_labels), dtype=bool)
            precision_points = []
            
            for rank, pred_idx in enumerate(sorted_indices):
                best_gt_idx = np.argmax(iou_matrix[pred_idx])
                best_iou = iou_matrix[pred_idx, best_gt_idx]
                
                if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                    gt_matched[best_gt_idx] = True
                
                tp = np.sum(gt_matched)
                precision = tp / (rank + 1)
                precision_points.append(precision)
            
            ap = np.mean(precision_points) if precision_points else 0.0
            return float(ap)
            
        except Exception as e:
            print(f"AP calculation error: {e}")
            return 0.0
    
    @classmethod
    def compute_all_metrics(cls, gt_mask: np.ndarray, pred_mask: np.ndarray) -> Dict[str, float]:
        """计算所有评测指标"""
        try:
            # 确保是标签图
            if np.max(gt_mask) <= 1:
                gt_mask = measure.label(gt_mask > 0)
            if np.max(pred_mask) <= 1:
                pred_mask = measure.label(pred_mask > 0)
            
            # 二值化掩码
            gt_binary = (gt_mask > 0).astype(np.float32)
            pred_binary = (pred_mask > 0).astype(np.float32)
            
            # 基本像素级指标
            intersection = np.sum(gt_binary * pred_binary)
            union = np.sum(gt_binary) + np.sum(pred_binary) - intersection
            
            # IoU和Dice
            iou_score = intersection / (union + 1e-6)
            dice_score = 2 * intersection / (np.sum(gt_binary) + np.sum(pred_binary) + 1e-6)
            
            # HD95
            hd95 = cls.calculate_hausdorff_distance_95(gt_mask, pred_mask)
            
            # 实例数量
            gt_instances = len(np.unique(gt_mask)) - 1
            pred_instances = len(np.unique(pred_mask)) - 1
            
            # 计算AP50和AP75
            ap50 = cls.calculate_ap_at_threshold(gt_mask, pred_mask, 0.5)
            ap75 = cls.calculate_ap_at_threshold(gt_mask, pred_mask, 0.75)
            
            return {
                'ap50': float(ap50),
                'ap75': float(ap75),
                'iou_score': float(iou_score),
                'dice_score': float(dice_score),
                'hd95': float(hd95),
                'gt_instances': gt_instances,
                'pred_instances': pred_instances
            }
            
        except Exception as e:
            print(f"Comprehensive metrics calculation error: {e}")
            return {
                'ap50': 0.0,
                'ap75': 0.0,
                'iou_score': 0.0,
                'dice_score': 0.0,
                'hd95': float('inf'),
                'gt_instances': 0,
                'pred_instances': 0
            }

def merge_lora_weights(checkpoint_data):
    """手动合并LoRA权重到原始权重中"""
    try:
        print("  Manually merging LoRA weights...")
        
        # 找到所有需要合并的LoRA层
        lora_layers = {}
        original_layers = {}
        
        for key, value in checkpoint_data.items():
            if '.original_layer.' in key:
                # 提取层名称，例如 "image_encoder.blocks.0.attn.qkv"
                layer_name = key.replace('.original_layer.weight', '').replace('.original_layer.bias', '')
                param_type = 'weight' if key.endswith('.weight') else 'bias'
                
                if layer_name not in original_layers:
                    original_layers[layer_name] = {}
                original_layers[layer_name][param_type] = value
                
            elif '.lora.lora_A.' in key or '.lora.lora_B.' in key:
                # 提取层名称和LoRA参数类型
                if '.lora.lora_A.' in key:
                    layer_name = key.split('.lora.lora_A.')[0]
                    lora_type = 'A'
                    param_type = key.split('.lora.lora_A.')[1]  # weight or bias
                else:
                    layer_name = key.split('.lora.lora_B.')[0]
                    lora_type = 'B'
                    param_type = key.split('.lora.lora_B.')[1]  # weight or bias
                
                if layer_name not in lora_layers:
                    lora_layers[layer_name] = {}
                if lora_type not in lora_layers[layer_name]:
                    lora_layers[layer_name][lora_type] = {}
                lora_layers[layer_name][lora_type][param_type] = value
        
        print(f"    Found {len(original_layers)} original layers")
        print(f"    Found {len(lora_layers)} LoRA layers")
        
        # 合并权重
        merged_state_dict = {}
        
        # 首先复制所有非LoRA权重
        for key, value in checkpoint_data.items():
            if '.original_layer.' not in key and '.lora.' not in key:
                merged_state_dict[key] = value
        
        # 合并LoRA权重
        merged_count = 0
        for layer_name in original_layers:
            if layer_name in lora_layers:
                # 合并权重：W = W0 + B @ A
                if 'weight' in original_layers[layer_name] and 'A' in lora_layers[layer_name] and 'B' in lora_layers[layer_name]:
                    W0 = original_layers[layer_name]['weight']
                    A = lora_layers[layer_name]['A']['weight']
                    B = lora_layers[layer_name]['B']['weight']
                    
                    # 计算 ΔW = B @ A
                    delta_W = torch.matmul(B, A)
                    
                    # 合并权重
                    merged_weight = W0 + delta_W
                    merged_state_dict[f"{layer_name}.weight"] = merged_weight
                    merged_count += 1
                    
                # 处理bias
                if 'bias' in original_layers[layer_name]:
                    # 对于bias，通常只使用原始bias
                    merged_state_dict[f"{layer_name}.bias"] = original_layers[layer_name]['bias']
                    
                    # 如果LoRA_B有bias，则添加它
                    if 'B' in lora_layers[layer_name] and 'bias' in lora_layers[layer_name]['B']:
                        original_bias = original_layers[layer_name]['bias']
                        lora_bias = lora_layers[layer_name]['B']['bias']
                        merged_state_dict[f"{layer_name}.bias"] = original_bias + lora_bias
            else:
                # 没有LoRA的层，直接使用原始权重
                for param_type, param_value in original_layers[layer_name].items():
                    merged_state_dict[f"{layer_name}.{param_type}"] = param_value
        
        print(f"    Successfully merged {merged_count} LoRA layers")
        print(f"    Final merged state dict has {len(merged_state_dict)} parameters")
        
        return merged_state_dict
        
    except Exception as e:
        print(f"    Error merging LoRA weights: {e}")
        return None

def check_decoder_requirements(model_type: str, merged_state_dict: dict, original_checkpoint: dict):
    """检查是否需要decoder以及如何获取decoder"""
    try:
        # 检查merged权重中是否有decoder
        has_decoder_in_merged = any('decoder' in key for key in merged_state_dict.keys())
        
        # 检查原始checkpoint的分离组件中是否有decoder
        has_decoder_in_components = False
        decoder_state = None
        
        if 'mask_decoder_state_dict' in original_checkpoint:
            decoder_state = original_checkpoint['mask_decoder_state_dict']
            has_decoder_in_components = True
            print(f"    Found decoder in mask_decoder_state_dict")
        
        # 如果模型类型包含decoder，我们需要确保有decoder
        needs_decoder = model_type.endswith('_lm')  # lm模型通常需要decoder
        
        print(f"    Model type: {model_type}")
        print(f"    Needs decoder: {needs_decoder}")
        print(f"    Has decoder in merged: {has_decoder_in_merged}")
        print(f"    Has decoder in components: {has_decoder_in_components}")
        
        if needs_decoder and not has_decoder_in_merged and has_decoder_in_components:
            # 将分离的decoder添加到merged state dict
            print(f"    Adding decoder from components to merged state")
            for key, value in decoder_state.items():
                merged_state_dict[f"decoder.{key}"] = value
            return merged_state_dict, True
        elif has_decoder_in_merged:
            return merged_state_dict, True
        else:
            return merged_state_dict, False
            
    except Exception as e:
        print(f"    Error checking decoder requirements: {e}")
        return merged_state_dict, has_decoder_in_merged

def setup_model_safe(model_config: Dict, gpu_id: int = None):
    """安全的模型设置 - 支持LoRA checkpoint"""
    try:
        if gpu_id is not None and torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            device = f"cuda:{gpu_id}"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        model_type = model_config["model_type"]
        checkpoint_path = model_config.get("checkpoint_path", None)
        
        print(f"Setting up model: {model_config['name']}")
        print(f"  Model type: {model_type}")
        print(f"  Checkpoint: {checkpoint_path if checkpoint_path else 'Default'}")
        print(f"  Device: {device}")
        
        if checkpoint_path:
            # 检查LoRA实验的文件结构
            checkpoint_dir = Path(checkpoint_path).parent
            final_model_dir = checkpoint_dir.parent / "final_model"
            
            # 优先使用final_model中的merged模型
            merged_model_path = final_model_dir / "merged_sam_model.pth"
            lora_weights_path = final_model_dir / "sam_lora_weights.pth"
            
            print(f"  Checking LoRA model paths:")
            print(f"    Merged model: {merged_model_path} (exists: {merged_model_path.exists()})")
            print(f"    LoRA weights: {lora_weights_path} (exists: {lora_weights_path.exists()})")
            print(f"    Original checkpoint: {checkpoint_path} (exists: {Path(checkpoint_path).exists()})")
            
            actual_checkpoint_path = None
            use_peft = False
            
            # 选择最佳的模型文件
            if merged_model_path.exists():
                actual_checkpoint_path = str(merged_model_path)
                print(f"  Using merged SAM model: {actual_checkpoint_path}")
            elif lora_weights_path.exists():
                actual_checkpoint_path = str(lora_weights_path)
                use_peft = True
                print(f"  Using LoRA weights with PEFT: {actual_checkpoint_path}")
            elif Path(checkpoint_path).exists():
                actual_checkpoint_path = checkpoint_path
                use_peft = True
                print(f"  Using original checkpoint with PEFT: {actual_checkpoint_path}")
            else:
                print(f"  Warning: No valid checkpoint found, using default model")
                actual_checkpoint_path = None
            
            if actual_checkpoint_path:
                try:
                    # 安全加载checkpoint，处理PyTorch 2.6的weights_only问题
                    try:
                        # 首先尝试安全加载
                        checkpoint = torch.load(actual_checkpoint_path, map_location=device, weights_only=True)
                        print(f"  Loaded checkpoint safely with weights_only=True")
                    except Exception as safe_load_error:
                        print(f"  Safe load failed: {safe_load_error}")
                        print(f"  Trying to load with weights_only=False (trusted source)")
                        # 对于可信的LoRA checkpoint，使用weights_only=False
                        checkpoint = torch.load(actual_checkpoint_path, map_location=device, weights_only=False)
                        print(f"  Loaded checkpoint with weights_only=False")
                    
                    # 检查checkpoint结构并提取实际的模型权重
                    print(f"  Checkpoint keys: {list(checkpoint.keys())}")
                    
                    actual_model_state = None
                    has_decoder = False
                    
                    # 处理不同的checkpoint格式
                    if 'model_state_dict' in checkpoint:
                        # 格式1: 包含model_state_dict的训练checkpoint
                        actual_model_state = checkpoint['model_state_dict']
                        print(f"  Found model_state_dict in checkpoint")
                    elif 'image_encoder_state_dict' in checkpoint and 'prompt_encoder_state_dict' in checkpoint:
                        # 格式2: 分离的组件state_dict
                        print(f"  Found separate component state_dicts")
                        # 合并各个组件
                        actual_model_state = {}
                        for component_key in ['image_encoder_state_dict', 'prompt_encoder_state_dict', 'mask_decoder_state_dict']:
                            if component_key in checkpoint:
                                component_name = component_key.replace('_state_dict', '')
                                for key, value in checkpoint[component_key].items():
                                    actual_model_state[f"{component_name}.{key}"] = value
                    else:
                        # 格式3: 直接的模型权重
                        actual_model_state = checkpoint
                        print(f"  Using checkpoint as direct model weights")
                    
                    # 检查是否包含LoRA结构
                    has_lora_structure = False
                    if actual_model_state:
                        has_lora_structure = any('.original_layer.' in key or '.lora.' in key for key in actual_model_state.keys())
                        has_decoder = any('decoder' in key for key in actual_model_state.keys())
                        print(f"  LoRA structure found: {has_lora_structure}")
                        print(f"  Decoder found in checkpoint: {has_decoder}")
                    
                    # 如果发现LoRA结构，进行手动合并
                    if has_lora_structure:
                        merged_weights = merge_lora_weights(actual_model_state)
                        if merged_weights is not None:
                            # 检查decoder需求并处理
                            merged_weights, has_decoder = check_decoder_requirements(
                                model_type, merged_weights, checkpoint
                            )
                            actual_model_state = merged_weights
                            print(f"  Successfully merged LoRA weights")
                        else:
                            print(f"  Failed to merge LoRA weights, falling back to PEFT loading")
                            use_peft = True
                    
                    # 根据是否需要PEFT设置参数
                    if use_peft and has_lora_structure:
                        peft_kwargs = {
                            "rank": model_config.get("lora_rank", 8),
                            "attention_layers_to_update": model_config.get("attention_layers", [9, 10, 11])
                        }
                        print(f"  Using PEFT with rank={peft_kwargs['rank']}, layers={peft_kwargs['attention_layers_to_update']}")
                    else:
                        peft_kwargs = None
                        print(f"  Using merged model (no PEFT needed)")
                    
                    # 创建临时文件保存处理后的权重
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
                        temp_path = tmp_file.name
                        if actual_model_state:
                            torch.save(actual_model_state, temp_path)
                            print(f"  Saved processed weights to temporary file")
                        else:
                            # 如果处理失败，使用原始文件
                            temp_path = actual_checkpoint_path
                    
                    # 如果没有decoder但模型类型需要decoder，使用默认decoder
                    if not has_decoder and model_type.endswith('_lm'):
                        print(f"  Model type {model_type} needs decoder but none found, will use default setup")
                        # 让micro-sam自动下载和使用默认decoder
                        predictor, segmenter = get_predictor_and_segmenter(
                            model_type=model_type,
                            device=device,
                            amg=False,
                            is_tiled=False
                        )
                        
                        # 尝试加载我们的merged权重到predictor
                        try:
                            predictor.model.load_state_dict(actual_model_state, strict=False)
                            print("  Successfully loaded merged weights to default predictor")
                        except Exception as load_error:
                            print(f"  Warning: Could not load merged weights: {load_error}")
                            print("  Using default weights")
                        
                        print("  Using default segmenter with LoRA predictor")
                        
                    elif has_decoder:
                        # 如果有decoder，使用InstanceSegmentationWithDecoder
                        try:
                            from micro_sam.instance_segmentation import get_predictor_and_decoder
                            predictor, decoder = get_predictor_and_decoder(
                                model_type=model_type,
                                checkpoint_path=temp_path,
                                device=device,
                                peft_kwargs=peft_kwargs
                            )
                            segmenter = InstanceSegmentationWithDecoder(predictor, decoder)
                            print("  Using InstanceSegmentationWithDecoder with merged decoder")
                        except Exception as decoder_error:
                            print(f"  Decoder loading failed: {decoder_error}")
                            print("  Falling back to default segmenter")
                            predictor, segmenter = get_predictor_and_segmenter(
                                model_type=model_type,
                                device=device,
                                amg=False,
                                is_tiled=False
                            )
                            # 尝试加载我们的merged权重
                            try:
                                predictor.model.load_state_dict(actual_model_state, strict=False)
                                print("  Successfully loaded merged weights to fallback predictor")
                            except Exception as load_error:
                                print(f"  Warning: Could not load merged weights: {load_error}")
                    else:
                        # 如果没有decoder，使用普通的SAM
                        predictor = get_sam_model(
                            model_type=model_type,
                            checkpoint_path=temp_path,
                            device=device,
                            peft_kwargs=peft_kwargs
                        )
                        segmenter = None
                        print("  Using merged SAM model without decoder")
                    
                    # 清理临时文件
                    if temp_path != actual_checkpoint_path:
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                        
                except Exception as e:
                    print(f"  Error loading LoRA checkpoint: {e}")
                    print("  Falling back to default model setup...")
                    predictor, segmenter = get_predictor_and_segmenter(
                        model_type=model_type,
                        device=device,
                        amg=False,
                        is_tiled=False
                    )
            else:
                # 使用默认模型
                predictor, segmenter = get_predictor_and_segmenter(
                    model_type=model_type,
                    device=device,
                    amg=False,
                    is_tiled=False
                )
                print("  Using default model")
        else:
            # 使用默认模型
            predictor, segmenter = get_predictor_and_segmenter(
                model_type=model_type,
                device=device,
                amg=False,
                is_tiled=False
            )
            print("  Using default model")
        
        print(f"  Model setup completed successfully")
        return predictor, segmenter
        
    except Exception as e:
        print(f"Model setup failed for {model_config['name']}: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    """安全的模型设置 - 支持LoRA checkpoint"""
    try:
        if gpu_id is not None and torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            device = f"cuda:{gpu_id}"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        model_type = model_config["model_type"]
        checkpoint_path = model_config.get("checkpoint_path", None)
        
        print(f"Setting up model: {model_config['name']}")
        print(f"  Model type: {model_type}")
        print(f"  Checkpoint: {checkpoint_path if checkpoint_path else 'Default'}")
        print(f"  Device: {device}")
        
        if checkpoint_path:
            # 检查LoRA实验的文件结构
            checkpoint_dir = Path(checkpoint_path).parent
            final_model_dir = checkpoint_dir.parent / "final_model"
            
            # 优先使用final_model中的merged模型
            merged_model_path = final_model_dir / "merged_sam_model.pth"
            lora_weights_path = final_model_dir / "sam_lora_weights.pth"
            
            print(f"  Checking LoRA model paths:")
            print(f"    Merged model: {merged_model_path} (exists: {merged_model_path.exists()})")
            print(f"    LoRA weights: {lora_weights_path} (exists: {lora_weights_path.exists()})")
            print(f"    Original checkpoint: {checkpoint_path} (exists: {Path(checkpoint_path).exists()})")
            
            actual_checkpoint_path = None
            use_peft = False
            
            # 选择最佳的模型文件
            if merged_model_path.exists():
                actual_checkpoint_path = str(merged_model_path)
                print(f"  Using merged SAM model: {actual_checkpoint_path}")
            elif lora_weights_path.exists():
                actual_checkpoint_path = str(lora_weights_path)
                use_peft = True
                print(f"  Using LoRA weights with PEFT: {actual_checkpoint_path}")
            elif Path(checkpoint_path).exists():
                actual_checkpoint_path = checkpoint_path
                use_peft = True
                print(f"  Using original checkpoint with PEFT: {actual_checkpoint_path}")
            else:
                print(f"  Warning: No valid checkpoint found, using default model")
                actual_checkpoint_path = None
            
            if actual_checkpoint_path:
                try:
                    # 安全加载checkpoint，处理PyTorch 2.6的weights_only问题
                    try:
                        # 首先尝试安全加载
                        checkpoint = torch.load(actual_checkpoint_path, map_location=device, weights_only=True)
                        print(f"  Loaded checkpoint safely with weights_only=True")
                    except Exception as safe_load_error:
                        print(f"  Safe load failed: {safe_load_error}")
                        print(f"  Trying to load with weights_only=False (trusted source)")
                        # 对于可信的LoRA checkpoint，使用weights_only=False
                        checkpoint = torch.load(actual_checkpoint_path, map_location=device, weights_only=False)
                        print(f"  Loaded checkpoint with weights_only=False")
                    
                    # 检查checkpoint结构并提取实际的模型权重
                    print(f"  Checkpoint keys: {list(checkpoint.keys())}")
                    
                    actual_model_state = None
                    has_decoder = False
                    
                    # 处理不同的checkpoint格式
                    if 'model_state_dict' in checkpoint:
                        # 格式1: 包含model_state_dict的训练checkpoint
                        actual_model_state = checkpoint['model_state_dict']
                        print(f"  Found model_state_dict in checkpoint")
                    elif 'image_encoder_state_dict' in checkpoint and 'prompt_encoder_state_dict' in checkpoint:
                        # 格式2: 分离的组件state_dict
                        print(f"  Found separate component state_dicts")
                        # 合并各个组件
                        actual_model_state = {}
                        for component_key in ['image_encoder_state_dict', 'prompt_encoder_state_dict', 'mask_decoder_state_dict']:
                            if component_key in checkpoint:
                                component_name = component_key.replace('_state_dict', '')
                                for key, value in checkpoint[component_key].items():
                                    actual_model_state[f"{component_name}.{key}"] = value
                    else:
                        # 格式3: 直接的模型权重
                        actual_model_state = checkpoint
                        print(f"  Using checkpoint as direct model weights")
                    
                    # 检查是否包含LoRA结构
                    has_lora_structure = False
                    if actual_model_state:
                        has_lora_structure = any('.original_layer.' in key or '.lora.' in key for key in actual_model_state.keys())
                        has_decoder = any('decoder' in key for key in actual_model_state.keys())
                        print(f"  LoRA structure found: {has_lora_structure}")
                        print(f"  Decoder found in checkpoint: {has_decoder}")
                    
                    # 如果发现LoRA结构，进行手动合并
                    if has_lora_structure:
                        merged_weights = merge_lora_weights(actual_model_state)
                        if merged_weights is not None:
                            actual_model_state = merged_weights
                            print(f"  Successfully merged LoRA weights")
                        else:
                            print(f"  Failed to merge LoRA weights, falling back to PEFT loading")
                            use_peft = True
                    
                    # 根据是否需要PEFT设置参数
                    if use_peft and has_lora_structure:
                        peft_kwargs = {
                            "rank": model_config.get("lora_rank", 8),
                            "attention_layers_to_update": model_config.get("attention_layers", [9, 10, 11])
                        }
                        print(f"  Using PEFT with rank={peft_kwargs['rank']}, layers={peft_kwargs['attention_layers_to_update']}")
                    else:
                        peft_kwargs = None
                        print(f"  Using merged model (no PEFT needed)")
                    
                    # 创建临时文件保存处理后的权重
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
                        temp_path = tmp_file.name
                        if actual_model_state:
                            torch.save(actual_model_state, temp_path)
                            print(f"  Saved processed weights to temporary file")
                        else:
                            # 如果处理失败，使用原始文件
                            temp_path = actual_checkpoint_path
                    
                    if has_decoder:
                        # 如果有decoder，使用InstanceSegmentationWithDecoder
                        from micro_sam.instance_segmentation import get_predictor_and_decoder
                        predictor, decoder = get_predictor_and_decoder(
                            model_type=model_type,
                            checkpoint_path=temp_path,
                            device=device,
                            peft_kwargs=peft_kwargs
                        )
                        segmenter = InstanceSegmentationWithDecoder(predictor, decoder)
                        print("  Using InstanceSegmentationWithDecoder with LoRA decoder")
                    else:
                        # 如果没有decoder，使用普通的SAM
                        predictor = get_sam_model(
                            model_type=model_type,
                            checkpoint_path=temp_path,
                            device=device,
                            peft_kwargs=peft_kwargs
                        )
                        segmenter = None
                        print("  Using LoRA SAM model without decoder")
                    
                    # 清理临时文件
                    if temp_path != actual_checkpoint_path:
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                        
                except Exception as e:
                    print(f"  Error loading LoRA checkpoint: {e}")
                    print("  Falling back to default model setup...")
                    predictor, segmenter = get_predictor_and_segmenter(
                        model_type=model_type,
                        device=device,
                        amg=False,
                        is_tiled=False
                    )
            else:
                # 使用默认模型
                predictor, segmenter = get_predictor_and_segmenter(
                    model_type=model_type,
                    device=device,
                    amg=False,
                    is_tiled=False
                )
                print("  Using default model")
        else:
            # 使用默认模型
            predictor, segmenter = get_predictor_and_segmenter(
                model_type=model_type,
                device=device,
                amg=False,
                is_tiled=False
            )
            print("  Using default model")
        
        print(f"  Model setup completed successfully")
        return predictor, segmenter
        
    except Exception as e:
        print(f"Model setup failed for {model_config['name']}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def process_test_samples_worker(args):
    """Worker函数处理测试样本"""
    (test_samples, model_config, output_dir, config, test_set_name) = args
    
    timeout_handler = TimeoutHandler(config.process_timeout)
    timeout_handler.start_timer()
    
    try:
        # 设置输出目录
        model_output_dir = Path(output_dir) / model_config['name'] / test_set_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 根据配置决定是否创建可视化相关目录
        if config.enable_visualization:
            viz_dir = model_output_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
        
        if config.save_overlays:
            overlay_dir = model_output_dir / "overlays"
            overlay_dir.mkdir(exist_ok=True)
        
        # 检查是否已经处理过
        results_file = model_output_dir / "results.csv"
        if config.skip_existing and results_file.exists():
            print(f"Skipping {model_config['name']} on {test_set_name} - already processed")
            timeout_handler.stop_timer()
            return model_config['name'], test_set_name, "skipped"
        
        # 设置模型
        gpu_id = None
        predictor, segmenter = setup_model_safe(model_config, gpu_id)
        if predictor is None:
            timeout_handler.stop_timer()
            return model_config['name'], test_set_name, "model_setup_failed"
        
        # 处理测试样本
        results = []
        visualization_candidates = []
        overlay_files = []
        total_processing_time = 0.0
        
        print(f"Processing {len(test_samples)} test samples for {model_config['name']} on {test_set_name}")
        
        for sample in tqdm(test_samples, desc=f"Processing {model_config['name']} on {test_set_name}"):
            try:
                start_time = time.time()
                timeout_handler.start_timer()
                
                img_path = sample['image_path']
                mask_path = sample['mask_path']
                sample_id = sample['sample_id']
                
                # 检查文件是否存在
                if not Path(img_path).exists() or not Path(mask_path).exists():
                    print(f"Warning: Files not found for {sample_id}, skipping...")
                    continue
                
                # 加载和处理图像
                image = io.imread(img_path)
                if len(image.shape) > 2:
                    image_for_display = image.copy()
                    image_for_seg = image[:, :, 0]
                else:
                    image_for_display = image
                    image_for_seg = image
                
                # 预测
                if segmenter is not None:
                    if hasattr(segmenter, 'initialize') and hasattr(segmenter, 'generate'):
                        segmenter.initialize(image_for_seg)
                        masks = segmenter.generate()
                        
                        from micro_sam.instance_segmentation import mask_data_to_segmentation
                        segmentation = mask_data_to_segmentation(
                            masks, with_background=True, min_object_size=0
                        )
                    else:
                        segmentation = automatic_instance_segmentation(
                            predictor=predictor,
                            segmenter=segmenter,
                            input_path=image_for_seg,
                            ndim=2
                        )
                else:
                    # 没有segmenter，使用predictor进行AMG
                    from micro_sam.instance_segmentation import AutomaticMaskGenerator, mask_data_to_segmentation
                    amg = AutomaticMaskGenerator(predictor)
                    amg.initialize(image_for_seg)
                    masks = amg.generate()
                    segmentation = mask_data_to_segmentation(
                        masks, with_background=True, min_object_size=0
                    )
                
                # 加载GT
                gt_mask = io.imread(mask_path)
                if len(gt_mask.shape) > 2:
                    gt_mask = gt_mask[:, :, 0]
                
                # 计算处理时间
                processing_time = time.time() - start_time
                total_processing_time += processing_time
                
                # 计算所有指标
                metrics = ComprehensiveMetrics.compute_all_metrics(gt_mask, segmentation)
                
                # 添加元数据
                metrics.update({
                    'sample_id': sample_id,
                    'cell_type': sample['cell_type'],
                    'date': sample['date'],
                    'magnification': sample['magnification'],
                    'dataset_id': sample['dataset_id'],
                    'model': model_config['name'],
                    'test_set': test_set_name,
                    'processing_time': processing_time,
                    'image_path': str(img_path),
                    'mask_path': str(mask_path)
                })
                
                # 添加到结果列表
                results.append(metrics)
                
                # 存储图像数据用于可能的可视化
                if config.enable_visualization:
                    visualization_candidates.append({
                        'image': image_for_display,
                        'gt_mask': gt_mask,
                        'prediction': segmentation,
                        'metrics': metrics,
                        'img_id': sample_id
                    })
                
                # 保存叠加图像（如果启用）
                if config.save_overlays:
                    # 保存GT和预测的叠加图像
                    gt_overlay_path = overlay_dir / f"gt_overlay_{sample_id}.png"
                    save_overlay_image(
                        image_for_display, gt_mask, gt_overlay_path, 
                        alpha=0.5, draw_grid=config.draw_grid, grid_size=config.grid_size
                    )
                    
                    pred_overlay_path = overlay_dir / f"pred_overlay_{sample_id}.png"
                    save_overlay_image(
                        image_for_display, segmentation, pred_overlay_path, 
                        alpha=0.5, draw_grid=config.draw_grid, grid_size=config.grid_size
                    )
                    
                    overlay_files.append({
                        'sample_id': sample_id,
                        'gt_overlay': str(gt_overlay_path),
                        'pred_overlay': str(pred_overlay_path),
                        'gt_instances': metrics['gt_instances'],
                        'pred_instances': metrics['pred_instances']
                    })
                
                # 定期清理内存
                if len(results) % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing {sample['sample_id']}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 保存结果
        if results:
            df = pd.DataFrame(results)
            df.to_csv(results_file, index=False)
            
            # 保存叠加图像列表
            if overlay_files:
                overlay_df = pd.DataFrame(overlay_files)
                overlay_df.to_csv(model_output_dir / "overlay_files.csv", index=False)
            
            # 计算平均指标
            numeric_cols = ['ap50', 'ap75', 'iou_score', 'dice_score', 'hd95', 
                           'gt_instances', 'pred_instances', 'processing_time']
            
            avg_metrics = {}
            for col in numeric_cols:
                if col in df.columns:
                    values = df[col].dropna()
                    if col == 'hd95':
                        finite_values = values[np.isfinite(values)]
                        avg_metrics[col] = float(finite_values.mean()) if len(finite_values) > 0 else float('inf')
                    else:
                        avg_metrics[col] = float(values.mean()) if len(values) > 0 else 0.0
            
            # 添加元数据到摘要
            avg_metrics.update({
                'model': model_config['name'],
                'model_type': model_config['model_type'],
                'checkpoint_path': model_config.get('checkpoint_path', 'default'),
                'test_set': test_set_name,
                'processed_samples': len(results),
                'total_available_samples': len(test_samples),
                'success_rate': len(results) / len(test_samples) if test_samples else 0.0,
                'total_processing_time': total_processing_time,
                'average_processing_time_per_sample': total_processing_time / len(results) if results else 0.0,
                'overlay_images_saved': len(overlay_files) * 2 if overlay_files else 0,
                'visualization_enabled': config.enable_visualization,
                'overlays_enabled': config.save_overlays
            })
            
            # 保存摘要
            summary_file = model_output_dir / "summary.json"
            with open(summary_file, 'w') as f:
                json.dump(avg_metrics, f, indent=2)
            
            # 创建随机样本可视化（仅当启用时）
            if config.enable_visualization and visualization_candidates:
                visualizer = ResultVisualizer(viz_dir)
                
                num_to_visualize = min(config.visual_size, len(visualization_candidates))
                random_indices = random.sample(range(len(visualization_candidates)), num_to_visualize)
                
                print(f"Creating {num_to_visualize} visualizations for {model_config['name']} on {test_set_name}...")
                for idx in random_indices:
                    candidate = visualization_candidates[idx]
                    try:
                        visualizer.visualize_comparison(
                            candidate['image'],
                            candidate['gt_mask'],
                            candidate['prediction'],
                            candidate['metrics'],
                            candidate['img_id']
                        )
                    except Exception as viz_err:
                        print(f"Visualization error for {candidate['img_id']}: {viz_err}")
                
                print(f"Saved {num_to_visualize} visualizations to {viz_dir}")
            
            print(f"Completed {model_config['name']} on {test_set_name}: processed {len(results)}/{len(test_samples)} samples")
            print(f"Average metrics - AP50: {avg_metrics['ap50']:.3f}, AP75: {avg_metrics['ap75']:.3f}, "
                  f"IoU: {avg_metrics['iou_score']:.3f}, Dice: {avg_metrics['dice_score']:.3f}")
            
            if config.save_overlays:
                grid_info = " (with grid)" if config.draw_grid else ""
                print(f"Saved {len(overlay_files)} GT+Pred overlay image pairs{grid_info} to {overlay_dir}")
        else:
            print(f"No results for {model_config['name']} on {test_set_name}")
        
        timeout_handler.stop_timer()
        return model_config['name'], test_set_name, "completed"
        
    except Exception as e:
        timeout_handler.stop_timer()
        print(f"Error processing {model_config['name']} on {test_set_name}: {e}")
        import traceback
        traceback.print_exc()
        return model_config['name'], test_set_name, f"error: {str(e)}"

def process_test_samples_worker_optimized(args):
    """优化的Worker函数 - 批量预计算嵌入"""
    (test_samples, model_config, output_dir, config, test_set_name) = args
    
    timeout_handler = TimeoutHandler(config.process_timeout)
    timeout_handler.start_timer()
    
    try:
        # 设置输出目录
        model_output_dir = Path(output_dir) / model_config['name'] / test_set_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查是否已经处理过
        results_file = model_output_dir / "results.csv"
        if config.skip_existing and results_file.exists():
            print(f"Skipping {model_config['name']} on {test_set_name} - already processed")
            timeout_handler.stop_timer()
            return model_config['name'], test_set_name, "skipped"
        
        # 设置模型
        predictor, segmenter = setup_model_safe(model_config, None)
        if predictor is None:
            timeout_handler.stop_timer()
            return model_config['name'], test_set_name, "model_setup_failed"
        
        print(f"Processing {len(test_samples)} test samples for {model_config['name']} on {test_set_name}")
        
        # === 关键优化：批量预计算嵌入 ===
        print("Step 1: Precomputing image embeddings...")
        image_embeddings_cache = {}
        
        # 批量预计算嵌入
        for i, sample in enumerate(tqdm(test_samples[:50], desc="Precomputing embeddings")):  # 先处理前50个测试
            try:
                img_path = sample['image_path']
                sample_id = sample['sample_id']
                
                if not Path(img_path).exists():
                    continue
                
                # 加载图像
                image = io.imread(img_path)
                if len(image.shape) > 2:
                    image_for_seg = image[:, :, 0]
                else:
                    image_for_seg = image
                
                # 预计算嵌入并缓存
                if segmenter is not None and hasattr(segmenter, 'initialize'):
                    # 直接调用predictor预计算嵌入
                    predictor.set_image(image_for_seg)
                    # 获取嵌入并缓存
                    embeddings = {
                        'features': predictor.features,
                        'original_size': predictor.original_size,
                        'input_size': predictor.input_size
                    }
                    image_embeddings_cache[sample_id] = embeddings
                else:
                    # 对于AMG情况
                    from micro_sam.instance_segmentation import AutomaticMaskGenerator
                    if not hasattr(process_test_samples_worker_optimized, '_amg_cache'):
                        process_test_samples_worker_optimized._amg_cache = AutomaticMaskGenerator(predictor)
                    
                    amg = process_test_samples_worker_optimized._amg_cache
                    amg.initialize(image_for_seg)
                    # 缓存AMG状态
                    image_embeddings_cache[sample_id] = amg.get_state()
                
                # 每10个清理一次内存
                if i % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except Exception as e:
                print(f"Error precomputing embeddings for {sample['sample_id']}: {e}")
                continue
        
        print(f"Precomputed embeddings for {len(image_embeddings_cache)} samples")
        
        # === Step 2: 快速推理 ===
        print("Step 2: Fast inference using cached embeddings...")
        results = []
        total_processing_time = 0.0
        
        for sample in tqdm(test_samples[:50], desc=f"Fast inference"):  # 只处理前50个进行测试
            try:
                start_time = time.time()
                
                sample_id = sample['sample_id']
                img_path = sample['image_path']
                mask_path = sample['mask_path']
                
                # 检查缓存
                if sample_id not in image_embeddings_cache:
                    continue
                
                if not Path(img_path).exists() or not Path(mask_path).exists():
                    continue
                
                # 加载图像
                image = io.imread(img_path)
                if len(image.shape) > 2:
                    image_for_display = image.copy()
                    image_for_seg = image[:, :, 0]
                else:
                    image_for_display = image
                    image_for_seg = image
                
                # === 使用缓存的嵌入进行快速推理 ===
                if segmenter is not None and hasattr(segmenter, 'generate'):
                    # 直接设置缓存的嵌入
                    cached_embeddings = image_embeddings_cache[sample_id]
                    predictor.features = cached_embeddings['features']
                    predictor.original_size = cached_embeddings['original_size'] 
                    predictor.input_size = cached_embeddings['input_size']
                    predictor.is_image_set = True
                    
                    # 如果segmenter有decoder，直接用decoder
                    if hasattr(segmenter, 'decoder'):
                        # 计算decoder输出
                        with torch.no_grad():
                            decoder_outputs = segmenter.decoder(
                                predictor.features, 
                                predictor.input_size,
                                predictor.original_size
                            )
                        
                        # 直接从decoder输出生成分割
                        from micro_sam.instance_segmentation import mask_data_to_segmentation
                        # 这里需要根据具体的decoder输出格式调整
                        segmentation = process_decoder_output(decoder_outputs, predictor.original_size)
                    else:
                        # 使用AMG方式
                        masks = segmenter.generate()
                        from micro_sam.instance_segmentation import mask_data_to_segmentation
                        segmentation = mask_data_to_segmentation(
                            masks, with_background=True, min_object_size=0
                        )
                else:
                    # AMG情况 - 使用缓存状态
                    from micro_sam.instance_segmentation import AutomaticMaskGenerator, mask_data_to_segmentation
                    amg = AutomaticMaskGenerator(predictor)
                    amg.set_state(image_embeddings_cache[sample_id])
                    masks = amg.generate()
                    segmentation = mask_data_to_segmentation(
                        masks, with_background=True, min_object_size=0
                    )
                
                # 加载GT
                gt_mask = io.imread(mask_path)
                if len(gt_mask.shape) > 2:
                    gt_mask = gt_mask[:, :, 0]
                
                # 计算处理时间（应该显著减少）
                processing_time = time.time() - start_time
                total_processing_time += processing_time
                
                # 计算指标
                metrics = ComprehensiveMetrics.compute_all_metrics(gt_mask, segmentation)
                
                # 添加元数据
                metrics.update({
                    'sample_id': sample_id,
                    'cell_type': sample['cell_type'],
                    'model': model_config['name'],
                    'test_set': test_set_name,
                    'processing_time': processing_time,
                    'used_cached_embeddings': True
                })
                
                results.append(metrics)
                
                # 每处理10个样本显示一次时间
                if len(results) % 10 == 0:
                    avg_time = total_processing_time / len(results)
                    print(f"Processed {len(results)} samples, avg time: {avg_time:.2f}s per sample")
                
            except Exception as e:
                print(f"Error in fast inference for {sample['sample_id']}: {e}")
                continue
        
        # 保存结果
        if results:
            df = pd.DataFrame(results)
            df.to_csv(results_file, index=False)
            
            avg_time = total_processing_time / len(results) if results else 0
            print(f"Completed with cached embeddings: {len(results)} samples")
            print(f"Average processing time: {avg_time:.2f}s per sample (should be much faster!)")
            
            # 保存时间对比信息
            timing_info = {
                'total_samples': len(results),
                'total_time': total_processing_time,
                'avg_time_per_sample': avg_time,
                'used_embedding_cache': True,
                'cache_size': len(image_embeddings_cache)
            }
            
            with open(model_output_dir / "timing_info.json", 'w') as f:
                json.dump(timing_info, f, indent=2)
        
        timeout_handler.stop_timer()
        return model_config['name'], test_set_name, "completed_fast"
        
    except Exception as e:
        timeout_handler.stop_timer()
        print(f"Error in optimized processing: {e}")
        return model_config['name'], test_set_name, f"error: {str(e)}"

def process_decoder_output(decoder_outputs, original_size):
    """处理decoder输出为分割mask"""
    try:
        # 这个函数需要根据具体的decoder输出格式实现
        # 一般decoder输出包含前景、边界、距离等预测
        
        if isinstance(decoder_outputs, dict):
            # 如果有foreground预测
            if 'foreground' in decoder_outputs:
                foreground = decoder_outputs['foreground']
            else:
                # 假设第一个输出是foreground
                foreground = list(decoder_outputs.values())[0]
        else:
            # 直接tensor输出
            foreground = decoder_outputs
        
        # 转换为numpy并阈值化
        if torch.is_tensor(foreground):
            foreground = foreground.cpu().numpy()
        
        # 简单阈值化
        binary_mask = (foreground > 0.5).astype(np.uint8)
        
        # 连通组件标记
        from skimage import measure
        labeled_mask = measure.label(binary_mask)
        
        return labeled_mask
        
    except Exception as e:
        print(f"Error processing decoder output: {e}")
        # 返回空mask
        return np.zeros(original_size, dtype=np.uint32)

# 替换原始的处理函数
def run_optimized_evaluation(config):
    """运行优化的评测"""
    
    # 只处理第一个checkpoint和第一个测试集进行测试
    test_manager = LoRATestSetManager(config)
    all_test_sets = test_manager.get_all_test_sets()
    
    first_test_set_name = list(all_test_sets.keys())[0]
    first_test_samples = all_test_sets[first_test_set_name]
    first_checkpoint = config.checkpoints[0]
    
    print(f"Testing optimized inference on:")
    print(f"  Checkpoint: {first_checkpoint['name']}")
    print(f"  Test set: {first_test_set_name}")
    print(f"  Samples: {len(first_test_samples)} (will process first 50)")
    
    args = (
        first_test_samples[:50],  # 只处理前50个进行测试
        first_checkpoint,
        config.output_base_dir,
        config,
        first_test_set_name
    )
    
    start_time = time.time()
    result = process_test_samples_worker_optimized(args)
    end_time = time.time()
    
    print(f"\nOptimized inference result: {result}")
    print(f"Total time for 50 samples: {end_time - start_time:.2f}s")
    print(f"Average time per sample: {(end_time - start_time)/50:.2f}s")
    
    return result


class LoRAExperimentConfig:
    """LoRA实验批量评测配置"""
    
    def __init__(self):
        # 项目路径配置
        self.project_root = Path("/LD-FS/home/yunshuchen/DeepMicroSeg/microsam")
        self.cache_root = Path("/LD-FS/home/yunshuchen/DeepMicroSeg/microsam/Retrain/micro_sam_cache")
        
        # 测试集配置 - 三组测试集合
        self.test_sets = {
            'split_0.18_0.02_0.80': {
                'name': 'split_0.18_0.02_0.80',
                'base_dir': '/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/lora_split',
                'cell_types': ['293T', 'MSC', 'RBD', 'VERO']
            },
            'split_0.27_0.03_0.70': {
                'name': 'split_0.27_0.03_0.70',
                'base_dir': '/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/lora_split',
                'cell_types': ['293T', 'MSC', 'RBD', 'VERO']
            },
            'split_0.36_0.04_0.60': {
                'name': 'split_0.36_0.04_0.60',
                'base_dir': '/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/lora_split',
                'cell_types': ['293T', 'MSC', 'RBD', 'VERO']
            }
        }
        
        # LoRA checkpoint配置 - 修正为实际的vit_b_lm模型类型
        self.checkpoints = [
            {
                "name": "lora_293t_vit_b_lm_r8",
                "model_type": "vit_b_lm",  # 修正为实际的模型类型
                "checkpoint_path": "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/lora_experiments/lora_experiments_293t_train10_val10_test80/lora_finetune_vit_b_lm_r8/final_model/merged_sam_model.pth",
                "lora_rank": 8,
                "attention_layers": [9, 10, 11],
                "cell_type_trained": "293T"
            },
            {
                "name": "lora_msc_vit_b_lm_r8",
                "model_type": "vit_b_lm",  # 修正为实际的模型类型
                "checkpoint_path": "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/lora_experiments/lora_experiments_msc_train10_val10_test80/lora_finetune_vit_b_lm_r8/final_model/merged_sam_model.pth",
                "lora_rank": 8,
                "attention_layers": [9, 10, 11],
                "cell_type_trained": "MSC"
            },
            {
                "name": "lora_rbd_vit_b_lm_r8",
                "model_type": "vit_b_lm",  # 修正为实际的模型类型
                "checkpoint_path": "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/lora_experiments/lora_experiments_rbd_train10_val10_test80/lora_finetune_vit_b_lm_r8/final_model/merged_sam_model.pth",
                "lora_rank": 8,
                "attention_layers": [9, 10, 11],
                "cell_type_trained": "RBD"
            },
            {
                "name": "lora_vero_vit_b_lm_r8",
                "model_type": "vit_b_lm",  # 修正为实际的模型类型
                "checkpoint_path": "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/lora_experiments/lora_experiments_vero_train10_val10_test80/lora_finetune_vit_b_lm_r8/final_model/merged_sam_model.pth",
                "lora_rank": 8,
                "attention_layers": [9, 10, 11],
                "cell_type_trained": "VERO"
            }
        ]
        
        # 硬件配置
        self.max_gpu_workers = 4
        self.process_timeout = 600  # 10分钟超时
        
        # 输出配置
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_base_dir = str(self.cache_root / f"lora_evaluation_results_{timestamp}")
        
        # 可视化配置
        self.enable_visualization = True
        self.visual_size = 10  # 每个测试集随机选择的可视化样本数量
        self.save_overlays = True
        self.draw_grid = True
        self.grid_size = 50
        
        self.save_detailed_metrics = True
        
        # 缓存配置
        self.cache_dir = str(self.cache_root / "embeddings_cache")
        
        # 评测配置
        self.skip_existing = True
        self.create_summary_report = True
        self.generate_unified_csv = True

class LoRATestSetManager:
    """LoRA测试集管理器 - 从JSON文件中读取测试样本"""
    
    def __init__(self, config: LoRAExperimentConfig):
        self.config = config
        self.test_data = {}
        self._load_all_test_sets()
    
    def _load_all_test_sets(self):
        """加载所有测试集"""
        for test_set_key, test_set_info in self.config.test_sets.items():
            test_set_name = test_set_info['name']
            base_dir = Path(test_set_info['base_dir'])
            cell_types = test_set_info['cell_types']
            
            # 为每个测试集收集所有细胞类型的测试样本
            all_test_samples = []
            
            for cell_type in cell_types:
                # 构造JSON文件路径
                json_pattern = f"split_{test_set_name.split('_')[1]}_{test_set_name.split('_')[2]}_{test_set_name.split('_')[3]}_{cell_type}_*.json"
                json_files = list(base_dir.glob(json_pattern))
                
                if not json_files:
                    print(f"Warning: No JSON files found for {test_set_name} {cell_type}")
                    continue
                
                # 通常应该只有一个匹配的文件
                for json_file in json_files:
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                        
                        # 提取测试样本
                        test_samples = data.get('test_samples', [])
                        print(f"Loaded {len(test_samples)} test samples from {json_file.name}")
                        
                        # 为每个测试样本添加细胞类型信息
                        for sample in test_samples:
                            sample['cell_type'] = cell_type
                            sample['test_set_origin'] = test_set_name
                        
                        all_test_samples.extend(test_samples)
                        
                    except Exception as e:
                        print(f"Error loading {json_file}: {e}")
                        continue
            
            self.test_data[test_set_name] = all_test_samples
            print(f"Total test samples for {test_set_name}: {len(all_test_samples)}")
    
    def get_test_samples(self, test_set_name: str) -> List[Dict]:
        """获取指定测试集的样本"""
        return self.test_data.get(test_set_name, [])
    
    def get_all_test_sets(self) -> Dict[str, List[Dict]]:
        """获取所有测试集"""
        return self.test_data
    
    def get_summary(self) -> pd.DataFrame:
        """获取测试集摘要"""
        summary_data = []
        
        for test_set_name, samples in self.test_data.items():
            # 按细胞类型统计
            cell_type_counts = {}
            for sample in samples:
                cell_type = sample.get('cell_type', 'unknown')
                cell_type_counts[cell_type] = cell_type_counts.get(cell_type, 0) + 1
            
            for cell_type, count in cell_type_counts.items():
                summary_data.append({
                    'test_set': test_set_name,
                    'cell_type': cell_type,
                    'sample_count': count
                })
        
        return pd.DataFrame(summary_data)

class LoRABatchEvaluator:
    """LoRA批量评测器"""
    
    def __init__(self, config: LoRAExperimentConfig):
        self.config = config
        self.test_manager = None
        self.results_summary = []
    
    def setup(self):
        """设置测试集管理器"""
        self.test_manager = LoRATestSetManager(self.config)
        
        # 创建输出目录和缓存目录
        os.makedirs(self.config.output_base_dir, exist_ok=True)
        os.makedirs(self.config.cache_dir, exist_ok=True)
        os.environ["MICROSAM_CACHEDIR"] = self.config.cache_dir
        
        print(f"发现测试集:")
        summary_df = self.test_manager.get_summary()
        print(summary_df.to_string(index=False))
    
    def run_evaluation(self):
        """运行批量评测 - 交叉评测所有checkpoint和测试集"""
        if self.test_manager is None:
            raise ValueError("Please call setup() first")
        
        # 获取所有测试集
        all_test_sets = self.test_manager.get_all_test_sets()
        
        print(f"将进行交叉评测:")
        print(f"  Checkpoints: {len(self.config.checkpoints)}")
        print(f"  Test Sets: {len(all_test_sets)}")
        print(f"  Total Combinations: {len(self.config.checkpoints) * len(all_test_sets)}")
        
        # 创建任务列表 - 每个checkpoint在每个测试集上都要评测
        args_list = []
        for test_set_name, test_samples in all_test_sets.items():
            if not test_samples:
                print(f"Warning: No test samples for {test_set_name}, skipping...")
                continue
                
            for checkpoint_config in self.config.checkpoints:
                args_list.append((
                    test_samples, 
                    checkpoint_config, 
                    self.config.output_base_dir, 
                    self.config, 
                    test_set_name
                ))
        
        print(f"\n开始处理 {len(args_list)} 个评测任务...")
        
        # 逐个处理任务（避免GPU资源竞争）
        results = []
        for i, args in enumerate(args_list):
            model_name = args[1]['name']
            test_set_name = args[4]
            
            print(f"\n[{i+1}/{len(args_list)}] Processing {model_name} on {test_set_name}")
            result = process_test_samples_worker(args)
            results.append(result)
            print(f"Completed: {result[0]} on {result[1]} - {result[2]}")
            
            # 每处理完一个任务后清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 生成摘要报告
        if self.config.create_summary_report:
            self._create_comprehensive_summary_report(results)
    
    def _create_comprehensive_summary_report(self, results):
        """创建全面的摘要报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path(self.config.output_base_dir) / f"summary_report_{timestamp}"
        report_dir.mkdir(exist_ok=True)
        
        # 收集所有详细结果和摘要
        all_detailed_results = []
        all_summaries = []
        
        for checkpoint_config in self.config.checkpoints:
            checkpoint_name = checkpoint_config['name']
            checkpoint_dir = Path(self.config.output_base_dir) / checkpoint_name
            
            if not checkpoint_dir.exists():
                continue
                
            for test_set_dir in checkpoint_dir.iterdir():
                if not test_set_dir.is_dir():
                    continue
                    
                # 收集详细结果
                results_file = test_set_dir / "results.csv"
                if results_file.exists():
                    df = pd.read_csv(results_file)
                    all_detailed_results.append(df)
                
                # 收集摘要
                summary_file = test_set_dir / "summary.json"
                if summary_file.exists():
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                        summary['checkpoint_name'] = checkpoint_name
                        summary['test_set_name'] = test_set_dir.name
                        all_summaries.append(summary)
        
        # 生成统一的详细结果CSV
        if all_detailed_results and self.config.generate_unified_csv:
            unified_df = pd.concat(all_detailed_results, ignore_index=True)
            unified_csv_path = report_dir / "unified_detailed_results.csv"
            unified_df.to_csv(unified_csv_path, index=False)
            print(f"统一详细结果已保存到: {unified_csv_path}")
        
        # 生成交叉评测摘要表
        if all_summaries:
            summary_df = pd.DataFrame(all_summaries)
            
            # 重新排列列顺序，突出交叉评测结构
            column_order = [
                'checkpoint_name', 'model_type', 'checkpoint_path', 'test_set_name',
                'ap50', 'ap75', 'iou_score', 'dice_score', 'hd95',
                'gt_instances', 'pred_instances', 'processing_time',
                'processed_samples', 'total_processing_time'
            ]
            
            available_columns = [col for col in column_order if col in summary_df.columns]
            summary_df_ordered = summary_df[available_columns]
            
            # 保存交叉评测摘要
            cross_eval_summary_path = report_dir / "cross_evaluation_summary.csv"
            summary_df_ordered.to_csv(cross_eval_summary_path, index=False)
            print(f"交叉评测摘要已保存到: {cross_eval_summary_path}")
            
            # 创建交叉评测矩阵 (Checkpoint vs TestSet)
            self._create_cross_evaluation_matrix(summary_df, report_dir)
            
            # 生成按checkpoint分组的统计
            checkpoint_stats = summary_df.groupby('checkpoint_name')[
                ['ap50', 'ap75', 'iou_score', 'dice_score', 'hd95', 'processing_time']
            ].agg(['mean', 'std']).round(4)
            checkpoint_stats.to_csv(report_dir / "checkpoint_performance_statistics.csv")
            
            # 生成按测试集分组的统计
            testset_stats = summary_df.groupby('test_set_name')[
                ['ap50', 'ap75', 'iou_score', 'dice_score', 'hd95', 'processing_time']
            ].agg(['mean', 'std']).round(4)
            testset_stats.to_csv(report_dir / "testset_performance_statistics.csv")
            
            # 创建可视化
            if self.config.enable_visualization:
                self._create_cross_evaluation_visualizations(summary_df, report_dir)
            
            print(f"完整的LoRA交叉评测报告已保存到: {report_dir}")
            
            # 打印交叉评测统计
            self._print_cross_evaluation_summary(summary_df)
    
    def _create_cross_evaluation_matrix(self, summary_df: pd.DataFrame, output_dir: Path):
        """创建交叉评测矩阵"""
        metrics = ['ap50', 'ap75', 'iou_score', 'dice_score']
        
        for metric in metrics:
            # 创建数据透视表
            pivot_table = summary_df.pivot(
                index='checkpoint_name', 
                columns='test_set_name', 
                values=metric
            )
            
            # 保存为CSV
            matrix_path = output_dir / f"cross_evaluation_matrix_{metric}.csv"
            pivot_table.to_csv(matrix_path)
            
            # 创建热力图
            if self.config.enable_visualization:
                plt.figure(figsize=(12, 8))
                sns.heatmap(
                    pivot_table, 
                    annot=True, 
                    fmt='.3f', 
                    cmap='viridis',
                    cbar_kws={'label': metric.upper()}
                )
                plt.title(f'Cross-Evaluation Matrix: {metric.upper()}')
                plt.xlabel('Test Set')
                plt.ylabel('Checkpoint')
                plt.tight_layout()
                
                heatmap_path = output_dir / f"cross_evaluation_heatmap_{metric}.png"
                plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                plt.close()
    
    def _create_cross_evaluation_visualizations(self, summary_df: pd.DataFrame, output_dir: Path):
        """创建交叉评测可视化图表"""
        try:
            # 1. 整体性能对比图
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            metrics = ['ap50', 'ap75', 'iou_score', 'dice_score']
            metric_labels = ['AP50', 'AP75', 'IoU Score', 'Dice Score']
            
            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                ax = axes[i]
                
                # 创建分组箱线图
                checkpoint_names = summary_df['checkpoint_name'].unique()
                colors = plt.cm.Set3(np.linspace(0, 1, len(checkpoint_names)))
                
                positions = []
                all_data = []
                all_labels = []
                
                for j, checkpoint in enumerate(checkpoint_names):
                    checkpoint_data = summary_df[summary_df['checkpoint_name'] == checkpoint]
                    values = checkpoint_data[metric].values
                    
                    if len(values) > 0:
                        # 为每个checkpoint创建箱线图位置
                        pos = np.arange(len(values)) + j * (len(values) + 1)
                        positions.extend(pos)
                        all_data.extend(values)
                        
                        # 添加标签
                        test_sets = checkpoint_data['test_set_name'].values
                        labels = [f"{checkpoint[:10]}\n{ts}" for ts in test_sets]
                        all_labels.extend(labels)
                
                # 创建条形图而非箱线图（因为每个组合只有一个值）
                bars = ax.bar(range(len(all_data)), all_data, 
                             color=[colors[i//3] for i in range(len(all_data))])
                
                ax.set_title(label, fontsize=14, fontweight='bold')
                ax.set_xticks(range(len(all_labels)))
                ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)
                ax.grid(True, alpha=0.3)
                
                # 添加数值标签
                for bar, val in zip(bars, all_data):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(output_dir / "cross_evaluation_performance_comparison.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 按checkpoint分组的雷达图
            self._create_radar_chart(summary_df, output_dir)
            
            print("交叉评测可视化图表创建完成")
            
        except Exception as e:
            print(f"可视化错误: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_radar_chart(self, summary_df: pd.DataFrame, output_dir: Path):
        """创建雷达图显示各checkpoint的综合性能"""
        try:
            from math import pi
            
            # 选择用于雷达图的指标
            metrics = ['ap50', 'ap75', 'iou_score', 'dice_score']
            metric_labels = ['AP50', 'AP75', 'IoU', 'Dice']
            
            # 按checkpoint分组并计算平均值
            checkpoint_means = summary_df.groupby('checkpoint_name')[metrics].mean()
            
            # 设置雷达图
            N = len(metrics)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]  # 闭合图形
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(checkpoint_means)))
            
            for i, (checkpoint, values) in enumerate(checkpoint_means.iterrows()):
                values_list = values.tolist()
                values_list += values_list[:1]  # 闭合图形
                
                ax.plot(angles, values_list, 'o-', linewidth=2, 
                       label=checkpoint, color=colors[i])
                ax.fill(angles, values_list, alpha=0.25, color=colors[i])
            
            # 设置标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels)
            ax.set_ylim(0, 1)
            ax.set_title('Checkpoint Performance Radar Chart', size=16, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(output_dir / "checkpoint_performance_radar.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"雷达图创建错误: {e}")
    
    def _print_cross_evaluation_summary(self, summary_df: pd.DataFrame):
        """打印交叉评测摘要统计"""
        print("\n" + "="*80)
        print("LoRA交叉评测结果摘要")
        print("="*80)
        
        # 按checkpoint显示性能
        print("\n按Checkpoint性能排名 (平均AP50):")
        checkpoint_perf = summary_df.groupby('checkpoint_name')['ap50'].mean().sort_values(ascending=False)
        for i, (checkpoint, score) in enumerate(checkpoint_perf.items()):
            trained_cell = next(
                (c['cell_type_trained'] for c in self.config.checkpoints if c['name'] == checkpoint), 
                'Unknown'
            )
            print(f"  {i+1}. {checkpoint} (trained on {trained_cell}): {score:.3f}")
        
        # 按测试集显示难度
        print("\n按Test Set难度排名 (平均AP50, 从易到难):")
        testset_perf = summary_df.groupby('test_set_name')['ap50'].mean().sort_values(ascending=False)
        for i, (test_set, score) in enumerate(testset_perf.items()):
            print(f"  {i+1}. {test_set}: {score:.3f}")
        
        # 显示交叉表现最佳组合
        print("\n最佳组合 (Top 5 AP50):")
        top_combinations = summary_df.nlargest(5, 'ap50')[
            ['checkpoint_name', 'test_set_name', 'ap50', 'ap75', 'iou_score', 'dice_score']
        ]
        for _, row in top_combinations.iterrows():
            trained_cell = next(
                (c['cell_type_trained'] for c in self.config.checkpoints if c['name'] == row['checkpoint_name']), 
                'Unknown'
            )
            print(f"  {row['checkpoint_name']} (trained on {trained_cell}) + {row['test_set_name']}: "
                  f"AP50={row['ap50']:.3f}, AP75={row['ap75']:.3f}, IoU={row['iou_score']:.3f}, Dice={row['dice_score']:.3f}")
        
        # 域适应性分析
        print("\n域适应性分析:")
        for checkpoint_config in self.config.checkpoints:
            checkpoint_name = checkpoint_config['name']
            trained_cell = checkpoint_config['cell_type_trained']
            
            checkpoint_data = summary_df[summary_df['checkpoint_name'] == checkpoint_name]
            if len(checkpoint_data) == 0:
                continue
            
            print(f"\n  {checkpoint_name} (trained on {trained_cell}):")
            
            for _, row in checkpoint_data.iterrows():
                test_set = row['test_set_name']
                ap50 = row['ap50']
                
                # 分析是否在同域还是跨域
                if trained_cell.lower() in test_set.lower():
                    domain_type = "同域"
                else:
                    domain_type = "跨域"
                
                print(f"    {test_set} ({domain_type}): AP50={ap50:.3f}")

def main():
    """主函数 - LoRA实验交叉评测"""
    print("="*80)
    print("LoRA实验批量交叉评测系统")
    print("支持多checkpoint × 多测试集的完整交叉评测")
    print("包含AP50、AP75、IoU、Dice、HD95等完整指标")
    print("="*80)
    
    # 配置
    config = LoRAExperimentConfig()
    
    # 显示配置信息
    print(f"\n配置信息:")
    print(f"  输出目录: {config.output_base_dir}")
    print(f"  缓存目录: {config.cache_dir}")
    
    print(f"\n测试集配置:")
    for test_set_name, test_set_info in config.test_sets.items():
        print(f"  {test_set_name}:")
        print(f"    基础目录: {test_set_info['base_dir']}")
        print(f"    细胞类型: {test_set_info['cell_types']}")
    
    print(f"\nCheckpoint配置:")
    for i, checkpoint in enumerate(config.checkpoints):
        print(f"  {i+1}. {checkpoint['name']}")
        print(f"     模型类型: {checkpoint['model_type']}")
        print(f"     训练细胞: {checkpoint['cell_type_trained']}")
        print(f"     LoRA Rank: {checkpoint['lora_rank']}")
        print(f"     Checkpoint: {checkpoint['checkpoint_path']}")
        print(f"     文件存在: {Path(checkpoint['checkpoint_path']).exists()}")
        print()
    
    # 设置运行参数
    config.enable_visualization = True
    config.visual_size = 10
    config.save_overlays = True
    config.draw_grid = True
    config.grid_size = 50
    config.skip_existing = True
    config.generate_unified_csv = True
    
    print(f"运行配置:")
    print(f"  可视化模式: {'计算+可视化' if config.enable_visualization else '仅计算指标'}")
    if config.enable_visualization:
        print(f"  可视化样本数: {config.visual_size}")
    print(f"  保存叠加图像: {config.save_overlays}")
    if config.save_overlays:
        print(f"  绘制网格: {config.draw_grid}")
        if config.draw_grid:
            print(f"  网格大小: {config.grid_size}px")
    print(f"  跳过已有结果: {config.skip_existing}")
    
    # 创建评测器
    evaluator = LoRABatchEvaluator(config)
    
    # 设置评测器
    evaluator.setup()
    
    # 计算总工作量
    total_checkpoints = len(config.checkpoints)
    total_test_sets = len(config.test_sets)
    total_combinations = total_checkpoints * total_test_sets
    
    print(f"\n交叉评测工作量:")
    print(f"  Checkpoints: {total_checkpoints} (293T, MSC, RBD, VERO)")
    print(f"  Test Sets: {total_test_sets}")
    print(f"  总组合数: {total_combinations}")
    
    # 详细显示交叉评测矩阵
    print(f"\n交叉评测矩阵 ({total_checkpoints} × {total_test_sets}):")
    print("  Checkpoints (trained on):")
    for i, checkpoint in enumerate(config.checkpoints, 1):
        print(f"    {i}. {checkpoint['name']} (trained on {checkpoint['cell_type_trained']})")
    print("  Test Sets:")
    for i, test_set in enumerate(config.test_sets.keys(), 1):
        print(f"    {i}. {test_set}")
    print(f"  → 每个checkpoint将在所有{total_test_sets}个测试集上评测")
    
    # 运行评测
    print("\n" + "="*80)
    print("开始LoRA交叉评测...")
    print("每个checkpoint将在所有测试集上评测")
    if config.enable_visualization:
        print("模式: 计算指标 + 可视化")
    else:
        print("模式: 仅计算指标")
    print("="*80)
    
    start_time = time.time()
    
    # 运行交叉评测
    evaluator.run_evaluation()
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("LoRA交叉评测完成!")
    print("="*80)
    print(f"总耗时: {total_time/3600:.2f} 小时")
    print(f"结果保存在: {config.output_base_dir}")
    
    # 显示最终结果文件
    summary_dirs = list(Path(config.output_base_dir).glob("summary_report_*"))
    if summary_dirs:
        latest_summary = max(summary_dirs, key=lambda x: x.name)
        
        print(f"\n主要结果文件:")
        
        # 交叉评测摘要文件
        cross_eval_file = latest_summary / "cross_evaluation_summary.csv"
        if cross_eval_file.exists():
            print(f"  交叉评测摘要: {cross_eval_file}")
            
            # 显示简要统计
            df = pd.read_csv(cross_eval_file)
            print(f"  包含 {len(df)} 条交叉评测记录")
            print(f"  Checkpoints: {df['checkpoint_name'].unique().tolist()}")
            print(f"  Test Sets: {df['test_set_name'].unique().tolist()}")
        
        # 详细结果文件
        unified_csv = latest_summary / "unified_detailed_results.csv"
        if unified_csv.exists():
            print(f"  详细结果: {unified_csv}")
        
        # 交叉评测矩阵文件
        matrix_files = list(latest_summary.glob("cross_evaluation_matrix_*.csv"))
        if matrix_files:
            print(f"  交叉评测矩阵:")
            for matrix_file in matrix_files:
                metric = matrix_file.stem.split('_')[-1]
                print(f"    - {metric.upper()}: {matrix_file}")
        
        # 可视化文件
        if config.enable_visualization:
            viz_files = [
                "cross_evaluation_performance_comparison.png",
                "checkpoint_performance_radar.png"
            ]
            viz_files.extend([f"cross_evaluation_heatmap_{m}.png" for m in ['ap50', 'ap75', 'iou_score', 'dice_score']])
            
            print(f"  可视化图表: {latest_summary}")
            for viz_file in viz_files:
                if (latest_summary / viz_file).exists():
                    print(f"    - {viz_file}")
    
    print("\n重要发现:")
    print("- 这是LoRA模型的交叉域评测结果")
    print("- 可以分析域适应性：同域 vs 跨域性能")
    print("- 查看交叉评测矩阵了解各checkpoint在不同测试集上的表现")
    print("- 雷达图显示各checkpoint的综合性能对比")
    
    print("\n评测系统执行完毕！")
    mode_info = "计算+可视化" if config.enable_visualization else "仅计算指标"
    overlay_info = f"，叠加图像{'（含网格）' if config.draw_grid else ''}" if config.save_overlays else ""
    print(f"模式: {mode_info}{overlay_info}，完整的交叉评测报告已生成。")
    
    print("\n分析建议:")
    print("1. 查看 cross_evaluation_summary.csv 了解整体表现")
    print("2. 查看交叉评测矩阵分析checkpoint在不同数据集上的泛化能力")
    print("3. 比较同域和跨域性能，评估域适应性")
    print("4. 使用雷达图对比不同checkpoint的综合性能")

if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main()