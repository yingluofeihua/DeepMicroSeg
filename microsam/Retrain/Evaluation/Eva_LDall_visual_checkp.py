"""
优化的细胞分割批量评测系统
修复了问题并添加了仅计算/计算+可视化的参数控制
支持自定义checkpoint
适配新的项目路径结构
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
        # 如果没有标签，返回一个简单的colormap
        return ListedColormap([(0, 0, 0)])
    
    # Generate random colors for each label (excluding background)
    np.random.seed(42)  # 使用固定的随机种子确保颜色一致性
    colors = [(np.random.random(), np.random.random(), np.random.random()) for _ in range(n_labels)]
    # Insert black as the first color (for background)
    colors.insert(0, (0, 0, 0))
    return ListedColormap(colors)

def draw_grid_on_image(image, grid_size=50):
    """在图像上绘制网格线
    
    Args:
        image: 输入图像数组
        grid_size: 网格大小（像素）
    
    Returns:
        带网格线的图像
    """
    img_with_grid = image.copy()
    h, w = image.shape[:2]
    
    # 绘制垂直线
    for x in range(0, w, grid_size):
        if len(img_with_grid.shape) == 3:
            img_with_grid[:, x:x+1, :] = [255, 255, 255]  # 白色线条
        else:
            img_with_grid[:, x:x+1] = 255
    
    # 绘制水平线
    for y in range(0, h, grid_size):
        if len(img_with_grid.shape) == 3:
            img_with_grid[y:y+1, :, :] = [255, 255, 255]  # 白色线条
        else:
            img_with_grid[y:y+1, :] = 255
    
    return img_with_grid

def find_matched_pairs(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
    """找到有对应mask的image文件对"""
    matched_pairs = []
    
    # 获取所有图像文件
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    
    for img_path in image_files:
        img_stem = img_path.stem
        
        # 尝试不同的掩码命名模式
        mask_patterns = [
            f"{img_stem}_seg.png",
            f"{img_stem}.png",
            f"{img_stem}_mask.png",
            f"{img_stem}_seg.tif",
            f"{img_stem}.tif"
        ]
        
        for pattern in mask_patterns:
            potential_mask = masks_dir / pattern
            if potential_mask.exists():
                matched_pairs.append((img_path, potential_mask))
                break
    
    return matched_pairs

def check_visualizations(config):
    """检查每个数据集的可视化结果"""
    print("\n检查可视化结果:")
    
    all_viz_count = 0
    
    # 修复：从config.models中正确提取模型名称
    for model_config in config.models:
        model_name = model_config['name']  # 提取模型名称字符串
        model_dir = Path(config.output_base_dir) / model_name  # 使用字符串而不是整个字典
        
        if not model_dir.exists():
            print(f"  {model_name}: 目录不存在")
            continue
            
        model_viz_count = 0
        print(f"  {model_name}:")
        
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
                
            viz_dir = dataset_dir / "visualizations"
            if viz_dir.exists():
                # 查找所有可视化文件
                viz_files = list(viz_dir.glob("comparison_*.png"))
                if viz_files:
                    print(f"    - {dataset_dir.name}: {len(viz_files)} 可视化样本")
                    model_viz_count += len(viz_files)
                    
                    # 列出部分可视化文件名
                    if len(viz_files) > 0:
                        max_show = min(3, len(viz_files))
                        sample_files = [f.name for f in viz_files[:max_show]]
                        print(f"      样例: {', '.join(sample_files)}")
            else:
                print(f"    - {dataset_dir.name}: 无可视化目录")
        
        if model_viz_count > 0:
            print(f"    总计: {model_viz_count} 个可视化")
        else:
            print(f"    总计: 0 个可视化")
        all_viz_count += model_viz_count
    
    print(f"\n所有模型共计: {all_viz_count} 个可视化")
    
    if all_viz_count == 0:
        print("\n未发现任何可视化结果，可能需要检查:")
        print("  1. 确认 config.enable_visualization 是否设置为 True")
        print("  2. 检查权限是否允许写入可视化目录")
        print("  3. 查看脚本输出中是否有与可视化相关的错误消息")
        print("  4. 检查数据集是否成功处理完成")
    else:
        print(f"\n可视化检查完成！发现 {all_viz_count} 个可视化文件")

def check_overlay_results(config):
    """检查叠加图像结果"""
    print("\n检查叠加图像结果:")
    
    all_overlay_count = 0
    
    for model_config in config.models:
        model_name = model_config['name']
        model_dir = Path(config.output_base_dir) / model_name
        
        if not model_dir.exists():
            continue
            
        model_overlay_count = 0
        print(f"  {model_name}:")
        
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
                
            overlay_dir = dataset_dir / "overlays"
            if overlay_dir.exists():
                # 查找GT和预测叠加图像
                gt_overlays = list(overlay_dir.glob("gt_overlay_*.png"))
                pred_overlays = list(overlay_dir.glob("pred_overlay_*.png"))
                
                if gt_overlays or pred_overlays:
                    total_overlays = len(gt_overlays) + len(pred_overlays)
                    print(f"    - {dataset_dir.name}: {len(gt_overlays)} GT + {len(pred_overlays)} 预测叠加图像")
                    model_overlay_count += total_overlays
            else:
                print(f"    - {dataset_dir.name}: 无叠加图像目录")
        
        if model_overlay_count > 0:
            print(f"    总计: {model_overlay_count} 个叠加图像")
        else:
            print(f"    总计: 0 个叠加图像")
        all_overlay_count += model_overlay_count
    
    print(f"\n所有模型共计: {all_overlay_count} 个叠加图像")

class DatasetManager:
    """数据集管理器 - 自动发现和组织所有数据集"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.datasets = self._discover_datasets()
    
    def _discover_datasets(self) -> List[Dict]:
        """自动发现所有数据集"""
        datasets = []
        
        try:
            # 遍历所有可能的数据集路径
            for cell_type in self.base_dir.iterdir():
                if not cell_type.is_dir():
                    continue
                    
                for date_dir in cell_type.iterdir():
                    if not date_dir.is_dir():
                        continue
                        
                    for magnification in date_dir.iterdir():
                        if not magnification.is_dir():
                            continue
                            
                        images_dir = magnification / "images"
                        masks_dir = magnification / "masks"
                        
                        # 检查是否存在images和masks文件夹
                        if images_dir.exists() and masks_dir.exists():
                            # 检查是否有实际的图像文件
                            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
                            mask_files = list(masks_dir.glob("*.png")) + list(masks_dir.glob("*.tif"))
                            
                            if image_files and mask_files:
                                dataset_info = {
                                    'cell_type': cell_type.name,
                                    'date': date_dir.name,
                                    'magnification': magnification.name,
                                    'images_dir': str(images_dir),
                                    'masks_dir': str(masks_dir),
                                    'num_images': len(image_files),
                                    'num_masks': len(mask_files),
                                    'dataset_id': f"{cell_type.name}_{date_dir.name}_{magnification.name}"
                                }
                                datasets.append(dataset_info)
                                
        except Exception as e:
            print(f"Error discovering datasets: {e}")
        
        return sorted(datasets, key=lambda x: (x['cell_type'], x['date'], x['magnification']))
    
    def get_datasets_summary(self) -> pd.DataFrame:
        """获取数据集摘要"""
        return pd.DataFrame(self.datasets)
    
    def filter_datasets(self, cell_types=None, dates=None, magnifications=None) -> List[Dict]:
        """过滤数据集"""
        filtered = self.datasets
        
        if cell_types:
            filtered = [d for d in filtered if d['cell_type'] in cell_types]
        if dates:
            filtered = [d for d in filtered if d['date'] in dates]
        if magnifications:
            filtered = [d for d in filtered if d['magnification'] in magnifications]
            
        return filtered

class BatchEvaluationConfig:
    """批量评测配置"""
    
    def __init__(self):
        # 项目路径配置
        self.project_root = Path("/LD-FS/home/yunshuchen/DeepMicroSeg/microsam")
        self.cache_root = Path("/LD-FS/home/yunshuchen/DeepMicroSeg/microsam/Retrain/micro_sam_cache")
        
        # 模型配置 - 支持自定义checkpoint
        self.models = [
            {
                "name": "vit_t",
                "model_type": "vit_t_lm", 
                "checkpoint_path": None  # 使用默认模型
            },
            {
                "name": "vit_b", 
                "model_type": "vit_b_lm",
                "checkpoint_path": None  # 使用默认模型
            },
            {
                "name": "vit_l",
                "model_type": "vit_l_lm", 
                "checkpoint_path": None  # 使用默认模型
            },
            {
                "name": "Bfonly",  # 你的自定义模型
                "model_type": "vit_b_lm",  # 基础模型类型
                "checkpoint_path": "/LD-FS/home/yunshuchen/micro_sam/optimized_training_results_0522/checkpoints/checkpoints/microsam_optimized_patches/best.pt"
            },
                        {
                "name": "Flonly",  # 你的自定义模型
                "model_type": "vit_b_lm",  # 基础模型类型
                "checkpoint_path": "/LD-FS/home/yunshuchen/micro_sam/optimized_training_results/checkpoints/checkpoints/microsam_optimized_patches/best.pt"
            }
        ]
        
        # 硬件配置
        self.max_gpu_workers = 4
        self.process_timeout = 600  # 10分钟超时
        self.batch_size = None  # None表示处理所有图像，数字表示限制图像数量
        
        # 输出配置
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_base_dir = str(self.cache_root / f"batch_evaluation_results_{timestamp}")
        
        # **新增参数：控制是否启用可视化**
        self.enable_visualization = False  # False=仅计算指标, True=计算+可视化
        self.visual_size = 15  # 每个数据集随机选择的可视化样本数量
        self.save_overlays = True  # 是否保存叠加图像
        self.draw_grid = True  # 是否在叠加图像上绘制网格
        self.grid_size = 50  # 网格大小（像素）
        
        self.save_detailed_metrics = True
        
        # 缓存配置 - 统一到cache目录
        self.cache_dir = str(self.cache_root / "embeddings_cache")
        
        # 评测配置
        self.skip_existing = True  # 跳过已经评测过的数据集
        self.create_summary_report = True
        self.generate_unified_csv = True  # 生成统一的CSV报告

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
            # 转换为二值图
            gt_binary = (gt_mask > 0).astype(np.uint8)
            pred_binary = (pred_mask > 0).astype(np.uint8)
            
            # 查找轮廓
            gt_contours = measure.find_contours(gt_binary, 0.5)
            pred_contours = measure.find_contours(pred_binary, 0.5)
            
            if not gt_contours or not pred_contours:
                return float('inf')
            
            # 获取最大的轮廓
            gt_contour = max(gt_contours, key=len) if len(gt_contours) > 1 else gt_contours[0]
            pred_contour = max(pred_contours, key=len) if len(pred_contours) > 1 else pred_contours[0]
            
            if len(gt_contour) < 2 or len(pred_contour) < 2:
                return float('inf')
            
            # 计算距离矩阵
            distances_gt_to_pred = distance.cdist(gt_contour, pred_contour, 'euclidean')
            distances_pred_to_gt = distance.cdist(pred_contour, gt_contour, 'euclidean')
            
            # 计算单向距离
            min_distances_gt_to_pred = np.min(distances_gt_to_pred, axis=1)
            min_distances_pred_to_gt = np.min(distances_pred_to_gt, axis=1)
            
            # 计算95%分位数
            hd_gt_to_pred = np.percentile(min_distances_gt_to_pred, 95)
            hd_pred_to_gt = np.percentile(min_distances_pred_to_gt, 95)
            
            # HD95是两个方向的最大值
            hd95 = max(hd_gt_to_pred, hd_pred_to_gt)
            
            return float(hd95)
            
        except Exception as e:
            print(f"HD95 calculation error: {e}")
            return float('inf')
    
    @staticmethod
    def calculate_ap_at_threshold(gt_mask: np.ndarray, pred_mask: np.ndarray, iou_threshold: float) -> float:
        """计算指定IoU阈值下的平均精度"""
        try:
            # 提取实例
            gt_labels = np.unique(gt_mask)[1:]  # 排除背景
            pred_labels = np.unique(pred_mask)[1:]  # 排除背景
            
            if len(gt_labels) == 0:
                return 1.0 if len(pred_labels) == 0 else 0.0
            if len(pred_labels) == 0:
                return 0.0
            
            # 计算所有GT和预测实例之间的IoU
            iou_matrix = np.zeros((len(pred_labels), len(gt_labels)))
            
            for i, pred_id in enumerate(pred_labels):
                pred_region = (pred_mask == pred_id)
                for j, gt_id in enumerate(gt_labels):
                    gt_region = (gt_mask == gt_id)
                    
                    intersection = np.sum(pred_region & gt_region)
                    union = np.sum(pred_region | gt_region)
                    
                    if union > 0:
                        iou_matrix[i, j] = intersection / union
            
            # 按预测实例大小排序作为置信度
            pred_areas = [np.sum(pred_mask == pred_id) for pred_id in pred_labels]
            sorted_indices = np.argsort(pred_areas)[::-1]  # 从大到小排序
            
            # 匹配预测和GT实例
            gt_matched = np.zeros(len(gt_labels), dtype=bool)
            precision_points = []
            
            for rank, pred_idx in enumerate(sorted_indices):
                # 找到与当前预测实例IoU最高的GT实例
                best_gt_idx = np.argmax(iou_matrix[pred_idx])
                best_iou = iou_matrix[pred_idx, best_gt_idx]
                
                # 如果IoU超过阈值且GT实例未被匹配过，则为真正例
                if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                    gt_matched[best_gt_idx] = True
                
                # 计算当前的精度
                tp = np.sum(gt_matched)
                precision = tp / (rank + 1)
                precision_points.append(precision)
            
            # 计算平均精度
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

def setup_model_safe(model_config: Dict, gpu_id: int = None):
    """安全的模型设置 - 支持自定义checkpoint"""
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
        
        if checkpoint_path and Path(checkpoint_path).exists():
            # 使用自定义checkpoint
            print(f"Loading custom checkpoint from: {checkpoint_path}")
            
            # 检查checkpoint是否包含decoder（用于automatic instance segmentation）
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                has_decoder = any('decoder' in key for key in checkpoint.keys())
                print(f"  Decoder found in checkpoint: {has_decoder}")
                
                if has_decoder:
                    # 如果有decoder，使用InstanceSegmentationWithDecoder
                    from micro_sam.instance_segmentation import get_predictor_and_decoder
                    predictor, decoder = get_predictor_and_decoder(
                        model_type=model_type,
                        checkpoint_path=checkpoint_path,
                        device=device
                    )
                    segmenter = InstanceSegmentationWithDecoder(predictor, decoder)
                    print("  Using InstanceSegmentationWithDecoder with custom decoder")
                else:
                    # 如果没有decoder，使用普通的AMG
                    predictor = get_sam_model(
                        model_type=model_type,
                        checkpoint_path=checkpoint_path,
                        device=device
                    )
                    segmenter = None  # 将在后面创建AMG
                    print("  Using custom SAM model without decoder")
                    
            except Exception as e:
                print(f"  Error loading custom checkpoint: {e}")
                print("  Falling back to default model setup...")
                predictor, segmenter = get_predictor_and_segmenter(
                    model_type=model_type,
                    device=device,
                    amg=False,
                    is_tiled=False
                )
        else:
            # 使用默认模型
            if checkpoint_path:
                print(f"  Warning: Checkpoint path not found: {checkpoint_path}")
                print("  Using default model instead")
            
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

# 叠加图像保存函数（优化版本）
def save_overlay_image(image, mask, output_path, alpha=0.5, draw_grid=False, grid_size=50):
    """保存单张叠加图像
    
    Args:
        image: 原始图像数组
        mask: 掩码图像数组（实例标记）
        output_path: 输出文件路径
        alpha: 叠加透明度
        draw_grid: 是否绘制网格
        grid_size: 网格大小（像素）
    """
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
            # 重新规范化
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
            np.random.seed(42)  # 确保颜色一致性
            colors = [(np.random.random(), np.random.random(), np.random.random()) 
                      for _ in range(n_labels)]
            colors.insert(0, (0, 0, 0))  # 背景为黑色
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
        plt.close('all')  # 关闭所有图形
        
        # 强制垃圾回收
        gc.collect()
        
        return True
    except Exception as e:
        print(f"Error saving overlay to {output_path}: {e}")
        plt.close('all')  # 确保关闭图形
        return False

def process_dataset_worker(args):
    """Worker函数处理单个数据集 - 集成了图像叠加保存功能"""
    (dataset_info, model_config, output_dir, config) = args
    
    timeout_handler = TimeoutHandler(config.process_timeout)
    timeout_handler.start_timer()
    
    try:
        # 设置输出目录
        dataset_output_dir = Path(output_dir) / dataset_info['dataset_id']
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 根据配置决定是否创建可视化相关目录
        if config.enable_visualization:
            # 为可视化创建专门目录
            viz_dir = dataset_output_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
        
        if config.save_overlays:
            # 为叠加图像创建专门目录
            overlay_dir = dataset_output_dir / "overlays"
            overlay_dir.mkdir(exist_ok=True)
        
        # 检查是否已经处理过
        results_file = dataset_output_dir / "results.csv"
        if config.skip_existing and results_file.exists():
            print(f"Skipping {dataset_info['dataset_id']} - already processed")
            timeout_handler.stop_timer()
            return dataset_info['dataset_id'], "skipped"
        
        # 设置模型
        gpu_id = None
        predictor, segmenter = setup_model_safe(model_config, gpu_id)
        if predictor is None:
            timeout_handler.stop_timer()
            return dataset_info['dataset_id'], "model_setup_failed"
        
        # 获取图像和掩码的匹配对
        images_dir = Path(dataset_info['images_dir'])
        masks_dir = Path(dataset_info['masks_dir'])
        
        matched_pairs = find_matched_pairs(images_dir, masks_dir)
        print(f"Found {len(matched_pairs)} matched image-mask pairs for {dataset_info['dataset_id']}")
        
        if not matched_pairs:
            print(f"No matched pairs found for {dataset_info['dataset_id']}")
            timeout_handler.stop_timer()
            return dataset_info['dataset_id'], "no_matched_pairs"
        
        # 限制处理的图像数量
        if config.batch_size is None:
            # 处理所有匹配的图像对
            processing_pairs = matched_pairs
        else:
            # 限制处理数量
            max_pairs = min(config.batch_size, len(matched_pairs))
            processing_pairs = matched_pairs[:max_pairs]
        
        # 用于存储评测结果和可视化选择的图像
        results = []
        visualization_candidates = []
        overlay_files = []  # 存储叠加图像路径
        total_processing_time = 0.0
        
        # 处理每个匹配的图像-掩码对
        for img_path, mask_path in tqdm(processing_pairs, desc=f"Processing {dataset_info['dataset_id']}"):
            try:
                start_time = time.time()
                timeout_handler.start_timer()
                
                img_stem = img_path.stem
                
                # 加载和处理图像
                image = io.imread(img_path)
                if len(image.shape) > 2:
                    # 保存颜色图像用于叠加，但为分割提取单通道
                    image_for_display = image.copy()
                    image_for_seg = image[:, :, 0]
                else:
                    image_for_display = image
                    image_for_seg = image
                
                # 预测 - 根据segmenter类型选择不同的预测方法
                if segmenter is not None:
                    # 使用提供的segmenter（可能是InstanceSegmentationWithDecoder）
                    if hasattr(segmenter, 'initialize') and hasattr(segmenter, 'generate'):
                        # 如果是InstanceSegmentationWithDecoder
                        segmenter.initialize(image_for_seg)
                        masks = segmenter.generate()
                        
                        # 转换masks为segmentation
                        from micro_sam.instance_segmentation import mask_data_to_segmentation
                        segmentation = mask_data_to_segmentation(
                            masks, with_background=True, min_object_size=0
                        )
                    else:
                        # 使用automatic_instance_segmentation
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
                    'image_id': img_stem,
                    'cell_type': dataset_info['cell_type'],
                    'date': dataset_info['date'],
                    'magnification': dataset_info['magnification'],
                    'model': model_config['name'],
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
                        'img_id': img_stem
                    })
                
                # 保存叠加图像（如果启用）
                if config.save_overlays:
                    # 1. 保存原始图像与真实掩码的叠加
                    gt_overlay_path = overlay_dir / f"gt_overlay_{img_stem}.png"
                    save_overlay_image(
                        image_for_display, gt_mask, gt_overlay_path, 
                        alpha=0.5, draw_grid=config.draw_grid, grid_size=config.grid_size
                    )
                    
                    # 2. 保存原始图像与预测掩码的叠加
                    pred_overlay_path = overlay_dir / f"pred_overlay_{img_stem}.png"
                    save_overlay_image(
                        image_for_display, segmentation, pred_overlay_path, 
                        alpha=0.5, draw_grid=config.draw_grid, grid_size=config.grid_size
                    )
                    
                    # 3. 记录叠加图像路径
                    overlay_files.append({
                        'image_id': img_stem,
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
                print(f"Error processing {img_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 保存结果
        if results:
            df = pd.DataFrame(results)
            df.to_csv(results_file, index=False)
            
            # 保存叠加图像列表（如果有）
            if overlay_files:
                overlay_df = pd.DataFrame(overlay_files)
                overlay_df.to_csv(dataset_output_dir / "overlay_files.csv", index=False)
                grid_info = " (with grid)" if config.draw_grid else ""
                print(f"Saved {len(overlay_files)} overlay image pairs{grid_info} to {overlay_dir}")
            
            # 计算平均指标
            numeric_cols = ['ap50', 'ap75', 'iou_score', 'dice_score', 'hd95', 
                           'gt_instances', 'pred_instances', 'processing_time']
            
            avg_metrics = {}
            for col in numeric_cols:
                if col in df.columns:
                    values = df[col].dropna()
                    # 特殊处理HD95的无穷值
                    if col == 'hd95':
                        finite_values = values[np.isfinite(values)]
                        avg_metrics[col] = float(finite_values.mean()) if len(finite_values) > 0 else float('inf')
                    else:
                        avg_metrics[col] = float(values.mean()) if len(values) > 0 else 0.0
            
            # 添加元数据到摘要
            avg_metrics.update({
                'dataset_id': dataset_info['dataset_id'],
                'cell_type': dataset_info['cell_type'],
                'date': dataset_info['date'],
                'magnification': dataset_info['magnification'],
                'model': model_config['name'],
                'model_type': model_config['model_type'],
                'checkpoint_path': model_config.get('checkpoint_path', 'default'),
                'processed_images': len(results),
                'total_available_pairs': len(matched_pairs),
                'success_rate': len(results) / len(matched_pairs) if matched_pairs else 0.0,
                'total_processing_time': total_processing_time,
                'average_processing_time_per_image': total_processing_time / len(results) if results else 0.0,
                'overlay_images_saved': len(overlay_files) * 2 if overlay_files else 0,  # 真实和预测的叠加图像
                'visualization_enabled': config.enable_visualization,
                'overlays_enabled': config.save_overlays
            })
            
            # 保存摘要
            summary_file = dataset_output_dir / "summary.json"
            with open(summary_file, 'w') as f:
                json.dump(avg_metrics, f, indent=2)
            
            # 创建随机样本可视化（仅当启用时）
            if config.enable_visualization and visualization_candidates:
                # 创建可视化器
                visualizer = ResultVisualizer(viz_dir)
                
                # 随机选择样本
                num_to_visualize = min(config.visual_size, len(visualization_candidates))
                random.seed(config.visualization_seed)
                np.random.seed(config.visualization_seed)
                random_indices = random.sample(range(len(visualization_candidates)), num_to_visualize)
                print(f"Selected indices: {sorted(random_indices)}")  # 显示选择的索引
                
                print(f"Creating {num_to_visualize} visualizations for {dataset_info['dataset_id']}...")
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
            
            print(f"Completed {dataset_info['dataset_id']}: processed {len(results)}/{len(matched_pairs)} pairs")
            print(f"Average metrics - AP50: {avg_metrics['ap50']:.3f}, AP75: {avg_metrics['ap75']:.3f}, "
                  f"IoU: {avg_metrics['iou_score']:.3f}, Dice: {avg_metrics['dice_score']:.3f}")
            
            if config.save_overlays:
                grid_info = " (with grid)" if config.draw_grid else ""
                print(f"Saved {len(overlay_files)} GT+Pred overlay image pairs{grid_info} to {overlay_dir}")
        else:
            print(f"No results for {dataset_info['dataset_id']}")
        
        timeout_handler.stop_timer()
        return dataset_info['dataset_id'], "completed"
        
    except Exception as e:
        timeout_handler.stop_timer()
        print(f"Error processing dataset {dataset_info['dataset_id']}: {e}")
        import traceback
        traceback.print_exc()
        return dataset_info['dataset_id'], f"error: {str(e)}"

class BatchEvaluator:
    """批量评测器"""
    
    def __init__(self, config: BatchEvaluationConfig):
        self.config = config
        self.dataset_manager = None
        self.results_summary = []
    
    def setup(self, base_data_dir: str):
        """设置数据集管理器"""
        self.dataset_manager = DatasetManager(base_data_dir)
        print(f"发现 {len(self.dataset_manager.datasets)} 个数据集")
        
        # 创建输出目录和缓存目录
        os.makedirs(self.config.output_base_dir, exist_ok=True)
        os.makedirs(self.config.cache_dir, exist_ok=True)
        os.environ["MICROSAM_CACHEDIR"] = self.config.cache_dir
    
    def run_evaluation(self, cell_types=None, dates=None, magnifications=None):
        """运行批量评测"""
        if self.dataset_manager is None:
            raise ValueError("Please call setup() first")
        
        # 过滤数据集
        datasets_to_process = self.dataset_manager.filter_datasets(
            cell_types=cell_types,
            dates=dates,
            magnifications=magnifications
        )
        
        print(f"将处理 {len(datasets_to_process)} 个数据集")
        
        # 创建参数列表
        args_list = []
        for dataset_info in datasets_to_process:
            for model_config in self.config.models:
                output_dir = Path(self.config.output_base_dir) / model_config['name']
                args_list.append((dataset_info, model_config, output_dir, self.config))
        
        # 处理数据集（使用线程池避免GPU竞争）
        results = []
        for args in args_list:
            result = process_dataset_worker(args)
            results.append(result)
            print(f"Completed: {result[0]} - {result[1]}")
        
        # 生成摘要报告
        if self.config.create_summary_report:
            self._create_comprehensive_summary_report(results)
    
    def _create_comprehensive_summary_report(self, results):
        """创建全面的摘要报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path(self.config.output_base_dir) / f"summary_report_{timestamp}"
        report_dir.mkdir(exist_ok=True)
        
        # 收集所有详细结果
        all_detailed_results = []
        all_summaries = []
        
        for model_config in self.config.models:
            model_name = model_config['name']
            model_dir = Path(self.config.output_base_dir) / model_name
            
            if not model_dir.exists():
                continue
                
            for dataset_dir in model_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                    
                # 收集详细结果
                results_file = dataset_dir / "results.csv"
                if results_file.exists():
                    df = pd.read_csv(results_file)
                    all_detailed_results.append(df)
                
                # 收集摘要
                summary_file = dataset_dir / "summary.json"
                if summary_file.exists():
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                        summary['model'] = model_name
                        summary['dataset_id'] = dataset_dir.name
                        all_summaries.append(summary)
        
        # 生成统一的详细结果CSV
        if all_detailed_results and self.config.generate_unified_csv:
            unified_df = pd.concat(all_detailed_results, ignore_index=True)
            unified_csv_path = report_dir / "unified_detailed_results.csv"
            unified_df.to_csv(unified_csv_path, index=False)
            print(f"统一详细结果已保存到: {unified_csv_path}")
        
        # 生成最终摘要CSV - 按要求的列顺序
        if all_summaries:
            summary_df = pd.DataFrame(all_summaries)
            
            # 按要求的列顺序重新排列
            column_order = [
                'model', 'model_type', 'checkpoint_path', 'dataset_id', 'cell_type', 'date', 'magnification',
                'ap50', 'ap75', 'iou_score', 'dice_score', 'hd95',
                'gt_instances', 'pred_instances', 'processing_time',
                'processed_images', 'total_processing_time'
            ]
            
            # 只选择存在的列
            available_columns = [col for col in column_order if col in summary_df.columns]
            summary_df_ordered = summary_df[available_columns]
            
            # 保存最终摘要
            final_summary_path = report_dir / "final_evaluation_summary.csv"
            summary_df_ordered.to_csv(final_summary_path, index=False)
            print(f"最终摘要已保存到: {final_summary_path}")
            
            # 生成模型间对比统计
            model_comparison = summary_df.groupby('model')[
                ['ap50', 'ap75', 'iou_score', 'dice_score', 'hd95', 'processing_time']
            ].agg(['mean', 'std']).round(4)
            model_comparison.to_csv(report_dir / "model_comparison_statistics.csv")
            
            # 创建可视化（仅当启用时）
            if self.config.enable_visualization:
                self._create_final_visualizations(summary_df, report_dir)
            
            print(f"完整的评测报告已保存到: {report_dir}")
            
            # 打印最终统计
            print("\n=== 最终评测统计 ===")
            for model in summary_df['model'].unique():
                model_data = summary_df[summary_df['model'] == model]
                is_custom = any('custom' in str(path).lower() or 'best.pt' in str(path) 
                              for path in model_data['checkpoint_path'].fillna(''))
                model_label = f"{model}{'*' if is_custom else ''}"
                print(f"\n{model_label}:")
                if is_custom:
                    custom_path = model_data['checkpoint_path'].iloc[0]
                    print(f"  Checkpoint: {custom_path}")
                print(f"  AP50: {model_data['ap50'].mean():.3f} ± {model_data['ap50'].std():.3f}")
                print(f"  AP75: {model_data['ap75'].mean():.3f} ± {model_data['ap75'].std():.3f}")
                print(f"  IoU:  {model_data['iou_score'].mean():.3f} ± {model_data['iou_score'].std():.3f}")
                print(f"  Dice: {model_data['dice_score'].mean():.3f} ± {model_data['dice_score'].std():.3f}")
                finite_hd95 = model_data['hd95'][np.isfinite(model_data['hd95'])]
                if len(finite_hd95) > 0:
                    print(f"  HD95: {finite_hd95.mean():.3f} ± {finite_hd95.std():.3f}")
                else:
                    print(f"  HD95: N/A (all infinite)")
                print(f"  处理时间: {model_data['processing_time'].mean():.3f}s/image")
            print("\n* = 自定义checkpoint")
    
    def _create_final_visualizations(self, summary_df, output_dir):
        """创建最终的可视化图表"""
        try:
            # 1. 模型性能对比图
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            metrics = ['ap50', 'ap75', 'iou_score', 'dice_score', 'hd95', 'processing_time']
            metric_labels = ['AP50', 'AP75', 'IoU Score', 'Dice Score', 'HD95', 'Processing Time (s)']
            
            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                
                if metric in summary_df.columns:
                    # 特殊处理HD95的无穷值
                    if metric == 'hd95':
                        plot_data = summary_df[summary_df[metric] != np.inf]
                        if len(plot_data) == 0:
                            ax.text(0.5, 0.5, 'All HD95 values are infinite', 
                                   ha='center', va='center', transform=ax.transAxes)
                            ax.set_title(label)
                            continue
                    else:
                        plot_data = summary_df
                    
                    # 按模型分组绘制箱线图
                    models = plot_data['model'].unique()
                    box_data = [plot_data[plot_data['model'] == model][metric].values for model in models]
                    
                    bp = ax.boxplot(box_data, labels=models, patch_artist=True)
                    
                    # 设置颜色
                    colors = ['lightblue', 'lightgreen', 'lightcoral', 'orange']
                    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                        patch.set_facecolor(color)
                    
                    ax.set_title(label)
                    ax.grid(True, alpha=0.3)
                    
                    # 添加均值标记
                    for j, data in enumerate(box_data):
                        if len(data) > 0:
                            mean_val = np.mean(data)
                            ax.text(j+1, mean_val, f'{mean_val:.3f}', 
                                   ha='center', va='bottom', fontweight='bold', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(output_dir / "model_performance_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print("可视化图表创建完成")
            
        except Exception as e:
            print(f"可视化错误: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数 - 优化版本，适配新项目结构"""
    print("="*60)
    print("优化的细胞分割批量评测系统 - 支持自定义checkpoint")
    print("包含AP50、AP75、IoU、Dice、HD95等完整指标")
    print("支持仅计算指标或计算+可视化模式")
    print("适配新项目路径结构")
    print("="*60)
    
    # 配置
    config = BatchEvaluationConfig()
    
    # 显示项目路径信息
    print(f"项目路径配置:")
    print(f"  项目根目录: {config.project_root}")
    print(f"  缓存根目录: {config.cache_root}")
    print(f"  嵌入缓存目录: {config.cache_dir}")
    print(f"  输出目录: {config.output_base_dir}")
    
    # 显示模型配置
    print("\n模型配置:")
    for i, model_config in enumerate(config.models):
        print(f"  {i+1}. {model_config['name']}")
        print(f"     类型: {model_config['model_type']}")
        checkpoint = model_config.get('checkpoint_path', 'Default')
        if checkpoint and checkpoint != 'Default':
            print(f"     Checkpoint: {checkpoint}")
            print(f"     文件存在: {Path(checkpoint).exists()}")
        else:
            print(f"     Checkpoint: 使用默认模型")
        print()
    
    # 设置批处理大小和运行模式
    config.batch_size = None
    config.enable_visualization = True
    config.visualization_seed = 42
    config.visual_size = 15
    config.save_overlays = True
    config.draw_grid = True
    config.grid_size = 50
    config.skip_existing = True
    config.generate_unified_csv = True
    
    print(f"运行配置:")
    print(f"  批处理大小: {'全部有mask的图像' if config.batch_size is None else config.batch_size}")
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
    evaluator = BatchEvaluator(config)
    
    # 设置数据目录 - 根据实际情况修改
    base_data_dir = "/LD-FS/home/yunshuchen/DeepMicroSeg/microsam/Retrain/micro_sam_cache/LD_patch_0520"
    evaluator.setup(base_data_dir)
    
    # 显示发现的数据集
    datasets_summary = evaluator.dataset_manager.get_datasets_summary()
    print("\n发现的数据集:")
    print(datasets_summary[['cell_type', 'date', 'magnification', 'num_images', 'num_masks']].to_string(index=False))
    print(f"\n总共: {len(datasets_summary)} 个数据集")
    
    # 计算总工作量
    total_tasks = len(datasets_summary) * len(config.models)
    print(f"总任务数: {total_tasks} (数据集 × 模型)")
    
    # 运行评测
    print("\n" + "="*60)
    print("开始批量评测...")
    print("注意: 只处理有对应mask的图像")
    if config.enable_visualization:
        print("模式: 计算指标 + 可视化")
    else:
        print("模式: 仅计算指标")
    print("="*60)
    
    start_time = time.time()
    
    # 运行评测
    evaluator.run_evaluation()
    
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("批量评测完成!")
    print("="*60)
    print(f"总耗时: {total_time/3600:.2f} 小时")
    print(f"结果保存在: {config.output_base_dir}")
    
    # 显示最终的统一结果文件
    summary_dirs = list(Path(config.output_base_dir).glob("summary_report_*"))
    if summary_dirs:
        latest_summary = max(summary_dirs, key=lambda x: x.name)
        
        # 最终摘要文件
        final_summary_file = latest_summary / "final_evaluation_summary.csv"
        if final_summary_file.exists():
            print(f"\n主要结果文件:")
            print(f"  最终摘要: {final_summary_file}")
            
            # 显示简要统计
            df = pd.read_csv(final_summary_file)
            print(f"  包含 {len(df)} 条记录")
            print(f"  覆盖模型: {df['model'].unique().tolist()}")
            print(f"  覆盖细胞类型: {df['cell_type'].unique().tolist()}")
            
            # 显示各模型的最佳性能
            print("\n模型性能排名 (按AP50):")
            model_performance = df.groupby('model')['ap50'].mean().sort_values(ascending=False)
            for i, (model, score) in enumerate(model_performance.items()):
                # 检查是否为自定义模型
                model_data = df[df['model'] == model]
                is_custom = any('custom' in str(path).lower() or 'best.pt' in str(path) 
                              for path in model_data['checkpoint_path'].fillna(''))
                model_label = f"{model}{'*' if is_custom else ''}"
                print(f"  {i+1}. {model_label}: {score:.3f}")
            
            if any('*' in line for line in [f"{model}{'*' if any('custom' in str(path).lower() or 'best.pt' in str(path) for path in df[df['model'] == model]['checkpoint_path'].fillna('')) else ''}" for model in df['model'].unique()]):
                print("  * = 自定义checkpoint")
        
        # 详细结果文件
        unified_csv = latest_summary / "unified_detailed_results.csv"
        if unified_csv.exists():
            print(f"  详细结果: {unified_csv}")
        
        # 可视化文件（仅当启用时）
        if config.enable_visualization:
            viz_files = [
                "model_performance_comparison.png",
            ]
            print(f"  可视化图表: {latest_summary}")
            for viz_file in viz_files:
                if (latest_summary / viz_file).exists():
                    print(f"    - {viz_file}")
    
    # 检查各数据集的可视化结果（使用修复版本）
    if config.enable_visualization:
        check_visualizations(config)
    
    # 检查叠加图像结果
    if config.save_overlays:
        check_overlay_results(config)
    
    print("\n评测系统执行完毕！")
    mode_info = "计算+可视化" if config.enable_visualization else "仅计算指标"
    overlay_info = f"，叠加图像{'（含网格）' if config.draw_grid else ''}" if config.save_overlays else ""
    print(f"模式: {mode_info}{overlay_info}，统一CSV报告已生成。")
    print("\n重要提示:")
    print("- 所有缓存和结果文件已保存到: /LD-FS/home/yunshuchen/micro_sam_cache")
    print("- 你的自定义模型已包含在评测中")
    print("- 请查看结果中标有 '*' 的模型，这些是使用自定义checkpoint的模型")
    print("- 如需修改自定义checkpoint路径，请编辑 BatchEvaluationConfig 中的模型配置")

if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main()