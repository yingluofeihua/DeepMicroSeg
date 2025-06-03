"""
LoRA实验完整评测系统 - 支持所有样本评测和可视化
- 评测所有测试集样本（移除10个样本限制）
- 添加可视化功能（randomseed 42）
- 支持计算+可视化模式
- 保存叠加图像
"""

import os
import time
import json
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
import gc
from scipy.spatial import distance
import random
from matplotlib.colors import ListedColormap
import math

# Import micro_sam modules
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation
from micro_sam.util import get_sam_model
from micro_sam.instance_segmentation import get_predictor_and_decoder, InstanceSegmentationWithDecoder
from micro_sam.instance_segmentation import AutomaticMaskGenerator, mask_data_to_segmentation

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
        """Create visualization with comparison layout"""
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
            
            # Add metrics text
            iou_val = metrics.get('iou_score', 0)
            dice_val = metrics.get('dice_score', 0)
            hd95_val = metrics.get('hd95', 0)
            ap50_val = metrics.get('ap50', 0)
            ap75_val = metrics.get('ap75', 0)
            
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
            
            output_path = self.output_dir / f"comparison_{img_id}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
            plt.close('all')
            gc.collect()
            
        except Exception as e:
            print(f"Visualization error for {img_id}: {e}")
            plt.close('all')

def create_random_colormap(n_labels):
    """为每个标签创建随机颜色图"""
    if n_labels <= 0:
        return ListedColormap([(0, 0, 0)])
    
    np.random.seed(42)
    colors = [(np.random.random(), np.random.random(), np.random.random()) for _ in range(n_labels)]
    colors.insert(0, (0, 0, 0))
    return ListedColormap(colors)

def save_overlay_image(image, mask, output_path, alpha=0.5, draw_grid=False, grid_size=50):
    """保存叠加图像"""
    try:
        if len(image.shape) == 2:
            img_rgb = np.stack([image] * 3, axis=2)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            img_rgb = np.stack([image[:,:,0]] * 3, axis=2)
        else:
            img_rgb = image.copy()
        
        if img_rgb.max() > 1:
            img_rgb = img_rgb / 255.0
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img_rgb)
        
        unique_labels = np.unique(mask)
        unique_labels = unique_labels[unique_labels > 0]
        n_labels = len(unique_labels)
        
        if n_labels > 0:
            np.random.seed(42)
            colors = [(np.random.random(), np.random.random(), np.random.random()) 
                      for _ in range(n_labels)]
            colors.insert(0, (0, 0, 0))
            cmap = ListedColormap(colors)
            
            plt.imshow(mask, cmap=cmap, alpha=alpha)
        
        plt.title(f"Overlay ({n_labels} instances)")
        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close('all')
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
    """完整的评测指标计算类"""
    
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
            if np.max(gt_mask) <= 1:
                gt_mask = measure.label(gt_mask > 0)
            if np.max(pred_mask) <= 1:
                pred_mask = measure.label(pred_mask > 0)
            
            gt_binary = (gt_mask > 0).astype(np.float32)
            pred_binary = (pred_mask > 0).astype(np.float32)
            
            intersection = np.sum(gt_binary * pred_binary)
            union = np.sum(gt_binary) + np.sum(pred_binary) - intersection
            
            iou_score = intersection / (union + 1e-6)
            dice_score = 2 * intersection / (np.sum(gt_binary) + np.sum(pred_binary) + 1e-6)
            
            hd95 = cls.calculate_hausdorff_distance_95(gt_mask, pred_mask)
            
            gt_instances = len(np.unique(gt_mask)) - 1
            pred_instances = len(np.unique(pred_mask)) - 1
            
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

class LoRAExperimentConfig:
    """LoRA实验配置 - 增强版"""
    
    def __init__(self):
        # 项目路径配置
        self.project_root = Path("/LD-FS/home/yunshuchen/DeepMicroSeg/microsam")
        self.cache_root = Path("/LD-FS/home/yunshuchen/DeepMicroSeg/microsam/Retrain/micro_sam_cache")
        
        # 测试集配置
        self.test_sets = {
            'split_0.18_0.02_0.80': {
                'name': 'split_0.18_0.02_0.80',
                'base_dir': '/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/lora_split',
                'cell_types': ['293T', 'MSC', 'RBD', 'VERO']
            }
            # 'split_0.27_0.03_0.70': {
            #     'name': 'split_0.27_0.03_0.70',
            #     'base_dir': '/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/lora_split',
            #     'cell_types': ['293T', 'MSC', 'RBD', 'VERO']
            # },
            # 'split_0.36_0.04_0.60': {
            #     'name': 'split_0.36_0.04_0.60',
            #     'base_dir': '/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/lora_split',
            #     'cell_types': ['293T', 'MSC', 'RBD', 'VERO']
            # }
        }
        
        # LoRA checkpoint配置
        self.checkpoints = [
            {
                "name": "lora_293t_vit_b_lm_r8",
                "model_type": "vit_b_lm",
                "checkpoint_path": "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/results/293T/checkpoints/cells_lora/best.pt",
                "lora_rank": 8,
                "attention_layers": [9, 10, 11],
                "cell_type_trained": "293T"
            },
            {
                "name": "lora_rbd_vit_b_lm_r8", 
                "model_type": "vit_b_lm",
                "checkpoint_path": "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/results/RBD/checkpoints/cells_lora/best.pt",
                "lora_rank": 8,
                "attention_layers": [9, 10, 11],
                "cell_type_trained": "RBD"
            },
            {
                "name": "lora_rbd_vit_b_lm_r8", 
                "model_type": "vit_b_lm",
                "checkpoint_path": "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/results/MSC/checkpoints/cells_lora/best.pt",
                "lora_rank": 8,
                "attention_layers": [9, 10, 11],
                "cell_type_trained": "MSC"
            }
        ]
        
        # 硬件配置
        self.max_gpu_workers = 1
        self.process_timeout = 600
        
        # **新增：可视化配置**
        self.enable_visualization = True  # 启用可视化
        self.visualization_seed = 42     # 随机种子
        self.visual_size = 15           # 每个组合的可视化样本数
        self.save_overlays = True       # 保存叠加图像
        self.draw_grid = False          # 网格线
        self.grid_size = 50            # 网格大小
        
        # **移除样本数量限制 - 评测所有样本**
        self.process_all_samples = True  # 处理所有样本
        
        # 输出配置
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_base_dir = str(self.cache_root / f"lora_full_evaluation_results_{timestamp}")
        
        self.save_detailed_metrics = True
        self.cache_dir = str(self.cache_root / "embeddings_cache")
        
        # 评测配置
        self.skip_existing = True
        self.create_summary_report = True
        self.generate_unified_csv = True

def setup_model_safe(model_config: Dict, gpu_id: int = None):
    """基于示例代码的LoRA加载方式 - 直接加载权重到SAM模型"""
    import torch
    import sys
    import types
    
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
            print(f"  Loading LoRA checkpoint from: {checkpoint_path}")
            
            # 创建虚拟的datasets_simple模块来解决pickle问题
            if 'datasets_simple' not in sys.modules:
                print("  Creating dummy datasets_simple module...")
                datasets_simple = types.ModuleType('datasets_simple')
                
                # 创建完整的PatchDataset类（基于你提供的代码）
                class DummyPatchDataset:
                    def __init__(self, img_dir=None, mask_dir=None, *args, **kwargs):
                        self.img_dir = img_dir
                        self.mask_dir = mask_dir
                        # 其他可能的参数
                        for key, value in kwargs.items():
                            setattr(self, key, value)
                    
                    def __len__(self):
                        return 0
                    
                    def __getitem__(self, idx):
                        return None, None
                    
                    def __getstate__(self):
                        return self.__dict__
                    
                    def __setstate__(self, state):
                        self.__dict__.update(state)
                
                # 还需要添加encode_instance_map函数（虽然不会被调用）
                def encode_instance_map(inst_map, sigma=3):
                    import numpy as np
                    import torch
                    H, W = inst_map.shape
                    center = np.zeros((H, W), np.float32)
                    offsetx = np.zeros((H, W), np.float32)
                    offsety = np.zeros((H, W), np.float32)
                    scale = np.zeros((H, W), np.float32)
                    label = np.stack([center, offsetx, offsety, scale], axis=0)
                    return torch.from_numpy(label).float()
                
                datasets_simple.PatchDataset = DummyPatchDataset
                datasets_simple.encode_instance_map = encode_instance_map
                sys.modules['datasets_simple'] = datasets_simple
                print("  ✓ Dummy datasets_simple module with PatchDataset created")
            
            try:
                # 方法：仿照示例代码的直接加载方式
                print("  Method: Direct LoRA loading (based on sample code)...")
                
                # 1. 加载基础模型（返回predictor和sam_model）
                predictor, sam_model = get_sam_model(
                    model_type=model_type, 
                    device=device, 
                    return_sam=True
                )
                print("  ✓ Base model loaded")
                
                # 2. 加载LoRA权重
                state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                print(f"  Checkpoint top-level keys: {list(state.keys())}")
                
                # 3. 提取PEFT状态（更新优先级，包含model_state）
                peft_state = state.get('model_state', state.get('model', state.get('sam', state.get('state_dict', state))))
                
                if peft_state is state.get('model_state'):
                    print("  Using 'model_state' key from checkpoint")
                elif peft_state is state.get('model'):
                    print("  Using 'model' key from checkpoint")
                elif peft_state is state.get('sam'):
                    print("  Using 'sam' key from checkpoint")
                elif peft_state is state.get('state_dict'):
                    print("  Using 'state_dict' key from checkpoint")
                else:
                    print("  Using entire checkpoint as model state")
                
                print(f"  PEFT state keys count: {len(peft_state.keys()) if isinstance(peft_state, dict) else 'Not a dict'}")
                
                # 4. 检查并修复权重键的前缀问题
                if isinstance(peft_state, dict):
                    # 检查是否所有键都有sam.前缀
                    sample_keys = list(peft_state.keys())[:5]
                    print(f"  Sample keys: {sample_keys}")
                    
                    if all(key.startswith('sam.') for key in peft_state.keys()):
                        print("  Detected 'sam.' prefix in all keys, removing prefix...")
                        # 移除sam.前缀
                        fixed_state = {}
                        for key, value in peft_state.items():
                            new_key = key[4:] if key.startswith('sam.') else key  # 移除'sam.'前缀
                            fixed_state[new_key] = value
                        peft_state = fixed_state
                        print(f"  Fixed state keys count: {len(peft_state)}")
                        print(f"  Sample fixed keys: {list(peft_state.keys())[:5]}")
                    
                    # 检查LoRA权重（更新检测逻辑）
                    lora_keys = [k for k in peft_state.keys() if any(lora_term in k.lower() for lora_term in ['lora', 'w_a_linear', 'w_b_linear', 'qkv_proj'])]
                    if lora_keys:
                        print(f"  Found {len(lora_keys)} LoRA parameters")
                        print(f"  Sample LoRA keys: {lora_keys[:3]}{'...' if len(lora_keys) > 3 else ''}")
                    else:
                        print("  Warning: No LoRA keys found in PEFT state")
                
                # 5. 直接加载到sam_model，忽略不匹配的键（按示例代码）
                missing_keys, unexpected_keys = sam_model.load_state_dict(peft_state, strict=False)
                print(f"  Load result - Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
                
                if len(missing_keys) > 0:
                    print(f"  Sample missing keys: {missing_keys[:3]}{'...' if len(missing_keys) > 3 else ''}")
                if len(unexpected_keys) > 0:
                    print(f"  Sample unexpected keys: {unexpected_keys[:3]}{'...' if len(unexpected_keys) > 3 else ''}")
                
                # 6. 设置模型为评估模式
                sam_model.eval()
                if device.startswith('cuda'):
                    sam_model.cuda()
                
                print("  ✓ LoRA weights loaded successfully using direct method!")
                
                # 7. 为分割任务准备segmenter - 使用AutomaticMaskGenerator
                segmenter = None  # 将在后面使用AMG
                
                return predictor, segmenter
                
            except Exception as lora_error:
                print(f"  ✗ LoRA loading failed: {lora_error}")
                import traceback
                traceback.print_exc()
                print("  Falling back to default model...")
                    
        # 使用默认模型
        print("  Using default model")
        predictor, segmenter = get_predictor_and_segmenter(
            model_type=model_type,
            device=device,
            amg=False,
            is_tiled=False
        )
        print("  ✓ Default model loaded")
        if checkpoint_path:
            print("  WARNING: This is NOT using your LoRA checkpoint!")
        return predictor, segmenter
        
    except Exception as e:
        print(f"Model setup failed for {model_config['name']}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def process_test_samples_worker(args):
    from micro_sam.instance_segmentation import AutomaticMaskGenerator, mask_data_to_segmentation
    from segment_anything import SamAutomaticMaskGenerator  # 可选，仅当你仍可能 fallback 使用
    """Worker函数处理测试样本 - 增强版含可视化"""
    (test_samples, model_config, output_dir, config, test_set_name) = args
    
    timeout_handler = TimeoutHandler(config.process_timeout)
    timeout_handler.start_timer()
    
    try:
        # 设置输出目录
        model_output_dir = Path(output_dir) / model_config['name'] / test_set_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 为可视化创建目录
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
        predictor, segmenter = setup_model_safe(model_config, None)
        if predictor is None:
            timeout_handler.stop_timer()
            return model_config['name'], test_set_name, "model_setup_failed"
        
        # **重要修改：处理所有测试样本**
        if config.process_all_samples:
            sample_subset = test_samples  # 处理所有样本
            print(f"Processing ALL {len(sample_subset)} test samples for {model_config['name']} on {test_set_name}")
        else:
            max_samples = min(10, len(test_samples))
            sample_subset = test_samples[:max_samples]
            print(f"Processing {len(sample_subset)} test samples for {model_config['name']} on {test_set_name}")
        
        results = []
        visualization_candidates = []
        overlay_files = []
        total_processing_time = 0.0
        
        for sample in tqdm(sample_subset, desc=f"Processing {model_config['name']} on {test_set_name}"):
            try:
                start_time = time.time()
                timeout_handler.start_timer()
                
                img_path = sample['image_path']
                mask_path = sample['mask_path']
                sample_id = sample['sample_id']
                
                if not Path(img_path).exists() or not Path(mask_path).exists():
                    print(f"Warning: Files not found for {sample_id}, skipping...")
                    continue
                
                # 加载和处理图像 - 增强版错误处理
                try:
                    image = io.imread(img_path)
                    print(f"  Debug: Image {sample_id} shape: {image.shape}, dtype: {image.dtype}")
                    
                    # 新增预处理封装
                    def prepare_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
                        if len(image.shape) == 3:
                            if image.shape[2] == 3:
                                image_for_display = image.copy()
                                image_for_seg = image[:, :, 0]
                            elif image.shape[2] == 1:
                                image_for_display = image[:, :, 0]
                                image_for_seg = image[:, :, 0]
                            else:
                                image_for_display = image[:, :, 0]
                                image_for_seg = image[:, :, 0]
                        elif len(image.shape) == 2:
                            image_for_display = image
                            image_for_seg = image
                        else:
                            raise ValueError(f"Unsupported image shape: {image.shape}")
                        
                        if image_for_seg.max() <= 1:
                            image_for_seg = (image_for_seg * 255).astype(np.uint8)
                        else:
                            image_for_seg = image_for_seg.astype(np.uint8)
                        
                        # 为 segment_anything 准备 RGB 格式图像
                        image_for_amg = np.stack([image_for_seg]*3, axis=-1)
                        return image_for_display, image_for_seg, image_for_amg
                    
                    image_for_display, image_for_seg, image_for_amg = prepare_image(image)
                    print(f"  Debug: Final image_for_seg shape: {image_for_seg.shape}, dtype: {image_for_seg.dtype}")
                except Exception as img_error:
                    print(f"Error loading image {img_path}: {img_error}")
                    continue
                
                # 预测 - 使用示例代码的方式
                if segmenter is not None:
                    # 如果有专门的segmenter，使用它
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
                    # 使用AutomaticMaskGenerator（仿照示例代码）
                    try:
                        # 需要获取sam_model，这里需要重新加载
                        if hasattr(predictor, 'model'):
                            sam_model = predictor.model
                        else:
                            # 重新获取sam_model
                            _, sam_model = get_sam_model(
                                model_type=model_config["model_type"], 
                                device=device, 
                                return_sam=True
                            )
                            
                            # 如果有checkpoint，再次应用LoRA权重
                            if checkpoint_path and Path(checkpoint_path).exists():
                                state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                                peft_state = state.get('model', state.get('sam', state.get('state_dict', state)))
                                sam_model.load_state_dict(peft_state, strict=False)
                                sam_model.eval()
                        
                        # 创建AMG（使用示例代码的参数）
                        amg = SamAutomaticMaskGenerator(
                            sam_model,
                            points_per_side=32,
                            pred_iou_thresh=0.86,
                            stability_score_thresh=0.92,
                            crop_n_layers=0
                        )
                        masks = amg.generate(image_for_amg)
                        # amg = AutomaticMaskGenerator(predictor)
                        # amg.initialize(image_for_seg)
                        # masks = amg.generate()

                        segmentation = mask_data_to_segmentation(
                            masks, with_background=True, min_object_size=0
)
                        
                        # 转换为segmentation格式
                        if masks:
                            # 创建实例分割图
                            height, width = image_for_seg.shape[:2]
                            segmentation = np.zeros((height, width), dtype=np.uint16)
                            
                            # 按面积排序，大的先画
                            sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
                            
                            for idx, mask_data in enumerate(sorted_masks, 1):
                                mask = mask_data['segmentation']
                                segmentation[mask] = idx
                        else:
                            segmentation = np.zeros_like(image_for_seg, dtype=np.uint16)
                            
                    except ImportError:
                        print("  Warning: segment_anything not available, using micro_sam AMG")
                        # 使用micro_sam的AMG作为备选
                        from micro_sam.instance_segmentation import AutomaticMaskGenerator, mask_data_to_segmentation
                        amg = AutomaticMaskGenerator(predictor)
                        amg.initialize(image_for_seg)
                        masks = amg.generate()
                        segmentation = mask_data_to_segmentation(
                            masks, with_background=True, min_object_size=0
                        )
                
                # 加载GT - 增强错误处理
                try:
                    gt_mask = io.imread(mask_path)
                    print(f"  Debug: GT mask {sample_id} shape: {gt_mask.shape}, dtype: {gt_mask.dtype}")
                    
                    if len(gt_mask.shape) > 2:
                        gt_mask = gt_mask[:, :, 0]
                    
                    # 确保GT mask是2D的
                    if len(gt_mask.shape) != 2:
                        print(f"Error: GT mask still not 2D: {gt_mask.shape}")
                        continue
                        
                    print(f"  Debug: Final GT mask shape: {gt_mask.shape}")
                    
                except Exception as gt_error:
                    print(f"Error loading GT mask {mask_path}: {gt_error}")
                    continue
                
                processing_time = time.time() - start_time
                total_processing_time += processing_time
                
                # 计算所有指标
                metrics = ComprehensiveMetrics.compute_all_metrics(gt_mask, segmentation)
                
                # 添加元数据
                metrics.update({
                    'sample_id': sample_id,
                    'cell_type': sample['cell_type'],
                    'model': model_config['name'],
                    'test_set': test_set_name,
                    'processing_time': processing_time,
                    'image_path': str(img_path),
                    'mask_path': str(mask_path)
                })
                
                results.append(metrics)
                
                # **存储可视化候选样本**
                if config.enable_visualization:
                    visualization_candidates.append({
                        'image': image_for_display,
                        'gt_mask': gt_mask,
                        'prediction': segmentation,
                        'metrics': metrics,
                        'sample_id': sample_id
                    })
                
                # **保存叠加图像**
                if config.save_overlays:
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
                continue
        
        # 保存结果
        if results:
            df = pd.DataFrame(results)
            df.to_csv(results_file, index=False)
            
            # **保存叠加图像列表**
            if overlay_files:
                overlay_df = pd.DataFrame(overlay_files)
                overlay_df.to_csv(model_output_dir / "overlay_files.csv", index=False)
                print(f"Saved {len(overlay_files)} overlay image pairs to {overlay_dir}")
            
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
                'total_available_samples': len(sample_subset),
                'success_rate': len(results) / len(sample_subset) if sample_subset else 0.0,
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
            
            # **创建随机样本可视化（使用固定随机种子42）**
            if config.enable_visualization and visualization_candidates:
                visualizer = ResultVisualizer(viz_dir)
                
                # 使用固定随机种子42选择样本
                num_to_visualize = min(config.visual_size, len(visualization_candidates))
                random.seed(config.visualization_seed)
                np.random.seed(config.visualization_seed)
                random_indices = random.sample(range(len(visualization_candidates)), num_to_visualize)
                
                print(f"Creating {num_to_visualize} visualizations for {model_config['name']} on {test_set_name}")
                print(f"Selected indices (seed {config.visualization_seed}): {sorted(random_indices)}")
                
                for idx in random_indices:
                    candidate = visualization_candidates[idx]
                    try:
                        visualizer.visualize_comparison(
                            candidate['image'],
                            candidate['gt_mask'],
                            candidate['prediction'],
                            candidate['metrics'],
                            candidate['sample_id']
                        )
                    except Exception as viz_err:
                        print(f"Visualization error for {candidate['sample_id']}: {viz_err}")
                
                print(f"Saved {num_to_visualize} visualizations to {viz_dir}")
            
            print(f"Completed {model_config['name']} on {test_set_name}: processed {len(results)}/{len(sample_subset)} samples")
            print(f"Average metrics - AP50: {avg_metrics['ap50']:.3f}, AP75: {avg_metrics['ap75']:.3f}, "
                  f"IoU: {avg_metrics['iou_score']:.3f}, Dice: {avg_metrics['dice_score']:.3f}")
            
            if config.save_overlays:
                print(f"Saved {len(overlay_files)} GT+Pred overlay image pairs to {overlay_dir}")
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

class LoRATestSetManager:
    """LoRA测试集管理器"""
    
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
            
            all_test_samples = []
            
            for cell_type in cell_types:
                json_pattern = f"split_{test_set_name.split('_')[1]}_{test_set_name.split('_')[2]}_{test_set_name.split('_')[3]}_{cell_type}_*.json"
                json_files = list(base_dir.glob(json_pattern))
                
                if not json_files:
                    print(f"Warning: No JSON files found for {test_set_name} {cell_type}")
                    continue
                
                for json_file in json_files:
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                        
                        test_samples = data.get('test_samples', [])
                        print(f"Loaded {len(test_samples)} test samples from {json_file.name}")
                        
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
    """LoRA批量评测器 - 增强版"""
    
    def __init__(self, config: LoRAExperimentConfig):
        self.config = config
        self.test_manager = None
        self.results_summary = []
    
    def setup(self):
        """设置测试集管理器"""
        self.test_manager = LoRATestSetManager(self.config)
        
        os.makedirs(self.config.output_base_dir, exist_ok=True)
        os.makedirs(self.config.cache_dir, exist_ok=True)
        os.environ["MICROSAM_CACHEDIR"] = self.config.cache_dir
        
        print(f"发现测试集:")
        summary_df = self.test_manager.get_summary()
        print(summary_df.to_string(index=False))
    
    def run_evaluation(self):
        """运行完整批量评测"""
        if self.test_manager is None:
            raise ValueError("Please call setup() first")
        
        all_test_sets = self.test_manager.get_all_test_sets()
        
        print(f"将进行完整交叉评测:")
        print(f"  Checkpoints: {len(self.config.checkpoints)}")
        print(f"  Test Sets: {len(all_test_sets)}")
        print(f"  Total Combinations: {len(self.config.checkpoints) * len(all_test_sets)}")
        print(f"  Processing Mode: {'ALL samples' if self.config.process_all_samples else 'Limited samples'}")
        print(f"  Visualization: {'Enabled' if self.config.enable_visualization else 'Disabled'}")
        if self.config.enable_visualization:
            print(f"  Visualization Seed: {self.config.visualization_seed}")
            print(f"  Visualization Samples: {self.config.visual_size}")
        print(f"  Overlay Images: {'Enabled' if self.config.save_overlays else 'Disabled'}")
        
        # 创建任务列表 - 每个checkpoint在每个测试集上都要评测
        args_list = []
        total_samples = 0
        
        for test_set_name, test_samples in all_test_sets.items():
            if not test_samples:
                print(f"Warning: No test samples for {test_set_name}, skipping...")
                continue
                
            total_samples += len(test_samples)
            
            for checkpoint_config in self.config.checkpoints:
                # **重要：不再限制样本数量**
                args_list.append((
                    test_samples,  # 使用所有测试样本
                    checkpoint_config, 
                    self.config.output_base_dir, 
                    self.config, 
                    test_set_name
                ))
        
        print(f"\n开始处理 {len(args_list)} 个评测任务...")
        print(f"总测试样本数: {total_samples}")
        if self.config.process_all_samples:
            print("注意: 将处理所有测试样本（完整评测）")
        
        # 逐个处理任务
        results = []
        for i, args in enumerate(args_list):
            model_name = args[1]['name']
            test_set_name = args[4]
            sample_count = len(args[0])
            
            print(f"\n[{i+1}/{len(args_list)}] Processing {model_name} on {test_set_name} ({sample_count} samples)")
            result = process_test_samples_worker(args)
            results.append(result)
            print(f"Completed: {result[0]} on {result[1]} - {result[2]}")
            
            # 每处理完一个任务后清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 生成摘要报告
        if self.config.create_summary_report:
            self._create_summary_report(results)
    
    def _create_summary_report(self, results):
        """创建增强的摘要报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path(self.config.output_base_dir) / f"summary_report_{timestamp}"
        report_dir.mkdir(exist_ok=True)
        
        # 收集所有摘要
        all_summaries = []
        all_detailed_results = []
        
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
            print(f"包含 {len(unified_df)} 条详细记录")
        
        # 生成交叉评测摘要表
        if all_summaries:
            summary_df = pd.DataFrame(all_summaries)
            
            column_order = [
                'checkpoint_name', 'model_type', 'test_set_name',
                'ap50', 'ap75', 'iou_score', 'dice_score', 'hd95',
                'processed_samples', 'total_processing_time',
                'overlay_images_saved', 'visualization_enabled'
            ]
            
            available_columns = [col for col in column_order if col in summary_df.columns]
            summary_df_ordered = summary_df[available_columns]
            
            # 保存交叉评测摘要
            cross_eval_summary_path = report_dir / "cross_evaluation_summary.csv"
            summary_df_ordered.to_csv(cross_eval_summary_path, index=False)
            print(f"交叉评测摘要已保存到: {cross_eval_summary_path}")
            
            # 创建交叉评测矩阵
            self._create_cross_evaluation_matrix(summary_df, report_dir)
            
            # 创建可视化统计报告
            if self.config.enable_visualization:
                self._create_visualization_report(report_dir)
            
            # 打印交叉评测统计
            self._print_summary(summary_df)
    
    def _create_cross_evaluation_matrix(self, summary_df: pd.DataFrame, output_dir: Path):
        """创建交叉评测矩阵"""
        metrics = ['ap50', 'ap75', 'iou_score', 'dice_score']
        
        for metric in metrics:
            pivot_table = summary_df.pivot(
                index='checkpoint_name', 
                columns='test_set_name', 
                values=metric
            )
            
            matrix_path = output_dir / f"cross_evaluation_matrix_{metric}.csv"
            pivot_table.to_csv(matrix_path)
            print(f"交叉评测矩阵 {metric.upper()} 已保存到: {matrix_path}")
    
    def _create_visualization_report(self, output_dir: Path):
        """创建可视化统计报告"""
        viz_report = {
            'visualization_config': {
                'enabled': self.config.enable_visualization,
                'random_seed': self.config.visualization_seed,
                'samples_per_combination': self.config.visual_size,
                'overlay_images_enabled': self.config.save_overlays
            },
            'total_combinations': len(self.config.checkpoints) * len(self.config.test_sets),
            'expected_visualizations': len(self.config.checkpoints) * len(self.config.test_sets) * self.config.visual_size,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        viz_report_path = output_dir / "visualization_report.json"
        with open(viz_report_path, 'w') as f:
            json.dump(viz_report, f, indent=2)
        
        print(f"可视化报告已保存到: {viz_report_path}")
    
    def _print_summary(self, summary_df: pd.DataFrame):
        """打印摘要统计"""
        print("\n" + "="*80)
        print("LoRA完整交叉评测结果摘要")
        print("="*80)
        
        # 显示处理的样本统计
        total_processed = summary_df['processed_samples'].sum()
        print(f"\n总体统计:")
        print(f"  总处理样本数: {total_processed}")
        print(f"  总组合数: {len(summary_df)}")
        
        if self.config.enable_visualization:
            expected_viz = len(summary_df) * self.config.visual_size
            print(f"  预期可视化数量: {expected_viz} (seed: {self.config.visualization_seed})")
        
        if self.config.save_overlays:
            total_overlays = summary_df['overlay_images_saved'].sum()
            print(f"  总叠加图像数: {total_overlays}")
        
        # 按checkpoint显示性能
        print("\n按Checkpoint性能排名 (平均AP50):")
        checkpoint_perf = summary_df.groupby('checkpoint_name')['ap50'].mean().sort_values(ascending=False)
        for i, (checkpoint, score) in enumerate(checkpoint_perf.items()):
            trained_cell = next(
                (c['cell_type_trained'] for c in self.config.checkpoints if c['name'] == checkpoint), 
                'Unknown'
            )
            print(f"  {i+1}. {checkpoint} (trained on {trained_cell}): {score:.3f}")
        
        # 显示详细交叉结果
        print("\n详细交叉评测结果:")
        for _, row in summary_df.iterrows():
            trained_cell = next(
                (c['cell_type_trained'] for c in self.config.checkpoints if c['name'] == row['checkpoint_name']), 
                'Unknown'
            )
            samples_info = f"({row['processed_samples']} samples)"
            print(f"  {row['checkpoint_name']} (trained on {trained_cell}) → {row['test_set_name']} {samples_info}: "
                  f"AP50={row['ap50']:.3f}, AP75={row['ap75']:.3f}, IoU={row['iou_score']:.3f}")

def main():
    """主函数 - LoRA实验完整评测（含可视化）"""
    print("="*80)
    print("LoRA实验完整评测系统")
    print("处理所有测试样本 + 可视化 (randomseed 42)")
    print("="*80)
    
    # 配置
    config = LoRAExperimentConfig()
    
    # 验证checkpoint路径
    print(f"\n验证Checkpoint文件:")
    for i, checkpoint in enumerate(config.checkpoints):
        path_exists = Path(checkpoint['checkpoint_path']).exists()
        status = "✓" if path_exists else "✗"
        print(f"  {status} {checkpoint['name']}: {checkpoint['checkpoint_path']}")
        if not path_exists:
            print(f"     Warning: File not found!")
    
    print(f"\n测试集配置:")
    for test_set_name, test_set_info in config.test_sets.items():
        print(f"  {test_set_name}:")
        print(f"    基础目录: {test_set_info['base_dir']}")
        print(f"    细胞类型: {test_set_info['cell_types']}")
    
    print(f"\n评测配置:")
    print(f"  处理模式: {'所有样本' if config.process_all_samples else '限制样本'}")
    print(f"  可视化: {config.enable_visualization}")
    if config.enable_visualization:
        print(f"    随机种子: {config.visualization_seed}")
        print(f"    每组合可视化数: {config.visual_size}")
    print(f"  叠加图像: {config.save_overlays}")
    print(f"  跳过已有结果: {config.skip_existing}")
    
    # 创建评测器
    evaluator = LoRABatchEvaluator(config)
    
    # 设置评测器
    evaluator.setup()
    
    # 计算总工作量
    total_checkpoints = len(config.checkpoints)
    total_test_sets = len(config.test_sets)
    total_combinations = total_checkpoints * total_test_sets
    
    print(f"\n完整交叉评测工作量:")
    print(f"  Checkpoints: {total_checkpoints}")
    print(f"  Test Sets: {total_test_sets}")
    print(f"  总组合数: {total_combinations}")
    
    if config.enable_visualization:
        expected_viz = total_combinations * config.visual_size
        print(f"  预期可视化总数: {expected_viz}")
    
    # 详细显示交叉评测矩阵
    print(f"\n交叉评测矩阵 ({total_checkpoints} × {total_test_sets}):")
    print("  Checkpoints (trained on):")
    for i, checkpoint in enumerate(config.checkpoints, 1):
        print(f"    {i}. {checkpoint['name']} (trained on {checkpoint['cell_type_trained']})")
    print("  Test Sets:")
    for i, test_set in enumerate(config.test_sets.keys(), 1):
        print(f"    {i}. {test_set}")
    print(f"  → 每个checkpoint将在所有{total_test_sets}个测试集上完整评测")
    
    # 运行评测
    print("\n" + "="*80)
    print("开始LoRA完整交叉评测...")
    print("="*80)
    
    start_time = time.time()
    evaluator.run_evaluation()
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("LoRA完整交叉评测完成!")
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
            
            df = pd.read_csv(cross_eval_file)
            total_samples = df['processed_samples'].sum()
            print(f"  包含 {len(df)} 条交叉评测记录")
            print(f"  总处理样本数: {total_samples}")
            print(f"  Checkpoints: {df['checkpoint_name'].unique().tolist()}")
            print(f"  Test Sets: {df['test_set_name'].unique().tolist()}")
        
        # 统一详细结果
        unified_file = latest_summary / "unified_detailed_results.csv"
        if unified_file.exists():
            print(f"  统一详细结果: {unified_file}")
            
        # 交叉评测矩阵文件
        matrix_files = list(latest_summary.glob("cross_evaluation_matrix_*.csv"))
        if matrix_files:
            print(f"  交叉评测矩阵:")
            for matrix_file in matrix_files:
                metric = matrix_file.stem.split('_')[-1]
                print(f"    - {metric.upper()}: {matrix_file}")
        
        # 可视化报告
        if config.enable_visualization:
            viz_report = latest_summary / "visualization_report.json"
            if viz_report.exists():
                print(f"  可视化报告: {viz_report}")
    
    print("\n重要发现:")
    print("- 这是LoRA模型的完整交叉域评测结果")
    print("- 处理了所有测试样本（无数量限制）")
    if config.enable_visualization:
        print(f"- 使用随机种子{config.visualization_seed}生成可视化")
        print(f"- 每个组合生成{config.visual_size}个可视化样本")
    print("- 可以分析域适应性：同域 vs 跨域性能")
    print("- 查看交叉评测矩阵了解各checkpoint在不同测试集上的表现")
    
    print("\n评测系统执行完毕！")
    print("完整的交叉评测报告已生成。")
    
    print("\n分析建议:")
    print("1. 查看 cross_evaluation_summary.csv 了解整体表现")
    print("2. 查看 unified_detailed_results.csv 获取所有样本的详细指标")
    print("3. 查看交叉评测矩阵分析checkpoint在不同数据集上的泛化能力")
    print("4. 比较同域和跨域性能，评估域适应性")
    if config.enable_visualization:
        print("5. 查看各组合的可视化结果，了解分割质量")
        print("6. 检查叠加图像，直观比较GT和预测结果")

if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main()