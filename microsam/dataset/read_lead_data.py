"""
完整的细胞分割批量评测系统
包含AP50、AP75、IoU、Dice、HD95等完整指标
支持多模型评测并生成统一的CSV报告
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

# Import micro_sam modules
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

# 设置matplotlib使用非交互式后端
import matplotlib
matplotlib.use('Agg')
plt.ioff()

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
        # 模型配置 - 包含三个模型
        self.models = ["vit_t_lm", "vit_b_lm", "vit_l_lm"]
        
        # 硬件配置
        self.max_gpu_workers = 4
        self.process_timeout = 600  # 10分钟超时
        self.batch_size = None  # None表示处理所有图像，数字表示限制图像数量
        
        # 输出配置
        self.output_base_dir = "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/LDCellData/batch_evaluation_results_onlycal"
        self.save_visualizations = True
        self.save_detailed_metrics = True
        
        # 缓存配置
        self.cache_dir = "/LD-FS/home/zhenhuachen/.cache/micro_sam"
        
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

def setup_model_safe(model_type: str, gpu_id: int = None):
    """安全的模型设置"""
    try:
        if gpu_id is not None and torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            device = f"cuda:{gpu_id}"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        predictor, segmenter = get_predictor_and_segmenter(
            model_type=model_type,
            device=device,
            amg=False,
            is_tiled=False
        )
        
        return predictor, segmenter
    except Exception as e:
        print(f"Model setup failed: {e}")
        return None, None

def process_dataset_worker(args):
    """Worker函数处理单个数据集"""
    (dataset_info, model_type, output_dir, config) = args
    
    timeout_handler = TimeoutHandler(config.process_timeout)
    timeout_handler.start_timer()
    
    try:
        # 设置输出目录
        dataset_output_dir = Path(output_dir) / dataset_info['dataset_id']
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查是否已经处理过
        results_file = dataset_output_dir / "results.csv"
        if config.skip_existing and results_file.exists():
            print(f"Skipping {dataset_info['dataset_id']} - already processed")
            timeout_handler.stop_timer()
            return dataset_info['dataset_id'], "skipped"
        
        # 设置模型
        gpu_id = None
        predictor, segmenter = setup_model_safe(model_type, gpu_id)
        if predictor is None:
            timeout_handler.stop_timer()
            return dataset_info['dataset_id'], "model_setup_failed"
        
        # 获取图像列表
        images_dir = Path(dataset_info['images_dir'])
        masks_dir = Path(dataset_info['masks_dir'])
        
        image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
        
        # 限制处理的图像数量 - 设置为None表示处理所有图像
        if config.batch_size is None:
            # 处理所有图像
            max_images = len(image_files)
            image_files = image_files
        else:
            # 限制处理数量
            max_images = min(config.batch_size, len(image_files))
            image_files = image_files[:max_images]
        
        results = []
        total_processing_time = 0.0
        
        # 处理每张图像
        for img_path in tqdm(image_files, desc=f"Processing {dataset_info['dataset_id']}"):
            try:
                start_time = time.time()
                timeout_handler.start_timer()
                
                # 构建掩码路径
                img_stem = img_path.stem
                # 尝试不同的掩码命名模式
                mask_patterns = [
                    f"{img_stem}_seg.png",
                    f"{img_stem}.png",
                    f"{img_stem}_mask.png",
                    f"{img_stem}_seg.tif",
                    f"{img_stem}.tif"
                ]
                
                mask_path = None
                for pattern in mask_patterns:
                    potential_mask = masks_dir / pattern
                    if potential_mask.exists():
                        mask_path = potential_mask
                        break
                
                if mask_path is None:
                    print(f"Warning: No mask found for {img_stem}")
                    continue
                
                # 加载和处理图像
                image = io.imread(img_path)
                if len(image.shape) > 2:
                    image = image[:, :, 0]
                
                # 预测
                segmentation = automatic_instance_segmentation(
                    predictor=predictor,
                    segmenter=segmenter,
                    input_path=image,
                    ndim=2
                )
                
                # 加载GT并计算指标
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
                    'model': model_type,
                    'processing_time': processing_time,
                    'image_path': str(img_path),
                    'mask_path': str(mask_path)
                })
                
                results.append(metrics)
                
                # 定期清理内存
                if len(results) % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # 保存结果
        if results:
            df = pd.DataFrame(results)
            df.to_csv(results_file, index=False)
            
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
                'model': model_type,
                'processed_images': len(results),
                'total_available_images': len(image_files),
                'success_rate': len(results) / len(image_files) if image_files else 0.0,
                'total_processing_time': total_processing_time,
                'average_processing_time_per_image': total_processing_time / len(results) if results else 0.0
            })
            
            # 保存摘要
            summary_file = dataset_output_dir / "summary.json"
            with open(summary_file, 'w') as f:
                json.dump(avg_metrics, f, indent=2)
            
            print(f"Completed {dataset_info['dataset_id']}: processed {len(results)}/{len(image_files)} images")
            print(f"Average metrics - AP50: {avg_metrics['ap50']:.3f}, AP75: {avg_metrics['ap75']:.3f}, "
                  f"IoU: {avg_metrics['iou_score']:.3f}, Dice: {avg_metrics['dice_score']:.3f}")
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
        
        # 创建输出目录
        os.makedirs(self.config.output_base_dir, exist_ok=True)
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
            for model_type in self.config.models:
                output_dir = Path(self.config.output_base_dir) / model_type
                args_list.append((dataset_info, model_type, output_dir, self.config))
        
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
        
        for model_type in self.config.models:
            model_dir = Path(self.config.output_base_dir) / model_type
            
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
                        summary['model'] = model_type
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
                'model', 'dataset_id', 'cell_type', 'date', 'magnification',
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
            
            # 创建可视化
            self._create_final_visualizations(summary_df, report_dir)
            
            print(f"完整的评测报告已保存到: {report_dir}")
            
            # 打印最终统计
            print("\n=== 最终评测统计 ===")
            for model in summary_df['model'].unique():
                model_data = summary_df[summary_df['model'] == model]
                print(f"\n{model}:")
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
                    colors = ['lightblue', 'lightgreen', 'lightcoral']
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
            
            # 2. 按细胞类型的性能热图
            if 'cell_type' in summary_df.columns:
                fig, axes = plt.subplots(2, 3, figsize=(20, 14))
                axes = axes.flatten()
                
                key_metrics = ['ap50', 'ap75', 'iou_score', 'dice_score', 'processing_time']
                
                for i, metric in enumerate(key_metrics):
                    if i >= len(axes) or metric not in summary_df.columns:
                        continue
                    
                    ax = axes[i]
                    
                    try:
                        # 创建透视表
                        pivot_data = summary_df.pivot_table(
                            values=metric,
                            index='model',
                            columns='cell_type',
                            aggfunc='mean'
                        )
                        
                        # 特殊处理HD95
                        if metric == 'hd95':
                            # 将无穷值替换为NaN以便在热图中显示
                            pivot_data = pivot_data.replace([np.inf, -np.inf], np.nan)
                        
                        sns.heatmap(pivot_data, annot=True, fmt='.3f', ax=ax, 
                                   cmap='RdYlBu_r', cbar_kws={'label': 'Score'})
                        ax.set_title(f'{metric.upper()} by Model and Cell Type')
                        ax.set_xlabel('Cell Type')
                        ax.set_ylabel('Model')
                    except Exception as e:
                        print(f"Error creating heatmap for {metric}: {e}")
                        ax.text(0.5, 0.5, f'Error: {metric}', ha='center', va='center')
                        ax.set_title(f'{metric.upper()} (Error)')
                
                # 隐藏多余的子图
                for i in range(len(key_metrics), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                plt.savefig(output_dir / "performance_by_cell_type.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # 3. 性能指标相关性分析
            metrics_for_corr = ['ap50', 'ap75', 'iou_score', 'dice_score']
            available_metrics = [m for m in metrics_for_corr if m in summary_df.columns]
            
            if len(available_metrics) > 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                corr_matrix = summary_df[available_metrics].corr()
                
                sns.heatmap(corr_matrix, annot=True, fmt='.3f', ax=ax, 
                           cmap='coolwarm', center=0, square=True,
                           cbar_kws={'label': 'Correlation'})
                ax.set_title('Correlation Matrix of Performance Metrics')
                
                plt.tight_layout()
                plt.savefig(output_dir / "metrics_correlation.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            print("可视化图表创建完成")
            
        except Exception as e:
            print(f"可视化错误: {e}")
            import traceback
            traceback.print_exc()
                    
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
                    colors = ['lightblue', 'lightgreen', 'lightcoral']
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
            
            # 2. 按细胞类型的性能热图
            if 'cell_type' in summary_df.columns:
                fig, axes = plt.subplots(2, 3, figsize=(20, 14))
                axes = axes.flatten()
                
                key_metrics = ['ap50', 'ap75', 'iou_score', 'dice_score', 'processing_time']
                
                for i, metric in enumerate(key_metrics):
                    if i >= len(axes) or metric not in summary_df.columns:
                        continue
                    
                    ax = axes[i]
                    
                    try:
                        # 创建透视表
                        pivot_data = summary_df.pivot_table(
                            values=metric,
                            index='model',
                            columns='cell_type',
                            aggfunc='mean'
                        )
                        
                        # 特殊处理HD95
                        if metric == 'hd95':
                            # 将无穷值替换为NaN以便在热图中显示
                            pivot_data = pivot_data.replace([np.inf, -np.inf], np.nan)
                        
                        sns.heatmap(pivot_data, annot=True, fmt='.3f', ax=ax, 
                                   cmap='RdYlBu_r', cbar_kws={'label': 'Score'})
                        ax.set_title(f'{metric.upper()} by Model and Cell Type')
                        ax.set_xlabel('Cell Type')
                        ax.set_ylabel('Model')
                    except Exception as e:
                        print(f"Error creating heatmap for {metric}: {e}")
                        ax.text(0.5, 0.5, f'Error: {metric}', ha='center', va='center')
                        ax.set_title(f'{metric.upper()} (Error)')
                
                # 隐藏多余的子图
                for i in range(len(key_metrics), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                plt.savefig(output_dir / "performance_by_cell_type.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # 3. 性能指标相关性分析
            metrics_for_corr = ['ap50', 'ap75', 'iou_score', 'dice_score']
            available_metrics = [m for m in metrics_for_corr if m in summary_df.columns]
            
            if len(available_metrics) > 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                corr_matrix = summary_df[available_metrics].corr()
                
                sns.heatmap(corr_matrix, annot=True, fmt='.3f', ax=ax, 
                           cmap='coolwarm', center=0, square=True,
                           cbar_kws={'label': 'Correlation'})
                ax.set_title('Correlation Matrix of Performance Metrics')
                
                plt.tight_layout()
                plt.savefig(output_dir / "metrics_correlation.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            print("可视化图表创建完成")
            
        except Exception as e:
            print(f"可视化错误: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    print("="*60)
    print("细胞分割批量评测系统")
    print("包含AP50、AP75、IoU、Dice、HD95等完整指标")
    print("="*60)
    
    # 配置
    config = BatchEvaluationConfig()
    
    # 确保包含所有三个模型
    config.models = ["vit_t_lm", "vit_b_lm", "vit_l_lm"]
    
    # 设置批处理大小 - None表示处理所有图像，数字表示限制图像数量
    config.batch_size = None  # 处理所有图像
    # config.batch_size = 100  # 或者设置具体数量进行测试
    
    config.skip_existing = True  # 跳过已处理的数据集
    config.generate_unified_csv = True  # 生成统一的CSV
    
    print(f"配置信息:")
    print(f"  模型: {config.models}")
    print(f"  批处理大小: {'全部图像' if config.batch_size is None else config.batch_size}")
    print(f"  跳过已有结果: {config.skip_existing}")
    print(f"  输出目录: {config.output_base_dir}")
    
    # 创建评测器
    evaluator = BatchEvaluator(config)
    
    # 设置数据目录
    base_data_dir = "/LD-FS/home/yunshuchen/micro_sam/patch_0520"
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
    print("="*60)
    
    start_time = time.time()
    
    # 可以选择性地处理特定数据集
    evaluator.run_evaluation(
        cell_types=['MSC','Vero'],        # 可选：只处理特定细胞类型
        # magnifications=['20X'],     # 可选：只处理特定放大倍数
        # dates=['20250427']          # 可选：只处理特定日期
    )
    
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
                print(f"  {i+1}. {model}: {score:.3f}")
        
        # 详细结果文件
        unified_csv = latest_summary / "unified_detailed_results.csv"
        if unified_csv.exists():
            print(f"  详细结果: {unified_csv}")
        
        # 可视化文件
        viz_files = [
            "model_performance_comparison.png",
            "performance_by_cell_type.png",
            "metrics_correlation.png"
        ]
        print(f"  可视化图表: {latest_summary}")
        for viz_file in viz_files:
            if (latest_summary / viz_file).exists():
                print(f"    - {viz_file}")
    
    print("\n评测系统执行完毕！")

if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main()