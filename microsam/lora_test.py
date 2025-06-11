#!/usr/bin/env python3
"""
基于SAM的细胞分割LoRA微调系统 - 命令行预测评估工具
"""

import argparse
import torch
import os
import warnings
import sys
from glob import glob
from typing import Union, Tuple, Optional

import numpy as np
import imageio.v3 as imageio
from matplotlib import pyplot as plt
from skimage.measure import label as connected_components
from skimage import measure

import time
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import distance
from skimage.measure import regionprops
from scipy.optimize import linear_sum_assignment
import math
import gc

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.util.util import get_random_colors

import micro_sam.training as sam_training
import micro_sam.util
from micro_sam.sample_data import fetch_tracking_example_data, fetch_tracking_segmentation_data
from micro_sam.automatic_segmentation import automatic_instance_segmentation

warnings.filterwarnings("ignore")

# Set matplotlib to non-interactive backend
import matplotlib
matplotlib.use('Agg')
plt.ioff()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="基于SAM的细胞分割LoRA预测评估工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数
    parser.add_argument("--best_checkpoint", type=str, required=True,
                       help="最佳检查点文件路径")
    parser.add_argument("--test_image_dir", type=str, required=True,
                       help="测试图像目录路径")
    parser.add_argument("--test_mask_dir", type=str, required=True,
                       help="测试掩码目录路径")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="结果保存目录路径")
    
    # 可选参数
    parser.add_argument("--model_type", type=str, default="vit_b_lm",
                       choices=["vit_t", "vit_b", "vit_l", "vit_h", "vit_b_lm", "vit_l_lm"],
                       help="SAM模型类型")
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="指定使用的GPU ID")
    
    # LoRA参数
    parser.add_argument("--rank", type=int, default=8,
                       help="LoRA rank参数")
    parser.add_argument("--attention_layers_to_update", type=int, nargs="+", 
                       default=[9, 10, 11],
                       help="需要更新的注意力层索引列表")
    
    # 预测参数
    parser.add_argument("--center_distance_threshold", type=float, default=0.7,
                       help="中心距离阈值")
    parser.add_argument("--boundary_distance_threshold", type=float, default=0.7,
                       help="边界距离阈值")
    parser.add_argument("--foreground_threshold", type=float, default=0.2,
                       help="前景阈值")
    parser.add_argument("--min_size", type=int, default=100,
                       help="最小实例大小")
    
    # 其他参数
    parser.add_argument("--limit_images", type=int, default=None,
                       help="限制处理的图像数量（用于测试）")
    parser.add_argument("--save_visualizations", action="store_true",
                       help="是否保存可视化图像")
    parser.add_argument("--dpi", type=int, default=300,
                       help="保存图像的DPI")
    
    return parser.parse_args()


def setup_device(gpu_id):
    """设置GPU设备"""
    if torch.cuda.is_available():
        if gpu_id >= torch.cuda.device_count():
            raise ValueError(f"GPU ID {gpu_id} 不可用，总共有 {torch.cuda.device_count()} 个GPU")
        
        torch.cuda.set_device(gpu_id)
        device = "cuda"
        print(f"使用GPU: cuda:{gpu_id} ({torch.cuda.get_device_name(gpu_id)})")
    else:
        device = "cpu"
        print("CUDA不可用，使用CPU推理")
    
    return device


def validate_paths(args):
    """验证输入路径是否存在"""
    paths_to_check = [
        ("best_checkpoint", args.best_checkpoint),
        ("test_image_dir", args.test_image_dir),
        ("test_mask_dir", args.test_mask_dir),
    ]
    
    for name, path in paths_to_check:
        if not os.path.exists(path):
            raise ValueError(f"路径不存在: {name} = {path}")
    
    # 创建结果目录
    os.makedirs(args.results_dir, exist_ok=True)


def load_test_paths(test_image_dir, test_mask_dir):
    """加载测试数据路径"""
    image_paths = sorted(glob(os.path.join(test_image_dir, "*")))
    mask_paths = sorted(glob(os.path.join(test_mask_dir, "*")))
    
    if len(image_paths) == 0:
        raise ValueError(f"在 {test_image_dir} 中未找到图像文件")
    if len(mask_paths) == 0:
        raise ValueError(f"在 {test_mask_dir} 中未找到掩码文件")
    if len(image_paths) != len(mask_paths):
        raise ValueError(f"图像数量 ({len(image_paths)}) 与掩码数量 ({len(mask_paths)}) 不匹配")
    
    return image_paths, mask_paths


class ComprehensiveMetrics:
    """Complete evaluation metrics calculator including AP50, AP75, IoU, Dice, HD95"""
    
    @staticmethod
    def calculate_hausdorff_distance_95(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
        """Calculate HD95 metric using contour-based approach"""
        try:
            # Convert to binary images
            gt_binary = (gt_mask > 0).astype(np.uint8)
            pred_binary = (pred_mask > 0).astype(np.uint8)
            
            # Find contours
            gt_contours = measure.find_contours(gt_binary, 0.5)
            pred_contours = measure.find_contours(pred_binary, 0.5)
            
            if not gt_contours or not pred_contours:
                return float('inf')
            
            # Get the largest contour
            gt_contour = max(gt_contours, key=len) if len(gt_contours) > 1 else gt_contours[0]
            pred_contour = max(pred_contours, key=len) if len(pred_contours) > 1 else pred_contours[0]
            
            if len(gt_contour) < 2 or len(pred_contour) < 2:
                return float('inf')
            
            # Calculate distance matrices
            distances_gt_to_pred = distance.cdist(gt_contour, pred_contour, 'euclidean')
            distances_pred_to_gt = distance.cdist(pred_contour, gt_contour, 'euclidean')
            
            # Calculate one-way distances
            min_distances_gt_to_pred = np.min(distances_gt_to_pred, axis=1)
            min_distances_pred_to_gt = np.min(distances_pred_to_gt, axis=1)
            
            # Calculate 95th percentile
            hd_gt_to_pred = np.percentile(min_distances_gt_to_pred, 95)
            hd_pred_to_gt = np.percentile(min_distances_pred_to_gt, 95)
            
            # HD95 is the maximum of both directions
            hd95 = max(hd_gt_to_pred, hd_pred_to_gt)
            
            return float(hd95)
            
        except Exception as e:
            print(f"HD95 calculation error: {e}")
            return float('inf')
    
    @staticmethod
    def calculate_ap_at_threshold(gt_mask: np.ndarray, pred_mask: np.ndarray, iou_threshold: float) -> float:
        """Calculate Average Precision at specified IoU threshold"""
        try:
            # Extract instances
            gt_labels = np.unique(gt_mask)[1:]  # Exclude background
            pred_labels = np.unique(pred_mask)[1:]  # Exclude background
            
            if len(gt_labels) == 0:
                return 1.0 if len(pred_labels) == 0 else 0.0
            if len(pred_labels) == 0:
                return 0.0
            
            # Calculate IoU matrix between all GT and predicted instances
            iou_matrix = np.zeros((len(pred_labels), len(gt_labels)))
            
            for i, pred_id in enumerate(pred_labels):
                pred_region = (pred_mask == pred_id)
                for j, gt_id in enumerate(gt_labels):
                    gt_region = (gt_mask == gt_id)
                    
                    intersection = np.sum(pred_region & gt_region)
                    union = np.sum(pred_region | gt_region)
                    
                    if union > 0:
                        iou_matrix[i, j] = intersection / union
            
            # Sort predicted instances by area as confidence score
            pred_areas = [np.sum(pred_mask == pred_id) for pred_id in pred_labels]
            sorted_indices = np.argsort(pred_areas)[::-1]  # Sort from large to small
            
            # Match predictions and GT instances
            gt_matched = np.zeros(len(gt_labels), dtype=bool)
            precision_points = []
            
            for rank, pred_idx in enumerate(sorted_indices):
                # Find GT instance with highest IoU for current prediction
                best_gt_idx = np.argmax(iou_matrix[pred_idx])
                best_iou = iou_matrix[pred_idx, best_gt_idx]
                
                # If IoU exceeds threshold and GT instance hasn't been matched, it's a true positive
                if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                    gt_matched[best_gt_idx] = True
                
                # Calculate current precision
                tp = np.sum(gt_matched)
                precision = tp / (rank + 1)
                precision_points.append(precision)
            
            # Calculate average precision
            ap = np.mean(precision_points) if precision_points else 0.0
            return float(ap)
            
        except Exception as e:
            print(f"AP calculation error: {e}")
            return 0.0
    
    @classmethod
    def compute_all_metrics(cls, gt_mask: np.ndarray, pred_mask: np.ndarray) -> dict:
        """Calculate all evaluation metrics"""
        try:
            # Ensure label maps
            if np.max(gt_mask) <= 1:
                gt_mask = measure.label(gt_mask > 0)
            if np.max(pred_mask) <= 1:
                pred_mask = measure.label(pred_mask > 0)
            
            # Binary masks
            gt_binary = (gt_mask > 0).astype(np.float32)
            pred_binary = (pred_mask > 0).astype(np.float32)
            
            # Basic pixel-level metrics
            intersection = np.sum(gt_binary * pred_binary)
            union = np.sum(gt_binary) + np.sum(pred_binary) - intersection
            
            # IoU and Dice
            iou_score = intersection / (union + 1e-6)
            dice_score = 2 * intersection / (np.sum(gt_binary) + np.sum(pred_binary) + 1e-6)
            
            # HD95
            hd95 = cls.calculate_hausdorff_distance_95(gt_mask, pred_mask)
            
            # Instance counts
            gt_instances = len(np.unique(gt_mask)) - 1
            pred_instances = len(np.unique(pred_mask)) - 1
            
            # Calculate AP50 and AP75
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


class InstanceSegmentationMetrics:
    """Instance segmentation evaluation metrics calculator"""
    
    def __init__(self):
        self.metrics_results = []
    
    def evaluate_image(self, pred_instances, gt_instances, image_name, processing_time=0.0):
        """Evaluate all metrics for a single image"""
        start_time = time.time()
        
        # Calculate comprehensive metrics using the new method
        metrics = ComprehensiveMetrics.compute_all_metrics(gt_instances, pred_instances)
        
        # Add metadata
        metrics.update({
            'image_name': image_name,
            'processing_time': processing_time
        })
        
        self.metrics_results.append(metrics)
        
        eval_time = time.time() - start_time
        print(f"  Metrics calculated (time: {eval_time:.3f}s)")
        print(f"    AP50: {metrics['ap50']:.3f}, AP75: {metrics['ap75']:.3f}")
        print(f"    IoU: {metrics['iou_score']:.3f}, Dice: {metrics['dice_score']:.3f}")
        print(f"    HD95: {metrics['hd95']:.3f}, GT/Pred: {metrics['gt_instances']}/{metrics['pred_instances']}")
        
        return metrics
    
    def get_average_metrics(self):
        """Calculate average metrics"""
        if not self.metrics_results:
            return {}
        
        df = pd.DataFrame(self.metrics_results)
        
        # Handle infinite HD95 values
        finite_hd95 = df[df['hd95'] != float('inf')]['hd95']
        
        avg_metrics = {
            'Average_AP50': df['ap50'].mean(),
            'Average_AP75': df['ap75'].mean(),
            'Average_IoU_Score': df['iou_score'].mean(),
            'Average_Dice_Score': df['dice_score'].mean(),
            'Average_HD95': finite_hd95.mean() if len(finite_hd95) > 0 else float('inf'),
            'Average_GT_Instances': df['gt_instances'].mean(),
            'Average_Pred_Instances': df['pred_instances'].mean(),
            'Average_Processing_Time': df['processing_time'].mean(),
            'Total_Images': len(df),
            'Valid_HD95_Count': len(finite_hd95)
        }
        
        return avg_metrics
    
    def save_results(self, save_path):
        """Save detailed results to CSV"""
        if not self.metrics_results:
            print("No results to save")
            return
        
        df = pd.DataFrame(self.metrics_results)
        df.to_csv(save_path, index=False)
        print(f"Detailed results saved to: {save_path}")
        
        # Also save average metrics
        avg_metrics = self.get_average_metrics()
        avg_path = save_path.replace('.csv', '_averages.csv')
        
        avg_df = pd.DataFrame([avg_metrics])
        avg_df.to_csv(avg_path, index=False)
        print(f"Average metrics saved to: {avg_path}")
        
        return df, avg_df


def safe_format(val):
    """Safely format numerical values for display"""
    if val is None:
        return "N/A"
    if isinstance(val, (int, float)):
        if math.isnan(val) or math.isinf(val):
            return "N/A"
        return f"{val:.3f}"
    return "N/A"


def load_lora_model_for_prediction(checkpoint_path, model_type, device, rank, attention_layers):
    """
    Correctly load LoRA fine-tuned model for prediction
    """
    # Use the same method as during training to create the model
    peft_kwargs = {
        "rank": rank,
        "attention_layers_to_update": attention_layers
    }
    
    # Create trainable model with LoRA
    model = sam_training.get_trainable_sam_model(
        model_type=model_type,
        device=device,
        peft_kwargs=peft_kwargs
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try different key names
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    
    model.eval()
    
    return model, checkpoint


def predict_with_lora_model(image, checkpoint_path, model_type, device, args):
    """
    Predict using LoRA model and measure time
    """
    
    # Load model
    model, checkpoint = load_lora_model_for_prediction(
        checkpoint_path, model_type, device, 
        args.rank, args.attention_layers_to_update
    )
    
    # Manually create predictor and segmenter, as we need to support LoRA
    from segment_anything.predictor import SamPredictor
    from micro_sam.instance_segmentation import get_amg, get_decoder
    
    predictor = SamPredictor(model.sam)  # Use model.sam
    
    # Check if decoder exists
    decoder = None
    if isinstance(checkpoint, dict) and "decoder_state" in checkpoint:
        print("🎯 Decoder state found, using InstanceSegmentationWithDecoder")
        try:
            decoder = get_decoder(
                image_encoder=predictor.model.image_encoder, 
                decoder_state=checkpoint["decoder_state"], 
                device=device
            )
        except Exception as e:
            print(f"⚠️ Cannot load decoder: {e}, using AutomaticMaskGenerator")
            decoder = None
    else:
        print("🎯 No decoder state found, using AutomaticMaskGenerator")
    
    segmenter = get_amg(predictor=predictor, is_tiled=False, decoder=decoder)
    
    # Execute prediction with timing
    start_time = time.time()
    
    prediction = automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmenter,
        input_path=image,
        ndim=2,
        center_distance_threshold=args.center_distance_threshold,
        boundary_distance_threshold=args.boundary_distance_threshold,
        foreground_threshold=args.foreground_threshold,
        min_size=args.min_size,
    )
    
    processing_time = time.time() - start_time
    
    return prediction, processing_time


def create_visualization(image, gt_mask, prediction, metrics, image_name, save_path, dpi):
    """创建可视化图像"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Set journal-style font parameters
    title_fontsize = 11
    caption_fontsize = 9
    
    # Original image
    axes[0,0].imshow(image, cmap="gray")
    axes[0,0].axis("off")
    axes[0,0].set_title("(a) Original Image", fontsize=title_fontsize, fontfamily='serif', pad=10)
    
    # Ground Truth Mask
    axes[0,1].imshow(gt_mask, cmap="tab20", interpolation="nearest")
    axes[0,1].axis("off")
    axes[0,1].set_title(f"(b) Ground Truth ({metrics['gt_instances']} instances)", 
                       fontsize=title_fontsize, fontfamily='serif', pad=10)
    
    # Prediction results
    axes[1,0].imshow(prediction, cmap="tab20", interpolation="nearest")
    axes[1,0].axis("off")
    axes[1,0].set_title(f"(c) Prediction ({metrics['pred_instances']} instances)", 
                       fontsize=title_fontsize, fontfamily='serif', pad=10)
    
    # Prediction overlay on original image (with transparency)
    axes[1,1].imshow(image, cmap="gray")
    if np.any(prediction > 0):  # Only overlay if there are predictions
        axes[1,1].imshow(prediction, cmap="tab20", alpha=0.5, interpolation="nearest")
    axes[1,1].axis("off")
    axes[1,1].set_title("(d) Prediction Overlay", fontsize=title_fontsize, fontfamily='serif', pad=10)
    
    # Adjust layout with proper spacing
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.02, right=0.98, hspace=0.08, wspace=-0.4)
    
    # Add main title with proper positioning
    fig.suptitle(f"Cell Segmentation Results: {image_name}", 
                fontsize=12, fontfamily='serif', fontweight='normal')
    
    # Add performance metrics as figure caption with proper positioning
    metrics_caption = (f"AP50: {safe_format(metrics['ap50'])}, "
                      f"AP75: {safe_format(metrics['ap75'])}, "
                      f"IoU: {safe_format(metrics['iou_score'])}, "
                      f"Dice: {safe_format(metrics['dice_score'])}, "
                      f"HD95: {safe_format(metrics['hd95'])}")
    
    fig.text(0.5, 0.03, metrics_caption, ha='center', va='bottom', 
            fontsize=caption_fontsize, fontfamily='serif', style='italic')
    
    # Save result image
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close('all')


def print_config(args, device, num_images):
    """打印配置信息"""
    print("=" * 80)
    print("🔬 基于SAM的细胞分割LoRA预测评估系统")
    print("=" * 80)
    print("📂 数据配置:")
    print(f"  测试图像目录: {args.test_image_dir}")
    print(f"  测试掩码目录: {args.test_mask_dir}")
    print(f"  图像数量: {num_images}")
    print()
    print("🤖 模型配置:")
    print(f"  模型类型: {args.model_type}")
    print(f"  检查点路径: {args.best_checkpoint}")
    print(f"  LoRA Rank: {args.rank}")
    print(f"  LoRA 层: {args.attention_layers_to_update}")
    print()
    print("⚙️ 预测参数:")
    print(f"  中心距离阈值: {args.center_distance_threshold}")
    print(f"  边界距离阈值: {args.boundary_distance_threshold}")
    print(f"  前景阈值: {args.foreground_threshold}")
    print(f"  最小实例大小: {args.min_size}")
    print()
    print("💾 输出配置:")
    print(f"  结果目录: {args.results_dir}")
    print(f"  保存可视化: {args.save_visualizations}")
    print(f"  设备: {device}")
    print("=" * 80)


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 验证路径
    validate_paths(args)
    
    # 设置设备
    device = setup_device(args.gpu_id)
    
    # 加载测试数据路径
    print("📂 加载测试数据路径...")
    image_paths, mask_paths = load_test_paths(args.test_image_dir, args.test_mask_dir)
    
    # 限制图像数量（如果指定）
    if args.limit_images:
        image_paths = image_paths[:args.limit_images]
        mask_paths = mask_paths[:args.limit_images]
    
    # 打印配置
    print_config(args, device, len(image_paths))
    
    # 初始化评估器
    evaluator = InstanceSegmentationMetrics()
    
    print(f"\n🚀 开始评估 {len(image_paths)} 张图像...")
    print("="*80)
    
    # 处理每张图像
    for i, image_path in enumerate(image_paths):
        print(f"\nProcessing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        print("-" * 60)
        
        try:
            # 读取图像
            image = imageio.imread(image_path)
            print(f"Original image shape: {image.shape}")
            
            # 读取对应的真实掩码
            mask_path = mask_paths[i]
            gt_mask = imageio.imread(mask_path)
            print(f"GT mask shape: {gt_mask.shape}")
            
            # 确保图像是2D
            if len(image.shape) == 3:
                if image.shape[2] == 1:
                    image = image.squeeze(2)
                elif image.shape[2] == 3:
                    # 如果是彩色图像，转换为灰度
                    image = np.mean(image, axis=2).astype(image.dtype)
                    print("🔄 Converted color image to grayscale")
            
            # 确保掩码是2D
            if len(gt_mask.shape) == 3:
                if gt_mask.shape[2] == 1:
                    gt_mask = gt_mask.squeeze(2)
                elif gt_mask.shape[2] == 3:
                    # 如果是彩色掩码，取第一个通道或转换为灰度
                    gt_mask = gt_mask[:,:,0]
                    print("🔄 Converted color mask to grayscale")
            
            print(f"Processed image shape: {image.shape}")
            print(f"Processed mask shape: {gt_mask.shape}")
            
            # 使用增强预测功能
            print("🚀 Starting prediction...")
            prediction, processing_time = predict_with_lora_model(
                image=image,
                checkpoint_path=args.best_checkpoint,
                model_type=args.model_type,
                device=device,
                args=args
            )
            
            print(f"✅ Prediction completed (time: {processing_time:.3f}s)")
            
            # 计算评估指标
            print("📊 Calculating evaluation metrics...")
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            metrics = evaluator.evaluate_image(
                pred_instances=prediction,
                gt_instances=gt_mask,
                image_name=image_name,
                processing_time=processing_time
            )
            
            # 创建可视化（如果启用）
            if args.save_visualizations:
                output_path = os.path.join(args.results_dir, f"{image_name}_evaluation.png")
                create_visualization(image, gt_mask, prediction, metrics, image_name, output_path, args.dpi)
                print(f"💾 Visualization saved to: {output_path}")
            
            # 强制垃圾回收
            gc.collect()
            
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            import traceback
            traceback.print_exc()
            plt.close('all')  # 确保即使出错也关闭图像
            continue
    
    print("\n" + "="*80)
    print("🎉 评估完成!")
    print("="*80)
    
    # 计算并显示平均指标
    avg_metrics = evaluator.get_average_metrics()
    
    if avg_metrics:
        print("\n📊 Average Evaluation Metrics:")
        print("-" * 50)
        print(f"Average AP50:           {avg_metrics['Average_AP50']:.3f}")
        print(f"Average AP75:           {avg_metrics['Average_AP75']:.3f}")
        print(f"Average IoU Score:      {avg_metrics['Average_IoU_Score']:.3f}")
        print(f"Average Dice Score:     {avg_metrics['Average_Dice_Score']:.3f}")
        print(f"Average HD95:           {avg_metrics['Average_HD95']:.3f}")
        print(f"Average GT Instances:   {avg_metrics['Average_GT_Instances']:.1f}")
        print(f"Average Pred Instances: {avg_metrics['Average_Pred_Instances']:.1f}")
        print(f"Average Processing Time:{avg_metrics['Average_Processing_Time']:.3f}s")
        print(f"Total Images:           {avg_metrics['Total_Images']}")
        print(f"Valid HD95 Count:       {avg_metrics['Valid_HD95_Count']}")
        print("-" * 50)
    
    # 保存详细结果
    results_csv_path = os.path.join(args.results_dir, "detailed_evaluation_results.csv")
    detailed_df, avg_df = evaluator.save_results(results_csv_path)
    
    print(f"\n📁 所有结果已保存到: {args.results_dir}")
    print(f"   • 详细结果: detailed_evaluation_results.csv")
    print(f"   • 平均指标: detailed_evaluation_results_averages.csv")
    if args.save_visualizations:
        print(f"   • 可视化图像: {len(image_paths)} PNG 文件")
    
    print("\n✅ 预测和评估完成!")


if __name__ == "__main__":
    main()