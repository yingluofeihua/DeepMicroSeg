"""
评测指标计算模块
包含AP50、AP75、IoU、Dice、HD95等完整指标的计算
"""

import numpy as np
from typing import Dict, List
from scipy.spatial import distance
from skimage import measure
from dataclasses import dataclass


@dataclass
class MetricsResult:
    """评测指标结果"""
    ap50: float
    ap75: float
    iou_score: float
    dice_score: float
    hd95: float
    gt_instances: int
    pred_instances: int
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'ap50': self.ap50,
            'ap75': self.ap75,
            'iou_score': self.iou_score,
            'dice_score': self.dice_score,
            'hd95': self.hd95,
            'gt_instances': self.gt_instances,
            'pred_instances': self.pred_instances
        }


class InstanceMetrics:
    """实例级评测指标计算器"""
    
    @staticmethod
    def calculate_ap_at_threshold(gt_mask: np.ndarray, pred_mask: np.ndarray, 
                                iou_threshold: float) -> float:
        """计算指定IoU阈值下的平均精度"""
        try:
            # 提取实例标签
            gt_labels = np.unique(gt_mask)[1:]  # 排除背景(0)
            pred_labels = np.unique(pred_mask)[1:]  # 排除背景(0)
            
            # 边界情况处理
            if len(gt_labels) == 0:
                return 1.0 if len(pred_labels) == 0 else 0.0
            if len(pred_labels) == 0:
                return 0.0
            
            # 计算IoU矩阵
            iou_matrix = InstanceMetrics._compute_iou_matrix(gt_mask, pred_mask, 
                                                           gt_labels, pred_labels)
            
            # 按预测实例面积排序（作为置信度）
            pred_areas = [np.sum(pred_mask == pred_id) for pred_id in pred_labels]
            sorted_indices = np.argsort(pred_areas)[::-1]  # 从大到小
            
            # 执行匹配并计算精度
            gt_matched = np.zeros(len(gt_labels), dtype=bool)
            precision_points = []
            
            for rank, pred_idx in enumerate(sorted_indices):
                best_gt_idx = np.argmax(iou_matrix[pred_idx])
                best_iou = iou_matrix[pred_idx, best_gt_idx]
                
                # 匹配条件：IoU超过阈值且GT未被匹配
                if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                    gt_matched[best_gt_idx] = True
                
                # 计算当前精度
                tp = np.sum(gt_matched)
                precision = tp / (rank + 1)
                precision_points.append(precision)
            
            return float(np.mean(precision_points)) if precision_points else 0.0
            
        except Exception as e:
            print(f"AP计算错误: {e}")
            return 0.0
    
    @staticmethod
    def _compute_iou_matrix(gt_mask: np.ndarray, pred_mask: np.ndarray,
                          gt_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
        """计算IoU矩阵"""
        iou_matrix = np.zeros((len(pred_labels), len(gt_labels)))
        
        for i, pred_id in enumerate(pred_labels):
            pred_region = (pred_mask == pred_id)
            
            for j, gt_id in enumerate(gt_labels):
                gt_region = (gt_mask == gt_id)
                
                intersection = np.sum(pred_region & gt_region)
                union = np.sum(pred_region | gt_region)
                
                iou_matrix[i, j] = intersection / union if union > 0 else 0.0
        
        return iou_matrix


class PixelMetrics:
    """像素级评测指标计算器"""
    
    @staticmethod
    def calculate_iou(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
        """计算IoU（交并比）"""
        try:
            gt_binary = (gt_mask > 0).astype(np.float32)
            pred_binary = (pred_mask > 0).astype(np.float32)
            
            intersection = np.sum(gt_binary * pred_binary)
            union = np.sum(gt_binary) + np.sum(pred_binary) - intersection
            
            return float(intersection / (union + 1e-6))
        except Exception:
            return 0.0
    
    @staticmethod
    def calculate_dice(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
        """计算Dice系数"""
        try:
            gt_binary = (gt_mask > 0).astype(np.float32)
            pred_binary = (pred_mask > 0).astype(np.float32)
            
            intersection = np.sum(gt_binary * pred_binary)
            total = np.sum(gt_binary) + np.sum(pred_binary)
            
            return float(2 * intersection / (total + 1e-6))
        except Exception:
            return 0.0


class DistanceMetrics:
    """距离相关评测指标计算器"""
    
    @staticmethod
    def calculate_hausdorff_distance_95(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
        """计算95%豪斯多夫距离"""
        try:
            # 转换为二值图
            gt_binary = (gt_mask > 0).astype(np.uint8)
            pred_binary = (pred_mask > 0).astype(np.uint8)
            
            # 提取轮廓
            gt_contours = measure.find_contours(gt_binary, 0.5)
            pred_contours = measure.find_contours(pred_binary, 0.5)
            
            if not gt_contours or not pred_contours:
                return float('inf')
            
            # 选择最大轮廓
            gt_contour = max(gt_contours, key=len) if len(gt_contours) > 1 else gt_contours[0]
            pred_contour = max(pred_contours, key=len) if len(pred_contours) > 1 else pred_contours[0]
            
            if len(gt_contour) < 2 or len(pred_contour) < 2:
                return float('inf')
            
            # 计算双向距离
            hd_gt_to_pred = DistanceMetrics._directed_hausdorff_95(gt_contour, pred_contour)
            hd_pred_to_gt = DistanceMetrics._directed_hausdorff_95(pred_contour, gt_contour)
            
            return float(max(hd_gt_to_pred, hd_pred_to_gt))
            
        except Exception as e:
            print(f"HD95计算错误: {e}")
            return float('inf')
    
    @staticmethod
    def _directed_hausdorff_95(contour_a: np.ndarray, contour_b: np.ndarray) -> float:
        """计算单向95%豪斯多夫距离"""
        distances = distance.cdist(contour_a, contour_b, 'euclidean')
        min_distances = np.min(distances, axis=1)
        return np.percentile(min_distances, 95)


class ComprehensiveMetrics:
    """综合评测指标计算器"""
    
    def __init__(self, enable_hd95: bool = True):
        self.enable_hd95 = enable_hd95
        self.instance_metrics = InstanceMetrics()
        self.pixel_metrics = PixelMetrics()
        self.distance_metrics = DistanceMetrics()
    
    def compute_all_metrics(self, gt_mask: np.ndarray, pred_mask: np.ndarray) -> MetricsResult:
        """计算所有评测指标"""
        try:
            # 预处理：确保是标签图
            gt_mask = self._preprocess_mask(gt_mask)
            pred_mask = self._preprocess_mask(pred_mask)
            
            # 计算各类指标
            ap50 = self.instance_metrics.calculate_ap_at_threshold(gt_mask, pred_mask, 0.5)
            ap75 = self.instance_metrics.calculate_ap_at_threshold(gt_mask, pred_mask, 0.75)
            
            iou_score = self.pixel_metrics.calculate_iou(gt_mask, pred_mask)
            dice_score = self.pixel_metrics.calculate_dice(gt_mask, pred_mask)
            
            if self.enable_hd95:
                hd95 = self.distance_metrics.calculate_hausdorff_distance_95(gt_mask, pred_mask)
            else:
                hd95 = float('inf')
            
            # 统计实例数量
            gt_instances = len(np.unique(gt_mask)) - 1  # 排除背景
            pred_instances = len(np.unique(pred_mask)) - 1  # 排除背景
            
            return MetricsResult(
                ap50=ap50,
                ap75=ap75,
                iou_score=iou_score,
                dice_score=dice_score,
                hd95=hd95,
                gt_instances=gt_instances,
                pred_instances=pred_instances
            )
            
        except Exception as e:
            print(f"综合指标计算错误: {e}")
            return MetricsResult(
                ap50=0.0,
                ap75=0.0,
                iou_score=0.0,
                dice_score=0.0,
                hd95=float('inf'),
                gt_instances=0,
                pred_instances=0
            )
    
    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """预处理掩码：确保是标签图格式"""
        if mask is None or mask.size == 0:
            return np.zeros((1, 1), dtype=np.int32)
        
        # 如果是二值图，转换为标签图
        if np.max(mask) <= 1:
            return measure.label(mask > 0).astype(np.int32)
        
        return mask.astype(np.int32)
    
    def compute_batch_metrics(self, gt_masks: List[np.ndarray], 
                            pred_masks: List[np.ndarray]) -> List[MetricsResult]:
        """批量计算指标"""
        if len(gt_masks) != len(pred_masks):
            raise ValueError("GT掩码和预测掩码数量不匹配")
        
        results = []
        for gt_mask, pred_mask in zip(gt_masks, pred_masks):
            result = self.compute_all_metrics(gt_mask, pred_mask)
            results.append(result)
        
        return results
    
    def aggregate_metrics(self, results: List[MetricsResult], 
                        exclude_infinite: bool = True) -> Dict[str, float]:
        """聚合多个结果的平均指标"""
        if not results:
            return {}
        
        metrics_dict = {
            'ap50': [],
            'ap75': [],
            'iou_score': [],
            'dice_score': [],
            'hd95': [],
            'gt_instances': [],
            'pred_instances': []
        }
        
        # 收集所有值
        for result in results:
            result_dict = result.to_dict()
            for key in metrics_dict:
                value = result_dict[key]
                # 可选择排除无穷值
                if exclude_infinite and key == 'hd95' and not np.isfinite(value):
                    continue
                metrics_dict[key].append(value)
        
        # 计算平均值
        aggregated = {}
        for key, values in metrics_dict.items():
            if values:
                aggregated[f'mean_{key}'] = float(np.mean(values))
                aggregated[f'std_{key}'] = float(np.std(values))
                aggregated[f'median_{key}'] = float(np.median(values))
            else:
                aggregated[f'mean_{key}'] = float('inf') if key == 'hd95' else 0.0
                aggregated[f'std_{key}'] = 0.0
                aggregated[f'median_{key}'] = float('inf') if key == 'hd95' else 0.0
        
        return aggregated