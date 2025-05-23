"""
LoRA训练工具函数
包含SAM训练的损失函数、数据预处理等工具
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class SAMLoss(nn.Module):
    """SAM训练的综合损失函数"""
    
    def __init__(
        self,
        focal_loss_weight: float = 20.0,
        dice_loss_weight: float = 1.0,
        iou_loss_weight: float = 1.0,
        use_focal_loss: bool = True,
        use_dice_loss: bool = True,
        use_iou_loss: bool = True
    ):
        super().__init__()
        
        self.focal_loss_weight = focal_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.iou_loss_weight = iou_loss_weight
        self.use_focal_loss = use_focal_loss
        self.use_dice_loss = use_dice_loss
        self.use_iou_loss = use_iou_loss
        
        # Focal loss参数
        self.focal_alpha = 0.8
        self.focal_gamma = 2.0
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算总损失"""
        pred_masks = predictions['masks']  # [B, N, H, W]
        target_masks = targets['masks']    # [B, N, H, W]
        
        # 确保尺寸匹配
        if pred_masks.shape != target_masks.shape:
            target_masks = F.interpolate(
                target_masks.float(), 
                size=pred_masks.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        loss_dict = {}
        total_loss = 0.0
        
        # Focal Loss
        if self.use_focal_loss:
            focal_loss = self._focal_loss(pred_masks, target_masks)
            loss_dict['focal_loss'] = focal_loss
            total_loss += self.focal_loss_weight * focal_loss
        
        # Dice Loss
        if self.use_dice_loss:
            dice_loss = self._dice_loss(pred_masks, target_masks)
            loss_dict['dice_loss'] = dice_loss
            total_loss += self.dice_loss_weight * dice_loss
        
        # IoU Loss
        if self.use_iou_loss:
            iou_loss = self._iou_loss(pred_masks, target_masks)
            loss_dict['iou_loss'] = iou_loss
            total_loss += self.iou_loss_weight * iou_loss
        
        # IoU预测损失（如果有）
        if 'iou_predictions' in predictions and 'iou_targets' in targets:
            iou_pred_loss = F.mse_loss(
                predictions['iou_predictions'],
                targets['iou_targets']
            )
            loss_dict['iou_prediction_loss'] = iou_pred_loss
            total_loss += iou_pred_loss
        
        loss_dict['total_loss'] = total_loss
        return loss_dict
    
    def _focal_loss(self, pred_masks: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        """Focal Loss实现"""
        # 使用sigmoid激活
        pred_sigmoid = torch.sigmoid(pred_masks)
        
        # 计算focal loss
        ce_loss = F.binary_cross_entropy_with_logits(
            pred_masks, target_masks.float(), reduction='none'
        )
        
        p_t = pred_sigmoid * target_masks + (1 - pred_sigmoid) * (1 - target_masks)
        alpha_t = self.focal_alpha * target_masks + (1 - self.focal_alpha) * (1 - target_masks)
        
        focal_weight = alpha_t * (1 - p_t) ** self.focal_gamma
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()
    
    def _dice_loss(self, pred_masks: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        """Dice Loss实现"""
        pred_sigmoid = torch.sigmoid(pred_masks)
        
        # 平滑因子
        smooth = 1.0
        
        # 计算每个样本的dice loss
        intersection = (pred_sigmoid * target_masks).sum(dim=(-2, -1))
        total = pred_sigmoid.sum(dim=(-2, -1)) + target_masks.sum(dim=(-2, -1))
        
        dice_score = (2.0 * intersection + smooth) / (total + smooth)
        dice_loss = 1.0 - dice_score
        
        return dice_loss.mean()
    
    def _iou_loss(self, pred_masks: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        """IoU Loss实现"""
        pred_sigmoid = torch.sigmoid(pred_masks)
        
        # 计算IoU
        intersection = (pred_sigmoid * target_masks).sum(dim=(-2, -1))
        union = pred_sigmoid.sum(dim=(-2, -1)) + target_masks.sum(dim=(-2, -1)) - intersection
        
        # 避免除零
        iou_score = intersection / (union + 1e-6)
        iou_loss = 1.0 - iou_score
        
        return iou_loss.mean()


def calculate_sam_loss(predictions: Dict[str, torch.Tensor], 
                      ground_truth: Dict[str, torch.Tensor],
                      loss_config: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
    """计算SAM损失的便捷函数"""
    if loss_config is None:
        loss_config = {
            'focal_loss_weight': 20.0,
            'dice_loss_weight': 1.0,
            'iou_loss_weight': 1.0
        }
    
    loss_fn = SAMLoss(**loss_config)
    return loss_fn(predictions, ground_truth)


def prepare_sam_inputs(batch: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """准备SAM训练的输入和目标"""
    
    # 输入数据
    inputs = {
        'images': batch['images'],  # 已经是stacked tensor
        'point_coords': batch.get('point_coords', []),
        'point_labels': batch.get('point_labels', []),
        'boxes': batch.get('boxes', []),
        'mask_inputs': batch.get('mask_inputs', None),
        'multimask_output': batch.get('multimask_output', False)
    }
    
    # 目标数据 - 需要处理列表格式的ground_truth_masks
    ground_truth_masks = batch['ground_truth_masks']
    
    # 将列表中的掩码堆叠成批量张量
    if isinstance(ground_truth_masks, list):
        # 确保所有掩码都有相同的形状
        batch_masks = []
        max_num_objects = 0
        target_size = None
        
        # 首先确定最大对象数和目标尺寸
        for masks in ground_truth_masks:
            if isinstance(masks, torch.Tensor) and masks.numel() > 0:
                if target_size is None:
                    target_size = masks.shape[-2:]  # H, W
                max_num_objects = max(max_num_objects, masks.shape[0])
        
        # 如果没有有效的掩码，使用默认尺寸
        if target_size is None:
            target_size = (1024, 1024)  # SAM默认尺寸
        
        # 处理每个批次中的掩码
        for masks in ground_truth_masks:
            if isinstance(masks, torch.Tensor) and masks.numel() > 0:
                # 调整到目标尺寸
                if masks.shape[-2:] != target_size:
                    masks = torch.nn.functional.interpolate(
                        masks.unsqueeze(1).float(),  # 添加通道维度
                        size=target_size,
                        mode='nearest'
                    ).squeeze(1).long()  # 移除通道维度并转回long
                
                # 填充到最大对象数
                if masks.shape[0] < max_num_objects:
                    padding = torch.zeros(
                        max_num_objects - masks.shape[0], 
                        target_size[0], 
                        target_size[1], 
                        dtype=masks.dtype,
                        device=masks.device
                    )
                    masks = torch.cat([masks, padding], dim=0)
                elif masks.shape[0] > max_num_objects:
                    masks = masks[:max_num_objects]
                
                batch_masks.append(masks)
            else:
                # 空掩码的情况
                empty_masks = torch.zeros(
                    max_num_objects, 
                    target_size[0], 
                    target_size[1], 
                    dtype=torch.long
                )
                batch_masks.append(empty_masks)
        
        # 堆叠成批量张量
        if batch_masks:
            targets_masks = torch.stack(batch_masks)  # [B, N, H, W]
        else:
            targets_masks = torch.zeros(len(ground_truth_masks), 1, target_size[0], target_size[1], dtype=torch.long)
    else:
        targets_masks = ground_truth_masks
    
    targets = {
        'masks': targets_masks
    }
    
    # 计算IoU目标（如果需要）
    if targets_masks.numel() > 0:
        iou_targets = calculate_mask_iou_targets(targets_masks)
        targets['iou_targets'] = iou_targets
    
    return inputs, targets


def calculate_mask_iou_targets(masks: torch.Tensor) -> torch.Tensor:
    """计算掩码的IoU目标值"""
    if masks.numel() == 0:
        return torch.tensor([])
    
    # masks shape: [B, N, H, W]
    batch_size = masks.shape[0]
    num_objects = masks.shape[1] if len(masks.shape) > 3 else 1
    
    if len(masks.shape) == 3:
        # 如果是 [B, H, W]，添加对象维度
        masks = masks.unsqueeze(1)  # [B, 1, H, W]
        num_objects = 1
    
    # 计算每个掩码的面积比例作为IoU目标
    mask_areas = masks.sum(dim=(-2, -1))  # [B, N]
    total_area = masks.shape[-2] * masks.shape[-1]
    
    # 使用面积比例作为IoU的粗略估计，并限制在合理范围内
    iou_targets = torch.clamp(mask_areas.float() / total_area, 0.1, 1.0)
    
    return iou_targets


class MaskPostProcessor:
    """掩码后处理器"""
    
    def __init__(self, threshold: float = 0.0, remove_small_objects: bool = True, 
                 min_object_size: int = 100):
        self.threshold = threshold
        self.remove_small_objects = remove_small_objects
        self.min_object_size = min_object_size
    
    def process(self, masks: torch.Tensor) -> torch.Tensor:
        """后处理预测的掩码"""
        # 应用阈值
        if self.threshold > 0:
            masks = (masks > self.threshold).float()
        else:
            masks = torch.sigmoid(masks)
        
        # 移除小对象（如果需要）
        if self.remove_small_objects:
            masks = self._remove_small_objects(masks)
        
        return masks
    
    def _remove_small_objects(self, masks: torch.Tensor) -> torch.Tensor:
        """移除小对象"""
        processed_masks = masks.clone()
        
        for b in range(masks.shape[0]):
            for m in range(masks.shape[1]):
                mask = masks[b, m]
                mask_area = mask.sum()
                
                if mask_area < self.min_object_size:
                    processed_masks[b, m] = 0
        
        return processed_masks


class SAMDataAugmentation:
    """SAM训练的数据增强"""
    
    def __init__(self, 
                 flip_prob: float = 0.5,
                 rotation_prob: float = 0.3,
                 scale_prob: float = 0.3,
                 noise_prob: float = 0.2):
        self.flip_prob = flip_prob
        self.rotation_prob = rotation_prob
        self.scale_prob = scale_prob
        self.noise_prob = noise_prob
    
    def augment_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """对批次数据进行增强"""
        augmented_batch = batch.copy()
        
        # 随机水平翻转
        if torch.rand(1) < self.flip_prob:
            augmented_batch = self._horizontal_flip(augmented_batch)
        
        # 随机垂直翻转
        if torch.rand(1) < self.flip_prob:
            augmented_batch = self._vertical_flip(augmented_batch)
        
        # 添加噪声
        if torch.rand(1) < self.noise_prob:
            augmented_batch = self._add_noise(augmented_batch)
        
        return augmented_batch
    
    def _horizontal_flip(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """水平翻转"""
        batch['images'] = torch.flip(batch['images'], dims=[-1])
        if 'ground_truth_masks' in batch:
            batch['ground_truth_masks'] = torch.flip(batch['ground_truth_masks'], dims=[-1])
        
        # 调整点坐标
        if 'point_coords' in batch:
            for i, coords in enumerate(batch['point_coords']):
                if len(coords) > 0:
                    coords[:, 0] = batch['images'].shape[-1] - 1 - coords[:, 0]
        
        # 调整边界框
        if 'boxes' in batch:
            for i, boxes in enumerate(batch['boxes']):
                if len(boxes) > 0:
                    width = batch['images'].shape[-1]
                    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                    boxes[:, 0] = width - 1 - x2
                    boxes[:, 2] = width - 1 - x1
        
        return batch
    
    def _vertical_flip(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """垂直翻转"""
        batch['images'] = torch.flip(batch['images'], dims=[-2])
        if 'ground_truth_masks' in batch:
            batch['ground_truth_masks'] = torch.flip(batch['ground_truth_masks'], dims=[-2])
        
        # 调整点坐标
        if 'point_coords' in batch:
            for i, coords in enumerate(batch['point_coords']):
                if len(coords) > 0:
                    coords[:, 1] = batch['images'].shape[-2] - 1 - coords[:, 1]
        
        # 调整边界框
        if 'boxes' in batch:
            for i, boxes in enumerate(batch['boxes']):
                if len(boxes) > 0:
                    height = batch['images'].shape[-2]
                    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                    boxes[:, 1] = height - 1 - y2
                    boxes[:, 3] = height - 1 - y1
        
        return batch
    
    def _add_noise(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """添加噪声"""
        noise_level = 0.02 * torch.rand(1)
        noise = torch.randn_like(batch['images']) * noise_level
        batch['images'] = torch.clamp(batch['images'] + noise, 0, 1)
        
        return batch


class TrainingMetrics:
    """训练指标计算器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置指标"""
        self.total_loss = 0.0
        self.focal_loss = 0.0
        self.dice_loss = 0.0
        self.iou_loss = 0.0
        self.count = 0
    
    def update(self, loss_dict: Dict[str, torch.Tensor]):
        """更新指标"""
        self.total_loss += loss_dict.get('total_loss', 0.0).item()
        self.focal_loss += loss_dict.get('focal_loss', 0.0).item()
        self.dice_loss += loss_dict.get('dice_loss', 0.0).item()
        self.iou_loss += loss_dict.get('iou_loss', 0.0).item()
        self.count += 1
    
    def compute(self) -> Dict[str, float]:
        """计算平均指标"""
        if self.count == 0:
            return {}
        
        return {
            'avg_total_loss': self.total_loss / self.count,
            'avg_focal_loss': self.focal_loss / self.count,
            'avg_dice_loss': self.dice_loss / self.count,
            'avg_iou_loss': self.iou_loss / self.count
        }


def validate_sam_batch(batch: Dict[str, Any]) -> bool:
    """验证SAM批次数据的有效性"""
    required_keys = ['images', 'ground_truth_masks']
    
    for key in required_keys:
        if key not in batch:
            print(f"批次数据缺少必需的键: {key}")
            return False
    
    # 检查张量形状
    images = batch['images']
    masks = batch['ground_truth_masks']
    
    if len(images.shape) != 4:
        print(f"图像张量形状错误: {images.shape}，期望 [B, C, H, W]")
        return False
    
    if len(masks.shape) < 3:
        print(f"掩码张量形状错误: {masks.shape}，期望至少 [B, H, W] 或 [B, N, H, W]")
        return False
    
    # 检查尺寸匹配
    if images.shape[0] != masks.shape[0]:
        print(f"批次大小不匹配: 图像 {images.shape[0]} vs 掩码 {masks.shape[0]}")
        return False
    
    return True


def create_sam_training_step(model, optimizer, loss_fn, device):
    """创建SAM训练步骤函数"""
    
    def training_step(batch: Dict[str, Any]) -> Dict[str, float]:
        """执行一个训练步骤"""
        # 验证批次数据
        if not validate_sam_batch(batch):
            return {'error': 1.0}
        
        # 准备输入和目标
        inputs, targets = prepare_sam_inputs(batch)
        
        # 将数据移动到设备
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(device)
            elif isinstance(value, list):
                inputs[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
        
        for key, value in targets.items():
            if isinstance(value, torch.Tensor):
                targets[key] = value.to(device)
        
        # 前向传播
        model.train()
        predictions = model(inputs)
        
        # 计算损失
        loss_dict = loss_fn(predictions, targets)
        total_loss = loss_dict['total_loss']
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 返回损失信息
        return {key: value.item() if isinstance(value, torch.Tensor) else value 
                for key, value in loss_dict.items()}
    
    return training_step