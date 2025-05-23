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
    """准备SAM训练的输入和目标 - 处理张量掩码"""
    
    try:
        # 输入数据
        inputs = {
            'images': batch['images'],  # [B, C, H, W]
            'point_coords': batch.get('point_coords', []),
            'point_labels': batch.get('point_labels', []),
            'boxes': batch.get('boxes', []),
            'mask_inputs': batch.get('mask_inputs', None),
            'multimask_output': batch.get('multimask_output', False)
        }
        
        # 目标数据 - 现在应该是张量
        ground_truth_masks = batch['ground_truth_masks']  # [B, N, H, W]
        
        print(f"DEBUG: ground_truth_masks type: {type(ground_truth_masks)}")
        print(f"DEBUG: ground_truth_masks shape: {ground_truth_masks.shape}")
        
        # 确保在正确设备上
        device = inputs['images'].device
        
        if isinstance(ground_truth_masks, torch.Tensor):
            # 现在是张量，直接使用
            targets_masks = ground_truth_masks.to(device)
            print(f"DEBUG: 使用张量掩码，形状: {targets_masks.shape}")
            
        else:
            # 如果还是列表（向后兼容），转换为张量
            print(f"WARNING: ground_truth_masks still is list, converting to tensor")
            
            if isinstance(ground_truth_masks, list):
                processed_masks = []
                
                for i, masks in enumerate(ground_truth_masks):
                    if isinstance(masks, torch.Tensor):
                        masks = masks.to(device)
                        if len(masks.shape) == 2:
                            masks = masks.unsqueeze(0)
                        processed_masks.append(masks)
                    else:
                        h, w = inputs['images'].shape[-2:]
                        default_mask = torch.zeros(1, h, w, dtype=torch.long, device=device)
                        processed_masks.append(default_mask)
                
                # 统一形状并堆叠
                max_objects = max([mask.shape[0] for mask in processed_masks])
                target_size = processed_masks[0].shape[-2:]
                
                unified_masks = []
                for masks in processed_masks:
                    if masks.shape[-2:] != target_size:
                        masks = torch.nn.functional.interpolate(
                            masks.unsqueeze(1).float(),
                            size=target_size,
                            mode='nearest'
                        ).squeeze(1).long()
                    
                    if masks.shape[0] < max_objects:
                        padding_shape = (max_objects - masks.shape[0], target_size[0], target_size[1])
                        padding = torch.zeros(padding_shape, dtype=torch.long, device=device)
                        masks = torch.cat([masks, padding], dim=0)
                    elif masks.shape[0] > max_objects:
                        masks = masks[:max_objects]
                    
                    unified_masks.append(masks)
                
                targets_masks = torch.stack(unified_masks)
            else:
                # 创建默认张量
                batch_size = inputs['images'].shape[0]
                h, w = inputs['images'].shape[-2:]
                targets_masks = torch.zeros(batch_size, 1, h, w, dtype=torch.long, device=device)
        
        targets = {
            'masks': targets_masks  # [B, N, H, W]
        }
        
        # 计算IoU目标
        try:
            if targets_masks.numel() > 0:
                iou_targets = calculate_mask_iou_targets(targets_masks)
                targets['iou_targets'] = iou_targets
        except Exception as e:
            print(f"WARNING: IoU targets calculation failed: {e}")
            targets['iou_targets'] = torch.ones(targets_masks.shape[0], targets_masks.shape[1], device=device)
        
        print(f"DEBUG: prepare_sam_inputs successful, targets_masks shape: {targets_masks.shape}")
        return inputs, targets
        
    except Exception as e:
        print(f"ERROR in prepare_sam_inputs: {e}")
        import traceback
        traceback.print_exc()
        
        # 返回默认值
        batch_size = batch['images'].shape[0]
        device = batch['images'].device
        h, w = batch['images'].shape[-2:]
        
        inputs = {
            'images': batch['images'],
            'point_coords': [],
            'point_labels': [],
            'boxes': [],
            'mask_inputs': None,
            'multimask_output': False
        }
        
        targets = {
            'masks': torch.zeros(batch_size, 1, h, w, dtype=torch.long, device=device),
            'iou_targets': torch.ones(batch_size, 1, device=device)
        }
        
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
    """验证SAM批次数据的有效性 - 支持张量掩码"""
    required_keys = ['images', 'ground_truth_masks']
    
    for key in required_keys:
        if key not in batch:
            print(f"批次数据缺少必需的键: {key}")
            return False
    
    # 检查张量形状
    images = batch['images']
    masks = batch['ground_truth_masks']
    
    print("images.shape: ", images.shape)
    print("images.type: ", type(images))
    print("masks.shape: ", masks.shape)  # 现在应该是张量
    print("masks.type: ", type(masks))
    
    # 检查图像张量
    if not isinstance(images, torch.Tensor):
        print(f"images不是张量: {type(images)}")
        return False
        
    if len(images.shape) != 4:
        print(f"图像张量形状错误: {images.shape}，期望 [B, C, H, W]")
        return False
    
    # 检查掩码张量
    if not isinstance(masks, torch.Tensor):
        print(f"masks不是张量: {type(masks)}，期望 torch.Tensor")
        return False
    
    if len(masks.shape) != 4:
        print(f"掩码张量形状错误: {masks.shape}，期望 [B, N, H, W]")
        return False
    
    # 检查批次大小匹配
    if images.shape[0] != masks.shape[0]:
        print(f"批次大小不匹配: 图像 {images.shape[0]} vs 掩码 {masks.shape[0]}")
        return False
    
    # 检查空间尺寸匹配（可选，因为SAM可能会resize）
    # if images.shape[-2:] != masks.shape[-2:]:
    #     print(f"空间尺寸不匹配: 图像 {images.shape[-2:]} vs 掩码 {masks.shape[-2:]}")
    #     return False
    
    print(f"✅ 批次验证通过：图像 {images.shape}, 掩码 {masks.shape}")
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