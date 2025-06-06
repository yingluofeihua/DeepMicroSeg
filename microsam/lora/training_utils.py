"""
LoRA训练工具函数 - 修复版
修复SAM训练中的掩码形状不匹配问题
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
        """计算总损失 - 修复多实例掩码处理"""
        pred_masks = predictions['masks']  # [B, 1, H, W] - SAM输出单个掩码
        target_masks = targets['masks']    # [B, N, H, W] - 多个实例掩码
        
        # 🔧 关键修复：将多实例目标掩码合并为单个二进制掩码
        if target_masks.shape[1] > 1:
            # 将所有实例掩码合并为一个二进制掩码 [B, N, H, W] -> [B, 1, H, W]
            combined_target = (target_masks.sum(dim=1, keepdim=True) > 0).float()
        else:
            combined_target = target_masks.float()
        
        # 确保预测和目标形状匹配
        if pred_masks.shape != combined_target.shape:
            # 如果空间尺寸不匹配，调整目标掩码尺寸
            combined_target = F.interpolate(
                combined_target, 
                size=pred_masks.shape[-2:], 
                mode='nearest'
            )
        
        loss_dict = {}
        total_loss = 0.0
        
        # Focal Loss
        if self.use_focal_loss:
            focal_loss = self._focal_loss(pred_masks, combined_target)
            loss_dict['focal_loss'] = focal_loss
            total_loss += self.focal_loss_weight * focal_loss
        
        # Dice Loss
        if self.use_dice_loss:
            dice_loss = self._dice_loss(pred_masks, combined_target)
            loss_dict['dice_loss'] = dice_loss
            total_loss += self.dice_loss_weight * dice_loss
        
        # IoU Loss
        if self.use_iou_loss:
            iou_loss = self._iou_loss(pred_masks, combined_target)
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
            pred_masks, target_masks, reduction='none'
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


def prepare_sam_inputs(batch: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """准备SAM训练的输入和目标 - 修复多实例掩码处理"""
    
    try:
        # print(f"batch: {batch.keys()}")
        # 输入数据
        inputs = {
            'images': batch['images'],  # [B, C, H, W]
            'point_coords': batch.get('point_coords', []),
            'point_labels': batch.get('point_labels', []),
            'boxes': batch.get('boxes', []),
            'mask_inputs': batch.get('mask_inputs', None),
            'multimask_output': batch.get('multimask_output', False)
        }
        
        # 目标数据
        ground_truth_masks = batch['ground_truth_masks']  # [B, N, H, W]
        # print(f"ground_truth_masks: {ground_truth_masks}")
        # print(ground_truth_masks.shape)
        # 确保在正确设备上
        device = inputs['images'].device

        targets_masks = ground_truth_masks.to(device)
        
        # if isinstance(ground_truth_masks, torch.Tensor):
        #     targets_masks = ground_truth_masks.to(device)
            
        #     # 🔧 关键修复：处理多实例掩码
        #     if targets_masks.shape[1] > 1:
        #         # 方案1：合并所有实例为单个二进制掩码
        #         binary_masks = (targets_masks.sum(dim=1, keepdim=True) > 0).float()
        #         targets_masks = binary_masks
            
        # else:
        #     # 向后兼容处理
        #     print(f"WARNING: ground_truth_masks还是列表格式，转换为张量")
            
        #     if isinstance(ground_truth_masks, list):
        #         processed_masks = []
                
        #         for i, masks in enumerate(ground_truth_masks):
        #             if isinstance(masks, torch.Tensor):
        #                 masks = masks.to(device)
        #                 if len(masks.shape) == 2:
        #                     masks = masks.unsqueeze(0)
                        
        #                 # 如果有多个实例，合并为二进制掩码
        #                 if masks.shape[0] > 1:
        #                     binary_mask = (masks.sum(dim=0, keepdim=True) > 0).float()
        #                     processed_masks.append(binary_mask)
        #                 else:
        #                     processed_masks.append(masks)
        #             else:
        #                 h, w = inputs['images'].shape[-2:]
        #                 default_mask = torch.zeros(1, h, w, dtype=torch.float32, device=device)
        #                 processed_masks.append(default_mask)
                
        #         # 统一形状并堆叠
        #         target_size = processed_masks[0].shape[-2:]
        #         unified_masks = []
                
        #         for masks in processed_masks:
        #             if masks.shape[-2:] != target_size:
        #                 masks = torch.nn.functional.interpolate(
        #                     masks.unsqueeze(1).float(),
        #                     size=target_size,
        #                     mode='nearest'
        #                 ).squeeze(1)
        #             unified_masks.append(masks)
                
        #         targets_masks = torch.stack(unified_masks)
        #     else:
        #         # 创建默认张量
        #         batch_size = inputs['images'].shape[0]
        #         h, w = inputs['images'].shape[-2:]
        #         targets_masks = torch.zeros(batch_size, 1, h, w, dtype=torch.float32, device=device)
        
        targets = {
            'masks': targets_masks  # [B, 1, H, W] - 现在是单个二进制掩码
        }
        
        # 计算IoU目标
        try:
            if targets_masks.numel() > 0:
                iou_targets = calculate_mask_iou_targets(targets_masks)
                targets['iou_targets'] = iou_targets
        except Exception as e:
            print(f"WARNING: IoU targets calculation failed: {e}")
            targets['iou_targets'] = torch.ones(targets_masks.shape[0], targets_masks.shape[1], device=device)
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
            'masks': torch.zeros(batch_size, 1, h, w, dtype=torch.float32, device=device),
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
        def safe_item(value):
            """安全地获取数值，处理tensor和float"""
            if isinstance(value, torch.Tensor):
                return value.item()
            return float(value)
        
        self.total_loss += safe_item(loss_dict.get('total_loss', 0.0))
        self.focal_loss += safe_item(loss_dict.get('focal_loss', 0.0))
        self.dice_loss += safe_item(loss_dict.get('dice_loss', 0.0))
        self.iou_loss += safe_item(loss_dict.get('iou_loss', 0.0))
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
        def safe_item(value):
            """安全地获取数值"""
            if isinstance(value, torch.Tensor):
                return value.item()
            return float(value)
        
        return {key: safe_item(value) for key, value in loss_dict.items()}
    
    return training_step


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

def prepare_sam_inputs_multi_instance(batch: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """准备SAM训练的输入和目标 - 支持多实例"""
    
    try:
        # 输入数据
        inputs = {
            'images': batch['images'],  # [B, 3, H, W]
            'point_coords': batch.get('point_coords', []),
            'point_labels': batch.get('point_labels', []),
            'boxes': batch.get('boxes', []),
            'mask_inputs': batch.get('mask_inputs', None),
            'multimask_output': batch.get('multimask_output', False)
        }
        
        # 🔧 关键修复：保持多实例掩码格式
        ground_truth_masks = batch['ground_truth_masks']  # [B, N, H, W]
        device = inputs['images'].device
        
        if isinstance(ground_truth_masks, torch.Tensor):
            targets_masks = ground_truth_masks.to(device)
        else:
            # 处理列表格式（向后兼容）
            batch_size = inputs['images'].shape[0]
            h, w = inputs['images'].shape[-2:]
            targets_masks = torch.zeros(batch_size, 1, h, w, dtype=torch.float32, device=device)
        
        targets = {
            'masks': targets_masks,  # [B, N, H, W] - 保持多实例格式
            'num_instances': torch.tensor([masks.shape[1] for masks in [targets_masks]], device=device)
        }
        
        return inputs, targets
        
    except Exception as e:
        print(f"ERROR in prepare_sam_inputs_multi_instance: {e}")
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
            'masks': torch.zeros(batch_size, 1, h, w, dtype=torch.float32, device=device),
            'num_instances': torch.ones(batch_size, device=device)
        }
        
        return inputs, targets


class SAMLossMultiInstance(nn.Module):
    """SAM训练的多实例损失函数"""
    
    def __init__(
        self,
        focal_loss_weight: float = 20.0,
        dice_loss_weight: float = 1.0,
        iou_loss_weight: float = 1.0,
        instance_loss_weight: float = 5.0,
        use_focal_loss: bool = True,
        use_dice_loss: bool = True,
        use_iou_loss: bool = True,
        use_instance_loss: bool = True
    ):
        super().__init__()
        
        self.focal_loss_weight = focal_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.iou_loss_weight = iou_loss_weight
        self.instance_loss_weight = instance_loss_weight
        
        self.use_focal_loss = use_focal_loss
        self.use_dice_loss = use_dice_loss
        self.use_iou_loss = use_iou_loss
        self.use_instance_loss = use_instance_loss
        
        # Focal loss参数
        self.focal_alpha = 0.8
        self.focal_gamma = 2.0
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算多实例损失"""
        
        pred_masks = predictions['masks']  # [B, N_pred, H, W] 或 [B, 1, H, W]
        target_masks = targets['masks']    # [B, N_target, H, W]
        
        device = pred_masks.device
        batch_size = pred_masks.shape[0]
        
        loss_dict = {}
        total_loss = 0.0
        
        # 🔧 策略1：如果SAM输出单个掩码，与最佳匹配的目标计算损失
        if pred_masks.shape[1] == 1:
            # SAM输出单个掩码，找到最佳匹配的目标实例
            best_matches = []
            
            for b in range(batch_size):
                pred_mask = pred_masks[b, 0]  # [H, W]
                target_instances = target_masks[b]  # [N, H, W]
                
                best_iou = 0.0
                best_target = None
                
                # 找到IoU最高的目标实例
                for i in range(target_instances.shape[0]):
                    target_mask = target_instances[i]
                    if target_mask.sum() == 0:  # 跳过空掩码
                        continue
                    
                    iou = self._calculate_iou(pred_mask, target_mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_target = target_mask
                
                if best_target is not None:
                    best_matches.append(best_target)
                else:
                    # 如果没有找到匹配，使用合并的目标
                    combined_target = (target_instances.sum(dim=0) > 0).float()
                    best_matches.append(combined_target)
            
            # 重新组织为批次格式
            matched_targets = torch.stack(best_matches).unsqueeze(1)  # [B, 1, H, W]
            
            # 计算损失
            if self.use_focal_loss:
                focal_loss = self._focal_loss(pred_masks, matched_targets)
                loss_dict['focal_loss'] = focal_loss
                total_loss += self.focal_loss_weight * focal_loss
            
            if self.use_dice_loss:
                dice_loss = self._dice_loss(pred_masks, matched_targets)
                loss_dict['dice_loss'] = dice_loss
                total_loss += self.dice_loss_weight * dice_loss
            
            if self.use_iou_loss:
                iou_loss = self._iou_loss(pred_masks, matched_targets)
                loss_dict['iou_loss'] = iou_loss
                total_loss += self.iou_loss_weight * iou_loss
        
        # 🔧 策略2：如果SAM输出多个掩码，进行实例级匹配
        else:
            # 实例级损失计算（更复杂，但更准确）
            instance_losses = []
            
            for b in range(batch_size):
                pred_instances = pred_masks[b]  # [N_pred, H, W]
                target_instances = target_masks[b]  # [N_target, H, W]
                
                # 使用匈牙利算法进行最优匹配（简化版本）
                instance_loss = self._compute_instance_matching_loss(
                    pred_instances, target_instances
                )
                instance_losses.append(instance_loss)
            
            avg_instance_loss = torch.stack(instance_losses).mean()
            loss_dict['instance_loss'] = avg_instance_loss
            total_loss += self.instance_loss_weight * avg_instance_loss
        
        loss_dict['total_loss'] = total_loss
        return loss_dict
    
    def _calculate_iou(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """计算IoU"""
        pred_binary = (pred > 0.5).float()
        target_binary = (target > 0.5).float()
        
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection
        
        if union > 0:
            return (intersection / union).item()
        return 0.0
    
    def _compute_instance_matching_loss(self, pred_instances: torch.Tensor, 
                                      target_instances: torch.Tensor) -> torch.Tensor:
        """计算实例匹配损失（简化版本）"""
        # 这里实现简化的匹配策略
        # 实际应用中可以使用匈牙利算法进行最优匹配
        
        n_pred = pred_instances.shape[0]
        n_target = target_instances.shape[0]
        
        if n_pred == 0 or n_target == 0:
            return torch.tensor(1.0, device=pred_instances.device)
        
        # 计算所有预测-目标对的损失矩阵
        loss_matrix = torch.zeros(n_pred, n_target, device=pred_instances.device)
        
        for i in range(n_pred):
            for j in range(n_target):
                if target_instances[j].sum() == 0:  # 跳过空目标
                    continue
                
                pred_mask = pred_instances[i:i+1]  # [1, H, W]
                target_mask = target_instances[j:j+1]  # [1, H, W]
                
                # 计算二值交叉熵损失
                bce_loss = F.binary_cross_entropy_with_logits(pred_mask, target_mask)
                loss_matrix[i, j] = bce_loss
        
        # 简化匹配：选择每个预测的最小损失
        if loss_matrix.numel() > 0:
            min_losses = loss_matrix.min(dim=1)[0]
            return min_losses.mean()
        else:
            return torch.tensor(1.0, device=pred_instances.device)
    
    def _focal_loss(self, pred_masks: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        """Focal Loss实现"""
        pred_sigmoid = torch.sigmoid(pred_masks)
        
        ce_loss = F.binary_cross_entropy_with_logits(
            pred_masks, target_masks, reduction='none'
        )
        
        p_t = pred_sigmoid * target_masks + (1 - pred_sigmoid) * (1 - target_masks)
        alpha_t = self.focal_alpha * target_masks + (1 - self.focal_alpha) * (1 - target_masks)
        
        focal_weight = alpha_t * (1 - p_t) ** self.focal_gamma
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()
    
    def _dice_loss(self, pred_masks: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        """Dice Loss实现"""
        pred_sigmoid = torch.sigmoid(pred_masks)
        smooth = 1.0
        
        intersection = (pred_sigmoid * target_masks).sum(dim=(-2, -1))
        total = pred_sigmoid.sum(dim=(-2, -1)) + target_masks.sum(dim=(-2, -1))
        
        dice_score = (2.0 * intersection + smooth) / (total + smooth)
        dice_loss = 1.0 - dice_score
        
        return dice_loss.mean()
    
    def _iou_loss(self, pred_masks: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        """IoU Loss实现"""
        pred_sigmoid = torch.sigmoid(pred_masks)
        
        intersection = (pred_sigmoid * target_masks).sum(dim=(-2, -1))
        union = pred_sigmoid.sum(dim=(-2, -1)) + target_masks.sum(dim=(-2, -1)) - intersection
        
        iou_score = intersection / (union + 1e-6)
        iou_loss = 1.0 - iou_score
        
        return iou_loss.mean()

def validate_sam_batch_multi_instance(batch: Dict[str, Any]) -> bool:
    """验证SAM多实例批次数据的有效性"""
    required_keys = ['images', 'ground_truth_masks']
    
    for key in required_keys:
        if key not in batch:
            print(f"批次数据缺少必需的键: {key}")
            return False
    
    # 检查张量形状
    images = batch['images']
    masks = batch['ground_truth_masks']
    
    if not isinstance(images, torch.Tensor):
        print(f"images不是张量: {type(images)}")
        return False
        
    if len(images.shape) != 4:
        print(f"图像张量形状错误: {images.shape}，期望 [B, C, H, W]")
        return False
    
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
    
    return True

def create_sam_training_step_multi_instance(model, optimizer, loss_fn, device):
    """创建多实例SAM训练步骤函数"""
    
    def training_step(batch: Dict[str, Any]) -> Dict[str, float]:
        """执行一个多实例训练步骤"""
        
        # 验证批次数据
        if not validate_sam_batch_multi_instance(batch):
            return {'error': 1.0}
        
        # 准备输入和目标
        inputs, targets = prepare_sam_inputs_multi_instance(batch)
        
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
        def safe_item(value):
            if isinstance(value, torch.Tensor):
                return value.item()
            return float(value)
        
        return {key: safe_item(value) for key, value in loss_dict.items()}
    
    return training_step


