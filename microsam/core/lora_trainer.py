"""
LoRA训练器 - 完整多实例版本
支持SAM的多实例细胞分割训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import time
import json
from tqdm import tqdm
import wandb

from config.lora_config import LoRATrainingSettings
from lora.sam_lora_wrapper import SAMLoRAWrapper, create_sam_lora_model
from lora.data_loaders import create_data_loaders
from lora.training_utils import (
    SAMLoss, prepare_sam_inputs, TrainingMetrics,
    # 🔧 新增多实例相关函数
    SAMLossMultiInstance, 
    prepare_sam_inputs_multi_instance,
    create_sam_training_step_multi_instance,
    validate_sam_batch_multi_instance
)
from core.metrics import ComprehensiveMetrics, MetricsResult
from utils.file_utils import setup_logging
from utils.model_utils import optimize_memory, get_device_info, print_model_summary


class LoRATrainer:
    """LoRA训练器 - 支持多实例SAM细胞分割"""
    
    def __init__(self, config: LoRATrainingSettings):
        self.config = config
        self.device = self._setup_device()
        
        # 初始化组件
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.data_loaders = {}
        self.loss_fn = None
        self.metrics_calculator = ComprehensiveMetrics()
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.early_stopping_counter = 0
        
        # 🔧 多实例训练相关
        self.use_multi_instance = True  # 启用多实例训练
        self.training_step_fn = None
        
        # 日志和监控
        self.writer = None
        self.logger = setup_logging()
        
        # 创建输出目录
        self.output_dir = self.config.get_model_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        self.config.save_to_json(self.output_dir / "config.json")
    
    def _setup_device(self) -> torch.device:
        """设置计算设备"""
        if self.config.experiment.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.experiment.device)
        
        print(f"使用设备: {device}")
        
        if device.type == "cuda":
            device_info = get_device_info()
            print(f"GPU设备数量: {device_info['cuda_device_count']}")
            print(f"当前GPU: {torch.cuda.current_device()}")
            if device_info['gpu_memory']:
                current_gpu = f"gpu_{torch.cuda.current_device()}"
                if current_gpu in device_info['gpu_memory']:
                    gpu_info = device_info['gpu_memory'][current_gpu]
                    print(f"GPU内存: {gpu_info['total_memory'] / 1e9:.1f} GB")
        
        return device
    
    def setup_model(self) -> bool:
        """设置SAM LoRA模型 - 支持多实例"""
        try:
            print("正在设置SAM LoRA多实例模型...")
            
            # 创建LoRA配置
            lora_config = {
                'rank': self.config.lora.rank,
                'alpha': self.config.lora.alpha,
                'dropout': self.config.lora.dropout,
                'target_modules': self.config.lora.target_modules,
                'apply_lora_to': self.config.model.apply_lora_to,
                'freeze_image_encoder': self.config.model.freeze_backbone,
                'freeze_prompt_encoder': self.config.model.freeze_prompt_encoder,
                'freeze_mask_decoder': self.config.model.freeze_mask_decoder
            }
            
            # 创建SAM LoRA模型
            self.model = create_sam_lora_model(
                model_type=self.config.model.base_model_name,
                lora_config=lora_config,
                device=str(self.device)
            )
            
            if self.model is None:
                print("SAM LoRA模型创建失败")
                return False
            
            # 确保模型在正确设备上
            self.model = self.model.to(self.device)
            
            # 确保所有LoRA模块在正确设备上
            if hasattr(self.model, 'lora_modules'):
                for name, lora_module in self.model.lora_modules.items():
                    if hasattr(lora_module, 'lora'):
                        lora_module.lora = lora_module.lora.to(self.device)
                    lora_module = lora_module.to(self.device)
            
            # 打印模型信息
            self.model.print_model_info()
            print_model_summary(self.model)
            
            # 验证模型设备一致性
            self._verify_model_device_consistency()
            
            return True
            
        except Exception as e:
            print(f"模型设置失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _verify_model_device_consistency(self):
        """验证模型设备一致性"""
        print(f"\n验证模型设备一致性...")
        
        device_counts = {}
        
        # 检查主要组件
        for name, module in [
            ('image_encoder', self.model.image_encoder),
            ('prompt_encoder', self.model.prompt_encoder), 
            ('mask_decoder', self.model.mask_decoder)
        ]:
            if module is not None:
                for param_name, param in module.named_parameters():
                    device = str(param.device)
                    device_counts[device] = device_counts.get(device, 0) + 1
        
        # 检查LoRA模块
        if hasattr(self.model, 'lora_modules'):
            for lora_name, lora_module in self.model.lora_modules.items():
                if hasattr(lora_module, 'lora'):
                    for param_name, param in lora_module.lora.named_parameters():
                        device = str(param.device)
                        device_counts[device] = device_counts.get(device, 0) + 1
        
        print(f"设备分布: {device_counts}")
        
        if len(device_counts) > 1:
            print(f"⚠️  发现多个设备，正在统一到 {self.device}")
            self.model = self.model.to(self.device)
            print(f"✅ 所有模型组件已移动到 {self.device}")
        else:
            print(f"✅ 所有模型组件都在 {self.device} 上")
    
    def setup_data_loaders(self) -> bool:
        """设置数据加载器 - 支持多实例"""
        try:
            print("正在创建多实例数据加载器...")
            
            # 🔧 使用SAM数据集格式，支持多实例
            self.data_loaders = create_data_loaders(
                config=self.config.data,
                dataset_type="sam"  # 使用SAM数据集格式
            )
            
            print("多实例数据加载器创建成功:")
            for split, loader in self.data_loaders.items():
                print(f"  {split}: {len(loader)} 批次, {len(loader.dataset)} 样本")
                
                # 🔧 验证第一个批次的数据格式
                if len(loader) > 0:
                    try:
                        first_batch = next(iter(loader))
                        print(f"    - 图像形状: {first_batch['images'].shape}")
                        print(f"    - 掩码形状: {first_batch['ground_truth_masks'].shape}")
                        print(f"    - 最大实例数: {first_batch['ground_truth_masks'].shape[1]}")
                    except Exception as e:
                        print(f"    - 批次验证失败: {e}")
            
            return True
            
        except Exception as e:
            print(f"数据加载器设置失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def setup_optimizer_and_loss(self):
        """设置优化器和损失函数 - 多实例版本"""
        # 只优化LoRA参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        print(f"可训练参数数量: {len(trainable_params)}")
        total_trainable = sum(p.numel() for p in trainable_params)
        print(f"可训练参数总数: {total_trainable:,}")
        
        # 创建优化器
        if self.config.training.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
                eps=self.config.training.adam_epsilon
            )
        else:
            self.optimizer = optim.Adam(
                trainable_params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        
        # 学习率调度器
        if self.config.training.lr_scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=self.config.training.learning_rate * 0.01
            )
        elif self.config.training.lr_scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.num_epochs // 3,
                gamma=0.1
            )
        else:
            self.scheduler = None
        
        # 🔧 创建多实例损失函数
        if self.use_multi_instance:
            loss_config = {
                'focal_loss_weight': 20.0,
                'dice_loss_weight': 1.0,
                'iou_loss_weight': 1.0,
                'instance_loss_weight': 5.0,
                'use_focal_loss': True,
                'use_dice_loss': True,
                'use_iou_loss': True,
                'use_instance_loss': True
            }
            self.loss_fn = SAMLossMultiInstance(**loss_config)
            print("使用多实例SAM损失函数")
        else:
            # 传统单实例损失函数
            loss_config = {
                'focal_loss_weight': 20.0,
                'dice_loss_weight': 1.0,
                'iou_loss_weight': 1.0,
                'use_focal_loss': True,
                'use_dice_loss': True,
                'use_iou_loss': True
            }
            self.loss_fn = SAMLoss(**loss_config)
            print("使用传统SAM损失函数")
        
        print(f"优化器设置完成: {type(self.optimizer).__name__}")
        print(f"学习率调度器: {type(self.scheduler).__name__ if self.scheduler else 'None'}")
    
    def setup_logging(self):
        """设置日志记录"""
        # TensorBoard
        log_dir = self.config.get_logs_dir()
        self.writer = SummaryWriter(log_dir)
        
        # Weights & Biases
        if self.config.experiment.use_wandb:
            wandb.init(
                project=self.config.experiment.wandb_project,
                entity=self.config.experiment.wandb_entity,
                name=self.config.experiment.run_name,
                config=self.config.to_dict(),
                dir=str(self.output_dir)
            )
    
    def train(self) -> bool:
        """开始训练 - 多实例版本"""
        print("="*60)
        print("开始SAM LoRA多实例微调训练")
        print("="*60)
        
        # 设置所有组件
        if not self.setup_model():
            return False
        
        if not self.setup_data_loaders():
            return False
        
        self.setup_optimizer_and_loss()
        self.setup_logging()
        
        # 🔧 创建多实例训练步骤函数
        if self.use_multi_instance:
            self.training_step_fn = create_sam_training_step_multi_instance(
                self.model, self.optimizer, self.loss_fn, self.device
            )
            print("使用多实例训练步骤函数")
        else:
            # 使用传统训练步骤
            from lora.training_utils import create_sam_training_step
            self.training_step_fn = create_sam_training_step(
                self.model, self.optimizer, self.loss_fn, self.device
            )
            print("使用传统训练步骤函数")
        
        # 训练循环
        try:
            for epoch in range(self.config.training.num_epochs):
                self.current_epoch = epoch
                
                print(f"\n开始第 {epoch + 1}/{self.config.training.num_epochs} 轮训练")
                
                # 🔧 使用对应的训练函数
                if self.use_multi_instance:
                    train_metrics = self.train_epoch_multi_instance()
                else:
                    train_metrics = self.train_epoch()
                
                # 验证
                if self.use_multi_instance:
                    val_metrics = self.validate_multi_instance() if 'val' in self.data_loaders else {}
                else:
                    val_metrics = self.validate() if 'val' in self.data_loaders else {}

                # 记录日志
                self.log_metrics(train_metrics, val_metrics, epoch)
                
                # 保存检查点
                if (epoch + 1) % max(1, self.config.training.save_steps // len(self.data_loaders['train'])) == 0:
                    self.save_checkpoint(epoch, val_metrics)
                
                # 早停检查
                if self.check_early_stopping(val_metrics):
                    print(f"早停触发，在第 {epoch + 1} 轮停止训练")
                    break
                
                # 更新学习率
                if self.scheduler:
                    self.scheduler.step()
                
                # 内存优化
                optimize_memory()
            
            # 保存最终模型
            self.save_final_model()
            
            print("\n" + "="*60)
            print("多实例训练完成!")
            print("="*60)
            return True
            
        except Exception as e:
            print(f"训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            if self.writer:
                self.writer.close()
            if self.config.experiment.use_wandb:
                wandb.finish()
    
    def train_epoch_multi_instance(self) -> Dict[str, float]:
        """多实例训练一个epoch"""
        self.model.train()
        
        epoch_metrics = TrainingMetrics()
        
        progress_bar = tqdm(
            self.data_loaders['train'],
            desc=f"Epoch {self.current_epoch + 1}/{self.config.training.num_epochs}"
        )
        
        successful_steps = 0
        failed_steps = 0
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # 🔧 使用多实例训练步骤函数
                step_metrics = self.training_step_fn(batch)
                
                # 检查训练结果
                if 'error' not in step_metrics:
                    epoch_metrics.update(step_metrics)
                    self.global_step += 1
                    successful_steps += 1
                    
                    # 更新进度条
                    progress_bar.set_postfix({
                        'loss': f"{step_metrics.get('total_loss', 0):.4f}",
                        'focal': f"{step_metrics.get('focal_loss', 0):.4f}",
                        'dice': f"{step_metrics.get('dice_loss', 0):.4f}",
                        'instance': f"{step_metrics.get('instance_loss', 0):.4f}",
                        'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                        'success': f"{successful_steps}/{successful_steps + failed_steps}"
                    })
                    
                    # 记录步骤日志
                    if self.global_step % self.config.training.logging_steps == 0:
                        self.log_step_metrics(step_metrics, batch_idx)
                else:
                    failed_steps += 1
                    error_type = step_metrics.get('error_type', 'unknown')
                    print(f"批次 {batch_idx} 处理失败: {error_type}")
                    
                    # 如果失败率过高，停止训练
                    if failed_steps > successful_steps and batch_idx > 10:
                        print(f"失败率过高 ({failed_steps}/{successful_steps + failed_steps})，停止训练")
                        break
                
            except Exception as e:
                print(f"训练步骤异常 (批次 {batch_idx}): {e}")
                failed_steps += 1
                continue
        
        # 计算epoch平均指标
        avg_metrics = epoch_metrics.compute()
        avg_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
        avg_metrics['successful_steps'] = successful_steps
        avg_metrics['failed_steps'] = failed_steps
        avg_metrics['success_rate'] = successful_steps / (successful_steps + failed_steps) if (successful_steps + failed_steps) > 0 else 0
        
        print(f"Epoch完成: 成功 {successful_steps}, 失败 {failed_steps}, 成功率 {avg_metrics['success_rate']:.2%}")
        
        return avg_metrics
    
    def train_epoch(self) -> Dict[str, float]:
        """传统训练一个epoch（单实例）"""
        self.model.train()
        
        epoch_metrics = TrainingMetrics()
        
        progress_bar = tqdm(
            self.data_loaders['train'],
            desc=f"Epoch {self.current_epoch + 1}/{self.config.training.num_epochs}"
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # 执行训练步骤
                step_metrics = self.training_step_fn(batch)
                
                # 更新统计
                if 'error' not in step_metrics:
                    epoch_metrics.update(step_metrics)
                    self.global_step += 1
                    
                    # 更新进度条
                    progress_bar.set_postfix({
                        'loss': f"{step_metrics.get('total_loss', 0):.4f}",
                        'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                    })
                    
                    # 记录步骤日志
                    if self.global_step % self.config.training.logging_steps == 0:
                        self.log_step_metrics(step_metrics, batch_idx)
                else:
                    print(f"批次 {batch_idx} 处理失败")
                    
            except Exception as e:
                print(f"训练步骤失败 (批次 {batch_idx}): {e}")
                continue
        
        # 计算epoch平均指标
        avg_metrics = epoch_metrics.compute()
        avg_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
        
        return avg_metrics
    
    def validate_multi_instance(self) -> Dict[str, float]:
        """多实例验证模型"""
        if 'val' not in self.data_loaders:
            return {}
        
        print("正在进行多实例验证...")
        self.model.eval()
        
        val_metrics = TrainingMetrics()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.data_loaders['val'], desc="Validating")):
                try:
                    # 🔧 使用多实例输入准备
                    inputs, targets = prepare_sam_inputs_multi_instance(batch)
                    
                    # 将数据移动到设备
                    for key, value in inputs.items():
                        if isinstance(value, torch.Tensor):
                            inputs[key] = value.to(self.device)
                        elif isinstance(value, list):
                            inputs[key] = [v.to(self.device) if isinstance(v, torch.Tensor) else v for v in value]
                    
                    for key, value in targets.items():
                        if isinstance(value, torch.Tensor):
                            targets[key] = value.to(self.device)
                    
                    # 前向传播
                    predictions = self.model(inputs)
                    
                    # 计算损失
                    loss_dict = self.loss_fn(predictions, targets)
                    val_metrics.update(loss_dict)
                    
                    # 🔧 处理多实例预测和目标用于指标计算
                    pred_masks = torch.sigmoid(predictions['masks']).cpu().numpy()  # [B, N_pred, H, W]
                    target_masks = targets['masks'].cpu().numpy()  # [B, N_target, H, W]
                    
                    # 为每个样本选择最佳预测和合并目标
                    for i in range(pred_masks.shape[0]):
                        # 选择IoU得分最高的预测掩码
                        if pred_masks.shape[1] > 1 and 'iou_predictions' in predictions:
                            iou_scores = predictions['iou_predictions'][i].cpu().numpy()
                            best_pred_idx = np.argmax(iou_scores)
                            pred_mask = pred_masks[i, best_pred_idx]
                        else:
                            pred_mask = pred_masks[i, 0] if pred_masks.shape[1] > 0 else np.zeros_like(target_masks[i, 0])
                        
                        # 合并所有目标实例（用于整体评估）
                        target_combined = (target_masks[i].sum(axis=0) > 0).astype(np.float32)
                        
                        all_predictions.append(pred_mask)
                        all_targets.append(target_combined)
                    
                    # 限制验证批次数量
                    if batch_idx >= 10:
                        break
                        
                except Exception as e:
                    print(f"验证步骤失败: {e}")
                    continue
        
        # 计算平均损失
        avg_val_metrics = val_metrics.compute()
        
        # 计算分割指标
        if all_predictions and all_targets:
            print(f"计算分割指标，样本数量={len(all_predictions)}")
            seg_metrics = self._compute_segmentation_metrics_multi_instance(all_predictions, all_targets)
            avg_val_metrics.update(seg_metrics)
        else:
            print("没有有效的预测/目标数据")
            avg_val_metrics.update({'val_iou': 0.0, 'val_dice': 0.0})
        
        return avg_val_metrics
    
    def validate(self) -> Dict[str, float]:
        """传统验证模型（单实例）"""
        if 'val' not in self.data_loaders:
            return {}
        
        print("正在验证...")
        self.model.eval()
        
        val_metrics = TrainingMetrics()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.data_loaders['val'], desc="Validating")):
                try:
                    # 准备输入和目标
                    inputs, targets = prepare_sam_inputs(batch)
                    
                    # 将数据移动到设备
                    for key, value in inputs.items():
                        if isinstance(value, torch.Tensor):
                            inputs[key] = value.to(self.device)
                        elif isinstance(value, list):
                            inputs[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
                    
                    for key, value in targets.items():
                        if isinstance(value, torch.Tensor):
                            targets[key] = value.to(self.device)
                    
                    # 前向传播
                    predictions = self.model(inputs)
                    
                    # 计算损失
                    loss_dict = self.loss_fn(predictions, targets)
                    val_metrics.update(loss_dict)
                    
                    # 收集预测和目标
                    pred_masks = torch.sigmoid(predictions['masks']).cpu().numpy()
                    target_masks = targets['masks'].cpu().numpy()
                    
                    for i in range(pred_masks.shape[0]):
                        pred_mask = pred_masks[i]
                        target_mask = target_masks[i]
                        
                        all_predictions.append(pred_mask)
                        all_targets.append(target_mask)
                    
                    # 只处理前几个批次用于验证
                    if batch_idx >= 5:
                        break
                        
                except Exception as e:
                    print(f"验证步骤失败: {e}")
                    continue
        
        # 计算平均损失
        avg_val_metrics = val_metrics.compute()
        
        # 计算分割指标
        if all_predictions and all_targets:
            seg_metrics = self._compute_segmentation_metrics(all_predictions, all_targets)
            avg_val_metrics.update(seg_metrics)
        else:
            avg_val_metrics.update({'val_iou': 0.0, 'val_dice': 0.0})
        
        return avg_val_metrics
    
    def _compute_segmentation_metrics_multi_instance(self, predictions: List, targets: List) -> Dict[str, float]:
        """计算多实例分割指标"""
        try:
            all_ious = []
            all_dices = []
            
            for pred, target in zip(predictions, targets):
                # 处理维度
                if pred.ndim > 2:
                    pred = pred[0] if pred.shape[0] == 1 else pred.mean(axis=0)
                if target.ndim > 2:
                    target = target[0] if target.shape[0] == 1 else target.mean(axis=0)
                
                # 二值化
                pred_binary = (pred > 0.5).astype(np.int32)
                target_binary = (target > 0.5).astype(np.int32)
                
                # 计算IoU和Dice
                intersection = np.sum(pred_binary * target_binary)
                pred_sum = np.sum(pred_binary)
                target_sum = np.sum(target_binary)
                union = pred_sum + target_sum - intersection
                
                if union > 0:
                    iou = intersection / union
                    dice = 2 * intersection / (pred_sum + target_sum)
                    all_ious.append(iou)
                    all_dices.append(dice)
                else:
                    # 如果都是空的，认为完全匹配
                    if pred_sum == 0 and target_sum == 0:
                        all_ious.append(1.0)
                        all_dices.append(1.0)
                    else:
                        all_ious.append(0.0)
                        all_dices.append(0.0)
            
            avg_iou = np.mean(all_ious) if all_ious else 0.0
            avg_dice = np.mean(all_dices) if all_dices else 0.0
            
            print(f"多实例验证完成，平均IoU={avg_iou:.4f}, 平均Dice={avg_dice:.4f}")
            
            return {
                'val_iou': avg_iou,
                'val_dice': avg_dice,
                'val_samples': len(all_ious)
            }
            
        except Exception as e:
            print(f"多实例分割指标计算失败: {e}")
            import traceback
            traceback.print_exc()
            return {'val_iou': 0.0, 'val_dice': 0.0}
    
    def _compute_segmentation_metrics(self, predictions: List, targets: List) -> Dict[str, float]:
        """计算分割指标（传统版本）"""
        try:
            all_ious = []
            all_dices = []
            
            for pred, target in zip(predictions, targets):
                # 处理维度
                if pred.ndim > 2:
                    pred = pred[0] if pred.shape[0] == 1 else pred.mean(axis=0)
                if target.ndim > 2:
                    target = target[0] if target.shape[0] == 1 else target.mean(axis=0)
                
                # 尺寸匹配
                if pred.shape != target.shape:
                    import torch.nn.functional as F
                    import torch
                    
                    pred_tensor = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
                    target_size = target.shape
                    
                    pred_resized = F.interpolate(
                        pred_tensor, 
                        size=target_size, 
                        mode='bilinear',
                        align_corners=False
                    )
                    pred = pred_resized.squeeze().numpy()
                
                # 二值化
                pred_binary = (pred > 0.5).astype(np.int32)
                target_binary = (target > 0.5).astype(np.int32)
                
                # 计算IoU和Dice
                intersection = np.sum(pred_binary * target_binary)
                union = np.sum(pred_binary) + np.sum(target_binary) - intersection
                
                if union > 0:
                    iou = intersection / union
                    dice = 2 * intersection / (np.sum(pred_binary) + np.sum(target_binary))
                    
                    all_ious.append(iou)
                    all_dices.append(dice)
                else:
                    all_ious.append(0.0)
                    all_dices.append(0.0)
            
            avg_iou = np.mean(all_ious) if all_ious else 0.0
            avg_dice = np.mean(all_dices) if all_dices else 0.0
            
            print(f"验证完成，平均IoU={avg_iou:.4f}, 平均Dice={avg_dice:.4f}")
            
            return {
                'val_iou': avg_iou,
                'val_dice': avg_dice
            }
            
        except Exception as e:
            print(f"分割指标计算失败: {e}")
            import traceback
            traceback.print_exc()
            return {'val_iou': 0.0, 'val_dice': 0.0}
    
    def log_metrics(self, train_metrics: Dict[str, float], 
                   val_metrics: Dict[str, float], epoch: int):
        """记录指标"""
        # TensorBoard日志
        if self.writer:
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'train/{key}', value, epoch)
            
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)
        
        # Weights & Biases日志
        if self.config.experiment.use_wandb:
            log_dict = {f'train/{k}': v for k, v in train_metrics.items()}
            log_dict.update({f'val/{k}': v for k, v in val_metrics.items()})
            log_dict['epoch'] = epoch
            wandb.log(log_dict)
        
        # 控制台输出
        print(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs} - 指标摘要:")
        print("训练指标:")
        for key, value in train_metrics.items():
            if key in ['successful_steps', 'failed_steps']:
                print(f"  {key}: {int(value)}")
            elif key == 'success_rate':
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value:.4f}")
        
        if val_metrics:
            print("验证指标:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")
    
    def log_step_metrics(self, step_metrics: Dict[str, float], step: int):
        """记录步骤指标"""
        if self.writer:
            for key, value in step_metrics.items():
                if key not in ['error', 'success']:
                    self.writer.add_scalar(f'train/step_{key}', value, self.global_step)
        
        if self.config.experiment.use_wandb:
            log_dict = {f'train/step_{k}': v for k, v in step_metrics.items() 
                       if k not in ['error', 'success']}
            log_dict['global_step'] = self.global_step
            wandb.log(log_dict)
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """保存检查点"""
        checkpoint_dir = self.config.get_checkpoint_dir()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': self.config.to_dict(),
            'metrics': metrics,
            'use_multi_instance': self.use_multi_instance
        }
        
        # 保存当前检查点
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        current_metric = metrics.get('avg_total_loss', float('inf'))
        if current_metric < self.best_metric:
            self.best_metric = current_metric
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型 (loss: {current_metric:.4f})")
        
        # 保存LoRA权重
        lora_save_path = checkpoint_dir / f"lora_epoch_{epoch + 1}"
        self.model.save_lora_weights(lora_save_path)
        
        print(f"检查点已保存: {checkpoint_path}")
    
    def check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """检查早停条件"""
        if not metrics or 'avg_total_loss' not in metrics:
            return False
        
        current_metric = metrics['avg_total_loss']
        
        if current_metric < self.best_metric:
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        
        return self.early_stopping_counter >= self.config.training.early_stopping_patience
    
    def save_final_model(self):
        """保存最终模型"""
        # 保存LoRA权重
        final_dir = self.output_dir / "final_model"
        self.model.save_lora_weights(final_dir)
        
        # 保存合并后的完整模型
        merged_path = final_dir / "merged_sam_model.pth"
        self.model.merge_and_save_full_model(merged_path)
        
        # 保存训练摘要
        summary = {
            'training_completed': True,
            'total_epochs': self.current_epoch + 1,
            'total_steps': self.global_step,
            'best_metric': self.best_metric,
            'model_type': self.config.model.base_model_name,
            'use_multi_instance': self.use_multi_instance,
            'lora_config': {
                'rank': self.config.lora.rank,
                'alpha': self.config.lora.alpha,
                'dropout': self.config.lora.dropout
            },
            'config': self.config.to_dict()
        }
        
        with open(final_dir / "training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"最终模型已保存到: {final_dir}")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """加载检查点"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.current_epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.best_metric = checkpoint['best_metric']
            
            # 恢复多实例设置
            if 'use_multi_instance' in checkpoint:
                self.use_multi_instance = checkpoint['use_multi_instance']
            
            print(f"检查点加载成功: {checkpoint_path}")
            print(f"从第 {self.current_epoch + 1} 轮继续训练")
            print(f"使用多实例模式: {self.use_multi_instance}")
            
            return True
            
        except Exception as e:
            print(f"加载检查点失败: {e}")
            return False
    
    def evaluate_model(self, test_data_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """评估模型性能"""
        if test_data_loader is None:
            test_data_loader = self.data_loaders.get('test')
        
        if test_data_loader is None:
            print("没有测试数据，跳过评估")
            return {}
        
        self.model.eval()
        test_metrics = TrainingMetrics()
        all_predictions = []
        all_targets = []
        
        print("开始模型评估...")
        
        with torch.no_grad():
            for batch in tqdm(test_data_loader, desc="Evaluating"):
                try:
                    # 根据模式选择输入准备函数
                    if self.use_multi_instance:
                        inputs, targets = prepare_sam_inputs_multi_instance(batch)
                    else:
                        inputs, targets = prepare_sam_inputs(batch)
                    
                    # 将数据移动到设备
                    for key, value in inputs.items():
                        if isinstance(value, torch.Tensor):
                            inputs[key] = value.to(self.device)
                        elif isinstance(value, list):
                            inputs[key] = [v.to(self.device) if isinstance(v, torch.Tensor) else v for v in value]
                    
                    for key, value in targets.items():
                        if isinstance(value, torch.Tensor):
                            targets[key] = value.to(self.device)
                    
                    # 前向传播
                    predictions = self.model(inputs)
                    
                    # 计算损失
                    loss_dict = self.loss_fn(predictions, targets)
                    test_metrics.update(loss_dict)
                    
                    # 收集预测和目标
                    pred_masks = torch.sigmoid(predictions['masks']).cpu().numpy()
                    target_masks = targets['masks'].cpu().numpy()
                    
                    if self.use_multi_instance:
                        # 多实例处理
                        for i in range(pred_masks.shape[0]):
                            if pred_masks.shape[1] > 1 and 'iou_predictions' in predictions:
                                iou_scores = predictions['iou_predictions'][i].cpu().numpy()
                                best_pred_idx = np.argmax(iou_scores)
                                pred_mask = pred_masks[i, best_pred_idx]
                            else:
                                pred_mask = pred_masks[i, 0] if pred_masks.shape[1] > 0 else np.zeros_like(target_masks[i, 0])
                            
                            target_combined = (target_masks[i].sum(axis=0) > 0).astype(np.float32)
                            all_predictions.append(pred_mask)
                            all_targets.append(target_combined)
                    else:
                        # 单实例处理
                        all_predictions.extend(pred_masks)
                        all_targets.extend(target_masks)
                    
                except Exception as e:
                    print(f"评估步骤失败: {e}")
                    continue
        
        # 计算最终指标
        final_metrics = test_metrics.compute()
        
        # 计算分割指标
        if all_predictions and all_targets:
            if self.use_multi_instance:
                seg_metrics = self._compute_segmentation_metrics_multi_instance(all_predictions, all_targets)
            else:
                seg_metrics = self._compute_segmentation_metrics(all_predictions, all_targets)
            final_metrics.update(seg_metrics)
        
        print("评估完成:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return final_metrics


def create_trainer_from_config(config_path: str) -> LoRATrainer:
    """从配置文件创建训练器"""
    config = LoRATrainingSettings.from_json(config_path)
    return LoRATrainer(config)


def resume_training(checkpoint_path: str, config_path: Optional[str] = None) -> LoRATrainer:
    """恢复训练"""
    if config_path:
        trainer = create_trainer_from_config(config_path)
    else:
        # 从检查点目录加载配置
        checkpoint_dir = Path(checkpoint_path).parent
        config_file = checkpoint_dir.parent / "config.json"
        if config_file.exists():
            trainer = create_trainer_from_config(str(config_file))
        else:
            print("无法找到配置文件，使用默认配置")
            from config.lora_config import LoRATrainingSettings
            trainer = LoRATrainer(LoRATrainingSettings())
    
    trainer.load_checkpoint(checkpoint_path)
    return trainer


def create_multi_instance_trainer(data_dir: str, 
                                model_type: str = "vit_b_lm",
                                batch_size: int = 4,
                                epochs: int = 15,
                                learning_rate: float = 1e-4,
                                rank: int = 8,
                                output_dir: str = "./multi_instance_experiments") -> LoRATrainer:
    """快速创建多实例训练器"""
    
    from config.lora_config import LoRATrainingSettings
    
    config = LoRATrainingSettings()
    
    # 数据配置
    config.data.train_data_dir = data_dir
    config.data.batch_size = batch_size
    config.data.max_objects_per_image = 20
    config.data.use_data_augmentation = True
    
    # 模型配置
    config.model.base_model_name = model_type
    config.model.apply_lora_to = ['image_encoder']
    
    # LoRA配置
    config.lora.rank = rank
    config.lora.alpha = rank * 2.0
    config.lora.dropout = 0.1
    
    # 训练配置
    config.training.num_epochs = epochs
    config.training.learning_rate = learning_rate
    config.training.batch_size = batch_size
    config.training.save_steps = 200
    config.training.eval_steps = 50
    config.training.logging_steps = 10
    
    # 实验配置
    config.experiment.output_dir = output_dir
    config.experiment.experiment_name = f"multi_instance_{model_type}_r{rank}"
    
    trainer = LoRATrainer(config)
    trainer.use_multi_instance = True  # 确保启用多实例
    
    return trainer


# 使用示例
if __name__ == "__main__":
    # 创建多实例训练器并开始训练
    trainer = create_multi_instance_trainer(
        data_dir="/path/to/your/cell/data",
        model_type="vit_b_lm",
        batch_size=4,
        epochs=15,
        learning_rate=1e-4,
        rank=8
    )
    
    success = trainer.train()
    
    if success:
        print("🎉 多实例SAM LoRA训练成功完成!")
    else:
        print("❌ 训练失败")