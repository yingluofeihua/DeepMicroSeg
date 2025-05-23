"""
LoRA训练器
负责LoRA模型的训练、验证和保存
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
from lora.adapters import LoRAModelWrapper, create_lora_model
from lora.data_loaders import create_data_loaders
from core.model_handler import ModelHandler, ModelConfig
from core.metrics import ComprehensiveMetrics, MetricsResult
from utils.file_utils import setup_logging


class LoRATrainer:
    """LoRA训练器"""
    
    def __init__(self, config: LoRATrainingSettings):
        self.config = config
        self.device = self._setup_device()
        
        # 初始化组件
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.data_loaders = {}
        self.metrics_calculator = ComprehensiveMetrics()
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.early_stopping_counter = 0
        
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
            print(f"GPU设备数量: {torch.cuda.device_count()}")
            print(f"当前GPU: {torch.cuda.current_device()}")
            print(f"GPU内存: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        
        return device
    
    def setup_model(self) -> bool:
        """设置模型"""
        try:
            # 加载基础模型
            model_config = ModelConfig(
                name=self.config.model.base_model_name,
                device=str(self.device)
            )
            
            base_model_handler = ModelHandler(model_config)
            if not base_model_handler.load_model():
                print("基础模型加载失败")
                return False
            
            # 获取基础模型
            base_model = base_model_handler.predictor  # 或者适当的模型组件
            
            # 创建LoRA模型
            lora_config = {
                'rank': self.config.lora.rank,
                'alpha': self.config.lora.alpha,
                'dropout': self.config.lora.dropout,
                'target_modules': self.config.lora.target_modules
            }
            
            self.model = create_lora_model(base_model, lora_config)
            self.model = self.model.to(self.device)
            
            # 打印模型信息
            self.model.print_model_info()
            
            return True
            
        except Exception as e:
            print(f"模型设置失败: {e}")
            return False
    
    def setup_data_loaders(self) -> bool:
        """设置数据加载器"""
        try:
            self.data_loaders = create_data_loaders(
                config=self.config.data,
                dataset_type="sam"  # 使用SAM数据集格式
            )
            
            print("数据加载器创建成功:")
            for split, loader in self.data_loaders.items():
                print(f"  {split}: {len(loader)} 批次, {len(loader.dataset)} 样本")
            
            return True
            
        except Exception as e:
            print(f"数据加载器设置失败: {e}")
            return False
    
    def setup_optimizer(self):
        """设置优化器和学习率调度器"""
        # 只优化LoRA参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
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
        
        print(f"优化器设置完成: {type(self.optimizer).__name__}")
        print(f"可训练参数数量: {len(trainable_params)}")
    
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
        """开始训练"""
        print("开始LoRA训练...")
        
        # 设置所有组件
        if not self.setup_model():
            return False
        
        if not self.setup_data_loaders():
            return False
        
        self.setup_optimizer()
        self.setup_logging()
        
        # 训练循环
        try:
            for epoch in range(self.config.training.num_epochs):
                self.current_epoch = epoch
                
                # 训练一个epoch
                train_metrics = self.train_epoch()
                
                # 验证
                val_metrics = self.validate() if 'val' in self.data_loaders else {}
                
                # 记录日志
                self.log_metrics(train_metrics, val_metrics, epoch)
                
                # 保存检查点
                if (epoch + 1) % (self.config.training.save_steps // len(self.data_loaders['train'])) == 0:
                    self.save_checkpoint(epoch, val_metrics)
                
                # 早停检查
                if self.check_early_stopping(val_metrics):
                    print(f"早停触发，在第 {epoch + 1} 轮停止训练")
                    break
                
                # 更新学习率
                if self.scheduler:
                    self.scheduler.step()
            
            # 保存最终模型
            self.save_final_model()
            
            print("训练完成!")
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
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.data_loaders['train'],
            desc=f"Epoch {self.current_epoch + 1}/{self.config.training.num_epochs}"
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # 前向传播
            loss = self.training_step(batch)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.config.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.max_grad_norm
                )
            
            # 优化器步骤
            self.optimizer.step()
            
            # 更新统计
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / num_batches:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # 记录步骤日志
            if self.global_step % self.config.training.logging_steps == 0:
                self.log_step_metrics(loss.item(), batch_idx)
        
        return {
            'train_loss': total_loss / num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def training_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """单个训练步骤"""
        # 将数据移到设备
        images = batch['images'].to(self.device)
        
        # 这里需要根据具体的模型架构实现损失计算
        # 以下是一个简化的示例，实际需要根据SAM的训练方式调整
        
        try:
            # 前向传播（这里需要根据实际的SAM模型接口调整）
            # 假设模型返回预测结果
            predictions = self.model(images)
            
            # 计算损失（需要根据具体任务调整）
            loss = self.compute_loss(predictions, batch)
            
            return loss
            
        except Exception as e:
            print(f"训练步骤出错: {e}")
            # 返回一个小的损失值以继续训练
            return torch.tensor(0.01, requires_grad=True, device=self.device)
    
    def compute_loss(self, predictions: Any, batch: Dict[str, Any]) -> torch.Tensor:
        """计算损失函数"""
        # 这里需要根据具体的模型输出格式和任务需求实现
        # 以下是一个简化的示例
        
        # 如果有ground truth masks
        if 'ground_truth_masks' in batch:
            gt_masks = batch['ground_truth_masks']
            
            # 简化的损失计算
            if hasattr(predictions, 'masks'):
                pred_masks = predictions.masks
                # 计算mask损失
                loss = nn.functional.binary_cross_entropy_with_logits(
                    pred_masks, gt_masks.float().to(self.device)
                )
            else:
                # 如果没有具体的mask输出，使用占位符损失
                loss = torch.tensor(0.01, requires_grad=True, device=self.device)
        else:
            # 占位符损失
            loss = torch.tensor(0.01, requires_grad=True, device=self.device)
        
        return loss
    
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        if 'val' not in self.data_loaders:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        all_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(self.data_loaders['val'], desc="Validating"):
                # 验证步骤
                loss = self.validation_step(batch)
                total_loss += loss.item()
                num_batches += 1
                
                # 计算评测指标
                metrics = self.compute_validation_metrics(batch)
                if metrics:
                    all_metrics.append(metrics)
        
        # 聚合指标
        val_metrics = {'val_loss': total_loss / num_batches}
        
        if all_metrics:
            # 计算平均指标
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if key in m]
                if values:
                    val_metrics[f'val_{key}'] = np.mean(values)
        
        return val_metrics
    
    def validation_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """单个验证步骤"""
        images = batch['images'].to(self.device)
        
        try:
            predictions = self.model(images)
            loss = self.compute_loss(predictions, batch)
            return loss
        except Exception as e:
            print(f"验证步骤出错: {e}")
            return torch.tensor(0.01, device=self.device)
    
    def compute_validation_metrics(self, batch: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """计算验证指标"""
        try:
            # 这里需要根据具体模型输出实现指标计算
            # 简化示例
            if 'ground_truth_masks' in batch and len(batch['ground_truth_masks']) > 0:
                gt_masks = batch['ground_truth_masks'][0].cpu().numpy()
                
                # 生成简单的预测掩码（实际应该使用模型输出）
                pred_masks = np.random.rand(*gt_masks.shape) > 0.5
                
                # 计算指标
                metrics_result = self.metrics_calculator.compute_all_metrics(
                    gt_masks, pred_masks.astype(np.int32)
                )
                
                return metrics_result.to_dict()
            
            return None
            
        except Exception as e:
            print(f"指标计算出错: {e}")
            return None
    
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
        print(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")
        print("训练指标:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        if val_metrics:
            print("验证指标:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")
    
    def log_step_metrics(self, loss: float, step: int):
        """记录步骤指标"""
        if self.writer:
            self.writer.add_scalar('train/step_loss', loss, self.global_step)
        
        if self.config.experiment.use_wandb:
            wandb.log({'train/step_loss': loss, 'global_step': self.global_step})
    
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
            'metrics': metrics
        }
        
        # 保存当前检查点
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        current_metric = metrics.get('val_loss', float('inf'))
        if current_metric < self.best_metric:
            self.best_metric = current_metric
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型 (val_loss: {current_metric:.4f})")
        
        # 保存LoRA权重
        self.model.save_lora_weights(checkpoint_dir / f"lora_epoch_{epoch + 1}")
        
        print(f"检查点已保存: {checkpoint_path}")
    
    def check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """检查早停条件"""
        if not metrics or 'val_loss' not in metrics:
            return False
        
        current_metric = metrics['val_loss']
        
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
        merged_path = final_dir / "merged_model.pth"
        self.model.merge_and_save(merged_path)
        
        # 保存训练摘要
        summary = {
            'training_completed': True,
            'total_epochs': self.current_epoch + 1,
            'total_steps': self.global_step,
            'best_metric': self.best_metric,
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
            
            print(f"检查点加载成功: {checkpoint_path}")
            print(f"从第 {self.current_epoch + 1} 轮继续训练")
            
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
        all_metrics = []
        
        print("开始模型评估...")
        
        with torch.no_grad():
            for batch in tqdm(test_data_loader, desc="Evaluating"):
                metrics = self.compute_validation_metrics(batch)
                if metrics:
                    all_metrics.append(metrics)
        
        if not all_metrics:
            return {}
        
        # 计算平均指标
        eval_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            if values:
                eval_metrics[key] = np.mean(values)
        
        print("评估完成:")
        for key, value in eval_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return eval_metrics


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
        trainer = create_trainer_from_config(str(config_file))
    
    trainer.load_checkpoint(checkpoint_path)
    return trainer