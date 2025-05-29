"""
LoRA训练主入口文件 (增强版)
集成完整的LoRA微调、评测和批量推理测试功能
支持数据集划分、详细评测报告和性能分析
"""

import sys
import argparse
from pathlib import Path
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from config.lora_config import LoRATrainingSettings, LORA_PRESET_CONFIGS, get_config_for_model
from core.lora_trainer import LoRATrainer, create_trainer_from_config, resume_training
from core.evaluator import BatchEvaluator
from core.dataset_manager import DatasetManager
from config.settings import BatchEvaluationSettings
from lora.data_loaders import split_dataset, list_cached_splits, clean_old_splits, preview_data_split
from utils.file_utils import setup_logging
from utils.model_utils import get_device_info, optimize_memory
from utils.data_splitter import DatasetSplitter, print_split_summary, DataSplit

# 🔧 新增导入 - 评测相关
from lora.stable_sam_lora_wrapper import load_stable_sam_lora_model
from core.metrics import ComprehensiveMetrics
from lora.data_loaders import SAMDataset, collate_fn
from config.lora_config import DataConfig


class EnhancedLoRAEvaluator:
    """增强版LoRA评测器 - 集成到lora_main.py"""
    
    def __init__(self, lora_model_path: str, device: str = "auto"):
        self.lora_model_path = Path(lora_model_path)
        self.device = self._setup_device(device)
        self.model = None
        self.metrics_calculator = ComprehensiveMetrics(enable_hd95=True)
        
    def _setup_device(self, device: str) -> torch.device:
        """设置计算设备"""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)
    
    def load_model(self) -> bool:
        """加载LoRA模型"""
        try:
            print(f"正在加载LoRA模型: {self.lora_model_path}")
            
            # 自动检测模型类型
            config_file = self.lora_model_path / "sam_lora_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                model_type = config.get('model_type', 'vit_b_lm')
            else:
                model_type = 'vit_b_lm'
                print("未找到配置文件，使用默认模型类型: vit_b_lm")
            
            self.model = load_stable_sam_lora_model(model_type, str(self.lora_model_path), str(self.device))
            
            if self.model is None:
                print("❌ LoRA模型加载失败")
                return False
            
            self.model.eval()
            print(f"✅ LoRA模型加载成功，设备: {self.device}")
            return True
            
        except Exception as e:
            print(f"加载LoRA模型失败: {e}")
            return False
    
    def evaluate_with_split_file(self, split_file: str, args) -> dict:
        """使用数据划分文件进行评测"""
        print(f"使用数据划分文件进行评测: {split_file}")
        
        # 加载数据划分
        try:
            with open(split_file, 'r', encoding='utf-8') as f:
                split_data = json.load(f)
            
            split_result = DataSplit.from_dict(split_data)
            test_samples = split_result.test_samples
            
            print(f"测试集样本数: {len(test_samples)}")
            
        except Exception as e:
            print(f"加载数据划分文件失败: {e}")
            return {}
        
        # 验证并过滤有效样本
        valid_samples = []
        for sample in test_samples:
            img_path = Path(sample['image_path'])
            mask_path = Path(sample['mask_path'])
            if img_path.exists() and mask_path.exists():
                valid_samples.append(sample)
        
        print(f"有效测试样本数: {len(valid_samples)}")
        
        if not valid_samples:
            print("❌ 没有有效的测试样本")
            return {}
        
        # 限制评测数量
        max_samples = getattr(args, 'max_samples', None)
        if max_samples and max_samples < len(valid_samples):
            valid_samples = valid_samples[:max_samples]
            print(f"限制评测样本数为: {max_samples}")
        
        return self._run_evaluation(valid_samples, args)
    
    def _run_evaluation(self, test_samples: list, args) -> dict:
        """执行评测"""
        if not self.model:
            if not self.load_model():
                return {}
        
        # 创建数据集
        config = DataConfig()
        config.batch_size = getattr(args, 'eval_batch_size', 1)
        
        test_dataset = SAMDataset(
            data_dir=None,
            config=config,
            split='test',
            samples=test_samples
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn
        )
        
        print("开始模型评测...")
        
        results = []
        total_processing_time = 0.0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="评测进度")):
                try:
                    # 准备输入
                    from lora.training_utils import prepare_sam_inputs
                    
                    inputs, targets = prepare_sam_inputs(batch)
                    
                    # 移动到设备
                    for key, value in inputs.items():
                        if isinstance(value, torch.Tensor):
                            inputs[key] = value.to(self.device)
                        elif isinstance(value, list):
                            inputs[key] = [v.to(self.device) if isinstance(v, torch.Tensor) else v for v in value]
                    
                    for key, value in targets.items():
                        if isinstance(value, torch.Tensor):
                            targets[key] = value.to(self.device)
                    
                    # 计算处理时间
                    start_time = time.time()
                    
                    # 模型推理
                    predictions = self.model(inputs)
                    
                    processing_time = time.time() - start_time
                    total_processing_time += processing_time
                    
                    # 后处理预测结果
                    pred_masks = torch.sigmoid(predictions['masks']).cpu().numpy()
                    target_masks = targets['masks'].cpu().numpy()
                    
                    # 获取样本信息
                    sample_ids = batch.get('sample_ids', [f"sample_{batch_idx}"])
                    
                    # 处理每个批次中的样本
                    for i in range(pred_masks.shape[0]):
                        result = self._process_single_prediction(
                            pred_masks[i], target_masks[i], 
                            sample_ids[i] if i < len(sample_ids) else f"sample_{batch_idx}_{i}",
                            processing_time / pred_masks.shape[0],
                            test_samples[batch_idx * pred_masks.shape[0] + i] if test_samples else {}
                        )
                        
                        if result:
                            results.append(result)
                    
                except Exception as e:
                    print(f"处理批次 {batch_idx} 失败: {e}")
                    continue
        
        # 生成评测报告
        return self._generate_evaluation_report(results, total_processing_time, args)
    
    def _process_single_prediction(self, pred_mask: np.ndarray, target_mask: np.ndarray,
                                 sample_id: str, processing_time: float,
                                 sample_info: dict) -> dict:
        """处理单个预测结果"""
        try:
            # 处理预测掩码
            if len(pred_mask.shape) > 2:
                pred_mask = pred_mask[0] if pred_mask.shape[0] > 0 else np.zeros(pred_mask.shape[1:])
            
            # 处理目标掩码
            if len(target_mask.shape) > 2:
                target_mask = (target_mask.sum(axis=0) > 0).astype(float)
            
            # 调整尺寸匹配（如果需要）
            if pred_mask.shape != target_mask.shape:
                import torch.nn.functional as F
                pred_tensor = torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0).float()
                target_size = target_mask.shape
                pred_resized = F.interpolate(pred_tensor, size=target_size, mode='bilinear', align_corners=False)
                pred_mask = pred_resized.squeeze().numpy()
            
            # 二值化
            pred_binary = (pred_mask > 0.5).astype(int)
            target_binary = (target_mask > 0.5).astype(int)
            
            # 计算指标
            metrics_result = self.metrics_calculator.compute_all_metrics(target_binary, pred_binary)
            
            # 构建结果字典
            result = metrics_result.to_dict()
            result.update({
                'sample_id': sample_id,
                'processing_time': processing_time,
                'cell_type': sample_info.get('cell_type', 'unknown'),
                'date': sample_info.get('date', 'unknown'),
                'magnification': sample_info.get('magnification', 'unknown'),
                'dataset_id': sample_info.get('dataset_id', 'unknown'),
                'image_path': sample_info.get('image_path', ''),
                'mask_path': sample_info.get('mask_path', '')
            })
            
            return result
            
        except Exception as e:
            print(f"处理单个预测失败 {sample_id}: {e}")
            return None
    
    def _generate_evaluation_report(self, results: list, total_processing_time: float, args) -> dict:
        """生成评测报告"""
        if not results:
            print("❌ 没有有效的评测结果")
            return {}
        
        # 创建输出目录
        output_dir = getattr(args, 'eval_output', None)
        if output_dir:
            output_dir = Path(output_dir)
        else:
            output_dir = self.lora_model_path.parent / "evaluation_results"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存详细结果
        if getattr(args, 'save_detailed', True):
            df = pd.DataFrame(results)
            detailed_file = output_dir / "detailed_evaluation_results.csv"
            df.to_csv(detailed_file, index=False)
            print(f"详细结果已保存: {detailed_file}")
        
        # 计算平均指标
        avg_metrics = self._calculate_average_metrics(results)
        
        # 添加元数据
        avg_metrics.update({
            'model_path': str(self.lora_model_path),
            'total_samples': len(results),
            'total_processing_time': total_processing_time,
            'average_processing_time_per_sample': total_processing_time / len(results),
            'device': str(self.device),
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # 保存摘要结果
        summary_file = output_dir / "evaluation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(avg_metrics, f, indent=2, ensure_ascii=False, default=convert_numpy)
        
        print(f"评测摘要已保存: {summary_file}")
        
        # 打印结果摘要
        self._print_evaluation_summary(avg_metrics)
        
        return avg_metrics
    
    def _calculate_average_metrics(self, results: list) -> dict:
        """计算平均指标"""
        metrics_cols = ['ap50', 'ap75', 'iou_score', 'dice_score', 'hd95', 
                       'gt_instances', 'pred_instances', 'processing_time']
        
        avg_metrics = {}
        df = pd.DataFrame(results)
        
        for col in metrics_cols:
            if col in df.columns:
                if col == 'hd95':
                    finite_values = df[col][np.isfinite(df[col])]
                    avg_metrics[f'avg_{col}'] = finite_values.mean() if len(finite_values) > 0 else float('inf')
                    avg_metrics[f'median_{col}'] = finite_values.median() if len(finite_values) > 0 else float('inf')
                    avg_metrics[f'finite_count_{col}'] = len(finite_values)
                    avg_metrics[f'infinite_count_{col}'] = len(df) - len(finite_values)
                else:
                    avg_metrics[f'avg_{col}'] = df[col].mean()
                    avg_metrics[f'std_{col}'] = df[col].std()
                    avg_metrics[f'median_{col}'] = df[col].median()
        
        # 按细胞类型统计
        if 'cell_type' in df.columns:
            cell_type_stats = {}
            for cell_type in df['cell_type'].unique():
                if cell_type != 'unknown':
                    cell_data = df[df['cell_type'] == cell_type]
                    cell_stats = {}
                    for metric in ['ap50', 'ap75', 'iou_score', 'dice_score']:
                        if metric in cell_data.columns:
                            cell_stats[metric] = {
                                'mean': cell_data[metric].mean(),
                                'std': cell_data[metric].std(),
                                'count': len(cell_data)
                            }
                    cell_type_stats[cell_type] = cell_stats
            
            avg_metrics['by_cell_type'] = cell_type_stats
        
        return avg_metrics
    
    def _print_evaluation_summary(self, metrics: dict):
        """打印评测摘要"""
        print(f"\n{'='*60}")
        print("LoRA模型评测结果摘要")
        print(f"{'='*60}")
        
        print(f"模型路径: {metrics.get('model_path', 'N/A')}")
        print(f"评测样本数: {metrics.get('total_samples', 0)}")
        print(f"总处理时间: {metrics.get('total_processing_time', 0):.2f}s")
        print(f"平均处理时间: {metrics.get('average_processing_time_per_sample', 0):.4f}s/样本")
        
        print(f"\n主要性能指标:")
        key_metrics = ['avg_ap50', 'avg_ap75', 'avg_iou_score', 'avg_dice_score']
        for metric in key_metrics:
            if metric in metrics:
                value = metrics[metric]
                std_key = metric.replace('avg_', 'std_')
                std_value = metrics.get(std_key, 0)
                print(f"  {metric.replace('avg_', '').upper()}: {value:.4f} ± {std_value:.4f}")
        
        # HD95特殊处理
        if 'avg_hd95' in metrics:
            hd95_val = metrics['avg_hd95']
            finite_count = metrics.get('finite_count_hd95', 0)
            total_count = metrics.get('total_samples', 0)
            if hd95_val == float('inf'):
                print(f"  HD95: ∞ (所有值都是无穷)")
            else:
                print(f"  HD95: {hd95_val:.4f} (基于 {finite_count}/{total_count} 个有效值)")
        
        # 按细胞类型显示
        if 'by_cell_type' in metrics:
            print(f"\n按细胞类型统计:")
            for cell_type, stats in metrics['by_cell_type'].items():
                print(f"  {cell_type}:")
                for metric, values in stats.items():
                    print(f"    {metric.upper()}: {values['mean']:.4f} ± {values['std']:.4f} (n={values['count']})")
        
        print(f"{'='*60}")
    
    def batch_inference_test(self, test_samples: list, batch_sizes: list = [1, 2, 4, 8]) -> dict:
        """批量推理性能测试"""
        print(f"\n🚀 开始批量推理性能测试")
        
        if not self.model:
            if not self.load_model():
                return {}
        
        # 限制样本数量用于性能测试
        test_samples = test_samples[:min(50, len(test_samples))]
        
        batch_results = {}
        
        for batch_size in batch_sizes:
            print(f"\n测试批次大小: {batch_size}")
            
            if batch_size > len(test_samples):
                print(f"批次大小 {batch_size} 超过可用样本数 {len(test_samples)}，跳过")
                continue
            
            try:
                # 创建数据加载器
                config = DataConfig()
                config.batch_size = batch_size
                
                test_dataset = SAMDataset(
                    data_dir=None,
                    config=config,
                    split='test',
                    samples=test_samples[:batch_size * 3]  # 限制样本数
                )
                
                test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=1,
                    collate_fn=collate_fn
                )
                
                # 性能测试
                times = []
                
                with torch.no_grad():
                    for batch_idx, batch in enumerate(test_loader):
                        if batch_idx >= 3:  # 只测试前几个批次
                            break
                        
                        from lora.training_utils import prepare_sam_inputs
                        inputs, targets = prepare_sam_inputs(batch)
                        
                        # 移动到设备
                        for key, value in inputs.items():
                            if isinstance(value, torch.Tensor):
                                inputs[key] = value.to(self.device)
                            elif isinstance(value, list):
                                inputs[key] = [v.to(self.device) if isinstance(v, torch.Tensor) else v for v in value]
                        
                        # 计时
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        start_time = time.time()
                        predictions = self.model(inputs)
                        
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        end_time = time.time()
                        times.append(end_time - start_time)
                
                # 统计结果
                if times:
                    avg_time = np.mean(times)
                    time_per_image = avg_time / batch_size
                    throughput = batch_size / avg_time
                    
                    batch_results[batch_size] = {
                        'avg_batch_time': avg_time,
                        'time_per_image': time_per_image,
                        'throughput': throughput,
                        'num_batches_tested': len(times)
                    }
                    
                    print(f"  平均批次时间: {avg_time:.4f}s")
                    print(f"  每图像时间: {time_per_image:.4f}s")
                    print(f"  吞吐量: {throughput:.2f} 图像/秒")
                
            except Exception as e:
                print(f"批次大小 {batch_size} 测试失败: {e}")
                batch_results[batch_size] = {'error': str(e)}
        
        return batch_results


def parse_arguments():
    """解析命令行参数 - 增强版"""
    parser = argparse.ArgumentParser(
        description="SAM LoRA微调训练系统 (增强版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 快速训练
  python lora_main.py train --preset quick --data-dir /path/to/data
  
  # 使用数据划分文件评测
  python lora_main.py evaluate --lora-model /path/to/lora --split-file /path/to/split.json --batch-test
  
  # 详细评测与批量推理测试
  python lora_main.py evaluate --lora-model /path/to/lora --split-file /path/to/split.json --max-samples 1000 --batch-test --save-detailed
  
  # 训练后自动评测
  python lora_main.py train-and-eval --data-dir /path/to/data --eval-split-file /path/to/split.json
        """
    )
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练LoRA模型')
    add_train_arguments(train_parser)
    
    # 恢复训练命令
    resume_parser = subparsers.add_parser('resume', help='恢复训练')
    resume_parser.add_argument('--checkpoint', required=True, help='检查点文件路径')
    resume_parser.add_argument('--config', help='配置文件路径（可选）')
    
    # 🔧 增强版评测命令
    eval_parser = subparsers.add_parser('evaluate', help='评测LoRA模型 (增强版)')
    add_enhanced_eval_arguments(eval_parser)
    
    # 训练+评测命令
    train_eval_parser = subparsers.add_parser('train-and-eval', help='训练后自动评测')
    add_train_arguments(train_eval_parser)
    add_enhanced_eval_arguments(train_eval_parser)
    
    # 数据准备命令
    data_parser = subparsers.add_parser('prepare-data', help='准备训练数据/预览数据划分')
    data_parser.add_argument('--data-dir', required=True, help='数据目录')
    data_parser.add_argument('--train-ratio', type=float, default=0.8, help='训练集比例')
    data_parser.add_argument('--val-ratio', type=float, default=0.1, help='验证集比例')
    data_parser.add_argument('--test-split', type=float, default=0.1, help='测试集比例')
    data_parser.add_argument('--cell-types', nargs='+', help='要处理的细胞类型')
    data_parser.add_argument('--split-method', choices=['random', 'by_dataset'], default='random', help='划分方法')
    data_parser.add_argument('--seed', type=int, default=42, help='随机种子')
    data_parser.add_argument('--preview-only', action='store_true', help='只预览，不创建实际划分')
    
    # 数据划分管理命令
    splits_parser = subparsers.add_parser('manage-splits', help='管理数据划分缓存')
    splits_parser.add_argument('--list', action='store_true', help='列出所有缓存的划分')
    splits_parser.add_argument('--clean', action='store_true', help='清理旧的划分文件')
    splits_parser.add_argument('--keep', type=int, default=10, help='保留的最新划分数量')
    splits_parser.add_argument('--split-dir', default='./data/lora_split', help='划分文件存储目录')
    
    # 模型信息命令
    info_parser = subparsers.add_parser('info', help='显示模型和系统信息')
    info_parser.add_argument('--model', choices=['vit_t_lm', 'vit_b_lm', 'vit_l_lm'], 
                            default='vit_b_lm', help='模型类型')
    
    # 通用参数
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    return parser.parse_args()


def add_enhanced_eval_arguments(parser):
    """添加增强版评测相关参数"""
    # 🔧 基本评测参数
    parser.add_argument('--lora-model', required=True, help='LoRA模型路径')
    parser.add_argument('--split-file', help='数据划分文件路径 (推荐)')
    parser.add_argument('--eval-data', help='评测数据目录 (备选)')
    parser.add_argument('--eval-output', help='评测结果输出目录')
    
    # 🔧 新增：详细评测参数
    parser.add_argument('--max-samples', type=int, help='最大评测样本数')
    parser.add_argument('--eval-batch-size', type=int, default=1, help='评测批次大小')
    parser.add_argument('--save-detailed', action='store_true', default=True, help='保存详细结果')
    parser.add_argument('--cell-types', nargs='+', help='细胞类型过滤')
    
    # 🔧 新增：批量推理测试参数
    parser.add_argument('--batch-test', action='store_true', help='执行批量推理性能测试')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 2, 4, 8], 
                       help='批量推理测试的批次大小')
    
    # 🔧 新增：性能分析参数
    parser.add_argument('--benchmark', action='store_true', help='执行性能基准测试')
    parser.add_argument('--compare-baseline', action='store_true', help='与基础模型对比')


def add_train_arguments(parser):
    """添加训练相关参数"""
    # 数据配置
    parser.add_argument('--data-dir', required=True, help='训练数据目录')
    parser.add_argument('--val-data-dir', help='验证数据目录')
    parser.add_argument('--output-dir', default='./data/lora_experiments', help='输出目录')
    
    # 数据划分参数
    parser.add_argument('--test-split', type=float, default=0.1, help='测试集比例（0.0-1.0）')
    parser.add_argument('--val-split', type=float, help='验证集比例（如果不指定，从train_ratio计算）')
    parser.add_argument('--split-method', choices=['random', 'by_dataset'], default='random', help='数据划分方法')
    parser.add_argument('--split-seed', type=int, default=42, help='数据划分随机种子')
    parser.add_argument('--no-cached-split', action='store_true', help='不使用缓存的数据划分')
    
    # 模型配置
    parser.add_argument('--model', choices=['vit_t_lm', 'vit_b_lm', 'vit_l_lm'], 
                       default='vit_b_lm', help='基础模型')
    parser.add_argument('--preset', choices=list(LORA_PRESET_CONFIGS.keys()), 
                       help='预设配置')
    parser.add_argument('--config', help='配置文件路径')
    
    # LoRA配置
    parser.add_argument('--rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--alpha', type=float, default=16.0, help='LoRA alpha')
    parser.add_argument('--dropout', type=float, default=0.1, help='LoRA dropout')
    parser.add_argument('--lora-target', choices=['image_encoder', 'mask_decoder', 'both'],
                       default='image_encoder', help='LoRA应用目标')
    
    # 训练配置
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=8, help='批大小')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--save-steps', type=int, default=500, help='保存检查点的步数间隔')
    parser.add_argument('--eval-steps', type=int, default=100, help='验证的步数间隔')
    parser.add_argument('--logging-steps', type=int, default=50, help='日志记录的步数间隔')
    
    # 实验配置
    parser.add_argument('--experiment-name', default='sam_lora_finetune', help='实验名称')
    parser.add_argument('--use-wandb', action='store_true', help='使用Weights & Biases')
    parser.add_argument('--wandb-project', default='sam_lora_training', help='W&B项目名')
    parser.add_argument('--cell-types', nargs='+', help='要训练的细胞类型，如: --cell-types 293T MSC')


# 🔧 增强版评测函数
def evaluate_lora_model_enhanced(args, lora_model_path: str = None) -> bool:
    """增强版LoRA模型评测"""
    print("="*60)
    print("开始增强版SAM LoRA模型评测")
    print("="*60)
    
    # 确定模型路径
    if lora_model_path is None:
        lora_model_path = args.lora_model
    
    if not lora_model_path:
        print("错误: 未指定LoRA模型路径")
        return False
    
    try:
        # 创建增强版评测器
        evaluator = EnhancedLoRAEvaluator(lora_model_path, device="auto")
        
        print(f"LoRA模型路径: {lora_model_path}")
        
        # 🔧 优先使用数据划分文件
        if hasattr(args, 'split_file') and args.split_file:
            print(f"使用数据划分文件: {args.split_file}")
            
            # 执行评测
            eval_results = evaluator.evaluate_with_split_file(args.split_file, args)
            
            if not eval_results:
                print("❌ 评测失败")
                return False
            
            # 🔧 批量推理测试
            if getattr(args, 'batch_test', False):
                print("\n" + "="*60)
                print("执行批量推理性能测试")
                print("="*60)
                
                # 重新加载测试样本用于批量测试
                try:
                    with open(args.split_file, 'r', encoding='utf-8') as f:
                        split_data = json.load(f)
                    split_result = DataSplit.from_dict(split_data)
                    test_samples = split_result.test_samples
                    
                    # 验证样本有效性
                    valid_samples = []
                    for sample in test_samples:
                        if Path(sample['image_path']).exists() and Path(sample['mask_path']).exists():
                            valid_samples.append(sample)
                    
                    if valid_samples:
                        batch_sizes = getattr(args, 'batch_sizes', [1, 2, 4, 8])
                        batch_results = evaluator.batch_inference_test(valid_samples, batch_sizes)
                        
                        # 保存批量测试结果
                        if batch_results:
                            output_dir = Path(getattr(args, 'eval_output', evaluator.lora_model_path.parent / "evaluation_results"))
                            batch_file = output_dir / "batch_inference_results.json"
                            with open(batch_file, 'w') as f:
                                json.dump(batch_results, f, indent=2)
                            print(f"批量推理测试结果已保存: {batch_file}")
                            
                            # 打印批量测试摘要
                            print(f"\n📊 批量推理测试摘要:")
                            print(f"{'批次大小':<8} {'平均时间(s)':<12} {'吞吐量(图像/s)':<15} {'状态':<10}")
                            print("-" * 50)
                            for batch_size, result in batch_results.items():
                                if 'error' not in result:
                                    throughput = result['throughput']
                                    avg_time = result['avg_batch_time']
                                    status = "✅ 成功"
                                    print(f"{batch_size:<8} {avg_time:<12.4f} {throughput:<15.2f} {status:<10}")
                                else:
                                    print(f"{batch_size:<8} {'N/A':<12} {'N/A':<15} {'❌ 失败':<10}")
                
                except Exception as e:
                    print(f"批量推理测试失败: {e}")
        
        # 🔧 备选：使用数据目录评测
        elif hasattr(args, 'eval_data') and args.eval_data:
            print(f"使用数据目录: {args.eval_data}")
            
            from lora.data_loaders import create_data_loaders
            from config.lora_config import DataConfig
            
            # 创建数据配置
            config = DataConfig() 
            config.test_data_dir = args.eval_data
            config.batch_size = getattr(args, 'eval_batch_size', 1)
            config._cell_types_filter = getattr(args, 'cell_types', None)
            
            try:
                data_loaders = create_data_loaders(config, dataset_type="sam")
                
                if 'test' not in data_loaders:
                    print("❌ 无法创建测试数据加载器")
                    return False
                
                test_loader = data_loaders['test']
                print(f"测试数据: {len(test_loader)} 批次")
                
                # 这里可以进一步实现基于数据目录的评测...
                print("基于数据目录的评测功能开发中...")
                
            except Exception as e:
                print(f"使用数据目录评测失败: {e}")
                return False
        
        else:
            print("错误: 请指定 --split-file 或 --eval-data")
            return False
        
        print("\n✅ 増强版LoRA模型评测完成!")
        return True
        
    except Exception as e:
        print(f"评测过程中出现错误: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        return False


# 保持原有函数，但改名区分
def create_config_from_args(args) -> LoRATrainingSettings:
    """从命令行参数创建配置"""
    
    # 使用预设配置
    if hasattr(args, 'preset') and args.preset:
        config = LORA_PRESET_CONFIGS[args.preset]
        print(f"使用预设配置: {args.preset}")
    
    # 从配置文件加载
    elif hasattr(args, 'config') and args.config:
        config = LoRATrainingSettings.from_json(args.config)
        print(f"从配置文件加载: {args.config}")
    
    # 为特定模型创建配置
    elif hasattr(args, 'model'):
        config = get_config_for_model(args.model)
        print(f"为模型 {args.model} 创建配置")
    
    # 使用默认配置
    else:
        config = LoRATrainingSettings()
        print("使用默认配置")
    
    # 更新配置 - 保持原有逻辑
    if hasattr(args, 'data_dir'):
        config.data.train_data_dir = args.data_dir
    
    if hasattr(args, 'val_data_dir') and args.val_data_dir:
        config.data.val_data_dir = args.val_data_dir
    
    if hasattr(args, 'output_dir'):
        config.experiment.output_dir = args.output_dir
    
    if hasattr(args, 'model'):
        config.model.base_model_name = args.model
    
    # 数据划分配置
    if hasattr(args, 'test_split'):
        config.data.test_split_ratio = args.test_split
        
        if hasattr(args, 'val_split') and args.val_split is not None:
            config.data.val_split_ratio = args.val_split
            config.data.train_split_ratio = 1.0 - args.test_split - args.val_split
        else:
            if args.test_split >= 0.9:
                config.data.val_split_ratio = 0.0
                config.data.train_split_ratio = 1.0 - args.test_split
            else:
                remaining_ratio = 1.0 - args.test_split
                config.data.train_split_ratio = remaining_ratio * 0.9
                config.data.val_split_ratio = remaining_ratio * 0.1
        
        print(f"数据划分比例: train={config.data.train_split_ratio:.3f}, "
              f"val={config.data.val_split_ratio:.3f}, test={config.data.test_split_ratio:.3f}")
    
    if hasattr(args, 'split_method'):
        config.data.split_method = args.split_method
        
    if hasattr(args, 'split_seed'):
        config.data.split_seed = args.split_seed
        
    if hasattr(args, 'no_cached_split'):
        config.data.use_cached_split = not args.no_cached_split
    
    # LoRA参数
    if hasattr(args, 'rank'):
        config.lora.rank = args.rank
    if hasattr(args, 'alpha'):
        config.lora.alpha = args.alpha
    if hasattr(args, 'dropout'):
        config.lora.dropout = args.dropout
    
    # LoRA目标设置
    if hasattr(args, 'lora_target'):
        if args.lora_target == 'image_encoder':
            config.model.apply_lora_to = ['image_encoder']
        elif args.lora_target == 'mask_decoder':
            config.model.apply_lora_to = ['mask_decoder']
        elif args.lora_target == 'both':
            config.model.apply_lora_to = ['image_encoder', 'mask_decoder']
    
    # 训练参数
    if hasattr(args, 'epochs'):
        config.training.num_epochs = args.epochs
    if hasattr(args, 'batch_size'):
        config.training.batch_size = args.batch_size
    if hasattr(args, 'learning_rate'):
        config.training.learning_rate = args.learning_rate
    if hasattr(args, 'weight_decay'):
        config.training.weight_decay = args.weight_decay
    if hasattr(args, 'save_steps'):
        config.training.save_steps = args.save_steps
    if hasattr(args, 'eval_steps'):
        config.training.eval_steps = args.eval_steps  
    if hasattr(args, 'logging_steps'):
        config.training.logging_steps = args.logging_steps
    
    # 实验配置
    if hasattr(args, 'experiment_name'):
        config.experiment.experiment_name = args.experiment_name
    if hasattr(args, 'use_wandb'):
        config.experiment.use_wandb = args.use_wandb
    if hasattr(args, 'wandb_project'):
        config.experiment.wandb_project = args.wandb_project

    # 添加细胞类型过滤
    if hasattr(args, 'cell_types') and args.cell_types:
        config.data._cell_types_filter = args.cell_types
    else:
        config.data._cell_types_filter = None
    
    # 调试模式
    if hasattr(args, 'debug') and args.debug:
        config.experiment.debug_mode = True
        config.training.num_epochs = 2
        config.training.batch_size = 2
        config.training.save_steps = 10
        config.training.eval_steps = 5
        config.training.logging_steps = 1
        print("调试模式已启用")
    
    return config


def check_system_requirements():
    """检查系统要求"""
    print("检查系统要求...")
    
    # 检查PyTorch
    print(f"PyTorch版本: {torch.__version__}")
    
    # 检查设备信息
    device_info = get_device_info()
    print(f"CUDA可用: {device_info['cuda_available']}")
    
    if device_info['cuda_available']:
        print(f"GPU数量: {device_info['cuda_device_count']}")
        for gpu_name, gpu_info in device_info['gpu_memory'].items():
            print(f"  {gpu_name}: {gpu_info['name']}, "
                  f"内存: {gpu_info['total_memory'] / 1e9:.1f} GB")
    
    # 检查内存
    cpu_memory = device_info['cpu_memory']
    print(f"CPU内存: {cpu_memory['total'] / 1e9:.1f} GB "
          f"(可用: {cpu_memory['available'] / 1e9:.1f} GB)")
    
    # 检查micro_sam
    try:
        import micro_sam
        print(f"micro_sam已安装")
    except ImportError:
        print("警告: micro_sam未安装，可能影响模型加载")
    
    print("系统检查完成\n")


def train_lora_model(args) -> str:
    """训练LoRA模型 - 保持原有逻辑"""
    # 检查是否需要分别训练多个细胞类型
    if hasattr(args, 'cell_types') and args.cell_types and len(args.cell_types) > 1:
        return train_multiple_cell_types(args)
    
    print("="*60)
    print("开始SAM LoRA微调训练")
    print("="*60)
    
    check_system_requirements()
    config = create_config_from_args(args)
    
    if not config.validate():
        print("配置验证失败")
        return None
    
    # 如果是单个细胞类型，创建更详细的实验名称
    if hasattr(args, 'cell_types') and args.cell_types and len(args.cell_types) == 1:
        cell_type = args.cell_types[0]
        test_ratio = int(args.test_split * 100) if hasattr(args, 'test_split') else 10
        val_ratio = int(args.val_split * 100) if hasattr(args, 'val_split') and args.val_split else 10
        train_ratio = 100 - test_ratio - val_ratio
        
        split_suffix = f"train{train_ratio}_val{val_ratio}_test{test_ratio}"
        config.experiment.experiment_name = f"sam_lora_{cell_type.lower()}_{split_suffix}"
        config.experiment.output_dir = f"{config.experiment.output_dir}_{cell_type.lower()}_{split_suffix}"
    
    # 预览数据划分
    if hasattr(args, 'test_split') and args.test_split > 0:
        print(f"\n预览数据划分...")
        try:
            preview_stats = preview_data_split(
                data_dir=config.data.train_data_dir,
                train_ratio=config.data.train_split_ratio,
                val_ratio=config.data.val_split_ratio,
                test_ratio=config.data.test_split_ratio,
                cell_types=config.data._cell_types_filter,
                split_method=config.data.split_method,
                seed=config.data.split_seed,
                split_storage_dir=config.data.split_storage_dir
            )
            
            if preview_stats:
                print(f"  总样本数: {preview_stats['total_samples']}")
                print(f"  训练集: {preview_stats['train_count']} 样本")
                print(f"  验证集: {preview_stats['val_count']} 样本")
                print(f"  测试集: {preview_stats['test_count']} 样本")
                
                if 'cell_type_distribution' in preview_stats:
                    print(f"  细胞类型分布: {preview_stats['cell_type_distribution']}")
                    
        except Exception as e:
            print(f"预览数据划分失败: {e}")
    
    # 打印配置信息
    print(f"\n训练配置:")
    print(f"  基础模型: {config.model.base_model_name}")
    print(f"  LoRA配置:")
    print(f"    rank: {config.lora.rank}")
    print(f"    alpha: {config.lora.alpha}")
    print(f"    dropout: {config.lora.dropout}")
    print(f"    应用到: {config.model.apply_lora_to}")
    print(f"  训练配置:")
    print(f"    学习率: {config.training.learning_rate}")
    print(f"    批大小: {config.training.batch_size}")
    print(f"    训练轮数: {config.training.num_epochs}")
    print(f"  输出目录: {config.experiment.output_dir}")
    print(f"  数据目录: {config.data.train_data_dir}")
    
    trainer = LoRATrainer(config)
    success = trainer.train()
    
    if success:
        model_path = trainer.output_dir / "final_model"
        print(f"\n训练完成! 模型保存在: {model_path}")
        return str(model_path)
    else:
        print("\n训练失败!")
        return None


def train_multiple_cell_types(args) -> str:
    """为多个细胞类型分别训练模型 - 保持原有逻辑"""
    print("="*60)
    print("开始多细胞类型分别训练")
    print("="*60)
    print(f"将训练的细胞类型: {', '.join(args.cell_types)}")
    
    results = {}
    
    for cell_type in args.cell_types:
        print(f"\n🔄 开始训练 {cell_type} 模型...")
        
        # 创建单个细胞类型的参数副本
        single_args = type(args)()
        for attr in dir(args):
            if not attr.startswith('_'):
                setattr(single_args, attr, getattr(args, attr))
        
        # 设置为单个细胞类型
        single_args.cell_types = [cell_type]
        
        # 训练
        model_path = train_lora_model(single_args)
        results[cell_type] = model_path
        
        if model_path:
            print(f"✅ {cell_type} 训练完成: {model_path}")
        else:
            print(f"❌ {cell_type} 训练失败")
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 生成摘要
    print(f"\n{'='*60}")
    print("多细胞类型训练完成")
    print(f"{'='*60}")
    
    successful = sum(1 for path in results.values() if path)
    print(f"成功训练: {successful}/{len(args.cell_types)}")
    
    for cell_type, path in results.items():
        if path:
            print(f"  ✅ {cell_type}: {path}")
        else:
            print(f"  ❌ {cell_type}: 失败")
    
    return f"多细胞类型训练完成，成功: {successful}/{len(args.cell_types)}"


def resume_lora_training(args) -> str:
    """恢复LoRA训练 - 保持原有逻辑"""
    print("="*60)
    print("恢复SAM LoRA训练")
    print("="*60)
    
    trainer = resume_training(args.checkpoint, args.config)
    success = trainer.train()
    
    if success:
        model_path = trainer.output_dir / "final_model"
        print(f"\n训练完成! 模型保存在: {model_path}")
        return str(model_path)
    else:
        print("\n训练失败!")
        return None


def prepare_training_data(args):
    """准备训练数据/预览数据划分 - 保持原有逻辑"""
    print("="*60)
    print("准备训练数据 / 数据划分预览")
    print("="*60)
    
    data_dir = args.data_dir
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    test_ratio = getattr(args, 'test_split', 0.1)
    
    # 验证比例总和
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"警告: 比例总和不为1.0 ({total_ratio})，正在自动归一化...")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
    
    cell_types = getattr(args, 'cell_types', None)
    split_method = getattr(args, 'split_method', 'random')
    seed = getattr(args, 'seed', 42)
    preview_only = getattr(args, 'preview_only', False)
    
    print(f"数据目录: {data_dir}")
    print(f"分割比例 - 训练: {train_ratio:.3f}, 验证: {val_ratio:.3f}, 测试: {test_ratio:.3f}")
    print(f"细胞类型过滤: {cell_types}")
    print(f"分割方法: {split_method}")
    print(f"随机种子: {seed}")
    print(f"预览模式: {preview_only}")
    
    try:
        if preview_only:
            stats = preview_data_split(
                data_dir=data_dir,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                cell_types=cell_types,
                split_method=split_method,
                seed=seed
            )
            
            if stats:
                print(f"\n📊 数据划分预览:")
                print(f"  总样本数: {stats['total_samples']}")
                print(f"  训练集: {stats['train_count']} 样本 ({stats['train_count']/stats['total_samples']*100:.1f}%)")
                print(f"  验证集: {stats['val_count']} 样本 ({stats['val_count']/stats['total_samples']*100:.1f}%)")
                print(f"  测试集: {stats['test_count']} 样本 ({stats['test_count']/stats['total_samples']*100:.1f}%)")
                
                if 'cell_type_distribution' in stats:
                    print(f"  细胞类型分布:")
                    for cell_type, count in stats['cell_type_distribution'].items():
                        print(f"    {cell_type}: {count} 样本")
            else:
                print("预览失败")
        else:
            from utils.data_splitter import create_data_split, print_split_summary
            
            split_result = create_data_split(
                data_dir=data_dir,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                cell_types=cell_types,
                split_method=split_method,
                seed=seed,
                use_cached=True
            )
            
            print_split_summary(split_result)
            
        print("数据准备完成!")
        
    except Exception as e:
        print(f"数据准备失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


def manage_data_splits(args):
    """管理数据划分缓存 - 保持原有逻辑"""
    print("="*60)
    print("数据划分缓存管理")
    print("="*60)
    
    split_dir = args.split_dir
    
    if args.list:
        print(f"📋 列出缓存的数据划分 (目录: {split_dir})")
        try:
            cached_splits = list_cached_splits(split_dir)
            
            if not cached_splits:
                print("  没有找到缓存的数据划分")
            else:
                print(f"  找到 {len(cached_splits)} 个缓存文件:")
                
                for i, split_info in enumerate(cached_splits, 1):
                    print(f"\n  {i}. 文件: {Path(split_info['file_path']).name}")
                    print(f"     大小: {split_info['file_size_mb']:.2f} MB")
                    print(f"     数据目录: {split_info.get('data_dir', 'N/A')}")
                    print(f"     比例: train={split_info.get('train_ratio', 0):.2f}, "
                          f"val={split_info.get('val_ratio', 0):.2f}, "
                          f"test={split_info.get('test_ratio', 0):.2f}")
                    print(f"     样本数: {split_info.get('total_samples', 0)}")
                    if split_info.get('cell_types'):
                        print(f"     细胞类型: {split_info['cell_types']}")
                    print(f"     创建时间: {split_info.get('created_at', 'N/A')}")
                        
        except Exception as e:
            print(f"列出缓存失败: {e}")
    
    if args.clean:
        print(f"\n🧹 清理旧的数据划分文件 (保留最新 {args.keep} 个)")
        try:
            clean_old_splits(split_dir, args.keep)
            print("清理完成!")
        except Exception as e:
            print(f"清理失败: {e}")


def show_model_info(args):
    """显示模型和系统信息 - 保持原有逻辑"""
    print("="*60)
    print("模型和系统信息")
    print("="*60)
    
    # 系统信息
    check_system_requirements()
    
    # 尝试加载模型信息
    try:
        from core.sam_model_loader import create_sam_model_loader
        from utils.model_utils import print_model_summary
        
        print(f"正在加载模型信息: {args.model}")
        
        loader = create_sam_model_loader(args.model, "cpu")
        if loader.load_model():
            print("\n模型加载成功!")
            
            components = loader.get_trainable_components()
            print(f"\n模型组件:")
            for name, component in components.items():
                param_count = sum(p.numel() for p in component.parameters())
                print(f"  {name}: {param_count:,} 参数")
            
            if loader.model is not None:
                print_model_summary(loader.model)
            
        else:
            print("模型加载失败")
    
    except Exception as e:
        print(f"获取模型信息失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


def main():
    """主函数 - 增强版"""
    # 设置日志
    setup_logging()
    
    # 解析参数
    args = parse_arguments()
    
    if args.command is None:
        print("请指定命令。使用 --help 查看帮助信息")
        return
    
    try:
        if args.command == 'train':
            train_lora_model(args)
        
        elif args.command == 'resume':
            resume_lora_training(args)
        
        elif args.command == 'evaluate':
            # 🔧 使用增强版评测
            evaluate_lora_model_enhanced(args)
        
        elif args.command == 'train-and-eval':
            # 先训练
            lora_model_path = train_lora_model(args)
            
            # 再评测
            if lora_model_path:
                print("\n" + "="*60)
                print("开始自动评测")
                print("="*60)
                
                # 🔧 如果有split_file参数，直接使用增强版评测
                if hasattr(args, 'split_file') and args.split_file:
                    # 设置评测参数
                    args.lora_model = lora_model_path
                    evaluate_lora_model_enhanced(args)
                else:
                    # 回退到传统评测（如果需要的话）
                    print("训练完成，但未指定数据划分文件，跳过自动评测")
                    print("建议使用 --split-file 参数指定测试数据")
        
        elif args.command == 'prepare-data':
            prepare_training_data(args)
        
        elif args.command == 'manage-splits':
            manage_data_splits(args)
        
        elif args.command == 'info':
            show_model_info(args)
        
        else:
            print(f"未知命令: {args.command}")
    
    except KeyboardInterrupt:
        print("\n\n操作被用户中断")
    except Exception as e:
        print(f"\n执行过程中出现错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
    finally:
        # 清理资源
        optimize_memory()


if __name__ == "__main__":
    main()