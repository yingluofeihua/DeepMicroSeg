"""
LoRA训练主入口文件 (修改版)
支持LoRA微调和评测的完整流程
使用新的SAM模型架构
"""

import sys
import argparse
from pathlib import Path
import json
import torch

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from config.lora_config import LoRATrainingSettings, LORA_PRESET_CONFIGS, get_config_for_model
from core.lora_trainer import LoRATrainer, create_trainer_from_config, resume_training
from core.evaluator import BatchEvaluator
from core.dataset_manager import DatasetManager
from config.settings import BatchEvaluationSettings
from lora.data_loaders import split_dataset
from utils.file_utils import setup_logging
from utils.model_utils import get_device_info, optimize_memory


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="SAM LoRA微调训练系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 快速训练
  python lora_main.py train --preset quick --data-dir /path/to/data
  
  # 标准训练
  python lora_main.py train --data-dir /path/to/data --model vit_b_lm --epochs 10
  
  # 从配置文件训练
  python lora_main.py train --config config.json
  
  # 恢复训练
  python lora_main.py resume --checkpoint /path/to/checkpoint.pth
  
  # 评测LoRA模型
  python lora_main.py evaluate --lora-model /path/to/lora --data-dir /path/to/data
  
  # 训练后自动评测
  python lora_main.py train-and-eval --data-dir /path/to/data --eval-data /path/to/eval
  
  # 准备数据
  python lora_main.py prepare-data --data-dir /path/to/data
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
    
    # 评测命令
    eval_parser = subparsers.add_parser('evaluate', help='评测LoRA模型')
    add_eval_arguments(eval_parser)
    
    # 训练+评测命令
    train_eval_parser = subparsers.add_parser('train-and-eval', help='训练后自动评测')
    add_train_arguments(train_eval_parser)
    add_eval_arguments(train_eval_parser)
    
    # 数据准备命令
    data_parser = subparsers.add_parser('prepare-data', help='准备训练数据')
    data_parser.add_argument('--data-dir', required=True, help='数据目录')
    data_parser.add_argument('--train-ratio', type=float, default=0.8, help='训练集比例')
    data_parser.add_argument('--val-ratio', type=float, default=0.1, help='验证集比例')
    
    # 模型信息命令
    info_parser = subparsers.add_parser('info', help='显示模型和系统信息')
    info_parser.add_argument('--model', choices=['vit_t_lm', 'vit_b_lm', 'vit_l_lm'], 
                            default='vit_b_lm', help='模型类型')
    
    # 通用参数
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    return parser.parse_args()


def add_train_arguments(parser):
    """添加训练相关参数"""
    # 数据配置
    parser.add_argument('--data-dir', required=True, help='训练数据目录')
    parser.add_argument('--val-data-dir', help='验证数据目录')
    parser.add_argument('--output-dir', default='./data/lora_experiments', help='输出目录')
    
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
    
    # 实验配置
    parser.add_argument('--experiment-name', default='sam_lora_finetune', help='实验名称')
    parser.add_argument('--use-wandb', action='store_true', help='使用Weights & Biases')
    parser.add_argument('--wandb-project', default='sam_lora_training', help='W&B项目名')

    # 添加保存相关参数
    parser.add_argument('--save-steps', type=int, default=500, help='保存检查点的步数间隔')
    parser.add_argument('--eval-steps', type=int, default=100, help='验证的步数间隔')
    parser.add_argument('--logging-steps', type=int, default=50, help='日志记录的步数间隔')
    # 添加细胞类型过滤参数
    parser.add_argument('--cell-types', nargs='+', help='要训练的细胞类型，如: --cell-types 293T MSC')

def add_eval_arguments(parser):
    """添加评测相关参数"""
    parser.add_argument('--lora-model', help='LoRA模型路径')
    parser.add_argument('--eval-data', help='评测数据目录')
    parser.add_argument('--eval-output', help='评测结果输出目录')
    parser.add_argument('--compare-baseline', action='store_true', help='与基础模型对比')


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
    
    # 更新配置
    if hasattr(args, 'data_dir'):
        config.data.train_data_dir = args.data_dir
    
    if hasattr(args, 'val_data_dir') and args.val_data_dir:
        config.data.val_data_dir = args.val_data_dir
    
    if hasattr(args, 'output_dir'):
        config.experiment.output_dir = args.output_dir
    
    if hasattr(args, 'model'):
        config.model.base_model_name = args.model
    
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
        config._cell_types_filter = args.cell_types
    else:
        config._cell_types_filter = None
    
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
    """训练LoRA模型"""
    
    # 检查是否需要分别训练多个细胞类型
    if hasattr(args, 'cell_types') and args.cell_types and len(args.cell_types) > 1:
        return train_multiple_cell_types(args)
    
    # 原来的单模型训练逻辑
    print("="*60)
    print("开始SAM LoRA微调训练")
    print("="*60)
    
    check_system_requirements()
    config = create_config_from_args(args)
    
    if not config.validate():
        print("配置验证失败")
        return None
    
    # 如果是单个细胞类型，添加到实验名称中
    if hasattr(args, 'cell_types') and args.cell_types and len(args.cell_types) == 1:
        cell_type = args.cell_types[0]
        config.experiment.experiment_name = f"sam_lora_{cell_type.lower()}"
        config.experiment.output_dir = f"{config.experiment.output_dir}_{cell_type.lower()}"
        print(f"训练细胞类型: {cell_type}")
    
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
    """为多个细胞类型分别训练模型"""
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
        import torch
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
    """恢复LoRA训练"""
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


def evaluate_lora_model(args, lora_model_path: str = None) -> bool:
    """评测LoRA模型"""
    print("="*60)
    print("开始SAM LoRA模型评测")
    print("="*60)
    
    # 确定模型路径
    if lora_model_path is None:
        lora_model_path = args.lora_model
    
    if not lora_model_path:
        print("错误: 未指定LoRA模型路径")
        return False
    
    # 确定评测数据
    eval_data_dir = getattr(args, 'eval_data', None) or getattr(args, 'data_dir', None)
    if not eval_data_dir:
        print("错误: 未指定评测数据目录")
        return False
    
    try:
        # 加载LoRA模型进行评测
        from lora.sam_lora_wrapper import load_sam_lora_model
        from core.metrics import ComprehensiveMetrics
        from lora.data_loaders import create_data_loaders
        from config.lora_config import DataConfig
        
        print(f"LoRA模型路径: {lora_model_path}")
        print(f"评测数据目录: {eval_data_dir}")
        
        # 创建评测数据加载器
        data_config = DataConfig()
        data_config.test_data_dir = eval_data_dir
        
        data_loaders = create_data_loaders(data_config, dataset_type="sam")
        if 'test' not in data_loaders:
            print("无法创建测试数据加载器")
            return False
        
        test_loader = data_loaders['test']
        print(f"测试数据: {len(test_loader)} 批次")
        
        # 加载LoRA模型
        model_type = "vit_b_lm"  # 默认模型类型，可以从配置中读取
        lora_model = load_sam_lora_model(model_type, lora_model_path)
        
        if lora_model is None:
            print("LoRA模型加载失败")
            return False
        
        print("LoRA模型加载成功")
        
        # 创建指标计算器
        metrics_calculator = ComprehensiveMetrics()
        
        # 进行评测
        lora_model.eval()
        all_results = []
        
        print("开始评测...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= 10:  # 限制评测数量
                    break
                
                try:
                    # 准备输入
                    from lora.training_utils import prepare_sam_inputs
                    inputs, targets = prepare_sam_inputs(batch)
                    
                    # 模型预测
                    predictions = lora_model(inputs)
                    
                    # 计算指标
                    pred_masks = torch.sigmoid(predictions['masks']).cpu().numpy()
                    target_masks = targets['masks'].cpu().numpy()
                    
                    for pred, target in zip(pred_masks, target_masks):
                        if pred.ndim > 2:
                            pred = pred[0]
                        if target.ndim > 2:
                            target = target[0]
                        
                        pred_binary = (pred > 0.5).astype(int)
                        target_binary = (target > 0.5).astype(int)
                        
                        result = metrics_calculator.compute_all_metrics(target_binary, pred_binary)
                        all_results.append(result.to_dict())
                
                except Exception as e:
                    print(f"评测批次 {batch_idx} 失败: {e}")
                    continue
        
        # 计算平均指标
        if all_results:
            avg_metrics = {}
            for key in all_results[0].keys():
                values = [r[key] for r in all_results if key in r and r[key] is not None]
                if values:
                    if key == 'hd95':
                        finite_values = [v for v in values if not (v == float('inf') or v != v)]
                        avg_metrics[key] = sum(finite_values) / len(finite_values) if finite_values else float('inf')
                    else:
                        avg_metrics[key] = sum(values) / len(values)
            
            print(f"\n评测结果 (基于 {len(all_results)} 个样本):")
            for key, value in avg_metrics.items():
                print(f"  {key}: {value:.4f}")
            
            # 保存结果
            if hasattr(args, 'eval_output') and args.eval_output:
                output_dir = Path(args.eval_output)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                results_file = output_dir / "lora_evaluation_results.json"
                with open(results_file, 'w') as f:
                    json.dump({
                        'average_metrics': avg_metrics,
                        'individual_results': all_results,
                        'model_path': lora_model_path,
                        'data_path': eval_data_dir
                    }, f, indent=2)
                
                print(f"评测结果已保存到: {results_file}")
        
        else:
            print("没有有效的评测结果")
            return False
        
        print("LoRA模型评测完成!")
        return True
        
    except Exception as e:
        print(f"LoRA模型评测失败: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        return False


def prepare_training_data(args):
    """准备训练数据"""
    print("="*60)
    print("准备训练数据")
    print("="*60)
    
    data_dir = args.data_dir
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    
    print(f"数据目录: {data_dir}")
    print(f"分割比例 - 训练: {train_ratio}, 验证: {val_ratio}, 测试: {1-train_ratio-val_ratio}")
    
    try:
        split_dataset(data_dir, train_ratio, val_ratio)
        print("数据准备完成!")
    except Exception as e:
        print(f"数据准备失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


def show_model_info(args):
    """显示模型和系统信息"""
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
        
        loader = create_sam_model_loader(args.model, "cpu")  # 使用CPU加载以节省显存
        if loader.load_model():
            print("\n模型加载成功!")
            
            # 打印模型组件信息
            components = loader.get_trainable_components()
            print(f"\n模型组件:")
            for name, component in components.items():
                param_count = sum(p.numel() for p in component.parameters())
                print(f"  {name}: {param_count:,} 参数")
            
            # 打印详细的模型摘要（如果有完整模型）
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
    """主函数"""
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
            evaluate_lora_model(args)
        
        elif args.command == 'train-and-eval':
            # 先训练
            lora_model_path = train_lora_model(args)
            
            # 再评测
            if lora_model_path:
                print("\n" + "="*60)
                print("开始自动评测")
                print("="*60)
                evaluate_lora_model(args, lora_model_path)
        
        elif args.command == 'prepare-data':
            prepare_training_data(args)
        
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