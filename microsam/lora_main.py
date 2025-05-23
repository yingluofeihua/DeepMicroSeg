"""
LoRA训练主入口文件
支持LoRA微调和评测的完整流程
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


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="LoRA微调训练系统",
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
    
    # 通用参数
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    return parser.parse_args()


def add_train_arguments(parser):
    """添加训练相关参数"""
    # 数据配置
    parser.add_argument('--data-dir', required=True, help='训练数据目录')
    parser.add_argument('--val-data-dir', help='验证数据目录')
    parser.add_argument('--output-dir', default='./lora_experiments', help='输出目录')
    
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
    
    # 训练配置
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=8, help='批大小')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='权重衰减')
    
    # 实验配置
    parser.add_argument('--experiment-name', default='lora_finetune', help='实验名称')
    parser.add_argument('--use-wandb', action='store_true', help='使用Weights & Biases')
    parser.add_argument('--wandb-project', default='micro_sam_lora', help='W&B项目名')


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
    
    # 训练参数
    if hasattr(args, 'epochs'):
        config.training.num_epochs = args.epochs
    if hasattr(args, 'batch_size'):
        config.training.batch_size = args.batch_size
    if hasattr(args, 'learning_rate'):
        config.training.learning_rate = args.learning_rate
    if hasattr(args, 'weight_decay'):
        config.training.weight_decay = args.weight_decay
    
    # 实验配置
    if hasattr(args, 'experiment_name'):
        config.experiment.experiment_name = args.experiment_name
    if hasattr(args, 'use_wandb'):
        config.experiment.use_wandb = args.use_wandb
    if hasattr(args, 'wandb_project'):
        config.experiment.wandb_project = args.wandb_project
    
    # 调试模式
    if hasattr(args, 'debug') and args.debug:
        config.experiment.debug_mode = True
        config.training.num_epochs = 1
        config.training.batch_size = 2
        config.training.save_steps = 10
    
    return config


def train_lora_model(args) -> str:
    """训练LoRA模型"""
    print("="*60)
    print("开始LoRA微调训练")
    print("="*60)
    
    # 创建配置
    config = create_config_from_args(args)
    
    # 验证配置
    if not config.validate():
        print("配置验证失败")
        return None
    
    # 打印配置信息
    print(f"\n训练配置:")
    print(f"  基础模型: {config.model.base_model_name}")
    print(f"  LoRA rank: {config.lora.rank}")
    print(f"  LoRA alpha: {config.lora.alpha}")
    print(f"  学习率: {config.training.learning_rate}")
    print(f"  批大小: {config.training.batch_size}")
    print(f"  训练轮数: {config.training.num_epochs}")
    print(f"  输出目录: {config.experiment.output_dir}")
    
    # 创建训练器
    trainer = LoRATrainer(config)
    
    # 开始训练
    success = trainer.train()
    
    if success:
        model_path = trainer.output_dir / "final_model"
        print(f"\n训练完成! 模型保存在: {model_path}")
        return str(model_path)
    else:
        print("\n训练失败!")
        return None


def resume_lora_training(args) -> str:
    """恢复LoRA训练"""
    print("="*60)
    print("恢复LoRA训练")
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
    print("开始LoRA模型评测")
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
    
    # 创建评测配置
    eval_config = BatchEvaluationSettings()
    
    if hasattr(args, 'eval_output') and args.eval_output:
        eval_config.output_base_dir = args.eval_output
    else:
        eval_config.output_base_dir = str(Path(lora_model_path).parent / "evaluation_results")
    
    # 设置模型（这里需要扩展支持LoRA模型）
    # 暂时使用原始模型配置
    
    # 创建评测器
    evaluator = BatchEvaluator(eval_config)
    
    # 设置评测环境
    if not evaluator.setup(eval_data_dir):
        print("评测环境设置失败")
        return False
    
    # 运行评测
    success = evaluator.run_evaluation()
    
    if success:
        print(f"\n评测完成! 结果保存在: {eval_config.output_base_dir}")
        return True
    else:
        print("\n评测失败!")
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
        
        else:
            print(f"未知命令: {args.command}")
    
    except KeyboardInterrupt:
        print("\n\n操作被用户中断")
    except Exception as e:
        print(f"\n执行过程中出现错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()