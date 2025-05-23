"""
细胞分割批量评测系统 - 主入口文件
包含AP50、AP75、IoU、Dice、HD95等完整指标
支持多模型评测并生成统一的CSV报告
"""

import sys
import time
import argparse
from pathlib import Path
import torch
import torch.multiprocessing as mp

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from config.settings import BatchEvaluationSettings, PRESET_CONFIGS
from config.paths import validate_output_permissions, setup_environment_paths
from core.evaluator import BatchEvaluator
from utils.file_utils import setup_logging


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="细胞分割模型批量评测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py --data-dir /path/to/data --output-dir /path/to/output
  python main.py --preset fast --data-dir /path/to/data
  python main.py --config config.json --cell-types MSC Vero
        """
    )
    
    # 基本参数
    parser.add_argument(
        '--data-dir', '-d',
        required=True,
        help='数据集根目录路径'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        default=None,
        help='输出目录路径（默认：./batch_evaluation_results）'
    )
    
    parser.add_argument(
        '--cache-dir',
        default=None,
        help='缓存目录路径（默认：~/.cache/micro_sam）'
    )
    
    # 配置选项
    parser.add_argument(
        '--preset',
        choices=['fast', 'comprehensive', 'debug'],
        help='使用预设配置'
    )
    
    parser.add_argument(
        '--config',
        help='配置文件路径（JSON格式）'
    )
    
    # 模型选择
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['vit_t_lm', 'vit_b_lm', 'vit_l_lm'],
        help='要评测的模型列表'
    )
    
    # 数据集过滤
    parser.add_argument(
        '--cell-types',
        nargs='+',
        help='要处理的细胞类型'
    )
    
    parser.add_argument(
        '--dates',
        nargs='+',
        help='要处理的日期'
    )
    
    parser.add_argument(
        '--magnifications',
        nargs='+',
        help='要处理的放大倍数'
    )
    
    # 处理选项
    parser.add_argument(
        '--batch-size',
        type=int,
        help='每个数据集处理的图像数量限制'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=True,
        help='跳过已处理的数据集（默认：启用）'
    )
    
    parser.add_argument(
        '--no-skip-existing',
        action='store_false',
        dest='skip_existing',
        help='不跳过已处理的数据集'
    )
    
    parser.add_argument(
        '--no-visualizations',
        action='store_false',
        dest='save_visualizations',
        help='不生成可视化图表'
    )
    
    # 系统选项
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='预览模式，只显示将要处理的数据集'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出'
    )
    
    return parser.parse_args()


def load_config_from_args(args) -> BatchEvaluationSettings:
    """从命令行参数加载配置"""
    
    # 1. 使用预设配置
    if args.preset:
        config = PRESET_CONFIGS[args.preset]
        print(f"使用预设配置: {args.preset}")
    
    # 2. 从配置文件加载
    elif args.config:
        import json
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = BatchEvaluationSettings.from_dict(config_dict)
        print(f"从配置文件加载: {args.config}")
    
    # 3. 使用默认配置
    else:
        config = BatchEvaluationSettings()
        print("使用默认配置")
    
    # 更新路径
    if args.output_dir:
        config.output_base_dir = args.output_dir
    elif args.output_dir is None:
        config.output_base_dir = str(Path.cwd() / "batch_evaluation_results")
    
    if args.cache_dir:
        config.cache_dir = args.cache_dir
    
    config.update_paths(config.output_base_dir, config.cache_dir)
    
    # 更新模型列表
    if args.models:
        from config.settings import ModelConfig
        config.models = [ModelConfig(name) for name in args.models]
    
    # 更新处理选项
    if args.batch_size is not None:
        config.evaluation.batch_size = args.batch_size
    
    config.evaluation.skip_existing = args.skip_existing
    
    if hasattr(args, 'save_visualizations'):
        config.evaluation.save_visualizations = args.save_visualizations
    
    return config


def validate_environment(config: BatchEvaluationSettings, data_dir: str) -> bool:
    """验证运行环境"""
    
    print("正在验证环境...")
    
    # 检查数据目录
    if not Path(data_dir).exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        return False
    
    # 检查输出目录权限
    if not validate_output_permissions(config.output_base_dir):
        print(f"错误: 输出目录权限不足: {config.output_base_dir}")
        return False
    
    # 设置环境路径
    setup_environment_paths(config.cache_dir)
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"CUDA可用，设备数量: {torch.cuda.device_count()}")
    else:
        print("CUDA不可用，将使用CPU")
    
    print("环境验证通过")
    return True


def dry_run_preview(evaluator: BatchEvaluator, args):
    """预览模式 - 显示将要处理的数据集"""
    
    print("\n" + "="*60)
    print("预览模式 - 将要处理的数据集")
    print("="*60)
    
    # 获取过滤后的数据集
    datasets_to_process = evaluator.dataset_manager.filter_datasets(
        cell_types=args.cell_types,
        dates=args.dates,
        magnifications=args.magnifications
    )
    
    if not datasets_to_process:
        print("没有符合条件的数据集")
        return
    
    print(f"符合条件的数据集: {len(datasets_to_process)}")
    print(f"评测模型: {[model.name for model in evaluator.config.models]}")
    print(f"总任务数: {len(datasets_to_process) * len(evaluator.config.models)}")
    
    print("\n数据集详情:")
    for i, dataset in enumerate(datasets_to_process, 1):
        print(f"{i:2d}. {dataset.dataset_id}")
        print(f"    细胞类型: {dataset.cell_type}")
        print(f"    日期: {dataset.date}")
        print(f"    放大倍数: {dataset.magnification}")
        print(f"    有效图像对: {dataset.valid_pairs}")
    
    # 估算处理时间
    estimated_time = len(datasets_to_process) * len(evaluator.config.models) * 2  # 假设每个任务2分钟
    print(f"\n预估处理时间: {estimated_time:.0f} 分钟")
    
    print("="*60)


def main():
    """主函数"""
    
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    # 设置日志
    setup_logging()
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 打印欢迎信息
    print("="*60)
    print("细胞分割批量评测系统")
    print("包含AP50、AP75、IoU、Dice、HD95等完整指标")
    print("="*60)
    
    try:
        # 加载配置
        config = load_config_from_args(args)
        
        # 验证环境
        if not validate_environment(config, args.data_dir):
            sys.exit(1)
        
        # 创建评测器
        evaluator = BatchEvaluator(config)
        
        # 设置评测环境
        if not evaluator.setup(args.data_dir):
            print("评测环境设置失败")
            sys.exit(1)
        
        # 显示配置信息
        print(f"\n配置信息:")
        print(f"  数据目录: {args.data_dir}")
        print(f"  输出目录: {config.output_base_dir}")
        print(f"  缓存目录: {config.cache_dir}")
        print(f"  评测模型: {[m.name for m in config.models]}")
        print(f"  批处理大小: {'全部图像' if config.evaluation.batch_size is None else config.evaluation.batch_size}")
        print(f"  跳过已有结果: {config.evaluation.skip_existing}")
        
        # 显示数据集摘要
        evaluator.dataset_manager.print_summary()
        
        # 预览模式
        if args.dry_run:
            dry_run_preview(evaluator, args)
            return
        
        # 确认开始评测
        if not args.verbose:
            confirm = input("\n是否开始评测? (y/N): ").lower().strip()
            if confirm != 'y':
                print("评测已取消")
                return
        
        # 开始评测
        print("\n" + "="*60)
        print("开始批量评测...")
        print("="*60)
        
        start_time = time.time()
        
        success = evaluator.run_evaluation(
            cell_types=args.cell_types,
            dates=args.dates,
            magnifications=args.magnifications
        )
        
        total_time = time.time() - start_time
        
        # 显示结果
        print("\n" + "="*60)
        if success:
            print("批量评测完成!")
        else:
            print("批量评测出现问题!")
        print("="*60)
        
        print(f"总耗时: {total_time/3600:.2f} 小时")
        print(f"结果保存在: {config.output_base_dir}")
        
        # 显示最终结果文件
        summary_dirs = list(Path(config.output_base_dir).glob("summary_report_*"))
        if summary_dirs:
            latest_summary = max(summary_dirs, key=lambda x: x.name)
            
            final_summary_file = latest_summary / "final_evaluation_summary.csv"
            if final_summary_file.exists():
                print(f"\n主要结果文件:")
                print(f"  最终摘要: {final_summary_file}")
                
                # 显示简要统计
                import pandas as pd
                df = pd.read_csv(final_summary_file)
                print(f"  包含 {len(df)} 条记录")
                print(f"  覆盖模型: {df['model'].unique().tolist()}")
                
                if 'cell_type' in df.columns:
                    print(f"  覆盖细胞类型: {df['cell_type'].unique().tolist()}")
                
                # 显示各模型的最佳性能
                if 'ap50' in df.columns:
                    print("\n模型性能排名 (按AP50):")
                    model_performance = df.groupby('model')['ap50'].mean().sort_values(ascending=False)
                    for i, (model, score) in enumerate(model_performance.items()):
                        print(f"  {i+1}. {model}: {score:.3f}")
        
        print("\n评测系统执行完毕！")
        
    except KeyboardInterrupt:
        print("\n\n评测被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n评测过程中出现错误: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)
    finally:
        # 清理资源
        try:
            if 'evaluator' in locals():
                evaluator.cleanup()
        except:
            pass


if __name__ == "__main__":
    main()