#!/usr/bin/env python3
"""
基于SAM的细胞分割LoRA微调系统 - 命令行训练工具
"""

import argparse
import warnings
import os
import sys
from glob import glob
from typing import Union, Tuple, Optional, List

import numpy as np
import imageio.v3 as imageio
from matplotlib import pyplot as plt
from skimage.measure import label as connected_components

import torch

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.util.util import get_random_colors

import micro_sam.training as sam_training
from micro_sam.sample_data import fetch_tracking_example_data, fetch_tracking_segmentation_data
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

warnings.filterwarnings("ignore")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="基于SAM的细胞分割LoRA微调训练工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 数据路径参数
    parser.add_argument("--train_image_dir", type=str, required=True,
                       help="训练图像目录路径")
    parser.add_argument("--train_mask_dir", type=str, required=True,
                       help="训练掩码目录路径")
    parser.add_argument("--val_image_dir", type=str, required=True,
                       help="验证图像目录路径")
    parser.add_argument("--val_mask_dir", type=str, required=True,
                       help="验证掩码目录路径")
    
    # 模型参数
    parser.add_argument("--model_type", type=str, default="vit_b_lm",
                       choices=["vit_t", "vit_b", "vit_l", "vit_h", "vit_b_lm", "vit_l_lm"],
                       help="SAM模型类型")
    parser.add_argument("--checkpoint_name", type=str, required=True,
                       help="保存的检查点名称")
    parser.add_argument("--pretrained_checkpoint", type=str, default=None,
                       help="预训练检查点路径（可选）")
    
    # LoRA参数
    parser.add_argument("--rank", type=int, default=8,
                       help="LoRA rank参数")
    parser.add_argument("--attention_layers_to_update", type=int, nargs="+", 
                       default=[9, 10, 11],
                       help="需要更新的注意力层索引列表")
    
    # 训练参数
    parser.add_argument("--save_root", type=str, required=True,
                       help="模型保存根目录")
    parser.add_argument("--train_number", type=int, default=0,
                       help="训练的图片数量")
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="指定使用的GPU ID")
    parser.add_argument("--n_epochs", type=int, default=100,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="批处理大小")
    parser.add_argument("--n_objects_per_batch", type=int, default=5,
                       help="每批次采样的对象数量")
    parser.add_argument("--patch_height", type=int, default=512,
                       help="训练补丁高度")
    parser.add_argument("--patch_width", type=int, default=512,
                       help="训练补丁宽度")
    parser.add_argument("--min_instance_size", type=int, default=25,
                       help="最小实例大小")
    
    # 其他参数
    parser.add_argument("--train_instance_segmentation", action="store_true",
                       help="是否训练实例分割解码器")
    parser.add_argument("--freeze_prompt_encoder", action="store_true",
                       help="是否冻结prompt encoder")
    parser.add_argument("--visualize_first_sample", action="store_true",
                       help="是否可视化第一个样本")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    
    return parser.parse_args()


def validate_paths(args):
    """验证输入路径是否存在"""
    paths_to_check = [
        ("train_image_dir", args.train_image_dir),
        ("train_mask_dir", args.train_mask_dir),
        ("val_image_dir", args.val_image_dir),
        ("val_mask_dir", args.val_mask_dir),
    ]
    
    for name, path in paths_to_check:
        if not os.path.exists(path):
            raise ValueError(f"路径不存在: {name} = {path}")
    
    # 检查预训练检查点
    if args.pretrained_checkpoint and not os.path.exists(args.pretrained_checkpoint):
        print(f"Warning: 预训练检查点不存在: {args.pretrained_checkpoint}")
        print("将使用默认的SAM预训练权重")
        args.pretrained_checkpoint = None
    
    # 创建保存目录
    os.makedirs(args.save_root, exist_ok=True)


def setup_device(gpu_id):
    """设置GPU设备"""
    if torch.cuda.is_available():
        if gpu_id >= torch.cuda.device_count():
            raise ValueError(f"GPU ID {gpu_id} 不可用，总共有 {torch.cuda.device_count()} 个GPU")
        
        # 设置当前CUDA设备
        torch.cuda.set_device(gpu_id)
        
        # micro_sam 只接受 'cuda' 作为设备名，而不是 'cuda:X'
        device = "cuda"
        print(f"使用GPU: cuda:{gpu_id} ({torch.cuda.get_device_name(gpu_id)})")
        print(f"CUDA设备已设置为: {torch.cuda.current_device()}")
    else:
        device = "cpu"
        print("CUDA不可用，使用CPU训练")
    
    return device


def load_data_paths(image_dir, mask_dir):
    """加载数据路径"""
    image_paths = sorted(glob(os.path.join(image_dir, "*")))
    mask_paths = sorted(glob(os.path.join(mask_dir, "*")))
    
    if len(image_paths) == 0:
        raise ValueError(f"在 {image_dir} 中未找到图像文件")
    if len(mask_paths) == 0:
        raise ValueError(f"在 {mask_dir} 中未找到掩码文件")
    if len(image_paths) != len(mask_paths):
        raise ValueError(f"图像数量 ({len(image_paths)}) 与掩码数量 ({len(mask_paths)}) 不匹配")
    
    return image_paths, mask_paths


def visualize_sample(image_paths, mask_paths):
    """可视化第一个样本"""
    image_path, mask_path = image_paths[0], mask_paths[0]
    image = imageio.imread(image_path)
    mask = imageio.imread(mask_path)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Input Image")
    ax[0].axis("off")

    mask = connected_components(mask)
    ax[1].imshow(mask, cmap=get_random_colors(mask), interpolation="nearest")
    ax[1].set_title("Ground Truth Instances")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()


def create_data_loader(image_paths, mask_paths, args, is_train=True):
    """创建数据加载器"""
    sampler = MinInstanceSampler(min_size=args.min_instance_size)
    patch_shape = (1, args.patch_height, args.patch_width)

    if args.train_number>0:
        image_paths = image_paths[:args.train_number]
        mask_paths = mask_paths[:args.train_number]
    
    loader = sam_training.default_sam_loader(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=mask_paths,
        label_key=None,
        with_segmentation_decoder=args.train_instance_segmentation,
        patch_shape=patch_shape,
        batch_size=args.batch_size,
        is_seg_dataset=True,
        shuffle=is_train,
        raw_transform=sam_training.identity,
        sampler=sampler,
    )
    
    return loader


def print_training_config(args, device):
    """打印训练配置"""
    print("=" * 80)
    print("🔬 基于SAM的细胞分割LoRA微调系统")
    print("=" * 80)
    print("📊 数据配置:")
    print(f"  训练图像目录: {args.train_image_dir}")
    print(f"  训练掩码目录: {args.train_mask_dir}")
    print(f"  验证图像目录: {args.val_image_dir}")
    print(f"  验证掩码目录: {args.val_mask_dir}")
    print()
    print("🤖 模型配置:")
    print(f"  模型类型: {args.model_type}")
    print(f"  检查点名称: {args.checkpoint_name}")
    print(f"  预训练检查点: {args.pretrained_checkpoint or '使用默认SAM权重'}")
    print()
    print("🔧 LoRA配置:")
    print(f"  Rank: {args.rank}")
    print(f"  更新层: {args.attention_layers_to_update}")
    print(f"  冻结Prompt Encoder: {args.freeze_prompt_encoder}")
    print()
    print("🏋️ 训练配置:")
    print(f"  训练轮数: {args.n_epochs}")
    print(f"  批处理大小: {args.batch_size}")
    print(f"  每批次对象数: {args.n_objects_per_batch}")
    print(f"  补丁大小: {args.patch_height}x{args.patch_width}")
    print(f"  最小实例大小: {args.min_instance_size}")
    print(f"  设备: cuda:{args.gpu_id} (当前设置: {device})")
    print(f"  保存路径: {args.save_root}")
    print("=" * 80)


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 验证路径
    validate_paths(args)
    
    # 设置设备
    device = setup_device(args.gpu_id)
    
    # 加载数据路径
    print("📂 加载数据路径...")
    train_image_paths, train_mask_paths = load_data_paths(args.train_image_dir, args.train_mask_dir)
    val_image_paths, val_mask_paths = load_data_paths(args.val_image_dir, args.val_mask_dir)
    
    print(f"训练样本数: {len(train_image_paths)}")
    print(f"验证样本数: {len(val_image_paths)}")
    
    # 可视化样本（可选）
    if args.visualize_first_sample:
        print("🖼️ 可视化第一个训练样本...")
        visualize_sample(train_image_paths, train_mask_paths)
    
    # 创建数据加载器
    print("🔄 创建数据加载器...")

    # if args.train_number==0:
    train_loader = create_data_loader(train_image_paths, train_mask_paths, args, is_train=True)
    val_loader = create_data_loader(val_image_paths, val_mask_paths, args, is_train=False)
    # else:
    #     train_loader = create_data_loader(train_image_paths[:args.train_number], train_mask_paths, args, is_train=True)
    #     val_loader = create_data_loader(val_image_paths[:args.train_number], val_mask_paths, args, is_train=False)
    
    # 打印配置
    print_training_config(args, device)
    
    # 设置LoRA参数
    peft_kwargs = {
        "rank": args.rank,
        "attention_layers_to_update": args.attention_layers_to_update,
    }
    
    if args.freeze_prompt_encoder:
        freeze=['prompt_encoder']
    else:
        freeze=[]
    
    # 开始训练
    print("🚀 开始训练...")
    try:
        sam_training.train_sam(
            name=args.checkpoint_name,
            save_root=args.save_root,
            model_type=args.model_type,
            checkpoint_path=args.pretrained_checkpoint,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=args.n_epochs,
            n_objects_per_batch=args.n_objects_per_batch,
            with_segmentation_decoder=args.train_instance_segmentation,
            device=device,
            peft_kwargs=peft_kwargs,
            freeze=freeze,
        )
        
        print("=" * 80)
        print("✅ 训练完成!")
        print(f"📁 模型保存到: {args.save_root}")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()