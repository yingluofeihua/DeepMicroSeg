# test_lora_with_split.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

from lora.sam_lora_wrapper import load_sam_lora_model
from core.metrics import ComprehensiveMetrics
from lora.data_loaders import SAMDataset, collate_fn
from config.lora_config import DataConfig
from utils.data_splitter import DataSplit
from utils.file_utils import load_image, load_mask

def test_lora_with_existing_split():
    # 配置路径
    lora_model_path = "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/lora_model_293t_train-5_val10_test95/lora_finetune_vit_b_lm_r8/final_model"
    split_file = "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/lora_split/split_0.05_0.00_0.95_293T_32f483d5bd91b97e.json"
    output_dir = "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/results/lora_test_0.95_293T"
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("正在加载数据划分文件...")
    # 加载数据划分
    with open(split_file, 'r', encoding='utf-8') as f:
        split_data = json.load(f)
    
    split_result = DataSplit.from_dict(split_data)
    test_samples = split_result.test_samples
    
    print(f"测试集样本数: {len(test_samples)}")
    print(f"训练集样本数: {len(split_result.train_samples)}")
    print(f"验证集样本数: {len(split_result.val_samples)}")
    
    # 验证测试样本路径
    valid_test_samples = []
    for sample in test_samples:
        img_path = Path(sample['image_path'])
        mask_path = Path(sample['mask_path'])
        if img_path.exists() and mask_path.exists():
            valid_test_samples.append(sample)
        else:
            print(f"警告: 样本路径不存在 - {sample['sample_id']}")
    
    print(f"有效测试样本数: {len(valid_test_samples)}")
    
    # 配置数据加载器
    config = DataConfig()
    config.batch_size = 1
    config._cell_types_filter = ['293T']
    
    # 创建测试数据集
    test_dataset = SAMDataset(
        data_dir=None,  # 不使用目录，直接传入样本
        config=config,
        split='test',
        samples=valid_test_samples
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    print("正在加载LoRA模型...")
    # 加载LoRA模型
    lora_model = load_sam_lora_model("vit_b_lm", lora_model_path)
    if lora_model is None:
        print("LoRA模型加载失败")
        return
    
    lora_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_model = lora_model.to(device)
    print(f"模型已加载到设备: {device}")
    
    # 创建指标计算器
    metrics_calculator = ComprehensiveMetrics(enable_hd95=True)
    
    # 测试循环
    results = []
    print("开始测试...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing LoRA Model")):
            try:
                # 准备输入
                from lora.training_utils import prepare_sam_inputs
                
                inputs, targets = prepare_sam_inputs(batch)
                
                # 移动到设备
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        inputs[key] = value.to(device)
                    elif isinstance(value, list):
                        inputs[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
                
                for key, value in targets.items():
                    if isinstance(value, torch.Tensor):
                        targets[key] = value.to(device)
                
                # 计算处理时间
                start_time = time.time()
                
                # 模型推理
                predictions = lora_model(inputs)
                
                processing_time = time.time() - start_time
                
                # 后处理预测结果
                pred_masks = torch.sigmoid(predictions['masks']).cpu().numpy()
                target_masks = targets['masks'].cpu().numpy()
                
                # 获取样本信息
                sample_id = batch['sample_ids'][0] if batch['sample_ids'] else f"sample_{batch_idx}"
                
                # 处理每个批次中的样本（虽然batch_size=1）
                for i in range(pred_masks.shape[0]):
                    # 处理预测掩码
                    pred_mask = pred_masks[i]
                    target_mask = target_masks[i]
                    
                    # 合并多个实例为单个二进制掩码
                    if len(pred_mask.shape) > 2:
                        pred_mask = pred_mask[0] if pred_mask.shape[0] > 0 else np.zeros(pred_mask.shape[1:])
                    
                    if len(target_mask.shape) > 2:
                        # 将多实例目标合并为二进制掩码
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
                    metrics_result = metrics_calculator.compute_all_metrics(target_binary, pred_binary)
                    
                    # 保存结果
                    result = metrics_result.to_dict()
                    result.update({
                        'sample_id': sample_id,
                        'cell_type': '293T',
                        'processing_time': processing_time,
                        'image_path': valid_test_samples[batch_idx]['image_path'] if batch_idx < len(valid_test_samples) else '',
                        'mask_path': valid_test_samples[batch_idx]['mask_path'] if batch_idx < len(valid_test_samples) else ''
                    })
                    
                    results.append(result)
                
                # 可选：限制测试数量
                # if batch_idx >= 100:
                #     break
                    
            except Exception as e:
                print(f"处理批次 {batch_idx} 失败: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 保存和分析结果
    if results:
        # 保存详细结果
        df = pd.DataFrame(results)
        df.to_csv(f"{output_dir}/detailed_test_results.csv", index=False)
        
        # 计算平均指标
        metrics_cols = ['ap50', 'ap75', 'iou_score', 'dice_score', 'hd95', 'gt_instances', 'pred_instances', 'processing_time']
        avg_metrics = {}
        
        for col in metrics_cols:
            if col in df.columns:
                if col == 'hd95':
                    # 特殊处理HD95的无穷值
                    finite_values = df[col][np.isfinite(df[col])]
                    avg_metrics[f'avg_{col}'] = finite_values.mean() if len(finite_values) > 0 else float('inf')
                    avg_metrics[f'median_{col}'] = finite_values.median() if len(finite_values) > 0 else float('inf')
                    avg_metrics[f'finite_count_{col}'] = len(finite_values)
                else:
                    avg_metrics[f'avg_{col}'] = df[col].mean()
                    avg_metrics[f'std_{col}'] = df[col].std()
                    avg_metrics[f'median_{col}'] = df[col].median()
        
        # 保存平均结果
        with open(f"{output_dir}/summary_test_results.json", 'w') as f:
            json.dump(avg_metrics, f, indent=2)
        
        # 保存测试配置信息
        test_info = {
            'lora_model_path': lora_model_path,
            'split_file': split_file,
            'total_test_samples': len(test_samples),
            'valid_test_samples': len(valid_test_samples),
            'successfully_processed': len(results),
            'device': str(device),
            'model_type': 'vit_b_lm',
            'cell_type': '293T'
        }
        
        with open(f"{output_dir}/test_info.json", 'w') as f:
            json.dump(test_info, f, indent=2)
        
        # 打印结果摘要
        print(f"\n{'='*60}")
        print("LoRA模型测试完成!")
        print(f"{'='*60}")
        print(f"结果保存在: {output_dir}")
        print(f"测试样本数: {len(results)}")
        print(f"数据划分文件: {Path(split_file).name}")
        
        print(f"\n指标摘要:")
        key_metrics = ['avg_ap50', 'avg_ap75', 'avg_iou_score', 'avg_dice_score', 'avg_hd95', 'avg_processing_time']
        for metric in key_metrics:
            if metric in avg_metrics:
                value = avg_metrics[metric]
                if 'hd95' in metric and value == float('inf'):
                    print(f"  {metric.replace('avg_', '').upper()}: ∞")
                else:
                    print(f"  {metric.replace('avg_', '').upper()}: {value:.4f}")
        
        print(f"\n实例统计:")
        print(f"  平均GT实例数: {avg_metrics.get('avg_gt_instances', 0):.1f}")
        print(f"  平均预测实例数: {avg_metrics.get('avg_pred_instances', 0):.1f}")
        
    else:
        print("没有成功处理的测试样本")

if __name__ == "__main__":
    test_lora_with_existing_split()