'''
 # @ Author: Zhenhua Chen
 # @ Create Time: 2025-05-29 05:39:19
 # @ Email: Zhenhua.Chen@gmail.com
 # @ Description:
 '''

# test_batch_fixed_with_real_data.py - 使用真实数据的批量推理测试
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import numpy as np
import time
import json
from tqdm import tqdm
from typing import List, Dict, Optional

class RealDataBatchInferenceTest:
    """使用真实测试数据的批量推理测试"""
    
    def __init__(self, lora_model_path: str, split_file: str):
        self.lora_model_path = lora_model_path
        self.split_file = split_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.test_samples = []
        
    def load_test_samples(self):
        """从数据划分文件加载测试样本"""
        print(f"正在加载数据划分文件: {self.split_file}")
        
        try:
            # 加载数据划分
            with open(self.split_file, 'r', encoding='utf-8') as f:
                split_data = json.load(f)
            
            from utils.data_splitter import DataSplit
            split_result = DataSplit.from_dict(split_data)
            self.test_samples = split_result.test_samples
            
            print(f"加载了 {len(self.test_samples)} 个测试样本")
            
            # 验证样本路径的有效性
            valid_samples = []
            for sample in self.test_samples:
                img_path = Path(sample['image_path'])
                mask_path = Path(sample['mask_path'])
                if img_path.exists() and mask_path.exists():
                    valid_samples.append(sample)
                else:
                    print(f"警告: 样本路径不存在 - {sample['sample_id']}")
            
            self.test_samples = valid_samples
            print(f"有效测试样本数: {len(self.test_samples)}")
            
            return len(self.test_samples) > 0
            
        except Exception as e:
            print(f"加载测试样本失败: {e}")
            return False
    
    def load_fixed_model(self):
        """加载修复版本的LoRA模型"""
        from lora.stable_sam_lora_wrapper import load_stable_sam_lora_model as load_sam_lora_model
        
        print("正在加载修复版本的LoRA模型...")
        self.model = load_sam_lora_model("vit_b_lm", self.lora_model_path, str(self.device))
        
        if self.model is None:
            print("❌ 修复版本模型加载失败")
            return False
        
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"✅ 修复版本模型加载成功，设备: {self.device}")
        return True
    
    def load_real_images(self, sample_indices: List[int]) -> Optional[torch.Tensor]:
        """加载真实图像数据"""
        from utils.file_utils import load_image
        
        images = []
        valid_indices = []
        
        for idx in sample_indices:
            if idx >= len(self.test_samples):
                continue
                
            sample = self.test_samples[idx]
            img_path = sample['image_path']
            
            try:
                # 加载图像
                image = load_image(img_path, convert_to_grayscale=False)
                if image is None:
                    continue
                
                # 转换为张量
                if len(image.shape) == 2:
                    # 灰度图转RGB
                    image = np.stack([image] * 3, axis=-1)
                elif image.shape[-1] == 1:
                    image = np.repeat(image, 3, axis=-1)
                
                # 转换为张量并归一化
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                
                # 调整到1024x1024
                if image_tensor.shape[-2:] != (1024, 1024):
                    image_tensor = torch.nn.functional.interpolate(
                        image_tensor.unsqueeze(0), 
                        size=(1024, 1024), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)
                
                images.append(image_tensor)
                valid_indices.append(idx)
                
            except Exception as e:
                print(f"加载图像失败 {img_path}: {e}")
                continue
        
        if not images:
            return None
        
        batch_images = torch.stack(images).to(self.device)
        print(f"成功加载 {len(images)} 张真实图像，形状: {batch_images.shape}")
        
        return batch_images, valid_indices
    
    def test_batch_sizes_with_real_data(self, batch_sizes: List[int] = [1, 2, 4, 8, 16]):
        """使用真实数据测试不同的批次大小"""
        
        # 先加载测试样本
        if not self.load_test_samples():
            print("❌ 无法加载测试样本")
            return
        
        # 加载模型
        if not self.load_fixed_model():
            print("❌ 无法加载模型")
            return
        
        print(f"\n{'='*80}")
        print("🧪 使用真实数据测试修复版本的批量推理")
        print(f"{'='*80}")
        print(f"测试样本总数: {len(self.test_samples)}")
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\n📊 测试 Batch Size = {batch_size}")
            print("-" * 50)
            
            # 检查是否有足够的样本
            if batch_size > len(self.test_samples):
                print(f"   ⚠️  批次大小 {batch_size} 超过可用样本数 {len(self.test_samples)}，跳过")
                results[batch_size] = {
                    'success': False,
                    'error': f'Insufficient samples ({len(self.test_samples)} < {batch_size})'
                }
                continue
            
            try:
                # 选择样本索引
                sample_indices = list(range(batch_size))
                
                # 加载真实图像
                loaded_data = self.load_real_images(sample_indices)
                if loaded_data is None:
                    print(f"   ❌ 无法加载足够的有效图像")
                    results[batch_size] = {
                        'success': False,
                        'error': 'Failed to load valid images'
                    }
                    continue
                
                real_images, valid_indices = loaded_data
                actual_batch_size = real_images.shape[0]
                
                print(f"   📷 实际加载图像数: {actual_batch_size}")
                
                # 准备输入
                batch_inputs = {
                    'images': real_images,
                    'point_coords': [],
                    'point_labels': [],
                    'boxes': [],
                    'mask_inputs': None,
                    'multimask_output': False
                }
                
                # 测试推理
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = self.model(batch_inputs)
                
                inference_time = time.time() - start_time
                
                # 验证输出
                masks = outputs['masks']
                iou_predictions = outputs['iou_predictions']
                
                print(f"   ✅ 推理成功!")
                print(f"   📏 输出形状:")
                print(f"      掩码: {masks.shape}")
                print(f"      IoU预测: {iou_predictions.shape}")
                print(f"   ⏱️  推理时间: {inference_time:.4f}s")
                print(f"   📈 平均时间/图像: {inference_time/actual_batch_size:.4f}s")
                
                # 显示处理的样本信息
                print(f"   📋 处理的样本:")
                for i, idx in enumerate(valid_indices[:3]):  # 只显示前3个
                    sample = self.test_samples[idx]
                    print(f"      {i+1}. {sample['sample_id']} ({sample['cell_type']})")
                if len(valid_indices) > 3:
                    print(f"      ... 还有 {len(valid_indices) - 3} 个样本")
                
                # 验证输出的合理性
                validation_result = self._validate_outputs(masks, iou_predictions, actual_batch_size)
                if validation_result['valid']:
                    print(f"   ✅ 输出验证通过")
                else:
                    print(f"   ⚠️  输出验证警告: {validation_result['warnings']}")
                
                # 保存结果
                results[batch_size] = {
                    'success': True,
                    'inference_time': inference_time,
                    'time_per_image': inference_time / actual_batch_size,
                    'actual_batch_size': actual_batch_size,
                    'processed_samples': [self.test_samples[idx]['sample_id'] for idx in valid_indices],
                    'output_shapes': {
                        'masks': list(masks.shape),
                        'iou_predictions': list(iou_predictions.shape)
                    },
                    'validation': validation_result
                }
                
            except Exception as e:
                print(f"   ❌ 推理失败: {e}")
                import traceback
                traceback.print_exc()
                results[batch_size] = {
                    'success': False,
                    'error': str(e)
                }
        
        # 打印总结
        self._print_test_summary(results)
        return results
    
    def _validate_outputs(self, masks: torch.Tensor, iou_predictions: torch.Tensor, batch_size: int) -> dict:
        """验证输出的合理性"""
        validation = {'valid': True, 'warnings': []}
        
        # 检查形状
        expected_mask_shape = (batch_size, 1, 256, 256)
        expected_iou_shape = (batch_size, 1)
        
        if masks.shape != expected_mask_shape:
            validation['warnings'].append(f"掩码形状异常: 期望{expected_mask_shape}, 实际{masks.shape}")
        
        if iou_predictions.shape != expected_iou_shape:
            validation['warnings'].append(f"IoU形状异常: 期望{expected_iou_shape}, 实际{iou_predictions.shape}")
        
        # 检查数值范围
        mask_min, mask_max = masks.min().item(), masks.max().item()
        iou_min, iou_max = iou_predictions.min().item(), iou_predictions.max().item()
        
        if mask_min < -10 or mask_max > 10:
            validation['warnings'].append(f"掩码数值范围异常: [{mask_min:.3f}, {mask_max:.3f}]")
        
        if iou_min < 0 or iou_max > 1:
            validation['warnings'].append(f"IoU数值范围异常: [{iou_min:.3f}, {iou_max:.3f}]")
        
        # 检查是否有NaN或Inf
        if torch.isnan(masks).any():
            validation['warnings'].append("掩码包含NaN值")
        
        if torch.isinf(masks).any():
            validation['warnings'].append("掩码包含Inf值")
        
        if torch.isnan(iou_predictions).any():
            validation['warnings'].append("IoU预测包含NaN值")
        
        if validation['warnings']:
            validation['valid'] = False
        
        return validation
    
    def _print_test_summary(self, results: dict):
        """打印测试总结"""
        print(f"\n{'='*80}")
        print("📋 真实数据批量推理测试总结")
        print(f"{'='*80}")
        
        successful_batches = [bs for bs, result in results.items() if result['success']]
        failed_batches = [bs for bs, result in results.items() if not result['success']]
        
        print(f"✅ 成功的批次大小: {successful_batches}")
        if failed_batches:
            print(f"❌ 失败的批次大小: {failed_batches}")
            print(f"失败原因:")
            for bs in failed_batches:
                print(f"  {bs}: {results[bs]['error']}")
        
        if successful_batches:
            print(f"\n📊 性能统计:")
            print(f"{'批次大小':<8} {'实际批次':<8} {'总时间(s)':<12} {'平均时间/图像(s)':<18} {'吞吐量(图像/s)':<15}")
            print("-" * 70)
            
            for batch_size in successful_batches:
                result = results[batch_size]
                actual_size = result['actual_batch_size']
                total_time = result['inference_time']
                time_per_image = result['time_per_image']
                throughput = 1.0 / time_per_image
                
                print(f"{batch_size:<8} {actual_size:<8} {total_time:<12.4f} {time_per_image:<18.4f} {throughput:<15.2f}")
        
        print(f"\n🎯 结论:")
        if len(successful_batches) == len(results):
            print("   ✅ 所有批次大小都成功！真实数据批量推理完全正常！")
        elif successful_batches:
            max_successful = max(successful_batches)
            print(f"   ⚠️  部分成功，最大支持批次大小: {max_successful}")
        else:
            print("   ❌ 所有批次都失败，需要进一步调试")
    
    def test_specific_samples(self, sample_indices: List[int]):
        """测试特定的样本"""
        if not self.test_samples:
            print("❌ 请先加载测试样本")
            return
        
        if not self.model:
            if not self.load_fixed_model():
                return
        
        print(f"\n🎯 测试特定样本 (索引: {sample_indices})")
        print("-" * 50)
        
        try:
            # 加载指定样本的图像
            loaded_data = self.load_real_images(sample_indices)
            if loaded_data is None:
                print("❌ 无法加载指定样本的图像")
                return
            
            real_images, valid_indices = loaded_data
            
            # 准备输入
            batch_inputs = {
                'images': real_images,
                'point_coords': [],
                'point_labels': [],
                'boxes': [],
                'mask_inputs': None,
                'multimask_output': False
            }
            
            # 推理
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(batch_inputs)
            inference_time = time.time() - start_time
            
            print(f"✅ 推理成功!")
            print(f"📏 输出形状: 掩码{outputs['masks'].shape}, IoU{outputs['iou_predictions'].shape}")
            print(f"⏱️  推理时间: {inference_time:.4f}s")
            print(f"📈 平均时间/图像: {inference_time/len(valid_indices):.4f}s")
            
            # 显示处理的样本详情
            print(f"\n📋 处理的样本详情:")
            for i, idx in enumerate(valid_indices):
                sample = self.test_samples[idx]
                mask_val = torch.sigmoid(outputs['masks'][i]).mean().item()
                iou_val = outputs['iou_predictions'][i].item()
                print(f"  {i+1}. {sample['sample_id']} ({sample['cell_type']})")
                print(f"     掩码均值: {mask_val:.4f}, IoU预测: {iou_val:.4f}")
                print(f"     图像路径: {sample['image_path']}")
            
            return outputs
            
        except Exception as e:
            print(f"❌ 测试特定样本失败: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """主测试函数"""
    # 配置路径
    lora_model_path = "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/lora_model/lora_model_293t_train0_val10_test90/lora_finetune_vit_b_lm_r8/final_model"
    split_file = "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/lora_split/split_0.05_0.00_0.95_293T_32f483d5bd91b97e.json"
    
    print("🚀 开始使用真实数据测试修复版本的批量推理")
    print(f"LoRA模型路径: {lora_model_path}")
    print(f"数据划分文件: {split_file}")
    
    # 创建测试实例
    tester = RealDataBatchInferenceTest(lora_model_path, split_file)
    
    # 测试1: 不同批次大小
    print("\n" + "="*80)
    print("测试1: 不同批次大小的真实数据推理能力")
    print("="*80)
    batch_results = tester.test_batch_sizes_with_real_data([1, 2, 4, 8, 16])
    
    # 测试2: 测试特定样本
    print("\n" + "="*80)
    print("测试2: 特定样本详细测试")
    print("="*80)
    if tester.test_samples:
        # 测试前5个样本
        sample_indices = list(range(min(5, len(tester.test_samples))))
        tester.test_specific_samples(sample_indices)
    
    print(f"\n{'='*80}")
    print("🎉 真实数据批量推理测试完成!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()