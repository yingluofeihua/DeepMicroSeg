'''
 # @ Author: Zhenhua Chen
 # @ Create Time: 2025-05-28 03:13:12
 # @ Email: Zhenhua.Chen@gmail.com
 # @ Description:
 '''

# test_batch_fixed.py - 测试修复版本的批量推理
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import numpy as np
import time
from tqdm import tqdm

class FixedBatchInferenceTest:
    """修复版本的批量推理测试"""
    
    def __init__(self, lora_model_path: str):
        self.lora_model_path = lora_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
    def load_fixed_model(self):
        """加载修复版本的模型"""
        # 直接导入修复版本的函数
        # from lora.stable_sam_lora_wrapper import load_fixed_sam_lora_model
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
    
    def test_batch_sizes(self, batch_sizes: list = [1, 2, 4, 8, 16]):
        """测试不同的批次大小"""
        if not self.load_fixed_model():
            return
        
        print(f"\n{'='*80}")
        print("🧪 测试修复版本的批量推理")
        print(f"{'='*80}")
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\n📊 测试 Batch Size = {batch_size}")
            print("-" * 50)
            
            try:
                # 创建测试数据
                test_images = torch.randn(batch_size, 3, 1024, 1024, device=self.device)
                
                # 准备输入
                batch_inputs = {
                    'images': test_images,
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
                print(f"   📈 平均时间/图像: {inference_time/batch_size:.4f}s")
                
                # 验证输出的合理性
                validation_result = self._validate_outputs(masks, iou_predictions, batch_size)
                if validation_result['valid']:
                    print(f"   ✅ 输出验证通过")
                else:
                    print(f"   ⚠️  输出验证警告: {validation_result['warnings']}")
                
                # 保存结果
                results[batch_size] = {
                    'success': True,
                    'inference_time': inference_time,
                    'time_per_image': inference_time / batch_size,
                    'output_shapes': {
                        'masks': list(masks.shape),
                        'iou_predictions': list(iou_predictions.shape)
                    },
                    'validation': validation_result
                }
                
            except Exception as e:
                print(f"   ❌ 推理失败: {e}")
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
        print("📋 批量推理测试总结")
        print(f"{'='*80}")
        
        successful_batches = [bs for bs, result in results.items() if result['success']]
        failed_batches = [bs for bs, result in results.items() if not result['success']]
        
        print(f"✅ 成功的批次大小: {successful_batches}")
        if failed_batches:
            print(f"❌ 失败的批次大小: {failed_batches}")
        
        if successful_batches:
            print(f"\n📊 性能统计:")
            print(f"{'批次大小':<8} {'总时间(s)':<12} {'平均时间/图像(s)':<18} {'吞吐量(图像/s)':<15}")
            print("-" * 60)
            
            for batch_size in successful_batches:
                result = results[batch_size]
                total_time = result['inference_time']
                time_per_image = result['time_per_image']
                throughput = 1.0 / time_per_image
                
                print(f"{batch_size:<8} {total_time:<12.4f} {time_per_image:<18.4f} {throughput:<15.2f}")
        
        print(f"\n🎯 结论:")
        if len(successful_batches) == len(results):
            print("   ✅ 所有批次大小都成功！批量推理修复成功！")
        elif successful_batches:
            max_successful = max(successful_batches)
            print(f"   ⚠️  部分成功，最大支持批次大小: {max_successful}")
        else:
            print("   ❌ 所有批次都失败，需要进一步调试")
    
    def benchmark_performance(self, batch_size: int = 8, num_iterations: int = 10):
        """性能基准测试"""
        if not self.model:
            if not self.load_fixed_model():
                return
        
        print(f"\n🏃‍♂️ 性能基准测试 (批次大小: {batch_size}, 迭代次数: {num_iterations})")
        print("-" * 60)
        
        # 预热
        print("预热中...")
        for _ in range(3):
            test_images = torch.randn(batch_size, 3, 1024, 1024, device=self.device)
            batch_inputs = {
                'images': test_images,
                'point_coords': [],
                'point_labels': [],
                'boxes': [],
                'mask_inputs': None,
                'multimask_output': False
            }
            
            with torch.no_grad():
                _ = self.model(batch_inputs)
        
        # 正式测试
        print("开始基准测试...")
        times = []
        
        for i in tqdm(range(num_iterations), desc="基准测试"):
            test_images = torch.randn(batch_size, 3, 1024, 1024, device=self.device)
            batch_inputs = {
                'images': test_images,
                'point_coords': [],
                'point_labels': [],
                'boxes': [],
                'mask_inputs': None,
                'multimask_output': False
            }
            
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model(batch_inputs)
            
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        # 统计结果
        times = np.array(times)
        mean_time = times.mean()
        std_time = times.std()
        min_time = times.min()
        max_time = times.max()
        
        time_per_image = mean_time / batch_size
        throughput = batch_size / mean_time
        
        print(f"\n📊 基准测试结果:")
        print(f"   平均推理时间: {mean_time:.4f}s ± {std_time:.4f}s")
        print(f"   最快推理时间: {min_time:.4f}s")
        print(f"   最慢推理时间: {max_time:.4f}s")
        print(f"   平均时间/图像: {time_per_image:.4f}s")
        print(f"   吞吐量: {throughput:.2f} 图像/秒")
        
        return {
            'mean_time': mean_time,
            'std_time': std_time,
            'min_time': min_time,
            'max_time': max_time,
            'time_per_image': time_per_image,
            'throughput': throughput
        }
    
    def test_with_real_data(self, data_samples: list):
        """使用真实数据测试"""
        if not self.model:
            if not self.load_fixed_model():
                return
        
        print(f"\n🖼️  真实数据测试 ({len(data_samples)} 个样本)")
        print("-" * 50)
        
        try:
            # 准备真实数据批次
            images_list = []
            for sample in data_samples:
                # 假设sample是图像路径或已加载的图像
                if isinstance(sample, str):
                    # 加载图像的逻辑
                    from utils.file_utils import load_image
                    image = load_image(sample, convert_to_grayscale=False)
                    if image is not None:
                        images_list.append(torch.from_numpy(image).permute(2, 0, 1).float() / 255.0)
                elif isinstance(sample, torch.Tensor):
                    images_list.append(sample)
            
            if not images_list:
                print("   ❌ 没有有效的图像数据")
                return
            
            # 调整到统一尺寸并创建批次
            processed_images = []
            for img in images_list:
                if img.shape[-2:] != (1024, 1024):
                    img = torch.nn.functional.interpolate(
                        img.unsqueeze(0), size=(1024, 1024), mode='bilinear', align_corners=False
                    ).squeeze(0)
                processed_images.append(img)
            
            # 创建批次张量
            batch_images = torch.stack(processed_images).to(self.device)
            
            batch_inputs = {
                'images': batch_images,
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
            
            print(f"   ✅ 真实数据推理成功!")
            print(f"   📏 输出形状: 掩码{outputs['masks'].shape}, IoU{outputs['iou_predictions'].shape}")
            print(f"   ⏱️  推理时间: {inference_time:.4f}s")
            print(f"   📈 平均时间/图像: {inference_time/len(data_samples):.4f}s")
            
            return outputs
            
        except Exception as e:
            print(f"   ❌ 真实数据测试失败: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """主测试函数"""
    # 配置路径
    lora_model_path = "/LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/lora_model/lora_model_293t_train0_val10_test90/lora_finetune_vit_b_lm_r8/final_model"
    
    print("🚀 开始测试修复版本的批量推理")
    print(f"模型路径: {lora_model_path}")
    
    # 创建测试实例
    tester = FixedBatchInferenceTest(lora_model_path)
    
    # 测试1: 不同批次大小
    print("\n" + "="*80)
    print("测试1: 不同批次大小的推理能力")
    print("="*80)
    batch_results = tester.test_batch_sizes([1, 2, 4, 8, 16, 32])
    
    # 测试2: 性能基准测试
    if batch_results and any(r['success'] for r in batch_results.values()):
        print("\n" + "="*80)
        print("测试2: 性能基准测试")
        print("="*80)
        
        # 找到最大成功的批次大小进行基准测试
        successful_batches = [bs for bs, result in batch_results.items() if result['success']]
        if successful_batches:
            optimal_batch_size = min(8, max(successful_batches))  # 最大8，或者最大成功批次
            tester.benchmark_performance(batch_size=optimal_batch_size, num_iterations=20)
    
    print(f"\n{'='*80}")
    print("🎉 修复版本批量推理测试完成!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()