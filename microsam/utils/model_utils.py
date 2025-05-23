"""
模型工具函数
包含模型权重转换、设备管理、内存优化等工具
"""

import torch
import torch.nn as nn
import gc
import psutil
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import time

# 尝试导入GPUtil（可选）
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False


def get_device_info() -> Dict[str, Any]:
    """获取设备信息"""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': 0,
        'current_device': None,
        'gpu_memory': {},
        'cpu_memory': {}
    }
    
    if torch.cuda.is_available():
        info['cuda_device_count'] = torch.cuda.device_count()
        info['current_device'] = torch.cuda.current_device()
        
        # GPU内存信息
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_info = torch.cuda.mem_get_info(i)
            
            info['gpu_memory'][f'gpu_{i}'] = {
                'name': props.name,
                'total_memory': props.total_memory,
                'free_memory': memory_info[0],
                'used_memory': props.total_memory - memory_info[0],
                'memory_utilization': (props.total_memory - memory_info[0]) / props.total_memory * 100
            }
    
    # CPU内存信息
    memory = psutil.virtual_memory()
    info['cpu_memory'] = {
        'total': memory.total,
        'available': memory.available,
        'used': memory.used,
        'percentage': memory.percent
    }
    
    return info


def optimize_memory():
    """优化内存使用"""
    # Python垃圾回收
    gc.collect()
    
    # CUDA内存清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_model_memory_usage(model: nn.Module) -> Dict[str, int]:
    """获取模型内存使用情况"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    return {
        'parameters_size_bytes': param_size,
        'buffers_size_bytes': buffer_size,
        'total_size_bytes': total_size,
        'parameters_size_mb': param_size / (1024 ** 2),
        'buffers_size_mb': buffer_size / (1024 ** 2),
        'total_size_mb': total_size / (1024 ** 2)
    }


def count_parameters(model: nn.Module, only_trainable: bool = False) -> Dict[str, int]:
    """统计模型参数"""
    if only_trainable:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'trainable_parameters': total_params}
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0
        }


def freeze_model_components(model: nn.Module, components_to_freeze: List[str]):
    """冻结模型的指定组件"""
    frozen_count = 0
    
    for name, module in model.named_modules():
        if any(component in name for component in components_to_freeze):
            for param in module.parameters():
                param.requires_grad = False
                frozen_count += 1
    
    print(f"冻结了 {frozen_count} 个参数")


def unfreeze_model_components(model: nn.Module, components_to_unfreeze: List[str]):
    """解冻模型的指定组件"""
    unfrozen_count = 0
    
    for name, module in model.named_modules():
        if any(component in name for component in components_to_unfreeze):
            for param in module.parameters():
                param.requires_grad = True
                unfrozen_count += 1
    
    print(f"解冻了 {unfrozen_count} 个参数")


def move_model_to_device(model: nn.Module, device: str, verbose: bool = True) -> nn.Module:
    """将模型移动到指定设备"""
    try:
        if verbose:
            print(f"正在将模型移动到设备: {device}")
        
        model = model.to(device)
        
        if verbose:
            print(f"模型已成功移动到: {device}")
        
        return model
        
    except Exception as e:
        print(f"移动模型到设备失败: {e}")
        return model


def save_model_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    loss: float,
    save_path: str,
    additional_info: Optional[Dict] = None
):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'timestamp': time.time()
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if additional_info:
        checkpoint.update(additional_info)
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        torch.save(checkpoint, save_path)
        print(f"检查点已保存: {save_path}")
        return True
    except Exception as e:
        print(f"保存检查点失败: {e}")
        return False


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cuda"
) -> Dict[str, Any]:
    """加载模型检查点"""
    try:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"检查点文件不存在: {checkpoint_path}")
            return {}
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 加载模型状态
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"模型状态已加载")
        
        # 加载优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"优化器状态已加载")
        
        print(f"检查点加载成功: {checkpoint_path}")
        return checkpoint
        
    except Exception as e:
        print(f"加载检查点失败: {e}")
        return {}


def convert_model_precision(model: nn.Module, precision: str = "fp16") -> nn.Module:
    """转换模型精度"""
    if precision == "fp16":
        model = model.half()
        print("模型已转换为FP16精度")
    elif precision == "fp32":
        model = model.float()
        print("模型已转换为FP32精度")
    else:
        print(f"不支持的精度类型: {precision}")
    
    return model


def enable_gradient_checkpointing(model: nn.Module):
    """启用梯度检查点以节省内存"""
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("梯度检查点已启用")
    else:
        print("模型不支持梯度检查点")


def get_model_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> Optional[int]:
    """计算模型的FLOPs（需要额外的库支持）"""
    try:
        # 这里可以集成FLOPs计算库，如thop或fvcore
        # from thop import profile
        # input_tensor = torch.randn(1, *input_shape)
        # flops, params = profile(model, inputs=(input_tensor,))
        # return flops
        print("FLOPs计算需要额外的库支持")
        return None
    except ImportError:
        print("FLOPs计算库未安装")
        return None


def compare_model_outputs(
    model1: nn.Module, 
    model2: nn.Module, 
    input_tensor: torch.Tensor,
    tolerance: float = 1e-5
) -> Dict[str, Any]:
    """比较两个模型的输出"""
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        output1 = model1(input_tensor)
        output2 = model2(input_tensor)
    
    # 计算差异
    if isinstance(output1, dict) and isinstance(output2, dict):
        differences = {}
        for key in output1.keys():
            if key in output2:
                diff = torch.abs(output1[key] - output2[key])
                differences[key] = {
                    'max_diff': diff.max().item(),
                    'mean_diff': diff.mean().item(),
                    'within_tolerance': (diff < tolerance).all().item()
                }
    else:
        diff = torch.abs(output1 - output2)
        differences = {
            'max_diff': diff.max().item(),
            'mean_diff': diff.mean().item(),
            'within_tolerance': (diff < tolerance).all().item()
        }
    
    return differences


def profile_model_memory(model: nn.Module, input_shape: Tuple[int, ...], device: str = "cuda"):
    """分析模型内存使用"""
    if not torch.cuda.is_available() or device == "cpu":
        print("内存分析需要CUDA支持")
        return None
    
    model = model.to(device)
    input_tensor = torch.randn(1, *input_shape, device=device)
    
    # 记录初始内存
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated()
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    
    # 记录前向传播后内存
    forward_memory = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()
    
    memory_info = {
        'initial_memory_mb': initial_memory / (1024 ** 2),
        'forward_memory_mb': forward_memory / (1024 ** 2),
        'peak_memory_mb': peak_memory / (1024 ** 2),
        'memory_increase_mb': (forward_memory - initial_memory) / (1024 ** 2)
    }
    
    return memory_info


def export_model_to_onnx(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    export_path: str,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None
) -> bool:
    """导出模型到ONNX格式"""
    try:
        model.eval()
        dummy_input = torch.randn(1, *input_shape)
        
        torch.onnx.export(
            model,
            dummy_input,
            export_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=input_names or ['input'],
            output_names=output_names or ['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"模型已导出到ONNX: {export_path}")
        return True
        
    except Exception as e:
        print(f"ONNX导出失败: {e}")
        return False


class ModelProfiler:
    """模型性能分析器"""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.timing_results = []
        self.memory_results = []
    
    def profile_inference(self, input_tensor: torch.Tensor, num_runs: int = 100) -> Dict[str, float]:
        """分析推理性能"""
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        
        # 预热
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(input_tensor)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 计时
        times = []
        
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            with torch.no_grad():
                _ = self.model(input_tensor)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            'mean_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'std_time': (sum([(t - sum(times)/len(times))**2 for t in times]) / len(times)) ** 0.5
        }
    
    def profile_training_step(self, 
                            input_tensor: torch.Tensor, 
                            target_tensor: torch.Tensor,
                            loss_fn: nn.Module,
                            optimizer: torch.optim.Optimizer,
                            num_runs: int = 10) -> Dict[str, float]:
        """分析训练步骤性能"""
        self.model.train()
        input_tensor = input_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)
        
        forward_times = []
        backward_times = []
        total_times = []
        
        for _ in range(num_runs):
            optimizer.zero_grad()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            # 前向传播
            output = self.model(input_tensor)
            loss = loss_fn(output, target_tensor)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            forward_time = time.time()
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            forward_times.append(forward_time - start_time)
            backward_times.append(end_time - forward_time)
            total_times.append(end_time - start_time)
        
        return {
            'mean_forward_time': sum(forward_times) / len(forward_times),
            'mean_backward_time': sum(backward_times) / len(backward_times),
            'mean_total_time': sum(total_times) / len(total_times)
        }


def print_model_summary(model: nn.Module, input_shape: Optional[Tuple[int, ...]] = None):
    """打印模型摘要信息"""
    print("="*60)
    print("模型摘要")
    print("="*60)
    
    # 参数统计
    param_stats = count_parameters(model)
    print(f"总参数数: {param_stats['total_parameters']:,}")
    print(f"可训练参数数: {param_stats['trainable_parameters']:,}")
    print(f"冻结参数数: {param_stats['frozen_parameters']:,}")
    print(f"可训练参数比例: {param_stats['trainable_ratio']:.2%}")
    
    # 内存使用
    memory_stats = get_model_memory_usage(model)
    print(f"模型内存使用: {memory_stats['total_size_mb']:.2f} MB")
    
    # 设备信息
    device_info = get_device_info()
    print(f"CUDA可用: {device_info['cuda_available']}")
    if device_info['cuda_available']:
        print(f"GPU数量: {device_info['cuda_device_count']}")
        print(f"当前设备: {device_info['current_device']}")
    
    print("="*60)


def validate_model_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    """验证模型状态字典的有效性"""
    if not isinstance(state_dict, dict):
        print("状态字典必须是字典类型")
        return False
    
    if len(state_dict) == 0:
        print("状态字典为空")
        return False
    
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            print(f"键 {key} 对应的值不是张量")
            return False
        
        if tensor.numel() == 0:
            print(f"键 {key} 对应的张量为空")
            return False
    
    return True


def get_model_device(model: nn.Module) -> str:
    """获取模型所在的设备"""
    try:
        return next(model.parameters()).device.type
    except StopIteration:
        return "unknown"


def sync_batch_norm_to_device(model: nn.Module, device: str):
    """同步BatchNorm到指定设备"""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.to(device)


def get_module_by_name(model: nn.Module, module_name: str) -> Optional[nn.Module]:
    """根据名称获取模型中的模块"""
    try:
        names = module_name.split('.')
        module = model
        for name in names:
            module = getattr(module, name)
        return module
    except AttributeError:
        return None


def replace_module_by_name(model: nn.Module, module_name: str, new_module: nn.Module):
    """根据名称替换模型中的模块"""
    names = module_name.split('.')
    parent = model
    
    for name in names[:-1]:
        parent = getattr(parent, name)
    
    setattr(parent, names[-1], new_module)


def calculate_model_size_mb(model: nn.Module) -> float:
    """计算模型大小（MB）"""
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    
    # 假设每个参数是float32（4字节）
    model_size_bytes = total_params * 4
    model_size_mb = model_size_bytes / (1024 * 1024)
    
    return model_size_mb


def model_requires_grad_stats(model: nn.Module) -> Dict[str, int]:
    """统计模型中需要梯度的参数"""
    requires_grad_params = 0
    frozen_params = 0
    
    for param in model.parameters():
        if param.requires_grad:
            requires_grad_params += param.numel()
        else:
            frozen_params += param.numel()
    
    return {
        'requires_grad': requires_grad_params,
        'frozen': frozen_params,
        'total': requires_grad_params + frozen_params
    }