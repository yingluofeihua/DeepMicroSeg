"""
模型处理模块
负责模型加载、推理和资源管理
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import gc
import time
from contextlib import contextmanager

# Import micro_sam modules
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

from config.settings import ModelConfig


class ModelLoadError(Exception):
    """模型加载错误"""
    pass


class ModelInferenceError(Exception):
    """模型推理错误"""
    pass


class ModelHandler:
    """模型处理器 - 负责模型的加载和推理"""
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.predictor = None
        self.segmenter = None
        self.device = model_config.device
        self.is_loaded = False
        self.load_time = None
        
    def load_model(self) -> bool:
        """加载模型"""
        try:
            start_time = time.time()
            
            # 设置设备
            if self._setup_device():
                print(f"正在加载模型: {self.model_config.name}")
                
                # 清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 加载预测器和分割器
                self.predictor, self.segmenter = get_predictor_and_segmenter(
                    model_type=self.model_config.name,
                    device=self.device,
                    amg=False,
                    is_tiled=False
                )
                
                if self.predictor is None or self.segmenter is None:
                    raise ModelLoadError(f"模型加载失败: {self.model_config.name}")
                
                self.is_loaded = True
                self.load_time = time.time() - start_time
                
                print(f"模型加载成功: {self.model_config.name} "
                      f"(耗时: {self.load_time:.2f}s, 设备: {self.device})")
                
                return True
                
        except Exception as e:
            print(f"模型加载失败: {self.model_config.name}, 错误: {e}")
            self.cleanup()
            return False
    
    def _setup_device(self) -> bool:
        """设置计算设备"""
        try:
            if self.model_config.device == "cuda" and torch.cuda.is_available():
                # 尝试使用CUDA
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    self.device = "cuda"
                    print(f"使用GPU设备，可用设备数: {device_count}")
                    return True
                else:
                    print("CUDA不可用，回退到CPU")
                    self.device = "cpu"
                    return True
            else:
                # 使用CPU
                self.device = "cpu" 
                print("使用CPU设备")
                return True
                
        except Exception as e:
            print(f"设备设置失败: {e}")
            return False
    
    def predict(self, image: np.ndarray) -> Optional[np.ndarray]:
        """执行分割预测"""
        if not self.is_loaded:
            raise ModelInferenceError("模型未加载")
        
        try:
            # 预处理图像
            processed_image = self._preprocess_image(image)
            
            # 执行自动实例分割
            segmentation = automatic_instance_segmentation(
                predictor=self.predictor,
                segmenter=self.segmenter,
                input_path=processed_image,  # 直接传入numpy数组
                ndim=2
            )
            
            return segmentation
            
        except Exception as e:
            raise ModelInferenceError(f"模型推理失败: {e}")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """预处理输入图像"""
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")
        
        # 确保是numpy数组
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # 处理多通道图像 - 转换为灰度
        if len(image.shape) > 2:
            if image.shape[2] > 1:
                # 如果是多通道，取第一个通道或转换为灰度
                if image.shape[2] == 3:
                    # RGB转灰度
                    image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
                else:
                    # 取第一个通道
                    image = image[:, :, 0]
        
        # 确保数据类型正确
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        return image
    
    def batch_predict(self, images: list) -> list:
        """批量预测"""
        if not self.is_loaded:
            raise ModelInferenceError("模型未加载")
        
        results = []
        for i, image in enumerate(images):
            try:
                result = self.predict(image)
                results.append(result)
                
                # 定期清理内存
                if (i + 1) % 10 == 0:
                    self._cleanup_memory()
                    
            except Exception as e:
                print(f"批量预测第{i}张图像失败: {e}")
                results.append(None)
        
        return results
    
    def _cleanup_memory(self):
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            'name': self.model_config.name,
            'device': self.device,
            'is_loaded': self.is_loaded,
            'load_time': self.load_time,
            'gpu_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.predictor is not None:
                del self.predictor
                self.predictor = None
            
            if self.segmenter is not None:
                del self.segmenter
                self.segmenter = None
            
            self._cleanup_memory()
            self.is_loaded = False
            
            print(f"模型资源已清理: {self.model_config.name}")
            
        except Exception as e:
            print(f"清理资源时出错: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        if not self.load_model():
            raise ModelLoadError(f"无法加载模型: {self.model_config.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()


class ModelManager:
    """模型管理器 - 管理多个模型的生命周期"""
    
    def __init__(self):
        self.handlers = {}
        self.active_handler = None
    
    def register_model(self, model_config: ModelConfig) -> ModelHandler:
        """注册模型"""
        handler = ModelHandler(model_config)
        self.handlers[model_config.name] = handler
        return handler
    
    def get_model_handler(self, model_name: str) -> Optional[ModelHandler]:
        """获取模型处理器"""
        return self.handlers.get(model_name)
    
    def load_model(self, model_name: str) -> bool:
        """加载指定模型"""
        handler = self.handlers.get(model_name)
        if handler is None:
            print(f"模型未注册: {model_name}")
            return False
        
        # 卸载当前活跃模型
        if self.active_handler and self.active_handler != handler:
            self.active_handler.cleanup()
        
        # 加载新模型
        if handler.load_model():
            self.active_handler = handler
            return True
        
        return False
    
    def switch_model(self, model_name: str) -> bool:
        """切换到指定模型"""
        return self.load_model(model_name)
    
    def predict_with_model(self, model_name: str, image: np.ndarray) -> Optional[np.ndarray]:
        """使用指定模型进行预测"""
        handler = self.handlers.get(model_name)
        if handler is None:
            raise ValueError(f"模型未注册: {model_name}")
        
        if not handler.is_loaded:
            if not handler.load_model():
                return None
        
        return handler.predict(image)
    
    def cleanup_all(self):
        """清理所有模型"""
        for handler in self.handlers.values():
            handler.cleanup()
        self.active_handler = None
    
    def get_all_model_info(self) -> dict:
        """获取所有模型信息"""
        info = {}
        for name, handler in self.handlers.items():
            info[name] = handler.get_model_info()
        return info


@contextmanager
def model_context(model_config: ModelConfig):
    """模型上下文管理器"""
    handler = ModelHandler(model_config)
    try:
        if not handler.load_model():
            raise ModelLoadError(f"无法加载模型: {model_config.name}")
        yield handler
    finally:
        handler.cleanup()


def create_model_handler(model_name: str, device: str = "cuda") -> ModelHandler:
    """创建模型处理器的便捷函数"""
    model_config = ModelConfig(name=model_name, device=device)
    return ModelHandler(model_config)


def safe_model_prediction(model_handler: ModelHandler, 
                         image: np.ndarray, 
                         max_retries: int = 3) -> Optional[np.ndarray]:
    """安全的模型预测（带重试机制）"""
    for attempt in range(max_retries):
        try:
            result = model_handler.predict(image)
            return result
        except Exception as e:
            print(f"预测尝试 {attempt + 1} 失败: {e}")
            if attempt < max_retries - 1:
                # 清理内存后重试
                model_handler._cleanup_memory()
                time.sleep(1)
            else:
                print(f"预测最终失败，已重试 {max_retries} 次")
                return None