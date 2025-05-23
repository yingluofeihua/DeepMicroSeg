"""
文件处理工具模块
提供图像加载、保存和验证功能
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from skimage import io
from PIL import Image
import tifffile
import logging


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


class ImageLoadError(Exception):
    """图像加载错误"""
    pass


def load_image(image_path: Path, convert_to_grayscale: bool = True) -> Optional[np.ndarray]:
    """
    加载图像文件
    
    Args:
        image_path: 图像文件路径
        convert_to_grayscale: 是否转换为灰度图
    
    Returns:
        图像数组，失败时返回None
    """
    try:
        image_path = Path(image_path)
        
        if not image_path.exists():
            logging.error(f"图像文件不存在: {image_path}")
            return None
        
        # 尝试多种加载方法
        image = _robust_image_load(image_path)
        
        if image is None:
            return None
        
        # 处理通道
        if convert_to_grayscale and len(image.shape) > 2:
            image = _convert_to_grayscale(image)
        
        # 确保数据类型
        if image.dtype != np.uint8:
            image = _normalize_to_uint8(image)
        
        return image
        
    except Exception as e:
        logging.error(f"加载图像失败 {image_path}: {e}")
        return None


def load_mask(mask_path: Path) -> Optional[np.ndarray]:
    """
    加载掩码文件
    
    Args:
        mask_path: 掩码文件路径
    
    Returns:
        掩码数组，失败时返回None
    """
    try:
        mask_path = Path(mask_path)
        
        if not mask_path.exists():
            logging.error(f"掩码文件不存在: {mask_path}")
            return None
        
        # 加载掩码
        mask = _robust_image_load(mask_path)
        
        if mask is None:
            return None
        
        # 确保掩码是单通道
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]  # 取第一个通道
        
        # 确保是2D数组
        if len(mask.shape) != 2:
            logging.error(f"掩码维度错误: {mask.shape}")
            return None
        
        return mask.astype(np.int32)
        
    except Exception as e:
        logging.error(f"加载掩码失败 {mask_path}: {e}")
        return None


def _robust_image_load(image_path: Path, max_retries: int = 3) -> Optional[np.ndarray]:
    """
    鲁棒的图像加载，尝试多种方法
    """
    load_methods = [
        ("PIL", _load_with_pil),
        ("skimage", _load_with_skimage),
        ("tifffile", _load_with_tifffile)
    ]
    
    for method_name, load_func in load_methods:
        for attempt in range(max_retries):
            try:
                image = load_func(image_path)
                if image is not None and image.size > 0:
                    logging.debug(f"{method_name} 成功加载: {image_path}")
                    return image
            except Exception as e:
                logging.debug(f"{method_name} 尝试 {attempt + 1} 失败: {e}")
    
    logging.error(f"所有加载方法都失败了: {image_path}")
    return None


def _load_with_pil(image_path: Path) -> Optional[np.ndarray]:
    """使用PIL加载图像"""
    with Image.open(image_path) as img:
        return np.array(img)


def _load_with_skimage(image_path: Path) -> Optional[np.ndarray]:
    """使用skimage加载图像"""
    return io.imread(str(image_path))


def _load_with_tifffile(image_path: Path) -> Optional[np.ndarray]:
    """使用tifffile加载图像（仅适用于TIFF）"""
    if image_path.suffix.lower() in ['.tif', '.tiff']:
        return tifffile.imread(str(image_path))
    return None


def _convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """将多通道图像转换为灰度"""
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            # RGB转灰度
            return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]).astype(image.dtype)
        elif image.shape[2] == 4:
            # RGBA转灰度，忽略alpha通道
            return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]).astype(image.dtype)
        else:
            # 取第一个通道
            return image[:, :, 0]
    return image


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """标准化图像到uint8格式"""
    if image.dtype == np.uint8:
        return image
    
    if image.max() <= 1.0:
        # 0-1范围的浮点数
        return (image * 255).astype(np.uint8)
    elif image.dtype == np.uint16:
        # 16位图像
        return (image / 256).astype(np.uint8)
    else:
        # 其他情况，直接截断
        return np.clip(image, 0, 255).astype(np.uint8)


def save_image(image: np.ndarray, output_path: Path, 
               quality: int = 95) -> bool:
    """
    保存图像
    
    Args:
        image: 图像数组
        output_path: 输出路径
        quality: JPEG质量（1-100）
    
    Returns:
        是否保存成功
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为PIL图像
        if len(image.shape) == 2:
            pil_image = Image.fromarray(image, mode='L')
        elif len(image.shape) == 3 and image.shape[2] == 3:
            pil_image = Image.fromarray(image, mode='RGB')
        else:
            raise ValueError(f"不支持的图像形状: {image.shape}")
        
        # 根据文件扩展名选择保存格式
        if output_path.suffix.lower() in ['.jpg', '.jpeg']:
            pil_image.save(output_path, 'JPEG', quality=quality)
        elif output_path.suffix.lower() == '.png':
            pil_image.save(output_path, 'PNG')
        elif output_path.suffix.lower() in ['.tif', '.tiff']:
            pil_image.save(output_path, 'TIFF')
        else:
            # 默认PNG
            pil_image.save(output_path, 'PNG')
        
        return True
        
    except Exception as e:
        logging.error(f"保存图像失败 {output_path}: {e}")
        return False


def validate_image_pair(image_path: Path, mask_path: Path) -> Tuple[bool, str]:
    """
    验证图像-掩码对的有效性
    
    Returns:
        (是否有效, 错误信息)
    """
    try:
        # 检查文件存在性
        if not image_path.exists():
            return False, f"图像文件不存在: {image_path}"
        
        if not mask_path.exists():
            return False, f"掩码文件不存在: {mask_path}"
        
        # 尝试加载
        image = load_image(image_path, convert_to_grayscale=False)
        mask = load_mask(mask_path)
        
        if image is None:
            return False, f"无法加载图像: {image_path}"
        
        if mask is None:
            return False, f"无法加载掩码: {mask_path}"
        
        # 检查尺寸匹配
        if image.shape[:2] != mask.shape:
            return False, f"图像和掩码尺寸不匹配: {image.shape[:2]} vs {mask.shape}"
        
        return True, "有效"
        
    except Exception as e:
        return False, f"验证失败: {e}"


def get_image_info(image_path: Path) -> Optional[Dict]:
    """
    获取图像信息
    """
    try:
        image = _robust_image_load(image_path)
        if image is None:
            return None
        
        file_size = image_path.stat().st_size
        
        return {
            'path': str(image_path),
            'shape': image.shape,
            'dtype': str(image.dtype),
            'size_bytes': file_size,
            'size_mb': file_size / (1024 * 1024),
            'min_value': image.min(),
            'max_value': image.max(),
            'mean_value': image.mean()
        }
        
    except Exception as e:
        logging.error(f"获取图像信息失败 {image_path}: {e}")
        return None


def batch_validate_images(image_paths: List[Path], 
                         mask_paths: List[Path]) -> Dict:
    """
    批量验证图像
    
    Returns:
        验证结果统计
    """
    if len(image_paths) != len(mask_paths):
        raise ValueError("图像和掩码列表长度不匹配")
    
    results = {
        'total': len(image_paths),
        'valid': 0,
        'invalid': 0,
        'errors': []
    }
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        is_valid, error_msg = validate_image_pair(img_path, mask_path)
        
        if is_valid:
            results['valid'] += 1
        else:
            results['invalid'] += 1
            results['errors'].append({
                'image': str(img_path),
                'mask': str(mask_path),
                'error': error_msg
            })
    
    return results


def cleanup_temp_files(temp_dir: Path, max_age_hours: int = 24):
    """
    清理临时文件
    
    Args:
        temp_dir: 临时文件目录
        max_age_hours: 最大保留时间（小时）
    """
    try:
        import time
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        if not temp_dir.exists():
            return
        
        removed_count = 0
        for file_path in temp_dir.rglob('*'):
            if file_path.is_file():
                if file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        removed_count += 1
                    except Exception as e:
                        logging.warning(f"删除临时文件失败 {file_path}: {e}")
        
        logging.info(f"清理了 {removed_count} 个临时文件")
        
    except Exception as e:
        logging.error(f"清理临时文件失败: {e}")


# 便捷函数
def load_image_mask_pair(image_path: Path, mask_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """加载图像-掩码对"""
    image = load_image(image_path)
    mask = load_mask(mask_path)
    
    if image is not None and mask is not None:
        return image, mask
    return None