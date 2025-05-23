"""
路径管理模块
处理所有与路径相关的操作和验证
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime


class PathManager:
    """路径管理器"""
    
    def __init__(self, base_output_dir: str):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_model_output_dir(self, model_name: str) -> Path:
        """获取模型输出目录"""
        model_dir = self.base_output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    def get_dataset_output_dir(self, model_name: str, dataset_id: str) -> Path:
        """获取数据集输出目录"""
        dataset_dir = self.get_model_output_dir(model_name) / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir
    
    def get_summary_report_dir(self) -> Path:
        """获取摘要报告目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.base_output_dir / f"summary_report_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir
    
    def get_results_file_path(self, model_name: str, dataset_id: str) -> Path:
        """获取结果文件路径"""
        return self.get_dataset_output_dir(model_name, dataset_id) / "results.csv"
    
    def get_summary_file_path(self, model_name: str, dataset_id: str) -> Path:
        """获取摘要文件路径"""
        return self.get_dataset_output_dir(model_name, dataset_id) / "summary.json"
    
    def check_existing_results(self, model_name: str, dataset_id: str) -> bool:
        """检查是否已存在结果文件"""
        results_file = self.get_results_file_path(model_name, dataset_id)
        return results_file.exists() and results_file.stat().st_size > 0


class DatasetPathValidator:
    """数据集路径验证器"""
    
    @staticmethod
    def validate_dataset_structure(base_dir: Path) -> List[dict]:
        """验证数据集目录结构"""
        valid_datasets = []
        
        if not base_dir.exists():
            raise FileNotFoundError(f"基础目录不存在: {base_dir}")
        
        try:
            for cell_type_dir in base_dir.iterdir():
                if not cell_type_dir.is_dir():
                    continue
                
                for date_dir in cell_type_dir.iterdir():
                    if not date_dir.is_dir():
                        continue
                    
                    for magnification_dir in date_dir.iterdir():
                        if not magnification_dir.is_dir():
                            continue
                        
                        images_dir = magnification_dir / "images"
                        masks_dir = magnification_dir / "masks"
                        
                        if DatasetPathValidator._validate_image_mask_dirs(images_dir, masks_dir):
                            valid_datasets.append({
                                'cell_type': cell_type_dir.name,
                                'date': date_dir.name,
                                'magnification': magnification_dir.name,
                                'images_dir': str(images_dir),
                                'masks_dir': str(masks_dir),
                                'dataset_id': f"{cell_type_dir.name}_{date_dir.name}_{magnification_dir.name}"
                            })
        
        except Exception as e:
            print(f"验证数据集结构时出错: {e}")
        
        return valid_datasets
    
    @staticmethod
    def _validate_image_mask_dirs(images_dir: Path, masks_dir: Path) -> bool:
        """验证图像和掩码目录"""
        if not (images_dir.exists() and masks_dir.exists()):
            return False
        
        # 检查是否有有效的图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        mask_extensions = ['.png', '.tif', '.tiff']
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(images_dir.glob(f"*{ext}")))
        
        mask_files = []
        for ext in mask_extensions:
            mask_files.extend(list(masks_dir.glob(f"*{ext}")))
        
        return len(image_files) > 0 and len(mask_files) > 0
    
    @staticmethod
    def find_matching_mask(image_path: Path, masks_dir: Path) -> Optional[Path]:
        """为图像文件查找匹配的掩码文件"""
        image_stem = image_path.stem
        
        # 尝试不同的掩码命名模式
        patterns = [
            f"{image_stem}_seg.png",
            f"{image_stem}.png", 
            f"{image_stem}_mask.png",
            f"{image_stem}_seg.tif",
            f"{image_stem}.tif",
            f"{image_stem}_mask.tif"
        ]
        
        for pattern in patterns:
            mask_path = masks_dir / pattern
            if mask_path.exists():
                return mask_path
        
        return None
    
    @staticmethod
    def count_valid_pairs(images_dir: Path, masks_dir: Path) -> Tuple[int, int]:
        """统计有效的图像-掩码对数量"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(images_dir.glob(f"*{ext}")))
        
        valid_pairs = 0
        for image_file in image_files:
            if DatasetPathValidator.find_matching_mask(image_file, masks_dir):
                valid_pairs += 1
        
        return len(image_files), valid_pairs


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_model_cache_dir(self, model_name: str) -> Path:
        """获取模型缓存目录"""
        model_cache = self.cache_dir / model_name
        model_cache.mkdir(parents=True, exist_ok=True)
        return model_cache
    
    def clear_cache(self, model_name: str = None):
        """清理缓存"""
        if model_name:
            model_cache = self.get_model_cache_dir(model_name)
            if model_cache.exists():
                import shutil
                shutil.rmtree(model_cache)
        else:
            if self.cache_dir.exists():
                import shutil
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_size(self) -> int:
        """获取缓存大小（字节）"""
        total_size = 0
        for file_path in self.cache_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def cleanup_old_cache(self, days: int = 7):
        """清理旧缓存文件"""
        import time
        cutoff_time = time.time() - (days * 24 * 3600)
        
        for file_path in self.cache_dir.rglob('*'):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"删除缓存文件失败 {file_path}: {e}")


def setup_environment_paths(cache_dir: str):
    """设置环境路径"""
    os.environ["MICROSAM_CACHEDIR"] = cache_dir
    
    # 创建必要的目录
    Path(cache_dir).mkdir(parents=True, exist_ok=True)


def validate_output_permissions(output_dir: str) -> bool:
    """验证输出目录权限"""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 测试写入权限
        test_file = output_path / ".test_write_permission"
        test_file.write_text("test")
        test_file.unlink()
        
        return True
    except Exception as e:
        print(f"输出目录权限验证失败: {e}")
        return False