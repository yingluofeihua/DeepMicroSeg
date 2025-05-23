"""
数据集管理模块
自动发现、验证和组织数据集
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from config.paths import DatasetPathValidator


@dataclass
class DatasetInfo:
    """数据集信息"""
    cell_type: str
    date: str
    magnification: str
    images_dir: str
    masks_dir: str
    num_images: int
    num_masks: int
    dataset_id: str
    valid_pairs: int = 0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'cell_type': self.cell_type,
            'date': self.date,
            'magnification': self.magnification,
            'images_dir': self.images_dir,
            'masks_dir': self.masks_dir,
            'num_images': self.num_images,
            'num_masks': self.num_masks,
            'dataset_id': self.dataset_id,
            'valid_pairs': self.valid_pairs
        }


class DatasetManager:
    """数据集管理器 - 自动发现和组织所有数据集"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.datasets: List[DatasetInfo] = []
        self.discovery_stats = {}
        
        # 自动发现数据集
        self._discover_datasets()
    
    def _discover_datasets(self) -> None:
        """自动发现所有数据集"""
        print(f"正在扫描数据集目录: {self.base_dir}")
        
        if not self.base_dir.exists():
            raise FileNotFoundError(f"基础目录不存在: {self.base_dir}")
        
        try:
            raw_datasets = DatasetPathValidator.validate_dataset_structure(self.base_dir)
            
            for dataset_dict in raw_datasets:
                dataset_info = self._create_dataset_info(dataset_dict)
                if dataset_info:
                    self.datasets.append(dataset_info)
            
            # 按类型、日期、放大倍数排序
            self.datasets.sort(key=lambda x: (x.cell_type, x.date, x.magnification))
            
            # 更新发现统计
            self._update_discovery_stats()
            
            print(f"发现 {len(self.datasets)} 个有效数据集")
            
        except Exception as e:
            print(f"数据集发现错误: {e}")
            self.datasets = []
    
    def _create_dataset_info(self, dataset_dict: Dict) -> Optional[DatasetInfo]:
        """从字典创建数据集信息对象"""
        try:
            images_dir = Path(dataset_dict['images_dir'])
            masks_dir = Path(dataset_dict['masks_dir'])
            
            # 统计文件数量
            image_files = self._get_image_files(images_dir)
            mask_files = self._get_mask_files(masks_dir)
            
            # 统计有效配对
            total_images, valid_pairs = DatasetPathValidator.count_valid_pairs(
                images_dir, masks_dir
            )
            
            return DatasetInfo(
                cell_type=dataset_dict['cell_type'],
                date=dataset_dict['date'],
                magnification=dataset_dict['magnification'],
                images_dir=str(images_dir),
                masks_dir=str(masks_dir),
                num_images=len(image_files),
                num_masks=len(mask_files),
                dataset_id=dataset_dict['dataset_id'],
                valid_pairs=valid_pairs
            )
            
        except Exception as e:
            print(f"创建数据集信息失败: {e}")
            return None
    
    def _get_image_files(self, images_dir: Path) -> List[Path]:
        """获取图像文件列表"""
        extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        image_files = []
        
        for ext in extensions:
            image_files.extend(list(images_dir.glob(f"*{ext}")))
            image_files.extend(list(images_dir.glob(f"*{ext.upper()}")))
        
        return sorted(image_files)
    
    def _get_mask_files(self, masks_dir: Path) -> List[Path]:
        """获取掩码文件列表"""
        extensions = ['.png', '.tif', '.tiff']
        mask_files = []
        
        for ext in extensions:
            mask_files.extend(list(masks_dir.glob(f"*{ext}")))
            mask_files.extend(list(masks_dir.glob(f"*{ext.upper()}")))
        
        return sorted(mask_files)
    
    def _update_discovery_stats(self) -> None:
        """更新发现统计信息"""
        self.discovery_stats = {
            'total_datasets': len(self.datasets),
            'cell_types': list(set(d.cell_type for d in self.datasets)),
            'dates': list(set(d.date for d in self.datasets)),
            'magnifications': list(set(d.magnification for d in self.datasets)),
            'total_images': sum(d.num_images for d in self.datasets),
            'total_masks': sum(d.num_masks for d in self.datasets),
            'total_valid_pairs': sum(d.valid_pairs for d in self.datasets)
        }
    
    def get_datasets_summary(self) -> pd.DataFrame:
        """获取数据集摘要DataFrame"""
        if not self.datasets:
            return pd.DataFrame()
        
        data = [dataset.to_dict() for dataset in self.datasets]
        return pd.DataFrame(data)
    
    def filter_datasets(self, 
                       cell_types: Optional[List[str]] = None,
                       dates: Optional[List[str]] = None,
                       magnifications: Optional[List[str]] = None,
                       min_valid_pairs: int = 0) -> List[DatasetInfo]:
        """过滤数据集"""
        filtered = self.datasets
        
        if cell_types:
            filtered = [d for d in filtered if d.cell_type in cell_types]
        
        if dates:
            filtered = [d for d in filtered if d.date in dates]
        
        if magnifications:
            filtered = [d for d in filtered if d.magnification in magnifications]
        
        if min_valid_pairs > 0:
            filtered = [d for d in filtered if d.valid_pairs >= min_valid_pairs]
        
        return filtered
    
    def get_dataset_by_id(self, dataset_id: str) -> Optional[DatasetInfo]:
        """根据ID获取数据集"""
        for dataset in self.datasets:
            if dataset.dataset_id == dataset_id:
                return dataset
        return None
    
    def get_image_mask_pairs(self, dataset_info: DatasetInfo) -> List[Tuple[Path, Path]]:
        """获取数据集中的图像-掩码对"""
        images_dir = Path(dataset_info.images_dir)
        masks_dir = Path(dataset_info.masks_dir)
        
        image_files = self._get_image_files(images_dir)
        pairs = []
        
        for image_file in image_files:
            mask_file = DatasetPathValidator.find_matching_mask(image_file, masks_dir)
            if mask_file:
                pairs.append((image_file, mask_file))
        
        return pairs
    
    def validate_dataset(self, dataset_info: DatasetInfo) -> Dict[str, any]:
        """验证单个数据集的完整性"""
        validation_result = {
            'dataset_id': dataset_info.dataset_id,
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # 检查目录存在性
            images_dir = Path(dataset_info.images_dir)
            masks_dir = Path(dataset_info.masks_dir)
            
            if not images_dir.exists():
                validation_result['errors'].append(f"图像目录不存在: {images_dir}")
                validation_result['valid'] = False
            
            if not masks_dir.exists():
                validation_result['errors'].append(f"掩码目录不存在: {masks_dir}")
                validation_result['valid'] = False
            
            if not validation_result['valid']:
                return validation_result
            
            # 获取图像-掩码对
            pairs = self.get_image_mask_pairs(dataset_info)
            
            # 统计信息
            validation_result['statistics'] = {
                'total_images': dataset_info.num_images,
                'total_masks': dataset_info.num_masks,
                'valid_pairs': len(pairs),
                'unpaired_images': dataset_info.num_images - len(pairs)
            }
            
            # 检查配对率
            if len(pairs) == 0:
                validation_result['errors'].append("没有找到有效的图像-掩码对")
                validation_result['valid'] = False
            elif len(pairs) < dataset_info.num_images * 0.8:
                validation_result['warnings'].append(
                    f"配对率较低: {len(pairs)}/{dataset_info.num_images} "
                    f"({len(pairs)/dataset_info.num_images*100:.1f}%)"
                )
            
        except Exception as e:
            validation_result['errors'].append(f"验证过程出错: {e}")
            validation_result['valid'] = False
        
        return validation_result
    
    def print_summary(self) -> None:
        """打印数据集摘要"""
        stats = self.discovery_stats
        
        print("\n" + "="*60)
        print("数据集发现摘要")
        print("="*60)
        print(f"总数据集: {stats['total_datasets']}")
        print(f"细胞类型: {', '.join(stats['cell_types'])}")
        print(f"日期: {', '.join(stats['dates'])}")
        print(f"放大倍数: {', '.join(stats['magnifications'])}")
        print(f"总图像数: {stats['total_images']}")
        print(f"总掩码数: {stats['total_masks']}")
        print(f"有效配对: {stats['total_valid_pairs']}")
        
        # 按细胞类型统计
        print(f"\n按细胞类型分布:")
        cell_type_stats = {}
        for dataset in self.datasets:
            if dataset.cell_type not in cell_type_stats:
                cell_type_stats[dataset.cell_type] = {
                    'datasets': 0, 'images': 0, 'pairs': 0
                }
            cell_type_stats[dataset.cell_type]['datasets'] += 1
            cell_type_stats[dataset.cell_type]['images'] += dataset.num_images
            cell_type_stats[dataset.cell_type]['pairs'] += dataset.valid_pairs
        
        for cell_type, stats in cell_type_stats.items():
            print(f"  {cell_type}: {stats['datasets']}个数据集, "
                  f"{stats['images']}张图像, {stats['pairs']}个有效对")
        
        print("="*60)
    
    def export_summary(self, output_file: str) -> None:
        """导出数据集摘要到文件"""
        summary_df = self.get_datasets_summary()
        
        if output_file.endswith('.csv'):
            summary_df.to_csv(output_file, index=False)
        elif output_file.endswith('.json'):
            import json
            data = {
                'discovery_stats': self.discovery_stats,
                'datasets': [d.to_dict() for d in self.datasets]
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError("不支持的文件格式，请使用 .csv 或 .json")
        
        print(f"数据集摘要已导出到: {output_file}")
    
    def get_dataset_statistics(self) -> Dict:
        """获取详细的数据集统计信息"""
        if not self.datasets:
            return {}
        
        # 计算各种统计信息
        valid_pairs = [d.valid_pairs for d in self.datasets]
        num_images = [d.num_images for d in self.datasets]
        
        return {
            'basic_stats': self.discovery_stats,
            'pair_statistics': {
                'min_pairs': min(valid_pairs) if valid_pairs else 0,
                'max_pairs': max(valid_pairs) if valid_pairs else 0,
                'avg_pairs': sum(valid_pairs) / len(valid_pairs) if valid_pairs else 0,
                'total_pairs': sum(valid_pairs)
            },
            'image_statistics': {
                'min_images': min(num_images) if num_images else 0,
                'max_images': max(num_images) if num_images else 0,
                'avg_images': sum(num_images) / len(num_images) if num_images else 0,
                'total_images': sum(num_images)
            },
            'distribution': {
                'by_cell_type': self._get_distribution_by_field('cell_type'),
                'by_date': self._get_distribution_by_field('date'),
                'by_magnification': self._get_distribution_by_field('magnification')
            }
        }
    
    def _get_distribution_by_field(self, field: str) -> Dict:
        """按指定字段获取分布统计"""
        distribution = {}
        for dataset in self.datasets:
            value = getattr(dataset, field)
            if value not in distribution:
                distribution[value] = {'count': 0, 'images': 0, 'pairs': 0}
            distribution[value]['count'] += 1
            distribution[value]['images'] += dataset.num_images
            distribution[value]['pairs'] += dataset.valid_pairs
        return distribution