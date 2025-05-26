"""
数据集划分工具
支持数据集的训练/验证/测试划分，并保存划分结果到JSON文件
"""

import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import datetime

from config.paths import DatasetPathValidator


@dataclass
class DataSplit:
    """数据划分结果"""
    train_samples: List[Dict]
    val_samples: List[Dict]
    test_samples: List[Dict]
    split_info: Dict
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'train_samples': self.train_samples,
            'val_samples': self.val_samples,
            'test_samples': self.test_samples,
            'split_info': self.split_info
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """从字典创建对象"""
        return cls(
            train_samples=data['train_samples'],
            val_samples=data['val_samples'], 
            test_samples=data['test_samples'],
            split_info=data['split_info']
        )


class DatasetSplitter:
    """数据集划分器"""
    
    def __init__(self, data_dir: str, split_storage_dir: str = "./data/lora_split"):
        self.data_dir = Path(data_dir)
        self.split_storage_dir = Path(split_storage_dir)
        self.split_storage_dir.mkdir(parents=True, exist_ok=True)
    
    def get_split_file_path(self, 
                           train_ratio: float, 
                           val_ratio: float, 
                           test_ratio: float,
                           cell_types: Optional[List[str]] = None,
                           split_method: str = "random",
                           seed: int = 42) -> Path:
        """获取划分文件的路径"""
        
        # 创建唯一标识符
        identifier_parts = [
            str(self.data_dir.resolve()),
            f"train_{train_ratio:.3f}",
            f"val_{val_ratio:.3f}",
            f"test_{test_ratio:.3f}",
            f"method_{split_method}",
            f"seed_{seed}"
        ]
        
        if cell_types:
            identifier_parts.append(f"cells_{'-'.join(sorted(cell_types))}")
        
        # 使用MD5生成短文件名
        identifier_str = "|".join(identifier_parts)
        file_hash = hashlib.md5(identifier_str.encode()).hexdigest()[:16]
        
        # 创建可读的文件名
        cell_suffix = f"_{'_'.join(cell_types)}" if cell_types else ""
        # filename = f"split_{train_ratio:.1f}_{val_ratio:.1f}_{test_ratio:.1f}{cell_suffix}_{file_hash}.json"
        filename = f"split_{train_ratio:.2f}_{val_ratio:.2f}_{test_ratio:.2f}{cell_suffix}_{file_hash}.json"
        
        return self.split_storage_dir / filename
    
    def load_cached_split(self, split_file_path: Path) -> Optional[DataSplit]:
        """加载缓存的划分结果"""
        try:
            if not split_file_path.exists():
                return None
            
            with open(split_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证文件完整性
            required_keys = ['train_samples', 'val_samples', 'test_samples', 'split_info']
            if not all(key in data for key in required_keys):
                print(f"划分文件格式不完整: {split_file_path}")
                return None
            
            # 验证样本路径是否仍然有效
            split_result = DataSplit.from_dict(data)
            if self._validate_split_paths(split_result):
                print(f"加载缓存的数据划分: {split_file_path}")
                return split_result
            else:
                print(f"缓存的划分文件中的路径已失效: {split_file_path}")
                return None
                
        except Exception as e:
            print(f"加载划分文件失败 {split_file_path}: {e}")
            return None
    
    def _validate_split_paths(self, split_result: DataSplit) -> bool:
        """验证划分结果中的路径是否有效"""
        all_samples = (split_result.train_samples + 
                      split_result.val_samples + 
                      split_result.test_samples)
        
        # 检查前10个样本的路径
        check_count = min(10, len(all_samples))
        valid_count = 0
        
        for i, sample in enumerate(all_samples[:check_count]):
            image_path = Path(sample['image_path'])
            mask_path = Path(sample['mask_path'])
            
            if image_path.exists() and mask_path.exists():
                valid_count += 1
        
        # 至少80%的样本路径有效才认为划分有效
        validity_ratio = valid_count / check_count if check_count > 0 else 0
        return validity_ratio >= 0.8
    
    def create_new_split(self,
                        train_ratio: float = 0.8,
                        val_ratio: float = 0.1, 
                        test_ratio: float = 0.1,
                        cell_types: Optional[List[str]] = None,
                        split_method: str = "random",
                        seed: int = 42) -> DataSplit:
        """创建新的数据划分"""
        
        print(f"创建新的数据划分...")
        print(f"  数据目录: {self.data_dir}")
        print(f"  划分比例: train={train_ratio:.1f}, val={val_ratio:.1f}, test={test_ratio:.1f}")
        print(f"  细胞类型过滤: {cell_types}")
        print(f"  划分方法: {split_method}")
        print(f"  随机种子: {seed}")
        
        # 发现所有数据集
        valid_datasets = DatasetPathValidator.validate_dataset_structure(self.data_dir)
        print(f"发现 {len(valid_datasets)} 个有效数据集")
        
        # 过滤细胞类型
        if cell_types:
            filtered_datasets = [d for d in valid_datasets if d['cell_type'] in cell_types]
            print(f"细胞类型过滤后: {len(filtered_datasets)} 个数据集")
            valid_datasets = filtered_datasets
        
        # 收集所有样本
        all_samples = []
        for dataset_info in valid_datasets:
            samples = self._collect_samples_from_dataset(dataset_info)
            all_samples.extend(samples)
        
        print(f"总样本数: {len(all_samples)}")
        
        if len(all_samples) == 0:
            raise ValueError("没有找到有效的样本")
        
        # 执行划分
        if split_method == "random":
            split_result = self._random_split(all_samples, train_ratio, val_ratio, test_ratio, seed)
        elif split_method == "by_dataset":
            split_result = self._dataset_aware_split(all_samples, train_ratio, val_ratio, test_ratio, seed)
        else:
            raise ValueError(f"不支持的划分方法: {split_method}")
        
        # 添加划分信息
        split_result.split_info = {
            'data_dir': str(self.data_dir),
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'cell_types': cell_types,
            'split_method': split_method,
            'seed': seed,
            'total_samples': len(all_samples),
            'train_count': len(split_result.train_samples),
            'val_count': len(split_result.val_samples),
            'test_count': len(split_result.test_samples),
            'created_at': datetime.datetime.now().isoformat()
        }
        
        print(f"划分完成: train={len(split_result.train_samples)}, val={len(split_result.val_samples)}, test={len(split_result.test_samples)}")
        
        return split_result
    
    def _collect_samples_from_dataset(self, dataset_info: Dict) -> List[Dict]:
        """从单个数据集收集样本"""
        samples = []
        
        images_dir = Path(dataset_info['images_dir'])
        masks_dir = Path(dataset_info['masks_dir'])
        
        # 获取图像-掩码对
        image_mask_pairs = self._get_image_mask_pairs(images_dir, masks_dir)
        
        for img_path, mask_path in image_mask_pairs:
            sample = {
                'image_path': str(img_path),
                'mask_path': str(mask_path),
                'sample_id': img_path.stem,
                'cell_type': dataset_info['cell_type'],
                'date': dataset_info['date'],
                'magnification': dataset_info['magnification'],
                'dataset_id': dataset_info['dataset_id']
            }
            samples.append(sample)
        
        return samples
    
    def _get_image_mask_pairs(self, images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
        """获取图像-掩码对"""
        pairs = []
        
        # 获取所有图像文件
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            image_files.extend(list(images_dir.glob(f"*{ext}")))
            image_files.extend(list(images_dir.glob(f"*{ext.upper()}")))
        
        # 为每个图像文件查找对应的掩码
        for img_file in image_files:
            mask_file = DatasetPathValidator.find_matching_mask(img_file, masks_dir)
            if mask_file:
                pairs.append((img_file, mask_file))
        
        return pairs
    
    def _random_split(self, samples: List[Dict], 
                     train_ratio: float, val_ratio: float, test_ratio: float,
                     seed: int) -> DataSplit:
        """随机划分数据"""
        
        # 设置随机种子
        random.seed(seed)
        shuffled_samples = samples.copy()
        random.shuffle(shuffled_samples)
        
        total_count = len(shuffled_samples)
        train_count = int(total_count * train_ratio)
        val_count = int(total_count * val_ratio)
        
        train_samples = shuffled_samples[:train_count]
        val_samples = shuffled_samples[train_count:train_count + val_count]
        test_samples = shuffled_samples[train_count + val_count:]
        
        return DataSplit(train_samples, val_samples, test_samples, {})
    
    def _dataset_aware_split(self, samples: List[Dict], 
                           train_ratio: float, val_ratio: float, test_ratio: float,
                           seed: int) -> DataSplit:
        """按数据集感知的方式划分数据"""
        
        # 按数据集分组
        dataset_groups = {}
        for sample in samples:
            dataset_id = sample['dataset_id']
            if dataset_id not in dataset_groups:
                dataset_groups[dataset_id] = []
            dataset_groups[dataset_id].append(sample)
        
        print(f"按数据集分组: {len(dataset_groups)} 个数据集")
        
        # 对每个数据集内部进行划分
        train_samples = []
        val_samples = []
        test_samples = []
        
        random.seed(seed)
        
        for dataset_id, dataset_samples in dataset_groups.items():
            random.shuffle(dataset_samples)
            
            count = len(dataset_samples)
            train_count = int(count * train_ratio)
            val_count = int(count * val_ratio)
            
            train_samples.extend(dataset_samples[:train_count])
            val_samples.extend(dataset_samples[train_count:train_count + val_count])
            test_samples.extend(dataset_samples[train_count + val_count:])
        
        return DataSplit(train_samples, val_samples, test_samples, {})
    
    def save_split(self, split_result: DataSplit, split_file_path: Path) -> bool:
        """保存划分结果到文件"""
        try:
            # 确保目录存在
            split_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存到JSON文件
            with open(split_file_path, 'w', encoding='utf-8') as f:
                json.dump(split_result.to_dict(), f, indent=2, ensure_ascii=False)
            
            print(f"数据划分已保存到: {split_file_path}")
            return True
            
        except Exception as e:
            print(f"保存划分文件失败: {e}")
            return False
    
    def split_data(self,
                   train_ratio: float = 0.8,
                   val_ratio: float = 0.1,
                   test_ratio: float = 0.1,
                   cell_types: Optional[List[str]] = None,
                   split_method: str = "random",
                   seed: int = 42,
                   use_cached: bool = True) -> DataSplit:
        """主要的数据划分接口"""
        
        # 获取划分文件路径
        split_file_path = self.get_split_file_path(
            train_ratio, val_ratio, test_ratio, cell_types, split_method, seed
        )
        
        # 尝试加载缓存的划分
        if use_cached:
            cached_split = self.load_cached_split(split_file_path)
            if cached_split is not None:
                return cached_split
        
        # 创建新的划分
        split_result = self.create_new_split(
            train_ratio, val_ratio, test_ratio, cell_types, split_method, seed
        )
        
        # 保存划分结果
        if self.save_split(split_result, split_file_path):
            print(f"新的数据划分已创建并保存")
        else:
            print(f"警告: 数据划分创建成功但保存失败")
        
        return split_result
    
    def list_cached_splits(self) -> List[Dict]:
        """列出所有缓存的划分"""
        cached_splits = []
        
        for split_file in self.split_storage_dir.glob("split_*.json"):
            try:
                with open(split_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'split_info' in data:
                    info = data['split_info'].copy()
                    info['file_path'] = str(split_file)
                    info['file_size_mb'] = split_file.stat().st_size / (1024 * 1024)
                    cached_splits.append(info)
                    
            except Exception as e:
                print(f"读取划分文件失败 {split_file}: {e}")
        
        return cached_splits
    
    def clean_old_splits(self, keep_recent: int = 10):
        """清理旧的划分文件，保留最近的几个"""
        try:
            split_files = list(self.split_storage_dir.glob("split_*.json"))
            
            if len(split_files) <= keep_recent:
                return
            
            # 按修改时间排序
            split_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # 删除多余的文件
            files_to_delete = split_files[keep_recent:]
            for file_path in files_to_delete:
                try:
                    file_path.unlink()
                    print(f"删除旧的划分文件: {file_path.name}")
                except Exception as e:
                    print(f"删除文件失败 {file_path}: {e}")
                    
        except Exception as e:
            print(f"清理旧划分文件失败: {e}")


def create_data_split(data_dir: str,
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1, 
                     test_ratio: float = 0.1,
                     cell_types: Optional[List[str]] = None,
                     split_method: str = "random",
                     seed: int = 42,
                     split_storage_dir: str = "./data/lora_split",
                     use_cached: bool = True) -> DataSplit:
    """便捷的数据划分函数"""
    
    splitter = DatasetSplitter(data_dir, split_storage_dir)
    return splitter.split_data(
        train_ratio, val_ratio, test_ratio, cell_types, 
        split_method, seed, use_cached
    )


def print_split_summary(split_result: DataSplit):
    """打印划分摘要"""
    info = split_result.split_info
    
    print("\n" + "="*60)
    print("数据集划分摘要")
    print("="*60)
    print(f"数据目录: {info.get('data_dir', 'N/A')}")
    print(f"划分方法: {info.get('split_method', 'N/A')}")
    print(f"随机种子: {info.get('seed', 'N/A')}")
    
    if info.get('cell_types'):
        print(f"细胞类型: {', '.join(info['cell_types'])}")
    
    print(f"\n划分结果:")
    print(f"  训练集: {info.get('train_count', len(split_result.train_samples))} 样本 ({info.get('train_ratio', 0):.1%})")
    print(f"  验证集: {info.get('val_count', len(split_result.val_samples))} 样本 ({info.get('val_ratio', 0):.1%})")
    print(f"  测试集: {info.get('test_count', len(split_result.test_samples))} 样本 ({info.get('test_ratio', 0):.1%})")
    print(f"  总计: {info.get('total_samples', 0)} 样本")
    
    # 按细胞类型统计
    if split_result.train_samples:
        print(f"\n按细胞类型分布:")
        
        for split_name, samples in [("训练集", split_result.train_samples), 
                                   ("验证集", split_result.val_samples),
                                   ("测试集", split_result.test_samples)]:
            if samples:
                cell_type_counts = {}
                for sample in samples:
                    cell_type = sample.get('cell_type', 'unknown')
                    cell_type_counts[cell_type] = cell_type_counts.get(cell_type, 0) + 1
                
                print(f"  {split_name}: {dict(cell_type_counts)}")
    
    print("="*60)