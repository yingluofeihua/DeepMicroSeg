"""
修复后的SAM数据集 - 补充缺失的方法
在 lora/data_loaders.py 中替换或修改SAMDataset类
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import json
import datetime

from config.lora_config import DataConfig
from utils.file_utils import load_image, load_mask
from config.paths import DatasetPathValidator
from utils.data_splitter import DatasetSplitter, DataSplit, create_data_split, print_split_summary


class CellSegmentationDataset(Dataset):
    """细胞分割数据集 - 支持层次化数据结构和数据划分"""
    
    def __init__(
        self,
        data_dir: str = None,
        config: DataConfig = None,
        split: str = "train",
        transform: Optional[Any] = None,
        # 🔧 新增参数：支持直接传入样本列表
        samples: Optional[List[Dict]] = None
    ):
        self.data_dir = Path(data_dir) if data_dir else None
        self.config = config
        self.split = split
        self.transform = transform
        
        # 🔧 修改：支持从预划分的样本加载数据
        if samples is not None:
            # 直接使用传入的样本列表
            self.samples = samples
            print(f"使用预划分的{split}样本: {len(self.samples)} 个")
        else:
            # 传统方式：从目录结构加载数据
            self.samples = self._load_samples()
        
        # 创建增强变换
        if transform is None:
            self.transform = self._create_default_transforms()
        
        print(f"数据集 {split} 加载完成: {len(self.samples)} 个样本")
    
    def _load_samples(self) -> List[Dict]:
        """加载数据样本 - 支持细胞类型过滤"""
        if self.data_dir is None:
            return []
            
        samples = []
        
        try:
            valid_datasets = DatasetPathValidator.validate_dataset_structure(self.data_dir)
            print(f"发现 {len(valid_datasets)} 个有效数据集")
            
            for dataset_info in valid_datasets:
                # 🔧 添加细胞类型过滤
                if hasattr(self.config, '_cell_types_filter') and self.config._cell_types_filter:
                    if dataset_info['cell_type'] not in self.config._cell_types_filter:
                        continue  # 跳过不匹配的细胞类型
                
                images_dir = Path(dataset_info['images_dir'])
                masks_dir = Path(dataset_info['masks_dir'])
                
                image_mask_pairs = self._get_image_mask_pairs(images_dir, masks_dir)
                
                for img_path, mask_path in image_mask_pairs:
                    samples.append({
                        'image_path': str(img_path),
                        'mask_path': str(mask_path),
                        'sample_id': img_path.stem,
                        'cell_type': dataset_info['cell_type'],
                        'date': dataset_info['date'],
                        'magnification': dataset_info['magnification'],
                        'dataset_id': dataset_info['dataset_id']
                    })
            
            # 打印过滤后的统计
            if hasattr(self.config, '_cell_types_filter') and self.config._cell_types_filter:
                print(f"过滤后样本数 ({self.config._cell_types_filter}): {len(samples)}")
            
            samples = self._split_samples(samples)
            
            valid_samples = []
            for sample in samples:
                if self._validate_sample(sample):
                    valid_samples.append(sample)
            
            return valid_samples
            
        except Exception as e:
            print(f"加载数据样本失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
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
    
    def _split_samples(self, samples: List[Dict]) -> List[Dict]:
        """根据split参数分割数据 - 保留原有逻辑作为后备"""
        if not samples:
            return []
        
        # 按数据集分组
        dataset_groups = {}
        for sample in samples:
            dataset_id = sample['dataset_id']
            if dataset_id not in dataset_groups:
                dataset_groups[dataset_id] = []
            dataset_groups[dataset_id].append(sample)
        
        # 为每个数据集进行分割
        split_samples = []
        for dataset_id, dataset_samples in dataset_groups.items():
            n_total = len(dataset_samples)
            n_train = int(n_total * self.config.train_split_ratio)
            n_val = int(n_total * self.config.val_split_ratio)
            
            # 随机打乱（使用固定种子确保可重现）
            random.seed(self.config.split_seed)
            random.shuffle(dataset_samples)
            
            if self.split == "train":
                split_samples.extend(dataset_samples[:n_train])
            elif self.split == "val":
                split_samples.extend(dataset_samples[n_train:n_train + n_val])
            elif self.split == "test":
                split_samples.extend(dataset_samples[n_train + n_val:])
            else:  # "all"
                split_samples.extend(dataset_samples)
        
        return split_samples
    
    def _validate_sample(self, sample: Dict) -> bool:
        """验证样本有效性"""
        img_path = Path(sample['image_path'])
        mask_path = Path(sample['mask_path'])
        
        if not (img_path.exists() and mask_path.exists()):
            return False
        
        try:
            # 尝试加载图像和掩码
            image = load_image(img_path, convert_to_grayscale=False)
            mask = load_mask(mask_path)
            
            if image is None or mask is None:
                return False
            
            # 检查尺寸匹配
            if image.shape[:2] != mask.shape:
                return False
            
            # 检查掩码是否有前景对象
            if self.config.filter_empty_images and np.max(mask) == 0:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _create_default_transforms(self):
        """创建默认的数据变换"""
        if self.split == "train" and self.config.use_data_augmentation:
            # 训练时使用数据增强
            transform = A.Compose([
                A.Resize(self.config.image_size[0], self.config.image_size[1]),
                A.OneOf([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                ], p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, 
                        contrast_limit=0.2, 
                        p=0.5
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=20,
                        val_shift_limit=20,
                        p=0.5
                    ),
                ], p=0.3),
                A.OneOf([
                    A.GaussianBlur(blur_limit=3, p=0.3),
                    A.GaussNoise(p=0.3),  # 移除var_limit参数
                ], p=0.2),
                A.Normalize(mean=self.config.normalize_mean, std=self.config.normalize_std),
                ToTensorV2()
            ])
        else:
            # 验证/测试时只做基本变换
            transform = A.Compose([
                A.Resize(self.config.image_size[0], self.config.image_size[1]),
                A.Normalize(mean=self.config.normalize_mean, std=self.config.normalize_std),
                ToTensorV2()
            ])
        
        return transform
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # 加载图像和掩码
        image = load_image(sample['image_path'], convert_to_grayscale=False)
        mask = load_mask(sample['mask_path'])
        
        if image is None or mask is None:
            # 返回一个有效的空样本
            return self._get_empty_sample()
        
        # 确保图像是RGB格式
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        
        # 应用变换
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # 处理掩码格式
        if isinstance(mask, torch.Tensor):
            mask = mask.long()
        else:
            mask = torch.from_numpy(mask).long()
        
        # 生成实例掩码和边界框
        instance_masks, boxes = self._process_mask(mask)
        return {
            'image': image,
            'masks': instance_masks,
            'boxes': boxes,
            'labels': torch.ones(len(boxes), dtype=torch.long),  # 所有对象都是细胞
            'sample_id': sample['sample_id'],
            'cell_type': sample['cell_type'],
            'date': sample['date'],
            'magnification': sample['magnification'],
            'dataset_id': sample['dataset_id']
        }
    
    def _get_empty_sample(self) -> Dict[str, torch.Tensor]:
        """获取空样本"""
        h, w = self.config.image_size
        return {
            'image': torch.zeros(3, h, w),
            'masks': torch.zeros(0, h, w),
            'boxes': torch.zeros(0, 4),
            'labels': torch.zeros(0, dtype=torch.long),
            'sample_id': 'empty',
            'cell_type': 'unknown',
            'date': 'unknown',
            'magnification': 'unknown',
            'dataset_id': 'unknown'
        }
    
    def _process_mask(self, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """处理掩码，提取实例和边界框"""
        # 获取所有实例ID
        unique_ids = torch.unique(mask)
        unique_ids = unique_ids[unique_ids > 0]  # 排除背景
        
        if len(unique_ids) == 0:
            return torch.zeros(0, mask.shape[-2], mask.shape[-1]), torch.zeros(0, 4)
        
        # 限制实例数量
        if len(unique_ids) > self.config.max_objects_per_image:
            unique_ids = unique_ids[:self.config.max_objects_per_image]
        
        instance_masks = []
        boxes = []
        
        for instance_id in unique_ids:
            # 创建单个实例掩码
            instance_mask = (mask == instance_id).float()
            
            # 过滤太小的对象
            if torch.sum(instance_mask) < self.config.min_object_size:
                continue
            
            # 过滤太大的对象
            if (self.config.max_object_size is not None and 
                torch.sum(instance_mask) > self.config.max_object_size):
                continue
            
            instance_masks.append(instance_mask)
            
            # 计算边界框
            pos = torch.where(instance_mask)
            if len(pos[0]) > 0:
                xmin = torch.min(pos[1])
                xmax = torch.max(pos[1])
                ymin = torch.min(pos[0])
                ymax = torch.max(pos[0])
                
                boxes.append([xmin, ymin, xmax, ymax])
        
        if len(instance_masks) == 0:
            return torch.zeros(0, mask.shape[-2], mask.shape[-1]), torch.zeros(0, 4)
        
        instance_masks = torch.stack(instance_masks)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        
        return instance_masks, boxes


class SAMDataset(CellSegmentationDataset):
    """专门为SAM模型优化的数据集 - 修复完整版"""
    
    def __init__(
        self,
        data_dir: str = None,
        config: DataConfig = None,
        split: str = "train",
        transform: Optional[Any] = None,
        samples: Optional[List[Dict]] = None
    ):
        self.data_dir = Path(data_dir) if data_dir else None
        self.config = config
        self.split = split
        self.transform = transform
        
        # 支持从预划分的样本加载数据
        if samples is not None:
            self.samples = samples
            print(f"使用预划分的{split}样本: {len(self.samples)} 个")
        else:
            self.samples = self._load_samples()
        
        # 创建增强变换
        if transform is None:
            self.transform = self._create_default_transforms()
        
        print(f"SAM数据集 {split} 加载完成: {len(self.samples)} 个样本")
    
    def _load_samples(self) -> List[Dict]:
        """加载数据样本 - 支持细胞类型过滤"""
        if self.data_dir is None:
            return []
            
        samples = []
        
        try:
            valid_datasets = DatasetPathValidator.validate_dataset_structure(self.data_dir)
            print(f"发现 {len(valid_datasets)} 个有效数据集")
            
            for dataset_info in valid_datasets:
                # 添加细胞类型过滤
                if hasattr(self.config, '_cell_types_filter') and self.config._cell_types_filter:
                    if dataset_info['cell_type'] not in self.config._cell_types_filter:
                        continue
                
                images_dir = Path(dataset_info['images_dir'])
                masks_dir = Path(dataset_info['masks_dir'])
                
                image_mask_pairs = self._get_image_mask_pairs(images_dir, masks_dir)
                
                for img_path, mask_path in image_mask_pairs:
                    samples.append({
                        'image_path': str(img_path),
                        'mask_path': str(mask_path),
                        'sample_id': img_path.stem,
                        'cell_type': dataset_info['cell_type'],
                        'date': dataset_info['date'],
                        'magnification': dataset_info['magnification'],
                        'dataset_id': dataset_info['dataset_id']
                    })
            
            if hasattr(self.config, '_cell_types_filter') and self.config._cell_types_filter:
                print(f"过滤后样本数 ({self.config._cell_types_filter}): {len(samples)}")
            
            samples = self._split_samples(samples)
            
            valid_samples = []
            for sample in samples:
                if self._validate_sample(sample):
                    valid_samples.append(sample)
            
            return valid_samples
            
        except Exception as e:
            print(f"加载数据样本失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
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
    
    def _split_samples(self, samples: List[Dict]) -> List[Dict]:
        """根据split参数分割数据"""
        if not samples:
            return []
        
        # 按数据集分组
        dataset_groups = {}
        for sample in samples:
            dataset_id = sample['dataset_id']
            if dataset_id not in dataset_groups:
                dataset_groups[dataset_id] = []
            dataset_groups[dataset_id].append(sample)
        
        # 为每个数据集进行分割
        split_samples = []
        for dataset_id, dataset_samples in dataset_groups.items():
            n_total = len(dataset_samples)
            n_train = int(n_total * self.config.train_split_ratio)
            n_val = int(n_total * self.config.val_split_ratio)
            
            # 随机打乱
            random.seed(self.config.split_seed)
            random.shuffle(dataset_samples)
            
            if self.split == "train":
                split_samples.extend(dataset_samples[:n_train])
            elif self.split == "val":
                split_samples.extend(dataset_samples[n_train:n_train + n_val])
            elif self.split == "test":
                split_samples.extend(dataset_samples[n_train + n_val:])
            else:  # "all"
                split_samples.extend(dataset_samples)
        
        return split_samples
    
    def _validate_sample(self, sample: Dict) -> bool:
        """验证样本有效性"""
        img_path = Path(sample['image_path'])
        mask_path = Path(sample['mask_path'])
        
        if not (img_path.exists() and mask_path.exists()):
            return False
        
        try:
            image = load_image(img_path, convert_to_grayscale=False)
            mask = load_mask(mask_path)
            
            if image is None or mask is None:
                return False
            
            if image.shape[:2] != mask.shape:
                return False
            
            if self.config.filter_empty_images and np.max(mask) == 0:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _create_default_transforms(self):
        """创建默认的数据变换"""
        if self.split == "train" and self.config.use_data_augmentation:
            # 训练时使用数据增强
            transform = A.Compose([
                A.Resize(self.config.image_size[0], self.config.image_size[1]),
                A.OneOf([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                ], p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, 
                        contrast_limit=0.2, 
                        p=0.5
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=20,
                        val_shift_limit=20,
                        p=0.5
                    ),
                ], p=0.3),
                A.OneOf([
                    A.GaussianBlur(blur_limit=3, p=0.3),
                    A.GaussNoise(p=0.3),
                ], p=0.2),
                A.Normalize(mean=self.config.normalize_mean, std=self.config.normalize_std),
                ToTensorV2()
            ])
        else:
            # 验证/测试时只做基本变换
            transform = A.Compose([
                A.Resize(self.config.image_size[0], self.config.image_size[1]),
                A.Normalize(mean=self.config.normalize_mean, std=self.config.normalize_std),
                ToTensorV2()
            ])
        
        return transform
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本 - 修复版本"""
        try:
            sample = self.samples[idx]
            
            # 加载图像和掩码
            image = load_image(sample['image_path'], convert_to_grayscale=False)
            mask = load_mask(sample['mask_path'])
            
            if image is None or mask is None:
                return self._get_empty_sample()
            
            # 确保图像是RGB格式
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=-1)
            
            # 应用变换
            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            
            # 🔧 关键修复：保持多实例掩码，不要合并！
            instance_masks, boxes = self._process_mask_for_sam(mask)
            
            # 为SAM生成点提示
            point_prompts = self._generate_point_prompts_for_instances(instance_masks)
            
            return {
                'images': image,  # [3, H, W]
                'point_coords': point_prompts['coords'],
                'point_labels': point_prompts['labels'],
                'boxes': boxes,
                'mask_inputs': None,
                'multimask_output': True,  # 启用多掩码输出
                'ground_truth_masks': instance_masks,  # 🎯 [N, H, W] - 多个二进制掩码
                'sample_id': sample['sample_id'],
                'cell_type': sample['cell_type'],
                'date': sample['date'],
                'magnification': sample['magnification'],
                'dataset_id': sample['dataset_id']
            }
            
        except Exception as e:
            print(f"加载样本失败 {idx}: {e}")
            return self._get_empty_sample()
    
    def _get_empty_sample(self) -> Dict[str, Any]:
        """获取空样本"""
        h, w = self.config.image_size
        return {
            'images': torch.zeros(3, h, w),
            'point_coords': torch.zeros(0, 2),
            'point_labels': torch.zeros(0),
            'boxes': torch.zeros(0, 4),
            'mask_inputs': None,
            'multimask_output': True,
            'ground_truth_masks': torch.zeros(1, h, w),  # 至少一个空掩码
            'sample_id': 'empty',
            'cell_type': 'unknown',
            'date': 'unknown',
            'magnification': 'unknown',
            'dataset_id': 'unknown'
        }
    
    def _process_mask_for_sam(self, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """为SAM处理多实例掩码 - 保持分离状态"""
        # 获取所有实例ID
        unique_ids = torch.unique(mask)
        unique_ids = unique_ids[unique_ids > 0]  # 排除背景
        
        if len(unique_ids) == 0:
            # 没有实例，返回空掩码
            return torch.zeros(1, mask.shape[-2], mask.shape[-1]), torch.zeros(1, 4)
        
        # 限制实例数量
        if len(unique_ids) > self.config.max_objects_per_image:
            unique_ids = unique_ids[:self.config.max_objects_per_image]
        
        instance_masks = []
        boxes = []
        
        for instance_id in unique_ids:
            # 🎯 关键：创建单个实例的二进制掩码
            instance_mask = (mask == instance_id).float()
            
            # 过滤太小的对象
            if torch.sum(instance_mask) < self.config.min_object_size:
                continue
            
            # 过滤太大的对象
            if (self.config.max_object_size is not None and 
                torch.sum(instance_mask) > self.config.max_object_size):
                continue
            
            instance_masks.append(instance_mask)
            
            # 计算边界框
            pos = torch.where(instance_mask)
            if len(pos[0]) > 0:
                xmin = torch.min(pos[1])
                xmax = torch.max(pos[1])
                ymin = torch.min(pos[0])
                ymax = torch.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
        
        if len(instance_masks) == 0:
            return torch.zeros(1, mask.shape[-2], mask.shape[-1]), torch.zeros(1, 4)
        
        # 🎯 返回多个二进制掩码 [N, H, W]
        instance_masks = torch.stack(instance_masks)  # [N, H, W]
        boxes = torch.tensor(boxes, dtype=torch.float32)
        
        return instance_masks, boxes
    
    def _generate_point_prompts_for_instances(self, instance_masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """为每个实例生成点提示 - 修复版本"""
        all_coords = []
        all_labels = []
        
        for mask in instance_masks:
            # 为每个实例生成1-2个正例点
            pos_coords = self._sample_points_from_mask(mask, num_points=1, positive=True)
            
            # 可选：添加负例点（在其他实例或背景区域）
            neg_coords = self._sample_points_from_mask(mask, num_points=1, positive=False)
            
            if len(pos_coords) > 0:
                coords = torch.cat([pos_coords, neg_coords], dim=0) if len(neg_coords) > 0 else pos_coords
                labels = torch.cat([
                    torch.ones(len(pos_coords)), 
                    torch.zeros(len(neg_coords))
                ]) if len(neg_coords) > 0 else torch.ones(len(pos_coords))
                
                all_coords.append(coords)
                all_labels.append(labels)
        
        if len(all_coords) > 0:
            return {
                'coords': torch.cat(all_coords, dim=0),
                'labels': torch.cat(all_labels, dim=0)
            }
        else:
            return {
                'coords': torch.zeros(0, 2),
                'labels': torch.zeros(0)
            }
    
    def _sample_points_from_mask(self, mask: torch.Tensor, num_points: int, positive: bool) -> torch.Tensor:
        """从掩码中采样点 - 🔧 新增缺失的方法"""
        try:
            if positive:
                # 从前景区域采样
                pos = torch.where(mask > 0.5)
            else:
                # 从背景区域采样
                pos = torch.where(mask <= 0.5)
            
            if len(pos[0]) == 0:
                return torch.zeros(0, 2)
            
            # 随机选择点
            num_available = len(pos[0])
            num_to_sample = min(num_points, num_available)
            
            if num_to_sample == 0:
                return torch.zeros(0, 2)
            
            indices = torch.randperm(num_available)[:num_to_sample]
            
            coords = torch.stack([
                pos[1][indices],  # x coordinates
                pos[0][indices]   # y coordinates
            ], dim=1)
            
            return coords.float()
            
        except Exception as e:
            print(f"采样点失败: {e}")
            return torch.zeros(0, 2)


# 🔧 修复collate_fn函数
def collate_fn_multi_instance(batch):
    """多实例SAM的批处理函数 - 修复版本"""
    if len(batch) == 0:
        return {}
    
    images = []
    all_point_coords = []
    all_point_labels = []
    all_boxes = []
    all_masks = []
    sample_ids = []
    
    max_instances = 0
    
    # 收集所有数据，找到最大实例数
    for item in batch:
        try:
            images.append(item['images'])
            
            if 'point_coords' in item and isinstance(item['point_coords'], torch.Tensor):
                all_point_coords.append(item['point_coords'])
            else:
                all_point_coords.append(torch.zeros(0, 2))
            
            if 'point_labels' in item and isinstance(item['point_labels'], torch.Tensor):
                all_point_labels.append(item['point_labels'])
            else:
                all_point_labels.append(torch.zeros(0))
            
            if 'boxes' in item and isinstance(item['boxes'], torch.Tensor):
                all_boxes.append(item['boxes'])
            else:
                all_boxes.append(torch.zeros(0, 4))
            
            # 处理多实例掩码
            if 'ground_truth_masks' in item:
                masks = item['ground_truth_masks']
                if isinstance(masks, torch.Tensor):
                    if len(masks.shape) == 2:
                        masks = masks.unsqueeze(0)  # [H, W] -> [1, H, W]
                    max_instances = max(max_instances, masks.shape[0])
                    all_masks.append(masks)
                else:
                    # 处理异常情况
                    h, w = item['images'].shape[-2:]
                    default_mask = torch.zeros(1, h, w, dtype=torch.float32)
                    all_masks.append(default_mask)
                    max_instances = max(max_instances, 1)
            else:
                # 如果没有掩码，创建默认掩码
                h, w = item['images'].shape[-2:]
                default_mask = torch.zeros(1, h, w, dtype=torch.float32)
                all_masks.append(default_mask)
                max_instances = max(max_instances, 1)
            
            sample_ids.append(item.get('sample_id', f'sample_{len(sample_ids)}'))
            
        except Exception as e:
            print(f"处理批次项目失败: {e}")
            continue
    
    if len(images) == 0:
        return {}
    
    # 堆叠图像
    try:
        images = torch.stack(images)
    except Exception as e:
        print(f"堆叠图像失败: {e}")
        return {}
    
    # 🔧 关键修复：正确处理多实例掩码
    batch_size = len(all_masks)
    h, w = all_masks[0].shape[-2:]
    
    # 创建统一的掩码张量 [B, max_instances, H, W]
    unified_masks = torch.zeros(batch_size, max_instances, h, w, dtype=torch.float32)
    
    for i, masks in enumerate(all_masks):
        try:
            num_instances = masks.shape[0]
            unified_masks[i, :num_instances] = masks
        except Exception as e:
            print(f"处理掩码 {i} 失败: {e}")
            continue
    
    return {
        'images': images,
        'point_coords': all_point_coords,
        'point_labels': all_point_labels,
        'boxes': all_boxes,
        'ground_truth_masks': unified_masks,  # [B, max_instances, H, W]
        'sample_ids': sample_ids
    }


# 🔧 修复create_data_loaders函数
def create_data_loaders(config: DataConfig, dataset_type: str = "standard") -> Dict[str, DataLoader]:
    """创建数据加载器 - 修复版本"""
    
    datasets = {}
    data_loaders = {}
    
    # 选择数据集类型
    dataset_class = SAMDataset if dataset_type == "sam" else SAMDataset  # 都使用SAM数据集
    
    # 🔧 检查是否使用数据划分
    use_data_splitting = (
        config.train_data_dir and 
        hasattr(config, 'test_split_ratio') and
        config.test_split_ratio > 0 and 
        hasattr(config, 'use_cached_split') and
        config.use_cached_split
    )
    
    if use_data_splitting:
        print("使用数据划分模式...")
        
        try:
            from utils.data_splitter import create_data_split, print_split_summary
            
            # 准备细胞类型过滤
            cell_types = getattr(config, '_cell_types_filter', None)
            
            # 执行数据划分
            split_result = create_data_split(
                data_dir=config.train_data_dir,
                train_ratio=config.train_split_ratio,
                val_ratio=config.val_split_ratio,
                test_ratio=config.test_split_ratio,
                cell_types=cell_types,
                split_method=config.split_method,
                seed=config.split_seed,
                split_storage_dir=config.split_storage_dir,
                use_cached=config.use_cached_split
            )
            
            # 打印划分摘要
            print_split_summary(split_result)
            
            # 创建数据集
            if len(split_result.train_samples) > 0:
                datasets['train'] = dataset_class(
                    data_dir=None,
                    config=config,
                    split='train',
                    samples=split_result.train_samples
                )
                
                data_loaders['train'] = DataLoader(
                    datasets['train'],
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=0,  # 🔧 设置为0避免多进程问题
                    pin_memory=config.pin_memory,
                    collate_fn=collate_fn_multi_instance
                )
            
            # 验证集
            if len(split_result.val_samples) > 0:
                datasets['val'] = dataset_class(
                    data_dir=None,
                    config=config,
                    split='val',
                    samples=split_result.val_samples
                )
                
                data_loaders['val'] = DataLoader(
                    datasets['val'],
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=0,  # 🔧 设置为0避免多进程问题
                    pin_memory=config.pin_memory,
                    collate_fn=collate_fn_multi_instance
                )
            
            # 测试集
            if len(split_result.test_samples) > 0:
                datasets['test'] = dataset_class(
                    data_dir=None,
                    config=config,
                    split='test',
                    samples=split_result.test_samples
                )
                
                data_loaders['test'] = DataLoader(
                    datasets['test'],
                    batch_size=1,  # 测试时batch_size=1
                    shuffle=False,
                    num_workers=0,  # 🔧 设置为0避免多进程问题
                    pin_memory=config.pin_memory,
                    collate_fn=collate_fn_multi_instance
                )
        
        except Exception as e:
            print(f"数据划分失败，回退到传统模式: {e}")
            use_data_splitting = False
    
    # 🔧 传统模式（不使用数据划分）
    if not use_data_splitting:
        print("使用传统数据加载模式...")
        
        # 训练集
        if config.train_data_dir:
            datasets['train'] = dataset_class(
                data_dir=config.train_data_dir,
                config=config,
                split='train'
            )
            
            if len(datasets['train']) > 0:
                data_loaders['train'] = DataLoader(
                    datasets['train'],
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=0,  # 🔧 设置为0避免多进程问题
                    pin_memory=config.pin_memory,
                    collate_fn=collate_fn_multi_instance
                )
            else:
                print("警告: 训练集为空，跳过创建训练数据加载器")
        
        # 验证集
        if hasattr(config, 'val_data_dir') and config.val_data_dir:
            datasets['val'] = dataset_class(
                data_dir=config.val_data_dir,
                config=config,
                split='val'
            )
            
            if len(datasets['val']) > 0:
                data_loaders['val'] = DataLoader(
                    datasets['val'],
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=0,  # 🔧 设置为0避免多进程问题
                    pin_memory=config.pin_memory,
                    collate_fn=collate_fn_multi_instance
                )
        elif config.train_data_dir and 'train' in datasets and len(datasets['train']) > 0:
            # 如果没有指定验证集，从训练数据中创建
            datasets['val'] = dataset_class(
                data_dir=config.train_data_dir,
                config=config,
                split='val'
            )
            
            if len(datasets['val']) > 0:
                data_loaders['val'] = DataLoader(
                    datasets['val'],
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=0,  # 🔧 设置为0避免多进程问题
                    pin_memory=config.pin_memory,
                    collate_fn=collate_fn_multi_instance
                )
        
        # 测试集
        if hasattr(config, 'test_data_dir') and config.test_data_dir:
            datasets['test'] = dataset_class(
                data_dir=config.test_data_dir,
                config=config,
                split='test'
            )
            
            if len(datasets['test']) > 0:
                data_loaders['test'] = DataLoader(
                    datasets['test'],
                    batch_size=1,  # 测试时batch_size=1
                    shuffle=False,
                    num_workers=0,  # 🔧 设置为0避免多进程问题
                    pin_memory=config.pin_memory,
                    collate_fn=collate_fn_multi_instance
                )
    
    return data_loaders


# 🔧 修复原有的collate_fn函数
def collate_fn(batch):
    """兼容性collate函数 - 调用多实例版本"""
    return collate_fn_multi_instance(batch)

def split_dataset(data_dir: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """将数据集分割为训练/验证/测试集 - 已废弃，建议使用数据划分功能"""
    print(f"⚠️  split_dataset 函数已废弃")
    print(f"建议使用新的数据划分功能，支持缓存和更好的管理")
    print(f"请在配置中设置 test_split_ratio 参数")
    
    # 计算测试集比例
    test_ratio = 1.0 - train_ratio - val_ratio
    
    print(f"数据分割功能已整合到数据集类中")
    print(f"训练/验证/测试比例: {train_ratio}/{val_ratio}/{test_ratio}")
    print(f"数据目录: {data_dir}")
    
    # 创建一个临时数据集来验证数据结构
    from config.lora_config import DataConfig
    config = DataConfig()
    config.train_split_ratio = train_ratio
    config.val_split_ratio = val_ratio
    config.test_split_ratio = test_ratio
    
    try:
        # 测试数据加载
        train_dataset = CellSegmentationDataset(data_dir, config, split='train')
        val_dataset = CellSegmentationDataset(data_dir, config, split='val')
        test_dataset = CellSegmentationDataset(data_dir, config, split='test')
        
        print(f"数据分割结果:")
        print(f"  训练集: {len(train_dataset)} 样本")
        print(f"  验证集: {len(val_dataset)} 样本")
        print(f"  测试集: {len(test_dataset)} 样本")
        
    except Exception as e:
        print(f"数据分割测试失败: {e}")
        import traceback
        traceback.print_exc()


# 🔧 新增：数据划分管理功能
def list_cached_splits(split_storage_dir: str = "./data/lora_split") -> List[Dict]:
    """列出所有缓存的数据划分"""
    try:
        splitter = DatasetSplitter("", split_storage_dir)
        return splitter.list_cached_splits()
    except Exception as e:
        print(f"列出缓存划分失败: {e}")
        return []


def clean_old_splits(split_storage_dir: str = "./data/lora_split", keep_recent: int = 10):
    """清理旧的数据划分文件"""
    try:
        splitter = DatasetSplitter("", split_storage_dir)
        splitter.clean_old_splits(keep_recent)
    except Exception as e:
        print(f"清理旧划分失败: {e}")


def preview_data_split(data_dir: str,
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      test_ratio: float = 0.1,
                      cell_types: Optional[List[str]] = None,
                      split_method: str = "random",
                      seed: int = 42,
                      split_storage_dir: str = "./data/lora_split") -> Dict:
    """预览数据划分结果，不实际创建文件"""
    try:
        splitter = DatasetSplitter(data_dir, split_storage_dir)
        
        # 创建临时划分
        split_result = splitter.create_new_split(
            train_ratio, val_ratio, test_ratio, cell_types, split_method, seed
        )
        
        # 返回统计信息
        stats = {
            'total_samples': len(split_result.train_samples) + len(split_result.val_samples) + len(split_result.test_samples),
            'train_count': len(split_result.train_samples),
            'val_count': len(split_result.val_samples),
            'test_count': len(split_result.test_samples),
            'split_info': split_result.split_info
        }
        
        # 按细胞类型统计
        all_samples = split_result.train_samples + split_result.val_samples + split_result.test_samples
        cell_type_counts = {}
        for sample in all_samples:
            cell_type = sample.get('cell_type', 'unknown')
            cell_type_counts[cell_type] = cell_type_counts.get(cell_type, 0) + 1
        
        stats['cell_type_distribution'] = cell_type_counts
        
        return stats
        
    except Exception as e:
        print(f"预览数据划分失败: {e}")
        return {}


def validate_data_split_config(train_ratio: float, val_ratio: float, test_ratio: float) -> bool:
    """验证数据划分配置的有效性"""
    total_ratio = train_ratio + val_ratio + test_ratio
    
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"错误: 数据划分比例总和必须为1.0，当前为 {total_ratio}")
        return False
    
    if train_ratio <= 0:
        print("错误: 训练集比例必须大于0")
        return False
    
    if val_ratio < 0 or test_ratio < 0:
        print("错误: 验证集和测试集比例不能为负数")
        return False
    
    if train_ratio >= 1.0:
        print("错误: 训练集比例不能大于等于1.0")
        return False
    
    return True


def create_balanced_cell_type_split(data_dir: str,
                                   cell_types: List[str],
                                   train_ratio: float = 0.8,
                                   val_ratio: float = 0.1,
                                   test_ratio: float = 0.1,
                                   seed: int = 42,
                                   split_storage_dir: str = "./data/lora_split") -> Dict[str, DataSplit]:
    """为每种细胞类型创建平衡的数据划分"""
    splits = {}
    
    for cell_type in cell_types:
        print(f"为细胞类型 {cell_type} 创建数据划分...")
        
        try:
            split_result = create_data_split(
                data_dir=data_dir,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                cell_types=[cell_type],  # 只处理当前细胞类型
                split_method="random",
                seed=seed,
                split_storage_dir=split_storage_dir,
                use_cached=True
            )
            
            splits[cell_type] = split_result
            print(f"  {cell_type}: train={len(split_result.train_samples)}, "
                  f"val={len(split_result.val_samples)}, test={len(split_result.test_samples)}")
            
        except Exception as e:
            print(f"为细胞类型 {cell_type} 创建划分失败: {e}")
    
    return splits


def merge_data_splits(splits: Dict[str, DataSplit]) -> DataSplit:
    """合并多个数据划分结果"""
    merged_train = []
    merged_val = []
    merged_test = []
    
    for cell_type, split_result in splits.items():
        merged_train.extend(split_result.train_samples)
        merged_val.extend(split_result.val_samples)
        merged_test.extend(split_result.test_samples)
    
    # 打乱合并后的数据
    import random
    random.shuffle(merged_train)
    random.shuffle(merged_val)
    random.shuffle(merged_test)
    
    # 创建合并的划分信息
    merged_info = {
        'merged_from': list(splits.keys()),
        'total_samples': len(merged_train) + len(merged_val) + len(merged_test),
        'train_count': len(merged_train),
        'val_count': len(merged_val),
        'test_count': len(merged_test),
        'created_at': datetime.datetime.now().isoformat()
    }
    
    return DataSplit(merged_train, merged_val, merged_test, merged_info)


def get_data_split_statistics(split_result: DataSplit) -> Dict:
    """获取数据划分的详细统计信息"""
    all_samples = (split_result.train_samples + 
                  split_result.val_samples + 
                  split_result.test_samples)
    
    if not all_samples:
        return {}
    
    # 基本统计
    stats = {
        'total_samples': len(all_samples),
        'train_count': len(split_result.train_samples),
        'val_count': len(split_result.val_samples),
        'test_count': len(split_result.test_samples)
    }
    
    # 按细胞类型统计
    cell_type_stats = {}
    for sample in all_samples:
        cell_type = sample.get('cell_type', 'unknown')
        if cell_type not in cell_type_stats:
            cell_type_stats[cell_type] = {'total': 0, 'train': 0, 'val': 0, 'test': 0}
        cell_type_stats[cell_type]['total'] += 1
    
    # 分别统计各个划分中的细胞类型
    for sample in split_result.train_samples:
        cell_type = sample.get('cell_type', 'unknown')
        if cell_type in cell_type_stats:
            cell_type_stats[cell_type]['train'] += 1
    
    for sample in split_result.val_samples:
        cell_type = sample.get('cell_type', 'unknown')
        if cell_type in cell_type_stats:
            cell_type_stats[cell_type]['val'] += 1
    
    for sample in split_result.test_samples:
        cell_type = sample.get('cell_type', 'unknown')
        if cell_type in cell_type_stats:
            cell_type_stats[cell_type]['test'] += 1
    
    stats['cell_type_distribution'] = cell_type_stats
    
    # 按数据集统计
    dataset_stats = {}
    for sample in all_samples:
        dataset_id = sample.get('dataset_id', 'unknown')
        if dataset_id not in dataset_stats:
            dataset_stats[dataset_id] = {'total': 0, 'train': 0, 'val': 0, 'test': 0}
        dataset_stats[dataset_id]['total'] += 1
    
    for sample in split_result.train_samples:
        dataset_id = sample.get('dataset_id', 'unknown')
        if dataset_id in dataset_stats:
            dataset_stats[dataset_id]['train'] += 1
    
    for sample in split_result.val_samples:
        dataset_id = sample.get('dataset_id', 'unknown')
        if dataset_id in dataset_stats:
            dataset_stats[dataset_id]['val'] += 1
    
    for sample in split_result.test_samples:
        dataset_id = sample.get('dataset_id', 'unknown')
        if dataset_id in dataset_stats:
            dataset_stats[dataset_id]['test'] += 1
    
    stats['dataset_distribution'] = dataset_stats
    
    return stats


def export_data_split_report(split_result: DataSplit, output_path: str):
    """导出数据划分报告"""
    import json
    
    # 获取统计信息
    stats = get_data_split_statistics(split_result)
    
    # 创建报告
    report = {
        'generation_time': datetime.datetime.now().isoformat(),
        'split_info': split_result.split_info,
        'statistics': stats,
        'sample_details': {
            'train_samples': split_result.train_samples,
            'val_samples': split_result.val_samples,
            'test_samples': split_result.test_samples
        }
    }
    
    # 保存报告
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"数据划分报告已导出到: {output_path}")


def load_data_split_from_report(report_path: str) -> Optional[DataSplit]:
    """从报告文件加载数据划分"""
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        if 'sample_details' not in report:
            print(f"报告文件格式不正确: {report_path}")
            return None
        
        sample_details = report['sample_details']
        split_info = report.get('split_info', {})
        
        return DataSplit(
            train_samples=sample_details['train_samples'],
            val_samples=sample_details['val_samples'],
            test_samples=sample_details['test_samples'],
            split_info=split_info
        )
        
    except Exception as e:
        print(f"从报告加载数据划分失败: {e}")
        return None