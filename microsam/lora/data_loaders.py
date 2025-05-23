"""
LoRA训练数据加载器
支持细胞分割任务的数据加载和预处理
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

from config.lora_config import DataConfig
from utils.file_utils import load_image, load_mask


class CellSegmentationDataset(Dataset):
    """细胞分割数据集"""
    
    def __init__(
        self,
        data_dir: str,
        config: DataConfig,
        split: str = "train",
        transform: Optional[Any] = None
    ):
        self.data_dir = Path(data_dir)
        self.config = config
        self.split = split
        self.transform = transform
        
        # 加载数据
        self.samples = self._load_samples()
        
        # 创建增强变换
        if transform is None:
            self.transform = self._create_default_transforms()
        
        print(f"加载了 {len(self.samples)} 个{split}样本")
    
    def _load_samples(self) -> List[Dict]:
        """加载数据样本"""
        samples = []
        
        # 检查数据目录结构
        if self._is_standard_structure():
            samples = self._load_standard_structure()
        elif self._is_mapping_structure():
            samples = self._load_mapping_structure()
        else:
            samples = self._load_flat_structure()
        
        # 过滤无效样本
        valid_samples = []
        for sample in samples:
            if self._validate_sample(sample):
                valid_samples.append(sample)
        
        return valid_samples
    
    def _is_standard_structure(self) -> bool:
        """检查是否为标准结构 (images/ 和 masks/ 文件夹)"""
        return (self.data_dir / "images").exists() and (self.data_dir / "masks").exists()
    
    def _is_mapping_structure(self) -> bool:
        """检查是否为映射结构 (有mapping.json文件)"""
        return (self.data_dir / "mapping.json").exists()
    
    def _load_standard_structure(self) -> List[Dict]:
        """加载标准结构数据"""
        samples = []
        images_dir = self.data_dir / "images"
        masks_dir = self.data_dir / "masks"
        
        for img_file in images_dir.glob("*"):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                # 查找对应的掩码
                mask_patterns = [
                    masks_dir / f"{img_file.stem}_seg.png",
                    masks_dir / f"{img_file.stem}.png",
                    masks_dir / f"{img_file.stem}_mask.png",
                    masks_dir / f"{img_file.stem}.tif"
                ]
                
                for mask_path in mask_patterns:
                    if mask_path.exists():
                        samples.append({
                            'image_path': str(img_file),
                            'mask_path': str(mask_path),
                            'sample_id': img_file.stem
                        })
                        break
        
        return samples
    
    def _load_mapping_structure(self) -> List[Dict]:
        """加载映射结构数据"""
        samples = []
        mapping_file = self.data_dir / "mapping.json"
        
        with open(mapping_file, 'r') as f:
            mapping_data = json.load(f)
        
        images_dir = Path(mapping_data.get('images_path', self.data_dir / "images"))
        masks_dir = Path(mapping_data.get('masks_path', self.data_dir / "masks"))
        
        for img_name, info in mapping_data.get('mapping', {}).items():
            img_path = images_dir / img_name
            mask_path = masks_dir / info['mask_file']
            
            if img_path.exists() and mask_path.exists():
                samples.append({
                    'image_path': str(img_path),
                    'mask_path': str(mask_path),
                    'sample_id': Path(img_name).stem
                })
        
        return samples
    
    def _load_flat_structure(self) -> List[Dict]:
        """加载平铺结构数据"""
        samples = []
        
        # 假设图像和掩码在同一目录下，通过文件名区分
        image_files = []
        mask_files = []
        
        for file_path in self.data_dir.glob("*"):
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                if any(keyword in file_path.name.lower() for keyword in ['mask', 'seg', 'label']):
                    mask_files.append(file_path)
                else:
                    image_files.append(file_path)
        
        # 匹配图像和掩码
        for img_file in image_files:
            for mask_file in mask_files:
                if img_file.stem in mask_file.name:
                    samples.append({
                        'image_path': str(img_file),
                        'mask_path': str(mask_file),
                        'sample_id': img_file.stem
                    })
                    break
        
        return samples
    
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
                    A.GaussNoise(var_limit=(0, 0.1), p=0.3),
                ], p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            # 验证/测试时只做基本变换
            transform = A.Compose([
                A.Resize(self.config.image_size[0], self.config.image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
            'sample_id': sample['sample_id']
        }
    
    def _get_empty_sample(self) -> Dict[str, torch.Tensor]:
        """获取空样本"""
        h, w = self.config.image_size
        return {
            'image': torch.zeros(3, h, w),
            'masks': torch.zeros(0, h, w),
            'boxes': torch.zeros(0, 4),
            'labels': torch.zeros(0, dtype=torch.long),
            'sample_id': 'empty'
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
    """专门为SAM模型优化的数据集"""
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)
        
        # SAM需要的特殊格式
        image = sample['image']
        masks = sample['masks']
        boxes = sample['boxes']
        
        # 为SAM准备输入
        if len(boxes) > 0:
            # 随机选择一些点作为prompt
            point_prompts = self._generate_point_prompts(masks)
            
            return {
                'image': image,
                'point_coords': point_prompts['coords'],
                'point_labels': point_prompts['labels'],
                'boxes': boxes,
                'mask_inputs': None,  # 可以为None
                'multimask_output': False,
                'ground_truth_masks': masks,
                'sample_id': sample['sample_id']
            }
        else:
            # 没有对象的情况
            h, w = image.shape[-2:]
            return {
                'image': image,
                'point_coords': torch.zeros(0, 2),
                'point_labels': torch.zeros(0),
                'boxes': torch.zeros(0, 4),
                'mask_inputs': None,
                'multimask_output': False,
                'ground_truth_masks': torch.zeros(0, h, w),
                'sample_id': sample['sample_id']
            }
    
    def _generate_point_prompts(self, masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """为每个掩码生成点提示"""
        all_coords = []
        all_labels = []
        
        for mask in masks:
            # 正例点（在对象内部）
            pos_coords = self._sample_points_from_mask(mask, num_points=1, positive=True)
            
            # 负例点（在对象外部）
            neg_coords = self._sample_points_from_mask(mask, num_points=1, positive=False)
            
            coords = torch.cat([pos_coords, neg_coords], dim=0)
            labels = torch.tensor([1] * len(pos_coords) + [0] * len(neg_coords))
            
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
        """从掩码中采样点"""
        if positive:
            # 从前景区域采样
            pos = torch.where(mask > 0.5)
        else:
            # 从背景区域采样
            pos = torch.where(mask <= 0.5)
        
        if len(pos[0]) == 0:
            return torch.zeros(0, 2)
        
        # 随机选择点
        indices = torch.randperm(len(pos[0]))[:num_points]
        
        coords = torch.stack([
            pos[1][indices],  # x coordinates
            pos[0][indices]   # y coordinates
        ], dim=1)
        
        return coords.float()


def create_data_loaders(config: DataConfig, dataset_type: str = "standard") -> Dict[str, DataLoader]:
    """创建数据加载器"""
    
    datasets = {}
    data_loaders = {}
    
    # 选择数据集类型
    dataset_class = SAMDataset if dataset_type == "sam" else CellSegmentationDataset
    
    # 训练集
    if config.train_data_dir:
        datasets['train'] = dataset_class(
            data_dir=config.train_data_dir,
            config=config,
            split='train'
        )
        
        data_loaders['train'] = DataLoader(
            datasets['train'],
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            collate_fn=collate_fn
        )
    
    # 验证集
    if config.val_data_dir:
        datasets['val'] = dataset_class(
            data_dir=config.val_data_dir,
            config=config,
            split='val'
        )
        
        data_loaders['val'] = DataLoader(
            datasets['val'],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            collate_fn=collate_fn
        )
    
    # 测试集
    if config.test_data_dir:
        datasets['test'] = dataset_class(
            data_dir=config.test_data_dir,
            config=config,
            split='test'
        )
        
        data_loaders['test'] = DataLoader(
            datasets['test'],
            batch_size=1,  # 测试时batch_size=1
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            collate_fn=collate_fn
        )
    
    return data_loaders


def collate_fn(batch):
    """自定义的批处理函数"""
    # 处理变长数据
    images = []
    all_point_coords = []
    all_point_labels = []
    all_boxes = []
    all_masks = []
    sample_ids = []
    
    for item in batch:
        images.append(item['image'])
        
        if 'point_coords' in item:
            all_point_coords.append(item['point_coords'])
            all_point_labels.append(item['point_labels'])
        
        if 'boxes' in item:
            all_boxes.append(item['boxes'])
        
        if 'ground_truth_masks' in item:
            all_masks.append(item['ground_truth_masks'])
        elif 'masks' in item:
            all_masks.append(item['masks'])
        
        sample_ids.append(item['sample_id'])
    
    # 堆叠图像
    images = torch.stack(images)
    
    # 返回批处理数据
    batch_data = {
        'images': images,
        'point_coords': all_point_coords,
        'point_labels': all_point_labels,
        'boxes': all_boxes,
        'ground_truth_masks': all_masks,
        'sample_ids': sample_ids
    }
    
    return batch_data


def split_dataset(data_dir: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """将数据集分割为训练/验证/测试集"""
    data_path = Path(data_dir)
    
    # 获取所有样本
    dataset = CellSegmentationDataset(data_dir, DataConfig(), split='all')
    samples = dataset.samples
    
    # 随机打乱
    random.shuffle(samples)
    
    # 计算分割点
    n_total = len(samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # 分割数据
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]
    
    # 创建分割目录
    for split_name, split_samples in [
        ('train', train_samples),
        ('val', val_samples), 
        ('test', test_samples)
    ]:
        split_dir = data_path / split_name
        split_dir.mkdir(exist_ok=True)
        
        # 保存样本列表
        with open(split_dir / 'samples.json', 'w') as f:
            json.dump(split_samples, f, indent=2)
    
    print(f"数据集分割完成:")
    print(f"  训练集: {len(train_samples)} 样本")
    print(f"  验证集: {len(val_samples)} 样本") 
    print(f"  测试集: {len(test_samples)} 样本")