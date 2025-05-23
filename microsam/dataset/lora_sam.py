import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import imageio.v3 as imageio
from matplotlib import pyplot as plt
from glob import glob
from skimage.measure import label as connected_components

from typing import Dict, List, Tuple, Any, Optional
import torch
from torch_em.data import MinInstanceSampler
from torch_em.util.util import get_random_colors
from torch_em.util.debug import check_loader

import micro_sam.training as sam_training
from micro_sam.sample_data import fetch_tracking_example_data, fetch_tracking_segmentation_data
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation
import json
from pathlib import Path
import logging
from peft import LoraConfig, get_peft_model  # Hugging Face PEFT 库用于 LoRA
import pandas as pd

# 设置环境和路径
# root_dir = os.getcwd()  # 可修改为你的数据路径
# DATA_FOLDER = os.path.join(root_dir, "data")
# os.makedirs(DATA_FOLDER, exist_ok=True)
# device = "cuda" if torch.cuda.is_available() else "cpu"

class DatasetManager(object):
    """数据集管理器 - 自动发现和组织所有数据集"""
    
    def __init__(self, base_dir: str):
        """初始化数据集管理器

        Args:
            base_dir (str): 数据集根目录路径
        """
        self.base_dir = Path(base_dir)
        self.datasets = self._discover_datasets()
    
    def _discover_datasets(self) -> List[Dict]:
        """自动发现所有数据集"""
        datasets = []
        
        try:
            # 遍历所有可能的数据集路径
            for cell_type in self.base_dir.iterdir():
                if not cell_type.is_dir():
                    continue
                    
                for date_dir in cell_type.iterdir():
                    if not date_dir.is_dir():
                        continue
                        
                    for magnification in date_dir.iterdir():
                        if not magnification.is_dir():
                            continue
                            
                        images_dir = magnification / "images"
                        masks_dir = magnification / "masks"
                        
                        # 检查是否存在images和masks文件夹
                        if images_dir.exists() and masks_dir.exists():
                            # 检查是否有实际的图像文件
                            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
                            mask_files = list(masks_dir.glob("*.png")) + list(masks_dir.glob("*.tif"))
                            
                            if image_files and mask_files:
                                dataset_info = {
                                    'cell_type': cell_type.name,
                                    'date': date_dir.name,
                                    'magnification': magnification.name,
                                    'images_dir': str(images_dir),
                                    'masks_dir': str(masks_dir),
                                    'num_images': len(image_files),
                                    'num_masks': len(mask_files),
                                    'dataset_id': f"{cell_type.name}_{date_dir.name}_{magnification.name}"
                                }
                                datasets.append(dataset_info)
                                
        except Exception as e:
            print(f"Error discovering datasets: {e}")
        
        return sorted(datasets, key=lambda x: (x['cell_type'], x['date'], x['magnification']))
    
    def get_datasets_summary(self) -> pd.DataFrame:
        """获取数据集摘要"""
        return pd.DataFrame(self.datasets)
    
    def filter_datasets(self, cell_types=None, dates=None, magnifications=None) -> List[Dict]:
        """过滤数据集"""
        filtered = self.datasets
        
        if cell_types:
            filtered = [d for d in filtered if d['cell_type'] in cell_types]
        if dates:
            filtered = [d for d in filtered if d['date'] in dates]
        if magnifications:
            filtered = [d for d in filtered if d['magnification'] in magnifications]
            
        return filtered

dm = DatasetManager(base_dir="/LD-FS/home/yunshuchen/micro_sam/patch_0520")
print(dm.datasets)

# # 下载数据集
# image_dir = fetch_tracking_example_data(DATA_FOLDER)
# segmentation_dir = fetch_tracking_segmentation_data(DATA_FOLDER)

# # 创建数据加载器
# raw_key, label_key = "*.tif", "*.tif"
# train_roi = np.s_[:70, :, :]  # 前 70 帧用于训练
# val_roi = np.s_[70:, :, :]    # 其余用于验证
# batch_size = 1
# patch_shape = (1, 512, 512)
# train_instance_segmentation = True  # 训练卷积解码器以支持 AIS
# sampler = MinInstanceSampler(min_size=25)

# train_loader = sam_training.default_sam_loader(
#     raw_paths=image_dir,
#     raw_key=raw_key,
#     label_paths=segmentation_dir,
#     label_key=label_key,
#     with_segmentation_decoder=train_instance_segmentation,
#     patch_shape=patch_shape,
#     batch_size=batch_size,
#     is_seg_dataset=True,
#     rois=train_roi,
#     shuffle=True,
#     raw_transform=sam_training.identity,
#     sampler=sampler,
# )

# val_loader = sam_training.default_sam_loader(
#     raw_paths=image_dir,
#     raw_key=raw_key,
#     label_paths=segmentation_dir,
#     label_key=label_key,
#     with_segmentation_decoder=train_instance_segmentation,
#     patch_shape=patch_shape,
#     batch_size=batch_size,
#     is_seg_dataset=True,
#     rois=val_roi,
#     shuffle=True,
#     raw_transform=sam_training.identity,
#     sampler=sampler,
# )

# # 检查数据加载器
# check_loader(train_loader, 2, plt=True)

# # LoRA 配置
# lora_config = LoraConfig(
#     r=8,  # LoRA 秩
#     lora_alpha=16,  # 缩放因子
#     target_modules=["q_proj", "k_proj", "v_proj"],  # 注意力层
#     lora_dropout=0.05,
#     bias="none",
# )

# # 加载 SAM 模型并应用 LoRA
# model_type = "vit_b"
# checkpoint_name = "sam_hela_lora"
# save_root = os.path.join(root_dir, "models")

# # 初始化 SAM 模型
# from micro_sam.models import get_sam_model
# model = get_sam_model(model_type=model_type, device=device, with_segmentation_decoder=True)

# # 应用 LoRA
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()  # 查看可训练参数数量

# # 训练超参数
# n_epochs = 3
# n_objects_per_batch = 5
# learning_rate = 1e-4

# # 优化器和损失函数
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# loss_fn = sam_training.get_loss(with_segmentation_decoder=True)

# # 自定义训练循环（因为 train_sam 可能不支持 LoRA）
# def train_one_epoch(model, loader, optimizer, loss_fn, device):
#     model.train()
#     total_loss = 0
#     for batch_idx, (x, y) in enumerate(loader):
#         x, y = x.to(device), y.to(device)
#         optimizer.zero_grad()
#         pred = model(x)
#         loss = loss_fn(pred, y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#         if batch_idx % 10 == 0:
#             print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
#     return total_loss / len(loader)

# # 验证函数
# def validate(model, loader, loss_fn, device):
#     model.eval()
#     total_loss = 0
#     with torch.no_grad():
#         for x, y in loader:
#             x, y = x.to(device), y.to(device)
#             pred = model(x)
#             loss = loss_fn(pred, y)
#             total_loss += loss.item()
#     return total_loss / len(loader)

# # 训练循环
# best_val_loss = float("inf")
# best_checkpoint = os.path.join(save_root, "checkpoints", checkpoint_name, "best.pt")
# os.makedirs(os.path.dirname(best_checkpoint), exist_ok=True)

# for epoch in range(n_epochs):
#     print(f"Epoch {epoch + 1}/{n_epochs}")
#     train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
#     val_loss = validate(model, val_loader, loss_fn, device)
#     print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

#     # 保存最佳模型
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         model.save_pretrained(best_checkpoint)
#         print(f"Saved best model at {best_checkpoint}")

# # 推理：自动实例分割
# def run_automatic_instance_segmentation(image, checkpoint_path, model_type, device):
#     from peft import PeftModel
#     predictor, segmenter = get_predictor_and_segmenter(
#         model_type=model_type,
#         checkpoint=checkpoint_path,
#         device=device,
#         is_tiled=False,
#     )
#     # 加载 LoRA 模型
#     predictor.model = PeftModel.from_pretrained(predictor.model, checkpoint_path).merge_and_unload()

#     prediction = automatic_instance_segmentation(
#         predictor=predictor,
#         segmenter=segmenter,
#         input_path=image,
#         ndim=2,
#         tile_shape=None,
#         halo=None,
#     )
#     return prediction

# # 测试推理
# image_paths = sorted(glob(os.path.join(image_dir, "*.tif")))[:2]  # 测试前 2 张图像
# for image_path in image_paths:
#     image = imageio.imread(image_path)
#     prediction = run_automatic_instance_segmentation(
#         image=image,
#         checkpoint_path=best_checkpoint,
#         model_type=model_type,
#         device=device,
#     )

#     # 可视化
#     fig, ax = plt.subplots(1, 2, figsize=(10, 10))
#     ax[0].imshow(image, cmap="gray")
#     ax[0].set_title("Input Image")
#     ax[0].axis("off")
#     ax[1].imshow(prediction, cmap=get_random_colors(prediction), interpolation="nearest")
#     ax[1].set_title("Predictions (AIS)")
#     ax[1].axis("off")
#     plt.show()
#     plt.close()