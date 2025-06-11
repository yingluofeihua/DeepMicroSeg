## git 分支命名

| 命名格式              | 示例                    | 说明         |
| ----------------- | --------------------- | ---------- |
| `feature/<功能名>`   | `feature/user-auth`   | 开发新功能      |
| `fix/<问题描述>`      | `fix/login-bug`       | 修复 bug     |
| `dev/<模块名>`       | `dev/payment-service` | 某模块或阶段性开发  |
| `task/<任务编号>`     | `task/123-add-search` | 对应某个任务系统编号 |
| `experiment/<名字>` | `experiment/new-UI`   | 原型尝试或实验性功能 |

## 



## train

### 293T(_num_1)

```bash
nohup python microsam/lora_train.py \
    --train_image_dir /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/293T/train/images \
    --train_mask_dir /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/293T/train/masks \
    --val_image_dir /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/293T/val/images \
    --val_mask_dir /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/293T/val/masks \
    --model_type vit_b_lm \
    --checkpoint_name sam_lora_293T \
    --pretrained_checkpoint /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/LDCellData/checkpoint/fl_best.pt \
    --rank 8 \
    --attention_layers_to_update 9 10 11 \
    --save_root /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/results1/293T/b_lm_sam_lora_fl \
    --gpu_id 4 \
    --n_epochs 100 \
    --batch_size 8 \
    --train_instance_segmentation\
    --freeze_prompt_encoder &
```

### MSC

```bash
nohup python microsam/lora_train.py \
    --train_image_dir /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/MSC/train/images \
    --train_mask_dir /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/MSC/train/masks \
    --val_image_dir /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/MSC/val/images \
    --val_mask_dir /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/MSC/val/masks \
    --model_type vit_b_lm \
    --checkpoint_name sam_lora_MSC \
    --pretrained_checkpoint /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/LDCellData/checkpoint/fl_best.pt \
    --rank 8 \
    --attention_layers_to_update 9 10 11 \
    --save_root /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/results1/MSC/b_lm_sam_lora_fl \
    --gpu_id 3 \
    --n_epochs 100 \
    --batch_size 8 \
    --train_instance_segmentation \
    --freeze_prompt_encoder &
```

**不同图片数量**

```bash
nohup python microsam/lora_train.py \
    --train_image_dir /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/MSC/train/images \
    --train_mask_dir /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/MSC/train/masks \
    --val_image_dir /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/MSC/val/images \
    --val_mask_dir /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/MSC/val/masks \
    --model_type vit_b_lm \
    --checkpoint_name sam_lora_MSC \
    --pretrained_checkpoint /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/LDCellData/checkpoint/fl_best.pt \
    --rank 8 \
    --attention_layers_to_update 9 10 11 \
    --save_root /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/results1/MSC/b_lm_sam_lora_fl_number_1 \
    --train_number 1 \
    --gpu_id 1 \
    --n_epochs 100 \
    --batch_size 8 \
    --train_instance_segmentation \
    --freeze_prompt_encoder &
```

### RBD
```bash
nohup python microsam/lora_train.py \
    --train_image_dir /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/RBD/train/images \
    --train_mask_dir /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/RBD/train/masks \
    --val_image_dir /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/RBD/val/images \
    --val_mask_dir /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/RBD/val/masks \
    --model_type vit_b_lm \
    --checkpoint_name sam_lora_RBD \
    --pretrained_checkpoint /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/LDCellData/checkpoint/fl_best.pt \
    --rank 8 \
    --attention_layers_to_update 9 10 11 \
    --save_root /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/results1/RBD/b_lm_sam_lora_fl \
    --gpu_id 3 \
    --n_epochs 100 \
    --batch_size 8 \
    --train_instance_segmentation \
    --freeze_prompt_encoder &
```

## test

### 293T

```bash
python microsam/lora_test.py \
    --best_checkpoint /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/results1/293T/b_lm_sam_lora_fl/checkpoints/sam_lora_293T/best.pt \
    --test_image_dir /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/293T/test/images \
    --test_mask_dir /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/293T/test/masks \
    --results_dir /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/results1/293T/b_lm_sam_lora_fl/test \
    --model_type vit_b_lm \
    --gpu_id 2 \
    --rank 8 \
    --attention_layers_to_update 9 10 11 \
    --save_visualizations \
    --dpi 300
```

### MSC

```bash
python microsam/lora_test.py \
    --best_checkpoint /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/results1/MSC/b_lm_sam_lora_fl/checkpoints/sam_lora_MSC/best.pt \
    --test_image_dir /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/MSC/test/images \
    --test_mask_dir /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/MSC/test/masks \
    --results_dir /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/results1/MSC/b_lm_sam_lora_fl/test \
    --model_type vit_b_lm \
    --gpu_id 3 \
    --rank 8 \
    --attention_layers_to_update 9 10 11 \
    --save_visualizations \
    --dpi 300
```

### RBD

```bash
python microsam/lora_test.py \
    --best_checkpoint /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/results1/RBD/b_lm_sam_lora_fl/checkpoints/sam_lora_RBD/best.pt \
    --test_image_dir /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/RBD/test/images \
    --test_mask_dir /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/patch_0520/RBD/test/masks \
    --results_dir /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/results1/RBD/b_lm_sam_lora_fl/test \
    --model_type vit_b_lm \
    --gpu_id 4 \
    --rank 8 \
    --attention_layers_to_update 9 10 11 \
    --save_visualizations \
    --dpi 300
```



