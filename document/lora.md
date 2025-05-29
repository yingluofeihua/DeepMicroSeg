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

test 0.5

``` bash
nohup bash -c 'CUDA_VISIBLE_DEVICES=2 python microsam/lora_main.py train \
  --data-dir /LD-FS/home/yunshuchen/micro_sam/patch_0520 \
  --model vit_b_lm \
  --epochs 10 \
  --batch-size 8 \
  --cell-types VERO \
  --test-split 0.5 \
  --val-split 0.0 \
  --output-dir ./data/lora_model \
  --use-wandb \
  --wandb-project "sam_lora_VERO"' \
  > ./data/logs/train_VERO_vit_b_lm_0.4_0.1_0.5.log 2>&1 &
```

test 0.6

``` bash
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 python microsam/lora_main.py train \
  --data-dir /LD-FS/home/yunshuchen/micro_sam/patch_0520 \
  --model vit_b_lm \
  --epochs 10 \
  --batch-size 8 \
  --cell-types VERO \
  --test-split 0.6 \
  --val-split 0.0 \
  --output-dir ./data/lora_model \
  --use-wandb \
  --wandb-project "sam_lora_VERO"' \
  > ./data/logs/train_VERO_vit_b_lm_0.4_0.0_0.6.log 2>&1 &
```


test 0.7
``` bash
nohup bash -c 'CUDA_VISIBLE_DEVICES=1 python microsam/lora_main.py train \
  --data-dir /LD-FS/home/yunshuchen/micro_sam/patch_0520 \
  --model vit_b_lm \
  --epochs 10 \
  --batch-size 8 \
  --cell-types VERO \
  --test-split 0.7 \
  --val-split 0.0 \
  --output-dir ./data/lora_model \
  --use-wandb \
  --wandb-project "sam_lora_VERO"' \
  > ./data/logs/train_VERO_vit_b_lm_0.3_0.0_0.7.log 2>&1 &
```

test 0.8
``` bash
nohup bash -c 'CUDA_VISIBLE_DEVICES=3 python microsam/lora_main.py train \
  --data-dir /LD-FS/home/yunshuchen/micro_sam/patch_0520 \
  --model vit_b_lm \
  --epochs 10 \
  --batch-size 8 \
  --cell-types VERO \
  --test-split 0.8 \
  --val-split 0.0 \
  --output-dir ./data/lora_model \
  --use-wandb \
  --wandb-project "sam_lora_VERO"' \
  > ./data/logs/train_VERO_vit_b_lm_0.2_0.0_0.8.log 2>&1 &
```

test 0.9
``` bash
nohup bash -c 'CUDA_VISIBLE_DEVICES=4 python microsam/lora_main.py train \
  --data-dir /LD-FS/home/yunshuchen/micro_sam/patch_0520 \
  --model vit_b_lm \
  --epochs 10 \
  --batch-size 8 \
  --cell-types VERO \
  --test-split 0.9 \
  --val-split 0.0 \
  --output-dir ./data/lora_model \
  --use-wandb \
  --wandb-project "sam_lora_VERO"' \
  > ./data/logs/train_VERO_vit_b_lm_0.1_0.0_0.9.log 2>&1 &
```

test 0.95
``` bash
nohup bash -c 'CUDA_VISIBLE_DEVICES=5 python microsam/lora_main.py train \
  --data-dir /LD-FS/home/yunshuchen/micro_sam/patch_0520 \
  --model vit_b_lm \
  --epochs 10 \
  --batch-size 8 \
  --cell-types VERO \
  --test-split 0.95 \
  --val-split 0.0 \
  --output-dir ./data/lora_model \
  --use-wandb \
  --wandb-project "sam_lora_VERO"' \
  > ./data/logs/train_VERO_vit_b_lm_0.05_0.0_0.95.log 2>&1 &
```

