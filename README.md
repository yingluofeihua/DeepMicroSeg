# MicroSAM - 细胞分割模型训练与评测系统

一个基于SAM (Segment Anything Model) 的细胞分割模型训练与批量评测系统，专门针对显微镜细胞图像设计，支持LoRA微调和多模型性能对比。

## 🌟 主要特性

### 🔬 专业的细胞分割
- 基于Meta SAM模型的细胞分割
- 支持多种细胞类型（MSC、Vero等）
- 适配显微镜图像的特殊需求

### 🎯 LoRA微调训练
- 轻量级LoRA适配器微调
- 显著减少训练参数和时间
- 支持多种SAM模型变体（vit_t_lm、vit_b_lm、vit_l_lm）
- 自动处理多实例掩码合并

### 📊 完整的评测指标
- **AP50/AP75** - 不同IoU阈值下的平均精度
- **IoU** - 交并比
- **Dice** - Dice系数
- **HD95** - 95%豪斯多夫距离
- **处理时间** - 推理效率评估

### 🚀 批量评测系统
- 自动发现和组织数据集
- 多模型并行评测
- 丰富的可视化报告
- 统一的CSV结果输出

## 📁 项目结构

```
microsam/
├── config/                 # 配置管理
│   ├── lora_config.py     # LoRA训练配置
│   ├── settings.py        # 评测系统配置
│   └── paths.py           # 路径管理
├── core/                   # 核心模块
│   ├── dataset_manager.py # 数据集管理
│   ├── evaluator.py       # 批量评测器
│   ├── lora_trainer.py    # LoRA训练器
│   ├── metrics.py         # 评测指标
│   ├── model_handler.py   # 模型管理
│   └── sam_model_loader.py # SAM模型加载
├── lora/                   # LoRA微调
│   ├── adapters.py        # LoRA适配器
│   ├── data_loaders.py    # 数据加载
│   ├── sam_lora_wrapper.py # SAM LoRA包装
│   └── training_utils.py  # 训练工具
├── utils/                  # 工具函数
│   ├── file_utils.py      # 文件处理
│   ├── model_utils.py     # 模型工具
│   ├── report_generator.py # 报告生成
│   └── visualization.py   # 可视化
├── main.py                 # 批量评测入口
└── lora_main.py           # LoRA训练入口
```

## 🔧 环境安装

### 系统要求
- Python 3.8+
- CUDA 11.0+ (推荐)
- 8GB+ GPU内存

### 安装依赖

```bash
# 克隆项目
git clone <your-repo-url>
cd microsam

# 安装基础依赖
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn
pip install scikit-image pillow tifffile
pip install albumentations opencv-python
pip install tqdm wandb tensorboard

# 安装micro_sam
pip install micro-sam

# 可选：性能监控
pip install gputil psutil
```

## 📂 数据格式

支持的数据集目录结构：

```
data/
├── 细胞类型1/
│   ├── 日期1/
│   │   ├── 放大倍数1/
│   │   │   ├── images/        # 原始图像
│   │   │   │   ├── img1.jpg
│   │   │   │   └── img2.png
│   │   │   └── masks/         # 分割掩码
│   │   │       ├── img1_seg.png
│   │   │       └── img2_seg.png
│   │   └── 放大倍数2/
│   └── 日期2/
└── 细胞类型2/
```

**支持的图像格式：** JPG, PNG, TIF, TIFF  
**掩码格式：** PNG, TIF (标签图或二值图)

## 🚀 快速开始

### 1. 批量评测 (零代码训练)

```bash
# 基本评测
python main.py --data-dir /path/to/data --output-dir ./results

# 指定模型和细胞类型
python main.py \
    --data-dir /path/to/data \
    --models vit_b_lm vit_l_lm \
    --cell-types MSC Vero \
    --output-dir ./results

# 使用预设配置
python main.py --preset fast --data-dir /path/to/data
```

### 2. LoRA微调训练

```bash
# 快速训练 (调试用)
python lora_main.py train \
    --preset quick \
    --data-dir /path/to/data

# 标准训练
python lora_main.py train \
    --data-dir /path/to/data \
    --model vit_b_lm \
    --epochs 10 \
    --batch-size 8 \
    --learning-rate 1e-4

# 训练后自动评测
python lora_main.py train-and-eval \
    --data-dir /path/to/train_data \
    --eval-data /path/to/test_data
```

### 3. 恢复训练

```bash
python lora_main.py resume \
    --checkpoint ./lora_experiments/checkpoints/checkpoint_epoch_5.pth
```

## ⚙️ 高级配置

### LoRA训练配置

```python
# config/lora_config.py
config = LoRATrainingSettings(
    lora=LoRAConfig(
        rank=8,                    # LoRA rank
        alpha=16.0,               # LoRA alpha
        dropout=0.1,              # Dropout率
        target_modules=["qkv", "proj"]  # 目标模块
    ),
    training=TrainingConfig(
        learning_rate=1e-4,
        batch_size=8,
        num_epochs=10
    ),
    model=ModelConfig(
        base_model_name="vit_b_lm",
        apply_lora_to=["image_encoder"]
    )
)
```

### 评测系统配置

```python
# config/settings.py
config = BatchEvaluationSettings(
    models=[
        ModelConfig("vit_t_lm"),
        ModelConfig("vit_b_lm"),
        ModelConfig("vit_l_lm")
    ],
    evaluation=EvaluationConfig(
        batch_size=None,  # 处理所有图像
        skip_existing=True,
        save_visualizations=True
    )
)
```

## 📊 结果输出

### 训练结果
```
lora_experiments/
├── experiment_name/
│   ├── config.json          # 训练配置
│   ├── checkpoints/         # 模型检查点
│   ├── final_model/         # 最终模型
│   │   ├── lora_weights.pth
│   │   └── merged_sam_model.pth
│   └── logs/               # TensorBoard日志
```

### 评测结果
```
batch_evaluation_results/
├── model_name/
│   └── dataset_id/
│       ├── results.csv      # 详细结果
│       └── summary.json     # 摘要统计
└── summary_report_timestamp/
    ├── final_evaluation_summary.csv
    ├── model_comparison_statistics.csv
    ├── executive_summary.json
    └── visualizations/      # 可视化图表
```

## 📈 性能监控

### 使用Weights & Biases

```bash
# 启用W&B监控
python lora_main.py train \
    --data-dir /path/to/data \
    --use-wandb \
    --wandb-project "sam_cell_segmentation"
```

### TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir ./lora_experiments/logs
```

## 🛠️ 命令行工具

### 训练相关命令

```bash
# 查看模型信息
python lora_main.py info --model vit_b_lm

# 准备数据集
python lora_main.py prepare-data \
    --data-dir /path/to/data \
    --train-ratio 0.8 \
    --val-ratio 0.1

# 评测LoRA模型
python lora_main.py evaluate \
    --lora-model ./lora_experiments/final_model \
    --eval-data /path/to/test_data
```

### 评测相关命令

```bash
# 预览模式 (不实际运行)
python main.py --dry-run --data-dir /path/to/data

# 指定特定数据
python main.py \
    --data-dir /path/to/data \
    --cell-types MSC \
    --dates 2024-01-01 \
    --magnifications 10x
```

## 🔬 技术特性

### LoRA微调优势
- **参数效率**: 只训练1-5%的参数
- **内存友好**: 显著降低GPU内存需求
- **训练快速**: 大幅减少训练时间
- **易于部署**: 轻量级适配器便于分发

### 多实例掩码处理
- 自动检测多实例掩码
- 智能合并为二进制掩码
- 兼容SAM单掩码输出限制
- 保持训练稳定性

### 评测系统特点
- **自动发现**: 智能扫描数据集目录
- **并行处理**: 支持多GPU并行评测
- **容错机制**: 自动跳过损坏数据
- **进度追踪**: 实时显示处理进度

## ⚡ 性能优化建议

### 训练优化
```bash
# 大批量训练 (需要更多GPU内存)
python lora_main.py train \
    --batch-size 16 \
    --learning-rate 2e-4

# 小内存训练
python lora_main.py train \
    --batch-size 4 \
    --rank 4 \
    --alpha 8
```

### 评测优化
```bash
# 限制处理数量 (快速评测)
python main.py \
    --data-dir /path/to/data \
    --batch-size 10

# 跳过可视化 (节省时间)
python main.py \
    --data-dir /path/to/data \
    --no-visualizations
```

## 🐛 常见问题

### Q: 训练时出现内存不足错误
**A**: 减少批大小或使用更小的LoRA rank
```bash
python lora_main.py train --batch-size 4 --rank 4
```

### Q: 数据集未被识别
**A**: 检查目录结构，确保images和masks目录存在且配对
```bash
python lora_main.py prepare-data --data-dir /path/to/data
```

### Q: 评测结果异常
**A**: 检查掩码格式，确保是标签图或二值图
```bash
# 查看数据集统计
python main.py --dry-run --data-dir /path/to/data
```

## 📝 开发指南

### 添加新的评测指标

```python
# core/metrics.py
class CustomMetrics:
    def calculate_custom_metric(self, gt_mask, pred_mask):
        # 实现自定义指标
        return metric_value
```

### 自定义LoRA配置

```python
# config/lora_config.py
custom_config = LoRATrainingSettings(
    lora=LoRAConfig(
        rank=16,
        target_modules=["custom_module"]
    )
)
```

## 🔍 系统要求检查

```bash
# 检查系统环境
python lora_main.py info

# 验证数据集
python main.py --dry-run --data-dir /path/to/data
```

## 📞 支持与贡献

- **问题报告**: 请在GitHub Issues中报告bug
- **功能建议**: 欢迎提交feature requests
- **贡献代码**: 请遵循代码规范，提交PR前请测试

## 📄 许可证

本项目基于MIT许可证开源，详见LICENSE文件。

## 🙏 致谢

- Meta AI的Segment Anything Model (SAM)
- MicroSAM项目的优秀工作
- 开源社区的宝贵贡献

---

**快速开始**: `python main.py --preset fast --data-dir /your/data/path`

**获取帮助**: `python main.py --help` 或 `python lora_main.py --help`