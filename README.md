project/
├── core/
│   ├── lora_trainer.py      # LoRA训练器 🆕
│   ├── lora_model_handler.py # LoRA模型处理 🆕
│   ├── dataset_manager.py   
│   ├── metrics.py          
│   ├── model_handler.py    # 需要扩展
│   └── evaluator.py        # 需要扩展
├── lora/                    # LoRA专用模块 🆕
│   ├── adapters.py         # LoRA适配器
│   ├── training_utils.py   # 训练工具
│   ├── data_loaders.py     # 数据加载器
│   └── callbacks.py        # 训练回调
├── config/
│   ├── lora_config.py      # LoRA配置 🆕
│   ├── settings.py         # 需要扩展
│   └── paths.py           
├── utils/
│   ├── lora_utils.py       # LoRA工具函数 🆕
│   ├── file_utils.py      
│   ├── visualization.py   
│   └── report_generator.py X
├── main.py                 # 需要扩展
└── lora_main.py           # LoRA训练入口 🆕


# 基本使用

```bash
python microsam/main.py --data-dir /LD-FS/home/yunshuchen/micro_sam/patch_0520  --output-dir /LD-FS/home/zhenhuachen/code/github/DeepMicroSeg/data/LDCellData/batch_evaluation_results_onlycal --cache-dir /LD-FS/home/zhenhuachen/.cache/micro_sam
```

# 使用预设配置
```bash
python main.py --preset fast --data-dir /path/to/data
```

# 只评测特定细胞类型
```bash
python main.py --data-dir /path/to/data --cell-types MSC Vero
```

# 预览模式
```bash
python main.py --data-dir /path/to/data --dry-run
```

# result

数据集发现摘要
============================================================
总数据集: 7
细胞类型: VERO, 293T, MSC, RBD
日期: 20250509, 20250506, 20250416, 20250427
放大倍数: 20X, 40X, 10X
总图像数: 11986
总掩码数: 1810
有效配对: 1810

按细胞类型分布:
| 细胞类型 | 数据集 | 图像数 | 有效对 |
| -------- | ------ | ------ | ------ |
| 293T     | 2      | 848    | 368    |
| MSC      | 3      | 658    | 262    |
| RBD      | 1      | 750    | 690    |
| VERO     | 1      | 9730   | 490    |




