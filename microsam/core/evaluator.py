"""
批量评测模块
负责协调数据集管理、模型推理和指标计算
"""

import time
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import threading
import os

from core.dataset_manager import DatasetManager, DatasetInfo
from core.model_handler import ModelHandler, model_context
from core.metrics import ComprehensiveMetrics, MetricsResult
from config.settings import BatchEvaluationSettings, ModelConfig
from config.paths import PathManager
from utils.file_utils import load_image, load_mask
from utils.report_generator import ReportGenerator
import numpy as np

class TimeoutHandler:
    """超时处理器"""
    def __init__(self, timeout_seconds: int = 300):
        self.timeout_seconds = timeout_seconds
        self.timer = None
    
    def timeout_handler(self):
        print(f"处理超时 {self.timeout_seconds} 秒!")
        os._exit(1)
    
    def start_timer(self):
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(self.timeout_seconds, self.timeout_handler)
        self.timer.start()
    
    def stop_timer(self):
        if self.timer:
            self.timer.cancel()


class DatasetProcessor:
    """单个数据集处理器"""
    
    def __init__(self, dataset_info: DatasetInfo, model_handler: ModelHandler,
                 metrics_calculator: ComprehensiveMetrics, 
                 path_manager: PathManager,
                 config: BatchEvaluationSettings):
        self.dataset_info = dataset_info
        self.model_handler = model_handler
        self.metrics_calculator = metrics_calculator
        self.path_manager = path_manager
        self.config = config
        self.timeout_handler = TimeoutHandler(config.evaluation.process_timeout)
        
    def process(self) -> Tuple[str, str]:
        """处理单个数据集"""
        self.timeout_handler.start_timer()
        
        try:
            model_name = self.model_handler.model_config.name
            dataset_id = self.dataset_info.dataset_id
            
            # 检查是否跳过已有结果
            if self.config.evaluation.skip_existing:
                if self.path_manager.check_existing_results(model_name, dataset_id):
                    print(f"跳过已处理的数据集: {dataset_id}")
                    self.timeout_handler.stop_timer()
                    return dataset_id, "skipped"
            
            # 确保模型已加载
            if not self.model_handler.is_loaded:
                if not self.model_handler.load_model():
                    self.timeout_handler.stop_timer()
                    return dataset_id, "model_load_failed"
            
            # 获取图像-掩码对
            # from core.dataset_manager import DatasetManager
            # temp_manager = DatasetManager(str(Path(self.dataset_info.images_dir).parent.parent.parent))
            # image_mask_pairs = temp_manager.get_image_mask_pairs(self.dataset_info)
            images_dir = Path(self.dataset_info.images_dir)
            masks_dir = Path(self.dataset_info.masks_dir)
            image_mask_pairs = self._get_direct_image_mask_pairs(images_dir, masks_dir)
            
            # 限制处理数量
            if self.config.evaluation.batch_size is not None:
                image_mask_pairs = image_mask_pairs[:self.config.evaluation.batch_size]
            
            if not image_mask_pairs:
                print(f"数据集无有效图像对: {dataset_id}")
                self.timeout_handler.stop_timer()
                return dataset_id, "no_valid_pairs"
            
            # 处理每个图像对
            results = []
            total_processing_time = 0.0
            
            for img_path, mask_path in tqdm(image_mask_pairs, 
                                          desc=f"处理 {dataset_id}"):
                try:
                    result = self._process_single_image(img_path, mask_path)
                    if result:
                        results.append(result)
                        total_processing_time += result.get('processing_time', 0)
                    
                    # 重置超时计时器
                    self.timeout_handler.start_timer()
                    
                except Exception as e:
                    print(f"处理图像失败 {img_path}: {e}")
                    continue
            
            # 保存结果
            if results:
                self._save_results(results, total_processing_time)
                print(f"完成 {dataset_id}: 处理了 {len(results)}/{len(image_mask_pairs)} 张图像")
                self.timeout_handler.stop_timer()
                return dataset_id, "completed"
            else:
                print(f"数据集无有效结果: {dataset_id}")
                self.timeout_handler.stop_timer()
                return dataset_id, "no_results"
                
        except Exception as e:
            self.timeout_handler.stop_timer()
            print(f"处理数据集失败 {self.dataset_info.dataset_id}: {e}")
            return self.dataset_info.dataset_id, f"error: {str(e)}"
    
    def _process_single_image(self, img_path: Path, mask_path: Path) -> Optional[Dict]:
        """处理单张图像"""
        try:
            start_time = time.time()
            
            # 加载图像和掩码
            image = load_image(img_path)
            gt_mask = load_mask(mask_path)
            
            if image is None or gt_mask is None:
                return None
            
            # 模型推理
            pred_mask = self.model_handler.predict(image)
            if pred_mask is None:
                return None
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 计算评测指标
            metrics_result = self.metrics_calculator.compute_all_metrics(gt_mask, pred_mask)
            
            # 构建结果字典
            result = metrics_result.to_dict()
            result.update({
                'image_id': img_path.stem,
                'cell_type': self.dataset_info.cell_type,
                'date': self.dataset_info.date,
                'magnification': self.dataset_info.magnification,
                'model': self.model_handler.model_config.name,
                'processing_time': processing_time,
                'image_path': str(img_path),
                'mask_path': str(mask_path)
            })
            
            return result
            
        except Exception as e:
            print(f"单图像处理失败 {img_path}: {e}")
            return None
    
    def _save_results(self, results: List[Dict], total_processing_time: float):
        """保存处理结果"""
        model_name = self.model_handler.model_config.name
        dataset_id = self.dataset_info.dataset_id
        
        # 保存详细结果CSV
        df = pd.DataFrame(results)
        results_file = self.path_manager.get_results_file_path(model_name, dataset_id)
        df.to_csv(results_file, index=False)
        
        # 计算平均指标
        avg_metrics = self._calculate_average_metrics(results, total_processing_time)
        
        # 保存摘要JSON
        summary_file = self.path_manager.get_summary_file_path(model_name, dataset_id)
        with open(summary_file, 'w') as f:
            json.dump(avg_metrics, f, indent=2)
    
    def _calculate_average_metrics(self, results: List[Dict], 
                                 total_processing_time: float) -> Dict:
        """计算平均指标"""
        if not results:
            return {}
        
        numeric_cols = ['ap50', 'ap75', 'iou_score', 'dice_score', 'hd95', 
                       'gt_instances', 'pred_instances', 'processing_time']
        
        avg_metrics = {}
        df = pd.DataFrame(results)
        
        for col in numeric_cols:
            if col in df.columns:
                values = df[col].dropna()
                # 特殊处理HD95的无穷值
                if col == 'hd95':
                    finite_values = values[pd.isna(values) == False]
                    # finite_values = finite_values[pd.isinf(finite_values) == False]
                    finite_values = finite_values[~np.isinf(finite_values)]
                    avg_metrics[col] = float(finite_values.mean()) if len(finite_values) > 0 else float('inf')
                else:
                    avg_metrics[col] = float(values.mean()) if len(values) > 0 else 0.0
        
        # 添加元数据
        avg_metrics.update({
            'dataset_id': self.dataset_info.dataset_id,
            'cell_type': self.dataset_info.cell_type,
            'date': self.dataset_info.date,
            'magnification': self.dataset_info.magnification,
            'model': self.model_handler.model_config.name,
            'processed_images': len(results),
            'total_processing_time': total_processing_time,
            'average_processing_time_per_image': total_processing_time / len(results) if results else 0.0
        })
        
        return avg_metrics
    
    def _get_direct_image_mask_pairs(self, images_dir, masks_dir):
        from config.paths import DatasetPathValidator
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(images_dir.glob(f"*{ext}")))
            image_files.extend(list(images_dir.glob(f"*{ext.upper()}")))
        
        pairs = []
        for img_file in image_files:
            mask_file = DatasetPathValidator.find_matching_mask(img_file, masks_dir)
            if mask_file:
                pairs.append((img_file, mask_file))
        return pairs


class BatchEvaluator:
    """批量评测器 - 主要的评测协调器"""
    
    def __init__(self, config: BatchEvaluationSettings):
        self.config = config
        self.dataset_manager: Optional[DatasetManager] = None
        self.path_manager: Optional[PathManager] = None
        self.metrics_calculator = ComprehensiveMetrics(
            enable_hd95=config.metrics_config.get('calculate_hd95', True)
        )
        self.results_summary = []
    
    def setup(self, base_data_dir: str) -> bool:
        """设置评测环境"""
        try:
            # 验证配置
            if not self.config.validate_config():
                return False
            
            # 初始化数据集管理器
            self.dataset_manager = DatasetManager(base_data_dir)
            if not self.dataset_manager.datasets:
                print("未发现任何有效数据集")
                return False
            
            # 初始化路径管理器
            self.path_manager = PathManager(self.config.output_base_dir)
            
            # 设置环境变量
            os.environ["MICROSAM_CACHEDIR"] = self.config.cache_dir
            
            print(f"评测环境设置完成")
            print(f"  数据集数量: {len(self.dataset_manager.datasets)}")
            print(f"  模型数量: {len(self.config.models)}")
            print(f"  输出目录: {self.config.output_base_dir}")
            
            return True
            
        except Exception as e:
            print(f"评测环境设置失败: {e}")
            return False
    
    def run_evaluation(self, 
                      cell_types: Optional[List[str]] = None,
                      dates: Optional[List[str]] = None,
                      magnifications: Optional[List[str]] = None) -> bool:
        """运行批量评测"""
        if self.dataset_manager is None:
            raise ValueError("请先调用 setup() 方法")
        
        # 过滤数据集
        datasets_to_process = self.dataset_manager.filter_datasets(
            cell_types=cell_types,
            dates=dates,
            magnifications=magnifications
        )
        
        if not datasets_to_process:
            print("没有符合条件的数据集")
            return False
        
        print(f"将处理 {len(datasets_to_process)} 个数据集")
        
        # 创建任务列表
        tasks = []
        for dataset_info in datasets_to_process:
            for model_config in self.config.models:
                tasks.append((dataset_info, model_config))
        
        print(f"总任务数: {len(tasks)} (数据集 × 模型)")
        
        # 执行任务
        results = []
        for dataset_info, model_config in tasks:
            result = self._process_single_task(dataset_info, model_config)
            results.append(result)
            print(f"完成: {result[0]} - {result[1]}")
        
        # 生成摘要报告
        if self.config.evaluation.create_summary_report:
            self._create_comprehensive_summary_report(results)
        
        return True
    
    def _process_single_task(self, dataset_info: DatasetInfo, 
                           model_config: ModelConfig) -> Tuple[str, str]:
        """处理单个任务（数据集-模型组合）"""
        try:
            with model_context(model_config) as model_handler:
                processor = DatasetProcessor(
                    dataset_info=dataset_info,
                    model_handler=model_handler,
                    metrics_calculator=self.metrics_calculator,
                    path_manager=self.path_manager,
                    config=self.config
                )
                return processor.process()
                
        except Exception as e:
            print(f"任务处理失败 {dataset_info.dataset_id}-{model_config.name}: {e}")
            return f"{dataset_info.dataset_id}-{model_config.name}", f"error: {str(e)}"
    
    def _create_comprehensive_summary_report(self, results: List[Tuple[str, str]]):
        """创建综合摘要报告"""
        try:
            report_generator = ReportGenerator(
                output_dir=self.path_manager.get_summary_report_dir(),
                config=self.config
            )
            
            # 收集所有结果数据
            all_summaries = self._collect_all_summaries()
            all_detailed_results = self._collect_all_detailed_results()
            
            # 生成报告
            report_generator.generate_comprehensive_report(
                summaries=all_summaries,
                detailed_results=all_detailed_results,
                task_results=results
            )
            
            print(f"综合报告已生成")
            
        except Exception as e:
            print(f"生成摘要报告失败: {e}")
    
    def _collect_all_summaries(self) -> List[Dict]:
        """收集所有摘要数据"""
        all_summaries = []
        
        for model_config in self.config.models:
            model_dir = self.path_manager.get_model_output_dir(model_config.name)
            
            for dataset_dir in model_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                
                summary_file = dataset_dir / "summary.json"
                if summary_file.exists():
                    try:
                        with open(summary_file, 'r') as f:
                            summary = json.load(f)
                            summary['model'] = model_config.name
                            summary['dataset_id'] = dataset_dir.name
                            all_summaries.append(summary)
                    except Exception as e:
                        print(f"读取摘要文件失败 {summary_file}: {e}")
        
        return all_summaries
    
    def _collect_all_detailed_results(self) -> List[pd.DataFrame]:
        """收集所有详细结果"""
        all_results = []
        
        for model_config in self.config.models:
            model_dir = self.path_manager.get_model_output_dir(model_config.name)
            
            for dataset_dir in model_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                
                results_file = dataset_dir / "results.csv"
                if results_file.exists():
                    try:
                        df = pd.read_csv(results_file)
                        all_results.append(df)
                    except Exception as e:
                        print(f"读取结果文件失败 {results_file}: {e}")
        
        return all_results
    
    def get_evaluation_summary(self) -> Dict:
        """获取评测摘要"""
        if self.dataset_manager is None:
            return {}
        
        return {
            'config': self.config.to_dict(),
            'datasets': self.dataset_manager.get_dataset_statistics(),
            'models': [model.name for model in self.config.models],
            'output_directory': self.config.output_base_dir,
            'cache_directory': self.config.cache_dir
        }

    
    def cleanup(self):
        """清理资源"""
        # 这里可以添加清理逻辑，比如清理临时文件等
        pass