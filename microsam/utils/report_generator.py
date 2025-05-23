"""
报告生成模块
生成综合评测报告和统计信息
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from utils.visualization import VisualizationGenerator
from config.settings import BatchEvaluationSettings


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, output_dir: Path, config: BatchEvaluationSettings):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.visualization_generator = VisualizationGenerator(output_dir)
    
    def generate_comprehensive_report(self, 
                                    summaries: List[Dict],
                                    detailed_results: List[pd.DataFrame],
                                    task_results: List[Tuple[str, str]]) -> Dict[str, Path]:
        """生成综合评测报告"""
        report_files = {}
        
        print("正在生成综合评测报告...")
        
        # 1. 生成统一的详细结果CSV
        if detailed_results and self.config.evaluation.generate_unified_csv:
            unified_csv = self._generate_unified_detailed_csv(detailed_results)
            if unified_csv:
                report_files['unified_detailed_results'] = unified_csv
                print(f"  ✓ 统一详细结果: {unified_csv.name}")
        
        # 2. 生成最终摘要CSV
        if summaries:
            summary_csv = self._generate_final_summary_csv(summaries)
            if summary_csv:
                report_files['final_summary'] = summary_csv
                print(f"  ✓ 最终摘要: {summary_csv.name}")
        
        # 3. 生成模型对比统计
        if summaries:
            comparison_csv = self._generate_model_comparison_stats(summaries)
            if comparison_csv:
                report_files['model_comparison'] = comparison_csv
                print(f"  ✓ 模型对比统计: {comparison_csv.name}")
        
        # 4. 生成可视化图表
        if summaries:
            summary_df = pd.DataFrame(summaries)
            detailed_df = pd.concat(detailed_results, ignore_index=True) if detailed_results else None
            
            viz_files = self.visualization_generator.generate_all_visualizations(
                summary_df, detailed_df
            )
            
            for i, viz_file in enumerate(viz_files):
                report_files[f'visualization_{i}'] = viz_file
        
        # 5. 生成执行摘要报告
        exec_summary = self._generate_executive_summary(summaries, task_results)
        if exec_summary:
            report_files['executive_summary'] = exec_summary
            print(f"  ✓ 执行摘要: {exec_summary.name}")
        
        # 6. 生成详细分析报告
        if summaries:
            analysis_report = self._generate_detailed_analysis(summaries)
            if analysis_report:
                report_files['detailed_analysis'] = analysis_report
                print(f"  ✓ 详细分析: {analysis_report.name}")
        
        print(f"报告生成完成，共 {len(report_files)} 个文件")
        return report_files
    
    def _generate_unified_detailed_csv(self, detailed_results: List[pd.DataFrame]) -> Optional[Path]:
        """生成统一的详细结果CSV"""
        try:
            if not detailed_results:
                return None
            
            unified_df = pd.concat(detailed_results, ignore_index=True)
            output_path = self.output_dir / "unified_detailed_results.csv"
            unified_df.to_csv(output_path, index=False)
            
            return output_path
            
        except Exception as e:
            print(f"生成统一详细结果失败: {e}")
            return None
    
    def _generate_final_summary_csv(self, summaries: List[Dict]) -> Optional[Path]:
        """生成最终摘要CSV"""
        try:
            summary_df = pd.DataFrame(summaries)
            
            # 按要求的列顺序重新排列
            column_order = [
                'model', 'dataset_id', 'cell_type', 'date', 'magnification',
                'ap50', 'ap75', 'iou_score', 'dice_score', 'hd95',
                'gt_instances', 'pred_instances', 'processing_time',
                'processed_images', 'total_processing_time'
            ]
            
            # 只选择存在的列
            available_columns = [col for col in column_order if col in summary_df.columns]
            summary_df_ordered = summary_df[available_columns]
            
            output_path = self.output_dir / "final_evaluation_summary.csv"
            summary_df_ordered.to_csv(output_path, index=False)
            
            return output_path
            
        except Exception as e:
            print(f"生成最终摘要失败: {e}")
            return None
    
    def _generate_model_comparison_stats(self, summaries: List[Dict]) -> Optional[Path]:
        """生成模型对比统计"""
        try:
            summary_df = pd.DataFrame(summaries)
            
            if 'model' not in summary_df.columns:
                return None
            
            # 计算各模型的平均值和标准差
            numeric_cols = ['ap50', 'ap75', 'iou_score', 'dice_score', 'hd95', 'processing_time']
            available_cols = [col for col in numeric_cols if col in summary_df.columns]
            
            # 特殊处理HD95
            df_for_stats = summary_df.copy()
            if 'hd95' in df_for_stats.columns:
                df_for_stats['hd95'] = df_for_stats['hd95'].replace([np.inf, -np.inf], np.nan)
            
            comparison_stats = df_for_stats.groupby('model')[available_cols].agg(['mean', 'std']).round(4)
            
            output_path = self.output_dir / "model_comparison_statistics.csv"
            comparison_stats.to_csv(output_path)
            
            return output_path
            
        except Exception as e:
            print(f"生成模型对比统计失败: {e}")
            return None
    
    def _generate_executive_summary(self, summaries: List[Dict], 
                                  task_results: List[Tuple[str, str]]) -> Optional[Path]:
        """生成执行摘要报告"""
        try:
            summary_df = pd.DataFrame(summaries)
            
            # 统计任务执行结果
            task_stats = {}
            for task_id, status in task_results:
                task_stats[status] = task_stats.get(status, 0) + 1
            
            # 生成摘要信息
            exec_summary = {
                'generation_time': datetime.now().isoformat(),
                'configuration': self.config.to_dict(),
                'task_execution': {
                    'total_tasks': len(task_results),
                    'task_breakdown': task_stats,
                    'success_rate': task_stats.get('completed', 0) / len(task_results) * 100 if task_results else 0
                }
            }
            
            # 模型性能摘要
            if not summary_df.empty and 'model' in summary_df.columns:
                model_performance = {}
                for model in summary_df['model'].unique():
                    model_data = summary_df[summary_df['model'] == model]
                    
                    performance = {}
                    for metric in ['ap50', 'ap75', 'iou_score', 'dice_score']:
                        if metric in model_data.columns:
                            values = model_data[metric].dropna()
                            if len(values) > 0:
                                performance[metric] = {
                                    'mean': float(values.mean()),
                                    'std': float(values.std()),
                                    'min': float(values.min()),
                                    'max': float(values.max())
                                }
                    
                    model_performance[model] = performance
                
                exec_summary['model_performance'] = model_performance
                
                # 最佳模型推荐
                if 'ap50' in summary_df.columns:
                    best_model = summary_df.groupby('model')['ap50'].mean().idxmax()
                    exec_summary['recommendations'] = {
                        'best_overall_model': best_model,
                        'evaluation_criteria': 'Based on AP50 score'
                    }
            
            # 数据集统计
            if not summary_df.empty:
                dataset_stats = {
                    'total_datasets': summary_df['dataset_id'].nunique() if 'dataset_id' in summary_df.columns else 0,
                    'cell_types': summary_df['cell_type'].unique().tolist() if 'cell_type' in summary_df.columns else [],
                    'total_images_processed': summary_df['processed_images'].sum() if 'processed_images' in summary_df.columns else 0
                }
                exec_summary['dataset_statistics'] = dataset_stats
            
            # 保存执行摘要
            output_path = self.output_dir / "executive_summary.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(exec_summary, f, indent=2, ensure_ascii=False)
            
            return output_path
            
        except Exception as e:
            print(f"生成执行摘要失败: {e}")
            return None
    
    def _generate_detailed_analysis(self, summaries: List[Dict]) -> Optional[Path]:
        """生成详细分析报告"""
        try:
            summary_df = pd.DataFrame(summaries)
            
            if summary_df.empty:
                return None
            
            analysis = {
                'generation_time': datetime.now().isoformat(),
                'data_overview': self._analyze_data_overview(summary_df),
                'performance_analysis': self._analyze_performance(summary_df),
                'efficiency_analysis': self._analyze_efficiency(summary_df),
                'recommendations': self._generate_recommendations(summary_df)
            }
            
            output_path = self.output_dir / "detailed_analysis.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            
            return output_path
            
        except Exception as e:
            print(f"生成详细分析失败: {e}")
            return None
    
    def _analyze_data_overview(self, summary_df: pd.DataFrame) -> Dict:
        """分析数据概览"""
        overview = {}
        
        # 基本统计
        overview['total_evaluations'] = len(summary_df)
        
        if 'model' in summary_df.columns:
            overview['models_evaluated'] = summary_df['model'].nunique()
            overview['model_list'] = summary_df['model'].unique().tolist()
        
        if 'dataset_id' in summary_df.columns:
            overview['datasets_evaluated'] = summary_df['dataset_id'].nunique()
        
        if 'cell_type' in summary_df.columns:
            overview['cell_types'] = summary_df['cell_type'].value_counts().to_dict()
        
        if 'processed_images' in summary_df.columns:
            overview['total_images_processed'] = int(summary_df['processed_images'].sum())
            overview['avg_images_per_dataset'] = float(summary_df['processed_images'].mean())
        
        return overview
    
    def _analyze_performance(self, summary_df: pd.DataFrame) -> Dict:
        """分析性能表现"""
        performance = {}
        
        metrics = ['ap50', 'ap75', 'iou_score', 'dice_score']
        
        for metric in metrics:
            if metric in summary_df.columns:
                values = summary_df[metric].dropna()
                if len(values) > 0:
                    performance[metric] = {
                        'overall_mean': float(values.mean()),
                        'overall_std': float(values.std()),
                        'best_score': float(values.max()),
                        'worst_score': float(values.min())
                    }
                    
                    # 按模型分析
                    if 'model' in summary_df.columns:
                        model_performance = summary_df.groupby('model')[metric].agg(['mean', 'std']).round(4)
                        performance[metric]['by_model'] = model_performance.to_dict('index')
        
        # 整体排名
        if 'model' in summary_df.columns and 'ap50' in summary_df.columns:
            model_ranking = summary_df.groupby('model')['ap50'].mean().sort_values(ascending=False)
            performance['model_ranking_by_ap50'] = model_ranking.to_dict()
        
        return performance
    
    def _analyze_efficiency(self, summary_df: pd.DataFrame) -> Dict:
        """分析效率表现"""
        efficiency = {}
        
        if 'processing_time' in summary_df.columns:
            proc_times = summary_df['processing_time'].dropna()
            if len(proc_times) > 0:
                efficiency['processing_time'] = {
                    'mean_seconds_per_image': float(proc_times.mean()),
                    'std_seconds_per_image': float(proc_times.std()),
                    'fastest_processing': float(proc_times.min()),
                    'slowest_processing': float(proc_times.max())
                }
                
                # 按模型分析处理时间
                if 'model' in summary_df.columns:
                    model_efficiency = summary_df.groupby('model')['processing_time'].agg(['mean', 'std']).round(4)
                    efficiency['processing_time']['by_model'] = model_efficiency.to_dict('index')
        
        # 性能效率比（AP50/处理时间）
        if 'ap50' in summary_df.columns and 'processing_time' in summary_df.columns:
            summary_df_copy = summary_df.copy()
            summary_df_copy['efficiency_ratio'] = summary_df_copy['ap50'] / summary_df_copy['processing_time']
            
            if 'model' in summary_df.columns:
                efficiency_ranking = summary_df_copy.groupby('model')['efficiency_ratio'].mean().sort_values(ascending=False)
                efficiency['efficiency_ranking'] = efficiency_ranking.to_dict()
        
        return efficiency
    
    def _generate_recommendations(self, summary_df: pd.DataFrame) -> Dict:
        """生成推荐建议"""
        recommendations = {}
        
        if 'model' in summary_df.columns:
            # 性能推荐
            if 'ap50' in summary_df.columns:
                best_ap50_model = summary_df.groupby('model')['ap50'].mean().idxmax()
                recommendations['best_accuracy_model'] = {
                    'model': best_ap50_model,
                    'reason': 'Highest average AP50 score'
                }
            
            # 效率推荐
            if 'processing_time' in summary_df.columns:
                fastest_model = summary_df.groupby('model')['processing_time'].mean().idxmin()
                recommendations['fastest_model'] = {
                    'model': fastest_model,
                    'reason': 'Lowest average processing time'
                }
            
            # 平衡推荐
            if 'ap50' in summary_df.columns and 'processing_time' in summary_df.columns:
                summary_df_copy = summary_df.copy()
                summary_df_copy['efficiency_score'] = summary_df_copy['ap50'] / summary_df_copy['processing_time']
                balanced_model = summary_df_copy.groupby('model')['efficiency_score'].mean().idxmax()
                recommendations['balanced_model'] = {
                    'model': balanced_model,
                    'reason': 'Best balance of accuracy and speed'
                }
        
        # 数据质量建议
        data_quality_issues = []
        
        if 'hd95' in summary_df.columns:
            inf_count = summary_df['hd95'].isin([np.inf, -np.inf]).sum()
            if inf_count > 0:
                data_quality_issues.append(f"{inf_count} evaluations had infinite HD95 values")
        
        if 'processed_images' in summary_df.columns and 'dataset_id' in summary_df.columns:
            low_image_datasets = summary_df[summary_df['processed_images'] < 10]['dataset_id'].tolist()
            if low_image_datasets:
                data_quality_issues.append(f"Datasets with <10 images: {low_image_datasets}")
        
        if data_quality_issues:
            recommendations['data_quality_notes'] = data_quality_issues
        
        return recommendations
    
    def generate_markdown_report(self, summaries: List[Dict]) -> Optional[Path]:
        """生成Markdown格式的报告"""
        try:
            summary_df = pd.DataFrame(summaries)
            
            if summary_df.empty:
                return None
            
            # 创建Markdown内容
            markdown_content = self._create_markdown_content(summary_df)
            
            output_path = self.output_dir / "evaluation_report.md"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            return output_path
            
        except Exception as e:
            print(f"生成Markdown报告失败: {e}")
            return None
    
    def _create_markdown_content(self, summary_df: pd.DataFrame) -> str:
        """创建Markdown报告内容"""
        content = []
        
        # 标题和概览
        content.append("# 细胞分割模型评测报告\n")
        content.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        content.append(f"**评测数量**: {len(summary_df)} 次评测\n")
        
        if 'model' in summary_df.columns:
            models = summary_df['model'].unique().tolist()
            content.append(f"**评测模型**: {', '.join(models)}\n")
        
        if 'dataset_id' in summary_df.columns:
            datasets = summary_df['dataset_id'].nunique()
            content.append(f"**数据集数量**: {datasets}\n")
        
        content.append("\n---\n")
        
        # 性能摘要
        content.append("## 性能摘要\n")
        
        if 'model' in summary_df.columns:
            for metric in ['ap50', 'ap75', 'iou_score', 'dice_score']:
                if metric in summary_df.columns:
                    content.append(f"### {metric.upper()}\n")
                    
                    model_stats = summary_df.groupby('model')[metric].agg(['mean', 'std']).round(4)
                    
                    content.append("| 模型 | 平均值 | 标准差 |\n")
                    content.append("|------|--------|--------|\n")
                    
                    for model, stats in model_stats.iterrows():
                        content.append(f"| {model} | {stats['mean']:.4f} | {stats['std']:.4f} |\n")
                    
                    content.append("\n")
        
        # 效率分析
        if 'processing_time' in summary_df.columns:
            content.append("## 效率分析\n")
            
            if 'model' in summary_df.columns:
                time_stats = summary_df.groupby('model')['processing_time'].agg(['mean', 'std']).round(4)
                
                content.append("| 模型 | 平均处理时间(s) | 标准差 |\n")
                content.append("|------|----------------|--------|\n")
                
                for model, stats in time_stats.iterrows():
                    content.append(f"| {model} | {stats['mean']:.4f} | {stats['std']:.4f} |\n")
                
                content.append("\n")
        
        # 推荐建议
        content.append("## 推荐建议\n")
        
        recommendations = self._generate_recommendations(summary_df)
        
        if 'best_accuracy_model' in recommendations:
            best_acc = recommendations['best_accuracy_model']
            content.append(f"- **最佳精度模型**: {best_acc['model']} ({best_acc['reason']})\n")
        
        if 'fastest_model' in recommendations:
            fastest = recommendations['fastest_model']
            content.append(f"- **最快模型**: {fastest['model']} ({fastest['reason']})\n")
        
        if 'balanced_model' in recommendations:
            balanced = recommendations['balanced_model']
            content.append(f"- **平衡推荐**: {balanced['model']} ({balanced['reason']})\n")
        
        content.append("\n---\n")
        content.append("*本报告由批量评测系统自动生成*")
        
        return "\n".join(content)
    
    def print_summary_statistics(self, summaries: List[Dict]):
        """打印摘要统计信息"""
        if not summaries:
            print("没有可用的摘要数据")
            return
        
        summary_df = pd.DataFrame(summaries)
        
        print("\n" + "="*60)
        print("最终评测统计")
        print("="*60)
        
        if 'model' in summary_df.columns:
            for model in summary_df['model'].unique():
                model_data = summary_df[summary_df['model'] == model]
                print(f"\n{model}:")
                
                for metric in ['ap50', 'ap75', 'iou_score', 'dice_score']:
                    if metric in model_data.columns:
                        values = model_data[metric].dropna()
                        if len(values) > 0:
                            print(f"  {metric.upper()}: {values.mean():.3f} ± {values.std():.3f}")
                
                # HD95特殊处理
                if 'hd95' in model_data.columns:
                    finite_hd95 = model_data['hd95'][np.isfinite(model_data['hd95'])]
                    if len(finite_hd95) > 0:
                        print(f"  HD95: {finite_hd95.mean():.3f} ± {finite_hd95.std():.3f}")
                    else:
                        print(f"  HD95: N/A (all infinite)")
                
                if 'processing_time' in model_data.columns:
                    proc_time = model_data['processing_time'].mean()
                    print(f"  处理时间: {proc_time:.3f}s/image")
        
        print("="*60)


class QuickReportGenerator:
    """快速报告生成器 - 用于生成简单的统计报告"""
    
    @staticmethod
    def generate_quick_summary(results_dir: Path) -> Dict:
        """快速生成摘要统计"""
        summary = {
            'total_models': 0,
            'total_datasets': 0,
            'model_performance': {},
            'generation_time': datetime.now().isoformat()
        }
        
        try:
            # 扫描结果目录
            model_dirs = [d for d in results_dir.iterdir() if d.is_dir() and not d.name.startswith('summary_report')]
            summary['total_models'] = len(model_dirs)
            
            for model_dir in model_dirs:
                model_name = model_dir.name
                dataset_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
                summary['total_datasets'] = max(summary['total_datasets'], len(dataset_dirs))
                
                # 收集该模型的性能数据
                model_metrics = []
                for dataset_dir in dataset_dirs:
                    summary_file = dataset_dir / "summary.json"
                    if summary_file.exists():
                        try:
                            with open(summary_file, 'r') as f:
                                metrics = json.load(f)
                                model_metrics.append(metrics)
                        except:
                            continue
                
                # 计算平均性能
                if model_metrics:
                    avg_metrics = {}
                    for metric in ['ap50', 'ap75', 'iou_score', 'dice_score', 'processing_time']:
                        values = [m.get(metric, 0) for m in model_metrics if m.get(metric) is not None]
                        if values:
                            avg_metrics[metric] = {
                                'mean': np.mean(values),
                                'std': np.std(values),
                                'count': len(values)
                            }
                    
                    summary['model_performance'][model_name] = avg_metrics
            
        except Exception as e:
            print(f"快速摘要生成失败: {e}")
        
        return summary