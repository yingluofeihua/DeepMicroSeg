"""
可视化生成模块
生成各种评测结果的可视化图表
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
plt.ioff()

import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


class VisualizationGenerator:
    """可视化生成器"""
    
    def __init__(self, output_dir: Path, dpi: int = 300):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
    
    def create_model_performance_comparison(self, summary_df: pd.DataFrame) -> Path:
        """创建模型性能对比图"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            metrics = ['ap50', 'ap75', 'iou_score', 'dice_score', 'hd95', 'processing_time']
            metric_labels = ['AP50', 'AP75', 'IoU Score', 'Dice Score', 'HD95', 'Processing Time (s)']
            
            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                if i >= len(axes) or metric not in summary_df.columns:
                    axes[i].set_visible(False)
                    continue
                    
                ax = axes[i]
                
                # 特殊处理HD95的无穷值
                if metric == 'hd95':
                    plot_data = summary_df[summary_df[metric] != np.inf]
                    if len(plot_data) == 0:
                        ax.text(0.5, 0.5, 'All HD95 values are infinite', 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(label)
                        continue
                else:
                    plot_data = summary_df
                
                # 按模型分组绘制箱线图
                if 'model' in plot_data.columns:
                    models = plot_data['model'].unique()
                    box_data = [plot_data[plot_data['model'] == model][metric].values 
                               for model in models]
                    
                    bp = ax.boxplot(box_data, labels=models, patch_artist=True)
                    
                    # 设置颜色
                    colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                    
                    ax.set_title(label, fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    
                    # 添加均值标记
                    for j, data in enumerate(box_data):
                        if len(data) > 0:
                            mean_val = np.mean(data)
                            ax.text(j+1, mean_val, f'{mean_val:.3f}', 
                                   ha='center', va='bottom', fontweight='bold', 
                                   fontsize=8, bbox=dict(boxstyle="round,pad=0.3", 
                                                        facecolor="white", alpha=0.8))
            
            plt.tight_layout()
            output_path = self.output_dir / "model_performance_comparison.png"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            print(f"创建模型性能对比图失败: {e}")
            plt.close()
            return None
    
    def create_performance_heatmap(self, summary_df: pd.DataFrame) -> Path:
        """创建性能热图"""
        try:
            if 'cell_type' not in summary_df.columns or 'model' not in summary_df.columns:
                print("数据中缺少必要的列，跳过热图生成")
                return None
                
            fig, axes = plt.subplots(2, 3, figsize=(20, 14))
            axes = axes.flatten()
            
            key_metrics = ['ap50', 'ap75', 'iou_score', 'dice_score', 'processing_time']
            
            for i, metric in enumerate(key_metrics):
                if i >= len(axes) or metric not in summary_df.columns:
                    if i < len(axes):
                        axes[i].set_visible(False)
                    continue
                
                ax = axes[i]
                
                try:
                    # 创建透视表
                    pivot_data = summary_df.pivot_table(
                        values=metric,
                        index='model',
                        columns='cell_type',
                        aggfunc='mean'
                    )
                    
                    # 特殊处理HD95
                    if metric == 'hd95':
                        pivot_data = pivot_data.replace([np.inf, -np.inf], np.nan)
                    
                    # 绘制热图
                    sns.heatmap(pivot_data, annot=True, fmt='.3f', ax=ax, 
                               cmap='RdYlBu_r', cbar_kws={'label': 'Score'},
                               square=True, linewidths=0.5)
                    ax.set_title(f'{metric.upper()} by Model and Cell Type', 
                                fontsize=12, fontweight='bold')
                    ax.set_xlabel('Cell Type', fontsize=10)
                    ax.set_ylabel('Model', fontsize=10)
                    
                except Exception as e:
                    print(f"创建 {metric} 热图失败: {e}")
                    ax.text(0.5, 0.5, f'Error: {metric}', ha='center', va='center',
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{metric.upper()} (Error)', fontsize=12)
            
            # 隐藏多余的子图
            for i in range(len(key_metrics), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            output_path = self.output_dir / "performance_by_cell_type.png"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            print(f"创建性能热图失败: {e}")
            plt.close()
            return None
    
    def create_metrics_correlation(self, summary_df: pd.DataFrame) -> Path:
        """创建指标相关性分析图"""
        try:
            metrics_for_corr = ['ap50', 'ap75', 'iou_score', 'dice_score']
            available_metrics = [m for m in metrics_for_corr if m in summary_df.columns]
            
            if len(available_metrics) < 2:
                print("可用指标少于2个，跳过相关性分析")
                return None
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 计算相关性矩阵
            corr_matrix = summary_df[available_metrics].corr()
            
            # 创建遮罩（上三角）
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # 绘制热图
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', ax=ax, 
                       cmap='coolwarm', center=0, square=True,
                       cbar_kws={'label': 'Correlation Coefficient'},
                       linewidths=0.5)
            
            ax.set_title('Correlation Matrix of Performance Metrics', 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            output_path = self.output_dir / "metrics_correlation.png"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            print(f"创建相关性分析图失败: {e}")
            plt.close()
            return None
    
    def create_performance_trends(self, summary_df: pd.DataFrame) -> Path:
        """创建性能趋势图"""
        try:
            if 'date' not in summary_df.columns:
                print("数据中缺少日期信息，跳过趋势图生成")
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            metrics = ['ap50', 'ap75', 'iou_score', 'dice_score']
            
            for i, metric in enumerate(metrics):
                if i >= len(axes) or metric not in summary_df.columns:
                    if i < len(axes):
                        axes[i].set_visible(False)
                    continue
                
                ax = axes[i]
                
                # 按日期和模型分组
                if 'model' in summary_df.columns:
                    for model in summary_df['model'].unique():
                        model_data = summary_df[summary_df['model'] == model]
                        trend_data = model_data.groupby('date')[metric].mean()
                        
                        ax.plot(trend_data.index, trend_data.values, 
                               marker='o', linewidth=2, label=model)
                
                ax.set_title(f'{metric.upper()} Trends Over Time', 
                            fontsize=12, fontweight='bold')
                ax.set_xlabel('Date', fontsize=10)
                ax.set_ylabel(metric.upper(), fontsize=10)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # 旋转x轴标签
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            output_path = self.output_dir / "performance_trends.png"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            print(f"创建趋势图失败: {e}")
            plt.close()
            return None
    
    def create_distribution_plots(self, detailed_df: pd.DataFrame) -> Path:
        """创建指标分布图"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            metrics = ['ap50', 'ap75', 'iou_score', 'dice_score', 'hd95', 'processing_time']
            
            for i, metric in enumerate(metrics):
                if i >= len(axes) or metric not in detailed_df.columns:
                    if i < len(axes):
                        axes[i].set_visible(False)
                    continue
                
                ax = axes[i]
                
                # 过滤有效数据
                if metric == 'hd95':
                    data = detailed_df[detailed_df[metric] != np.inf][metric]
                else:
                    data = detailed_df[metric].dropna()
                
                if len(data) == 0:
                    ax.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                           transform=ax.transAxes)
                    ax.set_title(f'{metric.upper()} Distribution')
                    continue
                
                # 绘制直方图和密度曲线
                ax.hist(data, bins=30, alpha=0.7, density=True, 
                       color='skyblue', edgecolor='black')
                
                # 添加统计信息
                mean_val = data.mean()
                std_val = data.std()
                ax.axvline(mean_val, color='red', linestyle='--', 
                          label=f'Mean: {mean_val:.3f}')
                ax.axvline(mean_val + std_val, color='orange', linestyle=':', 
                          label=f'+1σ: {mean_val + std_val:.3f}')
                ax.axvline(mean_val - std_val, color='orange', linestyle=':', 
                          label=f'-1σ: {mean_val - std_val:.3f}')
                
                ax.set_title(f'{metric.upper()} Distribution', 
                            fontsize=12, fontweight='bold')
                ax.set_xlabel(metric.upper(), fontsize=10)
                ax.set_ylabel('Density', fontsize=10)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_path = self.output_dir / "metrics_distribution.png"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            print(f"创建分布图失败: {e}")
            plt.close()
            return None
    
    def create_model_comparison_radar(self, summary_df: pd.DataFrame) -> Path:
        """创建模型对比雷达图"""
        try:
            if 'model' not in summary_df.columns:
                print("数据中缺少模型信息，跳过雷达图生成")
                return None
            
            # 选择关键指标
            metrics = ['ap50', 'ap75', 'iou_score', 'dice_score']
            available_metrics = [m for m in metrics if m in summary_df.columns]
            
            if len(available_metrics) < 3:
                print("可用指标少于3个，跳过雷达图生成")
                return None
            
            # 计算各模型的平均指标
            model_means = summary_df.groupby('model')[available_metrics].mean()
            
            # 设置雷达图
            angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))  # 闭合图形
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(model_means)))
            
            for i, (model, values) in enumerate(model_means.iterrows()):
                # 添加第一个值到末尾以闭合图形
                values_closed = np.concatenate((values.values, [values.values[0]]))
                
                ax.plot(angles, values_closed, 'o-', linewidth=2, 
                       label=model, color=colors[i])
                ax.fill(angles, values_closed, alpha=0.25, color=colors[i])
            
            # 设置标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([m.upper() for m in available_metrics])
            ax.set_ylim(0, 1)
            
            # 添加网格和标签
            ax.grid(True)
            ax.set_title('Model Performance Comparison (Radar Chart)', 
                        size=14, fontweight='bold', y=1.08)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            
            plt.tight_layout()
            output_path = self.output_dir / "model_comparison_radar.png"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            print(f"创建雷达图失败: {e}")
            plt.close()
            return None
    
    def generate_all_visualizations(self, summary_df: pd.DataFrame, 
                                   detailed_df: Optional[pd.DataFrame] = None) -> List[Path]:
        """生成所有可视化图表"""
        generated_files = []
        
        print("正在生成可视化图表...")
        
        # 模型性能对比
        file_path = self.create_model_performance_comparison(summary_df)
        if file_path:
            generated_files.append(file_path)
            print(f"  ✓ 模型性能对比图: {file_path.name}")
        
        # 性能热图
        file_path = self.create_performance_heatmap(summary_df)
        if file_path:
            generated_files.append(file_path)
            print(f"  ✓ 性能热图: {file_path.name}")
        
        # 指标相关性
        file_path = self.create_metrics_correlation(summary_df)
        if file_path:
            generated_files.append(file_path)
            print(f"  ✓ 指标相关性图: {file_path.name}")
        
        # 性能趋势
        file_path = self.create_performance_trends(summary_df)
        if file_path:
            generated_files.append(file_path)
            print(f"  ✓ 性能趋势图: {file_path.name}")
        
        # 雷达图
        file_path = self.create_model_comparison_radar(summary_df)
        if file_path:
            generated_files.append(file_path)
            print(f"  ✓ 模型对比雷达图: {file_path.name}")
        
        # 分布图（如果有详细数据）
        if detailed_df is not None and not detailed_df.empty:
            file_path = self.create_distribution_plots(detailed_df)
            if file_path:
                generated_files.append(file_path)
                print(f"  ✓ 指标分布图: {file_path.name}")
        
        print(f"共生成 {len(generated_files)} 个可视化文件")
        return generated_files


def create_simple_bar_chart(data: Dict[str, float], title: str, 
                           output_path: Path, xlabel: str = "", ylabel: str = ""):
    """创建简单的柱状图"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        keys = list(data.keys())
        values = list(data.values())
        
        bars = ax.bar(keys, values, color=plt.cm.Set3(np.arange(len(keys))))
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"创建柱状图失败: {e}")
        plt.close()
        return False