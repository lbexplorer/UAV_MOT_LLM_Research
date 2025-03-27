"""
绘图工具类
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path
import seaborn as sns
from ..utils.color_utils import ColorUtils

class PlotUtils:
    """绘图工具类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化绘图工具
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'output/plots'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_metrics(self,
                    metrics: Dict[str, List[float]],
                    title: str = 'Tracking Metrics Over Time',
                    output_path: str = None) -> None:
        """
        绘制评估指标
        
        Args:
            metrics: 评估指标字典
            title: 图表标题
            output_path: 输出路径
        """
        plt.figure(figsize=(12, 6))
        
        for metric_name, values in metrics.items():
            plt.plot(values, label=metric_name, linewidth=2)
            
        plt.title(title)
        plt.xlabel('Frame')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        if output_path is None:
            output_path = self.output_dir / 'metrics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_trajectories(self,
                         tracks_history: Dict[int, List[Tuple[float, float]]],
                         image_shape: Tuple[int, int],
                         title: str = 'Target Trajectories',
                         output_path: str = None) -> None:
        """
        绘制目标轨迹
        
        Args:
            tracks_history: 轨迹历史
            image_shape: 图像尺寸
            title: 图表标题
            output_path: 输出路径
        """
        plt.figure(figsize=(12, 8))
        plt.imshow(np.zeros(image_shape), cmap='gray')
        
        color_map = {}
        for track_id, positions in tracks_history.items():
            if len(positions) > 1:
                x_coords = [p[0] for p in positions]
                y_coords = [p[1] for p in positions]
                color = ColorUtils.get_color_by_id(track_id, color_map)
                plt.plot(x_coords, y_coords, color=color, label=f'Track {track_id}', linewidth=2)
                
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        if output_path is None:
            output_path = self.output_dir / 'trajectories.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_heatmap(self,
                    data: np.ndarray,
                    title: str = 'Density Heatmap',
                    output_path: str = None) -> None:
        """
        绘制热力图
        
        Args:
            data: 热力图数据
            title: 图表标题
            output_path: 输出路径
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(data, cmap='jet', cbar=True)
        plt.title(title)
        
        if output_path is None:
            output_path = self.output_dir / 'heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_confidence_distribution(self,
                                   confidences: List[float],
                                   title: str = 'Confidence Distribution',
                                   output_path: str = None) -> None:
        """
        绘制置信度分布
        
        Args:
            confidences: 置信度列表
            title: 图表标题
            output_path: 输出路径
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(confidences, bins=20)
        plt.title(title)
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        
        if output_path is None:
            output_path = self.output_dir / 'confidence_dist.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_tracking_stats(self,
                           stats: Dict[str, Any],
                           title: str = 'Tracking Statistics',
                           output_path: str = None) -> None:
        """
        绘制跟踪统计信息
        
        Args:
            stats: 统计信息字典
            title: 图表标题
            output_path: 输出路径
        """
        plt.figure(figsize=(12, 6))
        
        # 创建条形图
        categories = list(stats.keys())
        values = list(stats.values())
        
        bars = plt.bar(categories, values)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
            
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if output_path is None:
            output_path = self.output_dir / 'tracking_stats.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close() 