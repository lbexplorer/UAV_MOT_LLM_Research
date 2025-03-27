"""
可视化工具模块
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

logger = logging.getLogger(__name__)

class Visualizer:
    """可视化工具类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化可视化器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.colors = self._generate_colors(100)  # 为100个目标生成不同的颜色
        
    def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """生成不同的颜色"""
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            rgb = plt.cm.hsv(hue)[:3]
            colors.append(tuple(int(x * 255) for x in rgb))
        return colors
    
    def draw_frame(self,
                  image: np.ndarray,
                  tracks: List[Tuple[int, Tuple[float, float, float, float]]],
                  detections: List[Tuple[float, float, float, float]] = None) -> np.ndarray:
        """
        绘制单帧图像
        
        Args:
            image: 输入图像
            tracks: 跟踪结果 [(track_id, bbox), ...]
            detections: 检测结果 [bbox, ...]
            
        Returns:
            绘制后的图像
        """
        vis_image = image.copy()
        
        # 绘制检测框
        if detections is not None:
            for det in detections:
                x1, y1, x2, y2 = map(int, det)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
        # 绘制跟踪框
        for track_id, bbox in tracks:
            x1, y1, x2, y2 = map(int, bbox)
            color = self.colors[track_id % len(self.colors)]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_image, f"ID: {track_id}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return vis_image
    
    def create_animation(self,
                        frames: List[np.ndarray],
                        output_path: str,
                        fps: int = 30) -> None:
        """
        创建动画
        
        Args:
            frames: 帧图像列表
            output_path: 输出路径
            fps: 帧率
        """
        fig, ax = plt.subplots()
        img = ax.imshow(frames[0])
        
        def update(frame):
            img.set_array(frame)
            return [img]
            
        anim = FuncAnimation(fig, update, frames=frames,
                           interval=1000/fps, blit=True)
        anim.save(output_path, writer='pillow')
        plt.close()
        
    def plot_metrics(self,
                    metrics: Dict[str, List[float]],
                    output_path: str) -> None:
        """
        绘制评估指标
        
        Args:
            metrics: 评估指标字典
            output_path: 输出路径
        """
        plt.figure(figsize=(10, 6))
        for metric_name, values in metrics.items():
            plt.plot(values, label=metric_name)
            
        plt.title('Tracking Metrics Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        
    def visualize_tracks(self,
                        tracks_history: Dict[int, List[Tuple[float, float]]],
                        image_shape: Tuple[int, int],
                        output_path: str) -> None:
        """
        可视化目标轨迹
        
        Args:
            tracks_history: 轨迹历史 {track_id: [(x, y), ...]}
            image_shape: 图像尺寸
            output_path: 输出路径
        """
        plt.figure(figsize=(10, 6))
        plt.imshow(np.zeros(image_shape), cmap='gray')
        
        for track_id, positions in tracks_history.items():
            if len(positions) > 1:
                x_coords = [p[0] for p in positions]
                y_coords = [p[1] for p in positions]
                color = self.colors[track_id % len(self.colors)]
                plt.plot(x_coords, y_coords, color=color, label=f'Track {track_id}')
                
        plt.title('Target Trajectories')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        
    def save_visualization(self,
                          image: np.ndarray,
                          output_path: str) -> None:
        """
        保存可视化结果
        
        Args:
            image: 可视化图像
            output_path: 输出路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        logger.info(f"Visualization saved to {output_path}") 