"""
动画工具类
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ..utils.color_utils import ColorUtils

class AnimationUtils:
    """动画工具类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化动画工具
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'output/animations'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.color_map = {}
        
    def create_tracking_animation(self,
                                frames: List[np.ndarray],
                                tracks_history: List[Dict[int, Tuple[float, float, float, float]]],
                                detections_history: List[List[Tuple[float, float, float, float]]] = None,
                                fps: int = 30,
                                output_path: str = None) -> None:
        """
        创建跟踪动画
        
        Args:
            frames: 原始帧图像列表
            tracks_history: 跟踪历史
            detections_history: 检测历史
            fps: 帧率
            output_path: 输出路径
        """
        if output_path is None:
            output_path = self.output_dir / 'tracking_animation.gif'
            
        fig, ax = plt.subplots(figsize=(12, 8))
        img = ax.imshow(frames[0])
        
        def update(frame_idx):
            img.set_array(frames[frame_idx])
            
            # 清除之前的绘制
            for artist in ax.lines + ax.collections:
                artist.remove()
                
            # 绘制当前帧的跟踪结果
            tracks = tracks_history[frame_idx]
            for track_id, bbox in tracks.items():
                x1, y1, x2, y2 = bbox
                color = ColorUtils.get_color_by_id(track_id, self.color_map)
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                   fill=False, color=color, linewidth=2)
                ax.add_patch(rect)
                plt.text(x1, y1-10, f'ID: {track_id}',
                        color=color, fontsize=8)
                
            # 绘制检测结果
            if detections_history is not None:
                detections = detections_history[frame_idx]
                for det in detections:
                    x1, y1, x2, y2 = det
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                       fill=False, color=(0, 255, 0),
                                       linewidth=1, linestyle='--')
                    ax.add_patch(rect)
                    
            return [img]
            
        anim = FuncAnimation(fig, update, frames=len(frames),
                           interval=1000/fps, blit=True)
        anim.save(output_path, writer='pillow')
        plt.close()
        
    def create_trajectory_animation(self,
                                  image_shape: Tuple[int, int],
                                  tracks_history: Dict[int, List[Tuple[float, float]]],
                                  fps: int = 30,
                                  output_path: str = None) -> None:
        """
        创建轨迹动画
        
        Args:
            image_shape: 图像尺寸
            tracks_history: 轨迹历史
            fps: 帧率
            output_path: 输出路径
        """
        if output_path is None:
            output_path = self.output_dir / 'trajectory_animation.gif'
            
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(np.zeros(image_shape), cmap='gray')
        
        # 创建轨迹线
        lines = {}
        for track_id in tracks_history.keys():
            line, = ax.plot([], [], label=f'Track {track_id}',
                          color=ColorUtils.get_color_by_id(track_id, self.color_map),
                          linewidth=2)
            lines[track_id] = line
            
        def update(frame):
            for track_id, positions in tracks_history.items():
                if len(positions) > frame:
                    x_coords = [p[0] for p in positions[:frame+1]]
                    y_coords = [p[1] for p in positions[:frame+1]]
                    lines[track_id].set_data(x_coords, y_coords)
            return list(lines.values())
            
        anim = FuncAnimation(fig, update, frames=max(len(positions)
                                                   for positions in tracks_history.values()),
                           interval=1000/fps, blit=True)
        anim.save(output_path, writer='pillow')
        plt.close()
        
    def create_heatmap_animation(self,
                               heatmaps: List[np.ndarray],
                               fps: int = 30,
                               output_path: str = None) -> None:
        """
        创建热力图动画
        
        Args:
            heatmaps: 热力图列表
            fps: 帧率
            output_path: 输出路径
        """
        if output_path is None:
            output_path = self.output_dir / 'heatmap_animation.gif'
            
        fig, ax = plt.subplots(figsize=(10, 8))
        img = ax.imshow(heatmaps[0], cmap='jet')
        plt.colorbar(img)
        
        def update(frame):
            img.set_array(heatmaps[frame])
            return [img]
            
        anim = FuncAnimation(fig, update, frames=len(heatmaps),
                           interval=1000/fps, blit=True)
        anim.save(output_path, writer='pillow')
        plt.close()
        
    def create_video(self,
                    frames: List[np.ndarray],
                    output_path: str = None,
                    fps: int = 30) -> None:
        """
        创建视频
        
        Args:
            frames: 帧图像列表
            output_path: 输出路径
            fps: 帧率
        """
        if output_path is None:
            output_path = self.output_dir / 'tracking_video.mp4'
            
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        out.release() 