"""
颜色工具类
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

class ColorUtils:
    """颜色工具类"""
    
    @staticmethod
    def generate_colors(num_colors: int, colormap: str = 'hsv') -> List[Tuple[int, int, int]]:
        """
        生成不同的颜色
        
        Args:
            num_colors: 需要生成的颜色数量
            colormap: 使用的颜色映射
            
        Returns:
            颜色列表，每个颜色为RGB三元组
        """
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            rgb = plt.cm.get_cmap(colormap)(hue)[:3]
            colors.append(tuple(int(x * 255) for x in rgb))
        return colors
    
    @staticmethod
    def get_color_by_id(track_id: int, color_map: Dict[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
        """
        根据ID获取颜色
        
        Args:
            track_id: 目标ID
            color_map: 颜色映射字典
            
        Returns:
            RGB颜色值
        """
        if track_id not in color_map:
            # 生成新的颜色
            new_color = ColorUtils.generate_colors(1)[0]
            color_map[track_id] = new_color
        return color_map[track_id]
    
    @staticmethod
    def get_heatmap_color(value: float, min_val: float = 0.0, max_val: float = 1.0) -> Tuple[int, int, int]:
        """
        根据值生成热力图颜色
        
        Args:
            value: 输入值
            min_val: 最小值
            max_val: 最大值
            
        Returns:
            RGB颜色值
        """
        # 归一化值
        normalized = (value - min_val) / (max_val - min_val)
        # 使用jet颜色映射
        rgb = plt.cm.jet(normalized)[:3]
        return tuple(int(x * 255) for x in rgb)
    
    @staticmethod
    def get_confidence_color(confidence: float) -> Tuple[int, int, int]:
        """
        根据置信度生成颜色
        
        Args:
            confidence: 置信度值
            
        Returns:
            RGB颜色值
        """
        # 使用从红色到绿色的渐变
        if confidence < 0.5:
            # 红色到黄色的渐变
            ratio = confidence * 2
            return (255, int(255 * ratio), 0)
        else:
            # 黄色到绿色的渐变
            ratio = (confidence - 0.5) * 2
            return (int(255 * (1 - ratio)), 255, 0)
    
    @staticmethod
    def get_status_color(status: str) -> Tuple[int, int, int]:
        """
        根据状态获取颜色
        
        Args:
            status: 状态字符串
            
        Returns:
            RGB颜色值
        """
        status_colors = {
            'active': (0, 255, 0),      # 绿色
            'lost': (255, 0, 0),        # 红色
            'occluded': (255, 165, 0),  # 橙色
            'new': (0, 0, 255),         # 蓝色
            'deleted': (128, 128, 128)  # 灰色
        }
        return status_colors.get(status.lower(), (255, 255, 255))  # 默认白色 