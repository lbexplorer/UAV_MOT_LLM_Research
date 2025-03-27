"""
数据预处理模块
"""
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
import cv2
from pathlib import Path

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """数据预处理类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据预处理器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.image_size = config.get('image_size', (640, 640))
        self.normalize = config.get('normalize', True)
        
    def preprocess_detections(self, 
                            detections: List[Tuple[float, float, float, float]],
                            image_shape: Tuple[int, int]) -> List[Tuple[float, float, float, float]]:
        """
        预处理检测结果
        
        Args:
            detections: 原始检测框列表
            image_shape: 图像尺寸 (height, width)
            
        Returns:
            处理后的检测框列表
        """
        processed_dets = []
        for det in detections:
            # 归一化坐标
            if self.normalize:
                x1, y1, x2, y2 = det
                x1 = x1 / image_shape[1]
                y1 = y1 / image_shape[0]
                x2 = x2 / image_shape[1]
                y2 = y2 / image_shape[0]
                processed_dets.append((x1, y1, x2, y2))
            else:
                processed_dets.append(det)
                
        return processed_dets
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像
        
        Args:
            image: 输入图像
            
        Returns:
            处理后的图像
        """
        # 调整图像大小
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size)
            
        # 归一化
        if self.normalize:
            image = image.astype(np.float32) / 255.0
            
        return image
    
    def augment_data(self, 
                    image: np.ndarray,
                    detections: List[Tuple[float, float, float, float]]) -> Tuple[np.ndarray, List[Tuple[float, float, float, float]]]:
        """
        数据增强
        
        Args:
            image: 输入图像
            detections: 检测框列表
            
        Returns:
            增强后的图像和检测框
        """
        augmented_image = image.copy()
        augmented_dets = detections.copy()
        
        # 随机水平翻转
        if np.random.random() < 0.5:
            augmented_image = cv2.flip(augmented_image, 1)
            augmented_dets = self._flip_boxes(augmented_dets, image.shape[1])
            
        # 随机亮度调整
        if np.random.random() < 0.5:
            augmented_image = self._adjust_brightness(augmented_image)
            
        return augmented_image, augmented_dets
    
    def _flip_boxes(self, 
                   boxes: List[Tuple[float, float, float, float]],
                   image_width: int) -> List[Tuple[float, float, float, float]]:
        """水平翻转检测框"""
        flipped_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            new_x1 = image_width - x2
            new_x2 = image_width - x1
            flipped_boxes.append((new_x1, y1, new_x2, y2))
        return flipped_boxes
    
    def _adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        """调整图像亮度"""
        alpha = np.random.uniform(0.8, 1.2)
        beta = np.random.uniform(-10, 10)
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    def save_preprocessed_data(self,
                             data: Dict[str, Any],
                             output_dir: str) -> None:
        """
        保存预处理后的数据
        
        Args:
            data: 预处理后的数据
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存图像
        for frame_id, frame_data in data.items():
            image_path = output_path / f"frame_{frame_id:06d}.jpg"
            cv2.imwrite(str(image_path), frame_data['image'])
            
        logger.info(f"Preprocessed data saved to {output_dir}") 