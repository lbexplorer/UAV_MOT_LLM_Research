from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class LLMInterface(ABC):
    """LLM接口基类，定义与大语言模型交互的基本接口"""
    
    @abstractmethod
    def initialize(self) -> None:
        """初始化LLM接口"""
        pass
    
    @abstractmethod
    def get_tracking_suggestion(self, 
                              frame_context: Dict[str, Any],
                              tracking_history: List[Dict],
                              scene_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取跟踪建议
        
        Args:
            frame_context: 当前帧上下文信息，包含：
                - frame_id: 帧ID
                - detections: 当前帧检测结果 [(x1,y1,x2,y2),...]
                - traditional_result: 传统跟踪器结果
            tracking_history: 历史跟踪信息列表
            scene_info: 场景分析信息
            
        Returns:
            Dict包含：
                - id_associations: List[Tuple[int, int]] 建议的ID关联
                - confidence_scores: Dict[int, float] 每个跟踪目标的置信度
                - risk_assessment: Dict 风险评估结果
                - suggested_corrections: List 建议的修正操作
        """
        pass
    
    @abstractmethod
    def analyze_scene(self, 
                     frame_data: Dict[str, Any],
                     scene_history: List[Dict]) -> Dict[str, Any]:
        """
        场景分析
        
        Args:
            frame_data: 当前帧数据
            scene_history: 历史场景信息
            
        Returns:
            Dict包含：
                - scene_type: str 场景类型
                - complexity: float 场景复杂度
                - risk_areas: List 风险区域
                - interaction_patterns: List 目标交互模式
        """
        pass
    
    @abstractmethod
    def evaluate_tracking_quality(self,
                                tracking_result: Dict[str, Any],
                                frame_context: Dict[str, Any]) -> Dict[str, float]:
        """
        评估跟踪质量
        
        Args:
            tracking_result: 跟踪结果
            frame_context: 当前帧上下文
            
        Returns:
            Dict包含：
                - overall_quality: float 整体质量分数
                - id_stability: float ID稳定性分数
                - position_accuracy: float 位置准确度分数
        """
        pass 