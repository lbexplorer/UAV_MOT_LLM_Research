import openai
from typing import Dict, List, Any, Tuple
import json
import logging
from .interface import LLMInterface
from .prompts.tracking_prompts import (
    TRACKING_PROMPT_TEMPLATE,
    SCENE_ANALYSIS_PROMPT_TEMPLATE,
    QUALITY_EVALUATION_PROMPT_TEMPLATE
)

logger = logging.getLogger(__name__)

class OpenAIInterface(LLMInterface):
    """OpenAI API接口实现"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        初始化OpenAI接口
        
        Args:
            api_key: OpenAI API密钥
            model: 使用的模型名称
        """
        self.api_key = api_key
        self.model = model
        self.is_initialized = False
        
    def initialize(self) -> None:
        """初始化OpenAI客户端"""
        openai.api_key = self.api_key
        self.is_initialized = True
        logger.info(f"OpenAI interface initialized with model: {self.model}")
        
    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """
        调用OpenAI API
        
        Args:
            messages: 消息列表，包含system和user消息
            
        Returns:
            API响应的文本内容
        """
        if not self.is_initialized:
            raise RuntimeError("OpenAI interface not initialized. Call initialize() first.")
            
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise
            
    def get_tracking_suggestion(self,
                              frame_context: Dict[str, Any],
                              tracking_history: List[Dict],
                              scene_info: Dict[str, Any]) -> Dict[str, Any]:
        """获取跟踪建议"""
        # 构建提示词
        prompt = TRACKING_PROMPT_TEMPLATE.format(
            frame_id=frame_context['frame_id'],
            detections=frame_context['detections'],
            traditional_result=frame_context['traditional_result'],
            tracking_history=json.dumps(tracking_history[-5:], indent=2),  # 只使用最近5帧历史
            scene_info=json.dumps(scene_info, indent=2)
        )
        
        messages = [
            {"role": "system", "content": "你是一个多目标跟踪专家系统，专门负责分析和优化跟踪结果。"},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self._call_api(messages)
            result = json.loads(response)
            logger.debug(f"Tracking suggestion generated: {result}")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return {
                "id_associations": [],
                "confidence_scores": {},
                "risk_assessment": {"level": "unknown", "details": "Error parsing response"},
                "suggested_corrections": []
            }
            
    def analyze_scene(self,
                     frame_data: Dict[str, Any],
                     scene_history: List[Dict]) -> Dict[str, Any]:
        """场景分析"""
        prompt = SCENE_ANALYSIS_PROMPT_TEMPLATE.format(
            frame_data=json.dumps(frame_data, indent=2),
            scene_history=json.dumps(scene_history[-5:], indent=2)
        )
        
        messages = [
            {"role": "system", "content": "你是一个场景分析专家，负责理解和分析多目标跟踪场景。"},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self._call_api(messages)
            result = json.loads(response)
            logger.debug(f"Scene analysis completed: {result}")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing scene analysis response: {str(e)}")
            return {
                "scene_type": "unknown",
                "complexity": 0.0,
                "risk_areas": [],
                "interaction_patterns": []
            }
            
    def evaluate_tracking_quality(self,
                                tracking_result: Dict[str, Any],
                                frame_context: Dict[str, Any]) -> Dict[str, float]:
        """评估跟踪质量"""
        prompt = QUALITY_EVALUATION_PROMPT_TEMPLATE.format(
            tracking_result=json.dumps(tracking_result, indent=2),
            frame_context=json.dumps(frame_context, indent=2)
        )
        
        messages = [
            {"role": "system", "content": "你是一个跟踪质量评估专家，负责评估多目标跟踪的性能。"},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self._call_api(messages)
            result = json.loads(response)
            logger.debug(f"Quality evaluation completed: {result}")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing quality evaluation response: {str(e)}")
            return {
                "overall_quality": 0.0,
                "id_stability": 0.0,
                "position_accuracy": 0.0
            } 