from typing import Dict, Any, List
import logging
from openai import OpenAI

from llm.utils.response_parser import ResponseParser
from .interface import LLMInterface
from .utils.context_builder import ContextBuilder

logger = logging.getLogger(__name__)

class OpenAIInterface(LLMInterface):
    """OpenAI接口实现"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        初始化OpenAI接口
        
        Args:
            api_key: OpenAI API密钥
            model: 使用的模型名称
        """
        self.api_key = api_key
        self.model = model
        self.client = None
        
    def initialize(self) -> None:
        """初始化OpenAI客户端"""
        try:
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"OpenAI interface initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise
            
    def get_completion(self, prompt: str) -> str:
        """
        获取LLM响应
        
        Args:
            prompt: 输入提示词
            
        Returns:
            LLM的响应文本
        """
        if not self.client:
            raise RuntimeError("OpenAI interface not initialized. Call initialize() first.")
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个多目标跟踪专家系统，专门负责分析和优化跟踪结果。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise
            
    def analyze_scene(self, frame_data: Dict[str, Any], scene_history: List[Dict]) -> Dict[str, Any]:
        """
        场景分析
        
        Args:
            frame_data: 当前帧数据
            scene_history: 历史场景信息
            
        Returns:
            场景分析结果
        """
        try:
            # 使用ContextBuilder构建场景上下文
            current_frame = ContextBuilder.build_frame_context(
                frame_id=frame_data.get("frame_id"),
                detections=frame_data.get("detections", []),
                traditional_result=frame_data.get("traditional_result", {})
            )
            
            scene_context = ContextBuilder.build_scene_context(
                current_frame=current_frame,
                history=scene_history
            )
            
            # 构建提示词
            prompt = f"""
            请分析以下场景数据并提供详细分析：
            
            当前场景信息：
            - 检测目标数量：{scene_context['current_state']['statistics']['num_detections']}
            - 跟踪目标数量：{scene_context['current_state']['statistics']['num_tracks']}
            - 运动模式：{scene_context['motion_patterns']}
            - 密度分布：{scene_context['density_map']}
            - 交互区域：{scene_context['interaction_zones']}
            
            请提供以下分析结果（JSON格式）：
            1. scene_type: 场景类型（拥挤、稀疏、正常等）
            2. complexity: 场景复杂度评分（0-1）
            3. risk_areas: 风险区域列表
            4. interaction_patterns: 目标交互模式列表
            """
            
            response = self.get_completion(prompt)
            return ResponseParser.parse_scene_analysis(response)
            
        except Exception as e:
            logger.error(f"Scene analysis failed: {str(e)}")
            return ResponseParser._get_default_scene_analysis()
            
    def evaluate_tracking_quality(self, tracking_result: Dict[str, Any], 
                                frame_context: Dict[str, Any]) -> Dict[str, float]:
        """
        评估跟踪质量
        
        Args:
            tracking_result: 跟踪结果
            frame_context: 当前帧上下文
            
        Returns:
            质量评估结果
        """
        try:
            # 构建提示词
            prompt = f"""
            请评估以下跟踪结果的质量：
            
            跟踪结果：
            - 跟踪目标数量：{len(tracking_result.get('tracks', []))}
            - 目标详情：{tracking_result.get('tracks', [])}
            
            当前帧信息：
            - 检测目标数量：{len(frame_context.get('detections', []))}
            - 场景密度：{frame_context.get('density', 0)}
            
            请提供以下评分（JSON格式，分数范围0-1）：
            1. overall_quality: 整体跟踪质量
            2. id_stability: ID稳定性评分
            3. position_accuracy: 位置准确度评分
            """
            
            response = self.get_completion(prompt)
            return ResponseParser.parse_quality_evaluation(response)
            
        except Exception as e:
            logger.error(f"Quality evaluation failed: {str(e)}")
            return ResponseParser._get_default_quality_evaluation() 