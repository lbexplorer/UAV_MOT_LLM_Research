"""
LLM响应解析工具类
"""
from typing import Dict, List, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)

class ResponseParser:
    """解析和验证LLM的响应"""
    
    @staticmethod
    def parse_tracking_suggestion(response: str) -> Dict[str, Any]:
        """
        解析跟踪建议响应
        
        Args:
            response: LLM的原始响应文本
            
        Returns:
            解析后的跟踪建议
        """
        try:
            result = json.loads(response)
            
            # 验证必要字段
            required_fields = [
                "id_associations",
                "confidence_scores",
                "risk_assessment",
                "suggested_corrections"
            ]
            
            for field in required_fields:
                if field not in result:
                    logger.warning(f"Missing required field in tracking suggestion: {field}")
                    result[field] = ResponseParser._get_default_value(field)
            
            # 验证数据类型
            if not isinstance(result["id_associations"], list):
                logger.warning("Invalid id_associations type")
                result["id_associations"] = []
                
            if not isinstance(result["confidence_scores"], dict):
                logger.warning("Invalid confidence_scores type")
                result["confidence_scores"] = {}
                
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tracking suggestion: {str(e)}")
            return ResponseParser._get_default_tracking_suggestion()
    
    @staticmethod
    def parse_scene_analysis(response: str) -> Dict[str, Any]:
        """
        解析场景分析响应
        
        Args:
            response: LLM的原始响应文本
            
        Returns:
            解析后的场景分析结果
        """
        try:
            result = json.loads(response)
            
            # 验证必要字段
            required_fields = [
                "scene_type",
                "complexity",
                "risk_areas",
                "interaction_patterns"
            ]
            
            for field in required_fields:
                if field not in result:
                    logger.warning(f"Missing required field in scene analysis: {field}")
                    result[field] = ResponseParser._get_default_value(field)
            
            # 验证数据类型和范围
            if not isinstance(result["complexity"], (int, float)):
                logger.warning("Invalid complexity type")
                result["complexity"] = 0.0
            else:
                result["complexity"] = max(0.0, min(1.0, float(result["complexity"])))
                
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse scene analysis: {str(e)}")
            return ResponseParser._get_default_scene_analysis()
    
    @staticmethod
    def parse_quality_evaluation(response: str) -> Dict[str, float]:
        """
        解析质量评估响应
        
        Args:
            response: LLM的原始响应文本
            
        Returns:
            解析后的质量评估结果
        """
        try:
            result = json.loads(response)
            
            # 验证必要字段
            required_fields = [
                "overall_quality",
                "id_stability",
                "position_accuracy"
            ]
            
            for field in required_fields:
                if field not in result:
                    logger.warning(f"Missing required field in quality evaluation: {field}")
                    result[field] = 0.0
                    
                # 确保分数在[0,1]范围内
                if not isinstance(result[field], (int, float)):
                    logger.warning(f"Invalid {field} type")
                    result[field] = 0.0
                else:
                    result[field] = max(0.0, min(1.0, float(result[field])))
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse quality evaluation: {str(e)}")
            return ResponseParser._get_default_quality_evaluation()
    
    @staticmethod
    def _get_default_value(field_name: str) -> Any:
        """获取字段的默认值"""
        defaults = {
            "id_associations": [],
            "confidence_scores": {},
            "risk_assessment": {
                "level": "unknown",
                "details": "Failed to parse response"
            },
            "suggested_corrections": [],
            "scene_type": "unknown",
            "complexity": 0.0,
            "risk_areas": [],
            "interaction_patterns": []
        }
        return defaults.get(field_name, None)
    
    @staticmethod
    def _get_default_tracking_suggestion() -> Dict[str, Any]:
        """获取默认的跟踪建议"""
        return {
            "id_associations": [],
            "confidence_scores": {},
            "risk_assessment": {
                "level": "unknown",
                "details": "Failed to parse response"
            },
            "suggested_corrections": []
        }
    
    @staticmethod
    def _get_default_scene_analysis() -> Dict[str, Any]:
        """获取默认的场景分析结果"""
        return {
            "scene_type": "unknown",
            "complexity": 0.0,
            "risk_areas": [],
            "interaction_patterns": []
        }
    
    @staticmethod
    def _get_default_quality_evaluation() -> Dict[str, float]:
        """获取默认的质量评估结果"""
        return {
            "overall_quality": 0.0,
            "id_stability": 0.0,
            "position_accuracy": 0.0
        } 