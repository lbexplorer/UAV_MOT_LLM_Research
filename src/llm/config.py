"""
LLM配置文件
"""
from typing import Dict, Any
import os
from pathlib import Path

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent.parent

# API密钥配置
API_KEYS = {
    'openai': os.getenv('OPENAI_API_KEY', 'your-api-key-here')
}

# 模型配置
MODEL_CONFIG = {
    'openai': {
        'default_model': 'gpt-4',
        'models': {
            'gpt-4': {
                'name': 'gpt-4',
                'max_tokens': 4096,
                'temperature': 0.7
            }
        }
    }
}

# 提示词模板配置
PROMPT_CONFIG = {
    'tracking_suggestion': {
        'system_prompt': """你是一个多目标跟踪专家系统，专门负责分析和优化跟踪结果。
        你需要基于当前帧的检测结果、历史跟踪信息和场景分析，提供最优的跟踪建议。""",
        'max_history_frames': 5,
        'temperature': 0.7
    }
}

# 主配置
LLM_CONFIG: Dict[str, Any] = {
    'api_keys': API_KEYS,
    'model_config': MODEL_CONFIG,
    'prompt_config': PROMPT_CONFIG
}

def get_config() -> Dict[str, Any]:
    """
    获取LLM配置
    
    Returns:
        配置字典
    """
    return LLM_CONFIG

def get_api_key(provider: str) -> str:
    """
    获取指定提供商的API密钥
    
    Args:
        provider: API提供商名称
        
    Returns:
        API密钥
    """
    return API_KEYS.get(provider, '')

def get_model_config(provider: str, model_name: str = None) -> Dict[str, Any]:
    """
    获取模型配置
    
    Args:
        provider: API提供商名称
        model_name: 模型名称，如果为None则使用默认模型
        
    Returns:
        模型配置字典
    """
    provider_config = MODEL_CONFIG.get(provider, {})
    if model_name is None:
        model_name = provider_config.get('default_model')
    return provider_config.get('models', {}).get(model_name, {}) 