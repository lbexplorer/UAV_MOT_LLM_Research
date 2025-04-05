"""
环境变量加载工具
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class EnvLoader:
    """环境变量加载器"""
    
    @staticmethod
    def load_env():
        """
        加载环境变量
        """
        # 获取项目根目录
        root_dir = Path(__file__).parent.parent.parent.parent
        
        # 尝试加载.env文件
        env_path = root_dir / '.env'
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment variables from {env_path}")
        else:
            logger.warning(f"Environment file not found at {env_path}")
            
    @staticmethod
    def get_env_var(key: str, default: str = None) -> str:
        """
        获取环境变量值
        
        Args:
            key: 环境变量名
            default: 默认值
            
        Returns:
            环境变量值
        """
        value = os.getenv(key, default)
        if value is None:
            logger.warning(f"Environment variable {key} not found")
        return value
        
    @staticmethod
    def set_env_var(key: str, value: str):
        """
        设置环境变量值
        
        Args:
            key: 环境变量名
            value: 环境变量值
        """
        os.environ[key] = value
        logger.info(f"Set environment variable {key}") 