import pytest
from unittest.mock import Mock, patch
import sys
import os
# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llm.openai_interface import OpenAIInterface


class TestOpenAIInterface:
    def setup_method(self):
        """每个测试方法执行前的设置"""
        print("\n设置测试环境")
        
    def teardown_method(self):
        """每个测试方法执行后的清理"""
        print("清理测试环境")

    @pytest.fixture
    def mock_openai_client(self):
        """创建模拟的OpenAI客户端"""
        print("创建模拟客户端")
        with patch('llm.openai_interface.OpenAI') as mock_client:
            yield mock_client

    @pytest.fixture
    def interface(self):
        """创建OpenAIInterface实例"""
        print("创建接口实例")
        return OpenAIInterface(api_key="test_key", model="gpt-3.5-turbo")

    def test_initialization(self, interface, mock_openai_client):
        """测试初始化功能"""
        print("执行初始化测试")
        interface.initialize()
        mock_openai_client.assert_called_once_with(api_key="test_key")
        assert True

    def test_get_completion(self, interface, mock_openai_client):
        """测试获取LLM响应功能"""
        print("执行获取响应测试")
        # 设置模拟响应
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="测试响应"))]
        mock_openai_client.return_value.chat.completions.create.return_value = mock_response

        interface.initialize()
        response = interface.get_completion("测试提示词")
        
        assert response == "测试响应"

    def test_analyze_scene(self, interface, mock_openai_client):
        """测试场景分析功能"""
        print("执行场景分析测试")
        # 准备测试数据
        frame_data = {
            "frame_id": 1,
            "detections": [
                {"id": 1, "bbox": [100, 100, 150, 150]},
                {"id": 2, "bbox": [200, 200, 250, 250]}
            ],
            "traditional_result": {
                "tracks": [
                    {"track_id": 1, "bbox": [100, 100, 150, 150]},
                ]
            }
        }
        
        scene_history = [
            {
                "frame_id": 0,
                "tracks": [{"track_id": 1, "bbox": [90, 90, 140, 140]}]
            }
        ]

        # 设置模拟响应
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="""
        {
            "scene_type": "正常",
            "complexity": 0.5,
            "risk_areas": ["区域1"],
            "interaction_patterns": ["pattern1"]
        }
        """))]
        mock_openai_client.return_value.chat.completions.create.return_value = mock_response

        interface.initialize()
        result = interface.analyze_scene(frame_data, scene_history)
        
        assert "scene_type" in result
        assert "complexity" in result

if __name__ == "__main__":
    pytest.main(["-v"])