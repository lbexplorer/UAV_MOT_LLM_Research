import pytest
import numpy as np
from src.llm.utils.context_builder import ContextBuilder

class TestContextBuilder:
    @pytest.fixture
    def sample_detections(self):
        """提供示例检测数据"""
        return [
            [100, 100, 150, 150],  # [x1, y1, x2, y2]
            [200, 200, 250, 250],
            [300, 300, 350, 350]
        ]

    @pytest.fixture
    def sample_traditional_result(self):
        """提供示例传统跟踪结果"""
        return {
            "confirmed": [
                {"id": 1, "bbox": [100, 100, 150, 150]},
                {"id": 2, "bbox": [200, 200, 250, 250]}
            ]
        }

    @pytest.fixture
    def sample_history(self):
        """提供示例历史数据"""
        return [
            {
                "frame_id": 1,
                "tracks": [
                    {
                        "id": 1,
                        "center": [125, 125],
                        "bbox": [100, 100, 150, 150]
                    }
                ]
            },
            {
                "frame_id": 2,
                "tracks": [
                    {
                        "id": 1,
                        "center": [130, 130],
                        "bbox": [105, 105, 155, 155]
                    }
                ]
            }
        ]

    def test_build_frame_context(self, sample_detections, sample_traditional_result):
        """测试帧上下文构建"""
        frame_context = ContextBuilder.build_frame_context(
            frame_id=1,
            detections=sample_detections,
            traditional_result=sample_traditional_result
        )

        # 验证基本结构
        assert "frame_id" in frame_context
        assert "detections" in frame_context
        assert "traditional_result" in frame_context
        assert "statistics" in frame_context

        # 验证统计信息
        stats = frame_context["statistics"]
        assert stats["num_detections"] == len(sample_detections)
        assert stats["num_tracks"] == len(sample_traditional_result["confirmed"])
        assert isinstance(stats["frame_area"], float)
        assert stats["frame_area"] > 0

        # 验证检测结果处理
        for det in frame_context["detections"]:
            assert "bbox" in det
            assert "center" in det
            assert len(det["center"]) == 2

    def test_build_scene_context(self, sample_detections, sample_history):
        """测试场景上下文构建"""
        current_frame = ContextBuilder.build_frame_context(
            frame_id=3,
            detections=sample_detections,
            traditional_result={"confirmed": []}
        )

        scene_context = ContextBuilder.build_scene_context(
            current_frame=current_frame,
            history=sample_history
        )

        # 验证基本结构
        assert "current_state" in scene_context
        assert "motion_patterns" in scene_context
        assert "density_map" in scene_context
        assert "interaction_zones" in scene_context

        # 验证密度图
        density_map = scene_context["density_map"]
        assert "density" in density_map
        assert "hotspots" in density_map
        assert isinstance(density_map["density"], float)
        assert density_map["density"] >= 0

    def test_calculate_frame_area(self, sample_detections):
        """测试帧区域计算"""
        area = ContextBuilder._calculate_frame_area(sample_detections)
        assert isinstance(area, float)
        assert area > 0

        # 测试空检测列表
        empty_area = ContextBuilder._calculate_frame_area([])
        assert empty_area == 0.0

    def test_analyze_motion_patterns(self, sample_history):
        """测试运动模式分析"""
        patterns = ContextBuilder._analyze_motion_patterns(sample_history)
        
        # 验证返回的运动模式
        assert isinstance(patterns, list)
        if patterns:  # 如果有检测到的模式
            pattern = patterns[0]
            assert "track_id" in pattern
            assert "speed" in pattern
            assert "direction" in pattern
            assert "pattern_type" in pattern
            assert pattern["pattern_type"] in ["stationary", "slow_moving", 
                                             "normal_moving", "fast_moving"]

    def test_identify_interaction_zones(self):
        """测试交互区域识别"""
        current_frame = {
            "detections": [
                {
                    "center": [100, 100],
                    "bbox": [90, 90, 110, 110]
                },
                {
                    "center": [150, 150],
                    "bbox": [140, 140, 160, 160]
                }
            ]
        }

        zones = ContextBuilder._identify_interaction_zones(current_frame)
        
        # 验证交互区域
        assert isinstance(zones, list)
        if zones:  # 如果检测到交互
            zone = zones[0]
            assert "objects" in zone
            assert "center" in zone
            assert "distance" in zone
            assert "interaction_type" in zone

    def test_classify_motion_pattern(self):
        """测试运动模式分类"""
        # 测试不同速度下的分类
        assert ContextBuilder._classify_motion_pattern(0.5, 0) == "stationary"
        assert ContextBuilder._classify_motion_pattern(3, 0) == "slow_moving"
        assert ContextBuilder._classify_motion_pattern(7, 0) == "normal_moving"
        assert ContextBuilder._classify_motion_pattern(15, 0) == "fast_moving"

    def test_edge_cases(self):
        """测试边界情况"""
        # 测试空检测列表的场景上下文构建
        empty_frame = ContextBuilder.build_frame_context(
            frame_id=1,
            detections=[],
            traditional_result={"confirmed": []}
        )
        
        empty_scene = ContextBuilder.build_scene_context(
            current_frame=empty_frame,
            history=[]
        )

        assert empty_scene["density_map"]["density"] == 0.0
        assert len(empty_scene["motion_patterns"]) == 0
        assert len(empty_scene["interaction_zones"]) == 0