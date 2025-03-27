"""
跟踪相关的提示词模板
"""

TRACKING_PROMPT_TEMPLATE = """
请分析当前帧的跟踪情况并提供优化建议：

当前帧信息：
- 帧ID: {frame_id}
- 检测结果: {detections}

传统跟踪器结果：
{traditional_result}

场景分析信息：
{scene_info}

历史跟踪信息（最近5帧）：
{tracking_history}

请提供以下分析和建议：
1. ID关联建议
2. 每个跟踪目标的置信度评估
3. 潜在风险评估
4. 跟踪结果修正建议

请以JSON格式返回，包含以下字段：
{
    "id_associations": [[track_id, detection_id], ...],
    "confidence_scores": {"track_id": confidence_score, ...},
    "risk_assessment": {
        "level": "low/medium/high",
        "details": "风险详情描述"
    },
    "suggested_corrections": [
        {
            "track_id": "目标ID",
            "correction_type": "修正类型",
            "details": "具体修正建议"
        }
    ]
}
"""

SCENE_ANALYSIS_PROMPT_TEMPLATE = """
请分析当前跟踪场景：

当前帧数据：
{frame_data}

历史场景信息（最近5帧）：
{scene_history}

请提供以下分析：
1. 场景类型判断
2. 场景复杂度评估
3. 潜在风险区域识别
4. 目标交互模式分析

请以JSON格式返回，包含以下字段：
{
    "scene_type": "场景类型",
    "complexity": 复杂度分数,
    "risk_areas": [
        {
            "area": [x1, y1, x2, y2],
            "risk_level": "风险等级",
            "reason": "风险原因"
        }
    ],
    "interaction_patterns": [
        {
            "pattern_type": "交互类型",
            "objects_involved": [目标ID列表],
            "description": "交互描述"
        }
    ]
}
"""

QUALITY_EVALUATION_PROMPT_TEMPLATE = """
请评估当前跟踪结果的质量：

跟踪结果：
{tracking_result}

当前帧上下文：
{frame_context}

请评估以下方面：
1. 整体跟踪质量
2. ID稳定性
3. 位置准确度

请以JSON格式返回，包含以下字段：
{
    "overall_quality": 总体质量分数,
    "id_stability": ID稳定性分数,
    "position_accuracy": 位置准确度分数,
    "details": {
        "strengths": ["优势1", "优势2", ...],
        "weaknesses": ["不足1", "不足2", ...],
        "suggestions": ["建议1", "建议2", ...]
    }
}
""" 