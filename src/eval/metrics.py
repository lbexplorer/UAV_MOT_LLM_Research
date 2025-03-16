# src/eval/metrics.py
import os
import motmetrics as mm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


class MOTEvaluator:
    """
    多目标跟踪性能评估模块。
    支持 MOTA、MOTP 等指标的计算与可视化。
    """

    def __init__(self):
        """
        初始化评估模块。
        """
        # MOT数据累加器，用于存储每帧的匹配结果
        self.acc = mm.MOTAccumulator(auto_id=True)  # 启用自动帧 ID

        # 存储每帧的性能指标历史数据
        self.metrics_history = {
            'fp': [],  # 每帧的误报数
            'fn': [],  # 每帧的漏检数
            'idsw': []  # 每帧的 ID 切换数
        }

        # 指标计算器，用于生成最终报告
        self.metrics = mm.metrics.create()

    def update(self,
               gt_boxes: List[Tuple],
               pred_boxes: List[Tuple]) -> None:
        """
        更新评估状态。
        :param gt_boxes: ground truth 数据，格式为 [(id, x1, y1, x2, y2), ...]
        :param pred_boxes: 预测结果数据，格式为 [(id, x1, y1, x2, y2), ...]
        """
        # 提取 ground truth 和预测结果的 ID 和边界框
        gt_ids = [int(b[0]) for b in gt_boxes]  # Ground truth ID 列表
        gt_bbox = [b[1:] for b in gt_boxes]  # Ground truth 边界框列表

        pred_ids = [int(b[0]) for b in pred_boxes]  # 预测结果 ID 列表
        pred_bbox = [b[1:] for b in pred_boxes]  # 预测结果边界框列表

        # 计算 IOU 距离矩阵（1 - IOU）
        dist_matrix = mm.distances.iou_matrix(gt_bbox, pred_bbox, max_iou=0.5)

        # 更新 MOT 数据累加器（不传入 frameid）
        self.acc.update(
            gt_ids,  # Ground truth IDs
            pred_ids,  # 预测结果 IDs
            dist_matrix  # 距离矩阵
        )

    def get_summary(self) -> pd.DataFrame:
        """
        获取完整的性能评估报告。
        :return: 包含所有指标的 DataFrame
        """
        # 使用 metrics 实例计算指标
        summary = self.metrics.compute(
            self.acc,
            metrics=mm.metrics.motchallenge_metrics,  # 使用 MOTChallenge 标准指标
            name='Overall'
        )
        return summary

    def visualize_trends(self, output_dir: str = 'results') -> None:
        """
        绘制性能趋势图（MOTA 和 MOTP）。
        :param output_dir: 输出目录
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 获取完整报告
        summary = self.get_summary()

        # 创建画布
        plt.figure(figsize=(12, 6))

        # 提取 MOTA 和 MOTP 值（假设数据量较小，直接使用全局值）
        mota = summary['mota'].iloc[0]
        motp = summary['motp'].iloc[0]

        # 绘制指标值
        plt.bar(['MOTA', 'MOTP'], [mota, motp])
        plt.ylabel('Score')
        plt.title('Performance Summary')
        plt.ylim(0, 1)
        plt.grid(True)

        # 保存图像
        plt.savefig(os.path.join(output_dir, 'performance_summary.png'))
        plt.close()

    def save_report(self, output_dir: str = 'results') -> None:
        """
        保存详细的性能评估报告。
        :param output_dir: 输出目录
        """
        # 获取完整评估报告
        summary = self.get_summary()

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 将报告保存为文本文件
        with open(os.path.join(output_dir, 'metrics_report.txt'), 'w') as f:
            f.write(mm.io.render_summary(
                summary,
                formatters=self.metrics.formatters,  # 使用默认格式化器
                namemap=mm.io.motchallenge_metric_names  # 使用 MOTChallenge 指标名称
            ))