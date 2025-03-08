import os

import motmetrics as mm
import matplotlib.pyplot as plt
import numpy as np


class MOTEvaluator:
    """
    性能评价模块，用于评估目标追踪系统的性能。
    """

    def __init__(self):
        self.acc = mm.MOTAccumulator(auto_id=True)  # MOTAccumulator 实例
        self.metrics = mm.metrics.create()  # 性能指标计算器

    def update(self, frame_id, gt_data, track_results):
        """
        更新性能评估器。
        :param frame_id: 当前帧ID
        :param gt_data: 当前帧的ground truth数据
        :param track_results: 当前帧的追踪结果
        """
        gt_ids = [d['id'] for d in gt_data]  # Ground truth ID列表
        gt_boxes = [d['bbox'] for d in gt_data]  # Ground truth边界框列表

        track_ids = [tr[0] for tr in track_results]  # 追踪结果ID列表
        track_boxes = [tr[1] for tr in track_results]  # 追踪结果边界框列表

        # 计算IoU距离矩阵
        iou_matrix = self._compute_iou_matrix(gt_boxes, track_boxes)

        # 更新评估器
        self.acc.update(
            gt_ids,  # Ground truth IDs
            track_ids,  # Tracker IDs
            iou_matrix,  # 距离矩阵
            frameid=frame_id  # 帧ID
        )

    def _compute_iou_matrix(self, boxes1, boxes2):
        """
        计算IoU距离矩阵。
        :param boxes1: 边界框列表1
        :param boxes2: 边界框列表2
        :return: IoU距离矩阵
        """
        iou_matrix = np.zeros((len(boxes1), len(boxes2)))
        for i, b1 in enumerate(boxes1):
            for j, b2 in enumerate(boxes2):
                iou_matrix[i, j] = 1 - mm.distances.iou_matrix(
                    [b1], [b2], [0.5])[0][0]  # 转换为距离（1-IoU）
        return iou_matrix

    def get_metrics(self):
        """
        获取完整评估指标。
        :return: 性能指标
        """
        return mm.metrics.motchallenge_metrics(self.acc)

    def visualize_results(self, output_dir='results'):
        """
        可视化评估结果。
        :param output_dir: 输出目录
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 生成指标报告
        summary = self.get_metrics()
        mm.io.render_summary(
            summary,
            formatters=self.metrics.formatters,
            namemap=mm.io.motchallenge_metric_names
        )

        # 绘制趋势图
        plt.figure(figsize=(12, 6))
        for metric in ['mota', 'motp']:
            plt.plot(self.acc.events[metric], label=metric.upper())
        plt.xlabel('Frames')
        plt.ylabel('Score')
        plt.title('Performance Trends')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'performance_trend.png'))
        plt.close()
