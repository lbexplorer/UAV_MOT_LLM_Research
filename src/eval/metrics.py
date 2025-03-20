# src/eval/metrics.py
import os
import motmetrics as mm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import seaborn as sns


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
        self.acc = mm.MOTAccumulator(auto_id=True)

        # 存储每帧的性能指标历史数据
        self.metrics_history = {
            'fp': [],  # 每帧的误报数
            'fn': [],  # 每帧的漏检数
            'idsw': [],  # 每帧的 ID 切换数
            'mota': [],  # 每帧的 MOTA
            'motp': []   # 每帧的 MOTP
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

        # 更新 MOT 数据累加器
        self.acc.update(
            gt_ids,  # Ground truth IDs
            pred_ids,  # 预测结果 IDs
            dist_matrix  # 距离矩阵
        )

        # 计算当前帧的指标
        frame_metrics = self.metrics.compute(
            self.acc,
            metrics=['num_frames', 'num_objects', 'num_predictions', 
                    'num_false_positives', 'num_misses', 'num_switches',
                    'mota', 'motp'],
            name='frame'
        )

        # 更新历史数据
        self.metrics_history['fp'].append(frame_metrics['num_false_positives'].iloc[0])
        self.metrics_history['fn'].append(frame_metrics['num_misses'].iloc[0])
        self.metrics_history['idsw'].append(frame_metrics['num_switches'].iloc[0])
        self.metrics_history['mota'].append(frame_metrics['mota'].iloc[0])
        self.metrics_history['motp'].append(frame_metrics['motp'].iloc[0])

    def get_summary(self) -> pd.DataFrame:
        """
        获取完整的性能评估报告。
        :return: 包含所有指标的 DataFrame
        """
        summary = self.metrics.compute(
            self.acc,
            metrics=mm.metrics.motchallenge_metrics,
            name='Overall'
        )
        return summary

    def visualize_trends(self, output_dir: str = 'results') -> None:
        """
        绘制性能趋势图。
        :param output_dir: 输出目录
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Tracking Performance Analysis')

        # 1. MOTA 和 MOTP 趋势
        axes[0, 0].plot(self.metrics_history['mota'], label='MOTA')
        axes[0, 0].plot(self.metrics_history['motp'], label='MOTP')
        axes[0, 0].set_title('MOTA and MOTP Trends')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 2. 误报和漏检趋势
        axes[0, 1].plot(self.metrics_history['fp'], label='False Positives')
        axes[0, 1].plot(self.metrics_history['fn'], label='False Negatives')
        axes[0, 1].set_title('Error Trends')
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 3. ID切换趋势
        axes[1, 0].plot(self.metrics_history['idsw'], label='ID Switches')
        axes[1, 0].set_title('ID Switch Trends')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 4. 总体性能指标
        summary = self.get_summary()
        metrics = ['mota', 'motp', 'num_false_positives', 'num_misses', 'num_switches']
        values = [summary[m].iloc[0] for m in metrics]
        axes[1, 1].bar(metrics, values)
        axes[1, 1].set_title('Overall Performance')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_analysis.png'))
        plt.close()

    def save_report(self, output_dir: str = 'results') -> None:
        """
        保存详细的性能评估报告。
        :param output_dir: 输出目录
        """
        summary = self.get_summary()
        os.makedirs(output_dir, exist_ok=True)

        # 保存详细报告
        with open(os.path.join(output_dir, 'metrics_report.txt'), 'w') as f:
            f.write("Detailed MOT Metrics Report\n")
            f.write("=========================\n\n")
            
            # 写入总体指标
            f.write("Overall Metrics:\n")
            f.write("---------------\n")
            f.write(mm.io.render_summary(
                summary,
                formatters=self.metrics.formatters,
                namemap=mm.io.motchallenge_metric_names
            ))
            
            # 写入趋势分析
            f.write("\nTrend Analysis:\n")
            f.write("--------------\n")
            f.write(f"Average MOTA: {np.mean(self.metrics_history['mota']):.4f}\n")
            f.write(f"Average MOTP: {np.mean(self.metrics_history['motp']):.4f}\n")
            f.write(f"Total ID Switches: {sum(self.metrics_history['idsw'])}\n")
            f.write(f"Average False Positives per Frame: {np.mean(self.metrics_history['fp']):.2f}\n")
            f.write(f"Average False Negatives per Frame: {np.mean(self.metrics_history['fn']):.2f}\n")