import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

class FrameMetrics:
    def __init__(self, tp=0, fp=0, fn=0, id_switches=0):
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.id_switches = id_switches

    def calculate_precision_recall_f1(self):
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

class MetricsEvaluator:
    def __init__(self, gt_data):
        """
        初始化性能评估器
        :param gt_data: Ground Truth 数据列表，每个元素是一个字典
        """
        self.gt_data = gt_data

    def evaluate_performance(self, tracked_results):
        """
        计算 Precision, Recall, F1, MOTA, MOTP
        :param tracked_results: 跟踪结果列表，每个元素是一个元组 (frame, id, x1, y1)
        :return: 各项性能指标
        """
        # 将元组格式的跟踪结果转换为字典格式
        tracked_results_dict = [{'frame': res[0], 'id': res[1], 'x1': res[2], 'y1': res[3]} for res in tracked_results]

        frame_metrics = []
        frame_set = set()
        for result in self.gt_data:
            frame_set.add(result['frame'])

        frames = sorted(frame_set)

        for frame in frames:
            frame_gt = [gt for gt in self.gt_data if gt['frame'] == frame]
            frame_results = [result for result in tracked_results_dict if result['frame'] == frame]
            tp, fp, fn, id_switches = self.calculate_frame_metrics(frame_gt, frame_results)
            frame_metrics.append(FrameMetrics(tp, fp, fn, id_switches))

        precision, recall, f1 = zip(*[metrics.calculate_precision_recall_f1() for metrics in frame_metrics])
        mota, motp = self.calculate_mota_motp(tracked_results_dict)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mota': mota,
            'motp': motp
        }


    def calculate_frame_metrics(self, frame_gt, frame_results):
        """
        计算单帧的 TP、FP、FN 和 ID Switches
        :param frame_gt: Ground Truth 数据列表
        :param frame_results: 预测结果列表
        :return: TP、FP、FN、ID Switches
        """
        tp, fp, fn, id_switches = 0, 0, len(frame_gt), 0
        matched_gt_indices = set()

        # 计算 TP 和 FP
        for result in frame_results:
            matched_gt = self.find_matching_gt(result, frame_gt)
            if matched_gt is not None:
                tp += 1
                matched_gt_indices.add(matched_gt['id'])
            else:
                fp += 1

        # 计算 FN
        fn = len(frame_gt) - len(matched_gt_indices)

        # 计算 ID Switches (此处简化处理，实际可能需要更复杂的逻辑)
        id_switches = 0

        return tp, fp, fn, id_switches

    def find_matching_gt(self, result, frame_gt):
        """
        寻找与预测结果匹配的 GT（根据距离等规则）
        :param result: 预测结果字典
        :param frame_gt: Ground Truth 数据列表
        :return: 匹配的 GT 字典，若没有匹配返回 None
        """
        min_distance = float('inf')
        best_match = None
        for gt in frame_gt:
            distance = np.linalg.norm([result['x1'] - gt['x1'], result['y1'] - gt['y1']])
            if distance < min_distance:
                min_distance = distance
                best_match = gt
        return best_match if min_distance < 50 else None

    def calculate_mota_motp(self, tracked_results):
        """
        计算 MOTA 和 MOTP 性能指标
        :param tracked_results: 跟踪结果列表，每个元素是一个字典
        :return: MOTA 和 MOTP 值
        """
        total_tp = total_fp = total_fn = total_id_switches = 0
        total_distance = 0
        total_matches = 0
        frames = sorted(set([result['frame'] for result in self.gt_data]))

        for frame in frames:
            frame_gt = [gt for gt in self.gt_data if gt['frame'] == frame]
            frame_results = [result for result in tracked_results if result['frame'] == frame]

            tp, fp, fn, id_switches = self.calculate_frame_metrics(frame_gt, frame_results)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_id_switches += id_switches

            # Ensure correct 2D array format for coordinates (x1, y1)
            gt_coords = np.array([[gt['x1'], gt['y1']] for gt in frame_gt])
            result_coords = np.array([[res['x1'], res['y1']] for res in frame_results])

            if len(gt_coords) > 0 and len(result_coords) > 0:
                # 计算 MOTP
                distances = cdist(result_coords, gt_coords)
                for i, gt in enumerate(frame_gt):
                    # 找到每个 ground truth 与结果之间的最小距离
                    min_dist = np.min(distances[:, i]) if distances.size else 0
                    total_distance += min_dist
                    total_matches += 1

        # 计算 MOTA
        mota = (total_tp - total_id_switches) / (total_tp + 0.5 * total_fp + total_fn) if (
                                                                                                      total_tp + 0.5 * total_fp + total_fn) > 0 else 0

        # 计算 MOTP
        motp = total_distance / total_matches if total_matches > 0 else 0

        return mota, motp

def plot_metrics(frames, precision, recall, f1):
    """
    绘制 Precision, Recall, F1-score 的折线图
    :param frames: 帧数列表
    :param precision: Precision 值
    :param recall: Recall 值
    :param f1: F1-score 值
    """
    plt.figure(figsize=(12, 6))
    plt.plot(frames, precision, label='Precision', color='blue', marker='o')
    plt.plot(frames, recall, label='Recall', color='green', marker='s')
    plt.plot(frames, f1, label='F1-score', color='red', marker='^')

    plt.title('Precision, Recall, F1-score over Frames')
    plt.xlabel('Frames')
    plt.ylabel('Metric Value')

    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_mota_motp(frames, mota, motp):
    """
    绘制 MOTA 和 MOTP 的折线图
    :param frames: 帧数列表
    :param mota: MOTA 值
    :param motp: MOTP 值
    """
    plt.figure(figsize=(10, 6))
    plt.plot(frames, mota, label='MOTA', color='purple', marker='o')
    plt.plot(frames, motp, label='MOTP', color='orange', marker='s')

    plt.title('MOTA and MOTP over Frames')
    plt.xlabel('Frames')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
