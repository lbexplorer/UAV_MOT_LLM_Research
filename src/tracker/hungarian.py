# src/tracker/hungarian.py
import numpy as np
from scipy.optimize import linear_sum_assignment
import time  # 用于性能分析
import logging
from typing import List, Tuple, Dict


class HungarianTracker:
    """基于匈牙利算法的多目标跟踪器（支持IOU匹配）"""

    def __init__(self, iou_threshold: float = 0.5, max_unmatched: int = 3):
        """
        初始化跟踪器
        :param iou_threshold: IOU匹配阈值，默认0.5
        :param max_unmatched: 目标最大失配帧数，默认3
        """
        self.tracks: Dict[int, dict] = {}  # 跟踪目标字典 {track_id: {'bbox': bbox, 'unmatched': count}}
        self.next_id = 1  # ID分配计数器
        self.iou_threshold = iou_threshold
        self.max_unmatched = max_unmatched
        self.frame_count = 0  # 帧计数器
        self.total_time = 0.0  # 总耗时统计

    def update(self, detections: List[Tuple[float, float, float, float]]) -> Dict:
        """
        执行目标匹配与状态更新
        :param detections: 当前帧检测列表 [(x1, y1, x2, y2), ...]
        :return: 跟踪结果字典 {'matches': [(track_id, det_idx),...], 'unmatched': [track_id,...]}
        """
        start_time = time.time()
        self.frame_count += 1

        # 初始化返回结果
        results = {'matches': [], 'unmatched': []}

        # 步骤1：生成代价矩阵（1 - IOU）
        cost_matrix = self._iou_cost_matrix(detections)

        # 步骤2：使用匈牙利算法进行匹配
        if cost_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matches = [(row, col) for row, col in zip(row_ind, col_ind)]
        else:
            matches = []

        # 步骤3：处理匹配结果
        matched_tracks = set()
        matched_detections = set()

        for trk_idx, det_idx in matches:
            iou = 1 - cost_matrix[trk_idx, det_idx]
            if iou >= self.iou_threshold:
                track_id = list(self.tracks.keys())[trk_idx]
                self.tracks[track_id]['bbox'] = detections[det_idx]
                self.tracks[track_id]['unmatched'] = 0  # 重置失配计数
                results['matches'].append((track_id, det_idx))
                matched_tracks.add(trk_idx)
                matched_detections.add(det_idx)

        # 步骤4：处理未匹配的跟踪目标（失配处理）
        for trk_idx in range(len(self.tracks)):
            if trk_idx not in matched_tracks:
                track_id = list(self.tracks.keys())[trk_idx]
                self.tracks[track_id]['unmatched'] += 1
                if self.tracks[track_id]['unmatched'] > self.max_unmatched:
                    del self.tracks[track_id]
                    results['unmatched'].append(track_id)

        # 步骤5：处理未匹配的检测（新目标初始化）
        for det_idx in range(len(detections)):
            if det_idx not in matched_detections:
                self.tracks[self.next_id] = {
                    'bbox': detections[det_idx],
                    'unmatched': 0
                }
                results['matches'].append((self.next_id, det_idx))
                self.next_id += 1

        # 性能分析
        elapsed = time.time() - start_time
        self.total_time += elapsed
        logging.info(f"Frame {self.frame_count}: 匹配耗时 {elapsed:.4f}s | 目标数 {len(self.tracks)}")

        return results

    def _iou_cost_matrix(self, detections: List[Tuple]) -> np.ndarray:
        """
        计算IOU距离矩阵（1 - IOU）
        :param detections: 当前帧检测框列表 [(x1,y1,x2,y2), ...]
        :return: 成本矩阵 shape=(n_tracks, n_detections)
        """
        n_tracks = len(self.tracks)
        n_det = len(detections)

        if n_tracks == 0 or n_det == 0:
            return np.empty((0, 0))

        cost_matrix = np.zeros((n_tracks, n_det))

        # 提取所有跟踪框和检测框
        track_boxes = [track['bbox'] for track in self.tracks.values()]

        # 向量化计算IOU
        for i, trk_box in enumerate(track_boxes):
            for j, det_box in enumerate(detections):
                cost_matrix[i, j] = 1 - self._calc_iou(trk_box, det_box)

        return cost_matrix

    @staticmethod
    def _calc_iou(box_a: Tuple, box_b: Tuple) -> float:
        """
        计算两个矩形框的IOU（交并比）
        :param box_a: (x1, y1, x2, y2)
        :param box_b: (x1, y1, x2, y2)
        :return: IOU值 [0, 1]
        """
        # 计算交集区域
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])

        inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)

        # 计算各自面积
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        # 计算并集面积
        union_area = area_a + area_b - inter_area

        return inter_area / union_area if union_area != 0 else 0

    def get_performance(self) -> Dict:
        """
        获取性能统计信息
        :return: 包含平均处理时间和目标数的字典
        """
        return {
            'avg_time': self.total_time / self.frame_count,
            'max_targets': max(len(self.tracks), 1),
            'total_frames': self.frame_count
        }