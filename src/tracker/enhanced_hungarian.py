# src/tracker/enhanced_hungarian.py
import numpy as np
from scipy.optimize import linear_sum_assignment
from .kalman_filter import ExtendedKalmanFilter
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Target:
    """已确认目标类"""

    def __init__(self, obj_id: int, bbox: Tuple[float, float, float, float]):
        self.obj_id = obj_id
        self.ekf = ExtendedKalmanFilter([bbox[0], bbox[1], 0, 0])  # 初始化EKF
        self.bbox = bbox
        self.miss_count = 0
        self.status = "confirmed"


class PendingTarget:
    """候选目标类"""

    def __init__(self, bbox: Tuple[float, float, float, float]):
        self.ekf = ExtendedKalmanFilter([bbox[0], bbox[1], 0, 0])
        self.bbox = bbox
        self.hit_count = 1
        self.last_seen = 0


class EnhancedHungarianTracker:
    """综合优化的多目标跟踪器"""

    def __init__(self, base_threshold=0.5, confirm_frames=3, reserve_frames=3):
        """
        :param base_threshold: 基础IOU匹配阈值
        :param confirm_frames: 候选目标确认所需帧数
        :param reserve_frames: 目标丢失后保留的最大帧数
        """
        self.base_threshold = base_threshold
        self.confirm_frames = confirm_frames
        self.reserve_frames = reserve_frames
        self.confirmed_targets: Dict[int, Target] = {}
        self.pending_targets: List[PendingTarget] = []
        self.next_id = 1
        self.is_initialized = False  # 初始化标志位

    def dynamic_threshold(self, velocity: np.ndarray, size: float, miss_count: int) -> float:
        """动态阈值函数"""
        speed_factor = np.linalg.norm(velocity) * 0.1
        size_factor = size * 0.05
        miss_penalty = miss_count * 0.2
        return max(self.base_threshold + speed_factor + size_factor - miss_penalty, 0.3)

    def _iou_cost_matrix(self, tracks: List[Target], detections: List[Tuple]) -> np.ndarray:
        """向量化IOU计算"""
        # 处理空输入
        if len(tracks) == 0 or len(detections) == 0:
            return np.zeros((len(tracks), len(detections)))

        track_boxes = np.array([t.bbox for t in tracks])
        det_boxes = np.array(detections)

        # 确保数组维度正确（至少二维）
        if track_boxes.ndim == 1:
            track_boxes = track_boxes.reshape(-1, 4)
        if det_boxes.ndim == 1:
            det_boxes = det_boxes.reshape(-1, 4)

        # 计算交集坐标（修复维度问题）
        x_a = np.maximum(track_boxes[:, 0][:, None], det_boxes[:, 0])
        y_a = np.maximum(track_boxes[:, 1][:, None], det_boxes[:, 1])
        x_b = np.minimum(track_boxes[:, 2][:, None], det_boxes[:, 2])
        y_b = np.minimum(track_boxes[:, 3][:, None], det_boxes[:, 3])

        inter_area = np.maximum(0, x_b - x_a) * np.maximum(0, y_b - y_a)
        area_track = (track_boxes[:, 2] - track_boxes[:, 0]) * (track_boxes[:, 3] - track_boxes[:, 1])
        area_det = (det_boxes[:, 2] - det_boxes[:, 0]) * (det_boxes[:, 3] - det_boxes[:, 1])
        union_area = area_track[:, None] + area_det - inter_area
        iou = inter_area / (union_area + 1e-8)
        return 1 - iou

    def update(self, detections: List[Tuple]) -> Dict:
        # ================== 新增代码：处理初始帧 ==================
        if not self.is_initialized:
            self.is_initialized = True
            # 初始帧直接分配所有检测目标为已确认目标
            if len(detections) > 0:
                for det in detections:
                    self.confirmed_targets[self.next_id] = Target(self.next_id, det)
                    self.next_id += 1
            return {
                "confirmed": [(k, v.bbox) for k, v in self.confirmed_targets.items()],
                "pending": []
            }
        """主更新流程"""
        # 步骤1：预测已确认目标
        for target in self.confirmed_targets.values():
            predicted_bbox = target.ekf.predict()
            target.bbox = (*predicted_bbox, target.bbox[2], target.bbox[3])  # 保持宽高

        # 步骤2：匈牙利匹配
        cost_matrix = self._iou_cost_matrix(list(self.confirmed_targets.values()), detections)
        print(f"cost_matrix:{cost_matrix}")
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 动态阈值筛选
        valid_matches = []
        for r, c in zip(row_ind, col_ind):
            target = list(self.confirmed_targets.values())[r]
            iou = 1 - cost_matrix[r, c]
            threshold = self.dynamic_threshold(target.ekf.state[2:4],
                                               (target.bbox[2] - target.bbox[0]) * (target.bbox[3] - target.bbox[1]),
                                               target.miss_count)
            print(f"Matching target {target.obj_id} (IOU: {iou:.2f}) with detection {c} (Threshold: {threshold:.2f})")
            if iou >= threshold:
                valid_matches.append((r, c))
        print(f"Valid matches after thresholding: {valid_matches}")

        # 步骤3：更新匹配目标
        matched_tracks = set()
        matched_dets = set()
        for r, c in valid_matches:
            target_id = list(self.confirmed_targets.keys())[r]
            # 确保 measurement 是 [x, y] 格式
            measurement = np.array(detections[c][:2], dtype=np.float64).reshape(2, 1)
            self.confirmed_targets[target_id].ekf.update(measurement)
            self.confirmed_targets[target_id].miss_count = 0
            print(f"Updated target {target_id}: new bbox {self.confirmed_targets[target_id].bbox}")
            matched_tracks.add(r)
            matched_dets.add(c)

        # 步骤4：处理未匹配的已确认目标
        for idx, target in enumerate(self.confirmed_targets.values()):
            if idx not in matched_tracks:
                target.miss_count += 1
                if target.miss_count > self.reserve_frames:
                    print(f"Target {target.obj_id} lost and removed after {target.miss_count} misses.")
                    del self.confirmed_targets[target.obj_id]

        # 步骤5：处理未匹配的检测（新目标或候选目标）
        unmatched_detections = [detections[i] for i in range(len(detections)) if i not in matched_dets]
        for det in unmatched_detections:
            # 尝试与候选目标匹配
            matched = False
            for pt in self.pending_targets:
                predicted_bbox = pt.ekf.predict()
                iou = self._calc_iou(predicted_bbox, det)
                if iou >= self.base_threshold:
                    # 确保 measurement 是 [x, y] 格式
                    measurement = np.array(det[:2], dtype=np.float64).reshape(2, 1)
                    pt.ekf.update(measurement)
                    pt.hit_count += 1
                    pt.last_seen = 0
                    matched = True
                    print(f"Matched pending target: {pt.bbox} with detection {det}")
                    break
            if not matched:
                # 新增候选目标
                self.pending_targets.append(PendingTarget(det))
                print(f"Added new pending target: {det}")

        # 步骤6：更新候选目标状态
        new_pending = []
        for pt in self.pending_targets:
            pt.last_seen += 1
            if pt.last_seen <= self.reserve_frames:
                if pt.hit_count >= self.confirm_frames:
                    # 升级为已确认目标
                    self.confirmed_targets[self.next_id] = Target(self.next_id, pt.bbox)
                    self.next_id += 1
                else:
                    new_pending.append(pt)
        self.pending_targets = new_pending

        return {
            "confirmed": [(k, v.bbox) for k, v in self.confirmed_targets.items()],
            "pending": [(pt.bbox, pt.hit_count) for pt in self.pending_targets]
        }

    @staticmethod
    def _calc_iou(box_a: np.ndarray, box_b: Tuple) -> float:
        """计算两个矩形框的IOU"""
        # 确保 box_a 是 (x1, y1, x2, y2) 格式
        if len(box_a) == 2:
            # 如果 box_a 只有 [x, y]，假设宽高与 box_b 相同
            box_a = np.array([box_a[0], box_a[1], box_b[2], box_b[3]])
        elif len(box_a) == 4:
            box_a = np.array(box_a)
        else:
            raise ValueError("box_a 必须是长度为 2 或 4 的数组")

        # 确保 box_b 是 (x1, y1, x2, y2) 格式
        box_b = np.array(box_b)

        # 计算交集坐标
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])

        # 计算交集面积
        inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)

        # 计算各自面积
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        # 计算并集面积
        union_area = area_a + area_b - inter_area

        return inter_area / union_area if union_area != 0 else 0