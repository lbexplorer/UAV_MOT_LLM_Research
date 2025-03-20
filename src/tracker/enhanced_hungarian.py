# src/tracker/enhanced_hungarian.py
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from .kalman_filter import ExtendedKalmanFilter
from typing import List, Tuple, Dict
import logging

# 创建一个日志文件路径，使用绝对路径
log_file = r'E:\python\MOT\UAV_MOT_LLM_Research\logs\match.log'

# 如果文件夹不存在则创建
log_dir = os.path.dirname(log_file)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 配置日志输出
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),  # 输出到指定文件
        logging.StreamHandler()  # 也输出到控制台
    ]
)
logger = logging.getLogger(__name__)
logging.debug("这是一条调试信息")
logging.info("这是一条信息级别的日志")
logging.warning("这是一条警告级别的日志")
logging.error("这是一条错误级别的日志")
logging.critical("这是一条严重级别的日志")
# 检查是否有写入权限
if os.access(log_dir, os.W_OK):
    print(f"目录 {log_dir} 可写")
else:
    print(f"没有权限在 {log_dir} 写入文件")

class Target:
    """已确认目标类"""

    def __init__(self, obj_id: int, bbox: Tuple[float, float, float, float]):
        self.obj_id = obj_id
        self.ekf = ExtendedKalmanFilter([bbox[0], bbox[1], 0, 0])  # 初始化EKF
        self.bbox = bbox
        self.miss_count = 0
        self.status = "confirmed"
        logger.info(f"Created new confirmed target: ID={self.obj_id}, bbox={self.bbox}")


class PendingTarget:
    """候选目标类"""

    def __init__(self, bbox: Tuple[float, float, float, float]):
        self.ekf = ExtendedKalmanFilter([bbox[0], bbox[1], 0, 0])
        self.bbox = bbox
        self.hit_count = 1
        self.last_seen = 0
        logger.info(f"Created new pending target: bbox={self.bbox}")


class EnhancedHungarianTracker:
    """综合优化的多目标跟踪器"""

    def __init__(self, base_threshold=0.5, confirm_frames=3, reserve_frames=3):
        self.base_threshold = base_threshold
        self.confirm_frames = confirm_frames
        self.reserve_frames = reserve_frames
        self.confirmed_targets: Dict[int, Target] = {}
        self.pending_targets: List[PendingTarget] = []
        self.next_id = 1
        self.is_initialized = False
        logger.info(f"Tracker initialized with base_threshold={base_threshold}, "
                    f"confirm_frames={confirm_frames}, reserve_frames={reserve_frames}")

    def _iou_cost_matrix(self, tracks: List[Target], detections: List[Tuple]) -> np.ndarray:
        if len(tracks) == 0 or len(detections) == 0:
            logger.debug("Empty tracks or detections, returning zero cost matrix.")
            return np.zeros((len(tracks), len(detections)))

        track_boxes = np.array([t.bbox for t in tracks])
        det_boxes = np.array(detections)

        if track_boxes.ndim == 1:
            track_boxes = track_boxes.reshape(-1, 4)
        if det_boxes.ndim == 1:
            det_boxes = det_boxes.reshape(-1, 4)

        x_a = np.maximum(track_boxes[:, 0][:, None], det_boxes[:, 0])
        y_a = np.maximum(track_boxes[:, 1][:, None], det_boxes[:, 1])
        x_b = np.minimum(track_boxes[:, 2][:, None], det_boxes[:, 2])
        y_b = np.minimum(track_boxes[:, 3][:, None], det_boxes[:, 3])

        inter_area = np.maximum(0, x_b - x_a) * np.maximum(0, y_b - y_a)
        area_track = (track_boxes[:, 2] - track_boxes[:, 0]) * (track_boxes[:, 3] - track_boxes[:, 1])
        area_det = (det_boxes[:, 2] - det_boxes[:, 0]) * (det_boxes[:, 3] - det_boxes[:, 1])
        union_area = area_track[:, None] + area_det - inter_area
        iou = inter_area / (union_area + 1e-8)
        cost_matrix = 1 - iou
        logger.debug(f"Cost matrix calculated: {cost_matrix}")
        return cost_matrix

    def update(self, frame_id: int, detections: List[Tuple]) -> Dict:
        # 打印帧起始分隔符
        logger.info(f"\n========== Frame {frame_id} Start ==========")

        # 初始化阶段
        if not self.is_initialized:
            self.is_initialized = True
            if len(detections) > 0:
                for det in detections:
                    self.confirmed_targets[self.next_id] = Target(self.next_id, det)
                    self.next_id += 1
            logger.info("Tracker initialized with initial detections.")
            logger.info(f"========== Frame {frame_id} End ==========\n")
            return {
                "confirmed": [(k, v.bbox) for k, v in self.confirmed_targets.items()],
                "pending": []
            }

        logger.info(f"Processing new frame with {len(detections)} detections.")

        # 步骤1：预测已确认目标
        for target in self.confirmed_targets.values():
            predicted_state = target.ekf.predict()
            predicted_x, predicted_y = predicted_state[:2]
            original_width = target.bbox[2] - target.bbox[0]
            original_height = target.bbox[3] - target.bbox[1]
            target.bbox = (predicted_x, predicted_y, predicted_x + original_width, predicted_y + original_height)
            logger.debug(f"Predicted target {target.obj_id}: new bbox={target.bbox}")

        # 步骤2：匈牙利匹配
        cost_matrix = self._iou_cost_matrix(list(self.confirmed_targets.values()), detections)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        logger.debug(f"Raw matches from Hungarian algorithm: rows={row_ind}, cols={col_ind}")

        valid_matches = []
        for r, c in zip(row_ind, col_ind):
            iou = 1 - cost_matrix[r, c]
            if iou >= self.base_threshold:
                valid_matches.append((r, c))
                logger.debug(f"Valid match: track_idx={r}, det_idx={c}, IOU={iou}")
        logger.info(f"Valid matches after thresholding: {valid_matches}")

        # 步骤3：更新匹配目标（直接使用检测框）
        matched_dets = set()
        for r, c in valid_matches:
            target_id = list(self.confirmed_targets.keys())[r]
            det_bbox = detections[c]
            measurement = np.array(det_bbox[:2], dtype=np.float64).reshape(2, 1)
            target = self.confirmed_targets[target_id]
            target.ekf.update(measurement)
            target.bbox = det_bbox  # 直接更新为检测框
            target.miss_count = 0
            matched_dets.add(c)
            logger.info(f"Updated target {target_id}: new bbox={target.bbox}")

        # 步骤4：处理未匹配的已确认目标
        to_delete = []
        for target_id, target in self.confirmed_targets.items():
            if target_id not in [list(self.confirmed_targets.keys())[r] for r, _ in valid_matches]:
                target.miss_count += 1
                logger.info(f"Target {target_id} miss_count increased to {target.miss_count}")
                if target.miss_count > self.reserve_frames:
                    to_delete.append(target_id)
                    logger.info(f"Target {target_id} lost and removed after {target.miss_count} misses.")
        for target_id in to_delete:
            del self.confirmed_targets[target_id]

        # 步骤5：处理未匹配的检测
        unmatched_detections = [detections[i] for i in range(len(detections)) if i not in matched_dets]
        logger.debug(f"Unmatched detections: {unmatched_detections}")
        for det in unmatched_detections:
            matched = False
            for pt in self.pending_targets:
                predicted_state = pt.ekf.predict()
                predicted_x, predicted_y = predicted_state[:2]
                original_width = pt.bbox[2] - pt.bbox[0]
                original_height = pt.bbox[3] - pt.bbox[1]
                predicted_bbox = (predicted_x, predicted_y, predicted_x + original_width, predicted_y + original_height)
                iou = self._calc_iou(predicted_bbox, det)
                if iou >= self.base_threshold:
                    measurement = np.array(det[:2], dtype=np.float64).reshape(2, 1)
                    pt.ekf.update(measurement)
                    pt.hit_count += 1
                    pt.last_seen = 0
                    pt.bbox = det  # 更新候选目标bbox
                    matched = True
                    logger.debug(f"Matched pending target: bbox={pt.bbox} with detection {det}")
                    break
            if not matched:
                self.pending_targets.append(PendingTarget(det))
                logger.info(f"Added new pending target: bbox={det}")

        # 步骤6：更新候选目标状态
        new_pending = []
        for pt in self.pending_targets:
            pt.last_seen += 1
            if pt.last_seen <= self.reserve_frames:
                if pt.hit_count >= self.confirm_frames:
                    self.confirmed_targets[self.next_id] = Target(self.next_id, pt.bbox)
                    logger.info(f"Promoted pending target to confirmed: ID={self.next_id}, bbox={pt.bbox}")
                    self.next_id += 1
                else:
                    new_pending.append(pt)
        self.pending_targets = new_pending

        # 帧结束日志分隔符
        logger.info(f"========== Frame {frame_id} End ==========\n")
        return {
            "confirmed": [(k, v.bbox) for k, v in self.confirmed_targets.items()],
            "pending": [(pt.bbox, pt.hit_count) for pt in self.pending_targets]
        }

    @staticmethod
    def _calc_iou(box_a: np.ndarray, box_b: Tuple) -> float:
        if len(box_a) == 2:
            box_a = np.array([box_a[0], box_a[1], box_b[2], box_b[3]])
        elif len(box_a) == 4:
            box_a = np.array(box_a)
        else:
            raise ValueError("Invalid box_a format")

        box_b = np.array(box_b)
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])

        inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union_area = area_a + area_b - inter_area
        return inter_area / union_area if union_area != 0 else 0
