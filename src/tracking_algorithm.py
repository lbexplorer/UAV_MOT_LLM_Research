import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

from src.performance_metrics import MetricsEvaluator


def dynamic_threshold(velocity, size, miss_count):
    base_threshold = 20
    speed_factor = np.linalg.norm(velocity) / 10
    size_factor = size / 100
    miss_penalty = miss_count * 5
    return base_threshold + speed_factor + size_factor - miss_penalty

# 马氏距离计算, 用于匹配距离计算
def calculate_mahalanobis_distance(prediction, detection, covariance_matrix):
    # 提取与 [x, y] 相关的 2x2 协方差子矩阵
    covariance_matrix_2d = covariance_matrix[:2, :2]
    diff = prediction - detection
    return distance.mahalanobis(diff, np.zeros_like(diff), np.linalg.inv(covariance_matrix_2d))

# 扩展卡尔曼滤波器类,用于每个目标的预测与更新
class ExtendedKalmanFilter:
    def __init__(self, initial_state, time_interval=1):
        self.state = np.array(initial_state, dtype=np.float64)
        self.P = np.eye(4)
        self.F = np.eye(4)
        self.F[0, 2] = self.F[1, 3] = time_interval
        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 0.1
        self.previous_velocity = np.zeros(2)
        self.time_interval = time_interval

    def predict(self):
        self.optimize_process_noise()
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.state[:2]

    def update(self, measurement):
        measurement = np.array(measurement, dtype=np.float64)
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        y = measurement - np.dot(H, self.state)
        S = np.dot(H, np.dot(self.P, H.T)) + self.R
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        self.state += np.dot(K, y)
        self.P = np.dot(np.eye(4) - np.dot(K, H), self.P)

    def optimize_process_noise(self):
        # 实现过程噪声优化
        acceleration = np.linalg.norm(self.state[2:4] - self.previous_velocity) / self.time_interval
        self.Q[:2, :2] *= 1 + 0.1 * acceleration  # 根据加速度动态调整位置噪声
        self.previous_velocity = self.state[2:4]

# 目标信息类：封装目标的属性（位置、速度、大小、未匹配次数等）
class TargetInfo:
    def __init__(self, obj_id, initial_detection):
        self.obj_id = obj_id
        self.velocity = np.zeros(2)
        self.size = initial_detection[2] * initial_detection[3]
        self.reserve_counter = 0
        self.previous_position = initial_detection[:2]
        self.frame_count = 1

    def update_velocity(self, measurement, time_interval=1):
        self.velocity = (measurement - self.previous_position) / (time_interval * self.frame_count)
        self.previous_position = measurement
        self.frame_count += 1

# 目标类：封装目标的EKF实例和目标信息
class Target:
    def __init__(self, obj_id, ekf, target_info):
        self.obj_id = obj_id
        self.ekf = ekf
        self.target_info = target_info

    def update(self, measurement):
        self.ekf.update(measurement)
        self.target_info.update_velocity(measurement)

# 匹配器类：专门处理目标的匹配逻辑
class TargetMatcher:
    def __init__(self):
        pass

    def match(self, predictions, current_detections):
        distance_matrix = np.zeros((len(predictions), len(current_detections)))
        pred_ids = list(predictions.keys())
        print(f"Predictions: {len(predictions)}")
        print(f"Current detections: {len(current_detections)}")
        for i, pid in enumerate(pred_ids):
            for j, det in enumerate(current_detections):
                covariance_matrix = predictions[pid].ekf.P
                distance_matrix[i, j] = calculate_mahalanobis_distance(predictions[pid].ekf.state[:2], det[:2], covariance_matrix)
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        return [(pred_ids[row], col) for row, col in zip(row_ind, col_ind) if distance_matrix[row, col] < dynamic_threshold(predictions[pred_ids[row]].target_info.velocity, predictions[pred_ids[row]].target_info.size, predictions[pred_ids[row]].target_info.reserve_counter)]

# 目标跟踪器类：管理目标的预测、匹配和删除等
class ObjectTracker:
    def __init__(self, reserve=3, hit=3):
        self.tracked_objects = {}
        self.tracked_objects_info = {}
        self.reserve = reserve
        self.hit = hit
        self.matcher = TargetMatcher()

    def predict_targets(self):
        return {obj_id: obj for obj_id, obj in self.tracked_objects.items()}

    def update_matched_targets(self, matches, current_detections):
        for obj_id, det_index in matches:
            measurement = current_detections[det_index][:2]
            self.tracked_objects[obj_id].update(measurement)

    def handle_unmatched_targets(self, matches):
        matched_ids = {match[0] for match in matches}
        for obj_id in list(self.tracked_objects.keys()):
            if obj_id not in matched_ids:
                if obj_id not in self.tracked_objects_info:
                    self.tracked_objects_info[obj_id] = TargetInfo(obj_id, None)
                self.tracked_objects_info[obj_id].reserve_counter += 1
                if self.tracked_objects_info[obj_id].reserve_counter >= self.reserve:
                    del self.tracked_objects[obj_id]
                    del self.tracked_objects_info[obj_id]

    def add_new_targets(self, current_detections, matches):
        unmatched_detections = {i for i in range(len(current_detections))} - {m[1] for m in matches}
        for det_index in unmatched_detections:
            det = current_detections[det_index]
            new_target_id = len(self.tracked_objects) + 1
            ekf = ExtendedKalmanFilter(initial_state=np.array([det[0], det[1], 0, 0]))  # 初始假设目标静止
            target_info = TargetInfo(new_target_id, det)
            new_target = Target(new_target_id, ekf, target_info)
            self.tracked_objects[new_target_id] = new_target

# 运行目标追踪

import numpy as np


def run_tracking_objects(gt_data):
    """
    运行目标跟踪算法并计算每一帧的性能指标
    :param gt_data: Ground Truth 数据 (DataFrame)
    :return: 包含每一帧的性能指标的字典，计算出的 MOTA 和 MOTP
    """
    # 初始化性能评估器
    evaluator = MetricsEvaluator(gt_data)

    # 存储每一帧的指标数据
    frame_metrics = []
    tracked_results = []  # 跟踪结果 [(frame, id, x1, y1, w, h), ...]

    # 创建目标跟踪器实例
    object_tracker = ObjectTracker(reserve=3, hit=3)

    # 获取每一帧的 Ground Truth 数据
    frames = gt_data['frame'].unique()

    # 遍历每一帧
    for frame in frames:
        # 获取当前帧的 Ground Truth 数据
        frame_gt = gt_data[gt_data['frame'] == frame]
        current_detections = frame_gt[['x1', 'y1', 'w', 'h']].values

        # 如果是第一帧，初始化目标
        if frame == frames[0]:
            for det in current_detections:
                ekf = ExtendedKalmanFilter(initial_state=np.array([det[0], det[1], 0, 0]))  # 假设目标静止
                target_info = TargetInfo(len(tracked_results) + 1, det)
                target = Target(len(tracked_results) + 1, ekf, target_info)
                object_tracker.tracked_objects[target.obj_id] = target

        # 预测当前帧的目标位置
        predictions = object_tracker.predict_targets()

        # 使用匈牙利算法匹配当前帧的目标与预测
        matches = object_tracker.matcher.match(predictions, current_detections)

        # 更新已匹配的目标
        object_tracker.update_matched_targets(matches, current_detections)

        # 处理未匹配的目标
        object_tracker.handle_unmatched_targets(matches)

        # 添加新的目标
        object_tracker.add_new_targets(current_detections, matches)

        # 收集当前帧的跟踪结果
        tracked_results.extend([(frame, obj_id, *current_detections[match[1]]) for obj_id, match in matches])

        # 计算每一帧的 TP、FP、FN、ID Switches
        tp, fp, fn, id_switches = evaluator.calculate_frame_metrics(frame_gt, tracked_results)

        # 将当前帧的指标存储在字典中
        frame_metrics.append({
            'frame': frame,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'ID Switches': id_switches
        })

    # 计算 Precision、Recall、F1-Score
    frames, precision_list, recall_list, f1_list = evaluator.calculate_precision_recall_f1(frame_metrics)

    # 计算 MOTA 和 MOTP
    mota, motp = evaluator.calculate_git(tracked_results)

    # 返回每一帧的性能指标以及全局的 MOTA 和 MOTP
    return {
        'frames': frames,
        'precision': precision_list,
        'recall': recall_list,
        'f1': f1_list,
        'mota': mota,
        'motp': motp
    }