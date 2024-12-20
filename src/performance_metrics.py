import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

# 动态调整过程噪声
def optimize_process_noise(self):
    acceleration = np.linalg.norm(self.state[2:4] - self.previous_velocity) / self.time_interval
    self.Q[:2, :2] *= 1 + 0.1 * acceleration  # 根据加速度动态调整位置噪声
    self.previous_velocity = self.state[2:4]

def dynamic_threshold(velocity, size, miss_count):
    base_threshold = 20
    speed_factor = np.linalg.norm(velocity) / 10
    size_factor = size / 100
    miss_penalty = miss_count * 5
    return base_threshold + speed_factor + size_factor - miss_penalty

def calculate_mahalanobis_distance(prediction, detection, covariance_matrix):
    covariance_matrix_2d = covariance_matrix[:2, :2]
    diff = prediction - detection
    return distance.mahalanobis(diff, np.zeros_like(diff), np.linalg.inv(covariance_matrix_2d))

# 扩展卡尔曼滤波器类
class ExtendedKalmanFilter:
    def __init__(self, initial_state, time_interval=1):
        self.state = initial_state
        self.P = np.eye(4)
        self.F = np.eye(4)
        self.F[0, 2] = self.F[1, 3] = time_interval
        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 0.1
        self.previous_velocity = np.zeros(2)

    def predict(self):
        self.optimize_process_noise()
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.state[:2]

    def update(self, measurement):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        y = measurement - np.dot(H, self.state)
        S = np.dot(H, np.dot(self.P, H.T)) + self.R
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        self.state += np.dot(K, y)
        self.P = np.dot(np.eye(4) - np.dot(K, H), self.P)

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
        return {obj_id: obj.ekf.predict() for obj_id, obj in self.tracked_objects.items()}

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
