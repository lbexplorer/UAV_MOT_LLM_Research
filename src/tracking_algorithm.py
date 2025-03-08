import logging
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

# 配置日志记录器
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])


# 动态阈值函数（优化版）
def dynamic_threshold(velocity, size, miss_count):
    base_threshold = 25  # 提高基础阈值
    speed_factor = np.linalg.norm(velocity) * 0.5  # 调整速度系数
    size_factor = size * 0.2  # 调整大小系数
    miss_penalty = miss_count * 3  # 降低丢失惩罚
    return max(base_threshold + speed_factor + size_factor - miss_penalty, 10)  # 设置最小阈值


# 马氏距离计算（保持不变）
def calculate_mahalanobis_distance(prediction, detection, covariance_matrix):
    covariance_matrix_2d = covariance_matrix[:2, :2]
    diff = prediction[:2] - detection[:2]
    inv_cov = np.linalg.inv(covariance_matrix_2d)
    return distance.mahalanobis(diff, np.zeros_like(diff), inv_cov)


# 扩展卡尔曼滤波器（保持不变）
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
        acceleration = np.linalg.norm(self.state[2:4] - self.previous_velocity) / self.time_interval
        self.Q[:2, :2] *= 1 + 0.1 * acceleration
        self.previous_velocity = self.state[2:4]


# 目标类（增加状态字段）
class Target:
    def __init__(self, obj_id, ekf):
        self.obj_id = obj_id
        self.ekf = ekf
        self.miss_count = 0
        self.status = "confirmed"  # 新增状态标识


# 新增候选目标类
class PendingTarget:
    def __init__(self, detection):
        self.ekf = ExtendedKalmanFilter([detection[0], detection[1], 0, 0])
        self.hit_count = 1
        self.last_seen = 0  # 最后出现帧数
        self.status = "pending"


# 目标匹配器（优化匹配逻辑）
class TargetMatcher:
    def match(self, predictions, detections, threshold_func):
        distance_matrix = np.zeros((len(predictions), len(detections)))
        pred_ids = list(predictions.keys())

        for i, pid in enumerate(pred_ids):
            obj = predictions[pid]
            for j, det in enumerate(detections):
                covariance = obj.ekf.P
                dist = calculate_mahalanobis_distance(
                    obj.ekf.state[:2], det[:2], covariance)
                # 应用动态阈值
                threshold = threshold_func(
                    obj.ekf.state[2:4],  # 速度
                    np.linalg.norm(det[2:4]),  # 检测目标大小
                    obj.miss_count
                )
                distance_matrix[i, j] = dist if dist < threshold else 1e6  # 超出阈值的设为极大值
        # 使用匈牙利算法匹配
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        valid_matches = []
        for r, c in zip(row_ind, col_ind):
            if distance_matrix[r, c] < 1e6:
                valid_matches.append((pred_ids[r], c))
        return valid_matches


# 改进后的目标追踪器
class ObjectTracker:
    def __init__(self, reserve=3, confirm_frames=3):
        self.tracked_objects = {}  # 已确认目标
        self.pending_targets = []  # 候选目标
        self.reserve = reserve  # 最大保留帧数
        self.confirm_frames = confirm_frames  # 确认所需帧数
        self.next_id = 1  # ID分配计数器
        self.matcher = TargetMatcher()
        self.frame_count = 0  # 总帧数计数器

    def predict_targets(self):
        # 预测所有已确认目标
        for obj in self.tracked_objects.values():
            obj.ekf.predict()
        return self.tracked_objects

    def update_targets(self, matches, current_detections):
        # 更新匹配成功的已确认目标
        for obj_id, det_index in matches:
            detection = current_detections[det_index]
            self.tracked_objects[obj_id].ekf.update(detection[:2])
            self.tracked_objects[obj_id].miss_count = 0  # 重置丢失计数器

    def process_pending_targets(self, unmatched_detections):
        # 预测候选目标
        for pt in self.pending_targets:
            pt.ekf.predict()

        # 构建候选目标匹配矩阵
        distance_matrix = np.zeros((len(self.pending_targets), len(unmatched_detections)))
        for i, pt in enumerate(self.pending_targets):
            for j, det in enumerate(unmatched_detections):
                covariance = pt.ekf.P
                distance_matrix[i, j] = calculate_mahalanobis_distance(
                    pt.ekf.state[:2], det[:2], covariance)

        # 执行匹配
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        matched_pending = set()
        matched_detections = set()

        # 处理有效匹配
        for r, c in zip(row_ind, col_ind):
            if distance_matrix[r, c] < dynamic_threshold([0, 0], 0, 0):  # 使用基础阈值
                pt = self.pending_targets[r]
                pt.ekf.update(unmatched_detections[c][:2])
                pt.hit_count += 1
                pt.last_seen = self.frame_count
                matched_pending.add(r)
                matched_detections.add(c)

        # 更新未匹配的候选目标
        new_pending = []
        for i, pt in enumerate(self.pending_targets):
            if i in matched_pending:
                continue
            if self.frame_count - pt.last_seen <= self.reserve:
                new_pending.append(pt)
        self.pending_targets = new_pending

        # 添加新候选目标
        for j, det in enumerate(unmatched_detections):
            if j not in matched_detections:
                new_pt = PendingTarget(det)
                new_pt.last_seen = self.frame_count
                self.pending_targets.append(new_pt)

        # 确认稳定目标
        confirmed = []
        for pt in self.pending_targets:
            if pt.hit_count >= self.confirm_frames:
                confirmed.append(pt)

        # 分配正式ID
        for pt in confirmed:
            new_id = self.next_id
            self.tracked_objects[new_id] = Target(new_id, pt.ekf)
            self.pending_targets.remove(pt)
            self.next_id += 1

    def remove_lost_targets(self):
        # 移除丢失的已确认目标
        to_remove = []
        for obj_id, obj in self.tracked_objects.items():
            if obj.miss_count >= self.reserve:
                to_remove.append(obj_id)
            else:
                obj.miss_count += 1
        for obj_id in to_remove:
            del self.tracked_objects[obj_id]

    def update_frame(self, detections):
        self.frame_count += 1

        # 主匹配流程
        predictions = self.predict_targets()
        matches = self.matcher.match(predictions, detections, dynamic_threshold)
        self.update_targets(matches, detections)

        # 分离未匹配检测
        matched_det_indices = {m[1] for m in matches}
        unmatched_detections = [d for i, d in enumerate(detections) if i not in matched_det_indices]

        # 处理候选目标
        self.process_pending_targets(unmatched_detections)

        # 清理丢失目标
        self.remove_lost_targets()

        # 返回当前追踪状态
        return {
            "confirmed": [(k, v.ekf.state[:2]) for k, v in self.tracked_objects.items()],
            "pending": [(pt.ekf.state[:2], pt.hit_count) for pt in self.pending_targets]
        }


# 示例使用
if __name__ == "__main__":
    # 模拟检测数据（每帧的检测列表）
    sample_detections = [
        [[100, 100, 20, 20]],  # 第0帧
        [[102, 98, 20, 20]],  # 第1帧
        [[105, 95, 20, 20]],  # 第2帧
        [],  # 第3帧（目标消失）
        [[108, 92, 20, 20]],  # 第4帧
        [[110, 90, 20, 20]],  # 第5帧
    ]

    tracker = ObjectTracker(reserve=3, confirm_frames=3)

    for frame, detections in enumerate(sample_detections):
        state = tracker.update_frame(detections)
        print(f"\nFrame {frame} Tracking Results:")
        print(f"Confirmed Targets: {state['confirmed']}")
        print(f"Pending Targets: {state['pending']}")