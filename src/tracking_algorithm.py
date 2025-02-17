import logging
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from src.performance_metrics import MetricsEvaluator

# 配置日志记录器，日志级别设置为INFO，以便输出关键信息
logging.basicConfig(level=logging.INFO,  # 设置日志级别为INFO
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

# 动态阈值函数，根据目标的速度、大小和丢失次数动态调整匹配的阈值
def dynamic_threshold(velocity, size, miss_count):
    base_threshold = 20  # 基础阈值
    speed_factor = np.linalg.norm(velocity) / 10  # 根据速度调整阈值
    size_factor = size / 100  # 根据目标大小调整阈值
    miss_penalty = miss_count * 5  # 根据丢失次数调整阈值
    return base_threshold + speed_factor + size_factor - miss_penalty

# 计算马氏距离，用于测量预测和检测之间的相似度
def calculate_mahalanobis_distance(prediction, detection, covariance_matrix):
    covariance_matrix_2d = covariance_matrix[:2, :2]  # 仅使用前两列和前两行（位置）
    diff = prediction[:2] - detection[:2]  # 计算位置的差异
    inv_cov = np.linalg.inv(covariance_matrix_2d)  # 计算协方差矩阵的逆
    return distance.mahalanobis(diff, np.zeros_like(diff), inv_cov)  # 返回马氏距离

# 扩展卡尔曼滤波器类，用于目标状态预测和更新
class ExtendedKalmanFilter:
    def __init__(self, initial_state, time_interval=1):
        self.state = np.array(initial_state, dtype=np.float64)  # 目标的初始状态
        self.P = np.eye(4)  # 初始协方差矩阵
        self.F = np.eye(4)  # 状态转移矩阵
        self.F[0, 2] = self.F[1, 3] = time_interval  # 根据时间间隔调整矩阵
        self.Q = np.eye(4) * 0.01  # 过程噪声矩阵
        self.R = np.eye(2) * 0.1  # 测量噪声矩阵
        self.previous_velocity = np.zeros(2)  # 初始速度为零
        self.time_interval = time_interval  # 时间间隔

    # 预测目标的下一位置
    def predict(self):
        self.optimize_process_noise()  # 优化过程噪声
        self.state = np.dot(self.F, self.state)  # 状态预测
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q  # 协方差预测
        return self.state[:2]  # 返回预测的前两维（位置）

    # 更新卡尔曼滤波器的状态
    def update(self, measurement):
        measurement = np.array(measurement, dtype=np.float64)  # 将测量值转换为numpy数组
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # 测量矩阵
        y = measurement - np.dot(H, self.state)  # 计算残差
        S = np.dot(H, np.dot(self.P, H.T)) + self.R  # 计算残差的协方差
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))  # 计算卡尔曼增益
        self.state += np.dot(K, y)  # 更新状态
        self.P = np.dot(np.eye(4) - np.dot(K, H), self.P)  # 更新协方差矩阵

    # 根据目标的加速度优化过程噪声
    def optimize_process_noise(self):
        acceleration = np.linalg.norm(self.state[2:4] - self.previous_velocity) / self.time_interval  # 计算加速度
        self.Q[:2, :2] *= 1 + 0.1 * acceleration  # 根据加速度调整过程噪声
        self.previous_velocity = self.state[2:4]  # 更新速度

# 目标类，用于存储每个目标的信息
class Target:
    def __init__(self, obj_id, ekf):
        self.obj_id = obj_id  # 目标ID
        self.ekf = ekf  # 扩展卡尔曼滤波器对象
        self.miss_count = 0  # 丢失计数

# 目标匹配器类，用于根据卡尔曼滤波器的预测结果进行目标匹配
class TargetMatcher:
    def match(self, predictions, current_detections):
        distance_matrix = np.zeros((len(predictions), len(current_detections)))  # 创建距离矩阵
        pred_ids = list(predictions.keys())  # 获取预测目标的ID
        # 计算所有预测目标和当前检测目标之间的马氏距离
        for i, pid in enumerate(pred_ids):
            for j, det in enumerate(current_detections):
                covariance_matrix = predictions[pid].ekf.P  # 获取预测目标的协方差矩阵
                distance_matrix[i, j] = calculate_mahalanobis_distance(
                    predictions[pid].ekf.state[:2], det[:2], covariance_matrix
                )
        row_ind, col_ind = linear_sum_assignment(distance_matrix)  # 使用匈牙利算法进行匹配
        matches = [(pred_ids[row], col) for row, col in zip(row_ind, col_ind)]  # 返回匹配的目标ID
        return matches

# 目标追踪器类，用于管理目标的跟踪状态和更新
class ObjectTracker:
    def __init__(self, reserve=3):
        self.tracked_objects = {}  # 存储跟踪目标的字典
        self.reserve = reserve  # 丢失目标的最大容忍次数
        self.matcher = TargetMatcher()  # 目标匹配器实例

    # 获取所有有效的预测目标（丢失次数小于reserve的目标）
    def predict_targets(self):
        return {obj_id: obj for obj_id, obj in self.tracked_objects.items() if obj.miss_count < self.reserve}

    # 根据匹配结果更新目标的状态
    def update_targets(self, matches, current_detections):
        for obj_id, det_index in matches:
            detection = current_detections[det_index]  # 获取当前检测目标
            self.tracked_objects[obj_id].ekf.update(detection[:2])  # 使用卡尔曼滤波器更新目标状态

    # 为未匹配到的检测目标添加新的目标
    def add_new_targets(self, unmatched_detections):
        for det in unmatched_detections:
            obj_id = len(self.tracked_objects) + 1  # 新目标的ID
            ekf = ExtendedKalmanFilter(initial_state=[det[0], det[1], 0, 0])  # 创建新的EKF实例
            self.tracked_objects[obj_id] = Target(obj_id, ekf)  # 添加新的目标

    # 移除丢失次数超过阈值的目标
    def remove_lost_targets(self):
        to_remove = []  # 待移除的目标列表
        for obj_id, obj in self.tracked_objects.items():
            if obj.miss_count >= self.reserve:  # 丢失次数达到最大值，移除目标
                to_remove.append(obj_id)
            else:
                obj.miss_count += 1  # 未匹配到时，丢失计数加1
        for obj_id in to_remove:
            del self.tracked_objects[obj_id]  # 移除目标

# 运行目标追踪的主函数
def run_tracking_objects(detections_list):
    tracker = ObjectTracker()  # 创建目标追踪器实例
    evaluator = MetricsEvaluator()  # 创建性能评估器实例

    # 遍历每一帧的检测结果
    for frame, detections in enumerate(detections_list):
        predictions = tracker.predict_targets()  # 获取当前的预测目标
        matches = tracker.matcher.match(predictions, detections)  # 匹配目标
        tracker.update_targets(matches, detections)  # 更新目标状态

        # 获取未匹配的检测目标
        unmatched_detections = [det for i, det in enumerate(detections) if i not in [match[1] for match in matches]]
        tracker.add_new_targets(unmatched_detections)  # 添加新的目标
        tracker.remove_lost_targets()  # 移除丢失目标

        # 更新当前帧的性能指标
        frame_metrics = evaluator.update_frame(predictions, detections)
        print(f"Tracked Results for Frame {frame}: {matches}")  # 输出匹配结果
        print(f"Performance Metrics for Frame {frame}: {frame_metrics}")  # 输出性能指标
