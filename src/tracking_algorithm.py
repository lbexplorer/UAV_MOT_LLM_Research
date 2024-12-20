import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

# 动态调整过程噪声
def optimize_process_noise(self):
    """
    动态调整过程噪声，根据加速度调整位置噪声。
    该方法绑定到EKF类中，以访问类的属性。
    """
    acceleration = np.linalg.norm(self.state[2:4] - self.previous_velocity) / self.time_interval
    self.Q[:2, :2] *= 1 + 0.1 * acceleration  # 根据加速度动态调整位置噪声
    self.previous_velocity = self.state[2:4]  # 更新历史速度


# 动态阈值函数
def dynamic_threshold(velocity, size, miss_count):
    """
    根据目标的速度、大小和未匹配次数动态调整匹配阈值。
    """
    base_threshold = 20  # 基础阈值
    speed_factor = np.linalg.norm(velocity) / 10  # 根据速度动态调整
    size_factor = size / 100  # 根据目标尺寸动态调整
    miss_penalty = miss_count * 5  # 未匹配次数的惩罚
    return base_threshold + speed_factor + size_factor - miss_penalty


# 马氏距离计算
def calculate_mahalanobis_distance(prediction, detection, covariance_matrix):
    """
    计算预测位置和检测位置之间的马氏距离。
    """
    covariance_matrix_2d = covariance_matrix[:2, :2]
    diff = prediction - detection
    return distance.mahalanobis(diff, np.zeros_like(diff), np.linalg.inv(covariance_matrix_2d))


# 扩展卡尔曼滤波器类
class ExtendedKalmanFilter:
    def __init__(self, initial_state, time_interval=1):
        """
        初始化扩展卡尔曼滤波器
        :param initial_state: 初始状态 [x, y, vx, vy]
        :param time_interval: 时间间隔
        """
        self.state = initial_state  # [x, y, vx, vy]
        self.P = np.eye(4)  # 协方差矩阵
        self.F = np.eye(4)  # 状态转移矩阵
        self.F[0, 2] = self.F[1, 3] = time_interval  # 时间间隔影响的位置和速度转移
        self.Q = np.eye(4) * 0.01  # 过程噪声
        self.R = np.eye(2) * 0.1  # 观测噪声
        self.previous_velocity = np.zeros(2)  # 初始化历史速度

    def predict(self):
        """
        执行预测步骤，返回预测位置。
        """
        self.optimize_process_noise()  # 动态调整过程噪声
        self.state = np.dot(self.F, self.state)  # 预测下一个状态
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q  # 更新协方差矩阵
        return self.state[:2]  # 返回预测位置 (x, y)

    def update(self, measurement):
        """
        执行更新步骤，结合观测信息更新状态。
        :param measurement: 当前观测数据 (x, y)
        """
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # 观测矩阵
        y = measurement - np.dot(H, self.state)  # 计算残差
        S = np.dot(H, np.dot(self.P, H.T)) + self.R  # 残差协方差
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))  # 卡尔曼增益
        self.state += np.dot(K, y)
        self.P = np.dot(np.eye(4) - np.dot(K, H), self.P)  # 更新协方差矩阵


# 目标跟踪信息
tracked_objects = {}  # 跟踪目标的EKF实例
tracked_objects_info = {}  # 目标信息（速度、大小、计数等）
candidate_objects = {}  # 候选目标


# 目标类：封装目标信息，包括EKF实例、速度、大小等
class Target:
    def __init__(self, obj_id, ekf, initial_detection, reserve_counter=0):
        """
        初始化目标类。
        :param obj_id: 目标ID
        :param ekf: EKF实例
        :param initial_detection: 初始检测位置和大小 [x, y, width, height]
        :param reserve_counter: 保留计数器，未匹配的帧数
        """
        self.obj_id = obj_id
        self.ekf = ekf
        self.velocity = ekf.state[2:4]
        self.size = initial_detection[2] * initial_detection[3]
        self.reserve_counter = reserve_counter
        self.previous_position = initial_detection[:2]
        self.frame_count = 1
        self.P = ekf.P[:2, :2]

    def update_velocity(self, measurement, time_interval=1):
        """
        更新目标速度，通过当前位置和上一位置估算。
        :param measurement: 当前观测位置 [x, y]
        :param time_interval: 时间间隔
        """
        self.velocity = (measurement - self.previous_position) / (time_interval * self.frame_count)
        self.previous_position = measurement
        self.frame_count += 1


# 目标跟踪器类：管理所有目标的预测、匹配和更新等过程
class ObjectTracker:
    def __init__(self, reserve=3, hit=3):
        """
        初始化目标跟踪器类。
        :param reserve: 未匹配目标的最大容忍帧数
        :param hit: 新目标被匹配的最小帧数
        """
        self.tracked_objects = {}  # 当前跟踪目标
        self.tracked_objects_info = {}  # 目标信息
        self.candidate_objects = {}  # 候选目标
        self.reserve = reserve  # 最大容忍帧数
        self.hit = hit  # 最小匹配帧数

    def predict_targets(self):
        """
        对所有跟踪目标进行预测。
        """
        return {obj_id: ekf.predict() for obj_id, ekf in self.tracked_objects.items()}

    def match_targets_with_predictions(self, predictions, current_detections):
        """
        使用匈牙利算法将目标匹配到当前检测。
        :param predictions: 当前所有目标的预测位置
        :param current_detections: 当前检测的位置和大小
        :return: 匹配的目标ID和检测索引
        """
        distance_matrix = np.zeros((len(predictions), len(current_detections)))
        pred_ids = list(predictions.keys())
        for i, pid in enumerate(pred_ids):
            for j, det in enumerate(current_detections):
                covariance_matrix = self.tracked_objects_info[pid].P
                distance_matrix[i, j] = calculate_mahalanobis_distance(predictions[pid], det[:2], covariance_matrix)
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        return [(pred_ids[row], col) for row, col in zip(row_ind, col_ind) if distance_matrix[row, col] < dynamic_threshold(self.tracked_objects_info[pred_ids[row]].velocity, self.tracked_objects_info[pred_ids[row]].size, self.tracked_objects_info[pred_ids[row]].reserve_counter)]

    def update_matched_targets(self, matches, current_detections):
        """
        更新匹配到的目标状态。
        :param matches: 匹配的目标ID和检测索引
        :param current_detections: 当前检测
        """
        for obj_id, det_index in matches:
            measurement = current_detections[det_index][:2]
            self.tracked_objects[obj_id].update(measurement)
            self.tracked_objects_info[obj_id].update_velocity(measurement)

    def handle_unmatched_targets(self, matches):
        """
        处理未匹配的目标。
        :param matches: 匹配的目标
        """
        matched_ids = {match[0] for match in matches}
        for obj_id in list(self.tracked_objects.keys()):
            if obj_id not in matched_ids:
                if obj_id not in self.tracked_objects_info:
                    self.tracked_objects_info[obj_id] = Target(obj_id, self.tracked_objects[obj_id], None)
                self.tracked_objects_info[obj_id].reserve_counter += 1
                if self.tracked_objects_info[obj_id].reserve_counter >= self.reserve:
                    del self.tracked_objects[obj_id]
                    del self.tracked_objects_info[obj_id]

    def add_new_targets(self, current_detections, matches):
        """
        添加新的目标，如果当前检测没有匹配。
        :param current_detections: 当前检测的数据
        :param matches: 已匹配的目标
        """
        unmatched_detections = {i for i in range(len(current_detections))} - {m[1] for m in matches}
        for det_index in unmatched_detections:
            det = current_detections[det_index]
            new_target_id = len(self.tracked_objects) + 1
            ekf = ExtendedKalmanFilter(initial_state=np.array([det[0], det[1], 0, 0]))  # 初始状态假设目标静止
            new_target = Target(new_target_id, ekf, det)
            self.tracked_objects[new_target_id] = ekf
            self.tracked_objects_info[new_target_id] = new_target
