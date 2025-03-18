# src/tracker/kalman_filter.py
import numpy as np


class ExtendedKalmanFilter:
    """扩展卡尔曼滤波器（EKF），用于目标状态预测"""

    def __init__(self, initial_state, time_interval=1):
        """
        :param initial_state: 初始状态 [x, y, vx, vy]
        :param time_interval: 时间间隔（单位：秒）
        """
        self.state = np.array(initial_state, dtype=np.float64).reshape(4, 1)
        self.P = np.eye(4)  # 状态协方差矩阵
        self.F = np.array([  # 状态转移矩阵
            [1, 0, time_interval, 0],
            [0, 1, 0, time_interval],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.Q = np.diag([0.1, 0.1, 0.5, 0.5])  # 过程噪声
        self.R = np.diag([0.5, 0.5])  # 观测噪声

    def predict(self):
        """预测下一状态"""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2].flatten()

    def update(self, measurement):
        """根据观测更新状态"""
        # 确保 measurement 是 (2, 1) 形状
        measurement = np.array(measurement, dtype=np.float64).reshape(2, 1)

        # 观测矩阵
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # 计算残差
        y = measurement - H @ self.state

        # 计算卡尔曼增益
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # 更新状态
        self.state += K @ y

        # 更新协方差矩阵
        self.P = (np.eye(4) - K @ H) @ self.P