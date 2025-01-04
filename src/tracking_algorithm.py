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


def calculate_mahalanobis_distance(prediction, detection, covariance_matrix):
    covariance_matrix_2d = covariance_matrix[:2, :2]
    diff = prediction - detection
    return distance.mahalanobis(diff, np.zeros_like(diff), np.linalg.inv(covariance_matrix_2d))


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


class Target:
    def __init__(self, obj_id, ekf):
        self.obj_id = obj_id
        self.ekf = ekf
        self.miss_count = 0


class TargetMatcher:
    def match(self, predictions, current_detections):
        distance_matrix = np.zeros((len(predictions), len(current_detections)))
        pred_ids = list(predictions.keys())
        for i, pid in enumerate(pred_ids):
            for j, det in enumerate(current_detections):
                covariance_matrix = predictions[pid].ekf.P
                distance_matrix[i, j] = calculate_mahalanobis_distance(
                    predictions[pid].ekf.state[:2], det[:2], covariance_matrix
                )
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        return [(pred_ids[row], col) for row, col in zip(row_ind, col_ind)]


class ObjectTracker:
    def __init__(self, reserve=3):
        self.tracked_objects = {}
        self.reserve = reserve
        self.matcher = TargetMatcher()

    def predict_targets(self):
        return {obj_id: obj for obj_id, obj in self.tracked_objects.items()}

    def update_targets(self, matches, current_detections):
        for obj_id, det_index in matches:
            detection = current_detections[det_index]
            self.tracked_objects[obj_id].ekf.update(detection[:2])

    def add_new_targets(self, unmatched_detections):
        for det in unmatched_detections:
            obj_id = len(self.tracked_objects) + 1
            ekf = ExtendedKalmanFilter(initial_state=[det[0], det[1], 0, 0])
            self.tracked_objects[obj_id] = Target(obj_id, ekf)

    def remove_lost_targets(self):
        to_remove = [obj_id for obj_id, obj in self.tracked_objects.items() if obj.miss_count >= self.reserve]
        for obj_id in to_remove:
            del self.tracked_objects[obj_id]


def run_tracking_objects(gt_data):
    evaluator = MetricsEvaluator(gt_data)
    tracker = ObjectTracker()
    frames = sorted(gt_data['frame'].unique())
    metrics = {"frames": [], "precision": [], "recall": [], "f1": [], "mota": [], "motp": []}

    for frame in frames:
        frame_gt = gt_data[gt_data["frame"] == frame]
        detections = frame_gt[["x1", "y1", "w", "h"]].values

        predictions = tracker.predict_targets()
        matches = tracker.matcher.match(predictions, detections)

        unmatched_detections = [d for i, d in enumerate(detections) if i not in [m[1] for m in matches]]
        tracker.update_targets(matches, detections)
        tracker.add_new_targets(unmatched_detections)
        tracker.remove_lost_targets()

        frame_metrics = evaluator.calculate_frame_metrics(frame_gt, tracker.tracked_objects)
        for k, v in frame_metrics.items():
            metrics[k].append(v)

    return metrics
