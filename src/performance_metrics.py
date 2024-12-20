import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

class MetricsEvaluator:
    def __init__(self, gt_data):
        self.gt_data = gt_data

    def calculate_precision_recall_f1(self, frame_metrics):
        precision_list, recall_list, f1_list = [], [], []
        frames = self.gt_data['frame'].unique()

        for frame, metrics in zip(frames, frame_metrics):
            tp = metrics.get("TP", 0)
            fp = metrics.get("FP", 0)
            fn = metrics.get("FN", 0)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        return frames, precision_list, recall_list, f1_list

    def calculate_mota_motp(self, tracked_results):
        # MOTA 和 MOTP 计算
        mota, motp = 0.9, 0.8  # 示例值
        return mota, motp

    def plot_metrics(self, frames, precision, recall, f1):
        plt.figure(figsize=(12, 6))
        plt.plot(frames, precision, label='Precision', marker='o', color='blue')
        plt.plot(frames, recall, label='Recall', marker='o', color='green')
        plt.plot(frames, f1, label='F1-score', marker='o', color='red')
        plt.title('Evaluation Metrics Over Frames')
        plt.xlabel('Frame Number')
        plt.ylabel('Score')
        plt.xticks(frames[::50])
        plt.ylim(0, 1)
        plt.legend()
        plt.grid()
        plt.show()

    def plot_heatmap(self, frames, precision, recall, f1):
        metrics_data = np.array([precision, recall, f1])
        metrics_df = pd.DataFrame(metrics_data, index=['Precision', 'Recall', 'F1-score'], columns=frames)

        plt.figure(figsize=(12, 8))
        sns.heatmap(metrics_df, cmap="YlGnBu", cbar_kws={'label': 'Score'}, annot=False, linewidths=0.5, xticklabels=10, yticklabels=True)
        plt.title('Evaluation Metrics Heatmap', fontsize=16)
        plt.xlabel('Frame Number', fontsize=14)
        plt.ylabel('Metrics', fontsize=14)
        plt.show()

    def visualize_trajectories(self, tracked_results):
        for obj_id in set(result[1] for result in tracked_results):
            tracked_points = [(res[0], res[2]) for res in tracked_results if res[1] == obj_id]
            gt_points = self.gt_data[self.gt_data['id'] == obj_id][['frame', 'x1', 'y1']].values
            plt.plot([p[0] for p in tracked_points], [p[1] for p in tracked_points], label=f"Tracked {obj_id}")
            plt.plot(gt_points[:, 0], gt_points[:, 1], '--', label=f"GT {obj_id}")
        plt.legend()
        plt.show()
