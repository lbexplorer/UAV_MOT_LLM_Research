import pandas as pd

from src.performance_metrics import MetricsEvaluator


def load_gt_data(gt_file):
    gt_data = pd.read_csv(gt_file, sep=',', header=None, usecols=range(9))
    gt_data.columns = ['frame', 'id', 'x1', 'y1', 'w', 'h', 'confidence', 'class', 'visibility']
    return gt_data

def main(gt_file):
    gt_data = load_gt_data(gt_file)

    # 目标跟踪
    tracked_results, metrics = track_objects_v3(gt_data)

    # 计算并可视化性能指标
    evaluator = MetricsEvaluator(gt_data)
    frames, precision, recall, f1 = evaluator.calculate_precision_recall_f1(metrics)
    mota, motp = evaluator.calculate_mota_motp(tracked_results)
    evaluator.plot_metrics(frames, precision, recall, f1)
    evaluator.plot_heatmap(frames, precision, recall, f1)
    evaluator.visualize_trajectories(tracked_results)

if __name__ == "__main__":
    gt_file = "path/to/gt.csv"
    main(gt_file)
