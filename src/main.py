import pandas as pd

from src.performance_metrics import MetricsEvaluator, plot_metrics, plot_mota_motp
from src.tracking_algorithm import run_tracking_objects


def load_gt_data(gt_file):
    gt_data = pd.read_csv(gt_file, sep=',', header=None, usecols=range(9))
    gt_data.columns = ['frame', 'id', 'x1', 'y1', 'w', 'h', 'confidence', 'class', 'visibility']
    return gt_data

def main(gt_file):
    gt_data = load_gt_data(gt_file)

    # 目标跟踪
    tracked_results =  run_tracking_objects(gt_data)

    # 4. 提取性能指标数据
    frames = tracked_results['frames']
    precision = tracked_results['precision']
    recall = tracked_results['recall']
    f1 = tracked_results['f1']
    mota = tracked_results['mota']
    motp = tracked_results['motp']

    # 5. 绘制 Precision, Recall, F1-score 的折线图
    plot_metrics(frames, precision, recall, f1)

    # 6. 绘制 MOTA 和 MOTP 的折线图
    plot_mota_motp(frames, mota, motp)


if __name__ == "__main__":
    gt_file = r"E:\python\MOT\UAV_MOT_LLM_Research\data\MOT17-02-FRCNN\gt\gt.txt"
    main(gt_file)
