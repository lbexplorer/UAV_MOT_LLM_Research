import pandas as pd
import logging

from src.performance_metrics import plot_metrics, plot_mota_motp
from src.tracking_algorithm import run_tracking_objects

# 配置日志记录器
logging.basicConfig(level=logging.INFO,  # 设置日志级别
                    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
                    handlers=[logging.StreamHandler()])  # 默认输出到控制台


def load_gt_data(gt_file):
    # 读取 CSV 文件
    gt_data_df = pd.read_csv(gt_file, sep=',', header=None,
                             names=['frame', 'id', 'x1', 'y1', 'w', 'h', 'confidence', 'class', 'visibility'])

    # 将 DataFrame 转换为字典列表
    gt_data = gt_data_df.to_dict(orient='records')

    return gt_data
def main(gt_file):
    gt_data = load_gt_data(gt_file)
    print(gt_data)
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
    gt_file = r"E:\python\MOT\UAV_MOT_LLM_Research\data\MOT17-02-FRCNN\gt\gt2.txt"
    main(gt_file)

# 运行目标追踪的主函数
