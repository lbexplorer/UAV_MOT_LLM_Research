# main.py 示例

import logging
import os

from src.data_loader import MOTLoader
from src.tracker.hungarian import HungarianTracker
from src.eval.metrics import MOTEvaluator
# 配置日志记录
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO，记录所有 INFO 级别及以上的日志
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('tracking.log', mode='w')  # 输出到文件
    ]
)
def main():
    # 初始化模块
    loader = MOTLoader()
    tracker = HungarianTracker(iou_threshold=0.5)
    evaluator = MOTEvaluator()

    # 加载数据
    gt_data = loader.load_gt()

    for frame_id, frame_objs in gt_data.items():
        # 转换检测框格式
        gt_boxes = [(obj['id'], *loader.det_to_bbox(obj)) for obj in frame_objs]
        detections = [loader.det_to_bbox(obj) for obj in frame_objs]

        # 运行跟踪算法
        track_result = tracker.update(detections)
        pred_boxes = [(tid, *detections[det_idx]) for tid, det_idx in track_result['matches']]

        # 更新评估器
        evaluator.update(gt_boxes, pred_boxes)

        # 打印进度
        print(f"Frame {frame_id} processed. Tracks: {len(pred_boxes)}")

    # 输出评估结果
    evaluator.visualize_trends()
    evaluator.save_report()
    print("Evaluation completed.")

    # 读取并打印 metrics_report.txt 内容
    report_path = 'results/metrics_report.txt'
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            print(f.read())
    else:
        print("Metrics report not found.")


if __name__ == "__main__":
    main()