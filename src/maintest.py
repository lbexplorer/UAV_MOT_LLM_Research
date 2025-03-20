import logging
import motmetrics as mm

from src.performance_metrics import MOTEvaluator
from src.tracking_algorithm import ObjectTracker


def main(gt_path, output_dir='results'):
    """
    主函数模块，用于运行目标追踪系统并评估性能。
    :param gt_path: ground truth文件路径
    :param output_dir: 输出目录
    """
    # 初始化模块
    loader = MOTLoader()
    tracker = ObjectTracker()
    evaluator = MOTEvaluator()

    # 加载数据
    gt_data = loader.load_gt(gt_path)
    total_frames = max(gt_data.keys())

    # 处理每帧数据
    for frame in range(1, total_frames + 1):
        # 获取当前帧数据
        current_gt = gt_data.get(frame, [])
        detections = loader.convert_to_detections(current_gt)

        # 运行跟踪算法
        track_result = tracker.update_frame(detections)
        track_boxes = [(tid, state) for tid, state in track_result['confirmed']]

        # 更新评估器
        evaluator.update(frame, current_gt, track_boxes)

        # 日志输出
        if frame % 50 == 0:
            logging.info(f"Processed frame {frame}/{total_frames}")

    # 生成评估结果
    evaluator.visualize_results(output_dir)

    # 保存最终结果
    final_metrics = evaluator.get_metrics()
    logging.info("\nFinal Evaluation Metrics:")
    print(mm.io.render_summary(
        final_metrics,
        formatters=evaluator.metrics.formatters,
        namemap=mm.io.motchallenge_metric_names
    ))

    return final_metrics


if __name__ == "__main__":
    # 使用示例
    gt_path = 'path/to/MOT17/gt/gt.txt'  # 修改为实际路径
    main(gt_path)