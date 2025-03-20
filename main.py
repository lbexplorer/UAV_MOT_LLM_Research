# main.py 示例

import logging
import os
import numpy as np
from src.tracker.enhanced_hungarian import EnhancedHungarianTracker
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
logger = logging.getLogger(__name__)

def generate_synthetic_data(num_frames=100, num_objects=5):
    """生成合成数据用于测试"""
    gt_data = {}
    for frame_id in range(num_frames):
        frame_objs = []
        for obj_id in range(num_objects):
            # 生成随机运动轨迹
            x = 100 + frame_id * 2 + np.random.normal(0, 1)
            y = 100 + np.sin(frame_id * 0.1) * 20 + np.random.normal(0, 1)
            w = 30 + np.random.normal(0, 2)
            h = 30 + np.random.normal(0, 2)
            frame_objs.append({
                'id': obj_id,
                'bbox': [x, y, x + w, y + h]
            })
        gt_data[frame_id] = frame_objs
    return gt_data

def main():
    # 初始化模块
    tracker = EnhancedHungarianTracker()
    evaluator = MOTEvaluator()

    # 生成或加载数据
    gt_data = generate_synthetic_data(num_frames=50, num_objects=3)
    
    # 处理每一帧
    for frame_id, frame_objs in gt_data.items():
        # 准备ground truth数据
        gt_boxes = [(obj['id'], *obj['bbox']) for obj in frame_objs]
        
        # 模拟检测结果（添加一些噪声和漏检）
        detections = []
        for obj in frame_objs:
            if np.random.random() > 0.1:  # 90%的检测率
                # 添加随机噪声
                noise = np.random.normal(0, 2, 4)
                bbox = np.array(obj['bbox']) + noise
                detections.append(bbox.tolist())

        # 运行跟踪算法
        track_result = tracker.update(frame_id, detections)
        
        # 准备预测结果
        pred_boxes = [(tid, *bbox) for tid, bbox in track_result['confirmed']]
        
        # 更新评估器
        evaluator.update(gt_boxes, pred_boxes)

        # 打印进度
        logger.info(f"Frame {frame_id} processed. "
                   f"Ground truth: {len(gt_boxes)}, "
                   f"Detections: {len(detections)}, "
                   f"Tracks: {len(pred_boxes)}")

    # 输出评估结果
    evaluator.visualize_trends()
    evaluator.save_report()
    
    # 读取并打印评估报告
    report_path = 'results/metrics_report.txt'
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            logger.info("\nEvaluation Results:")
            logger.info(f.read())
    else:
        logger.error("Metrics report not found.")

if __name__ == "__main__":
    main()