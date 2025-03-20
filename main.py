# main.py 示例

import logging
import os
import numpy as np
import json
from datetime import datetime
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

def save_test_data(gt_frames, det_frames, output_dir='test_data'):
    """
    保存测试数据集到文件
    :param gt_frames: ground truth数据
    :param det_frames: 检测数据
    :param output_dir: 输出目录
    :return: 保存的文件路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 准备数据
    data = {
        'timestamp': timestamp,
        'ground_truth': gt_frames,
        'detections': det_frames
    }
    
    # 保存为JSON文件
    output_file = os.path.join(output_dir, f'test_data_{timestamp}.json')
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Test data saved to: {output_file}")
    return output_file

def load_test_data(file_path):
    """
    从文件加载测试数据集
    :param file_path: 数据文件路径
    :return: ground truth数据和检测数据
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded test data from: {file_path}")
    logger.info(f"Data timestamp: {data['timestamp']}")
    return data['ground_truth'], data['detections']

def generate_test_data(num_frames=20, num_objects=3):
    """
    生成测试数据集
    :param num_frames: 帧数
    :param num_objects: 每帧的目标数
    :return: ground truth数据和检测数据
    """
    gt_frames = {}
    det_frames = {}
    
    # 生成目标轨迹
    trajectories = []
    for obj_id in range(num_objects):
        # 为每个目标生成一个轨迹
        x = 100 + np.random.normal(0, 10)  # 初始x位置
        y = 100 + np.random.normal(0, 10)  # 初始y位置
        vx = np.random.normal(2, 0.5)      # x方向速度
        vy = np.random.normal(0, 0.5)      # y方向速度
        w = 30 + np.random.normal(0, 2)    # 宽度
        h = 30 + np.random.normal(0, 2)    # 高度
        trajectories.append({
            'id': obj_id,
            'x': x, 'y': y, 'vx': vx, 'vy': vy,
            'w': w, 'h': h
        })
    
    # 生成每一帧的数据
    for frame_id in range(num_frames):
        # 生成ground truth数据
        frame_gt = []
        for traj in trajectories:
            # 更新位置
            traj['x'] += traj['vx']
            traj['y'] += traj['vy']
            
            # 添加一些随机扰动
            noise_x = np.random.normal(0, 0.5)
            noise_y = np.random.normal(0, 0.5)
            
            frame_gt.append({
                'id': traj['id'],
                'bbox': [
                    traj['x'] + noise_x,
                    traj['y'] + noise_y,
                    traj['x'] + traj['w'] + noise_x,
                    traj['y'] + traj['h'] + noise_y
                ]
            })
        
        # 生成检测数据（模拟检测器的不完美性）
        frame_det = []
        for gt in frame_gt:
            if np.random.random() > 0.1:  # 90%的检测率
                # 添加检测噪声
                noise = np.random.normal(0, 2, 4)
                det_bbox = np.array(gt['bbox']) + noise
                frame_det.append(det_bbox.tolist())
        
        gt_frames[frame_id] = frame_gt
        det_frames[frame_id] = frame_det
    
    return gt_frames, det_frames

def main():
    # 初始化模块
    tracker = EnhancedHungarianTracker()
    evaluator = MOTEvaluator()

    # 生成测试数据
    logger.info("Generating test dataset...")
    gt_frames, det_frames = generate_test_data(num_frames=20, num_objects=3)
    logger.info(f"Generated {len(gt_frames)} frames with {len(gt_frames[0])} objects per frame")
    
    # 保存测试数据
    test_data_file = save_test_data(gt_frames, det_frames)
    
    # 处理每一帧
    for frame_id in sorted(gt_frames.keys()):
        # 准备ground truth数据
        gt_boxes = [(obj['id'], *obj['bbox']) for obj in gt_frames[frame_id]]
        
        # 获取检测结果
        detections = det_frames[frame_id]

        # 运行跟踪算法
        track_result = tracker.update(frame_id, detections)
        
        # 准备预测结果
        pred_boxes = [(tid, *bbox) for tid, bbox in track_result['confirmed']]
        
        # 更新评估器
        evaluator.update(gt_boxes, pred_boxes)

        # 打印进度
        if frame_id % 5 == 0:  # 每5帧打印一次进度
            logger.info(f"Frame {frame_id} processed. "
                       f"Ground truth: {len(gt_boxes)}, "
                       f"Detections: {len(detections)}, "
                       f"Tracks: {len(pred_boxes)}")

    # 输出评估结果
    logger.info("Generating evaluation results...")
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