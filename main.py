# main.py 示例

import logging
import os
import pandas as pd
from datetime import datetime
from src.tracker.enhanced_hungarian import EnhancedHungarianTracker
from src.eval.metrics import MOTEvaluator
from src.data_loader import MOTLoader

def setup_logger(results_dir: str) -> logging.Logger:
    """
    设置日志配置
    :param results_dir: 结果目录路径
    :return: 日志记录器
    """
    # 确保目录存在
    os.makedirs(results_dir, exist_ok=True)
    log_file = os.path.join(results_dir, 'tracking.log')
    
    # 确保之前的处理程序被移除
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 创建新的处理程序
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    console_handler = logging.StreamHandler()
    
    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def create_results_dir() -> str:
    """
    创建带时间戳的结果目录
    :return: 结果目录路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('results', f'eval_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def load_detections(det_path: str, logger: logging.Logger) -> dict:
    """
    加载检测器结果
    :param det_path: 检测结果文件路径
    :param logger: 日志记录器
    :return: 按帧组织的检测结果字典
    """
    try:
        # 读取检测结果文件
        det_data = pd.read_csv(
            det_path,
            sep=',',
            header=None,
            names=['frame', 'id', 'x', 'y', 'w', 'h', 'score', 'class', 'visibility']
        )
        
        # 按帧组织检测结果
        det_frames = {}
        for frame_id, frame_data in det_data.groupby('frame'):
            # 转换为边界框格式 (x1, y1, x2, y2)
            detections = []
            for _, det in frame_data.iterrows():
                x1, y1 = det['x'], det['y']
                x2, y2 = x1 + det['w'], y1 + det['h']
                detections.append((x1, y1, x2, y2))
            det_frames[frame_id] = detections
            
        logger.info(f"Loaded {len(det_frames)} frames of detection data")
        return det_frames
    
    except Exception as e:
        logger.error(f"Error loading detection data: {e}")
        raise

def process_mot_sequence(sequence_path: str, results_dir: str, logger: logging.Logger) -> dict:
    """
    处理MOT序列数据
    :param sequence_path: MOT序列根目录路径
    :param results_dir: 结果保存目录
    :param logger: 日志记录器
    :return: 评估结果
    """
    # 构建相关路径
    gt_path = os.path.join(sequence_path, 'gt', 'gt.txt')
    det_path = os.path.join(sequence_path, 'det', 'det.txt')
    
    # 初始化模块
    tracker = EnhancedHungarianTracker()
    evaluator = MOTEvaluator()
    data_loader = MOTLoader()
    
    try:
        # 加载ground truth数据
        logger.info(f"Loading ground truth data from: {gt_path}")
        data_loader.gt_path = gt_path
        gt_frames = data_loader.load_gt()
        logger.info(f"Loaded {len(gt_frames)} frames of ground truth data")
        
        # 加载检测器结果
        logger.info(f"Loading detection data from: {det_path}")
        det_frames = load_detections(det_path, logger)
        
        # 确保GT和检测结果的帧ID一致
        frame_ids = sorted(set(gt_frames.keys()) & set(det_frames.keys()))
        logger.info(f"Processing {len(frame_ids)} common frames")
        
        # 处理每一帧
        for frame_id in frame_ids:
            # 准备ground truth数据
            frame_gt = gt_frames[frame_id]
            gt_boxes = []
            
            # 转换GT数据格式
            for obj in frame_gt:
                bbox = data_loader.det_to_bbox(obj)
                gt_boxes.append((obj['id'], *bbox))
            
            # 获取检测结果
            det_boxes = det_frames[frame_id]
            
            # 运行跟踪算法
            track_result = tracker.update(frame_id, det_boxes)
            
            # 准备预测结果
            pred_boxes = [(tid, *bbox) for tid, bbox in track_result['confirmed']]
            
            # 更新评估器
            evaluator.update(gt_boxes, pred_boxes)
            
            # 打印进度
            if frame_id % 100 == 0:
                logger.info(f"Frame {frame_id} processed. "
                          f"Ground truth: {len(gt_boxes)}, "
                          f"Detections: {len(det_boxes)}, "
                          f"Tracks: {len(pred_boxes)}")
        
        # 生成评估结果
        logger.info(f"Saving results to: {results_dir}")
        
        # 生成并保存性能趋势图
        logger.info("Generating performance trend plots...")
        evaluator.visualize_trends(results_dir)
        
        # 保存评估报告
        logger.info("Saving evaluation report...")
        evaluator.save_report(results_dir)
        
        # 额外生成详细的性能趋势数据
        metrics_history = evaluator.metrics_history
        plot_detailed_trends(metrics_history, results_dir)
        
        # 获取评估摘要
        summary = evaluator.get_summary()
        
        return {
            'summary': summary,
            'results_dir': results_dir
        }
        
    except Exception as e:
        logger.error(f"Error processing sequence: {str(e)}")
        raise

def plot_detailed_trends(metrics_history: dict, output_dir: str):
    """生成详细的性能趋势图"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 删除 plt.style.use('seaborn')
    # 直接使用seaborn的设置
    sns.set_theme(style="whitegrid")  # 使用seaborn的默认主题
    
    # 创建多个子图
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle('Detailed Tracking Performance Trends', fontsize=16)
    
    # 1. MOTA & MOTP
    axes[0].plot(metrics_history['mota'], label='MOTA', color='blue')
    axes[0].plot(metrics_history['motp'], label='MOTP', color='red')
    axes[0].set_title('MOTA and MOTP over Time')
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Score')
    axes[0].legend()
    
    # 2. False Positives & Negatives
    axes[1].plot(metrics_history['fp'], label='False Positives', color='orange')
    axes[1].plot(metrics_history['fn'], label='False Negatives', color='green')
    axes[1].set_title('Detection Errors over Time')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    
    # 3. ID Switches
    axes[2].plot(metrics_history['idsw'], label='ID Switches', color='purple')
    axes[2].set_title('ID Switches over Time')
    axes[2].set_xlabel('Frame')
    axes[2].set_ylabel('Count')
    axes[2].legend()
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_performance_trends.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main(sequence_path: str):
    """
    主函数
    :param sequence_path: MOT序列根目录路径
    """
    # 创建结果目录
    results_dir = create_results_dir()
    
    # 设置日志
    logger = setup_logger(results_dir)
    logger.info(f"Starting MOT evaluation with sequence: {sequence_path}")
    logger.info("=" * 50)
    
    try:
        # 处理序列
        results = process_mot_sequence(sequence_path, results_dir, logger)
        
        # 打印评估结果
        logger.info("\nEvaluation Results:")
        logger.info("=" * 50)
        
        # 读取并打印评估报告
        report_path = os.path.join(results_dir, 'metrics_report.txt')
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
                logger.info("\n" + report_content)
        else:
            logger.error("Metrics report not found.")
            
        logger.info("=" * 50)
        logger.info(f"Results have been saved to: {results_dir}")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    # 使用完整的序列路径
    sequence_path = r"E:\python\MOT\UAV_MOT_LLM_Research\data\MOT17-02-FRCNN"
    main(sequence_path)