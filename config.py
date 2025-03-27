import os
# 获取项目根目录


ROOT_DIR= os.path.dirname(os.path.abspath(__file__))

class Config:
    # 数据集路径配置
    DATA_DIR = os.path.join(ROOT_DIR, 'data')  # 数据目录
    MOT17_ROOT = os.path.join(DATA_DIR, 'MOT17-02-FRCNN')  # MOT17数据集目录
    GT_PATH = os.path.join(MOT17_ROOT, 'gt', 'gt2.txt')  # GT文件路径

    # 输出目录
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'results')  # 结果输出目录

    # 日志配置
    LOG_FILE = os.path.join(ROOT_DIR, 'tracking.log')  # 日志文件路径

def check_path(description, path, check_type):
    """通用路径检查函数"""
    exists = os.path.exists(path) if check_type == "dir" else os.path.isfile(path)
    print(f"{description}: {path} -> {'存在 ✅' if exists else '不存在 ❌'}")
    return exists

if __name__ == '__main__':
    print("====== 配置路径测试 ======")

    # 1. 检查数据集目录是否存在
    check_path("数据集目录", Config.MOT17_ROOT, "dir")

    # 2. 检查 GT 文件是否存在
    check_path("GT 文件", Config.GT_PATH, "file")

    # 3. 检查输出目录是否可写
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)  # 确保目录存在
    test_file = os.path.join(Config.OUTPUT_DIR, 'test_write.txt')

    try:
        with open(test_file, 'w') as f:
            f.write('test')
        print(f"输出目录: {Config.OUTPUT_DIR} -> 可写 ✅")
    except Exception as e:
        print(f"输出目录: {Config.OUTPUT_DIR} -> 不可写 ❌ ({e})")
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)  # 清理测试文件

    print("====== 测试完成 ======")

"""
配置文件
"""
from typing import Dict, Any

# 数据预处理配置
PREPROCESSOR_CONFIG = {
    'image_size': (640, 640),
    'normalize': True,
    'augmentation': {
        'enabled': True,
        'flip_prob': 0.5,
        'brightness_prob': 0.5,
        'brightness_range': (0.8, 1.2),
        'contrast_range': (-10, 10)
    }
}

# 可视化配置
VISUALIZER_CONFIG = {
    'save_images': True,
    'save_video': True,
    'save_trajectories': True,
    'fps': 30,
    'output_dir': 'output/visualization',
    'colors': {
        'detection': (0, 255, 0),  # 绿色
        'track': None,  # 自动生成
        'text': (255, 255, 255)  # 白色
    }
}

# 跟踪器配置
TRACKER_CONFIG = {
    'base_threshold': 0.5,
    'confirm_frames': 3,
    'reserve_frames': 3,
    'max_age': 30,
    'min_hits': 3,
    'iou_threshold': 0.3,
    'max_distance': 100,
    'motion_weight': 0.7,
    'appearance_weight': 0.3
}

# 评估配置
EVALUATOR_CONFIG = {
    'metrics': [
        'MOTA',
        'MOTP',
        'IDF1',
        'MT',
        'ML',
        'FP',
        'FN',
        'IDSW'
    ],
    'output_dir': 'output/evaluation',
    'save_plots': True,
    'plot_interval': 10
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/tracking.log',
    'console': True
}

# 主配置
CONFIG: Dict[str, Any] = {
    'preprocessor': PREPROCESSOR_CONFIG,
    'visualizer': VISUALIZER_CONFIG,
    'tracker': TRACKER_CONFIG,
    'evaluator': EVALUATOR_CONFIG,
    'logging': LOGGING_CONFIG,
    'data_dir': 'data',
    'output_dir': 'output',
    'model_dir': 'models',
    'device': 'cuda'  # 或 'cpu'
}