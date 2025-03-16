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