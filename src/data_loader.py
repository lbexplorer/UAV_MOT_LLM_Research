import pandas as pd
import os
from config import Config


class MOTLoader:
    """MOT格式数据加载器"""

    def __init__(self):
        self.gt_path = Config.GT_PATH  # 使用Config中的路径配置
        print("\n")
        print(f"加载GT文件路径: {self.gt_path}")  # 打印路径以调试

        # 检查文件是否存在
        if not os.path.exists(self.gt_path):
            raise FileNotFoundError(f"GT文件未找到: {self.gt_path}")

    def load_gt(self) -> dict:
        """
        加载ground truth数据，按帧组织
        :return: 字典格式 {frame_id: [目标1信息, 目标2信息, ...]}
        """
        try:
            # 读取gt.txt文件，指定分隔符为逗号，读取9列数据
            gt_data = pd.read_csv(
                self.gt_path,
                sep=',',
                header=None,
                usecols=range(9)
            )

            # 设置列名
            gt_data.columns = [
                'frame', 'id', 'x1', 'y1', 'w', 'h',
                'confidence', 'class', 'visibility'
            ]

            # 按帧号分组，转换为字典格式
            gt_dict = {
                frame: group.to_dict('records')
                for frame, group in gt_data.groupby('frame')
            }
            return gt_dict
        except Exception as e:
            raise Exception(f"加载GT数据失败: {str(e)}")

    @staticmethod
    def det_to_xywh(det: dict) -> tuple:
        """
        将检测框信息转换为(x, y, w, h)格式
        :param det: 单个目标信息字典
        :return: (x, y, w, h)
        """
        return (det['x1'], det['y1'], det['w'], det['h'])

    @staticmethod
    def det_to_bbox(det: dict) -> tuple:
        """
        将检测框信息转换为(x1, y1, x2, y2)格式
        :param det: 单个目标信息字典
        :return: (x1, y1, x2, y2)
        """
        x1, y1, w, h = det['x1'], det['y1'], det['w'], det['h']
        return (x1, y1, x1 + w, y1 + h)


def test_data_loader():
    try:
        loader = MOTLoader()
        gt_data = loader.load_gt()

        # 打印前5帧数据
        for frame_id in range(1, 6):
            print(f"Frame {frame_id}:")
            for obj in gt_data.get(frame_id, []):
                print(f"  ID: {obj['id']}, BBox: {loader.det_to_xywh(obj)}")
            print()
    except Exception as e:
        print(f"测试失败: {str(e)}")


if __name__ == "__main__":
    test_data_loader()