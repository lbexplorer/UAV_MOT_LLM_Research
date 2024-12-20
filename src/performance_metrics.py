import numpy as np


class MetricsEvaluator:
    def __init__(self, gt_data):
        """
        初始化性能评估器
        :param gt_data: Ground Truth 数据
        """
        self.gt_data = gt_data

    def calculate_precision_recall_f1(self, frame_metrics):
        """
        使用算法输出的 TP、FP、FN 计算 Precision、Recall、F1-Score
        :param frame_metrics: 每帧的性能指标列表，包含 TP、FP、FN、ID Switch 等
        :return: 帧序列、Precision、Recall、F1-score 列表
        """
        precision_list, recall_list, f1_list = [], [], []
        frames = self.gt_data['frame'].unique()

        for frame, metrics in zip(frames, frame_metrics):
            tp = metrics.get("TP", 0)
            fp = metrics.get("FP", 0)
            fn = metrics.get("FN", 0)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        return frames, precision_list, recall_list, f1_list

    def calculate_git (self, tracked_results):
        """
        计算 MOTA 和 MOTP 性能指标
        :param tracked_results: 跟踪结果列表 [(frame, id, x1, y1), ...]
        :return: MOTA 和 MOTP 值
        """
        # 获取 Ground Truth 数据
        gt_data = self.gt_data
        total_frames = len(gt_data['frame'].unique())

        # 计算 MOTA
        mota = 0
        total_tp, total_fp, total_fn, total_id_switches = 0, 0, 0, 0

        # 计算每帧的指标
        for frame in gt_data['frame'].unique():
            frame_gt = gt_data[gt_data['frame'] == frame]
            frame_results = [result for result in tracked_results if result[0] == frame]

            tp, fp, fn, id_switches = self.calculate_frame_metrics(frame_gt, frame_results)
            mota += tp / (tp + fp + fn + id_switches)  # MOTA公式

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_id_switches += id_switches

        mota /= total_frames  # 平均 MOTA

        # 计算 MOTP
        motp = 0
        total_distance = 0
        total_matches = 0
        for frame in gt_data['frame'].unique():
            frame_gt = gt_data[gt_data['frame'] == frame]
            frame_results = [result for result in tracked_results if result[0] == frame]

            for gt in frame_gt.itertuples():
                matching_result = self.find_closest_match(gt, frame_results)
                if matching_result:
                    distance = np.linalg.norm([gt.x1 - matching_result[2], gt.y1 - matching_result[3]])
                    total_distance += distance
                    total_matches += 1

        motp = total_distance / total_matches if total_matches > 0 else 0

        return mota, motp

    def calculate_frame_metrics(self, frame_gt, frame_results):
        """
        计算单帧的 TP、FP、FN 和 ID Switches
        :param frame_gt: Ground Truth 数据
        :param frame_results: 预测结果
        :return: TP、FP、FN、ID Switches
        """
        tp, fp, fn, id_switches = 0, 0, 0, 0
        matched_ids = set()

        # 计算 TP 和 FP
        for result in frame_results:
            matched_gt = self.find_matching_gt(result, frame_gt)
            if matched_gt:
                tp += 1
                matched_ids.add(matched_gt.id)
            else:
                fp += 1

        # 计算 FN
        fn = len(frame_gt) - len(matched_ids)

        # 计算 ID Switches
        id_switches = self.calculate_id_switches(frame_results, frame_gt)

        return tp, fp, fn, id_switches

    def find_matching_gt(self, result, frame_gt):
        """
        寻找与预测结果匹配的 GT（根据距离等规则）
        :param result: 预测结果 (frame, id, x1, y1)
        :param frame_gt: Ground Truth 数据
        :return: 匹配的 GT 行，若没有匹配返回 None
        """
        for gt in frame_gt.itertuples():
            if np.linalg.norm([result[2] - gt.x1, result[3] - gt.y1]) < 50:  # 假设匹配的阈值为 50
                return gt
        return None

    def calculate_id_switches(self, frame_results, frame_gt):
        """
        计算 ID Switches（目标身份的切换）
        :param frame_results: 预测结果列表
        :param frame_gt: Ground Truth 数据
        :return: ID Switches 数量
        """
        # 实际情况可能需要跟踪每个目标的状态和历史轨迹来判断 ID Switches
        return 0  # 这里是一个简化示例，实际上需要根据 ID 变化来计算
