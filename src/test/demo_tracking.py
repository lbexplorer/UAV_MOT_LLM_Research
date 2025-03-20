# tests/demo_tracking.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tracker.enhanced_hungarian import EnhancedHungarianTracker


def test_tracking():
    tracker = EnhancedHungarianTracker()
    detections = [
        [(100, 100, 150, 150)],  # 帧1
        [(105, 95, 155, 145)],  # 帧2
        [],  # 帧3（丢失）
        [(110, 90, 160, 140)],  # 帧4
    ]
    print("\n")
    for frame_idx, dets in enumerate(detections):
        # 修改：传入帧编号和检测结果
        result = tracker.update(frame_idx, dets)
        print(result)

        confirmed_ids = [k for k, _ in result['confirmed']]
        print(f"Frame {frame_idx}: Confirmed IDs: {confirmed_ids}")
        print("\n")

if __name__ == "__main__":
    test_tracking()