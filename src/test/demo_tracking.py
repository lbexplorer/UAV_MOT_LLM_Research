# tests/demo_tracking.py
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
        result = tracker.update(dets)
        # 修复：直接提取已确认目标的ID
        confirmed_ids = [k for k, _ in result['confirmed']]

        print(f"Frame {frame_idx}: Confirmed IDs: {confirmed_ids}")


if __name__ == "__main__":
    test_tracking()