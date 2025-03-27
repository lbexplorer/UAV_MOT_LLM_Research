"""
上下文构建工具类
"""
from typing import Dict, List, Any
import numpy as np

class ContextBuilder:
    """构建LLM所需的上下文信息"""
    
    @staticmethod
    def build_frame_context(frame_id: int,
                          detections: List[List[float]],
                          traditional_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建当前帧上下文
        
        Args:
            frame_id: 帧ID
            detections: 检测结果列表
            traditional_result: 传统跟踪器结果
            
        Returns:
            包含当前帧信息的字典
        """
        return {
            "frame_id": frame_id,
            "detections": [
                {
                    "bbox": det,
                    "center": [(det[0] + det[2])/2, (det[1] + det[3])/2]
                }
                for det in detections
            ],
            "traditional_result": traditional_result,
            "statistics": {
                "num_detections": len(detections),
                "num_tracks": len(traditional_result.get("confirmed", [])),
                "frame_area": ContextBuilder._calculate_frame_area(detections)
            }
        }
    
    @staticmethod
    def build_scene_context(current_frame: Dict[str, Any],
                          history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        构建场景上下文
        
        Args:
            current_frame: 当前帧信息
            history: 历史帧信息
            
        Returns:
            场景上下文信息
        """
        return {
            "current_state": current_frame,
            "motion_patterns": ContextBuilder._analyze_motion_patterns(history),
            "density_map": ContextBuilder._calculate_density_map(current_frame, history),
            "interaction_zones": ContextBuilder._identify_interaction_zones(current_frame)
        }
    
    @staticmethod
    def _calculate_frame_area(detections: List[List[float]]) -> float:
        """计算帧区域"""
        if not detections:
            return 0.0
        
        all_coords = np.array(detections)
        x_min = np.min(all_coords[:, [0, 2]])
        x_max = np.max(all_coords[:, [0, 2]])
        y_min = np.min(all_coords[:, [1, 3]])
        y_max = np.max(all_coords[:, [1, 3]])
        
        return (x_max - x_min) * (y_max - y_min)
    
    @staticmethod
    def _analyze_motion_patterns(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分析运动模式"""
        patterns = []
        if len(history) < 2:
            return patterns
            
        # 分析最近几帧的运动趋势
        for track_id in set([t["id"] for t in history[-1].get("tracks", [])]):
            track_history = []
            for frame in history:
                for track in frame.get("tracks", []):
                    if track["id"] == track_id:
                        track_history.append(track)
            
            if len(track_history) >= 2:
                # 计算速度和方向
                recent_track = track_history[-2:]
                dx = recent_track[1]["center"][0] - recent_track[0]["center"][0]
                dy = recent_track[1]["center"][1] - recent_track[0]["center"][1]
                speed = np.sqrt(dx*dx + dy*dy)
                angle = np.arctan2(dy, dx)
                
                patterns.append({
                    "track_id": track_id,
                    "speed": speed,
                    "direction": angle,
                    "pattern_type": ContextBuilder._classify_motion_pattern(speed, angle)
                })
        
        return patterns
    
    @staticmethod
    def _calculate_density_map(current_frame: Dict[str, Any],
                             history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算密度图"""
        # 使用当前帧和历史帧的目标位置计算密度
        all_positions = []
        
        # 添加当前帧的位置
        for det in current_frame.get("detections", []):
            all_positions.append(det["center"])
        
        # 添加历史帧的位置（带衰减权重）
        for i, frame in enumerate(reversed(history[-5:])):
            weight = 0.8 ** (i + 1)  # 指数衰减权重
            for det in frame.get("detections", []):
                all_positions.append(det["center"])
        
        if not all_positions:
            return {"density": 0.0, "hotspots": []}
            
        # 计算密度
        positions = np.array(all_positions)
        density = len(positions) / ContextBuilder._calculate_frame_area(current_frame.get("detections", []))
        
        # 识别热点区域
        hotspots = []
        if len(positions) > 0:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=50, min_samples=3).fit(positions)
            unique_labels = set(clustering.labels_)
            
            for label in unique_labels:
                if label != -1:  # 排除噪声点
                    mask = clustering.labels_ == label
                    cluster_points = positions[mask]
                    hotspots.append({
                        "center": cluster_points.mean(axis=0).tolist(),
                        "radius": np.max(np.linalg.norm(cluster_points - cluster_points.mean(axis=0), axis=1)),
                        "num_points": int(mask.sum())
                    })
        
        return {
            "density": float(density),
            "hotspots": hotspots
        }
    
    @staticmethod
    def _identify_interaction_zones(current_frame: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别交互区域"""
        zones = []
        detections = current_frame.get("detections", [])
        
        if len(detections) < 2:
            return zones
            
        # 计算两两目标之间的距离
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                det1 = detections[i]
                det2 = detections[j]
                
                # 计算两个目标中心点之间的距离
                center1 = det1["center"]
                center2 = det2["center"]
                distance = np.sqrt(
                    (center1[0] - center2[0])**2 +
                    (center1[1] - center2[1])**2
                )
                
                # 如果距离小于阈值，认为存在交互
                if distance < 100:  # 可调整的阈值
                    zones.append({
                        "objects": [i, j],
                        "center": [(center1[0] + center2[0])/2,
                                 (center1[1] + center2[1])/2],
                        "distance": distance,
                        "interaction_type": "close_proximity"
                    })
        
        return zones
    
    @staticmethod
    def _classify_motion_pattern(speed: float, angle: float) -> str:
        """根据速度和方向分类运动模式"""
        if speed < 1:
            return "stationary"
        elif speed < 5:
            return "slow_moving"
        elif speed < 10:
            return "normal_moving"
        else:
            return "fast_moving" 