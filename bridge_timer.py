#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
桥式动作计时器 - 检测桥式姿势并计时
"""

import numpy as np
import time
from .base_counter import ExerciseCounter
from .pose_utils import PoseUtils
import logging # 添加 logging


class BridgeTimer(ExerciseCounter):
    """桥式动作计时器"""
    
    def __init__(self, confidence_threshold=0.5):
        """初始化桥式动作计时器
        
        Args:
            confidence_threshold: 关键点置信度阈值
        """
        super().__init__(confidence_threshold)
        
        # 桥式动作特定参数
        self.hip_angle_threshold = 160  # 髋部角度阈值（调高一点，更严格）
        self.ideal_hip_angle = 175      # 理想髋部角度（接近伸直）
        # self.min_hold_time = 1.0 # 不需要

        # 评估参数
        self.hip_angle_stability_threshold = 5 # 髋部角度晃动标准差阈值 (度)
        self.hip_height_stability_threshold = 10 # 髋部垂直晃动标准差阈值 (像素)

        # 质量评估权重
        self.w_pose = 0.5       # 平均髋部角度
        self.w_stability_angle = 0.25 # 髋部角度稳定性
        self.w_stability_height = 0.25 # 髋部高度稳定性
        
        # 状态变量
        self.in_bridge_position = False  # 是否处于桥式姿势 (由缓冲区决定)
        self.position_buffer = []  # 姿势缓冲区
        self.buffer_size = 3  # 缓冲区大小
        
        # --- 新增计时状态变量 ---
        self.last_valid_time = None # 上次姿势有效的时间戳
        self.accumulated_valid_time_in_rep = 0.0 # 当前有效姿势阶段累积的时间
        # -----------------------
        
        # 数据可视化相关
        self.visualization_history = [] 
        self.max_history_size = 30  
        
        # 姿势正确性
        self.posture_correct = False # 当前帧是否正确
        
        # 设置为计时模式
        self.is_counting = False
        
        self.logger = logging.getLogger(__name__) # 添加 logger
    
    def process(self, keypoints):
        """处理关键点数据，检测桥式姿势并计时 (重构逻辑)"""
        current_time = time.time()
        old_state = self.state

        # --- 1. 检测与过滤 ---
        person_detected = keypoints is not None and isinstance(keypoints, np.ndarray)
        smoothed_keypoints = None
        if person_detected:
            filtered_keypoints = self.filter_keypoints(keypoints)
            if filtered_keypoints is not None:
                smoothed_keypoints = self.smooth_keypoints(filtered_keypoints)

        # --- 2. 检查当前姿势是否有效 ---
        is_pose_currently_valid = False
        if smoothed_keypoints is not None:
            # 使用 _check_bridge_position 判断瞬时姿势
            self.posture_correct = self._check_bridge_position(smoothed_keypoints)
            
            # 使用缓冲区平滑姿势判断
            self.position_buffer.append(self.posture_correct)
            if len(self.position_buffer) > self.buffer_size:
                self.position_buffer.pop(0)
            is_pose_currently_valid = sum(self.position_buffer) > len(self.position_buffer) / 2
            self.in_bridge_position = is_pose_currently_valid # 更新平滑后的状态
        else:
            # 如果没有检测到点，则姿势无效
            self.posture_correct = False
            self.in_bridge_position = False
            is_pose_currently_valid = False # 确保是 False
        
        # --- 3. 状态机与计时逻辑 (与 PlankTimer 类似) ---
        if is_pose_currently_valid:
            if self.last_valid_time is None:
                # 刚进入有效状态
                self.logger.debug("进入有效桥式状态，开始计时/累积")
                self.last_valid_time = current_time
                self._start_new_rep() # 开始记录数据用于本次有效支撑评估
                self.accumulated_valid_time_in_rep = 0.0 # 重置本次累积时间
                self.state = "开始桥式"
            else:
                # 保持有效状态，累积时间
                delta_time = current_time - self.last_valid_time
                self.duration += delta_time
                self.accumulated_valid_time_in_rep += delta_time
                self.last_valid_time = current_time # 更新时间戳
                self.state = "进行中"
                # 发射实时更新的总时长
                self.time_updated.emit(self.duration)
                
        else: # 当前姿势无效或未检测到人
            if self.last_valid_time is not None:
                # 刚从有效状态退出
                self.logger.debug(f"退出有效桥式状态，本次有效时长: {self.accumulated_valid_time_in_rep:.2f}s")
                self.state = "计时中断"
                # 进行质量评估 (使用 self.current_rep_data 中记录的数据)
                if self.current_rep_data.get('timestamps'):
                    score, feedback = self._calculate_quality()
                    self._update_quality_and_feedback(score, feedback)
                else:
                     self.logger.warning("尝试评估质量，但没有记录到数据")
                
                self.last_valid_time = None # 重置计时锚点
            else:
                # 本来就处于无效状态
                if not person_detected:
                    self.state = "未检测到"
                else:
                    self.state = "等待桥式姿势" # 或 "姿势不标准"
        
        # --- 4. 记录数据 (只要检测到人且在有效计时期间) ---
        if smoothed_keypoints is not None and self.last_valid_time is not None:
            hip_angles = self._get_hip_angles(smoothed_keypoints)
            hip_center = self._get_hip_center(smoothed_keypoints)
            record_data = {}
            avg_hip_angle = None
            if hip_angles:
                 valid_angles = [a for a in hip_angles if a is not None]
                 if valid_angles: 
                     avg_hip_angle = np.mean(valid_angles)
                     record_data['hip'] = avg_hip_angle
            
            if 'hip_center' not in self.current_rep_data['positions']:
                 self.current_rep_data['positions']['hip_center'] = []
            if hip_center is not None:
                self.current_rep_data['positions']['hip_center'].append(hip_center)
            
            # 只有在有效计时期间才记录数据
            self._record_rep_data(current_time, smoothed_keypoints, **record_data)

        # --- 5. 更新可视化数据 --- 
        # 即使不在计时也更新可视化，显示当前状态
        if smoothed_keypoints is not None:
            hip_angles_viz = self._get_hip_angles(smoothed_keypoints)
            left_hip_viz = hip_angles_viz[0] if hip_angles_viz else None
            right_hip_viz = hip_angles_viz[1] if hip_angles_viz and len(hip_angles_viz) > 1 else None
            avg_hip_viz = None
            if hip_angles_viz:
                valid_viz = [a for a in hip_angles_viz if a is not None]
                if valid_viz: avg_hip_viz = np.mean(valid_viz)
            self._add_visualization_data(smoothed_keypoints, left_hip_viz, right_hip_viz, avg_hip_viz, self.state, self.duration)
        else:
            # 如果未检测到人，也发送空数据或状态，让UI知道
            self._add_visualization_data(None, None, None, None, self.state, self.duration)

        # --- 6. 发送状态变更信号 ---
        if old_state != self.state:
            self.state_changed.emit(self.state)

    def _check_bridge_position(self, keypoints):
        """检查是否处于桥式姿势 - 优化逻辑"""
        if keypoints is None: return False
        hip_angles = self._get_hip_angles(keypoints)
        if not hip_angles: return False
        
        left_hip_angle, right_hip_angle = hip_angles
        
        # 检查两侧角度是否有效
        left_valid = left_hip_angle is not None
        right_valid = right_hip_angle is not None
        
        # 如果两侧都无效，则姿势无效
        if not left_valid and not right_valid: return False
        
        # 判断逻辑：
        # 1. 如果两侧都可见，则需要两侧都满足阈值 (更严格的标准桥式)
        # 2. 如果只有一侧可见，则该侧满足阈值即可 (允许遮挡情况)
        if left_valid and right_valid:
            return left_hip_angle >= self.hip_angle_threshold and right_hip_angle >= self.hip_angle_threshold
        elif left_valid:
            return left_hip_angle >= self.hip_angle_threshold
        else: # 只有 right_valid
            return right_hip_angle >= self.hip_angle_threshold

    def _get_hip_angles(self, keypoints):
        """获取髋部角度"""
        if keypoints is None: return []
        left_hip_angle = PoseUtils.calculate_hip_angle(keypoints, 'left')
        right_hip_angle = PoseUtils.calculate_hip_angle(keypoints, 'right')
        if left_hip_angle is None and right_hip_angle is None: return []
        return [left_hip_angle, right_hip_angle]

    def _calculate_quality(self):
        """计算臀桥动作质量 (计时结束后评估整个过程)"""
        feedback = []
        pose_score = 0
        stability_angle_score = 0
        stability_height_score = 0
        
        # --- 姿态评估: 平均髋部角度 ---
        hip_angles = self.current_rep_data.get('angles', {}).get('hip', [])
        valid_hip_angles = [a for a in hip_angles if a is not None]
        if valid_hip_angles:
             avg_hip_angle = np.mean(valid_hip_angles)
             # 评分: 越接近 ideal_hip_angle 越好
             if avg_hip_angle >= self.ideal_hip_angle:
                 pose_score = 100
             elif avg_hip_angle < self.hip_angle_threshold:
                 pose_score = 0
                 feedback.append(f"平均臀部抬起不足 ({avg_hip_angle:.1f}° < {self.hip_angle_threshold}°)")
             else:
                 pose_score = max(0, 100 * (avg_hip_angle - self.hip_angle_threshold) / 
                                      (self.ideal_hip_angle - self.hip_angle_threshold))
             
             # --- 稳定性评估: 髋部角度晃动 ---
             std_dev_angle = np.std(valid_hip_angles)
             if std_dev_angle <= self.hip_angle_stability_threshold:
                 stability_angle_score = 100
             else:
                 stability_angle_score = max(0, 100 - 100 * (std_dev_angle - self.hip_angle_stability_threshold) / self.hip_angle_stability_threshold)
                 feedback.append(f"臀部角度晃动较大 (标准差 {std_dev_angle:.1f}° > {self.hip_angle_stability_threshold}°)")
        else:
             feedback.append("无法评估臀部姿态/角度稳定性 (数据不足)")

        # --- 稳定性评估: 髋部垂直晃动 ---
        hip_centers = self.current_rep_data.get('positions', {}).get('hip_center', [])
        valid_hip_centers = [p for p in hip_centers if p is not None]
        if len(valid_hip_centers) > 1:
             # --- 使用 _calculate_point_variance --- 
             std_dev_x, std_dev_y = self._calculate_point_variance(valid_hip_centers)
             # -----------------------------------
             if std_dev_y <= self.hip_height_stability_threshold:
                 stability_height_score = 100
             else:
                 stability_height_score = max(0, 100 - 100 * (std_dev_y - self.hip_height_stability_threshold) / self.hip_height_stability_threshold)
                 feedback.append(f"臀部高度晃动较大 (标准差 {std_dev_y:.1f} > {self.hip_height_stability_threshold})")
        else:
             feedback.append("无法评估臀部高度稳定性 (数据不足)")

        # --- 综合评分 ---
        total_score = int(self.w_pose * pose_score +
                        self.w_stability_angle * stability_angle_score +
                        self.w_stability_height * stability_height_score)
                        
        self.logger.debug(f"Bridge Quality: Total={total_score}, Pose={pose_score:.1f}, StabAng={stability_angle_score:.1f}, StabH={stability_height_score:.1f}, Feedback={feedback}")
        return total_score, feedback

    def _add_visualization_data(self, keypoints, left_hip_angle, right_hip_angle, avg_hip_angle, state, current_total_duration):
        """添加可视化数据"""
        viz_data = {
            'timestamp': time.time(),
            'left_hip_angle': left_hip_angle,
            'right_hip_angle': right_hip_angle,
            'hip_angle': avg_hip_angle, # 使用平均值
            'state': state,
            'duration': current_total_duration,
            'in_position': self.last_valid_time is not None,
            'posture_correct': self.posture_correct,
            'last_score': self.quality_scores[-1] if self.quality_scores else None,
            'last_feedback': self.feedback_messages[-1] if self.feedback_messages else []
        }
        self.visualization_history.append(viz_data)
        if len(self.visualization_history) > self.max_history_size: self.visualization_history.pop(0)
        # --- 修改信号发送 --- 
        vis_data_for_ui = {
             'hip_angle': avg_hip_angle # UI 主要关注平均髋角
        }
        self.visualization_data.emit(vis_data_for_ui)    # 使用基类定义的信号
        # self.visualization_data.emit(viz_data) # 移除旧信号
        # ------------------
    
    def reset(self):
        """重置计时器状态"""
        super().reset()
        self.in_bridge_position = False
        self.position_buffer = []
        self.posture_correct = False
        self.last_valid_time = None # 新增
        self.accumulated_valid_time_in_rep = 0.0 # 新增
        
    # 添加 _get_hip_center, _calculate_point_variance (如果不存在)
    # 这些通常在 BaseCounter 或需要的地方实现，这里假设 BaseCounter 有
    def _get_hip_center(self, keypoints):
        """计算髋部中心点 (与 SquatCounter 相同)"""
        left_hip = keypoints[PoseUtils.LEFT_HIP]
        right_hip = keypoints[PoseUtils.RIGHT_HIP]
        if left_hip is not None and right_hip is not None:
            # --- 添加长度检查 ---
            left_conf = left_hip[2] if len(left_hip) > 2 else 0.0
            right_conf = right_hip[2] if len(right_hip) > 2 else 0.0
            if left_conf >= self.confidence_threshold and right_conf >= self.confidence_threshold:
            # ------------------
                 return np.mean([left_hip[:2], right_hip[:2]], axis=0)
        elif left_hip is not None:
            # --- 添加长度检查 ---
            left_conf = left_hip[2] if len(left_hip) > 2 else 0.0
            if left_conf >= self.confidence_threshold:
            # ------------------
                 return left_hip[:2]
        elif right_hip is not None:
            # --- 添加长度检查 ---
            right_conf = right_hip[2] if len(right_hip) > 2 else 0.0
            if right_conf >= self.confidence_threshold:
            # ------------------
                 return right_hip[:2]
        return None

    # 这个方法应该在 BaseCounter 中，如果不在，需要添加
    def _calculate_point_variance(self, points):
        """计算点集在x和y方向上的标准差"""
        if not points or len(points) < 2:
            return 0.0, 0.0
        points_array = np.array(points)
        std_dev = np.std(points_array, axis=0)
        return std_dev[0], std_dev[1] # 返回 x 和 y 的标准差 