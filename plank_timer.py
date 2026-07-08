#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
平板支撑计时器 - 检测平板支撑姿势并计时
"""

import numpy as np
import time
from .base_counter import ExerciseCounter
from .pose_utils import PoseUtils
import logging # 添加 logging


class PlankTimer(ExerciseCounter):
    """平板支撑计时器"""
    
    def __init__(self, confidence_threshold=0.5):
        """初始化平板支撑计时器
        
        Args:
            confidence_threshold: 关键点置信度阈值
        """
        super().__init__(confidence_threshold)
        
        # 平板支撑特定参数
        self.torso_angle_threshold = 20  # 躯干与水平线的最大角度偏差阈值
        self.ideal_torso_horizontal_angle = 5 # 理想躯干与水平线偏差 (越小越好)
        self.elbow_angle_threshold = 100  # 肘部弯曲的最大角度阈值 (标准姿势通常是90度左右)
        self.ideal_elbow_angle = 90       # 理想肘部角度
        # self.min_hold_time = 1.0 # 基类有 duration, 这个似乎不需要
        
        # 稳定性评估参数
        self.hip_stability_threshold = 10 # 髋部垂直方向晃动标准差阈值 (像素)
        self.torso_angle_stability_threshold = 5 # 躯干水平角度变化标准差阈值 (度)

        # 质量评估权重
        self.w_pose_torso = 0.4
        self.w_pose_elbow = 0.2
        self.w_stability_hip = 0.2
        self.w_stability_torso = 0.2
        
        # 状态变量
        self.in_plank_position = False  # 是否处于平板支撑姿势
        self.position_buffer = []  # 姿势缓冲区
        self.buffer_size = 3  # 缓冲区大小
        
        # --- 新增计时状态变量 ---
        self.last_valid_time = None # 上次姿势有效的时间戳
        self.accumulated_valid_time_in_rep = 0.0 # 当前有效姿势阶段累积的时间（用于反馈或调试）
        # -----------------------
        
        # 数据可视化相关
        self.visualization_history = []  # 可视化历史数据
        self.max_history_size = 30  # 最大历史记录数
        
        # 姿势正确性
        self.posture_correct = False
        
        # 设置为计时模式
        self.is_counting = False
        
        # 累加计时相关变量 (这个逻辑可能需要调整，评估应在计时结束时进行)
        # self.last_increment_time = 0  # 上次增加计时的时间
        # self.timer_interval = 1.0  # 计时间隔，每秒加1
        # self.is_timing = False # 标记是否正在计时 (移除)
        
        self.logger = logging.getLogger(__name__) # 添加 logger
    
    def process(self, keypoints):
        """处理关键点数据，检测平板支撑并计时 (重构逻辑)"""
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
            is_pose_currently_valid = self._check_plank_position(smoothed_keypoints)
            
            # 使用缓冲区平滑姿势判断 (可选，如果需要更强的抗抖动性)
            # self.position_buffer.append(is_pose_currently_valid)
            # if len(self.position_buffer) > self.buffer_size:
            #     self.position_buffer.pop(0)
            # is_pose_currently_valid = sum(self.position_buffer) > len(self.position_buffer) / 2
        
        # --- 3. 状态机与计时逻辑 ---
        if is_pose_currently_valid:
            if self.last_valid_time is None:
                # 刚进入有效状态
                self.logger.debug("进入有效平板支撑状态，开始计时/累积")
                self.last_valid_time = current_time
                self._start_new_rep() # 开始记录数据用于本次有效支撑评估
                self.accumulated_valid_time_in_rep = 0.0 # 重置本次累积时间
                self.state = "开始支撑"
            else:
                # 保持有效状态，累积时间
                delta_time = current_time - self.last_valid_time
                self.duration += delta_time
                self.accumulated_valid_time_in_rep += delta_time
                self.last_valid_time = current_time # 更新时间戳
                self.state = "进行中"
                # 发射实时更新的总时长
                self.time_updated.emit(self.duration)
                
                # 记录数据用于质量评估
                # (移到下方公共部分记录)
                
        else: # 当前姿势无效或未检测到人
            if self.last_valid_time is not None:
                # 刚从有效状态退出
                self.logger.debug(f"退出有效平板支撑状态，本次有效时长: {self.accumulated_valid_time_in_rep:.2f}s")
                self.state = "计时中断"
                # 进行质量评估 (使用 self.current_rep_data 中记录的数据)
                if self.current_rep_data.get('timestamps'):
                    score, feedback = self._calculate_quality()
                    self._update_quality_and_feedback(score, feedback)
                else:
                     self.logger.warning("尝试评估质量，但没有记录到数据")
                
                self.last_valid_time = None # 重置计时锚点
                # self._start_new_rep() # 不需要立即清空，下次进入有效状态时会清空
            else:
                # 本来就处于无效状态
                if not person_detected:
                    self.state = "未检测到"
                else:
                    self.state = "等待平板支撑姿势" # 或 "姿势不标准"
        
        # --- 4. 记录数据 (只要检测到人就记录，供评估时筛选) ---
        if smoothed_keypoints is not None:
            torso_angle = PoseUtils.calculate_torso_angle(smoothed_keypoints)
            left_elbow_angle = PoseUtils.calculate_elbow_angle(smoothed_keypoints, 'left')
            right_elbow_angle = PoseUtils.calculate_elbow_angle(smoothed_keypoints, 'right')
            hip_center = self._get_hip_center(smoothed_keypoints)
            
            record_data = {}
            elbow_angle = None
            if left_elbow_angle is not None and right_elbow_angle is not None:
                elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
            elif left_elbow_angle is not None: elbow_angle = left_elbow_angle
            elif right_elbow_angle is not None: elbow_angle = right_elbow_angle
            if elbow_angle is not None: record_data['elbow'] = elbow_angle
            
            horizontal_angle = None
            if torso_angle is not None:
                horizontal_angle = abs(90 - torso_angle)
                record_data['torso_horizontal'] = horizontal_angle
                
            # 只有在有效计时期间才记录位置，用于稳定性评估
            if self.last_valid_time is not None:
                if 'hip_center' not in self.current_rep_data['positions']:
                     self.current_rep_data['positions']['hip_center'] = []
                if hip_center is not None: # 确保 hip_center 有效
                    self.current_rep_data['positions']['hip_center'].append(hip_center)
                # 记录有效角度
                self._record_rep_data(current_time, smoothed_keypoints, **record_data)
            
            # 更新可视化
            self._add_visualization_data(smoothed_keypoints, torso_angle, elbow_angle, self.state, self.duration)


        # --- 5. 发送状态变更信号 ---
        if old_state != self.state:
            self.state_changed.emit(self.state)

    def _check_plank_position(self, keypoints):
        """检查是否处于平板支撑姿势"""
        if keypoints is None: return False
        
        # 检查躯干角度
        torso_angle = PoseUtils.calculate_torso_angle(keypoints)
        if torso_angle is None: return False
        horizontal_angle = abs(90 - torso_angle)
        torso_horizontal = horizontal_angle <= self.torso_angle_threshold
        if not torso_horizontal: return False # 躯干不达标直接返回
        
        # 检查肘部角度 (至少一侧达标)
        left_elbow_angle = PoseUtils.calculate_elbow_angle(keypoints, 'left')
        right_elbow_angle = PoseUtils.calculate_elbow_angle(keypoints, 'right')
        
        # 如果两侧都不可见，无法判断
        if left_elbow_angle is None and right_elbow_angle is None: return False
        
        left_elbow_bent = (left_elbow_angle is not None and left_elbow_angle <= self.elbow_angle_threshold)
        right_elbow_bent = (right_elbow_angle is not None and right_elbow_angle <= self.elbow_angle_threshold)
        
        # 逻辑调整：标准平板支撑通常要求 *双肘* 都弯曲接近90度支撑，或者直臂支撑
        # 这里允许单肘弯曲可能是为了兼容某些情况，但可能需要调整
        # 考虑改为 AND 逻辑？ left_elbow_bent and right_elbow_bent
        # 或者定义直臂平板支撑的逻辑 (肘角接近180)
        # 当前逻辑: 至少有一边肘是弯曲的 (< 100度)
        elbow_bent = left_elbow_bent or right_elbow_bent
        
        # 也可以加入肩部位置检查，例如肩部不应低于肘部太多
        
        # return torso_horizontal and elbow_bent # 旧逻辑
        
        # 优化逻辑：同时检查躯干水平和至少一个肘部弯曲
        # 未来可以增加更多检查，例如肩/肘/腕对齐，臀部不能过高等
        return torso_horizontal and elbow_bent

    def _calculate_quality(self):
        """计算平板支撑动作质量 (评估记录的有效数据段)"""
        feedback = []
        pose_torso_score = 0
        pose_elbow_score = 0
        stability_hip_score = 0
        stability_torso_score = 0
        
        # --- 姿态评估: 躯干 --- 
        torso_angles = self.current_rep_data.get('angles', {}).get('torso_horizontal', [])
        valid_torso_angles = [a for a in torso_angles if a is not None]
        if valid_torso_angles:
             avg_torso_angle = np.mean(valid_torso_angles)
             # 评分: 越接近 ideal_torso_horizontal_angle 越好
             if avg_torso_angle <= self.ideal_torso_horizontal_angle:
                 pose_torso_score = 100
             elif avg_torso_angle > self.torso_angle_threshold:
                 pose_torso_score = 0
                 feedback.append(f"平均躯干偏差过大 ({avg_torso_angle:.1f}° > {self.torso_angle_threshold}°)，尝试降低臀部")
             else:
                 pose_torso_score = max(0, 100 * (self.torso_angle_threshold - avg_torso_angle) / 
                                      (self.torso_angle_threshold - self.ideal_torso_horizontal_angle))
             
             # --- 稳定性评估: 躯干角度晃动 ---
             std_dev_torso = np.std(valid_torso_angles)
             if std_dev_torso <= self.torso_angle_stability_threshold:
                 stability_torso_score = 100
             else:
                 stability_torso_score = max(0, 100 - 100 * (std_dev_torso - self.torso_angle_stability_threshold) / self.torso_angle_stability_threshold)
                 feedback.append(f"躯干角度晃动较大 (标准差 {std_dev_torso:.1f}° > {self.torso_angle_stability_threshold}°)")
        else:
             feedback.append("无法评估躯干姿态/稳定性 (数据不足)")

        # --- 姿态评估: 肘部 --- 
        elbow_angles = self.current_rep_data.get('angles', {}).get('elbow', [])
        valid_elbow_angles = [a for a in elbow_angles if a is not None]
        if valid_elbow_angles:
             avg_elbow_angle = np.mean(valid_elbow_angles)
             # 评分: 越接近 ideal_elbow_angle 越好 (允许一定误差)
             elbow_diff = abs(avg_elbow_angle - self.ideal_elbow_angle)
             max_allowed_elbow_diff = abs(self.elbow_angle_threshold - self.ideal_elbow_angle) # 基于阈值计算允许的最大差值
             if elbow_diff <= 5: # 理想范围内给满分
                 pose_elbow_score = 100
             elif elbow_diff > max_allowed_elbow_diff:
                  pose_elbow_score = 0
                  feedback.append(f"平均肘角偏差过大 ({avg_elbow_angle:.1f}°, 目标 {self.ideal_elbow_angle}°)")
             else:
                  pose_elbow_score = max(0, 100 * (max_allowed_elbow_diff - elbow_diff) / max_allowed_elbow_diff)
        else:
             feedback.append("无法评估肘部姿态 (数据不足)")
             
        # --- 稳定性评估: 髋部垂直晃动 ---
        hip_centers = self.current_rep_data.get('positions', {}).get('hip_center', [])
        valid_hip_centers = [p for p in hip_centers if p is not None]
        if len(valid_hip_centers) > 1:
             _, std_dev_y = self._calculate_point_variance(valid_hip_centers) # 只关心 Y 方向
             if std_dev_y <= self.hip_stability_threshold:
                 stability_hip_score = 100
             else:
                 stability_hip_score = max(0, 100 - 100 * (std_dev_y - self.hip_stability_threshold) / self.hip_stability_threshold)
                 feedback.append(f"臀部上下晃动较大 (标准差 {std_dev_y:.1f} > {self.hip_stability_threshold})")
        else:
             feedback.append("无法评估臀部稳定性 (数据不足)")

        # --- 综合评分 ---
        total_score = int(self.w_pose_torso * pose_torso_score +
                        self.w_pose_elbow * pose_elbow_score +
                        self.w_stability_hip * stability_hip_score +
                        self.w_stability_torso * stability_torso_score)
                        
        self.logger.debug(f"Plank Quality: Total={total_score}, PoseT={pose_torso_score:.1f}, PoseE={pose_elbow_score:.1f}, StabH={stability_hip_score:.1f}, StabT={stability_torso_score:.1f}, Feedback={feedback}")
        return total_score, feedback
        
    def _add_visualization_data(self, keypoints, torso_angle, elbow_angle, state, current_total_duration):
        """添加可视化数据"""
        horizontal_angle = None
        if torso_angle is not None:
            horizontal_angle = abs(90 - torso_angle)
        
        viz_data = {
            'timestamp': time.time(),
            'torso_angle': torso_angle,
            'horizontal_angle': horizontal_angle,
            'elbow_angle': elbow_angle,
            'state': state,
            'duration': current_total_duration, # 显示实时总时长
            'in_position': self.last_valid_time is not None,
            'posture_correct': self.posture_correct, # 当前帧姿势是否正确
            'last_score': self.quality_scores[-1] if self.quality_scores else None,
            'last_feedback': self.feedback_messages[-1] if self.feedback_messages else []
        }
        self.visualization_history.append(viz_data)
        if len(self.visualization_history) > self.max_history_size:
            self.visualization_history.pop(0)
        self.visualization_data.emit(viz_data)
    
    def reset(self):
        """重置计时器"""
        super().reset()
        self.in_plank_position = False
        self.position_buffer = []
        # visualization_history 在 super().reset() 中处理
        self.posture_correct = False
        # self.plank_start_time = 0  # 开始平板支撑的时间 (移除)
        self.last_valid_time = None # 重置计时锚点
        self.accumulated_valid_time_in_rep = 0.0 # 重置本次累积时间
        self.is_timing = False # 重置计时状态 