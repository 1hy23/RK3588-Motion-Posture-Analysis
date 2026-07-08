#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
仰卧起坐计数器 - 检测并计数仰卧起坐动作
"""

import numpy as np
import time
import logging
from .base_counter import ExerciseCounter
from .pose_utils import PoseUtils


class SitUpCounter(ExerciseCounter):
    """仰卧起坐计数器"""
    
    def __init__(self, confidence_threshold=0.5):
        """初始化仰卧起坐计数器
        
        Args:
            confidence_threshold: 关键点置信度阈值
        """
        super().__init__(confidence_threshold)
        
        # 仰卧起坐特定参数
        self.torso_angle_threshold_down = 80  # 仰卧时躯干与垂直方向的最小角度阈值
        self.torso_angle_threshold_up = 30  # 起坐时躯干与垂直方向的最大角度阈值 (调低一点，更严格)
        self.ideal_torso_angle_up = 15      # 理想起坐最高点角度
        # 评估参数
        self.hip_stability_threshold = 15       # 髋部中心点轨迹标准差阈值 (像素)
        self.knee_angle_threshold_stable = 20 # 膝盖角度变化阈值（评估腿部是否固定）

        # 质量评估权重
        self.w_amplitude = 0.4 # 最高点幅度
        self.w_pose = 0.3      # 腿部稳定
        self.w_stability = 0.3 # 核心/髋部稳定
        
        # 状态变量
        self.is_down = True  # 开始假设处于仰卧状态
        self.is_up = False  # 是否处于起坐状态
        self.last_torso_angle = None  # 上一帧躯干角度
        self.rep_started = False  # 是否已开始一次重复
        
        # 用于平滑判断的缓冲区
        self.down_position_buffer = [] 
        self.up_position_buffer = []   
        self.buffer_size = 1  
        
        # 数据可视化相关
        self.visualization_history = []  
        self.max_history_size = 30  
        
        # 设置为计数模式
        self.is_counting = True
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
    
    def process(self, keypoints):
        """处理关键点数据，检测并计数仰卧起坐"""
        current_time = time.time()
        if keypoints is None or not isinstance(keypoints, np.ndarray):
            if self.rep_started and self.current_rep_data.get('timestamps'):
                 score, feedback = self._calculate_quality()
                 self._update_quality_and_feedback(score, feedback)
            self.rep_started = False
            self._start_new_rep()
            return
        
        # 过滤和处理
        filtered_keypoints = self.filter_keypoints(keypoints)
        if filtered_keypoints is None: return
        smoothed_keypoints = self.smooth_keypoints(filtered_keypoints)
        if smoothed_keypoints is None: return
        
        # 计算角度和位置
        torso_angle = PoseUtils.calculate_torso_angle(smoothed_keypoints)
        left_knee_angle = PoseUtils.calculate_knee_angle(smoothed_keypoints, 'left')
        right_knee_angle = PoseUtils.calculate_knee_angle(smoothed_keypoints, 'right')
        hip_center = self._get_hip_center(smoothed_keypoints)
        
        # 处理躯干角度缺失
        if torso_angle is None:
            old_state = self.state
            needs_state_update = self.state != "等待检测躯干"
            self.state = "等待检测躯干"
            if self.rep_started and self.current_rep_data.get('timestamps'):
                 score, feedback = self._calculate_quality()
                 self._update_quality_and_feedback(score, feedback)
            self.rep_started = False
            self.is_down = True # 重置状态
            self.is_up = False
            self._start_new_rep() # 清空数据
            if needs_state_update: self.state_changed.emit(self.state)
            self._add_visualization_data(smoothed_keypoints, None, None, "等待") # 传入 None for knee_angle
            return
        
        # 计算平均膝盖角度 (用于腿部稳定性评估)
        knee_angle = None
        if left_knee_angle is not None and right_knee_angle is not None:
            knee_angle = (left_knee_angle + right_knee_angle) / 2
        else:
            knee_angle = left_knee_angle if left_knee_angle is not None else right_knee_angle

        # 记录数据
        if self.rep_started:
            record_data = {'torso': torso_angle}
            if knee_angle is not None: record_data['knee'] = knee_angle
            # 记录髋部中心位置
            if 'hip_center' not in self.current_rep_data['positions']:
                 self.current_rep_data['positions']['hip_center'] = []
            self.current_rep_data['positions']['hip_center'].append(hip_center)
            self._record_rep_data(current_time, smoothed_keypoints, **record_data)
        
        # 更新当前状态
        old_state = self.state
        
        # 检测位置状态
        in_down_position = torso_angle >= self.torso_angle_threshold_down
        self.down_position_buffer.append(in_down_position)
        if len(self.down_position_buffer) > self.buffer_size: self.down_position_buffer.pop(0)
        
        in_up_position = torso_angle <= self.torso_angle_threshold_up
        self.up_position_buffer.append(in_up_position)
        if len(self.up_position_buffer) > self.buffer_size: self.up_position_buffer.pop(0)
        
        is_down = sum(self.down_position_buffer) > len(self.down_position_buffer) / 2
        is_up = sum(self.up_position_buffer) > len(self.up_position_buffer) / 2
        
        # 状态判断逻辑
        if not self.rep_started and is_down:
            self._start_new_rep()
            self.rep_started = True
            self.state = "开始"
            # 记录初始数据
            record_data = {'torso': torso_angle}
            if knee_angle is not None: record_data['knee'] = knee_angle
            if 'hip_center' not in self.current_rep_data['positions']:
                 self.current_rep_data['positions']['hip_center'] = []
            self.current_rep_data['positions']['hip_center'].append(hip_center)
            self._record_rep_data(current_time, smoothed_keypoints, **record_data)
            
        elif self.rep_started and self.is_down and is_up:
            self.is_down = False
            self.is_up = True
            self.state = "起坐"
        elif self.rep_started and self.is_up and is_down:
            # 完成一次
            self.is_up = False
            self.is_down = True
            self.count += 1
            self.count_updated.emit(self.count)
            self.state = "仰卧"
            
            # 评估质量
            if self.current_rep_data.get('timestamps'): 
                score, feedback = self._calculate_quality()
                self._update_quality_and_feedback(score, feedback)
                
            self.rep_started = False  # 重置
        
        if old_state != self.state:
            self.state_changed.emit(self.state)
        
        self.last_torso_angle = torso_angle
        self._add_visualization_data(smoothed_keypoints, torso_angle, knee_angle, self.state)
    
    def _calculate_quality(self):
        """计算仰卧起坐动作质量"""
        feedback = []
        amplitude_score = 0
        pose_score = 0
        stability_score = 0
        
        # --- 幅度评估 (最高点躯干角度) ---
        torso_angles = self.current_rep_data.get('angles', {}).get('torso', [])
        valid_torso_angles = [a for a in torso_angles if a is not None]
        if valid_torso_angles:
            min_torso_angle = min(valid_torso_angles) # 角度越小表示起得越高
            self.min_angle_history.append(min_torso_angle)
            
            if min_torso_angle <= self.ideal_torso_angle_up:
                amplitude_score = 100
            elif min_torso_angle > self.torso_angle_threshold_up:
                amplitude_score = 0
                feedback.append(f"起身幅度不足 (最高 {min_torso_angle:.1f}°, 目标 < {self.torso_angle_threshold_up}°)")
            else:
                 amplitude_score = max(0, 100 * (self.torso_angle_threshold_up - min_torso_angle) / 
                                      (self.torso_angle_threshold_up - self.ideal_torso_angle_up))
        else:
            feedback.append("无法评估幅度 (无有效躯干角度)")

        # --- 姿态评估 (腿部稳定 - 膝盖角度变化) ---
        knee_angles = self.current_rep_data.get('angles', {}).get('knee', [])
        valid_knee_angles = [a for a in knee_angles if a is not None]
        if len(valid_knee_angles) > 1:
            angle_range = max(valid_knee_angles) - min(valid_knee_angles)
            if angle_range <= self.knee_angle_threshold_stable:
                 pose_score = 100
            else:
                 pose_score = max(0, 100 - 100 * (angle_range - self.knee_angle_threshold_stable) / self.knee_angle_threshold_stable)
                 feedback.append(f"腿部可能移动 (膝盖角度变化 {angle_range:.1f}° > {self.knee_angle_threshold_stable}°)")
        else:
             feedback.append("无法评估腿部稳定性 (膝盖角度数据不足)")

        # --- 稳定性评估 (髋部晃动) ---
        hip_centers = self.current_rep_data.get('positions', {}).get('hip_center', [])
        valid_hip_centers = [p for p in hip_centers if p is not None]
        if len(valid_hip_centers) > 1:
             std_dev_x, std_dev_y = self._calculate_point_variance(valid_hip_centers)
             # 主要关注左右晃动 (x轴)
             if std_dev_x <= self.hip_stability_threshold:
                 stability_score = 100
             else:
                 stability_score = max(0, 100 - 100 * (std_dev_x - self.hip_stability_threshold) / self.hip_stability_threshold)
                 feedback.append(f"核心/臀部不稳定 (左右晃动 {std_dev_x:.1f} > {self.hip_stability_threshold})")
        else:
             feedback.append("无法评估核心稳定性 (髋部位置数据不足)")
             
        # --- 综合评分 ---
        total_score = int(self.w_amplitude * amplitude_score + 
                        self.w_pose * pose_score + 
                        self.w_stability * stability_score)
                        
        self.logger.debug(f"Rep {self.count} Quality: Total={total_score}, Amp={amplitude_score:.1f}, Pose={pose_score:.1f}, Stab={stability_score:.1f}, Feedback={feedback}")
        return total_score, feedback

    # 修改可视化数据添加，加入膝盖角度
    def _add_visualization_data(self, keypoints, torso_angle, knee_angle, state):
        """添加可视化数据"""
        viz_data = {
            'timestamp': time.time(),
            'torso_angle': torso_angle,
            'knee_angle': knee_angle, # 添加膝盖角度
            'state': state,
            'count': self.count,
            'last_score': self.quality_scores[-1] if self.quality_scores else None,
            'last_feedback': self.feedback_messages[-1] if self.feedback_messages else []
        }
        self.visualization_history.append(viz_data)
        if len(self.visualization_history) > self.max_history_size: self.visualization_history.pop(0)
        self.visualization_data.emit(viz_data)
    
    # reset 利用基类逻辑
    def reset(self):
        """重置计数器"""
        super().reset()
        self.is_down = True
        self.is_up = False
        self.last_torso_angle = None
        self.rep_started = False
        self.down_position_buffer = []
        self.up_position_buffer = []
        self.visualization_history = [] 