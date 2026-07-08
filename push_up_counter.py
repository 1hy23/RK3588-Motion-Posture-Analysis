#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
俯卧撑计数器 - 检测并计数俯卧撑动作
"""

import numpy as np
import time
from .base_counter import ExerciseCounter
from .pose_utils import PoseUtils
import logging # 添加 logging

class PushUpCounter(ExerciseCounter):
    """俯卧撑计数器"""
    
    def __init__(self, confidence_threshold=0.5):
        """初始化俯卧撑计数器
        
        Args:
            confidence_threshold: 关键点置信度阈值
        """
        super().__init__(confidence_threshold)
        
        # 俯卧撑特定参数
        # 注意：原 elbow_angle_threshold_down = 140 似乎太高，调整为更合理的值
        self.elbow_angle_threshold_down = 100 # 俯卧撑最低点肘部角度阈值 (越小越好)
        self.elbow_angle_threshold_up = 160  # 俯卧撑最高点肘部角度阈值
        self.ideal_elbow_angle_down = 90    # 理想最低点肘部角度
        self.torso_angle_stability_threshold = 15 # 躯干角度变化阈值 (评估身体直线)
        self.hip_stability_threshold = 20       # 髋部垂直稳定性阈值 (像素)
        
        # 质量评估权重
        self.w_amplitude = 0.4
        self.w_pose = 0.3       # 身体直线
        self.w_stability = 0.3  # 髋部稳定
        
        # 状态变量
        # self.in_push_up_position = False # 这个变量似乎未使用，移除
        self.is_down = False  # 是否处于下降状态
        self.is_up = True  # 开始假设处于上升状态
        self.last_elbow_angle = None  # 上一帧肘部角度
        self.rep_started = False  # 是否已开始一次重复
        
        # 用于平滑判断的缓冲区 (保持为1)
        self.down_position_buffer = [] 
        self.up_position_buffer = []   
        self.buffer_size = 1  
        
        # 数据可视化相关
        self.visualization_history = []  
        self.max_history_size = 30  
        
        # 设置为计数模式
        self.is_counting = True
        self.logger = logging.getLogger(__name__) # 添加 logger
    
    def process(self, keypoints):
        """处理关键点数据，检测并计数俯卧撑"""
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
        left_elbow_angle = PoseUtils.calculate_elbow_angle(smoothed_keypoints, 'left')
        right_elbow_angle = PoseUtils.calculate_elbow_angle(smoothed_keypoints, 'right')
        torso_angle = PoseUtils.calculate_torso_angle(smoothed_keypoints)
        hip_center = self._get_hip_center(smoothed_keypoints)
        
        # 处理肘部角度缺失
        if left_elbow_angle is None and right_elbow_angle is None:
            old_state = self.state
            self.state = "等待检测肘部"
            if self.rep_started and self.current_rep_data.get('timestamps'):
                score, feedback = self._calculate_quality()
                self._update_quality_and_feedback(score, feedback)
            self.rep_started = False
            self._start_new_rep()
            if old_state != self.state: self.state_changed.emit(self.state)
            self._add_visualization_data(smoothed_keypoints, None, torso_angle, "等待")
            return
        
        # 计算平均或单侧肘部角度
        if left_elbow_angle is not None and right_elbow_angle is not None:
            elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
        else:
            elbow_angle = left_elbow_angle if left_elbow_angle is not None else right_elbow_angle
        
        # 记录数据
        if self.rep_started:
            record_data = {'elbow': elbow_angle}
            if torso_angle is not None: record_data['torso'] = torso_angle
            # 记录髋部中心位置
            if 'hip_center' not in self.current_rep_data['positions']:
                 self.current_rep_data['positions']['hip_center'] = []
            self.current_rep_data['positions']['hip_center'].append(hip_center)
            self._record_rep_data(current_time, smoothed_keypoints, **record_data)

        # 动态调整阈值 (暂未实现)
        # self.elbow_angle_threshold_down = self.adjust_threshold_based_on_user(self.elbow_angle_threshold_down)
        # self.elbow_angle_threshold_up = self.adjust_threshold_based_on_user(self.elbow_angle_threshold_up)
        
        old_state = self.state
        in_down_position = elbow_angle <= self.elbow_angle_threshold_down
        self.down_position_buffer.append(in_down_position)
        if len(self.down_position_buffer) > self.buffer_size: self.down_position_buffer.pop(0)
        
        in_up_position = elbow_angle >= self.elbow_angle_threshold_up
        self.up_position_buffer.append(in_up_position)
        if len(self.up_position_buffer) > self.buffer_size: self.up_position_buffer.pop(0)
        
        is_down = sum(self.down_position_buffer) > len(self.down_position_buffer) / 2
        is_up = sum(self.up_position_buffer) > len(self.up_position_buffer) / 2
   
        # 状态机
        if not self.rep_started and is_up:
            self._start_new_rep()
            self.rep_started = True
            self.state = "开始"
            # 记录初始数据
            record_data = {'elbow': elbow_angle}
            if torso_angle is not None: record_data['torso'] = torso_angle
            if 'hip_center' not in self.current_rep_data['positions']:
                self.current_rep_data['positions']['hip_center'] = []
            self.current_rep_data['positions']['hip_center'].append(hip_center)
            self._record_rep_data(current_time, smoothed_keypoints, **record_data)
            
        elif self.rep_started and self.is_up and is_down: # 从 上 -> 下
            self.is_down = True
            self.is_up = False
            self.state = "下降"
        elif self.rep_started and self.is_down and is_up: # 从 下 -> 上 (完成)
            self.is_up = True
            self.is_down = False
            self.count += 1
            self.count_updated.emit(self.count)
            self.state = "上升"
            
            # 评估质量
            if self.current_rep_data.get('timestamps'):
                score, feedback = self._calculate_quality()
                self._update_quality_and_feedback(score, feedback)
                
            self.rep_started = False

        if old_state != self.state:
            self.state_changed.emit(self.state)
        
        self.last_elbow_angle = elbow_angle
        self._add_visualization_data(smoothed_keypoints, elbow_angle, torso_angle, self.state)
    
    def _calculate_quality(self):
        """计算俯卧撑动作质量"""
        feedback = []
        amplitude_score = 0
        pose_score = 0
        stability_score = 0

        # --- 幅度评估 (肘部角度) --- 
        elbow_angles = self.current_rep_data.get('angles', {}).get('elbow', [])
        valid_elbow_angles = [a for a in elbow_angles if a is not None]
        if valid_elbow_angles:
            min_elbow_angle = min(valid_elbow_angles)
            self.min_angle_history.append(min_elbow_angle)
            
            if min_elbow_angle <= self.ideal_elbow_angle_down:
                 amplitude_score = 100
            elif min_elbow_angle > self.elbow_angle_threshold_down:
                 amplitude_score = 0
                 feedback.append(f"下降幅度不足 (最低 {min_elbow_angle:.1f}°, 目标 < {self.elbow_angle_threshold_down}°)")
            else:
                 amplitude_score = max(0, 100 * (self.elbow_angle_threshold_down - min_elbow_angle) / 
                                      (self.elbow_angle_threshold_down - self.ideal_elbow_angle_down))
            
            # 检查最高点是否充分伸直
            max_elbow_angle = max(valid_elbow_angles)
            if max_elbow_angle < self.elbow_angle_threshold_up:
                 amplitude_score *= 0.8 # 幅度略微扣分
                 feedback.append(f"最高点手臂未充分伸直 ({max_elbow_angle:.1f}° < {self.elbow_angle_threshold_up}°)")
        else:
            feedback.append("无法评估幅度 (无有效肘部角度)")

        # --- 姿态评估 (身体直线 - 躯干角度稳定) --- 
        torso_angles = self.current_rep_data.get('angles', {}).get('torso', [])
        valid_torso_angles = [a for a in torso_angles if a is not None]
        if len(valid_torso_angles) > 1:
            angle_range = max(valid_torso_angles) - min(valid_torso_angles)
            if angle_range <= self.torso_angle_stability_threshold:
                 pose_score = 100
            else:
                 pose_score = max(0, 100 - 100 * (angle_range - self.torso_angle_stability_threshold) / self.torso_angle_stability_threshold)
                 feedback.append(f"身体未保持直线 (躯干角度变化 {angle_range:.1f}° > {self.torso_angle_stability_threshold}°)")
        else:
             feedback.append("无法评估身体直线度 (躯干角度数据不足)")

        # --- 稳定性评估 (髋部垂直晃动) ---
        hip_centers = self.current_rep_data.get('positions', {}).get('hip_center', [])
        valid_hip_centers = [p for p in hip_centers if p is not None]
        if len(valid_hip_centers) > 1:
             _, std_dev_y = self._calculate_point_variance(valid_hip_centers)
             if std_dev_y <= self.hip_stability_threshold:
                 stability_score = 100
             else:
                 stability_score = max(0, 100 - 100 * (std_dev_y - self.hip_stability_threshold) / self.hip_stability_threshold)
                 feedback.append(f"核心不稳定，臀部晃动 ({std_dev_y:.1f} > {self.hip_stability_threshold})")
        else:
             feedback.append("无法评估核心稳定性 (髋部位置数据不足)")

        # --- 综合评分 ---
        total_score = int(self.w_amplitude * amplitude_score + 
                        self.w_pose * pose_score + 
                        self.w_stability * stability_score)
                        
        self.logger.debug(f"Rep {self.count} Quality: Total={total_score}, Amp={amplitude_score:.1f}, Pose={pose_score:.1f}, Stab={stability_score:.1f}, Feedback={feedback}")
        return total_score, feedback

    # adjust_threshold_based_on_user 保持不变
    def adjust_threshold_based_on_user(self, threshold):
        return threshold
    
    # 修改可视化数据添加
    def _add_visualization_data(self, keypoints, elbow_angle, torso_angle, state):
        """添加可视化数据"""
        viz_data = {
            'timestamp': time.time(),
            'elbow_angle': elbow_angle,
            'torso_angle': torso_angle, # 添加躯干角度
            'state': state,
            'count': self.count,
            # 'in_push_up_position': self.in_push_up_position # 移除未使用变量
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
        # self.in_push_up_position = False # 移除
        self.is_down = False
        self.is_up = True
        self.last_elbow_angle = None
        self.rep_started = False
        self.down_position_buffer = []
        self.up_position_buffer = [] 