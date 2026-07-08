#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
臂屈伸计数器 - 检测并计数臂屈伸动作
"""

import numpy as np
import time
from .base_counter import ExerciseCounter
from .pose_utils import PoseUtils
import logging


class TricepDipCounter(ExerciseCounter):
    """臂屈伸计数器"""
    
    def __init__(self, confidence_threshold=0.5):
        """初始化臂屈伸计数器
        
        Args:
            confidence_threshold: 关键点置信度阈值
        """
        super().__init__(confidence_threshold)
        
        # 臂屈伸特定参数
        self.elbow_angle_threshold_down = 120 # 下降时肘部角度阈值 (调低更合理)
        self.elbow_angle_threshold_up = 160  # 上升时肘部角度阈值
        self.ideal_elbow_angle_down = 90    # 理想最低点肘部角度
        # self.shoulder_hip_diff_threshold = 50 # 姿势约束，不在质量评估中使用

        # 评估参数
        self.hip_stability_threshold = 15 # 髋部垂直稳定性阈值 (像素)

        # 质量评估权重
        self.w_amplitude = 0.6 # 幅度 (下降和上升)
        self.w_stability = 0.4 # 稳定性
        # 姿态评估可以通过检查起始/结束是否满足条件隐含评估

        # 状态变量
        self.is_down = False  # 是否处于下降状态
        self.is_up = True  # 开始假设处于上升状态
        self.last_elbow_angle = None  # 上一帧肘部角度
        self.rep_started = False  # 是否已开始一次重复
        self.in_valid_pose = False # 记录是否处于有效臂屈伸姿势
        
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
        """处理关键点数据，检测并计数臂屈伸"""
        current_time = time.time()
        if keypoints is None or not isinstance(keypoints, np.ndarray):
            if self.rep_started and self.current_rep_data.get('timestamps'):
                 score, feedback = self._calculate_quality()
                 self._update_quality_and_feedback(score, feedback)
            self.rep_started = False
            self._start_new_rep()
            self.in_valid_pose = False # 重置姿势状态
            return
        
        # 过滤和处理
        filtered_keypoints = self.filter_keypoints(keypoints)
        if filtered_keypoints is None: return
        smoothed_keypoints = self.smooth_keypoints(filtered_keypoints)
        if smoothed_keypoints is None: return
        
        # 检查是否处于臂屈伸姿势（肩部高于髋部） - 这是前提条件
        self.in_valid_pose = self._is_in_dip_position(smoothed_keypoints)
        
        if not self.in_valid_pose:
            old_state = self.state
            needs_state_update = self.state != "等待臂屈伸姿势"
            self.state = "等待臂屈伸姿势"
            if self.rep_started and self.current_rep_data.get('timestamps'):
                score, feedback = self._calculate_quality()
                self._update_quality_and_feedback(score, feedback)
            self.rep_started = False
            self.is_down = False # 重置计数状态
            self.is_up = True
            self._start_new_rep()
            if needs_state_update: self.state_changed.emit(self.state)
            # 即使姿势无效，也尝试获取肘角用于可视化
            elbow_angle_viz = self._get_current_elbow_angle(smoothed_keypoints)
            self._add_visualization_data(smoothed_keypoints, elbow_angle_viz, "等待", False)
            return
        
        # 计算角度和位置 (只有在有效姿势下才进行计数和评估逻辑)
        elbow_angle = self._get_current_elbow_angle(smoothed_keypoints)
        hip_center = self._get_hip_center(smoothed_keypoints)

        # 处理肘部角度缺失
        if elbow_angle is None:
            old_state = self.state
            needs_state_update = self.state != "等待检测肘部"
            self.state = "等待检测肘部"
            if self.rep_started and self.current_rep_data.get('timestamps'):
                score, feedback = self._calculate_quality()
                self._update_quality_and_feedback(score, feedback)
            self.rep_started = False
            self.is_down = False
            self.is_up = True
            self._start_new_rep()
            if needs_state_update: self.state_changed.emit(self.state)
            self._add_visualization_data(smoothed_keypoints, None, "等待", True)
            return
        
        # 记录数据
        if self.rep_started:
            record_data = {'elbow': elbow_angle}
            # 记录髋部中心位置
            if 'hip_center' not in self.current_rep_data['positions']:
                 self.current_rep_data['positions']['hip_center'] = []
            self.current_rep_data['positions']['hip_center'].append(hip_center)
            self._record_rep_data(current_time, smoothed_keypoints, **record_data)
        
        # 更新当前状态
        old_state = self.state
        
        # 检测下降/上升位置
        in_down_position = elbow_angle <= self.elbow_angle_threshold_down
        self.down_position_buffer.append(in_down_position)
        if len(self.down_position_buffer) > self.buffer_size: self.down_position_buffer.pop(0)
        
        in_up_position = elbow_angle >= self.elbow_angle_threshold_up
        self.up_position_buffer.append(in_up_position)
        if len(self.up_position_buffer) > self.buffer_size: self.up_position_buffer.pop(0)
        
        is_down = sum(self.down_position_buffer) > len(self.down_position_buffer) / 2
        is_up = sum(self.up_position_buffer) > len(self.up_position_buffer) / 2
        
        # 状态判断逻辑 (与 PushUp 类似)
        if not self.rep_started and is_up:
            self._start_new_rep()
            self.rep_started = True
            self.state = "开始"
            # 记录初始数据
            record_data = {'elbow': elbow_angle}
            if 'hip_center' not in self.current_rep_data['positions']:
                 self.current_rep_data['positions']['hip_center'] = []
            self.current_rep_data['positions']['hip_center'].append(hip_center)
            self._record_rep_data(current_time, smoothed_keypoints, **record_data)
            
        elif self.rep_started and self.is_up and is_down: # 从上到下转换
            self.is_down = True
            self.is_up = False
            self.state = "下降"
        elif self.rep_started and self.is_down and is_up: # 从下到上转换，计为一次完成
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
        self._add_visualization_data(smoothed_keypoints, elbow_angle, self.state, self.in_valid_pose)
    
    def _get_current_elbow_angle(self, keypoints):
        """计算当前帧的平均或单侧肘部角度"""
        left_elbow_angle = PoseUtils.calculate_elbow_angle(keypoints, 'left')
        right_elbow_angle = PoseUtils.calculate_elbow_angle(keypoints, 'right')
        if left_elbow_angle is not None and right_elbow_angle is not None:
            return (left_elbow_angle + right_elbow_angle) / 2
        else:
            return left_elbow_angle if left_elbow_angle is not None else right_elbow_angle

    def _is_in_dip_position(self, keypoints):
        """检查是否处于臂屈伸姿势（肩部高于髋部）"""
        if keypoints is None: return False
        left_shoulder = keypoints[PoseUtils.LEFT_SHOULDER]
        right_shoulder = keypoints[PoseUtils.RIGHT_SHOULDER]
        left_hip = keypoints[PoseUtils.LEFT_HIP]
        right_hip = keypoints[PoseUtils.RIGHT_HIP]
        if (np.isnan(left_shoulder).any() or np.isnan(right_shoulder).any() or 
            np.isnan(left_hip).any() or np.isnan(right_hip).any()):
            # 如果单侧有效，也认为姿势可能有效，让角度计算去处理 None
            if (not np.isnan(left_shoulder).any() and not np.isnan(left_hip).any()) or \
               (not np.isnan(right_shoulder).any() and not np.isnan(right_hip).any()):
               pass # 继续检查
            else:
                return False
        
        # 尝试使用有效的单侧或双侧计算
        valid_shoulders_y = [s[1] for s in [left_shoulder, right_shoulder] if not np.isnan(s).any()]
        valid_hips_y = [h[1] for h in [left_hip, right_hip] if not np.isnan(h).any()]
        
        if not valid_shoulders_y or not valid_hips_y:
            return False # 无法比较
            
        shoulder_y = np.mean(valid_shoulders_y)
        hip_y = np.mean(valid_hips_y)
        
        # 检查肩部是否高于髋部 (Y值越小越高)
        # 稍微放宽条件，允许等于或略低于 (处理误差)
        return shoulder_y <= hip_y + 5 # 允许5像素误差
        # 移除距离阈值检查，这个阈值意义不大且可能误判
        # distance_sufficient = abs(shoulder_y - hip_y) > self.shoulder_hip_diff_threshold 
        # return shoulder_above_hip and distance_sufficient
        
    def _calculate_quality(self):
        """计算臂屈伸动作质量"""
        feedback = []
        amplitude_score = 0
        stability_score = 0

        # --- 幅度评估 --- 
        elbow_angles = self.current_rep_data.get('angles', {}).get('elbow', [])
        valid_elbow_angles = [a for a in elbow_angles if a is not None]
        if valid_elbow_angles:
            min_elbow_angle = min(valid_elbow_angles)
            max_elbow_angle = max(valid_elbow_angles)
            self.min_angle_history.append(min_elbow_angle)

            # 1. 最低点评分
            min_score = 0
            if min_elbow_angle <= self.ideal_elbow_angle_down:
                 min_score = 100
            elif min_elbow_angle > self.elbow_angle_threshold_down:
                 min_score = 0
                 feedback.append(f"下降幅度不足 (最低 {min_elbow_angle:.1f}°, 目标 < {self.elbow_angle_threshold_down}°)")
            else:
                 min_score = max(0, 100 * (self.elbow_angle_threshold_down - min_elbow_angle) / 
                                      (self.elbow_angle_threshold_down - self.ideal_elbow_angle_down))
            
            # 2. 最高点评分
            max_score = 0
            if max_elbow_angle >= self.elbow_angle_threshold_up:
                 max_score = 100
            else:
                 # 最高点未达到要求，扣分
                 max_score = max(0, 100 * (max_elbow_angle - self.ideal_elbow_angle_down) / 
                                    (self.elbow_angle_threshold_up - self.ideal_elbow_angle_down)) # 基于最低理想值和最高阈值评分
                 feedback.append(f"最高点手臂未充分伸直 ({max_elbow_angle:.1f}° < {self.elbow_angle_threshold_up}°)")
            
            # 幅度总分 = 最低点 * 60% + 最高点 * 40%
            amplitude_score = 0.6 * min_score + 0.4 * max_score
        else:
            feedback.append("无法评估幅度 (无有效肘部角度)")

        # --- 稳定性评估 (髋部垂直晃动) ---
        hip_centers = self.current_rep_data.get('positions', {}).get('hip_center', [])
        valid_hip_centers = [p for p in hip_centers if p is not None]
        if len(valid_hip_centers) > 1:
             _, std_dev_y = self._calculate_point_variance(valid_hip_centers)
             if std_dev_y <= self.hip_stability_threshold:
                 stability_score = 100
             else:
                 stability_score = max(0, 100 - 100 * (std_dev_y - self.hip_stability_threshold) / self.hip_stability_threshold)
                 feedback.append(f"身体上下晃动较大 ({std_dev_y:.1f} > {self.hip_stability_threshold})")
        else:
             feedback.append("无法评估稳定性 (髋部位置数据不足)")

        # --- 综合评分 ---
        total_score = int(self.w_amplitude * amplitude_score + 
                        self.w_stability * stability_score)
                        
        self.logger.debug(f"Rep {self.count} Quality: Total={total_score}, Amp={amplitude_score:.1f}, Stab={stability_score:.1f}, Feedback={feedback}")
        return total_score, feedback

    def _add_visualization_data(self, keypoints, elbow_angle, state, in_dip_position):
        """添加可视化数据"""
        viz_data = {
            'timestamp': time.time(),
            'elbow_angle': elbow_angle,
            'state': state,
            'count': self.count,
            'in_dip_position': in_dip_position,
            'last_score': self.quality_scores[-1] if self.quality_scores else None,
            'last_feedback': self.feedback_messages[-1] if self.feedback_messages else []
        }
        self.visualization_history.append(viz_data)
        if len(self.visualization_history) > self.max_history_size: self.visualization_history.pop(0)
        self.visualization_data.emit(viz_data)
    
    def reset(self):
        """重置计数器"""
        super().reset()
        self.is_down = False
        self.is_up = True
        self.last_elbow_angle = None
        self.rep_started = False
        self.down_position_buffer = []
        self.up_position_buffer = []
        self.in_valid_pose = False # 重置姿势状态 