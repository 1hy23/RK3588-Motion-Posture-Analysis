#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
深蹲计数器 - 检测并计数深蹲动作
"""

import numpy as np
import time
from .base_counter import ExerciseCounter
from .pose_utils import PoseUtils
import logging # 添加 logging


class SquatCounter(ExerciseCounter):
    """深蹲计数器"""
    
    def __init__(self, confidence_threshold=0.5):
        """初始化深蹲计数器
        
        Args:
            confidence_threshold: 关键点置信度阈值
        """
        super().__init__(confidence_threshold)
        
        # 深蹲特定参数
        self.knee_angle_threshold_down = 100  # 深蹲最低点膝盖角度阈值（标准可以更低，如90）
        self.knee_angle_threshold_up = 160  # 深蹲最高点膝盖角度阈值
        self.ideal_knee_angle_down = 90     # 理想深蹲最低点角度
        self.hip_angle_threshold_down = 95  # 深蹲最低点髋关节角度阈值 (需要调整)
        self.hip_angle_threshold_up = 160  # 深蹲最高点髋关节角度阈值 (需要调整)
        self.vertical_ratio_threshold_down = 0.7 # 髋/踝垂直距离相对于初始值的比例阈值 (表示下蹲深度)
        self.torso_angle_threshold_stable = 30 # 躯干角度变化阈值（评估背部稳定性）
        self.stability_threshold = 15       # 髋部中心点轨迹标准差阈值 (像素)

        # 质量评估权重
        self.w_amplitude = 0.4
        self.w_pose = 0.4
        self.w_stability = 0.2
        
        # 状态变量
        self.is_down = False  # 是否处于下蹲状态
        self.is_up = True  # 开始假设处于站立状态
        self.last_knee_angle = None  # 上一帧膝盖角度
        self.rep_started = False  # 是否已开始一次重复
        self.initial_hip_ankle_distance = None # 初始站立时髋踝垂直距离
        
        # 用于平滑判断的缓冲区 (保持为1，状态转换依赖即时判断)
        self.down_position_buffer = []  # 下蹲位置缓冲区
        self.up_position_buffer = []  # 站立位置缓冲区
        self.buffer_size = 3  # 缓冲区大小 (增大以提高稳定性)
        
        # 数据可视化相关
        self.visualization_history = []  # 可视化历史数据
        self.max_history_size = 30  # 最大历史记录数
        
        # 设置为计数模式
        self.is_counting = True
        self.logger = logging.getLogger(__name__) # 添加 logger
    
    def process(self, keypoints):
        """处理关键点数据，检测并计数深蹲"""
        current_time = time.time()
        if keypoints is None or not isinstance(keypoints, np.ndarray):
            # 如果没有关键点，可能意味着动作中断，需要评估之前的动作(如果存在)
            if self.rep_started and self.current_rep_data.get('timestamps'):
                 score, feedback = self._calculate_quality()
                 self._update_quality_and_feedback(score, feedback)
            self.rep_started = False # 重置状态
            self._start_new_rep() # 清空数据
            return
        
        # 过滤低置信度关键点
        filtered_keypoints = self.filter_keypoints(keypoints)
        if filtered_keypoints is None: return # 无法处理则跳过
        
        # 平滑关键点数据
        smoothed_keypoints = self.smooth_keypoints(filtered_keypoints)
        if smoothed_keypoints is None: return # 无法处理则跳过
        
        # 计算左右膝盖角度
        left_knee_angle = PoseUtils.calculate_knee_angle(smoothed_keypoints, 'left')
        right_knee_angle = PoseUtils.calculate_knee_angle(smoothed_keypoints, 'right')
        
        # --- 新增计算 ---
        left_hip_angle = PoseUtils.calculate_hip_angle(smoothed_keypoints, 'left')
        right_hip_angle = PoseUtils.calculate_hip_angle(smoothed_keypoints, 'right')
        hip_ankle_distance = self._calculate_hip_ankle_vertical_distance(smoothed_keypoints)
        # --------------
        
        # 计算躯干角度 (用于姿态评估)
        torso_angle = PoseUtils.calculate_torso_angle(smoothed_keypoints)
        
        # 使用可见度更高的一侧膝盖角度
        if left_knee_angle is None and right_knee_angle is None:
            old_state = self.state
            self.state = "等待检测膝盖"
            # 如果之前动作开始了但中断了，进行评估
            if self.rep_started and self.current_rep_data.get('timestamps'):
                 score, feedback = self._calculate_quality()
                 self._update_quality_and_feedback(score, feedback)
            self.rep_started = False # 重置状态
            self._start_new_rep() # 清空数据
            if old_state != self.state:
                self.state_changed.emit(self.state)
            self._add_visualization_data(smoothed_keypoints, None, None, None, torso_angle, "等待") # 调整可视化数据
            return
        
        # 优先使用两侧平均值
        if left_knee_angle is not None and right_knee_angle is not None:
            knee_angle = (left_knee_angle + right_knee_angle) / 2
        else:
            knee_angle = left_knee_angle if left_knee_angle is not None else right_knee_angle
        
        # --- 计算平均髋角度 (如果可用) ---
        hip_angle = None
        if left_hip_angle is not None and right_hip_angle is not None:
            hip_angle = (left_hip_angle + right_hip_angle) / 2
        elif left_hip_angle is not None:
            hip_angle = left_hip_angle
        elif right_hip_angle is not None:
            hip_angle = right_hip_angle
        # ------------------------------
        
        # 记录数据，如果动作已开始
        if self.rep_started:
            # 记录膝盖角度、髋角度和躯干角度
            record_data = {'knee': knee_angle}
            if hip_angle is not None: record_data['hip'] = hip_angle # 新增记录髋角度
            if torso_angle is not None: record_data['torso'] = torso_angle
            # 记录髋部中心位置
            if 'hip_center' not in self.current_rep_data['positions']:
                 self.current_rep_data['positions']['hip_center'] = []
            self.current_rep_data['positions']['hip_center'].append(self._get_hip_center(smoothed_keypoints))
            
            self._record_rep_data(current_time, smoothed_keypoints, **record_data)

        # 动态调整角度阈值 (暂未实现)
        # self.knee_angle_threshold_down = self.adjust_threshold_based_on_user(self.knee_angle_threshold_down)
        # self.knee_angle_threshold_up = self.adjust_threshold_based_on_user(self.knee_angle_threshold_up)
        
        old_state = self.state
        # --- 修改下蹲判断条件 --- 
        knee_down = knee_angle is not None and knee_angle <= self.knee_angle_threshold_down
        hip_down = hip_angle is not None and hip_angle <= self.hip_angle_threshold_down
        vertical_down = (hip_ankle_distance is not None and 
                         self.initial_hip_ankle_distance is not None and 
                         self.initial_hip_ankle_distance > 0 and 
                         (hip_ankle_distance / self.initial_hip_ankle_distance) <= self.vertical_ratio_threshold_down)
        in_down_position = knee_down or hip_down or vertical_down # 使用 OR 逻辑增加鲁棒性
        # ----------------------
        self.down_position_buffer.append(in_down_position)
        if len(self.down_position_buffer) > self.buffer_size:
            self.down_position_buffer.pop(0)
        
        # --- 修改站立判断条件 --- 
        knee_up = knee_angle is not None and knee_angle >= self.knee_angle_threshold_up
        # hip_up = hip_angle is not None and hip_angle >= self.hip_angle_threshold_up # 移除 hip_up 条件
        # 站立时垂直距离应恢复，这里简单用角度判断
        in_up_position = knee_up # 仅使用膝盖角度判断站立
        # ----------------------
        self.up_position_buffer.append(in_up_position)
        if len(self.up_position_buffer) > self.buffer_size:
            self.up_position_buffer.pop(0)
        
        is_down = sum(self.down_position_buffer) > len(self.down_position_buffer) / 2
        is_up = sum(self.up_position_buffer) > len(self.up_position_buffer) / 2
        
        # 状态机逻辑
        if not self.rep_started and is_up:
            self._start_new_rep() # 开始新动作，清空数据
            self.rep_started = True
            self.state = "开始"
            # --- 记录初始垂直距离 --- 
            if hip_ankle_distance is not None:
                self.initial_hip_ankle_distance = hip_ankle_distance
            else:
                self.initial_hip_ankle_distance = None # 如果开始时未检测到，则无法使用此指标
            # ----------------------
             # 记录初始数据
            record_data = {'knee': knee_angle}
            if hip_angle is not None: record_data['hip'] = hip_angle
            if torso_angle is not None: record_data['torso'] = torso_angle
            if 'hip_center' not in self.current_rep_data['positions']:
                 self.current_rep_data['positions']['hip_center'] = []
            self.current_rep_data['positions']['hip_center'].append(self._get_hip_center(smoothed_keypoints))
            self._record_rep_data(current_time, smoothed_keypoints, **record_data)
            
        elif self.rep_started and not self.is_down and is_down:
            self.is_down = True
            self.is_up = False
            self.state = "下蹲"
        elif self.rep_started and self.is_down and is_up:
            # 完成一次深蹲
            self.is_up = True
            self.is_down = False
            self.count += 1
            self.count_updated.emit(self.count)
            self.state = "站立"
            
            # 计算并更新质量评分
            if self.current_rep_data.get('timestamps'): # 确保有数据
                score, feedback = self._calculate_quality()
                self._update_quality_and_feedback(score, feedback)
            
            self.rep_started = False # 本次动作结束
            # self._start_new_rep() # 在下次动作开始时清空
        
        if old_state != self.state:
            self.state_changed.emit(self.state)
        
        self.last_knee_angle = knee_angle
        # --- 更新可视化数据传递 --- 
        self._add_visualization_data(smoothed_keypoints, knee_angle, hip_angle, hip_ankle_distance, torso_angle, self.state)
        # --------------------------
        
    def _calculate_quality(self):
        """计算深蹲动作质量"""
        feedback = []
        amplitude_score = 0
        pose_score = 0
        stability_score = 0
        
        # --- 幅度评估 --- 
        knee_angles = self.current_rep_data.get('angles', {}).get('knee', [])
        valid_knee_angles = [a for a in knee_angles if a is not None]
        if valid_knee_angles:
            min_knee_angle = min(valid_knee_angles)
            self.min_angle_history.append(min_knee_angle) # 记录历史用于可能的动态阈值
            
            # --- 幅度评估 (可选：结合垂直距离) --- 
            # 也可以结合 hip_ankle_distance / self.initial_hip_ankle_distance 来评分
            # 例如: vertical_ratio = min(hip_ankle_distances) / self.initial_hip_ankle_distance if ...
            # combined_amplitude_score = (score_from_knee + score_from_vertical) / 2
            # ------------------------------------
            
            # 评分逻辑: 越接近 ideal_knee_angle_down 越高分, 最低为0
            if min_knee_angle <= self.ideal_knee_angle_down:
                 amplitude_score = 100
            elif min_knee_angle > self.knee_angle_threshold_down:
                 amplitude_score = 0 # 未达到最低要求
                 feedback.append(f"幅度不足 (最低 {min_knee_angle:.1f}°, 目标 < {self.knee_angle_threshold_down}°)")
            else:
                 # 在理想和最低要求之间线性评分
                 amplitude_score = max(0, 100 * (self.knee_angle_threshold_down - min_knee_angle) / 
                                      (self.knee_angle_threshold_down - self.ideal_knee_angle_down))
        else:
            feedback.append("无法评估幅度 (无有效膝盖角度)")

        # --- 姿态评估 (背部挺直) --- 
        torso_angles = self.current_rep_data.get('angles', {}).get('torso', [])
        valid_torso_angles = [a for a in torso_angles if a is not None]
        if len(valid_torso_angles) > 1:
            # 检查躯干角度变化范围
            angle_range = max(valid_torso_angles) - min(valid_torso_angles)
            if angle_range <= self.torso_angle_threshold_stable:
                 pose_score = 100
            else:
                 # 超出越多，扣分越多，最低为0
                 pose_score = max(0, 100 - 100 * (angle_range - self.torso_angle_threshold_stable) / self.torso_angle_threshold_stable)
                 feedback.append(f"背部可能弯曲 (躯干角度变化 {angle_range:.1f}° > {self.torso_angle_threshold_stable}°)")
                 
            # 也可以检查最低点时的躯干角度是否过于前倾 (需要定义阈值)
            # min_torso_angle = min(valid_torso_angles)
            # if min_torso_angle < SOME_THRESHOLD:
            #     feedback.append("注意不要过度前倾")
            #     pose_score *= 0.8 # 稍微扣分
        else:
             feedback.append("无法评估背部姿态 (躯干角度数据不足)")

        # --- 稳定性评估 (髋部晃动) ---
        hip_centers = self.current_rep_data.get('positions', {}).get('hip_center', [])
        valid_hip_centers = [p for p in hip_centers if p is not None]
        if len(valid_hip_centers) > 1:
             std_dev_x, std_dev_y = self._calculate_point_variance(valid_hip_centers)
             avg_std_dev = (std_dev_x + std_dev_y) / 2
             
             # 评分逻辑: 标准差越小越好
             if avg_std_dev <= self.stability_threshold:
                 stability_score = 100
             else:
                 # 线性扣分，最低为0
                 stability_score = max(0, 100 - 100 * (avg_std_dev - self.stability_threshold) / self.stability_threshold)
                 feedback.append(f"身体晃动较大 (标准差 {avg_std_dev:.1f} > {self.stability_threshold})")
        else:
             feedback.append("无法评估稳定性 (髋部位置数据不足)")
             
        # --- 综合评分 ---
        total_score = int(self.w_amplitude * amplitude_score + 
                        self.w_pose * pose_score + 
                        self.w_stability * stability_score)
        
        self.logger.debug(f"Rep {self.count} Quality: Total={total_score}, Amp={amplitude_score:.1f}, Pose={pose_score:.1f}, Stab={stability_score:.1f}, Feedback={feedback}")
        return total_score, feedback

    def adjust_threshold_based_on_user(self, threshold):
        """根据用户的身高和体型动态调整阈值
        
        Args:
            threshold: 原始阈值
            
        Returns:
            调整后的阈值
        """
        # 这里可以根据用户的身高、体型等信息动态调整阈值
        # 例如，可以根据用户的身高调整角度阈值，使其更适合用户的体型
        return threshold  # 暂时返回原始阈值，后续可以实现具体的调整逻辑
    
    def _add_visualization_data(self, keypoints, knee_angle, hip_angle, hip_ankle_distance, torso_angle, state):
        """添加用于可视化的数据点 (包含髋角度和垂直距离)"""
        # 决定显示哪个角度，优先膝盖，其次髋部，最后躯干
        angle_to_plot = None
        if knee_angle is not None:
            angle_to_plot = knee_angle
        elif hip_angle is not None:
            angle_to_plot = hip_angle # 可视化可以考虑显示不同角度
        elif torso_angle is not None:
            angle_to_plot = torso_angle
            
        data_point = {
            'time': time.time(),
            'keypoints': keypoints, # 用于可能的未来绘制
            'knee_angle': knee_angle,
            'hip_angle': hip_angle,
            'hip_ankle_distance': hip_ankle_distance,
            'torso_angle': torso_angle,
            'state': state,
            'angle_for_plot': angle_to_plot # 传递给 UI 图表的数据
        }
        self.visualization_history.append(data_point)
        if len(self.visualization_history) > self.max_history_size:
            self.visualization_history.pop(0)
        
        # 发送信号，传递需要可视化的数据点
        # 注意: 发送给UI的数据应保持一致性，如果UI只显示一个角度，这里选择angle_to_plot
        vis_data_for_ui = {
             'knee_angle': knee_angle, # 可以考虑都传给UI，让UI决定显示哪个
             'hip_angle': hip_angle,
             'torso_angle': torso_angle
        }
        self.visualization_data.emit(vis_data_for_ui)    # 使用基类定义的信号
    
    def reset(self):
        """重置计数器"""
        super().reset()
        self.is_down = False
        self.is_up = True
        self.last_knee_angle = None
        self.rep_started = False
        self.down_position_buffer = []
        self.up_position_buffer = []
        # visualization_history 在 super().reset() 中已处理 

    def _get_hip_center(self, keypoints):
        """计算髋部中心点"""
        left_hip = keypoints[PoseUtils.LEFT_HIP]
        right_hip = keypoints[PoseUtils.RIGHT_HIP]
        if left_hip is not None and right_hip is not None:
             # 检查置信度是否足够
            left_conf = left_hip[2] if len(left_hip) > 2 else 0.0
            right_conf = right_hip[2] if len(right_hip) > 2 else 0.0
            if left_conf >= self.confidence_threshold and right_conf >= self.confidence_threshold:
                 return np.mean([left_hip[:2], right_hip[:2]], axis=0)
        elif left_hip is not None:
            left_conf = left_hip[2] if len(left_hip) > 2 else 0.0
            if left_conf >= self.confidence_threshold:
                 return left_hip[:2]
        elif right_hip is not None:
            right_conf = right_hip[2] if len(right_hip) > 2 else 0.0
            if right_conf >= self.confidence_threshold:
                 return right_hip[:2]
        return None
        
    def _get_ankle_center(self, keypoints):
        """计算脚踝中心点"""
        left_ankle = keypoints[PoseUtils.LEFT_ANKLE]
        right_ankle = keypoints[PoseUtils.RIGHT_ANKLE]
        if left_ankle is not None and right_ankle is not None:
             left_conf = left_ankle[2] if len(left_ankle) > 2 else 0.0
             right_conf = right_ankle[2] if len(right_ankle) > 2 else 0.0
             if left_conf >= self.confidence_threshold and right_conf >= self.confidence_threshold:
                 return np.mean([left_ankle[:2], right_ankle[:2]], axis=0)
        elif left_ankle is not None:
            left_conf = left_ankle[2] if len(left_ankle) > 2 else 0.0
            if left_conf >= self.confidence_threshold:
                 return left_ankle[:2]
        elif right_ankle is not None:
            right_conf = right_ankle[2] if len(right_ankle) > 2 else 0.0
            if right_conf >= self.confidence_threshold:
                 return right_ankle[:2]
        return None
        
    def _calculate_hip_ankle_vertical_distance(self, keypoints):
        """计算髋中心到踝中心的垂直距离"""
        hip_center = self._get_hip_center(keypoints)
        ankle_center = self._get_ankle_center(keypoints)
        
        if hip_center is not None and ankle_center is not None:
            return abs(hip_center[1] - ankle_center[1])
        return None 