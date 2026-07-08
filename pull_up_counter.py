#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
引体向上计数器 - 检测并计数引体向上动作
"""

import numpy as np
import time
from .base_counter import ExerciseCounter
from .pose_utils import PoseUtils
import logging


class PullUpCounter(ExerciseCounter):
    """引体向上计数器"""
    
    def __init__(self, confidence_threshold=0.5):
        """初始化引体向上计数器
        
        Args:
            confidence_threshold: 关键点置信度阈值
        """
        super().__init__(confidence_threshold)
        
        # 引体向上特定参数
        self.elbow_angle_threshold_down = 140 # 下垂时肘部角度阈值 (调高，接近伸直)
        self.elbow_angle_threshold_up = 60   # 上拉时肘部角度阈值 (调高一点)
        self.ideal_elbow_angle_down = 170   # 理想下放角度
        self.ideal_elbow_angle_up = 45     # 理想上拉角度
        # self.chin_over_bar_threshold = 100 # 像素阈值，改为在 _is_chin_over_bar 中直接比较 Y 坐标
        self.wrist_shoulder_alignment_threshold = 30 # 手腕和肩膀水平对齐阈值 (像素)
        
        # 质量评估权重
        self.w_amplitude_up = 0.4    # 上拉幅度 (下巴过杠+肘角)
        self.w_amplitude_down = 0.3  # 下放幅度 (肘角)
        self.w_pose = 0.3          # 手腕/肩膀对齐 (简化稳定性评估)

        # 状态变量
        self.is_down = True  
        self.is_up = False   
        self.last_elbow_angle = None 
        self.rep_started = False 
        
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
        """处理关键点数据，检测并计数引体向上"""
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
        elbow_angle = self._get_current_elbow_angle(smoothed_keypoints)
        chin_over_bar = self._is_chin_over_bar(smoothed_keypoints)
        wrist_alignment = self._check_wrist_alignment(smoothed_keypoints)
        
        # 处理肘部角度缺失
        if elbow_angle is None:
            old_state = self.state
            needs_state_update = self.state != "等待检测肘部"
            self.state = "等待检测肘部"
            if self.rep_started and self.current_rep_data.get('timestamps'):
                score, feedback = self._calculate_quality()
                self._update_quality_and_feedback(score, feedback)
            self.rep_started = False
            self.is_down = True
            self.is_up = False
            self._start_new_rep()
            if needs_state_update: self.state_changed.emit(self.state)
            self._add_visualization_data(smoothed_keypoints, None, "等待", False, False)
            return
        
        # 记录数据
        if self.rep_started:
            record_data = {'elbow': elbow_angle, 'chin_over': chin_over_bar, 'wrist_align': wrist_alignment}
            self._record_rep_data(current_time, smoothed_keypoints, **record_data)
        
        # 更新当前状态
        old_state = self.state
        
        # 检测位置
        in_down_position = elbow_angle >= self.elbow_angle_threshold_down
        self.down_position_buffer.append(in_down_position)
        if len(self.down_position_buffer) > self.buffer_size: self.down_position_buffer.pop(0)
        
        # 上拉位置: 肘部弯曲且下巴过杠
        in_up_position = (elbow_angle <= self.elbow_angle_threshold_up) and chin_over_bar
        self.up_position_buffer.append(in_up_position)
        if len(self.up_position_buffer) > self.buffer_size: self.up_position_buffer.pop(0)
        
        is_down = sum(self.down_position_buffer) > len(self.down_position_buffer) / 2
        is_up = sum(self.up_position_buffer) > len(self.up_position_buffer) / 2
        
        # 状态判断逻辑
        # 注意：原版在完成一次后没有重置 rep_started，这里修改为标准模式
        if not self.rep_started and is_down:
            self._start_new_rep()
            self.rep_started = True
            self.state = "开始"
            # 记录初始数据
            record_data = {'elbow': elbow_angle, 'chin_over': chin_over_bar, 'wrist_align': wrist_alignment}
            self._record_rep_data(current_time, smoothed_keypoints, **record_data)
            
        elif self.rep_started and self.is_down and is_up:
            self.is_down = False
            self.is_up = True
            self.state = "上拉"
        elif self.rep_started and self.is_up and is_down:
            # 完成一次
            self.is_up = False
            self.is_down = True
            self.count += 1
            self.count_updated.emit(self.count)
            self.state = "下垂"
            
            # 评估质量
            if self.current_rep_data.get('timestamps'): 
                score, feedback = self._calculate_quality()
                self._update_quality_and_feedback(score, feedback)
                
            self.rep_started = False # 重置状态
        
        if old_state != self.state:
            self.state_changed.emit(self.state)
        
        self.last_elbow_angle = elbow_angle
        self._add_visualization_data(smoothed_keypoints, elbow_angle, self.state, chin_over_bar, wrist_alignment)
    
    def _get_current_elbow_angle(self, keypoints):
        """计算当前帧的平均或单侧肘部角度"""
        left_elbow_angle = PoseUtils.calculate_elbow_angle(keypoints, 'left')
        right_elbow_angle = PoseUtils.calculate_elbow_angle(keypoints, 'right')
        if left_elbow_angle is not None and right_elbow_angle is not None:
            return (left_elbow_angle + right_elbow_angle) / 2
        else:
            return left_elbow_angle if left_elbow_angle is not None else right_elbow_angle
            
    def _is_chin_over_bar(self, keypoints):
        """检查下巴是否超过横杠（近似为鼻子高于手腕平均高度）"""
        if keypoints is None: return False
        nose = keypoints[PoseUtils.NOSE]
        left_wrist = keypoints[PoseUtils.LEFT_WRIST]
        right_wrist = keypoints[PoseUtils.RIGHT_WRIST]
        
        if np.isnan(nose).any(): return False
        
        # 检查至少一个手腕有效
        valid_wrists_y = [w[1] for w in [left_wrist, right_wrist] if not np.isnan(w).any()]
        if not valid_wrists_y: return False
            
        # 使用有效手腕的平均高度作为横杠的近似
        bar_y = np.mean(valid_wrists_y)
        
        # 检查鼻子是否高于横杠 (Y值越小越高)
        return nose[1] < bar_y 

    def _check_wrist_alignment(self, keypoints):
        """检查手腕是否大致在肩膀下方 (水平方向对齐)"""
        if keypoints is None: return False
        left_wrist = keypoints[PoseUtils.LEFT_WRIST]
        right_wrist = keypoints[PoseUtils.RIGHT_WRIST]
        left_shoulder = keypoints[PoseUtils.LEFT_SHOULDER]
        right_shoulder = keypoints[PoseUtils.RIGHT_SHOULDER]
        
        aligned = True
        # 检查左侧
        if not np.isnan(left_wrist).any() and not np.isnan(left_shoulder).any():
            if abs(left_wrist[0] - left_shoulder[0]) > self.wrist_shoulder_alignment_threshold:
                aligned = False
        # 检查右侧
        if not np.isnan(right_wrist).any() and not np.isnan(right_shoulder).any():
            if abs(right_wrist[0] - right_shoulder[0]) > self.wrist_shoulder_alignment_threshold:
                aligned = False
                
        # 如果两侧都无效，认为是对齐的 (避免误判)
        if np.isnan(left_wrist).any() and np.isnan(right_wrist).any():
             aligned = True 
             
        return aligned

    def _calculate_quality(self):
        """计算引体向上动作质量"""
        feedback = []
        amplitude_up_score = 0
        amplitude_down_score = 0
        pose_score = 0
        
        # --- 幅度评估 (上拉) ---
        chin_overs = self.current_rep_data.get('angles', {}).get('chin_over', [])
        elbow_angles = self.current_rep_data.get('angles', {}).get('elbow', [])
        valid_elbow_angles = [a for a in elbow_angles if a is not None]
        
        achieved_chin_over = any(c for c in chin_overs if c is True)
        min_elbow_angle = min(valid_elbow_angles) if valid_elbow_angles else None
        
        if achieved_chin_over:
            # 下巴过杠了，根据肘角评分
            if min_elbow_angle is not None:
                if min_elbow_angle <= self.ideal_elbow_angle_up:
                    amplitude_up_score = 100
                elif min_elbow_angle > self.elbow_angle_threshold_up:
                    amplitude_up_score = 50 # 过杠但幅度不够，给基础分
                    feedback.append(f"上拉幅度不足 (肘角 {min_elbow_angle:.1f}° > {self.elbow_angle_threshold_up}°)")
                else:
                    amplitude_up_score = 50 + 50 * (self.elbow_angle_threshold_up - min_elbow_angle) / \
                                           (self.elbow_angle_threshold_up - self.ideal_elbow_angle_up)
            else:
                 amplitude_up_score = 70 # 过杠但无法评估肘角，给较高基础分
        else:
             amplitude_up_score = 0
             feedback.append("下巴未过杠")
        
        # --- 幅度评估 (下放) ---
        if valid_elbow_angles:
            max_elbow_angle = max(valid_elbow_angles)
            self.max_angle_history.append(max_elbow_angle)
            
            if max_elbow_angle >= self.ideal_elbow_angle_down:
                 amplitude_down_score = 100
            elif max_elbow_angle < self.elbow_angle_threshold_down:
                 amplitude_down_score = 0
                 feedback.append(f"下放幅度不足 (肘角 {max_elbow_angle:.1f}° < {self.elbow_angle_threshold_down}°)")
            else:
                 amplitude_down_score = max(0, 100 * (max_elbow_angle - self.elbow_angle_threshold_down) / 
                                          (self.ideal_elbow_angle_down - self.elbow_angle_threshold_down))
        else:
            feedback.append("无法评估下放幅度 (无有效肘部角度)")
            
        # --- 姿态/稳定性评估 (手腕/肩膀对齐) ---
        alignments = self.current_rep_data.get('angles', {}).get('wrist_align', [])
        valid_alignments = [a for a in alignments if a is not None]
        if valid_alignments:
            # 计算对齐帧的比例
            aligned_ratio = sum(valid_alignments) / len(valid_alignments)
            pose_score = int(100 * aligned_ratio)
            if aligned_ratio < 0.8: # 如果大部分时间不对齐，给提示
                 feedback.append("注意保持手腕在肩膀下方")
        else:
            feedback.append("无法评估手腕对齐")
            pose_score = 50 # 给个基础分

        # --- 综合评分 ---
        total_score = int(self.w_amplitude_up * amplitude_up_score + 
                        self.w_amplitude_down * amplitude_down_score + 
                        self.w_pose * pose_score)
                        
        self.logger.debug(f"Rep {self.count} Quality: Total={total_score}, AmpUp={amplitude_up_score:.1f}, AmpDown={amplitude_down_score:.1f}, Pose={pose_score:.1f}, Feedback={feedback}")
        return total_score, feedback

    # 修改可视化数据添加
    def _add_visualization_data(self, keypoints, elbow_angle, state, chin_over_bar, wrist_alignment):
        """添加可视化数据"""
        viz_data = {
            'timestamp': time.time(),
            'elbow_angle': elbow_angle,
            'state': state,
            'count': self.count,
            'chin_over_bar': chin_over_bar,
            'wrist_alignment': wrist_alignment, # 添加手腕对齐状态
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
        self.last_elbow_angle = None
        self.rep_started = False
        self.down_position_buffer = []
        self.up_position_buffer = [] 