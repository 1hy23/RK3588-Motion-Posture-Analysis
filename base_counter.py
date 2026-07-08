#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
计数器/计时器基类 - 为所有训练项目提供基础功能
"""

import time
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
# import statistics  # 移除未使用的导入
from .pose_utils import PoseUtils # 添加 PoseUtils 导入


class ExerciseCounter(QObject):
    """运动计数/计时器基类"""
    
    # 信号定义
    count_updated = pyqtSignal(int)                  # 计数更新信号
    time_updated = pyqtSignal(float)                 # 计时更新信号
    state_changed = pyqtSignal(str)                  # 状态改变信号
    visualization_data = pyqtSignal(dict)            # 可视化数据信号
    quality_updated = pyqtSignal(list)               # 质量评分更新信号 (每次动作/时间段)
    feedback_updated = pyqtSignal(list)              # 反馈信息更新信号
    
    def __init__(self, confidence_threshold=0.5):
        """初始化
        
        Args:
            confidence_threshold: 关键点置信度阈值，低于此值的关键点将被忽略
        """
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.count = 0
        self.state = "待机"
        self.is_counting = True  # 是否为计数模式（否则为计时模式）
        self.last_update_time = time.time()
        self.duration = 0.0  # 持续时间（用于计时模式）
        self.history = []  # 历史数据
        
        # 数据平滑
        self.smoothing_window = 5
        self.data_buffer = []  # 数据缓冲区，用于平滑关键点数据

        # 质量评估相关
        self.quality_scores = []  # 存储每次动作/时间段的质量评分 (0-100)
        self.feedback_messages = [] # 存储每次评估的具体反馈信息
        self.current_rep_data = {} # 临时存储当前动作/时间段的数据 (如角度序列, 轨迹)
        # 添加用于记录幅度评估所需的数据列表
        self.min_angle_history = [] 
        self.max_angle_history = []
        
    def reset(self):
        """重置计数器/计时器"""
        self.count = 0
        self.duration = 0.0
        self.state = "待机"
        self.last_update_time = time.time()
        self.history = []
        self.data_buffer = []
        
        # 重置质量评估相关变量
        self.quality_scores = []
        self.feedback_messages = []
        self.current_rep_data = {}
        self.min_angle_history = []
        self.max_angle_history = []
        
        # 发送信号
        if self.is_counting:
            self.count_updated.emit(self.count)
        else:
            self.time_updated.emit(self.duration)
        self.state_changed.emit(self.state)
        # 发送空的质量和反馈信号，清空UI显示
        self.quality_updated.emit(self.quality_scores)
        self.feedback_updated.emit(self.feedback_messages)
        
    def filter_keypoints(self, keypoints):
        """过滤低置信度的关键点
        
        Args:
            keypoints: numpy数组形状为(N, 3)或(N, 2+C)，其中C是置信度
            
        Returns:
            过滤后的关键点
        """
        if keypoints is None:
            return None
            
        # 检查是否有置信度信息
        if keypoints.shape[1] >= 3:
            # 使用COCO格式 [x, y, confidence]
            valid_keypoints = keypoints.copy()
            # 将低于阈值的点置为NaN
            mask = valid_keypoints[:, 2] < self.confidence_threshold
            valid_keypoints[mask, :2] = np.nan
            return valid_keypoints
        return keypoints  # 没有置信度信息则直接返回
        
    def smooth_keypoints(self, keypoints):
        """平滑关键点数据
        
        Args:
            keypoints: numpy数组形状为(N, 3)或(N, 2+C)
            
        Returns:
            平滑后的关键点
        """
        if keypoints is None:
            return None
            
        # 添加到缓冲区
        self.data_buffer.append(keypoints.copy())
        
        # 保持缓冲区大小
        if len(self.data_buffer) > self.smoothing_window:
            self.data_buffer.pop(0)
            
        # 如果数据不足，无法平滑
        if len(self.data_buffer) < 3:
            return keypoints
            
        # 平滑处理 - 仅对坐标进行平滑，置信度保持不变
        smoothed = np.zeros_like(keypoints)
        
        for i in range(keypoints.shape[0]):
            # 收集有效数据 (不是NaN的点)
            valid_points = []
            valid_weights = []
            
            for idx, frame in enumerate(self.data_buffer):
                # 使用指数加权，最近的帧权重更高
                weight = np.exp(idx - len(self.data_buffer) + 1)
                
                # 检查是否有置信度通道
                if frame.shape[1] >= 3:
                    if frame[i, 2] >= self.confidence_threshold:
                        valid_points.append(frame[i, :2])
                        valid_weights.append(weight * frame[i, 2])  # 权重×置信度
                else:
                    # 没有置信度通道，使用所有点
                    valid_points.append(frame[i, :2])
                    valid_weights.append(weight)
            
            # 只有当有足够的有效点时才平滑
            if len(valid_points) >= 2:
                # 归一化权重
                valid_weights = np.array(valid_weights)
                valid_weights = valid_weights / np.sum(valid_weights)
                
                # 加权平均
                valid_points = np.array(valid_points)
                smoothed[i, :2] = np.sum(valid_points * valid_weights[:, np.newaxis], axis=0)
                
                # 保留原始置信度
                if keypoints.shape[1] >= 3:
                    smoothed[i, 2] = keypoints[i, 2]
            else:
                # 没有足够的有效点，保持原样
                smoothed[i] = keypoints[i]
                
        return smoothed
    
    def _start_new_rep(self):
        """开始新的一次动作重复或计时段，清空临时数据"""
        self.current_rep_data = {
            'timestamps': [],
            'keypoints_list': [],
            'angles': {}, # 存储不同角度的时间序列, e.g., {'knee': [], 'elbow': []}
            'positions': {} # 存储特定关键点的时间序列, e.g., {'hip': [], 'nose': []}
        }

    def _record_rep_data(self, timestamp, keypoints, **kwargs):
        """记录当前帧的数据到 current_rep_data"""
        if not self.current_rep_data: # 确保已经初始化
             self._start_new_rep()
             
        self.current_rep_data['timestamps'].append(timestamp)
        # 存储副本以防后续修改
        self.current_rep_data['keypoints_list'].append(keypoints.copy() if keypoints is not None else None)
        
        # 记录传入的其他数据 (通常是角度)
        for key, value in kwargs.items():
            if key not in self.current_rep_data['angles']:
                self.current_rep_data['angles'][key] = []
            self.current_rep_data['angles'][key].append(value)
            
        # (可选) 记录特定关键点位置用于轨迹分析
        # if keypoints is not None:
        #     hip_center = self._get_hip_center(keypoints) # 需要实现 _get_hip_center
        #     if 'hip_center' not in self.current_rep_data['positions']:
        #         self.current_rep_data['positions']['hip_center'] = []
        #     self.current_rep_data['positions']['hip_center'].append(hip_center)


    def _calculate_quality(self):
        """分析 current_rep_data，计算质量评分和反馈 (由子类实现)
        
        Returns:
            tuple: (总评分 (0-100), [反馈信息列表])
        """
        raise NotImplementedError("子类必须实现 _calculate_quality 方法")

    def _update_quality_and_feedback(self, score, feedback):
        """更新质量评分和反馈列表，并发送信号"""
        self.quality_scores.append(score)
        self.feedback_messages.append(feedback) # feedback 应该是一个字符串列表
        self.quality_updated.emit(self.quality_scores)
        self.feedback_updated.emit(self.feedback_messages)
        
    def _calculate_point_variance(self, points_trajectory):
        """计算点轨迹的方差 (评估稳定性)"""
        if not points_trajectory or len(points_trajectory) < 2:
            return 0.0, 0.0 # 或返回 None 表示无法计算

        # 移除 None 值
        valid_points = [p for p in points_trajectory if p is not None and not np.isnan(p).any()]
        
        if len(valid_points) < 2:
             return 0.0, 0.0

        points_array = np.array(valid_points)
        # 计算 x 和 y 坐标的标准差
        std_dev_x = np.std(points_array[:, 0])
        std_dev_y = np.std(points_array[:, 1])
        
        # 可以返回 x, y 的标准差，或者一个综合指标，例如平均标准差
        return std_dev_x, std_dev_y

    # 辅助函数：获取髋部中心点 (如果需要)
    def _get_hip_center(self, keypoints):
        """计算左右髋关节的中点"""
        # 使用导入的 PoseUtils
        if keypoints is None or len(keypoints) <= max(PoseUtils.LEFT_HIP, PoseUtils.RIGHT_HIP):
             return None
        left_hip = keypoints[PoseUtils.LEFT_HIP]
        right_hip = keypoints[PoseUtils.RIGHT_HIP]
        if np.isnan(left_hip).any() or np.isnan(right_hip).any():
             # 尝试使用单侧（如果另一侧无效）
             if not np.isnan(left_hip).any(): return left_hip[:2]
             if not np.isnan(right_hip).any(): return right_hip[:2]
             return None
        return (left_hip[:2] + right_hip[:2]) / 2

    def process(self, keypoints):
        """处理关键点数据 (由子类实现)
        
        Args:
            keypoints: 关键点数据，numpy数组形状为(N, 3)或(N, 2+C)
        """
        raise NotImplementedError("子类必须实现process方法") 