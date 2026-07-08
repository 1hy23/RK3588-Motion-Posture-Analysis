#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
姿态分析工具 - 提供关节角度计算和姿态分析功能
"""

import numpy as np
import math


class PoseUtils:
    """姿态分析工具类"""

    # COCO关键点索引定义
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    @staticmethod
    def calculate_angle(a, b, c):
        """计算三点间的角度
        
        Args:
            a: 第一个点的坐标 [x, y]
            b: 中间点的坐标 [x, y]
            c: 第三个点的坐标 [x, y]
            
        Returns:
            角度值(度)
        """
        # 检查输入是否有效
        if (a is None or b is None or c is None or 
            np.isnan(a).any() or np.isnan(b).any() or np.isnan(c).any()):
            return None
            
        # 将输入转换为numpy数组
        a = np.array(a[:2])  # 只使用x和y坐标
        b = np.array(b[:2])
        c = np.array(c[:2])
        
        # 计算向量
        ba = a - b
        bc = c - b
        
        # 计算点积
        dot_product = np.dot(ba, bc)
        
        # 计算模长
        magnitude_ba = np.linalg.norm(ba)
        magnitude_bc = np.linalg.norm(bc)
        
        # 避免除以零
        if magnitude_ba * magnitude_bc < 1e-10:
            return None
            
        # 计算夹角的余弦值
        cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
        
        # 处理数值误差，确保余弦值在-1到1之间
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        # 转换为角度
        angle = np.arccos(cosine_angle)
        angle = np.degrees(angle)
        
        return angle
    
    @staticmethod
    def calculate_distance(a, b):
        """计算两点间的欧氏距离
        
        Args:
            a: 第一个点的坐标 [x, y]
            b: 第二个点的坐标 [x, y]
            
        Returns:
            距离值
        """
        # 检查输入是否有效
        if (a is None or b is None or 
            np.isnan(a).any() or np.isnan(b).any()):
            return None
            
        # 将输入转换为numpy数组
        a = np.array(a[:2])  # 只使用x和y坐标
        b = np.array(b[:2])
        
        # 计算欧氏距离
        return np.linalg.norm(a - b)
    
    @staticmethod
    def calculate_vertical_distance(a, b):
        """计算两点间的垂直距离(y轴)
        
        Args:
            a: 第一个点的坐标 [x, y]
            b: 第二个点的坐标 [x, y]
            
        Returns:
            垂直距离值 (y轴方向)
        """
        # 检查输入是否有效
        if (a is None or b is None or 
            np.isnan(a).any() or np.isnan(b).any()):
            return None
            
        # 计算y轴距离
        return abs(a[1] - b[1])
    
    @staticmethod
    def calculate_horizontal_distance(a, b):
        """计算两点间的水平距离(x轴)
        
        Args:
            a: 第一个点的坐标 [x, y]
            b: 第二个点的坐标 [x, y]
            
        Returns:
            水平距离值 (x轴方向)
        """
        # 检查输入是否有效
        if (a is None or b is None or 
            np.isnan(a).any() or np.isnan(b).any()):
            return None
            
        # 计算x轴距离
        return abs(a[0] - b[0])
        
    @staticmethod
    def calculate_elbow_angle(keypoints, side='left'):
        """计算肘部角度
        
        Args:
            keypoints: 所有关键点数组
            side: 'left'或'right'，指定左侧或右侧
            
        Returns:
            肘部角度值
        """
        # 检查关键点数组是否有足够的元素
        if keypoints is None or len(keypoints) <= PoseUtils.RIGHT_WRIST:
            return None
            
        if side.lower() == 'left':
            shoulder_idx = PoseUtils.LEFT_SHOULDER
            elbow_idx = PoseUtils.LEFT_ELBOW
            wrist_idx = PoseUtils.LEFT_WRIST
        else:
            shoulder_idx = PoseUtils.RIGHT_SHOULDER
            elbow_idx = PoseUtils.RIGHT_ELBOW
            wrist_idx = PoseUtils.RIGHT_WRIST
        
        # 检查索引是否有效
        if (shoulder_idx >= len(keypoints) or 
            elbow_idx >= len(keypoints) or 
            wrist_idx >= len(keypoints)):
            return None
            
        shoulder = keypoints[shoulder_idx]
        elbow = keypoints[elbow_idx]
        wrist = keypoints[wrist_idx]
            
        return PoseUtils.calculate_angle(shoulder, elbow, wrist)
    
    @staticmethod
    def calculate_shoulder_angle(keypoints, side='left'):
        """计算肩部角度
        
        Args:
            keypoints: 所有关键点数组
            side: 'left'或'right'，指定左侧或右侧
            
        Returns:
            肩部角度值
        """
        # 检查关键点数组是否有足够的元素
        if keypoints is None or len(keypoints) <= PoseUtils.RIGHT_ELBOW:
            return None
            
        if side.lower() == 'left':
            hip_idx = PoseUtils.LEFT_HIP
            shoulder_idx = PoseUtils.LEFT_SHOULDER
            elbow_idx = PoseUtils.LEFT_ELBOW
        else:
            hip_idx = PoseUtils.RIGHT_HIP
            shoulder_idx = PoseUtils.RIGHT_SHOULDER
            elbow_idx = PoseUtils.RIGHT_ELBOW
        
        # 检查索引是否有效
        if (hip_idx >= len(keypoints) or 
            shoulder_idx >= len(keypoints) or 
            elbow_idx >= len(keypoints)):
            return None
            
        hip = keypoints[hip_idx]
        shoulder = keypoints[shoulder_idx]
        elbow = keypoints[elbow_idx]
            
        return PoseUtils.calculate_angle(hip, shoulder, elbow)
    
    @staticmethod
    def calculate_knee_angle(keypoints, side='left'):
        """计算膝盖角度
        
        Args:
            keypoints: 所有关键点数组
            side: 'left'或'right'，指定左侧或右侧
            
        Returns:
            膝盖角度值
        """
        # 检查关键点数组是否有足够的元素
        if keypoints is None or len(keypoints) <= PoseUtils.RIGHT_ANKLE:
            return None
            
        if side.lower() == 'left':
            hip_idx = PoseUtils.LEFT_HIP
            knee_idx = PoseUtils.LEFT_KNEE
            ankle_idx = PoseUtils.LEFT_ANKLE
        else:
            hip_idx = PoseUtils.RIGHT_HIP
            knee_idx = PoseUtils.RIGHT_KNEE
            ankle_idx = PoseUtils.RIGHT_ANKLE
        
        # 检查索引是否有效
        if (hip_idx >= len(keypoints) or 
            knee_idx >= len(keypoints) or 
            ankle_idx >= len(keypoints)):
            return None
            
        hip = keypoints[hip_idx]
        knee = keypoints[knee_idx]
        ankle = keypoints[ankle_idx]
        
        return PoseUtils.calculate_angle(hip, knee, ankle)
    
    @staticmethod
    def calculate_hip_angle(keypoints, side='left'):
        """计算髋部角度
        
        Args:
            keypoints: 所有关键点数组
            side: 'left'或'right'，指定左侧或右侧
            
        Returns:
            髋部角度值
        """
        # 检查关键点数组是否有足够的元素
        if keypoints is None or len(keypoints) <= PoseUtils.RIGHT_KNEE:
            return None
            
        if side.lower() == 'left':
            shoulder_idx = PoseUtils.LEFT_SHOULDER
            hip_idx = PoseUtils.LEFT_HIP
            knee_idx = PoseUtils.LEFT_KNEE
        else:
            shoulder_idx = PoseUtils.RIGHT_SHOULDER
            hip_idx = PoseUtils.RIGHT_HIP
            knee_idx = PoseUtils.RIGHT_KNEE
        
        # 检查索引是否有效
        if (shoulder_idx >= len(keypoints) or 
            hip_idx >= len(keypoints) or 
            knee_idx >= len(keypoints)):
            return None
            
        shoulder = keypoints[shoulder_idx]
        hip = keypoints[hip_idx]
        knee = keypoints[knee_idx]
            
        return PoseUtils.calculate_angle(shoulder, hip, knee)
    
    @staticmethod
    def calculate_torso_angle(keypoints):
        """计算躯干与水平线的夹角（用于检测是否保持躯干水平）
        
        Args:
            keypoints: 所有关键点数组
            
        Returns:
            躯干与水平线夹角 (0度为水平)
        """
        # 检查关键点数组是否有足够的元素
        if keypoints is None or len(keypoints) <= PoseUtils.RIGHT_HIP:
            return None
            
        # 使用左右肩膀和髋部的中点来确定躯干线
        left_shoulder_idx = PoseUtils.LEFT_SHOULDER
        right_shoulder_idx = PoseUtils.RIGHT_SHOULDER
        left_hip_idx = PoseUtils.LEFT_HIP
        right_hip_idx = PoseUtils.RIGHT_HIP
        
        # 检查索引是否有效
        if (left_shoulder_idx >= len(keypoints) or 
            right_shoulder_idx >= len(keypoints) or 
            left_hip_idx >= len(keypoints) or 
            right_hip_idx >= len(keypoints)):
            return None
            
        left_shoulder = keypoints[left_shoulder_idx]
        right_shoulder = keypoints[right_shoulder_idx]
        left_hip = keypoints[left_hip_idx]
        right_hip = keypoints[right_hip_idx]
        
        # 检查点是否有效
        if (np.isnan(left_shoulder).any() or np.isnan(right_shoulder).any() or
            np.isnan(left_hip).any() or np.isnan(right_hip).any()):
            return None
            
        # 计算肩部和髋部的中点
        shoulder_mid = (left_shoulder[:2] + right_shoulder[:2]) / 2
        hip_mid = (left_hip[:2] + right_hip[:2]) / 2
        
        # 计算躯干向量 (垂直向下为正y轴)
        torso_vector = shoulder_mid - hip_mid
        
        # 垂直向量 (垂直向下)
        vertical_vector = np.array([0, 1])
        
        # 计算两个向量的夹角
        dot_product = np.dot(torso_vector, vertical_vector)
        torso_magnitude = np.linalg.norm(torso_vector)
        
        # 避免除以零
        if torso_magnitude < 1e-10:
            return None
            
        cos_angle = dot_product / torso_magnitude  # vertical_magnitude = 1
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 处理数值误差
        
        # 计算角度 (0-180度)
        angle = np.degrees(np.arccos(cos_angle))
        
        # 判断躯干是向左还是向右倾斜
        cross_product = np.cross(np.append(torso_vector, 0), np.append(vertical_vector, 0))[2]
        if cross_product < 0:
            angle = 360 - angle
            
        # 将角度规范化到0-90度范围，表示与水平线的夹角
        if angle > 270:
            return 360 - angle
        elif angle > 180:
            return angle - 180
        elif angle > 90:
            return 180 - angle
        else:
            return angle
    
    @staticmethod
    def is_push_up_position(keypoints, threshold_angle=30):
        """检查是否处于俯卧撑姿势
        
        Args:
            keypoints: 关键点数组
            threshold_angle: 躯干与水平线的最大角度阈值
            
        Returns:
            是否处于俯卧撑姿势
        """
        # 检查关键点数组是否有足够的元素
        if keypoints is None or len(keypoints) <= PoseUtils.RIGHT_ANKLE:
            return False
            
        # 计算躯干与水平线的角度
        torso_angle = PoseUtils.calculate_torso_angle(keypoints)
        
        # 如果无法计算角度，返回False
        if torso_angle is None:
            return False
            
        # 检查躯干是否接近水平
        if torso_angle > threshold_angle:
            return False
            
        # 检查双肘是否弯曲
        left_elbow_angle = PoseUtils.calculate_elbow_angle(keypoints, 'left')
        right_elbow_angle = PoseUtils.calculate_elbow_angle(keypoints, 'right')
        
        # 至少有一侧肘部角度可计算
        if left_elbow_angle is None and right_elbow_angle is None:
            return False
            
        # 综合判断是否处于俯卧撑姿势
        return True
    
    @staticmethod
    def is_squat_position(keypoints, threshold_angle=80):
        """检查是否处于深蹲姿势
        
        Args:
            keypoints: 关键点数组
            threshold_angle: 膝盖弯曲的最大角度阈值
            
        Returns:
            是否处于深蹲姿势
        """
        # 检查关键点数组是否有足够的元素
        if keypoints is None or len(keypoints) <= PoseUtils.RIGHT_ANKLE:
            return False
            
        # 检查双膝是否弯曲
        left_knee_angle = PoseUtils.calculate_knee_angle(keypoints, 'left')
        right_knee_angle = PoseUtils.calculate_knee_angle(keypoints, 'right')
        
        # 至少有一侧膝盖角度可计算
        if left_knee_angle is None and right_knee_angle is None:
            return False
            
        # 判断膝盖弯曲程度
        if left_knee_angle is not None and left_knee_angle < threshold_angle:
            return True
        if right_knee_angle is not None and right_knee_angle < threshold_angle:
            return True
            
        return False 