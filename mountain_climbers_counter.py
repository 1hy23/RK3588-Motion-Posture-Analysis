#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
登山跑计数器 - 基于膝盖交替动作计数
"""

import numpy as np
import time
import logging
from .base_counter import ExerciseCounter
from .pose_utils import PoseUtils


class MountainClimbersCounter(ExerciseCounter):
    """登山跑计数器"""

    def __init__(self, confidence_threshold=0.5):
        """初始化登山跑计数器

        Args:
            confidence_threshold: 关键点置信度阈值
        """
        super().__init__(confidence_threshold)

        # 登山跑特定参数
        self.knee_up_threshold = 0.15  # 膝盖抬起高度阈值（相对于肩膀）
        self.min_knee_angle = 60  # 膝盖最小角度（防止过度弯曲）
        self.min_count_interval = 0.3  # 最小计数间隔（秒）

        # 状态变量
        self.left_knee_up = False  # 左膝盖是否抬起
        self.right_knee_up = False  # 右膝盖是否抬起
        self.last_left_up = False  # 上一帧左膝盖状态
        self.last_right_up = False  # 上一帧右膝盖状态
        self.last_count_time = 0  # 上次计数时间
        self.partial_count = 0  # 部分计数（0.5为单腿完成）

        # 计数器统计
        self.left_count = 0  # 左腿完成次数
        self.right_count = 0  # 右腿完成次数

        # 评估参数
        self.knee_height_stability_threshold = 0.05  # 膝盖高度稳定性阈值
        self.timing_consistency_threshold = 0.2  # 时间一致性阈值（秒）

        # 质量评估权重
        self.w_pose = 0.4  # 姿势正确性
        self.w_stability = 0.3  # 稳定性
        self.w_consistency = 0.3  # 一致性

        # 状态变量
        self.position_buffer = []  # 姿势缓冲区
        self.buffer_size = 3  # 缓冲区大小

        # 数据可视化相关
        self.visualization_history = []
        self.max_history_size = 30

        # 计时相关
        self.last_left_up_time = 0  # 左膝盖上次抬起时间
        self.last_right_up_time = 0  # 右膝盖上次抬起时间
        self.left_timing_history = []  # 左腿动作时间历史
        self.right_timing_history = []  # 右腿动作时间历史

        # 设置为计数模式
        self.is_counting = True

        self.logger = logging.getLogger(__name__)

    def process(self, keypoints):
        """处理关键点数据，检测登山跑动作并计数"""
        current_time = time.time()
        old_state = self.state

        # 1. 检测与过滤
        person_detected = keypoints is not None and isinstance(keypoints, np.ndarray)
        smoothed_keypoints = None
        if person_detected:
            filtered_keypoints = self.filter_keypoints(keypoints)
            if filtered_keypoints is not None:
                smoothed_keypoints = self.smooth_keypoints(filtered_keypoints)

        # 2. 检查膝盖状态
        left_knee_current = False
        right_knee_current = False

        if smoothed_keypoints is not None:
            left_knee_current = self._is_knee_up(smoothed_keypoints, 'left')
            right_knee_current = self._is_knee_up(smoothed_keypoints, 'right')

        # 3. 使用缓冲区平滑状态判断
        self.position_buffer.append((left_knee_current, right_knee_current))
        if len(self.position_buffer) > self.buffer_size:
            self.position_buffer.pop(0)

        # 计算缓冲区内主要状态
        if self.position_buffer:
            left_avg = sum([state[0] for state in self.position_buffer]) / len(self.position_buffer)
            right_avg = sum([state[1] for state in self.position_buffer]) / len(self.position_buffer)

            self.left_knee_up = left_avg > 0.5
            self.right_knee_up = right_avg > 0.5

        # 4. 更新状态显示
        if self.left_knee_up and self.right_knee_up:
            self.state = "双腿抬起"
        elif self.left_knee_up:
            self.state = "左腿抬起"
        elif self.right_knee_up:
            self.state = "右腿抬起"
        else:
            self.state = "起始姿势"

        # 5. 检测状态变化并计数
        time_since_last_count = current_time - self.last_count_time

        # 左腿：从抬起变为放下
        if not self.left_knee_up and self.last_left_up:
            if time_since_last_count > self.min_count_interval:
                self.partial_count += 0.5
                self.left_count += 1
                self.last_left_up_time = current_time

                # 记录左腿动作时间
                self.left_timing_history.append(current_time)
                if len(self.left_timing_history) > 10:
                    self.left_timing_history.pop(0)

                self.logger.debug(f"左腿完成动作，部分计数: {self.partial_count}, 左腿总数: {self.left_count}")

        # 右腿：从抬起变为放下
        if not self.right_knee_up and self.last_right_up:
            if time_since_last_count > self.min_count_interval:
                self.partial_count += 0.5
                self.right_count += 1
                self.last_right_up_time = current_time

                # 记录右腿动作时间
                self.right_timing_history.append(current_time)
                if len(self.right_timing_history) > 10:
                    self.right_timing_history.pop(0)

                self.logger.debug(f"右腿完成动作，部分计数: {self.partial_count}, 右腿总数: {self.right_count}")

        # 6. 完成一次完整的登山跑（左右各一次）
        if self.partial_count >= 1:
            self.count += 1
            self.partial_count -= 1
            self.last_count_time = current_time

            # 记录数据用于质量评估
            if smoothed_keypoints is not None:
                self._record_rep_data(current_time, smoothed_keypoints)

            self.logger.debug(f"✅ 登山跑计数: {self.count}")

        # 7. 保存当前状态
        self.last_left_up = self.left_knee_up
        self.last_right_up = self.right_knee_up

        # 8. 发送状态变更信号
        if old_state != self.state:
            self.state_changed.emit(self.state)

        # 9. 更新可视化数据
        if smoothed_keypoints is not None:
            # 获取膝盖高度和角度用于可视化
            left_knee_height = self._get_knee_height(smoothed_keypoints, 'left')
            right_knee_height = self._get_knee_height(smoothed_keypoints, 'right')
            left_knee_angle = self._get_knee_angle(smoothed_keypoints, 'left')
            right_knee_angle = self._get_knee_angle(smoothed_keypoints, 'right')

            self._add_visualization_data(
                smoothed_keypoints,
                left_knee_height,
                right_knee_height,
                left_knee_angle,
                right_knee_angle,
                self.state,
                self.count
            )

    def _is_knee_up(self, keypoints, side='left'):
        """检查膝盖是否抬起"""
        knee_height = self._get_knee_height(keypoints, side)
        if knee_height is None:
            return False

        # 膝盖高度超过阈值视为抬起
        return knee_height > self.knee_up_threshold

    def _get_knee_height(self, keypoints, side='left'):
        """获取膝盖相对于肩膀的高度"""
        if side == 'left':
            shoulder_idx = PoseUtils.LEFT_SHOULDER
            knee_idx = PoseUtils.LEFT_KNEE
        else:
            shoulder_idx = PoseUtils.RIGHT_SHOULDER
            knee_idx = PoseUtils.RIGHT_KNEE

        # 检查关键点是否存在
        if (keypoints is None or len(keypoints) <= max(shoulder_idx, knee_idx) or
                keypoints[shoulder_idx] is None or keypoints[knee_idx] is None):
            return None

        try:
            shoulder = keypoints[shoulder_idx]
            knee = keypoints[knee_idx]

            # 检查置信度
            shoulder_conf = shoulder[2] if len(shoulder) > 2 else 0.0
            knee_conf = knee[2] if len(knee) > 2 else 0.0

            if shoulder_conf < self.confidence_threshold or knee_conf < self.confidence_threshold:
                return None

            shoulder_y = float(shoulder[1])
            knee_y = float(knee[1])

            # 在图像坐标系中，y坐标向下增加
            # 膝盖比肩膀高时，knee_y < shoulder_y
            height = shoulder_y - knee_y  # 膝盖越高，这个值越大

            return height
        except Exception as e:
            self.logger.debug(f"计算膝盖高度出错: {e}")
            return None

    def _get_knee_angle(self, keypoints, side='left'):
        """获取膝盖角度"""
        if side == 'left':
            hip_idx = PoseUtils.LEFT_HIP
            knee_idx = PoseUtils.LEFT_KNEE
            ankle_idx = PoseUtils.LEFT_ANKLE
        else:
            hip_idx = PoseUtils.RIGHT_HIP
            knee_idx = PoseUtils.RIGHT_KNEE
            ankle_idx = PoseUtils.RIGHT_ANKLE

        # 检查关键点是否存在
        if (keypoints is None or len(keypoints) <= max(hip_idx, knee_idx, ankle_idx) or
                keypoints[hip_idx] is None or keypoints[knee_idx] is None or keypoints[ankle_idx] is None):
            return None

        try:
            # 获取坐标
            hip = keypoints[hip_idx]
            knee = keypoints[knee_idx]
            ankle = keypoints[ankle_idx]

            # 检查置信度
            hip_conf = hip[2] if len(hip) > 2 else 0.0
            knee_conf = knee[2] if len(knee) > 2 else 0.0
            ankle_conf = ankle[2] if len(ankle) > 2 else 0.0

            if (hip_conf < self.confidence_threshold or
                    knee_conf < self.confidence_threshold or
                    ankle_conf < self.confidence_threshold):
                return None

            # 计算角度
            angle = PoseUtils.calculate_angle(hip[:2], knee[:2], ankle[:2])
            return angle
        except Exception as e:
            self.logger.debug(f"计算膝盖角度出错: {e}")
            return None

    def _calculate_quality(self):
        """计算登山跑动作质量"""
        feedback = []
        pose_score = 0
        stability_score = 0
        consistency_score = 0

        # 1. 姿势正确性评估（膝盖高度）
        knee_heights = self.current_rep_data.get('heights', {}).get('knee', [])
        valid_heights = [h for h in knee_heights if h is not None]

        if valid_heights:
            avg_height = np.mean(valid_heights)
            if avg_height >= self.knee_up_threshold:
                pose_score = 100
            else:
                pose_score = max(0, 100 * avg_height / self.knee_up_threshold)
                feedback.append(f"平均抬腿高度不足 ({avg_height:.3f} < {self.knee_up_threshold})")
        else:
            feedback.append("无法评估抬腿高度（数据不足）")

        # 2. 稳定性评估（膝盖高度标准差）
        if len(valid_heights) > 1:
            std_dev_height = np.std(valid_heights)
            if std_dev_height <= self.knee_height_stability_threshold:
                stability_score = 100
            else:
                stability_score = max(0, 100 - 100 * std_dev_height / self.knee_height_stability_threshold)
                feedback.append(f"抬腿高度不稳定（标准差 {std_dev_height:.3f} > {self.knee_height_stability_threshold})")
        else:
            feedback.append("无法评估稳定性（数据不足）")

        # 3. 一致性评估（左右腿时间间隔）
        if len(self.left_timing_history) > 1 and len(self.right_timing_history) > 1:
            # 计算左腿动作间隔
            left_intervals = []
            for i in range(1, len(self.left_timing_history)):
                left_intervals.append(self.left_timing_history[i] - self.left_timing_history[i - 1])

            # 计算右腿动作间隔
            right_intervals = []
            for i in range(1, len(self.right_timing_history)):
                right_intervals.append(self.right_timing_history[i] - self.right_timing_history[i - 1])

            if left_intervals and right_intervals:
                avg_left_interval = np.mean(left_intervals)
                avg_right_interval = np.mean(right_intervals)

                # 计算左右腿间隔差异
                interval_diff = abs(avg_left_interval - avg_right_interval)
                if interval_diff <= self.timing_consistency_threshold:
                    consistency_score = 100
                else:
                    consistency_score = max(0, 100 - 100 * interval_diff / self.timing_consistency_threshold)
                    feedback.append(
                        f"左右腿节奏不一致（差异 {interval_diff:.2f}s > {self.timing_consistency_threshold}s）")
        else:
            feedback.append("无法评估动作一致性（数据不足）")

        # 4. 综合评分
        total_score = int(
            self.w_pose * pose_score +
            self.w_stability * stability_score +
            self.w_consistency * consistency_score
        )

        self.logger.debug(
            f"登山跑质量: 总分={total_score}, 姿势={pose_score:.1f}, 稳定={stability_score:.1f}, 一致={consistency_score:.1f}")

        return total_score, feedback

    def _add_visualization_data(self, keypoints, left_height, right_height, left_angle, right_angle, state, count):
        """添加可视化数据"""
        viz_data = {
            'timestamp': time.time(),
            'left_knee_height': left_height,
            'right_knee_height': right_height,
            'left_knee_angle': left_angle,
            'right_knee_angle': right_angle,
            'state': state,
            'count': count,
            'left_knee_up': self.left_knee_up,
            'right_knee_up': self.right_knee_up,
            'partial_count': self.partial_count,
            'last_score': self.quality_scores[-1] if self.quality_scores else None,
            'last_feedback': self.feedback_messages[-1] if self.feedback_messages else []
        }

        self.visualization_history.append(viz_data)
        if len(self.visualization_history) > self.max_history_size:
            self.visualization_history.pop(0)

        # 发送可视化数据
        vis_data_for_ui = {
            'left_height': left_height,
            'right_height': right_height,
            'left_angle': left_angle,
            'right_angle': right_angle,
            'state': state,
            'count': count
        }
        self.visualization_data.emit(vis_data_for_ui)

    def reset(self):
        """重置计数器状态"""
        super().reset()
        self.left_knee_up = False
        self.right_knee_up = False
        self.last_left_up = False
        self.last_right_up = False
        self.last_count_time = 0
        self.partial_count = 0
        self.left_count = 0
        self.right_count = 0
        self.position_buffer = []
        self.left_timing_history = []
        self.right_timing_history = []
        self.last_left_up_time = 0
        self.last_right_up_time = 0

    def get_stats(self):
        """获取详细统计信息"""
        return {
            "总次数": self.count,
            "左腿次数": self.left_count,
            "右腿次数": self.right_count,
            "当前状态": self.state,
            "部分计数": self.partial_count,
            "质量评分": self.quality_scores[-1] if self.quality_scores else None
        }

    def get_instructions(self):
        """获取登山跑的动作指导"""
        return [
            "1. 起始姿势：平板支撑姿势，双手与肩同宽",
            "2. 身体保持一条直线，核心收紧",
            "3. 快速交替将膝盖向胸部方向提膝",
            "4. 保持肩膀稳定，避免上下晃动",
            "5. 呼吸均匀，不要憋气",
            "6. 动作要快速、连贯，像在登山一样",
            "7. 每组30-60秒，休息30秒",
            "8. 建议每天3-5组"
        ]

    def get_exercise_info(self):
        """获取运动信息"""
        return {
            "name": "登山跑",
            "english_name": "Mountain Climbers",
            "category": "有氧运动/核心训练",
            "calories": "8-10千卡/分钟",
            "difficulty": "中级",
            "muscles": ["核心肌群", "股四头肌", "臀大肌", "肩部", "胸部"],
            "benefits": [
                "增强核心力量",
                "提高心肺功能",
                "燃脂效果显著",
                "提升协调性",
                "全身性训练"
            ]
        }

    def get_common_mistakes(self):
        """获取常见错误"""
        return [
            "✗ 臀部抬得过高或塌腰",
            "✗ 肩膀超过手腕，导致肩部压力过大",
            "✗ 膝盖提得不够高，动作幅度太小",
            "✗ 身体左右晃动，核心没有收紧",
            "✗ 呼吸不规律，容易疲劳",
            "✗ 动作节奏不均匀，时快时慢"
        ]