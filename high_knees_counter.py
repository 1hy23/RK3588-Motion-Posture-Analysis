#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高抬腿计数器 - 检测并计数高抬腿动作
使用MediaPipe姿势关键点（33个关键点）
"""

import numpy as np
import time
import sys
import os

# ============================ 导入问题修复 ============================
# 解决导入问题 - 使用绝对路径导入
current_dir = os.path.dirname(os.path.abspath(__file__))
counters_dir = current_dir
utils_dir = os.path.dirname(counters_dir)
src_dir = os.path.dirname(utils_dir)
app_dir = os.path.dirname(src_dir)
project_root = os.path.dirname(app_dir)

# 添加多个可能的路径
paths_to_add = [
    project_root,  # D:\姿态估计健身软件-PC\
    app_dir,  # D:\姿态估计健身软件-PC\app
    src_dir,  # D:\姿态估计健身软件-PC\app\src
    utils_dir,  # D:\姿态估计健身软件-PC\app\src\utils
    counters_dir  # D:\姿态估计健身软件-PC\app\src\utils\counters
]

for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# 导入基类和工具类
try:
    # 方法1：从src模块导入
    from src.utils.counters.base_counter import ExerciseCounter
    from src.utils.counters.pose_utils import PoseUtils

    print("✓ 从src模块导入成功")
except ImportError as e:
    print(f"✗ 导入失败: {e}")


    # 定义简单的基类作为后备
    class ExerciseCounter:
        def __init__(self, confidence_threshold=0.5):
            self.confidence_threshold = confidence_threshold
            self.count = 0
            self.state = "待机"

        def reset(self):
            self.count = 0
            self.state = "待机"

        def process(self, keypoints):
            pass


    class PoseUtils:
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28


# ============================ 高抬腿计数器类 ============================
class HighKneesCounter(ExerciseCounter):
    """高抬腿计数器类"""

    def __init__(self, confidence_threshold=0.5):
        """初始化高抬腿计数器"""
        super().__init__(confidence_threshold)

        # 高抬腿特定参数 - 重新调整
        self.knee_height_threshold = 0.3  # 膝盖高度阈值（绝对值）
        self.knee_angle_threshold = 120  # 膝盖角度阈值

        # 状态变量
        self.left_knee_up = False
        self.right_knee_up = False
        self.last_left_knee_state = False
        self.last_right_knee_state = False
        self.last_count_time = 0
        self.partial_count = 0

        # 历史数据
        self.left_knee_history = []
        self.right_knee_history = []
        self.max_history = 3

        # 调试模式
        self.debug = True

        print("高抬腿计数器初始化完成")

    def _calculate_knee_height(self, keypoints, side='left'):
        """计算膝盖高度（绝对高度差）"""
        if side == 'left':
            hip_idx = PoseUtils.LEFT_HIP
            knee_idx = PoseUtils.LEFT_KNEE
        else:
            hip_idx = PoseUtils.RIGHT_HIP
            knee_idx = PoseUtils.RIGHT_KNEE

        hip = keypoints[hip_idx]
        knee = keypoints[knee_idx]

        if hip is None or knee is None:
            return None

        # 检查置信度
        if len(hip) > 2 and hip[2] < self.confidence_threshold:
            return None
        if len(knee) > 2 and knee[2] < self.confidence_threshold:
            return None

        # 计算垂直高度差（膝盖比髋部高多少）
        # 注意：在图像坐标系中，y轴向下为正，所以数值越小位置越高
        height_difference = hip[1] - knee[1]  # 膝盖越高，这个值越大

        if self.debug:
            print(f"{side}膝盖高度差: {height_difference:.3f} (髋部y={hip[1]:.3f}, 膝盖y={knee[1]:.3f})")

        return height_difference

    def _calculate_knee_angle(self, keypoints, side='left'):
        """计算膝盖角度"""
        if side == 'left':
            hip_idx = PoseUtils.LEFT_HIP
            knee_idx = PoseUtils.LEFT_KNEE
            ankle_idx = PoseUtils.LEFT_ANKLE
        else:
            hip_idx = PoseUtils.RIGHT_HIP
            knee_idx = PoseUtils.RIGHT_KNEE
            ankle_idx = PoseUtils.RIGHT_ANKLE

        hip = keypoints[hip_idx]
        knee = keypoints[knee_idx]
        ankle = keypoints[ankle_idx]

        if hip is None or knee is None or ankle is None:
            return None

        # 检查置信度
        if len(hip) > 2 and hip[2] < self.confidence_threshold:
            return None
        if len(knee) > 2 and knee[2] < self.confidence_threshold:
            return None
        if len(ankle) > 2 and ankle[2] < self.confidence_threshold:
            return None

        # 计算向量
        hip_to_knee = np.array([knee[0] - hip[0], knee[1] - hip[1]])
        ankle_to_knee = np.array([knee[0] - ankle[0], knee[1] - ankle[1]])

        # 计算角度
        dot_product = np.dot(hip_to_knee, ankle_to_knee)
        norm_product = np.linalg.norm(hip_to_knee) * np.linalg.norm(ankle_to_knee)

        if norm_product < 0.001:
            return None

        cos_angle = dot_product / norm_product
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))

        if self.debug:
            print(f"{side}膝盖角度: {angle:.1f}°")

        return angle

    def _is_knee_up(self, keypoints, side='left'):
        """判断膝盖是否抬起"""
        # 计算膝盖高度
        knee_height = self._calculate_knee_height(keypoints, side)
        if knee_height is None:
            return False

        # 计算膝盖角度
        knee_angle = self._calculate_knee_angle(keypoints, side)
        if knee_angle is None:
            return False

        # 判断条件：膝盖足够高（膝盖在髋部上方）并且膝盖有一定弯曲
        # 注意：在图像中，y坐标向下增加，所以膝盖比髋部高时，knee_y < hip_y
        height_condition = knee_height > self.knee_height_threshold
        angle_condition = knee_angle < self.knee_angle_threshold  # 角度小表示膝盖弯曲

        return height_condition and angle_condition

    def process(self, keypoints):
        """处理关键点数据"""
        if keypoints is None or not isinstance(keypoints, np.ndarray):
            self.state = "等待检测"
            return

        # 检查关键点维度
        if len(keypoints) < 33:
            self.state = "关键点不足"
            return

        current_time = time.time()

        # 检测左右膝盖状态
        left_knee_up = self._is_knee_up(keypoints, 'left')
        right_knee_up = self._is_knee_up(keypoints, 'right')

        if self.debug:
            print(f"检测结果: 左膝抬起={left_knee_up}, 右膝抬起={right_knee_up}")

        # 记录历史状态
        self.left_knee_history.append(left_knee_up)
        self.right_knee_history.append(right_knee_up)

        # 保持历史记录长度
        if len(self.left_knee_history) > self.max_history:
            self.left_knee_history.pop(0)
        if len(self.right_knee_history) > self.max_history:
            self.right_knee_history.pop(0)

        # 使用历史数据平滑状态判断
        smoothed_left_up = False
        smoothed_right_up = False

        if len(self.left_knee_history) > 0:
            smoothed_left_up = sum(self.left_knee_history) > len(self.left_knee_history) / 2

        if len(self.right_knee_history) > 0:
            smoothed_right_up = sum(self.right_knee_history) > len(self.right_knee_history) / 2

        # 检测状态变化
        left_changed = smoothed_left_up != self.last_left_knee_state
        right_changed = smoothed_right_up != self.last_right_knee_state

        # 更新显示状态
        old_state = self.state

        if smoothed_left_up and smoothed_right_up:
            self.state = "双腿抬起"
        elif smoothed_left_up:
            self.state = "左腿抬起"
        elif smoothed_right_up:
            self.state = "右腿抬起"
        else:
            self.state = "站立"

        if self.debug and old_state != self.state:
            print(f"状态变化: {old_state} -> {self.state}")

        # 检测抬腿动作完成（从抬起到放下）
        time_since_last_count = current_time - self.last_count_time

        # 左腿：从抬起变为放下
        if left_changed and not smoothed_left_up and self.last_left_knee_state:
            if time_since_last_count > 0.3:  # 最小时间间隔
                self.partial_count += 0.5
                if self.debug:
                    print(f"左腿完成抬腿，部分计数: {self.partial_count}")

        # 右腿：从抬起变为放下
        if right_changed and not smoothed_right_up and self.last_right_knee_state:
            if time_since_last_count > 0.3:  # 最小时间间隔
                self.partial_count += 0.5
                if self.debug:
                    print(f"右腿完成抬腿，部分计数: {self.partial_count}")

        # 检查是否完成一次完整的高抬腿（左右各一次）
        if self.partial_count >= 1:
            self.count += 1
            self.partial_count -= 1
            self.last_count_time = current_time
            if self.debug:
                print(f"✅ 高抬腿计数: {self.count}")

        # 保存当前状态
        self.last_left_knee_state = smoothed_left_up
        self.last_right_knee_state = smoothed_right_up

    def reset(self):
        """重置计数器"""
        super().reset()
        self.left_knee_up = False
        self.right_knee_up = False
        self.last_left_knee_state = False
        self.last_right_knee_state = False
        self.last_count_time = 0
        self.partial_count = 0
        self.left_knee_history = []
        self.right_knee_history = []
        print("高抬腿计数器已重置")

    def get_instructions(self):
        """获取高抬腿的动作指导"""
        return [
            "1. 站立，双脚与肩同宽",
            "2. 快速交替抬腿，膝盖尽量抬至腰部高度",
            "3. 保持背部挺直，核心收紧",
            "4. 手臂自然摆动，配合腿部动作",
            "5. 前脚掌着地，动作轻盈",
            "6. 保持均匀呼吸，不要憋气",
            "7. 每组30-60秒，休息30秒",
            "8. 建议每天3-5组"
        ]

    def get_common_mistakes(self):
        """获取常见错误"""
        return [
            "✗ 弯腰驼背，身体前倾",
            "✗ 膝盖抬得不够高",
            "✗ 脚掌着地过重，产生噪音",
            "✗ 手臂摆动幅度过大或过小",
            "✗ 呼吸不规律，容易疲劳",
            "✗ 动作节奏不均匀",
            "✗ 低头看脚，颈部不适"
        ]

    def get_exercise_info(self):
        """获取运动信息"""
        return {
            "name": "高抬腿",
            "category": "有氧运动",
            "calories": "8.5千卡/分钟",
            "difficulty": "初级",
            "muscles": ["股四头肌", "核心肌群", "小腿", "臀部"]
        }


# ============================ 测试代码 ============================
if __name__ == "__main__":
    print("=" * 60)
    print("高抬腿计数器测试")
    print("=" * 60)

    # 创建计数器实例
    counter = HighKneesCounter()
    print("计数器创建成功")

    # 显示信息
    print(f"\n运动名称: {counter.get_exercise_info()['name']}")
    print(f"运动类别: {counter.get_exercise_info()['category']}")

    # 创建模拟关键点
    print("\n创建模拟关键点...")

    # 初始化关键点数组 - 使用归一化坐标（0-1）
    keypoints = np.full((33, 3), None)

    # 设置站立姿势的关键点
    # 注意：图像坐标中，y轴向下为正，所以数值越大位置越低

    # 站立姿势：髋部y=0.6，膝盖y=0.75，脚踝y=0.9
    # 左腿
    keypoints[23] = [0.4, 0.6, 0.9]  # 左髋
    keypoints[25] = [0.4, 0.75, 0.9]  # 左膝（在髋部下方）
    keypoints[27] = [0.4, 0.9, 0.9]  # 左踝

    # 右腿
    keypoints[24] = [0.6, 0.6, 0.9]  # 右髋
    keypoints[26] = [0.6, 0.75, 0.9]  # 右膝
    keypoints[28] = [0.6, 0.9, 0.9]  # 右踝

    print("处理站立姿势...")
    counter.process(keypoints)
    print(f"状态: {counter.state}, 计数: {counter.count}")

    # 模拟左膝抬起 - 膝盖提到腰部位置（y=0.3）
    print("\n模拟左膝抬起...")
    keypoints[25] = [0.4, 0.3, 0.9]  # 左膝抬高到腰部
    for _ in range(3):  # 多次处理确保检测
        counter.process(keypoints)
    print(f"状态: {counter.state}, 部分计数: {counter.partial_count}")

    # 模拟左膝放下
    print("\n模拟左膝放下...")
    keypoints[25] = [0.4, 0.75, 0.9]  # 左膝恢复
    for _ in range(3):
        counter.process(keypoints)
    print(f"状态: {counter.state}, 部分计数: {counter.partial_count}")

    # 模拟右膝抬起
    print("\n模拟右膝抬起...")
    keypoints[26] = [0.6, 0.3, 0.9]  # 右膝抬高
    for _ in range(3):
        counter.process(keypoints)
    print(f"状态: {counter.state}, 部分计数: {counter.partial_count}")

    # 模拟右膝放下
    print("\n模拟右膝放下...")
    keypoints[26] = [0.6, 0.75, 0.9]  # 右膝恢复
    for _ in range(3):
        counter.process(keypoints)
    print(f"状态: {counter.state}, 部分计数: {counter.partial_count}")
    print(f"总计数: {counter.count}")

    # 测试快速交替
    print("\n测试快速交替抬腿...")
    for i in range(4):
        print(f"\n第{i + 1}轮:")

        # 左膝抬起
        keypoints[25] = [0.4, 0.35, 0.9]
        for _ in range(2):
            counter.process(keypoints)
            time.sleep(0.05)

        # 左膝放下
        keypoints[25] = [0.4, 0.75, 0.9]
        for _ in range(2):
            counter.process(keypoints)
            time.sleep(0.05)

        # 右膝抬起
        keypoints[26] = [0.6, 0.35, 0.9]
        for _ in range(2):
            counter.process(keypoints)
            time.sleep(0.05)

        # 右膝放下
        keypoints[26] = [0.6, 0.75, 0.9]
        for _ in range(2):
            counter.process(keypoints)
            time.sleep(0.05)

        print(f"  计数: {counter.count}, 部分: {counter.partial_count}")

    print("\n" + "=" * 60)
    print(f"最终计数: {counter.count}")
    print("=" * 60)