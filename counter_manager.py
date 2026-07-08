#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
计数器/计时器管理器 - 管理不同训练项目的计数和计时功能
"""

from PyQt6.QtCore import QObject, pyqtSignal

from .push_up_counter import PushUpCounter
from .squat_counter import SquatCounter
from .plank_timer import PlankTimer
from .sit_up_counter import SitUpCounter
from .pull_up_counter import PullUpCounter
from .tricep_dip_counter import TricepDipCounter
from .bridge_timer import BridgeTimer
#shanchu
from .high_knees_counter import HighKneesCounter
from .mountain_climbers_counter import MountainClimbersCounter



class CounterManager(QObject):
    """计数器/计时器管理器类"""
    
    # 定义信号
    count_updated = pyqtSignal(int)
    time_updated = pyqtSignal(float)
    state_changed = pyqtSignal(str)
    visualization_data = pyqtSignal(dict)
    quality_updated = pyqtSignal(list)
    feedback_updated = pyqtSignal(list)
    
    def __init__(self, parent=None):
        """初始化管理器
        
        Args:
            parent: 父对象
        """
        super().__init__(parent)
        
        # 创建各类计数器/计时器实例
        self.counters = {
            "俯卧撑": PushUpCounter(),
            "深蹲": SquatCounter(),
            "平板支撑": PlankTimer(),
            "仰卧起坐": SitUpCounter(),
            "引体向上": PullUpCounter(),
            "臂屈伸": TricepDipCounter(),
            "桥": BridgeTimer(),
            "高抬腿": HighKneesCounter(),
            "登山跑": MountainClimbersCounter()
        }

        # 当前活动的计数器
        self.active_counter = None
        self.active_counter_name = None
        
        # 连接所有计数器的信号
        for name, counter in self.counters.items():
            try:
                counter.count_updated.connect(self.count_updated)
                counter.time_updated.connect(self.time_updated)
                counter.state_changed.connect(self.state_changed)
                counter.visualization_data.connect(self.visualization_data)
                if hasattr(counter, 'quality_updated'):
                     counter.quality_updated.connect(self.quality_updated)
                if hasattr(counter, 'feedback_updated'):
                     counter.feedback_updated.connect(self.feedback_updated)
            except TypeError as e:
                 print(f"Error connecting signals for {name}: {e}") # Debug connection errors
    
    def set_active_counter(self, exercise_name):
        """设置当前激活的计数器
        
        Args:
            exercise_name: 训练项目名称
            
        Returns:
            是否成功激活计数器
        """
        if exercise_name in self.counters:
            # 重置当前计数器（如果有）
            if self.active_counter is not None:
                self.active_counter.reset()
            
            # 激活新计数器
            self.active_counter = self.counters[exercise_name]
            self.active_counter_name = exercise_name
            
            # 重置新计数器
            self.active_counter.reset()
            
            print(f"Activated counter: {self.active_counter_name}") # Debug log
            return True
        print(f"Counter not found for: {exercise_name}") # Debug log
        return False
    
    def process_keypoints(self, keypoints):
        """处理关键点数据
        
        Args:
            keypoints: 关键点数据
        """
        if self.active_counter is not None:
            self.active_counter.process(keypoints)
    
    def reset_active_counter(self):
        """重置当前活动的计数器"""
        if self.active_counter is not None:
            self.active_counter.reset()
    
    def get_counter_type(self, exercise_name):
        """获取计数器类型
        
        Args:
            exercise_name: 训练项目名称
            
        Returns:
            "count"或"time"，表示计数或计时
        """
        if exercise_name in self.counters:
            return "time" if not self.counters[exercise_name].is_counting else "count"
        return None
    
    def get_active_counter_type(self):
        """获取当前活动计数器的类型
        
        Returns:
            "count"或"time"，表示计数或计时
        """
        if self.active_counter is not None:
            return "time" if not self.active_counter.is_counting else "count"
        return None
    
    def get_active_counter_name(self):
        """获取当前活动计数器的名称
        
        Returns:
            当前活动计数器的名称
        """
        return self.active_counter_name

    def get_active_counter_thresholds(self):
        """获取当前活动计数器的角度阈值 (如果存在)，适配不同计数器类的属性名称。
        
        Returns:
            dict or None: 包含阈值信息的字典 (例如 {'upper': 160, 'lower': 30})，
                          如果计数器不存在或没有定义合适的阈值，则返回 None。
        """
        if not self.active_counter:
            return None

        thresholds = {}
        counter = self.active_counter
        counter_name = self.active_counter_name

        # 高抬腿的阈值处理
        if counter_name == "高抬腿":
            # 检查是否有角度阈值属性
            if hasattr(counter, 'knee_angle_threshold_up') and hasattr(counter, 'knee_angle_threshold_down'):
                thresholds['upper'] = counter.knee_angle_threshold_up
                thresholds['lower'] = counter.knee_angle_threshold_down
            else:
                # 如果没有，使用默认值
                thresholds['upper'] = 170
                thresholds['lower'] = 90
            return thresholds if thresholds else None

        # 登山跑的阈值处理
        elif counter_name == "登山跑":
            if hasattr(counter, 'knee_angle_threshold_up') and hasattr(counter, 'knee_angle_threshold_down'):
                thresholds['upper'] = counter.knee_angle_threshold_up
                thresholds['lower'] = counter.knee_angle_threshold_down
            else:
                thresholds['upper'] = 180
                thresholds['lower'] = 100
            return thresholds if thresholds else None



        # 优先检查计数类常用的阈值对
        if hasattr(counter, 'knee_angle_threshold_up') and hasattr(counter, 'knee_angle_threshold_down'): # Squat
            thresholds['upper'] = counter.knee_angle_threshold_up
            thresholds['lower'] = counter.knee_angle_threshold_down
        elif hasattr(counter, 'elbow_angle_threshold_up') and hasattr(counter, 'elbow_angle_threshold_down'): # PushUp, PullUp, TricepDip
            # 特殊处理引体向上：up是小角度，down是大角度
            if counter_name == "引体向上":
                 thresholds['upper'] = counter.elbow_angle_threshold_down # 下垂是大角度，作为上限
                 thresholds['lower'] = counter.elbow_angle_threshold_up   # 上拉是小角度，作为下限
            else: # 俯卧撑, 臂屈伸 (up是大角度，down是小角度)
                 thresholds['upper'] = counter.elbow_angle_threshold_up
                 thresholds['lower'] = counter.elbow_angle_threshold_down
        elif hasattr(counter, 'torso_angle_threshold_up') and hasattr(counter, 'torso_angle_threshold_down'): # SitUp
             # 仰卧起坐：up是小角度，down是大角度
             thresholds['upper'] = counter.torso_angle_threshold_down # 仰卧是大角度，作为上限
             thresholds['lower'] = counter.torso_angle_threshold_up   # 起坐是小角度，作为下限
             
        # 检查计时类可能使用的单个阈值
        elif hasattr(counter, 'hip_angle_threshold'): # Bridge Timer
            # 桥式只有一个下限阈值
            thresholds['lower'] = counter.hip_angle_threshold
        elif hasattr(counter, 'torso_angle_threshold'): # Plank Timer
            # 平板支撑的躯干角度是与垂直线的夹角，阈值表示允许的最大偏差
            # 我们可以显示一个接近 90 度的参考线，或一个表示允许范围的区域
            # 简单起见，暂不为平板支撑绘制阈值线
            pass
        elif hasattr(counter, 'elbow_angle_threshold'): # Plank Timer (elbow)
             # 平板支撑的肘部角度也是一个约束，但通常不是图表主要显示内容
             pass
             
        # 检查通用的 MIN/MAX 阈值 (以防万一)
        if not thresholds: # 如果上面的特定检查都没匹配到
            if hasattr(counter, 'MAX_ANGLE'):
                thresholds['upper'] = counter.MAX_ANGLE
            if hasattr(counter, 'MIN_ANGLE'):
                thresholds['lower'] = counter.MIN_ANGLE

        return thresholds if thresholds else None 