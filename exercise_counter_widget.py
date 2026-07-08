#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练计数/计时控件 - 显示运动计数、计时和数据可视化
"""

import time
import numpy as np
# --- 移除 matplotlib 导入 ---
# import matplotlib
# matplotlib.use('QtAgg')
# from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import matplotlib.font_manager as fm
# import matplotlib.lines as mlines
# ---------------------------

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QFrame, QSizePolicy, QSplitter, QGridLayout, QTextEdit,
                             QProgressBar, QSpacerItem)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer, QSize
from PyQt6.QtGui import QFont, QIcon, QColor, QTextCursor, QLinearGradient, QPainter, QBrush
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# --- 添加 PyQtGraph 导入 ---
import pyqtgraph as pg
# ---------------------------

# --- 移除字体设置 (PyQtGraph 通常使用 Qt 的字体设置) ---
# try:
#    ...
# except Exception as e:
#    ...
# --------------------------------------------------------

# --- 移除旧的 RealtimeGraph (FigureCanvas) 类 --- 
# class RealtimeGraph(FigureCanvas): ...
# ------------------------------------------------

# --- 重写 RealtimeAngleChart 使用 PyQtGraph --- 
class RealtimeAngleChart(pg.PlotWidget):
    """使用 PyQtGraph 实现的实时角度变化图表 (增强样式 V3)"""
    def __init__(self, parent=None, max_points=150):
        super().__init__(parent=parent)
        self.max_points = max_points
        self.window_size_seconds = 10 # X轴显示时间窗口 (秒)
        self.target_x_pos_ratio = 0.7 # 最新数据点目标位置比例 (从左到右, 0.0 - 1.0)
        
        self.time_data = np.array([])
        self.angle_data = np.array([])
        self.start_time = None
        self.threshold_lines = {}

        # --- 现代高级配色方案 ---
        self.primary_color = QColor('#4361EE')  # 主曲线颜色
        # --- 调整 Alpha 值，使顶部更不透明，底部更透明 ---
        self.fill_gradient_start_color = QColor(67, 97, 238, 100) # 顶部 Alpha 增大
        self.fill_gradient_end_color = QColor(67, 97, 238, 0)    # 底部 Alpha 减小至完全透明
        # ------------------------------------------
        self.axis_pen_color = QColor('#CBD5E1')  # 轴线颜色
        self.axis_text_color = QColor('#64748B')  # 轴文本颜色
        self.grid_pen_color = QColor('#E2E8F0')  # 网格线颜色
        self.threshold_upper_color = QColor('#F59E0B')  # 上阈值颜色
        self.threshold_lower_color = QColor('#EF4444')  # 下阈值颜色
        self.background_color = QColor(0, 0, 0, 0)  # 完全透明背景
        # -----------------------

        # 设置透明背景
        self.setBackground(self.background_color)
        
        self.setup_plot()
        
        # --- 创建数据曲线 (更现代的样式) ---
        self.data_line = self.plot(
            self.time_data, 
            self.angle_data, 
            pen=pg.mkPen(
                color=self.primary_color, 
                width=3.5,  # 更粗的线条
                cosmetic=True  # 确保线条在缩放时保持宽度
            )
        )
        
        # --- 添加优雅的填充区域 ---
        self.zero_line = self.plot(
            [], [],
            pen=pg.mkPen(color=QColor(0, 0, 0, 0))  # 透明线
        )

        # 创建从上至下（底部靠近0透明，顶部远离0加深）渐变填充
        gradient = QLinearGradient(0, 0, 0, 1) # 从上(y=0)到下(y=1)
        gradient.setCoordinateMode(QLinearGradient.CoordinateMode.ObjectBoundingMode)
        # 颜色：顶部(y=0, 远离0轴)更深，底部(y=1, 靠近0轴)更透明
        gradient.setColorAt(0, self.fill_gradient_start_color) # 顶部 (y=0) 最深
        gradient.setColorAt(0.3, QColor(67, 97, 238, 60))   # 中间过渡调整
        gradient.setColorAt(0.7, QColor(67, 97, 238, 20))   # 中间过渡调整
        gradient.setColorAt(1, self.fill_gradient_end_color)    # 底部 (y=1) 最透明
        fill_brush = QBrush(gradient)

        # 添加填充区域
        self.fill_item = pg.FillBetweenItem(
            self.data_line, 
            self.zero_line, 
            brush=fill_brush
        )
        self.addItem(self.fill_item)

    def setup_plot(self):
        """设置 PyQtGraph 图表样式 (现代高级美观版)"""
        plot_item = self.getPlotItem()
        
        # --- 左侧 Y 轴 (现代简约风格) --- 
        left_axis = plot_item.getAxis('left')
        left_axis.setPen(pg.mkPen(color=self.axis_pen_color, width=1))
        left_axis.setTextPen(pg.mkPen(color=self.axis_text_color))
        left_axis.setLabel(
            '角度', 
            units='°', 
            **{
                'color': self.axis_text_color.name(), 
                'font-size': '9pt',
                'font-weight': 'bold'
            }
        )
        left_axis.setWidth(45)  # 稍微宽一点的轴
        left_axis.setTickFont(QFont("Arial", 8))  # 稍大的刻度字体
        
        # 优化刻度显示
        left_axis.setTicks([
            [(tick, str(tick)) for tick in range(0, 181, 30)],  # 主刻度
            [(tick, "") for tick in range(0, 181, 10) if tick % 30 != 0]  # 次刻度
        ])

        # --- 底部 X 轴 (彻底隐藏) ---
        plot_item.hideAxis('bottom')

        # 设置 Y 轴范围 (稍微扩大以提供视觉空间)
        plot_item.setYRange(-8, 188, padding=0)

        # 设置初始 X 轴范围
        plot_item.setXRange(0, self.window_size_seconds, padding=0.02)

        # 网格线 (更精致的样式)
        plot_item.showGrid(x=False, y=True)
        left_axis.setGrid(120)  # 更轻的网格透明度
        grid_pen = pg.mkPen(
            color=self.grid_pen_color, 
            width=0.8,  # 更细的网格线
            style=Qt.PenStyle.DotLine  # 点线风格更现代
        )
        left_axis.gridPen = grid_pen

        # 视图设置
        view_box = plot_item.getViewBox()
        view_box.setMouseEnabled(x=False, y=False)
        view_box.setLimits(xMin=0, yMin=-10, yMax=190)
        view_box.disableAutoRange()
        
        # 移除边框
        plot_item.getViewBox().setBorder(pen=None)

    def update_config(self, title=None, ylabel=None, y_range=None):
        """更新图表配置 (现代风格)"""
        if ylabel:
            self.getAxis('left').setLabel(
                ylabel, 
                units='°', 
                **{
                    'color': self.axis_text_color.name(), 
                    'font-size': '9pt',
                    'font-weight': 'bold'
                }
            )
        if y_range and len(y_range) == 2:
            self.setYRange(y_range[0], y_range[1], padding=0.03)  # 稍微增加padding

    def set_thresholds(self, thresholds):
        """设置阈值线 (现代高级样式)"""
        # 清除现有阈值线
        for line in self.threshold_lines.values(): 
            self.removeItem(line)
        self.threshold_lines.clear()

        if thresholds and isinstance(thresholds, dict):
            # 定义更现代的阈值线样式
            pens = {
                'upper': pg.mkPen(
                    color=self.threshold_upper_color, 
                    width=2.0,  # 稍粗的线条
                    style=Qt.PenStyle.DashLine,
                    cosmetic=True  # 确保线条在缩放时保持宽度
                ),
                'lower': pg.mkPen(
                    color=self.threshold_lower_color, 
                    width=2.0,
                    style=Qt.PenStyle.DashLine,
                    cosmetic=True
                )
            }
            
            # 添加阈值线
            for name, value in thresholds.items():
                if value is not None and name in pens:
                    # 创建带标签的阈值线
                    line = pg.InfiniteLine(
                        pos=value, 
                        angle=0, 
                        pen=pens[name], 
                        movable=False,
                        label=f"{name.title()}: {value}°",  # 添加标签
                        labelOpts={
                            'position': 0.97,  # 靠右侧
                            'color': pens[name].color(),
                            'fill': QColor(30, 30, 30, 40),  # 半透明背景
                            'movable': True
                        }
                    )
                    self.addItem(line)
                    self.threshold_lines[name] = line

    def add_data_point(self, angle_value):
        """添加新数据点并更新滚动位置 (优化版)"""
        if angle_value is None or not np.isfinite(angle_value):
            return
            
        current_time = time.time()
        if self.start_time is None: 
            self.start_time = current_time
        display_time = current_time - self.start_time
            
        # --- 添加数据点前，确保填充项可见 --- 
        if not self.fill_item.isVisible():
            self.fill_item.show()
        # --------------------------------
            
        # 添加数据点
        self.time_data = np.append(self.time_data, display_time)
        self.angle_data = np.append(self.angle_data, angle_value)

        # 限制数据点数量 (保持性能)
        if len(self.time_data) > self.max_points:
            self.time_data = self.time_data[-self.max_points:]
            self.angle_data = self.angle_data[-self.max_points:]

        # 更新数据线和填充
        try:
            self.data_line.setData(self.time_data, self.angle_data)
            
            # 更新零线 (用于填充)
            zeros = np.zeros_like(self.time_data)
            self.zero_line.setData(self.time_data, zeros)
        except Exception as e:
            print(f"[RealtimeAngleChart] setData 错误: {e}")
        
        # 平滑滚动效果
        if display_time > 0 and self.target_x_pos_ratio > 0:
            # 计算理想的视图范围
            target_left_edge = max(0, display_time - self.target_x_pos_ratio * self.window_size_seconds)
            target_right_edge = target_left_edge + self.window_size_seconds
            
            # 设置新的X轴范围 (无padding以获得精确控制)
            self.setXRange(target_left_edge, target_right_edge, padding=0) 
        else:
            # 初始视图
            self.setXRange(0, self.window_size_seconds, padding=0.02)
        
    def clear_data(self):
        """清除数据并重置图表 (现代版)"""
        # 清空数据
        self.time_data = np.array([])
        self.angle_data = np.array([])
        self.start_time = None
        
        # 更新视觉元素
        self.data_line.setData([], [])
        self.zero_line.setData([], [])

        # --- 显式移除并重新添加填充项以确保清除 ---
        # if self.fill_item in self.items():
        #     self.removeItem(self.fill_item)
        # self.addItem(self.fill_item)
        # --- 改为隐藏填充项 --- 
        if self.fill_item:
             self.fill_item.hide()
        # --------------------

        # 重置视图
        self.setXRange(0, self.window_size_seconds, padding=0.02)
# --------------------------------------------------

class ExerciseCounterWidget(QFrame):
    """训练计数/计时控件类 - 优化UI设计 (V3)"""

    # 信号定义
    reset_counter = pyqtSignal()  # 重置计数器信号

    # 定义基于持续时间计时的运动类型
    DURATION_TIMER_EXERCISES = ["平板支撑", "桥"]

    def __init__(self, parent=None, exercise_name=""):
        super().__init__(parent)
        self.exercise_name = exercise_name
        self.is_counting_mode = True  # 默认是计数模式
        self.elapsed_time = 0.0
        self.start_time = None
        self.display_thresholds = None # 用于存储阈值供图表使用

        # 获取应用程序设置 (模仿 WorkoutPage 获取方式)
        self.app_settings = self.get_app_settings()

        self.init_ui()
        self.update_exercise_display() # 初始化时更新显示
        self.set_counting_mode(True) # 默认初始化为计数模式

    def get_app_settings(self):
        """获取应用程序设置 (与 WorkoutPage 保持一致)"""
        # 实际应用中应从共享配置获取
        return {
            'theme_color': '#4361EE',      # 主题蓝
            'accent_color': '#0ea5e9',     # 强调蓝
            'success_color': '#10B981',    # 成功绿
            'warning_color': '#F59E0B',    # 警告黄
            'danger_color': '#EF4444',     # 危险红
            'background_color': '#F8FAFC', # 更柔和的背景色
            'card_background': '#FFFFFF',  # 卡片背景白色
            'text_color': '#1E293B',      # 深灰色文本
            'secondary_text': '#64748B', # 灰色次要文本
            'border_color': '#E2E8F0',      # 边框颜色
            'primary_button': f"""
                QPushButton {{
                    background-color: #4361EE; color: white; border: none; border-radius: 6px;
                    padding: 8px 14px; font-size: 13px; font-weight: 500;
                }}
                QPushButton:hover {{ background-color: #3A56D4; }}
                QPushButton:pressed {{ background-color: #2E4BBD; }}
            """,
            'secondary_button': f"""
                 QPushButton {{
                    background-color: #F1F5F9; color: #64748B;
                    border: 1px solid #E2E8F0; border-radius: 6px;
                    padding: 8px 14px; font-size: 13px; font-weight: 500;
                }}
                QPushButton:hover {{ background-color: #E2E8F0; border-color: #CBD5E1; }}
                QPushButton:pressed {{ background-color: #CBD5E1; }}
            """
        }

    def init_ui(self):
        """初始化UI (现代前端风格)"""
        self.setFrameShape(QFrame.Shape.StyledPanel)
        # --- 主容器样式 --- 
        self.setStyleSheet(f"""
            ExerciseCounterWidget {{
                background-color: {self.app_settings['card_background']};
                border-radius: 12px;
                border: 1px solid {self.app_settings['border_color']};
            }}
            QLabel {{ /* 默认 Label 样式 */
                color: {self.app_settings['text_color']};
                background-color: transparent;
                border: none;
                font-size: 13px; /* 默认字号 */
            }}
             /* 通用标题样式 */
            QLabel[objectName="sectionTitle"] {{
                 font-size: 14px;
                 font-weight: 500;
                 color: {self.app_settings['secondary_text']};
                 padding-bottom: 4px; /* 标题和内容间距 */
            }}
            /* 反馈文本框 */
            QTextEdit {{
                background-color: {self.app_settings['background_color']}; /* 使用主背景色 */
                border: 1px solid {self.app_settings['border_color']};
                border-radius: 8px;
                padding: 8px 10px;
                color: {self.app_settings['secondary_text']};
                font-size: 13px; /* 反馈文字大小 */
            }}
            /* 进度条 */
            QProgressBar {{
                border: none; border-radius: 6px; 
                background-color: #EAF0F6; /* 更浅的背景 */
                text-align: center; 
                color: {self.app_settings['text_color']};
                font-size: 10px; 
                font-weight: 500;
                height: 10px; /* 稍微减小高度 */
            }}
            QProgressBar::chunk {{
                border-radius: 6px;
                /* 颜色由 update_quality 动态设置 */
            }}
        """)
        self.setMinimumWidth(320)
        
        # --- 主垂直布局 --- 
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20) # 统一内边距
        main_layout.setSpacing(18) # 增大主要部分间距

        # --- 1. 顶部标题和重置按钮 --- 
        top_bar_layout = QHBoxLayout()
        top_bar_layout.setContentsMargins(0, 0, 0, 0)
        self.title_label = QLabel(self.exercise_name or "训练数据")
        self.title_label.setStyleSheet(f"""
            QLabel {{
                font-size: 19px; /* 增大标题 */
                font-weight: 600;
                color: {self.app_settings['text_color']};
            }}
        """)
        self.reset_button = QPushButton("重置")
        self.reset_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reset_button.setStyleSheet(self.app_settings['secondary_button'])
        self.reset_button.setFixedSize(QSize(70, 32))
        self.reset_button.clicked.connect(self.reset_counter_clicked)
        top_bar_layout.addWidget(self.title_label, alignment=Qt.AlignmentFlag.AlignLeft)
        top_bar_layout.addStretch()
        top_bar_layout.addWidget(self.reset_button, alignment=Qt.AlignmentFlag.AlignRight)
        main_layout.addLayout(top_bar_layout)

        # --- 2. 核心数据展示 (加一些背景或边框?) ---
        # 可以考虑将 data_layout 放入一个 QFrame 中增加视觉分组
        data_frame = QFrame() # 可选的容器
        data_frame.setStyleSheet("background-color: transparent; border: none;") # 默认透明
        data_layout = QHBoxLayout(data_frame) 
        data_layout.setContentsMargins(0, 5, 0, 5) 
        data_layout.setSpacing(25) # 增大计数和状态间距
        
        # 左侧: 计数/计时
        self.value_layout = QVBoxLayout()
        self.value_layout.setSpacing(0) # 紧凑数值和单位
        self.value_label = QLabel("0")
        self.value_label.setStyleSheet(f"""
            QLabel {{
                font-size: 56px; /* 再次增大 */
                font-weight: 600; /* 不用最粗 */
                color: {self.app_settings['theme_color']};
                qproperty-alignment: 'AlignLeft | AlignVCenter'; /* 垂直居中对齐 */
                padding-bottom: 0px; /* 移除底部填充 */
                line-height: 1; /* 尝试控制行高 */
            }}
        """)
        self.unit_label = QLabel("次")
        self.unit_label.setStyleSheet(f"""
            QLabel {{
                font-size: 16px; /* 减小单位字号 */
                font-weight: 400; /* 普通字重 */
                color: {self.app_settings['secondary_text']};
                padding-left: 5px; /* 轻微左移 */
                qproperty-alignment: 'AlignLeft | AlignVCenter'; /* 垂直居中对齐 */
            }}
        """)
        value_unit_layout = QHBoxLayout() 
        value_unit_layout.setSpacing(4) # 调整数值和单位间距
        value_unit_layout.addWidget(self.value_label)
        value_unit_layout.addWidget(self.unit_label, alignment=Qt.AlignmentFlag.AlignBottom) # 单位底部对齐
        value_unit_layout.addStretch()
        self.value_layout.addLayout(value_unit_layout)

        # 右侧: 状态
        self.status_layout = QVBoxLayout()
        self.status_layout.setSpacing(6) 
        status_title_label = QLabel("状态")
        status_title_label.setObjectName("sectionTitle") # 应用通用标题样式
        self.status_label = QLabel("未开始")
        # 状态标签样式通过 update_status 动态设置
        self.status_layout.addWidget(status_title_label, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        self.status_layout.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        self.status_layout.addStretch()

        data_layout.addLayout(self.value_layout, 1) # 计数/计时占主要空间
        data_layout.addLayout(self.status_layout, 0) # 状态靠右
        main_layout.addWidget(data_frame) # 添加容器 (如果使用)
        # main_layout.addLayout(data_layout) # 直接添加布局 (如果不使用容器)

        # --- 3. 质量评分区域 ---
        quality_layout = QVBoxLayout()
        quality_layout.setSpacing(6) 
        quality_title = QLabel("动作质量")
        quality_title.setObjectName("sectionTitle")
        self.quality_progress_bar = QProgressBar()
        # 进度条样式已在类级别设置
        self.quality_description_label = QLabel("暂无评分")
        self.quality_description_label.setStyleSheet(f"""
            QLabel {{
                font-size: 12px; /* 减小描述字号 */
                color: {self.app_settings['secondary_text']};
                padding-top: 2px; /* 微调上方间距 */
            }}
        """)
        quality_layout.addWidget(quality_title)
        quality_layout.addWidget(self.quality_progress_bar)
        quality_layout.addWidget(self.quality_description_label)
        main_layout.addLayout(quality_layout)

        # --- 4. 实时反馈区域 (使用 QLabel 替代 QTextEdit) ---
        feedback_layout = QVBoxLayout()
        feedback_layout.setSpacing(6) 
        feedback_title = QLabel("实时反馈")
        feedback_title.setObjectName("sectionTitle")
        
        # --- 使用 QLabel --- 
        self.feedback_label = QLabel("-") # 初始显示一个占位符
        self.feedback_label.setWordWrap(True)
        self.feedback_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.feedback_label.setMinimumHeight(40) # 设置最小高度防止跳动
        self.feedback_label.setStyleSheet(f"""
            QLabel {{
                color: {self.app_settings['secondary_text']};
                font-size: 13px;
                line-height: 1.4; /* 调整行高 */
                padding: 5px 0px; /* 微调垂直内边距 */
            }}
        """)
        # ----------------
        
        feedback_layout.addWidget(feedback_title)
        feedback_layout.addWidget(self.feedback_label)
        main_layout.addLayout(feedback_layout)

        # --- 5. 图表可视化区域 ---
        chart_layout = QVBoxLayout()
        chart_layout.setContentsMargins(0, 10, 0, 0) 
        chart_layout.setSpacing(6) 
        chart_title = QLabel("角度变化趋势")
        chart_title.setObjectName("sectionTitle")
        self.angle_chart = RealtimeAngleChart(max_points=150)
        self.angle_chart.setMinimumHeight(180) # 调整高度
        chart_layout.addWidget(chart_title)
        chart_layout.addWidget(self.angle_chart)
        main_layout.addLayout(chart_layout) 
        main_layout.setStretchFactor(chart_layout, 1) # 让图表区域获得更多垂直空间

        # 初始状态设置
        self.update_count(0)
        self.update_status("未开始")
        self.update_quality([]) # 确保应用初始样式

    def reset_counter_clicked(self):
        """处理重置按钮点击"""
        self.elapsed_time = 0.0
        if self.start_time: # 如果计时器在运行
            self.start_time = time.time() # 重置开始时间戳
        else: # 如果计时器未运行
            self.update_time(0) # 直接更新显示为0

        self.update_count(0)
        self.update_status("未开始")
        self.update_quality([]) # 重置质量评分
        self.update_feedback([]) # 清空反馈
        self.clear_visualization()
        self.reset_counter.emit() # 发出重置信号给 WorkoutPage

    @pyqtSlot(int)
    def update_count(self, count):
        """更新计数显示"""
        if self.is_counting_mode:
            self.value_label.setText(str(count))

    @pyqtSlot(float)
    def update_time(self, seconds=None):
        """更新计时显示"""
        if not self.is_counting_mode:
            if seconds is None:
                 # 如果没有提供秒数，则使用内部计时器计算
                 if self.start_time:
                     self.elapsed_time = time.time() - self.start_time
                 else:
                     self.elapsed_time = 0.0
                 seconds_to_display = self.elapsed_time
            else:
                 # 如果提供了秒数（例如，来自 CounterManager 的持续时间计时）
                 seconds_to_display = seconds
                 self.elapsed_time = seconds # 更新内部状态

            minutes = int(seconds_to_display // 60)
            secs = int(seconds_to_display % 60)
            hundredths = int((seconds_to_display * 100) % 100) # 显示到百分秒
            # self.value_label.setText(f"{minutes:02d}:{secs:02d}")
            self.value_label.setText(f"{minutes:01d}:{secs:02d}.{hundredths:02d}") # 更精确的显示

    @pyqtSlot(str)
    def update_status(self, status):
        """更新状态显示，并根据状态改变颜色和样式"""
        self.status_label.setText(status)
        # 基础样式
        base_style = f"""
            font-size: 15px; font-weight: 500;
            padding: 5px 12px;
            border-radius: 16px; /* 更圆的胶囊形状 */
            min-width: 70px;
            max-height: 28px; /* 控制高度 */
            qproperty-alignment: 'AlignCenter';
        """
        # 根据状态设置背景色和文字颜色
        if "进行中" in status or "向上" in status or "向下" in status:
             background_color = self.app_settings['accent_color'] + "E6" # 加透明度
             text_color = self.app_settings['theme_color']
             border = "none"
        elif "完成" in status or "标准" in status:
             background_color = self.app_settings['success_color'] + "E6"
             text_color = self.app_settings['success_color']
             border = "none"
        elif "错误" in status or "过快" in status or "过慢" in status:
             background_color = self.app_settings['danger_color'] + "E6"
             text_color = self.app_settings['danger_color']
             border = "none"
        elif "准备" in status:
             background_color = self.app_settings['warning_color'] + "E6"
             text_color = self.app_settings['warning_color']
             border = "none"
        else: # "未开始" 或其他默认状态
             background_color = "#F1F5F9"
             text_color = self.app_settings['secondary_text']
             border = f"1px solid {self.app_settings['border_color']}" # 默认状态加边框

        self.status_label.setStyleSheet(f"QLabel {{ {base_style} color: {text_color}; background-color: {background_color}; border: {border}; }}")

    @pyqtSlot(list)
    def update_quality(self, scores):
        """更新质量评分和描述 (使用最新分数)"""
   
        if not scores:
            latest_score_raw = 0 # Default to 0 if no scores
            description = "暂无评分"
            self.quality_progress_bar.setValue(0)
            self.quality_progress_bar.setFormat("N/A")
            chunk_color = "#D1D5DB"
        else:
            # --- 使用最后一个分数 --- 
            latest_score_raw = scores[-1] 
            if not np.isfinite(latest_score_raw):
                latest_score_raw = 0 # Use 0 if latest is invalid
            # ----------------------
            
            # 直接转换并限制最新分数
            latest_score = int(latest_score_raw)
            latest_score = max(0, min(100, latest_score)) # 强制限制在 0-100
            
            self.quality_progress_bar.setValue(latest_score)
            self.quality_progress_bar.setFormat(f"{latest_score}%")

            # 根据最新分数提供描述和颜色
            if latest_score >= 90:
                description = "质量优秀！"
                chunk_color = self.app_settings['success_color']
            elif latest_score >= 70:
                description = "质量良好"
                chunk_color = self.app_settings['success_color']
            elif latest_score >= 50:
                description = "质量一般，请注意动作规范"
                chunk_color = self.app_settings['warning_color']
            else:
                description = "质量较差，请调整姿势"
                chunk_color = self.app_settings['danger_color']

        # 更新进度条样式 (逻辑不变)
        self.quality_progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: none; border-radius: 6px; background-color: #EAF0F6;
                text-align: center; color: {self.app_settings['text_color']};
                font-size: 10px; font-weight: 500;
                height: 10px;
            }}
            QProgressBar::chunk {{
                border-radius: 6px;
                background-color: {chunk_color};
            }}
        """)
        # --- 强制更新 --- 
        self.quality_progress_bar.update()
        # ----------------
        self.quality_description_label.setText(description)

    @pyqtSlot(list)
    def update_feedback(self, feedback_list):
        """更新实时反馈标签 (Markdown 无序列表样式)"""
        if feedback_list:
            latest_feedback_items = feedback_list[-1]
            if isinstance(latest_feedback_items, list) and latest_feedback_items:
                # 如果是列表，格式化为无序列表
                formatted_items = [f"🔹 {str(item)}" for item in latest_feedback_items]
                display_text = "<br>".join(formatted_items) # 使用 HTML 换行
            elif isinstance(latest_feedback_items, str):
                # 如果是单个字符串
                display_text = f"🔹 {latest_feedback_items}"
            else:
                # 其他情况，尝试转换
                display_text = f"🔹 {str(latest_feedback_items)}"
            
            self.feedback_label.setText(display_text)
        else:
            self.feedback_label.setText("-") # 清空时显示占位符

    def update_visualization(self, data):
        """更新可视化图表 (PyQtGraph 版)"""
     
        # --- 决定要绘制哪个角度 --- 
        angle_to_plot = None
        if 'knee_angle' in data and data['knee_angle'] is not None: # 优先使用膝盖角度 (适用于深蹲等)
            angle_to_plot = data['knee_angle']
        elif 'elbow_angle' in data and data['elbow_angle'] is not None: # 备选肘部角度
            angle_to_plot = data['elbow_angle']
        elif 'torso_angle' in data and data['torso_angle'] is not None: # 备选躯干角度
             angle_to_plot = data['torso_angle']
        # --- 新增：检查髋部角度 --- 
        elif 'hip_angle' in data and data['hip_angle'] is not None: # 备选髋部角度 (桥式等)
             angle_to_plot = data['hip_angle']
        # -------------------------
        # ... 可以根据需要添加更多备选角度 ...
        # ---------------------------
        
        if angle_to_plot is not None:
            # 确保 self.angle_chart 是 PyQtGraph 控件
            if hasattr(self, 'angle_chart') and isinstance(self.angle_chart, pg.PlotWidget):
                 self.angle_chart.add_data_point(angle_to_plot)
            # else:
                 # print("警告: angle_chart 不是 PyQtGraph 控件或不可用")

    def clear_visualization(self):
        """清空图表数据 (PyQtGraph 版)"""
        if hasattr(self, 'angle_chart') and isinstance(self.angle_chart, pg.PlotWidget):
            self.angle_chart.clear_data()
            
    def set_exercise_name(self, exercise_name):
        """设置当前训练项目名称并更新UI"""
        self.exercise_name = exercise_name
        self.update_exercise_display()
        # 需要根据新的运动名称重新判断是否需要启动计时器
        self._set_mode(self.is_counting_mode) # 重新应用模式设置

    def update_exercise_display(self):
        """根据运动名称更新UI组件"""
        self.title_label.setText(self.exercise_name or "训练数据")
        # 根据运动类型判断是计数还是计时
        # 注意：这里假设了WorkoutPage在start_workout时会调用set_counting_mode
        # is_counting = self.exercise_name not in self.DURATION_TIMER_EXERCISES
        # self._set_mode(is_counting) # 应用模式

    def set_counting_mode(self, is_counting):
        """设置计数或计时模式"""
        self._set_mode(is_counting)
        # 根据模式更新显示
        if is_counting:
            self.update_count(0) # 重置计数显示
        else:
            self.update_time(0) # 重置计时显示

    def _set_mode(self, is_counting):
         """内部方法：设置模式并更新UI"""
         self.is_counting_mode = is_counting
         if is_counting:
             self.unit_label.setText("次")
             # 如果之前在计时，停止实时计时器
             # if self.realtime_timer.isActive():
             #     self.stop_realtime_timer()
             # 重置计时相关的变量
             self.elapsed_time = 0.0
             self.start_time = None
         else:
             self.unit_label.setText("时间")
             # 如果运动类型需要实时计时器，则启动
             # if self.exercise_name in self.REALTIME_TIMER_EXERCISES:
             #     self.start_realtime_timer()
             # else: # 对于持续时间计时，不需要启动实时计时器，由CounterManager提供时间
             #     if self.realtime_timer.isActive():
             #        self.stop_realtime_timer()
             #     self.elapsed_time = 0.0
             #     self.start_time = None # 由manager驱动时间
             # -----------------------------------------
             # 仅重置状态
             self.elapsed_time = 0.0
             self.start_time = None

         # 重置值和状态显示
         self.update_status("未开始")
         if is_counting:
             self.update_count(0)
         else:
             self.update_time(0.0)

    def set_display_thresholds(self, thresholds):
         """接收阈值并传递给图表 (PyQtGraph 版)"""
         self.display_thresholds = thresholds
         if hasattr(self, 'angle_chart') and isinstance(self.angle_chart, pg.PlotWidget):
             # print(f"ExerciseCounterWidget: 传递阈值 {thresholds} 给 PyQtGraph 图表")
             self.angle_chart.set_thresholds(thresholds)
         # else:
             # print("ExerciseCounterWidget: PyQtGraph 图表不可用，无法设置阈值")