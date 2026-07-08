#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
健身训练页面
"""

import os
import cv2
import numpy as np
import time
import threading
import logging
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFrame, QComboBox, QScrollArea,
                             QGridLayout, QSplitter, QStackedWidget, QMessageBox, QApplication, QSizePolicy, QLineEdit)
from PyQt6.QtGui import QPixmap, QIcon, QFont, QImage
from PyQt6.QtCore import Qt, QSize, pyqtSignal, pyqtSlot, QTimer, QPropertyAnimation, QEasingCurve, QPointF

from .pose_3d_viewer import Pose3DViewer
from ...utils.camera_manager import CameraManager
from ...utils.mmpose_detector import MMPoseDetector
from ...utils.model_manager import ModelManager
from ...utils.counters.counter_manager import CounterManager
from .exercise_counter_widget import ExerciseCounterWidget


class VideoWidget(QLabel):
    """视频显示控件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(320)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #f8f9fa; border-radius: 16px;")
        
        # 默认显示一个占位符
        self.set_image(None)
    
    def set_image(self, image):
        """设置图像
        
        Args:
            image: OpenCV格式的图像 (BGR)
        """
        if image is None:
            # 显示默认文本
            self.setText("摄像头未启动")
            self.setStyleSheet("background-color: #f8f9fa; border-radius: 16px; color: #6c757d; font-size: 16px;")
        else:
            # 转换为QImage
            if len(image.shape) == 3:
                height, width, channels = image.shape
                bytesPerLine = channels * width
                qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
            else:
                height, width = image.shape
                qImg = QImage(image.data, width, height, width, QImage.Format.Format_Grayscale8)
            
            # 按控件大小缩放图像
            pixmap = QPixmap.fromImage(qImg)
            scaled_pixmap = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
            self.setPixmap(scaled_pixmap)
            
            # 清除可能存在的文本和样式
            self.setText("")
            self.setStyleSheet("background-color: #f8f9fa; border-radius: 16px;")
    
    def display_frame(self, qimage):
        """显示QImage格式的帧
        
        Args:
            qimage: QImage对象
        """
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.setPixmap(scaled_pixmap)
        
        # 清除可能存在的文本和样式
        self.setText("")
        self.setStyleSheet("background-color: #f8f9fa; border-radius: 16px;")


class WorkoutCard(QFrame):
    """训练卡片"""

    def __init__(self, title, description, image_path=None, parent=None):
        super().__init__(parent)
        self.title = title
        self.description = description
        self.image_path = image_path
        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
            }
            
            QFrame:hover {
                background-color: #f8f9fa;
                border: 1px solid #cbd5e1;
            }
        """)
        self.setMinimumHeight(180)

        # 主布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # 标题
        title_label = QLabel(self.title)
        title_label.setStyleSheet("""
            QLabel {
                color: #212529;
                font-size: 18px;
                font-weight: bold;
                background-color: transparent;
                border: none;
            }
        """)
        layout.addWidget(title_label)

        # 描述
        desc_label = QLabel(self.description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("""
            QLabel {
                color: #6c757d;
                font-size: 13px;
                margin-top: 4px;
                background-color: transparent;
                border: none;
            }
        """)
        layout.addWidget(desc_label)

        # 开始按钮
        start_btn = QPushButton("开始训练")
        start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4361ee;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: 500;
                margin-top: 12px;
            }
            
            QPushButton:hover {
                background-color: #3a56d4;
            }
            
            QPushButton:pressed {
                background-color: #2e4bbd;
            }
        """)
        layout.addWidget(start_btn, alignment=Qt.AlignmentFlag.AlignLeft)


class WorkoutPage(QWidget):
    """健身训练页面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 摄像头和姿态检测相关
        self.camera_manager = CameraManager(self)  # 初始化摄像头管理器
        
        # 获取全局模型实例
        self.model_manager = ModelManager()
        self.mmpose_detector = self.model_manager.get_detector()
        self.mmpose_detector.is_webcam = True  # 设置为实时摄像头模式
        
        # 其他设置
        self.is_camera_running = False
        self.is_model_loading = False
        self.model_loaded = self.model_manager.is_loaded()
        
        # 获取应用程序设置
        self.app_settings = self.get_app_settings()
        
        # 创建计数器管理器
        self.counter_manager = CounterManager()
        
        # 初始化UI
        self.init_ui()
        
        # 页面加载完成后，开始异步加载相机列表
        QTimer.singleShot(500, self.camera_manager.scan_cameras_async)
        
        # 保存上一帧有效的姿态数据
        self.last_valid_keypoints = None
        
        # 连接信号
        self.connect_signals()

    def connect_signals(self):
        """连接信号到槽"""
        # 连接模型管理器的信号
        self.model_manager.model_loaded.connect(self.on_model_loaded)
        
        # 连接摄像头信号
        self.camera_manager.camera_list_ready.connect(self.update_camera_list)
        self.camera_manager.frame_ready.connect(self.on_frame_ready)
        self.camera_manager.camera_error.connect(self.on_camera_error)
        
        # 连接姿态检测信号
        self.mmpose_detector.frame_processed.connect(self.on_frame_processed)
        self.mmpose_detector.pose_detected.connect(self.on_pose_detected)
        self.mmpose_detector.detection_error.connect(self.on_detection_error)
        
        # 连接计数器管理器信号
        self.counter_manager.count_updated.connect(self.on_count_updated)
        self.counter_manager.time_updated.connect(self.on_time_updated)
        self.counter_manager.state_changed.connect(self.on_state_changed)
        self.counter_manager.visualization_data.connect(self.on_visualization_data)
        self.counter_manager.quality_updated.connect(self.on_quality_updated)
        self.counter_manager.feedback_updated.connect(self.on_feedback_updated)
        
        # 连接新的信号
        self.camera_combo.currentIndexChanged.connect(self.on_camera_selection_changed)
    
    @pyqtSlot(bool)
    def on_model_loaded(self, success):
        """模型加载完成的回调"""
        self.model_loaded = success
        if success:
            print("模型已加载")
        else:
            print("模型加载失败")

    def init_pose_detector(self):
        """初始化姿态检测器配置"""
        try:
            # 直接使用全局模型实例
            if not self.model_loaded:
                print("等待模型加载...")
                self.model_manager.load_model()
            else:
                print("模型已加载")
                
            # 设置性能参数
            self.mmpose_detector.set_performance_params(
                detection_interval=2,  # 每2帧检测一次
                input_size=(640, 480),
                use_threading=True
            )
            
            # 设置可视化参数
            self.mmpose_detector.set_visual_params(
                draw_bbox=True,
                draw_fps=True
            )
            
            return True
            
        except Exception as e:
            print(f"初始化出错: {str(e)}")
            return False

    def get_app_settings(self):
        """获取应用程序设置"""
        # 这里可以从全局设置或配置文件中获取配置
        # 简单起见，先返回一个默认配置
        return {
            'camera_id': 0,
            'resolution': (640, 480),
            'fps': 30,
            'theme_color': '#4361EE',      # 主题蓝
            'accent_color': '#0ea5e9',     # 强调蓝
            'success_color': '#10B981',    # 成功绿
            'warning_color': '#F59E0B',    # 警告黄
            'danger_color': '#EF4444',     # 危险红
            'background_color': '#F8FAFC', # 更柔和的背景色
            'card_background': '#FFFFFF',  # 卡片背景白色
            'text_color': '#1E293B',      # 深灰色文本
            'secondary_text': '#64748B', # 灰色次要文本
            'border_color': '#E2E8F0'       # 边框颜色
        }

    def init_ui(self):
        """初始化UI (布局优化版)"""
        # --- 按钮样式定义 ---
        self.button_style_primary = f"""
            QPushButton {{
                background-color: {self.app_settings['theme_color']};
                color: white; border: none; border-radius: 8px;
                padding: 8px 14px; font-size: 13px; font-weight: 500;
            }}
            QPushButton:hover {{ background-color: #3A56D4; }}
            QPushButton:pressed {{ background-color: #2E4BBD; }}
        """
        self.button_style_secondary = f"""
             QPushButton {{
                background-color: #F1F5F9; color: {self.app_settings['secondary_text']};
                border: 1px solid {self.app_settings['border_color']};
                border-radius: 8px; padding: 8px 14px; font-size: 13px; font-weight: 500;
            }}
            QPushButton:hover {{ background-color: #E2E8F0; border-color: #CBD5E1; }}
            QPushButton:pressed {{ background-color: #CBD5E1; }}
            QPushButton:checked {{
                background-color: {self.app_settings['accent_color']}; color: white; border: none;
            }}
        """
        self.button_style_warning = f"""
             QPushButton {{
                background-color: #F1F5F9; color: {self.app_settings['secondary_text']};
                border: 1px solid {self.app_settings['border_color']};
                border-radius: 8px; padding: 8px 14px; font-size: 13px; font-weight: 500;
            }}
            QPushButton:hover {{ background-color: #E2E8F0; border-color: #CBD5E1; }}
            QPushButton:checked {{ background-color: {self.app_settings['warning_color']}; color: white; border: none; }}
            QPushButton:pressed {{ background-color: #D97706; }}
        """
        self.stop_button_style = f"""
            QPushButton {{
                background-color: {self.app_settings['danger_color']};
                color: white; border: none; border-radius: 8px;
                padding: 8px 14px; font-size: 13px; font-weight: 500;
            }}
            QPushButton:hover {{ background-color: #DC2626; }}
            QPushButton:pressed {{ background-color: #B91C1C; }}
        """

        # --- 页面主体布局 ---
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setStyleSheet(f"background-color: {self.app_settings['background_color']};")

        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack)

        # --- 健身列表页面 --- 
        workout_list_page = QWidget()
        self.stack.addWidget(workout_list_page)
        workout_list_layout = QVBoxLayout(workout_list_page)
        workout_list_layout.setContentsMargins(25, 25, 25, 25) # 适度调整边距
        workout_list_layout.setSpacing(20) # 适度调整间距

        title_label = QLabel("选择训练项目")
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {self.app_settings['text_color']};
                font-size: 24px; /* 调整标题字号 */
                font-weight: 600; /* 加粗 */
                border: none; padding-bottom: 5px; /* 增加一点下边距 */
            }}
        """)
        workout_list_layout.addWidget(title_label)

        # 过滤器区域保持不变
        filter_frame = QFrame()
        filter_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {self.app_settings['card_background']};
                border-radius: 12px;
                border: 1px solid {self.app_settings['border_color']};
            }}
            QLabel {{
                color: {self.app_settings['secondary_text']};
                font-size: 14px;
                border: none; background: none;
            }}
        """)
        filter_layout = QHBoxLayout(filter_frame)
        filter_layout.setContentsMargins(16, 16, 16, 16) # 调整内边距
        filter_layout.setSpacing(12)

        difficulty_label = QLabel("难度:")
        difficulty_label.setStyleSheet("font-weight: 500; color: #475569; border: none; background: none;")
        filter_layout.addWidget(difficulty_label)
        difficulty_combo = QComboBox()
        difficulty_combo.addItems(["所有", "初级", "中级", "高级"])
        difficulty_combo.setStyleSheet(f"""
            QComboBox {{
                border: 1px solid {self.app_settings['border_color']};
                border-radius: 8px;
                padding: 8px 10px; /* 调整 padding */
                min-width: 120px; /* 调整宽度 */
                background-color: {self.app_settings['card_background']};
                color: {self.app_settings['text_color']};
                font-size: 13px;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding; subcontrol-position: top right; width: 25px;
                border-left-width: 0px; border-top-right-radius: 8px; border-bottom-right-radius: 8px;
            }}
            QComboBox QAbstractItemView {{
                border: 1px solid #edf0f5; border-radius: 8px; padding: 5px;
                selection-background-color: #EBF5FF; selection-color: {self.app_settings['theme_color']};
                background-color: {self.app_settings['card_background']}; color: #18191c;
            }}
        """)
        filter_layout.addWidget(difficulty_combo)
        filter_layout.addSpacing(18)

        body_part_label = QLabel("部位:")
        body_part_label.setStyleSheet("font-weight: 500; color: #475569; border: none; background: none;")
        filter_layout.addWidget(body_part_label)
        body_part_combo = QComboBox()
        body_part_combo.addItems(["所有", "核心", "上肢", "下肢", "全身"])
        body_part_combo.setStyleSheet(difficulty_combo.styleSheet())
        filter_layout.addWidget(body_part_combo)
        filter_layout.addStretch()
        workout_list_layout.addWidget(filter_frame)

        # 滚动区域和卡片保持不变
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{ background-color: transparent; border: none; }}
            QScrollBar:vertical {{ border: none; background: #E5E7EB; width: 7px; border-radius: 3px; margin: 0px; }}
            QScrollBar::handle:vertical {{ background: #9CA3AF; min-height: 25px; border-radius: 3px; }}
            QScrollBar::handle:vertical:hover {{ background: #6B7280; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
        """)
        scroll_content_list = QWidget()
        scroll_content_list.setStyleSheet("background-color: transparent;")
        scroll_layout_list = QVBoxLayout(scroll_content_list)
        scroll_layout_list.setContentsMargins(0, 0, 0, 0)
        scroll_layout_list.setSpacing(18) # 卡片垂直间距
        grid = QGridLayout()
        grid.setSpacing(18) # 卡片水平和垂直间距
        workouts = [
             {"title": "深蹲", "description": "锻炼大腿和臀部肌肉的基本动作", "level": "初级"},
             {"title": "俯卧撑", "description": "锻炼胸部、肩部和手臂肌肉的复合动作", "level": "初级"},
             {"title": "平板支撑", "description": "增强核心稳定性的静态训练", "level": "初级"},
             {"title": "高抬腿", "description": "锻炼大腿前侧、小腿和髋屈肌的复合动作", "level": "初级"},
             {"title": "引体向上", "description": "锻炼背部和手臂肌肉的上拉动作", "level": "中级"},
             {"title": "仰卧起坐", "description": "锻炼腹部肌肉的卷腹动作", "level": "中级"},
             {"title": "桥", "description": "锻炼臀部和核心肌肉的桥式动作", "level": "中级"},
             {"title": "臂屈伸", "description": "锻炼肱三头肌和前臂肌肉的屈伸动作", "level": "中级"},
             {"title": "登山跑", "description": "锻炼下腹、腹斜肌和髋屈肌的复合动作", "level": "中级"},
        ]
        for i, workout in enumerate(workouts):
            card = WorkoutCard(
                workout["title"],
                f"{workout['description']} | 难度: {workout['level']}"
            )
            row, col = divmod(i, 3) # 每行3个卡片
            grid.addWidget(card, row, col)
            card.findChild(QPushButton).clicked.connect(
                lambda checked=False, title=workout["title"]: self.start_workout(title))
        scroll_layout_list.addLayout(grid)
        scroll_layout_list.addStretch()
        scroll_area.setWidget(scroll_content_list)
        workout_list_layout.addWidget(scroll_area)


        # --- 创建训练视图页面 --- (布局优化)
        self.workout_view_page = QWidget()
        self.stack.addWidget(self.workout_view_page)
        workout_view_layout = QVBoxLayout(self.workout_view_page)
        workout_view_layout.setContentsMargins(15, 15, 15, 15) # 减小页面整体边距
        workout_view_layout.setSpacing(10) # 减小顶部栏和主内容间距

        # --- 顶部工具栏 --- (样式微调)
        top_bar = QFrame()
        top_bar.setStyleSheet(f"""
            QFrame {{
                background-color: {self.app_settings['card_background']};
                border-radius: 10px; /* 微调圆角 */
                border: 1px solid {self.app_settings['border_color']};
                padding: 8px 12px; /* 微调内边距 */
            }}
        """)
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(0,0,0,0) # 移除布局边距，由 QFrame padding 控制
        top_bar_layout.setSpacing(10)

        # 返回按钮
        back_button = QPushButton()
        back_button.setCursor(Qt.CursorShape.PointingHandCursor)
        back_button.setFixedSize(90, 32) # 微调尺寸
        button_layout = QHBoxLayout(back_button)
        button_layout.setContentsMargins(6, 0, 10, 0) # 微调内部边距
        button_layout.setSpacing(3)
        icon_label = QLabel()
        icon_label.setFixedSize(18, 18) # 微调图标尺寸
        icon_svg = f'''<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M15 19L8 12L15 5" stroke="{self.app_settings['theme_color']}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>'''
        icon_label.setStyleSheet(f"background: none; border: none;")
        icon_label.setPixmap(QPixmap.fromImage(QImage.fromData(icon_svg.encode(), 'SVG')))
        text_label = QLabel("返回")
        text_label.setStyleSheet(f"""
            color: {self.app_settings['theme_color']};
            font-size: 13px; font-weight: 500;
            background: none; border: none;
            border: none;
        """)
        button_layout.addWidget(icon_label)
        button_layout.addWidget(text_label)
        button_layout.addStretch()
        back_button.setStyleSheet(f"""
            QPushButton {{
                background-color: #F8FAFC; /* 更浅的背景 */
                border: 1px solid #E2E8F0; /* 匹配边框 */
                color: {self.app_settings['theme_color']};
                border-radius: 6px; padding: 0px;
            }}
            QPushButton:hover {{ background-color: #F1F5F9; border-color: #E2E8F0; }}
            QPushButton:pressed {{ background-color: #E2E8F0; }}
        """)
        back_button.clicked.connect(self.back_to_list)
        top_bar_layout.addWidget(back_button)

        # 当前训练标题
        self.workout_title_label = QLabel("当前训练")
        self.workout_title_label.setStyleSheet(f"""
            QLabel {{
                color: {self.app_settings['text_color']};
                font-size: 18px; /* 微调字号 */
                font-weight: 600;
                border: none; background: none; padding-left: 10px;
            }}
        """)
        top_bar_layout.addWidget(self.workout_title_label, alignment=Qt.AlignmentFlag.AlignCenter)
        top_bar_layout.addStretch(1) # 添加伸展因子确保标题居中

        workout_view_layout.addWidget(top_bar)

        # --- 主内容区域 (使用 QSplitter 替代 QHBoxLayout) ---
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(10) # 设置分隔条宽度
        # 修改分隔条样式，使其背景透明
        main_splitter.setStyleSheet(""" 
            QSplitter::handle:horizontal {
                background-color: transparent; /* 背景透明 */
                border: none; /* 移除边框 */
                width: 8px; /* 确保宽度 */
                margin: 0px 2px; /* 调整左右边距，增加点击区域 */
            }
            QSplitter::handle:horizontal:hover {
                /* 悬停时显示半透明浅灰色 */
                background-color: rgba(203, 213, 225, 0.5); /* #CBD5E1 with 50% alpha */
                border-radius: 3px;
            }
            QSplitter::handle:horizontal:pressed {
                background-color: rgba(160, 174, 192, 0.6); /* 稍深一点的按下效果 */
                border-radius: 3px;
            }
        """)

        # --- 左列: 训练数据控件 ---
        current_exercise_name = self.workout_title_label.text() if hasattr(self, 'workout_title_label') else ""
        self.counter_widget = ExerciseCounterWidget(exercise_name=current_exercise_name)
        self.counter_widget.reset_counter.connect(self.reset_counter)
        self.counter_widget.setMinimumWidth(300) # 保留最小宽度，防止过小
        self.counter_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        # 添加到 QSplitter
        main_splitter.addWidget(self.counter_widget)

        # --- 右列: 视频 和 3D模型 (垂直布局) ---
        right_column_widget = QWidget()
        right_column_widget.setStyleSheet("background: none; border: none;")
        right_column_layout = QVBoxLayout(right_column_widget)
        right_column_layout.setContentsMargins(0, 0, 0, 0) # 移除边距
        right_column_layout.setSpacing(10) # 保持视频和3D视图间距
        
        # -- 右列上方: 视频区域 --
        video_container = QFrame()
        video_container.setStyleSheet(f"""
            QFrame {{
                background-color: {self.app_settings['card_background']};
                border-radius: 12px; /* 统一圆角 */
                border: 1px solid {self.app_settings['border_color']};
                padding: 10px; /* 减小内边距 */
            }}
        """)
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0) # 内部边距由 QFrame padding 控制
        video_layout.setSpacing(8) # 减小标题、视频、控制栏间距

        video_title = QLabel("摄像头视图")
        video_title.setStyleSheet(f"""
            QLabel {{
                color: {self.app_settings['text_color']};
                font-size: 17px; /* 调整字号 */
                font-weight: 600;
                border: none; background: none;
                padding-bottom: 4px; /* 微调标题下方间距 */
            }}
        """)
        video_layout.addWidget(video_title)

        self.video_widget = VideoWidget() # VideoWidget 自带样式
        self.video_widget.setMinimumHeight(280) # 调整最小高度
        video_layout.addWidget(self.video_widget, 1) # 允许伸展

        video_controls = QHBoxLayout()
        video_controls.setContentsMargins(0, 5, 0, 0) # 微调上方边距
        video_controls.setSpacing(8) # 减小按钮间距

        self.camera_combo = QComboBox()
        # 使用之前定义的更紧凑的 QComboBox 样式
        self.camera_combo.setStyleSheet(difficulty_combo.styleSheet())
        self.camera_combo.setMinimumWidth(120) # 确保最小宽度
        self.camera_combo.addItem("检测中...", -1)
        
        # 添加RTSP URL输入框
        self.rtsp_url_input = QLineEdit()
        self.rtsp_url_input.setPlaceholderText("RTSP URL")
        self.rtsp_url_input.hide() # 默认隐藏
        # 借用 camera_combo 的样式，并替换为 QLineEdit
        rtsp_style = difficulty_combo.styleSheet().replace("QComboBox", "QLineEdit")\
                                               .replace("drop-down", "padding") # 简单替换避免错误
        self.rtsp_url_input.setStyleSheet(rtsp_style)
        self.rtsp_url_input.setMinimumWidth(150) # 给URL更多空间

        video_controls.addWidget(self.camera_combo)
        video_controls.addWidget(self.rtsp_url_input) # 添加到布局

        self.camera_btn = QPushButton("启动")
        self.camera_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.camera_btn.setStyleSheet(self.button_style_primary)
        self.camera_btn.clicked.connect(self.toggle_camera)
        video_controls.addWidget(self.camera_btn)

        self.show_skeleton_btn = QPushButton("骨架")
        self.show_skeleton_btn.setCheckable(True)
        self.show_skeleton_btn.setChecked(True)
        self.show_skeleton_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.show_skeleton_btn.setStyleSheet(self.button_style_secondary)
        self.show_skeleton_btn.clicked.connect(lambda checked: self.toggle_skeleton(checked))
        video_controls.addWidget(self.show_skeleton_btn)

        video_controls.addStretch()
        video_layout.addLayout(video_controls)
        right_column_layout.addWidget(video_container, 1) # 视频容器在垂直方向伸展

        # -- 右列下方: 3D 模型区域 --
        model_container = QFrame()
        model_container.setStyleSheet(video_container.styleSheet()) # 应用相同容器样式
        model_layout = QVBoxLayout(model_container)
        model_layout.setContentsMargins(0, 0, 0, 0) # 内部边距由 QFrame padding 控制
        model_layout.setSpacing(8) # 减小标题、视图、控制栏间距

        model_title = QLabel("3D姿态预览")
        model_title.setStyleSheet(video_title.styleSheet()) # 应用相同标题样式
        model_layout.addWidget(model_title)

        self.pose_3d_viewer = Pose3DViewer()
        self.pose_3d_viewer.setMinimumHeight(240) # 调整最小高度
        model_layout.addWidget(self.pose_3d_viewer, 1) # 允许伸展

        model_controls = QHBoxLayout()
        model_controls.setContentsMargins(0, 5, 0, 0) # 微调上方边距
        model_controls.setSpacing(8) # 减小按钮间距

        self.auto_rotate_btn = QPushButton("旋转")
        self.auto_rotate_btn.setCheckable(True)
        self.auto_rotate_btn.setChecked(True)
        self.auto_rotate_btn.setStyleSheet(self.button_style_warning) # 使用警告色突出
        self.auto_rotate_btn.clicked.connect(lambda checked: self.pose_3d_viewer.toggle_auto_rotate(checked))
        model_controls.addWidget(self.auto_rotate_btn)

        reset_view_btn = QPushButton("重置视角") # 更明确的文本
        reset_view_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reset_view_btn.setStyleSheet(self.button_style_secondary)
        reset_view_btn.clicked.connect(self.pose_3d_viewer.reset_view)
        model_controls.addWidget(reset_view_btn)

        model_controls.addStretch()
        model_layout.addLayout(model_controls)
        right_column_layout.addWidget(model_container, 1) # 3D容器在垂直方向伸展

        # 添加到 QSplitter
        main_splitter.addWidget(right_column_widget) 

        # 设置初始尺寸比例 (例如，左侧大约占 1/3)
        # 获取初始窗口宽度估算，或者直接设置固定比例
        # screen_width = QApplication.primaryScreen().availableGeometry().width()
        # initial_left = int(screen_width * 0.3)
        # initial_right = int(screen_width * 0.7)
        # 简单设置一个固定比例
        main_splitter.setSizes([350, 650]) # 尝试初始比例

        # 将 QSplitter 添加到主垂直布局，并允许伸展
        workout_view_layout.addWidget(main_splitter, 1) 

        # 初始化姿态检测模型
        self.init_pose_detector()

        # 默认显示健身列表页面
        self.stack.setCurrentIndex(0)

    def cleanup(self):
        """清理页面资源，停止后台线程 (增强等待)"""
        print("WorkoutPage: 开始清理...")
        
        self.is_camera_running = False 
        
        cam_thread = None
        pose_thread = None
        
        # 停止并尝试获取线程对象
        if hasattr(self, 'camera_manager') and self.camera_manager.is_running:
            print("WorkoutPage: 正在停止 CameraManager...")
            try:
                cam_thread = self.camera_manager.thread # 在 stop 前获取
                self.camera_manager.stop() # stop 内部有 join(timeout=1.0)
                print("WorkoutPage: CameraManager stop() called")
            except Exception as e:
                print(f"WorkoutPage 停止 CameraManager 时出错: {e}")
        
        if hasattr(self, 'mmpose_detector') and self.mmpose_detector.running:
            print("WorkoutPage: 正在停止 MMPoseDetector...")
            try:
                pose_thread = self.mmpose_detector.detection_thread # 在 stop 前获取
                self.mmpose_detector.stop() # stop 内部有 join(timeout=1.0)
                print("WorkoutPage: MMPoseDetector stop() called")
            except Exception as e:
                print(f"WorkoutPage 停止 MMPoseDetector 时出错: {e}")

        # --- 显式等待线程结束 --- 
        wait_timeout_ms = 500 # 等待 0.5 秒
        print(f"WorkoutPage: 等待后台线程结束 (最多 {wait_timeout_ms}ms)...")
        
        start_wait = time.time()
        
        if cam_thread and cam_thread.is_alive():
            print("WorkoutPage: 等待 Camera 线程...")
            # PyQt/Python threading 的 join 可能在主事件循环退出时行为不确定
            # 尝试使用 QThread 的 wait (如果 manager 的 thread 是 QThread)
            # 或者使用循环 + sleep 简单等待
            while cam_thread.is_alive() and (time.time() - start_wait) * 1000 < wait_timeout_ms:
                 time.sleep(0.05) # 短暂休眠
            if cam_thread.is_alive():
                 print("WorkoutPage: [警告] Camera 线程等待超时!")
            else:
                 print("WorkoutPage: Camera 线程已结束")
                 
        if pose_thread and pose_thread.is_alive():
            print("WorkoutPage: 等待 Pose 线程...")
            while pose_thread.is_alive() and (time.time() - start_wait) * 1000 < wait_timeout_ms:
                 time.sleep(0.05)
            if pose_thread.is_alive():
                 print("WorkoutPage: [警告] Pose 线程等待超时!")
            else:
                 print("WorkoutPage: Pose 线程已结束")
        # --------------------------
                
        print("WorkoutPage: 清理完成")

    def start_workout(self, workout_title):
        """开始健身训练，并设置计数器和阈值线"""
        # 清除上一帧的视图
        if hasattr(self, 'video_widget'):
            self.video_widget.set_image(None) # 清除摄像头视图
        if hasattr(self, 'pose_3d_viewer'): 
            # 发送空关键点以显示默认T姿态
            self.pose_3d_viewer.update_keypoints(np.zeros((17, 3), dtype=np.float32))
            self.pose_3d_viewer.reset_view() # 同时重置视角
        
        self.workout_title_label.setText(workout_title)
        self.stack.setCurrentIndex(1)
        print(f"已选择: {workout_title} - 点击'开始摄像头'开始训练")
        
        # 设置当前训练的计数器
        counter_type = self.counter_manager.get_counter_type(workout_title)
        thresholds = None # 初始化阈值
        if counter_type is not None:
            # 激活对应的计数器
            self.counter_manager.set_active_counter(workout_title)
            
            # 获取活动计数器的阈值
            thresholds = self.counter_manager.get_active_counter_thresholds()
            print(f"获取到 {workout_title} 的阈值: {thresholds}") # 添加日志
            
            # 更新 counter_widget 的运动名称
            self.counter_widget.set_exercise_name(workout_title)
            
            # 设置计数/计时模式
            self.counter_widget.set_counting_mode(counter_type == "count")
            
            # 传递阈值给 counter_widget
            if hasattr(self.counter_widget, 'set_display_thresholds'):
                self.counter_widget.set_display_thresholds(thresholds)
            else:
                print("警告: counter_widget 没有 set_display_thresholds 方法")

            # 显示计数/计时控件
            self.counter_widget.setVisible(True)
        else:
            # 没有找到对应的计数器，隐藏控件
            self.counter_widget.setVisible(False)
            # 如果没有计数器，也确保清空旧阈值
            if hasattr(self.counter_widget, 'set_display_thresholds'):
                 self.counter_widget.set_display_thresholds(None)

    def back_to_list(self):
        """返回健身列表"""
        # 如果摄像头正在运行，停止它
        if self.is_camera_running:
            self.toggle_camera()
            
        # 重置计数器
        self.reset_counter()
        
        # 隐藏计数/计时控件
        self.counter_widget.setVisible(False)

        self.stack.setCurrentIndex(0)

    def reset_counter(self):
        """重置计数器 (现在由 ExerciseCounterWidget 的信号触发)"""
        print("WorkoutPage.reset_counter called")
        self.counter_manager.reset_active_counter()
        # 清空图表也应该由 ExerciseCounterWidget 内部处理，或者通过信号传递
        # 如果 ExerciseCounterWidget 内部不能完全清除，可以在这里调用
        if hasattr(self, 'counter_widget') and hasattr(self.counter_widget, 'clear_visualization'):
             self.counter_widget.clear_visualization()
    
    @pyqtSlot(int)
    def on_count_updated(self, count):
        """处理计数更新信号"""
        if hasattr(self, 'counter_widget'):
            self.counter_widget.update_count(count)
    
    @pyqtSlot(float)
    def on_time_updated(self, seconds):
        """处理计时更新信号"""
        if hasattr(self, 'counter_widget'):
            self.counter_widget.update_time(seconds)
    
    @pyqtSlot(str)
    def on_state_changed(self, state):
        """处理状态改变信号"""
        if hasattr(self, 'counter_widget'):
            self.counter_widget.update_status(state)
    
    @pyqtSlot(dict)
    def on_visualization_data(self, data):
        """处理可视化数据信号"""
        # -------------------
        if hasattr(self, 'counter_widget'):
            self.counter_widget.update_visualization(data)
            
    @pyqtSlot(list)
    def on_quality_updated(self, scores):
        """处理质量评分更新信号"""
        if hasattr(self, 'counter_widget'):
            self.counter_widget.update_quality(scores)
            
    @pyqtSlot(list)
    def on_feedback_updated(self, feedback_list):
        """处理反馈信息更新信号"""
        if hasattr(self, 'counter_widget'):
            self.counter_widget.update_feedback(feedback_list)

    def toggle_camera(self):
        """切换摄像头状态（启动/停止），并更新按钮样式"""
        if not self.is_camera_running:
            # 启动摄像头
            if not self.camera_manager.is_running:
                # 获取摄像头ID或URL
                selected_data = self.camera_combo.currentData()
                camera_id_or_url = None
                is_rtsp_selected = False

                if selected_data == "rtsp":
                    rtsp_url = self.rtsp_url_input.text().strip()
                    if not rtsp_url:
                        self.on_camera_error("请输入有效的 RTSP URL")
                        return
                    if not rtsp_url.lower().startswith("rtsp://"):
                        self.on_camera_error("RTSP URL 必须以 rtsp:// 开头")
                        return
                    camera_id_or_url = rtsp_url
                    is_rtsp_selected = True
                elif isinstance(selected_data, int) and selected_data != -1:
                    camera_id_or_url = selected_data
                else:
                    self.on_camera_error("请先选择一个有效的摄像头或 RTSP 视频流")
                    self.camera_manager.scan_cameras_async() # 尝试刷新列表
                    return

                try:
                    self.camera_manager.set_camera_id(camera_id_or_url)
                    
                    # 注意：WorkoutPage 没有独立的宽度/高度/FPS设置控件
                    # 这些参数可能来自全局设置或在此页面不可配置
                    # 因此，我们只在非RTSP时，依赖CameraManager内部的默认值或已设值
                    if is_rtsp_selected:
                        logging.info("RTSP流选定，将忽略本地分辨率和帧率设置。")

                    if self.camera_manager.start():
                        self.is_camera_running = True
                        self.camera_btn.setText("停止") # 更新文本
                        self.camera_btn.setStyleSheet(self.stop_button_style) # 应用停止样式
                        logging.info(f"已启动摄像头 ID: {selected_data}")
                        
                        # 确保检测器已加载且线程已启动
                        if self.mmpose_detector and not self.mmpose_detector.running:
                            self.mmpose_detector.start()
                            logging.info("姿态检测线程已启动")
                        
                except Exception as e:
                    error_msg = f"启动摄像头失败: {e}"
                    logging.error(error_msg)
                    self.on_camera_error(error_msg)
                    self.is_camera_running = False  # 确保状态正确
                    # 恢复按钮状态
                    self.camera_btn.setText("启动") # 恢复文本
                    self.camera_btn.setStyleSheet(self.button_style_primary) # 恢复样式
            else:
                logging.warning("摄像头已在运行中，但 UI 状态未同步，正在停止...")
                self.stop_camera_resources() # 尝试停止
                # 更新按钮状态
                self.camera_btn.setText("启动")
                self.camera_btn.setStyleSheet(self.button_style_primary)
                self.is_camera_running = False
        else:
            # 停止摄像头
            # 检查 CameraManager 自身是否认为摄像头在运行
            if self.camera_manager.is_running:
                try:
                    # 停止检测线程
                    if self.mmpose_detector and self.mmpose_detector.running:
                        self.mmpose_detector.stop()
                        logging.info("姿态检测线程已停止")
                        
                    self.camera_manager.stop()
                    self.is_camera_running = False
                    self.camera_btn.setText("启动") # 更新文本
                    self.camera_btn.setStyleSheet(self.button_style_primary) # 应用启动样式
                    logging.info("摄像头已停止")
                    
                    # 清空视频显示
                    self.video_widget.set_image(None)
                    # self.pose_3d_viewer.clear_pose() # Pose3DViewer 没有 clear_pose
                    # 传递空关键点来清除3D视图
                    self.pose_3d_viewer.update_keypoints(np.zeros((17, 3), dtype=np.float32))
                except Exception as e:
                    error_msg = f"停止摄像头失败: {e}"
                    logging.error(error_msg)
                    self.on_camera_error(error_msg)
                    # 尝试恢复按钮状态
                    self.camera_btn.setText("停止") # 保持停止文本
                    self.camera_btn.setStyleSheet(self.stop_button_style) # 保持停止样式
            else:
                logging.warning("摄像头已停止")
                # 确保状态一致
                self.is_camera_running = False
                self.camera_btn.setText("启动") # 恢复文本
                self.camera_btn.setStyleSheet(self.button_style_primary) # 恢复样式

    @pyqtSlot(np.ndarray)
    def on_frame_ready(self, frame):
        """相机帧准备好时"""
        # 交给MMPose检测器处理
        self.mmpose_detector.process_frame(frame)

    @pyqtSlot(str)
    def on_camera_error(self, error_msg):
        """相机错误"""
        QMessageBox.critical(self, "相机错误", error_msg)
        if self.is_camera_running:
            self.toggle_camera()

    @pyqtSlot(str)
    def on_detection_error(self, error_msg):
        """检测错误"""
        QMessageBox.critical(self, "检测错误", error_msg)

    @pyqtSlot(QImage)
    def on_frame_processed(self, qimage):
        """接收处理后的帧"""
        try:
            # 使用VideoWidget的display_frame方法显示图像
            if self.video_widget is not None:
                self.video_widget.display_frame(qimage)
        except Exception as e:
            print(f"处理帧时出错: {str(e)}")

    @pyqtSlot(np.ndarray)
    def on_pose_detected(self, keypoints):
        """接收检测到的姿态关键点"""
        try:
            # 检查关键点是否有效
            is_valid = np.any(keypoints) and not np.all(keypoints == 0)
            
            if is_valid:
                # 如果有效，更新上一次有效的关键点
                self.last_valid_keypoints = keypoints.copy()
                # 更新3D模型
                self.pose_3d_viewer.update_keypoints(keypoints)
                
                # 将关键点传递给计数器管理器进行处理
                self.counter_manager.process_keypoints(keypoints)
            elif self.last_valid_keypoints is not None:
                # 如果无效但有之前的有效关键点，使用之前的关键点
                
                # 也可以传递上一帧有效的关键点给计数器管理器
                self.counter_manager.process_keypoints(self.last_valid_keypoints)
            # else:
            #     # 如果无效且没有之前的有效关键点
            #     print("未检测到完整姿态")
        except Exception as e:
            print(f"处理姿态关键点时出错: {str(e)}")

    def toggle_skeleton(self, show):
        """切换骨架显示

        Args:
            show: 是否显示骨架
        """
        if self.mmpose_detector is not None:
            # Read the current bbox state to avoid overriding it
            current_draw_bbox = self.mmpose_detector.draw_bbox
            # Toggle skeleton and keypoints together, keep bbox state
            self.mmpose_detector.set_visual_params(
                draw_skeleton=show, 
                draw_keypoints=show, 
                draw_bbox=current_draw_bbox
            )

    def update_camera_list(self, available_cameras):
        """更新摄像头列表"""
        if hasattr(self, 'camera_combo'):
            current_selection_data = self.camera_combo.currentData()
            self.camera_combo.clear()

            # 添加 RTSP 选项
            self.camera_combo.addItem("RTSP 视频流", "rtsp")

            if available_cameras:
                for camera_id in available_cameras:
                    self.camera_combo.addItem(f"摄像头 {camera_id}", camera_id)
                # 更新状态信息
                print(f"找到 {len(available_cameras)} 个可能的摄像头索引，使用时将验证可用性")
                # 默认选择第一个摄像头
                self.camera_combo.setCurrentIndex(0)
                # 启用启动按钮
                if hasattr(self, 'camera_btn'):
                    self.camera_btn.setEnabled(True)
                    self.camera_btn.setStyleSheet(self.button_style_primary) # 确保是启动样式

                # 手动触发一次选择检查
                self.on_camera_selection_changed(self.camera_combo.currentIndex())
            else:
                self.camera_combo.addItem("未检测到可用摄像头", -1)
                # 更新状态信息
                print("警告：未找到可能的摄像头索引")
                # 禁用启动按钮
                if hasattr(self, 'camera_btn'):
                    self.camera_btn.setEnabled(False)
            
            # 尝试恢复之前的选择
            index_to_select = self.camera_combo.findData(current_selection_data)
            if index_to_select != -1:
                self.camera_combo.setCurrentIndex(index_to_select)
            elif self.camera_combo.count() > 0: # 否则选择第一个（可能是RTSP）
                self.camera_combo.setCurrentIndex(0)
                # 确保启动按钮在有可用选项时启用（包括RTSP）
                if hasattr(self, 'camera_btn'):
                    self.camera_btn.setEnabled(True)
                    self.camera_btn.setStyleSheet(self.button_style_primary) # 确保是启动样式

    def on_camera_selection_changed(self, index):
        """处理摄像头下拉列表选择变化"""
        selected_data = self.camera_combo.itemData(index)
        is_rtsp = (selected_data == "rtsp")
        self.rtsp_url_input.setVisible(is_rtsp)

    def stop_camera_resources(self):
        """停止摄像头资源"""
        # 实现停止摄像头资源的逻辑
        pass
