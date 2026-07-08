#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MMPose姿态检测页面
"""

import os
import cv2
import numpy as np
import threading
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFrame, QComboBox, QFileDialog,
                             QGroupBox, QFormLayout, QDoubleSpinBox,
                             QCheckBox, QSpinBox, QMessageBox, QGridLayout,
                             QScrollArea, QSplitter, QLineEdit)
from PyQt6.QtGui import QPixmap, QIcon, QFont, QImage, QColor, QPainter
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtWidgets import QApplication

from .pose_3d_viewer import Pose3DViewer
from ...utils.camera_manager import CameraManager
from ...utils.mmpose_detector import MMPoseDetector
from ...utils.model_manager import ModelManager


class VideoWidget(QLabel):
    """视频显示组件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border-radius: 10px;
                padding: 0px;
                margin: 0px;
            }
        """)
        
        # 设置默认显示文本
        font = QFont("Arial", 13)
        font.setBold(True)
        self.setFont(font)
        self.setText("等待摄像头启动...")
        
        # 用于保存原始图像
        self.original_image = None

    def display_frame(self, qimage):
        """显示帧"""
        if qimage:
            self.original_image = qimage
            self.update_display()
    
    def update_display(self):
        """更新显示，确保图像完全填充并适应标签大小"""
        if self.original_image:
            # 获取控件的当前大小
            widget_size = self.size()
            
            if widget_size.width() <= 1 or widget_size.height() <= 1:
                return
                
            # 使用KeepAspectRatio确保图像不变形，但完全可见
            scaled_image = self.original_image.scaled(
                widget_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation
            )
            
            # 创建一个与控件大小相同的透明图像
            display_image = QImage(widget_size, QImage.Format.Format_ARGB32)
            display_image.fill(QColor(0, 0, 0, 0))
            
            # 计算居中位置
            x = (widget_size.width() - scaled_image.width()) // 2
            y = (widget_size.height() - scaled_image.height()) // 2
            
            # 在新图像上绘制缩放后的图像
            painter = QPainter(display_image)
            painter.drawImage(x, y, scaled_image)
            painter.end()
            
            # 设置显示
            self.setPixmap(QPixmap.fromImage(display_image))
    
    def resizeEvent(self, event):
        """重写调整大小事件以确保图像正确适应新大小"""
        super().resizeEvent(event)
        self.update_display()


class MMPosePage(QWidget):
    """MMPose姿态检测页面"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # 初始化模型路径
        self.det_config_path = None
        self.det_weight_path = None
        self.pose_config_path = None
        self.pose_weight_path = None

        # 创建摄像头管理器和姿态检测器
        self.camera_manager = CameraManager(self)
        self.camera_manager.camera_list_ready.connect(self.update_camera_list)
        
        # 使用全局模型实例
        self.model_manager = ModelManager()
        self.mmpose_detector = self.model_manager.get_detector()
        self.mmpose_detector.is_webcam = True
        
        # 初始化状态
        self.is_camera_running = False
        self.is_model_loading = False
        self.model_loaded = self.model_manager.is_loaded()
        
        # 信号连接
        self.camera_manager.camera_error.connect(self.on_camera_error)
        self.model_manager.model_loaded.connect(self.on_model_loaded)
        self.mmpose_detector.frame_processed.connect(self.on_frame_processed)
        self.mmpose_detector.pose_detected.connect(self.on_pose_detected)
        self.mmpose_detector.detection_error.connect(self.on_detection_error)

        # 初始化UI
        self.init_ui()

        # 创建models目录（如果不存在）- 修改为直接指向app/models目录
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.models_dir = os.path.join(app_dir, "models")
        
        # 确保models_dir路径存在
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(os.path.join(self.models_dir, "mmdetection_cfg"), exist_ok=True)
        
        # 保存上一帧有效的姿态数据
        self.last_valid_keypoints = None
        
        # 加载对话框引用
        self.loading_dialog = None
        
        # 页面加载完成后，开始异步加载相机列表
        QTimer.singleShot(500, self.camera_manager.scan_cameras_async)

    def init_ui(self):
        """初始化用户界面"""
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 创建主分割器
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setHandleWidth(2)
        self.main_splitter.setChildrenCollapsible(False)
        main_layout.addWidget(self.main_splitter)
        
        # 创建左侧分割器（垂直）
        self.left_splitter = QSplitter(Qt.Orientation.Vertical)
        self.left_splitter.setHandleWidth(1)
        self.left_splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(self.left_splitter)
        
        # 创建视频部分
        self.video_widget_container = self.create_video_section()
        self.left_splitter.addWidget(self.video_widget_container)
        
        # 创建3D模型部分
        self.model_widget = self.create_3d_model_section()
        self.left_splitter.addWidget(self.model_widget)
        
        # 创建右侧配置部分
        self.config_widget = self.create_config_section()
        self.main_splitter.addWidget(self.config_widget)
        
        # A设置初始分割比例
        self.main_splitter.setSizes([2000, 1000])  # 视频+3D模型区域占2/3，配置区域占1/3
        self.left_splitter.setSizes([1200, 800])   # 视频占60%，3D模型占40%
        
        # 应用全局样式
        self.apply_global_styles()
        
        # 连接摄像头信号
        self.camera_manager.frame_ready.connect(self.on_frame_ready)
        self.camera_manager.camera_error.connect(self.on_camera_error)
        self.camera_manager.camera_list_ready.connect(self.update_camera_list)
        
        # 连接检测器信号
        self.mmpose_detector.frame_processed.connect(self.on_frame_processed)
        self.mmpose_detector.pose_detected.connect(self.on_pose_detected)
        self.mmpose_detector.detection_error.connect(self.on_detection_error)
        
        # 初始化模型管理器
        self.model_manager.model_loaded.connect(self.on_model_loaded)
        
        # 如果检测器已加载模型，更新状态
        if self.mmpose_detector.is_model_loaded():
            self.model_loaded = True
        
        # 初始化摄像头列表
        self.camera_manager.scan_cameras_async()
        
        # 更新分割器大小
        QTimer.singleShot(100, self.update_splitter_sizes)

    def apply_global_styles(self):
        """应用全局样式"""
        # 设置应用程序的全局样式
        self.setStyleSheet("""
            QLabel {
                color: #374151;
            }
            QGroupBox {
                font-weight: 500;
                border: 1px solid #e9ecef;
                border-radius: 6px;
                margin-top: 14px;
                padding-top: 16px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #4b5563;
                background-color: #ffffff;
            }
            QDoubleSpinBox, QSpinBox {
                border: 1px solid #d1d5db;
                border-radius: 6px;
                padding: 6px 8px;
                color: #374151;
                background-color: #ffffff;
                min-height: 28px;
                selection-background-color: #dbeafe;
                selection-color: #1e40af;
            }
            QDoubleSpinBox:hover, QSpinBox:hover {
                border-color: #93c5fd;
            }
            QDoubleSpinBox:focus, QSpinBox:focus {
                border-color: #3b82f6;
                border-width: 1px;
            }
            QDoubleSpinBox::up-button, QSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 20px;
                border-top-right-radius: 4px;
                border: none;
                background-color: transparent;
                margin-top: 2px;
                margin-right: 2px;
            }
            QDoubleSpinBox::down-button, QSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 20px;
                border-bottom-right-radius: 4px;
                border: none;
                background-color: transparent;
                margin-bottom: 2px;
                margin-right: 2px;
            }
            QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {
                image: url(:/icons/arrow_up.png);
                width: 10px;
                height: 10px;
            }
            QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {
                image: url(:/icons/arrow_down.png);
                width: 10px;
                height: 10px;
            }
            QComboBox {
                border: 1px solid #edf0f5;
                border-radius: 6px;
                padding: 10px 12px;
                min-width: 150px;
                background-color: white;
                color: #18191c;
                font-size: 14px;
            }
            
            QComboBox:hover {
                border-color: #4361EE;
            }
            
            QComboBox:focus {
                border-color: #4361EE;
            }
            
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 30px;
                border-left-width: 0px;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
            }
            
            QComboBox::down-arrow {
                width: 12px;
                height: 12px;
            }
            
            QComboBox QAbstractItemView {
                border: 1px solid #edf0f5;
                border-radius: 4px;
                padding: 6px;
                selection-background-color: #EBF5FF;
                selection-color: #4361EE;
                background-color: white;
                color: #18191c;
            }
            QCheckBox {
                spacing: 8px;
                color: #374151;
                min-height: 24px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 1px solid #d1d5db;
                border-radius: 3px;
                background-color: #ffffff;
            }
            QCheckBox::indicator:hover {
                border-color: #93c5fd;
            }
            QCheckBox::indicator:checked {
                background-color: #3b82f6;
                border-color: #3b82f6;
                image: url(:/icons/check.png);
            }
            QScrollArea, QScrollArea > QWidget > QWidget {
                background-color: #ffffff;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #f8fafc;
                width: 12px;
                margin: 0px;
                border-radius: 6px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background-color: #cbd5e1;
                min-height: 30px;
                border-radius: 5px;
                margin: 2px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #94a3b8;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            QScrollBar:horizontal {
                background-color: #f8fafc;
                height: 12px;
                margin: 0px;
                border-radius: 6px;
                border: none;
            }
            QScrollBar::handle:horizontal {
                background-color: #cbd5e1;
                min-width: 30px;
                border-radius: 5px;
                margin: 2px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #94a3b8;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
            }
            QFrame {
                background-color: #ffffff;
            }
            /* 现代简约的分割器样式 */
            QSplitter::handle {
                background-color: transparent;
                border: none;
            }
            QSplitter::handle:horizontal {
                width: 2px;
                margin: 2px 4px;
                background-image: linear-gradient(to right, 
                                 transparent, 
                                 qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                         stop:0.2 #e2e8f0, 
                                         stop:0.5 #cbd5e1, 
                                         stop:0.8 #e2e8f0), 
                                 transparent);
            }
            QSplitter::handle:vertical {
                height: 2px;
                margin: 4px 2px;
                background-image: linear-gradient(to bottom, 
                                 transparent, 
                                 qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                         stop:0.2 #e2e8f0, 
                                         stop:0.5 #cbd5e1, 
                                         stop:0.8 #e2e8f0), 
                                 transparent);
            }
            QSplitter::handle:hover {
                background-color: transparent;
            }
            QSplitter::handle:horizontal:hover {
                background-image: linear-gradient(to right, 
                                 transparent, 
                                 qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                         stop:0.2 #94a3b8, 
                                         stop:0.5 #64748b, 
                                         stop:0.8 #94a3b8), 
                                 transparent);
            }
            QSplitter::handle:vertical:hover {
                background-image: linear-gradient(to bottom, 
                                 transparent, 
                                 qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                         stop:0.2 #94a3b8, 
                                         stop:0.5 #64748b, 
                                         stop:0.8 #94a3b8), 
                                 transparent);
            }
        """)

    def create_video_section(self):
        """创建视频区域"""
        video_frame = QFrame()
        video_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 10px;
                border: 1px solid #e9ecef;
            }
        """)
        
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(16, 16, 16, 16)
        video_layout.setSpacing(12)
        
        # 添加标题，去掉边框线
        video_title = QLabel("摄像头视图")
        video_title.setStyleSheet("""
            QLabel {
                color: #111827;
                font-size: 17px;
                font-weight: 600;
                border: none;
            }
        """)
        video_layout.addWidget(video_title)
        
        # 视频容器（使用QWidget而非直接使用VideoWidget）
        video_container = QWidget()
        video_container.setMinimumHeight(300)  # 降低最小高度以适应分割
        video_container_layout = QVBoxLayout(video_container)
        video_container_layout.setContentsMargins(0, 0, 0, 0)
        
        # 视频显示组件
        self.video_widget = VideoWidget()
        self.video_widget.setMinimumHeight(280)  # 降低最小高度
        video_container_layout.addWidget(self.video_widget)
        
        video_layout.addWidget(video_container, 1)  # 使用比例因子1允许扩展
        
        # 摄像头控制
        cam_controls = QHBoxLayout()
        cam_controls.setSpacing(10)
        
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(120)
        # RTSP URL 输入框
        self.rtsp_url_input = QLineEdit()
        self.rtsp_url_input.setPlaceholderText("RTSP URL")
        self.rtsp_url_input.hide() # 默认隐藏
        # 尝试应用类似 QComboBox 的样式，可能需要微调
        self.rtsp_url_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #edf0f5;
                border-radius: 8px;
                padding: 6px 10px;
                background-color: white;
                color: #18191c;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: #4361EE;
            }
        """)

        self.camera_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #edf0f5;
                border-radius: 8px;
                padding: 6px 10px;
                min-width: 75px;
                background-color: white;
                color: #18191c;
                font-size: 14px;
            }
            
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 25px;
                border-left-width: 0px;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
            }
            
            QComboBox::down-arrow {
                width: 10px;
                height: 10px;
            }
            
            QComboBox QAbstractItemView {
                border: 1px solid #edf0f5;
                border-radius: 4px;
                padding: 4px;
                selection-background-color: #EBF5FF;
                selection-color: #4361EE;
                background-color: white;
                color: #18191c;
            }
        """)
        
        # 初始状态下添加一个"正在检测摄像头..."选项
        self.camera_combo.addItem("正在检测摄像头...", -1)
        
        self.start_button = QPushButton("启动摄像头")
        self.start_button.setMinimumWidth(120)
        self.start_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4361ee;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: 500;
                padding: 8px 12px;
            }
            
            QPushButton:hover {
                background-color: #3a56d4;
            }
        """)
        self.start_button.clicked.connect(self.toggle_camera)
        
        cam_controls.addWidget(self.camera_combo)
        # 将 RTSP 输入框添加到摄像头控件布局中
        cam_controls.addWidget(self.rtsp_url_input)
        cam_controls.addWidget(self.start_button)
        cam_controls.addStretch()
        
        video_layout.addLayout(cam_controls)
        
        return video_frame

    def create_config_section(self):
        """创建配置区域"""
        config_frame = QFrame()
        config_frame.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 10px;
                border: 1px solid #e9ecef;
            }
        """)
        
        config_layout = QVBoxLayout(config_frame)
        config_layout.setContentsMargins(16, 16, 16, 16)
        config_layout.setSpacing(20)
        
        # 配置标题，去掉边框线
        config_title = QLabel("模型配置")
        config_title.setStyleSheet("""
            QLabel {
                color: #111827;
                font-size: 17px;
                font-weight: 600;
                border: none;
            }
        """)
        config_layout.addWidget(config_title)
        
        # 使用QScrollArea包装配置选项，防止窗口过小时显示不全
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }
        """)
        
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background-color: #ffffff;")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(20)
        
        # 添加模型文件设置区域
        model_files_group = self.create_model_files_section()
        scroll_layout.addWidget(model_files_group)
        
        # 添加参数设置区域
        param_group = self.create_parameters_section()
        scroll_layout.addWidget(param_group)
        
        # 添加可视化设置区域
        vis_group = self.create_visualization_section()
        scroll_layout.addWidget(vis_group)
        
        # 添加滚动区域
        scroll_area.setWidget(scroll_content)
        config_layout.addWidget(scroll_area)
        
        return config_frame

    def create_model_files_section(self):
        """创建模型文件设置区域"""
        model_files_group = QGroupBox("模型文件")
        model_files_group.setStyleSheet("""
            QGroupBox {
                background-color: #ffffff;
            }
            QLabel {
                border: none;
            }
        """)
        
        model_files_layout = QGridLayout(model_files_group)
        model_files_layout.setContentsMargins(16, 16, 16, 16)
        model_files_layout.setSpacing(12)
        model_files_layout.setColumnStretch(0, 1)  # 第一列伸展
        model_files_layout.setColumnStretch(1, 0)  # 第二列不伸展
        
        # 创建文本控件来展示当前选择的文件
        self.det_config_label = QLabel(os.path.basename(self.det_config_path) if self.det_config_path else "未选择")
        self.det_weight_label = QLabel(os.path.basename(self.det_weight_path) if self.det_weight_path else "未选择")
        self.pose_config_label = QLabel(os.path.basename(self.pose_config_path) if self.pose_config_path else "未选择")
        self.pose_weight_label = QLabel(os.path.basename(self.pose_weight_path) if self.pose_weight_path else "未选择")
        
        for label in [self.det_config_label, self.det_weight_label, 
                     self.pose_config_label, self.pose_weight_label]:
            label.setStyleSheet("""
                color: #6c757d; 
                padding: 4px;
                background-color: #f8f9fa;
                border-radius: 4px;
                border: none;
            """)
            label.setMinimumWidth(100)
        
        # 配置所有按钮的统一样式
        button_style = """
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """
        
        # 检测模型配置
        model_files_layout.addWidget(QLabel("检测模型配置:"), 0, 0)
        model_files_layout.addWidget(self.det_config_label, 0, 1)
        self.det_model_btn = QPushButton("选择文件")
        self.det_model_btn.setStyleSheet(button_style)
        self.det_model_btn.clicked.connect(self.select_det_config)
        model_files_layout.addWidget(self.det_model_btn, 0, 2)
        
        # 检测模型权重
        model_files_layout.addWidget(QLabel("检测模型权重:"), 1, 0)
        model_files_layout.addWidget(self.det_weight_label, 1, 1)
        self.det_weight_btn = QPushButton("选择文件")
        self.det_weight_btn.setStyleSheet(button_style)
        self.det_weight_btn.clicked.connect(self.select_det_weight)
        model_files_layout.addWidget(self.det_weight_btn, 1, 2)
        
        # 姿态模型配置
        model_files_layout.addWidget(QLabel("姿态模型配置:"), 2, 0)
        model_files_layout.addWidget(self.pose_config_label, 2, 1)
        self.pose_model_btn = QPushButton("选择文件")
        self.pose_model_btn.setStyleSheet(button_style)
        self.pose_model_btn.clicked.connect(self.select_pose_config)
        model_files_layout.addWidget(self.pose_model_btn, 2, 2)
        
        # 姿态模型权重
        model_files_layout.addWidget(QLabel("姿态模型权重:"), 3, 0)
        model_files_layout.addWidget(self.pose_weight_label, 3, 1)
        self.pose_weight_btn = QPushButton("选择文件")
        self.pose_weight_btn.setStyleSheet(button_style)
        self.pose_weight_btn.clicked.connect(self.select_pose_weight)
        model_files_layout.addWidget(self.pose_weight_btn, 3, 2)
        
        # 加载模型按钮
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.setStyleSheet("""
            QPushButton {
                background-color: #4361ee;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
                min-width: 120px;
            }
            
            QPushButton:hover {
                background-color: #3a56d4;
            }
            
            QPushButton:pressed {
                background-color: #2e4bbd;
            }
        """)
        self.load_model_btn.clicked.connect(self.load_models)
        
        # 添加Loading model按钮居中
        load_btn_layout = QHBoxLayout()
        load_btn_layout.addStretch(1)
        load_btn_layout.addWidget(self.load_model_btn)
        load_btn_layout.addStretch(1)
        model_files_layout.addLayout(load_btn_layout, 4, 0, 1, 3)  # 跨越3列
        
        return model_files_group

    def create_parameters_section(self):
        """创建参数设置区域"""
        param_group = QGroupBox("参数设置")
        param_group.setStyleSheet("""
            QLabel {
                border: none;
            }
            QSpinBox, QDoubleSpinBox {
                min-width: 80px;
            }
        """)
        
        param_layout = QGridLayout(param_group)
        param_layout.setContentsMargins(16, 16, 16, 16)
        param_layout.setHorizontalSpacing(12)
        param_layout.setVerticalSpacing(12)
        
        # 设备选择
        param_layout.addWidget(QLabel("设备:"), 0, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda:0", "cpu"])
        self.device_combo.currentTextChanged.connect(self.change_device)
        self.device_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #edf0f5;
                border-radius: 6px;
                padding: 10px 12px;
                min-width: 180px;
                background-color: white;
                color: #18191c;
                font-size: 14px;
            }
            
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 30px;
                border-left-width: 0px;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
            }
            
            QComboBox::down-arrow {
                width: 12px;
                height: 12px;
            }
            
            QComboBox QAbstractItemView {
                border: 1px solid #edf0f5;
                border-radius: 4px;
                padding: 6px;
                selection-background-color: #EBF5FF;
                selection-color: #4361EE;
                background-color: white;
                color: #18191c;
            }
        """)
        param_layout.addWidget(self.device_combo, 0, 1, 1, 2)
        
        # 添加分辨率设置
        param_layout.addWidget(QLabel("分辨率:"), 1, 0)
        resolution_layout = QHBoxLayout()
        
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(320, 1920)
        self.width_spinbox.setValue(640)
        self.width_spinbox.setSingleStep(16)
        
        self.height_spinbox = QSpinBox()
        self.height_spinbox.setRange(240, 1080)
        self.height_spinbox.setValue(480)
        self.height_spinbox.setSingleStep(16)
        
        resolution_layout.addWidget(self.width_spinbox)
        resolution_layout.addWidget(QLabel("x"))
        resolution_layout.addWidget(self.height_spinbox)
        
        param_layout.addLayout(resolution_layout, 1, 1, 1, 2)
        
        # 添加帧率设置
        param_layout.addWidget(QLabel("帧率:"), 2, 0)
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(5, 60)
        self.fps_spinbox.setValue(30)
        self.fps_spinbox.setSingleStep(5)
        param_layout.addWidget(self.fps_spinbox, 2, 1)
        
        # 设置阈值
        param_layout.addWidget(QLabel("边界框阈值:"), 3, 0)
        self.bbox_thr_spin = QDoubleSpinBox()
        self.bbox_thr_spin.setRange(0.01, 1.0)
        self.bbox_thr_spin.setValue(0.5)
        self.bbox_thr_spin.setSingleStep(0.05)
        self.bbox_thr_spin.valueChanged.connect(self.update_detection_params)
        param_layout.addWidget(self.bbox_thr_spin, 3, 1)
        
        param_layout.addWidget(QLabel("关键点阈值:"), 4, 0)
        self.kpt_thr_spin = QDoubleSpinBox()
        self.kpt_thr_spin.setRange(0.01, 1.0)
        self.kpt_thr_spin.setValue(0.5)
        self.kpt_thr_spin.setSingleStep(0.05)
        self.kpt_thr_spin.valueChanged.connect(self.update_detection_params)
        param_layout.addWidget(self.kpt_thr_spin, 4, 1)
        
        param_layout.addWidget(QLabel("NMS阈值:"), 5, 0)
        self.nms_thr_spin = QDoubleSpinBox()
        self.nms_thr_spin.setRange(0.01, 1.0)
        self.nms_thr_spin.setValue(0.3)
        self.nms_thr_spin.setSingleStep(0.05)
        self.nms_thr_spin.valueChanged.connect(self.update_detection_params)
        param_layout.addWidget(self.nms_thr_spin, 5, 1)
        
        return param_group

    def create_visualization_section(self):
        """创建可视化设置区域"""
        vis_group = QGroupBox("可视化设置")
        vis_group.setStyleSheet("""
            QLabel {
                border: none;
            }
            QSpinBox {
                min-width: 80px;
            }
        """)
        
        vis_layout = QGridLayout(vis_group)
        vis_layout.setContentsMargins(16, 16, 16, 16)
        vis_layout.setHorizontalSpacing(12)
        vis_layout.setVerticalSpacing(12)
        
        # 可视化选项
        self.draw_bbox_check = QCheckBox("显示边界框")
        self.draw_bbox_check.setChecked(True)
        self.draw_bbox_check.stateChanged.connect(self.update_visual_params)
        vis_layout.addWidget(self.draw_bbox_check, 0, 0)
        
        # 显示关键点
        self.draw_kpt_check = QCheckBox("显示关键点")
        self.draw_kpt_check.setChecked(True)
        self.draw_kpt_check.stateChanged.connect(self.update_visual_params)
        vis_layout.addWidget(self.draw_kpt_check, 0, 1)
        
        # 显示骨架
        self.draw_skeleton_check = QCheckBox("显示骨架")
        self.draw_skeleton_check.setChecked(True)
        self.draw_skeleton_check.stateChanged.connect(self.update_visual_params)
        vis_layout.addWidget(self.draw_skeleton_check, 1, 0)
        
        # 显示关键点索引
        self.show_kpt_idx_check = QCheckBox("显示关键点索引")
        self.show_kpt_idx_check.setChecked(False)
        self.show_kpt_idx_check.stateChanged.connect(self.update_visual_params)
        vis_layout.addWidget(self.show_kpt_idx_check, 1, 1)
        
        # 关键点半径
        vis_layout.addWidget(QLabel("关键点半径:"), 2, 0)
        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(1, 20)
        self.radius_spin.setValue(4)
        self.radius_spin.valueChanged.connect(self.update_visual_params)
        vis_layout.addWidget(self.radius_spin, 2, 1)
        
        # 线条粗细
        vis_layout.addWidget(QLabel("线条粗细:"), 3, 0)
        self.thickness_spin = QSpinBox()
        self.thickness_spin.setRange(1, 10)
        self.thickness_spin.setValue(2)
        self.thickness_spin.valueChanged.connect(self.update_visual_params)
        vis_layout.addWidget(self.thickness_spin, 3, 1)
        
        # 绘制热度图
        self.draw_heatmap_check = QCheckBox("绘制热度图")
        self.draw_heatmap_check.setChecked(False)
        self.draw_heatmap_check.stateChanged.connect(self.update_visual_params)
        vis_layout.addWidget(self.draw_heatmap_check, 4, 0)
        
        # 显示FPS信息
        self.draw_fps_check = QCheckBox("显示FPS信息")
        self.draw_fps_check.setChecked(False)
        self.draw_fps_check.stateChanged.connect(lambda checked: self.mmpose_detector.set_visual_params(draw_fps=checked))
        vis_layout.addWidget(self.draw_fps_check, 4, 1)
        
        return vis_group

    def create_3d_model_section(self):
        """创建3D模型显示区域"""
        model_frame = QFrame()
        model_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 10px;
                border: 1px solid #e9ecef;
            }
        """)
        
        model_layout = QVBoxLayout(model_frame)
        model_layout.setContentsMargins(16, 16, 16, 16)
        model_layout.setSpacing(12)
        
        # 添加标题，去掉边框线
        model_title = QLabel("3D姿态模型")
        model_title.setStyleSheet("""
            QLabel {
                color: #111827;
                font-size: 17px;
                font-weight: 600;
                border: none;
            }
        """)
        model_layout.addWidget(model_title)
        
        # 3D姿态显示器
        self.pose_3d_viewer = Pose3DViewer()
        self.pose_3d_viewer.setMinimumHeight(200)  # 降低最小高度以适应分割
        model_layout.addWidget(self.pose_3d_viewer, 1)  # 使用比例因子1允许扩展
        
        # 控制按钮
        pose_controls = QHBoxLayout()
        pose_controls.setSpacing(10)
        
        # 自动旋转按钮
        self.auto_rotate_btn = QPushButton("自动旋转")
        self.auto_rotate_btn.setMinimumWidth(120)
        self.auto_rotate_btn.setCheckable(True)
        self.auto_rotate_btn.setChecked(True)
        self.auto_rotate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_rotate_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9a3e;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: 500;
                padding: 8px 16px;
            }
            
            QPushButton:hover {
                background-color: #f58220;
            }
            
            QPushButton:checked {
                background-color: #f58220;
            }
        """)
        self.auto_rotate_btn.clicked.connect(
            lambda checked: self.pose_3d_viewer.toggle_auto_rotate(checked))
        pose_controls.addWidget(self.auto_rotate_btn)

        # 重置视图按钮
        reset_view_btn = QPushButton("重置视图")
        reset_view_btn.setMinimumWidth(120)
        reset_view_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reset_view_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: 500;
                padding: 8px 16px;
            }
            
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        reset_view_btn.clicked.connect(self.pose_3d_viewer.reset_view)
        pose_controls.addWidget(reset_view_btn)
        
        pose_controls.addStretch(1)  # 添加弹性空间
        model_layout.addLayout(pose_controls)
        
        return model_frame

    def toggle_camera(self):
        """切换摄像头状态"""
        if self.is_camera_running:
            # 停止摄像头
            self.camera_manager.stop()
            self.mmpose_detector.stop()
            
            # 更新按钮状态
            self.start_button.setText("启动摄像头")
            self.start_button.setStyleSheet(self.create_button_style("#4361ee"))
            
            # 更新状态
            self.is_camera_running = False
            
            # 重置姿态数据
            self.last_valid_keypoints = None
            self.pose_3d_viewer.reset_view()
            
        else:
            # 获取摄像头ID或RTSP URL
            selected_data = self.camera_combo.currentData()
            camera_id_or_url = None
            is_rtsp_selected = False

            if selected_data == "rtsp":
                rtsp_url = self.rtsp_url_input.text().strip()
                if not rtsp_url:
                    QMessageBox.warning(self, "警告", "请输入有效的 RTSP URL")
                    return
                if not rtsp_url.lower().startswith("rtsp://"):
                    QMessageBox.warning(self, "警告", "RTSP URL 必须以 rtsp:// 开头")
                    return
                camera_id_or_url = rtsp_url
                is_rtsp_selected = True
            elif isinstance(selected_data, int) and selected_data != -1:
                camera_id_or_url = selected_data
            else:
                QMessageBox.warning(self, "警告", "请先选择一个有效的摄像头或 RTSP 视频流")
                 # 尝试重新扫描摄像头以防列表过时
                self.camera_manager.scan_cameras_async()
                return

            # 设置摄像头ID或URL
            self.camera_manager.set_camera_id(camera_id_or_url)

            # 获取并设置参数（仅对非RTSP流有效）
            try:
                if not is_rtsp_selected:
                    # 设置分辨率
                    width = self.width_spinbox.value()
                    height = self.height_spinbox.value()
                    self.camera_manager.set_resolution(width, height)

                    # 设置帧率
                    fps = self.fps_spinbox.value()
                    self.camera_manager.set_fps(fps)
                else:
                    # 对于RTSP，可以记录但不设置分辨率/帧率，因为它们由流决定
                    logger = self.camera_manager._get_logger() # 获取logger实例
                    logger.info("RTSP流选定，将忽略本地分辨率和帧率设置。")

                # 启动相机
                if self.camera_manager.start():
                    # 更新按钮状态
                    self.start_button.setText("停止摄像头")
                    self.start_button.setStyleSheet(self.create_button_style("#ef4444"))
                    
                    # 启动检测器
                    self.mmpose_detector.start()
                    
                    # 更新状态
                    self.is_camera_running = True
                else:
                    QMessageBox.critical(self, "错误", "无法打开摄像头，请尝试以下操作：\n\n1. 检查是否选择了正确的摄像头\n2. 确认没有其他程序正在使用该摄像头\n3. 尝试使用其他摄像头索引\n4. 重新插拔摄像头或重启电脑")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"启动摄像头时出错: {str(e)}\n\n可能原因：\n1. 摄像头被占用\n2. 摄像头驱动问题\n3. 设备权限问题")
    
    def update_detector_params(self):
        """更新检测器参数"""
        try:
            # 检测参数
            bbox_thr = self.bbox_thr_spin.value()
            kpt_thr = self.kpt_thr_spin.value()
            nms_thr = self.nms_thr_spin.value()
            
            if hasattr(self, 'mmpose_detector'):
                self.mmpose_detector.set_detection_params(
                    bbox_thr=bbox_thr,
                    kpt_thr=kpt_thr,
                    nms_thr=nms_thr
                )
        except Exception as e:
            print(f"更新参数错误: {str(e)}")
    
    def update_visual_params(self):
        """更新可视化参数"""
        try:
            # 可视化参数
            radius = self.radius_spin.value()
            thickness = self.thickness_spin.value()
            draw_bbox = self.draw_bbox_check.isChecked()
            draw_heatmap = self.draw_heatmap_check.isChecked()
            show_kpt_idx = self.show_kpt_idx_check.isChecked()
            draw_fps = self.draw_fps_check.isChecked()
            draw_keypoints = self.draw_kpt_check.isChecked()
            draw_skeleton = self.draw_skeleton_check.isChecked()
            
            if hasattr(self, 'mmpose_detector'):
                self.mmpose_detector.set_visual_params(
                    radius=radius,
                    thickness=thickness,
                    draw_bbox=draw_bbox,
                    draw_heatmap=draw_heatmap,
                    show_kpt_idx=show_kpt_idx,
                    draw_fps=draw_fps,
                    draw_keypoints=draw_keypoints,
                    draw_skeleton=draw_skeleton
                )
        except Exception as e:
            print(f"更新参数错误: {str(e)}")
    
    def change_device(self, device):
        """切换设备"""
        if hasattr(self, 'mmpose_detector'):
            current_device = "cuda:0" if self.mmpose_detector.device.startswith("cuda") else "cpu"
            
            # 切换设备
            self.mmpose_detector.set_device(device)
            
    def update_device(self):
        """更新计算设备"""
        try:
            # 使用GPU
            use_gpu = self.gpu_radio.isChecked()
            
            if hasattr(self, 'mmpose_detector'):
                # 确定当前设备和新设备
                current_device = "cuda:0" if self.mmpose_detector.device.startswith("cuda") else "cpu"
                new_device = "cuda:0" if use_gpu else "cpu"
                
                # 仅当设备变更时重新加载模型
                if current_device != new_device:
                    self.mmpose_detector.set_device(new_device)
                    if self.is_camera_running:
                        self.toggle_camera()  # 停止摄像头
                        QMessageBox.information(self, "设备变更", "设备已更改，请重新启动摄像头")
            
        except Exception as e:
            print(f"更新设备错误: {str(e)}")

    def load_custom_model(self):
        """加载自定义模型"""
        try:
            # 获取当前配置
            det_config = self.det_config_edit.text()
            det_weight = self.det_weight_edit.text()
            pose_config = self.pose_config_edit.text()
            pose_weight = self.pose_weight_edit.text()
            
            # 验证路径
            if not all([det_config, det_weight, pose_config, pose_weight]):
                QMessageBox.warning(self, "路径不完整", "请确保所有模型路径都已填写")
                return
                
            # 验证文件是否存在
            for path, name in [
                (det_config, "检测模型配置"),
                (det_weight, "检测模型权重"),
                (pose_config, "姿态模型配置"),
                (pose_weight, "姿态模型权重")
            ]:
                if not os.path.exists(path):
                    QMessageBox.warning(self, "文件不存在", f"{name}文件不存在: {path}")
                    return
            
            # 停止摄像头（如果正在运行）
            was_running = self.is_camera_running
            if was_running:
                self.toggle_camera()
            
            loading_dialog = QMessageBox(self)
            loading_dialog.setWindowTitle("加载模型")
            loading_dialog.setText("正在加载自定义模型，请稍候...")
            loading_dialog.setStandardButtons(QMessageBox.StandardButton.NoButton)
            loading_dialog.setIcon(QMessageBox.Icon.Information)
            
            # 非模态显示
            loading_dialog.show()
            QApplication.processEvents()
            
            # 加载模型
            success = self.mmpose_detector.load_models(
                det_config=det_config,
                det_checkpoint=det_weight,
                pose_config=pose_config,
                pose_checkpoint=pose_weight
            )
            
            # 关闭对话框
            loading_dialog.close()
            
            # 更新状态
            if success:
                self.model_loaded = True
                QMessageBox.information(self, "成功", "自定义模型加载成功")
                
                # 如果之前正在运行，重新启动摄像头
                if was_running:
                    self.toggle_camera()
            else:
                QMessageBox.critical(self, "错误", "无法加载自定义模型，请检查文件")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载自定义模型时出错: {str(e)}")
    
    def connect_camera(self):
        """连接相机和检测器信号"""
        try:
            # 模型初始化
            if not self.model_manager.is_loaded() and not self.mmpose_detector.is_model_loaded():
                
                # 显示加载对话框
                loading_dialog = QMessageBox(self)
                loading_dialog.setWindowTitle("加载模型")
                loading_dialog.setText("正在加载姿态检测模型，请稍候...")
                loading_dialog.setStandardButtons(QMessageBox.StandardButton.NoButton)
                loading_dialog.setIcon(QMessageBox.Icon.Information)
                
                # 非模态显示
                loading_dialog.show()
                QApplication.processEvents()
                
                # 加载模型
                success = self.model_manager.load_model()
                
                # 关闭对话框
                loading_dialog.close()
                
                if success:
                    self.model_loaded = True
                else:
                    QMessageBox.critical(self, "错误", "无法加载检测模型，请检查模型文件")
                    return False
            
            # 启动摄像头
            self.toggle_camera()
            return True
        
        except Exception as e:
            QMessageBox.critical(self, "错误", f"连接摄像头时出错: {str(e)}")
            return False

    def select_det_config(self):
        """选择检测模型配置文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择检测模型配置文件", self.models_dir, "Python文件 (*.py)")
        if file_path:
            self.det_config_path = file_path
            self.det_config_label.setText(os.path.basename(file_path))

    def select_det_weight(self):
        """选择检测模型权重文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择检测模型权重文件", self.models_dir, "PyTorch权重文件 (*.pth)")
        if file_path:
            self.det_weight_path = file_path
            self.det_weight_label.setText(os.path.basename(file_path))

    def select_pose_config(self):
        """选择姿态模型配置文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择姿态模型配置文件", self.models_dir, "Python文件 (*.py)")
        if file_path:
            self.pose_config_path = file_path
            self.pose_config_label.setText(os.path.basename(file_path))

    def select_pose_weight(self):
        """选择姿态模型权重文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择姿态模型权重文件", self.models_dir, "PyTorch权重文件 (*.pth)")
        if file_path:
            self.pose_weight_path = file_path
            self.pose_weight_label.setText(os.path.basename(file_path))

    def load_models(self):
        """加载模型"""
        # 如果相机正在运行，先停止相机
        was_running = self.is_camera_running
        if was_running:
            self.toggle_camera()
        
        # 创建一个模态对话框以确保它能被关闭
        if self.loading_dialog is None:
            self.loading_dialog = QMessageBox()
            self.loading_dialog.setWindowTitle("加载模型")
            self.loading_dialog.setText("正在加载姿态检测模型，请稍候...")
            self.loading_dialog.setStandardButtons(QMessageBox.StandardButton.NoButton)
            self.loading_dialog.setIcon(QMessageBox.Icon.Information)
        
        # 让对话框不阻塞但在模型加载过程中可见
        self.loading_dialog.show()
        QApplication.processEvents()
        
        # 加载模型
        success = False
        try:
            success = self.mmpose_detector.load_models(
                det_config=self.det_config_path,
                det_checkpoint=self.det_weight_path,
                pose_config=self.pose_config_path,
                pose_checkpoint=self.pose_weight_path
            )
        except Exception as e:
            print(f"模型加载错误: {str(e)}")
        
        # 无论成功与否都关闭对话框
        if self.loading_dialog:
            self.loading_dialog.accept()  # 使用accept()强制关闭
            self.loading_dialog = None
            QApplication.processEvents()
        
        if success:
            QMessageBox.information(self, "提示", "模型加载成功")
            
            # 如果之前相机在运行，重新启动相机
            if was_running:
                self.toggle_camera()
        else:
            QMessageBox.critical(self, "错误", "模型加载失败，请检查模型文件路径是否正确")

    @pyqtSlot(np.ndarray)
    def on_frame_ready(self, frame):
        """当摄像头捕获到新帧时调用"""
        if frame is not None and self.mmpose_detector is not None:
            # 将帧传递给MMPose检测器处理
            self.mmpose_detector.process_frame(frame)

    @pyqtSlot(str)
    def on_camera_error(self, error_msg):
        """处理摄像头错误"""
        QMessageBox.critical(self, "摄像头错误", error_msg)
        # 如果摄像头正在运行，停止它
        if self.is_camera_running:
            self.toggle_camera()

    @pyqtSlot(str)
    def on_detection_error(self, error_msg):
        """处理检测错误"""
        QMessageBox.critical(self, "检测错误", error_msg)

    @pyqtSlot(QImage)
    def on_frame_processed(self, qimage):
        """显示处理后的帧"""
        if self.video_widget is not None:
            # 使用VideoWidget的display_frame方法显示图像
            self.video_widget.display_frame(qimage)

    @pyqtSlot(object)
    def on_pose_detected(self, keypoints):
        """处理检测到的姿态"""
        # 检查关键点是否有效
        if keypoints is not None and np.any(keypoints):
            # 保存有效的关键点
            self.last_valid_keypoints = keypoints.copy()
            # 更新3D模型
            if hasattr(self, 'pose_3d_viewer'):
                self.pose_3d_viewer.update_keypoints(keypoints)
            # 更新状态信息
        elif self.last_valid_keypoints is not None:
            # 使用上一次有效的关键点
            self.pose_3d_viewer.update_keypoints(self.last_valid_keypoints)

    def update_detection_params(self):
        """更新检测参数"""
        if hasattr(self, 'mmpose_detector'):
            self.mmpose_detector.set_detection_params(
                bbox_thr=self.bbox_thr_spin.value(),
                kpt_thr=self.kpt_thr_spin.value(),
                nms_thr=self.nms_thr_spin.value()
            )

    def update_visual_params(self):
        """更新可视化参数"""
        if hasattr(self, 'mmpose_detector'):
            self.mmpose_detector.set_visual_params(
                radius=self.radius_spin.value(),
                thickness=self.thickness_spin.value(),
                draw_bbox=self.draw_bbox_check.isChecked(),
                draw_heatmap=self.draw_heatmap_check.isChecked(),
                show_kpt_idx=self.show_kpt_idx_check.isChecked()
            )

    def resizeEvent(self, event):
        """窗口大小调整事件"""
        super().resizeEvent(event)
        
        # 重新调整QSplitter的大小比例
        if hasattr(self, 'main_splitter') and hasattr(self, 'left_splitter'):
            # 更新时保持分割器的相对比例
            self.update_splitter_sizes()

    def update_splitter_sizes(self):
        """更新分割器的大小，以保持适当的比例"""
        # 处理左右主分割器
        main_sizes = self.main_splitter.sizes()
        total_main = sum(main_sizes)
        if total_main > 0:
            main_ratio = main_sizes[0] / total_main
            # 主分割器使用当前的实际宽度计算
            new_main_width = self.width() - 40  # 考虑边距
            self.main_splitter.setSizes([int(new_main_width * main_ratio), 
                                        int(new_main_width * (1 - main_ratio))])
        
        # 处理上下垂直分割器
        left_sizes = self.left_splitter.sizes()
        total_left = sum(left_sizes)
        if total_left > 0:
            left_ratio = left_sizes[0] / total_left
            # 垂直分割器使用主分割器左侧的实际高度
            left_widget_height = self.main_splitter.widget(0).height() - 20  # 考虑边距
            self.left_splitter.setSizes([int(left_widget_height * 0.6), 
                                         int(left_widget_height * 0.4)])

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
                # 默认选择第一个摄像头
                self.camera_combo.setCurrentIndex(0)
                # 如果有摄像头，启用启动按钮
                if hasattr(self, 'start_button'):
                    self.start_button.setEnabled(True)
                    self.start_button.setStyleSheet(self.create_button_style("#4361ee"))  # 蓝色按钮
            else:
                self.camera_combo.addItem("未检测到可用摄像头", -1)
                # 如果摄像头列表为空，禁用启动按钮
                if hasattr(self, 'start_button'):
                    self.start_button.setEnabled(False)
                    self.start_button.setStyleSheet(self.create_button_style("#9ca3af"))  # 灰色按钮
            
            # 尝试恢复之前的选择
            index_to_select = self.camera_combo.findData(current_selection_data)
            if index_to_select != -1:
                self.camera_combo.setCurrentIndex(index_to_select)
            elif self.camera_combo.count() > 0: # 否则选择第一个（可能是RTSP）
                self.camera_combo.setCurrentIndex(0)
                # 确保启动按钮在有可用选项时启用（包括RTSP）
                self.start_button.setEnabled(True)
                self.start_button.setStyleSheet(self.create_button_style("#4361ee"))
            
            # 手动触发一次选择检查
            self.on_camera_selection_changed(self.camera_combo.currentIndex())

    @pyqtSlot(bool)
    def on_model_loaded(self, success):
        """模型加载完成的回调"""
        self.model_loaded = success
        # 更新UI状态
        if hasattr(self, 'model_status_label'):
            if success:
                self.model_status_label.setText("模型已加载")
                self.model_status_label.setStyleSheet("color: #10b981;")  # 绿色
            else:
                self.model_status_label.setText("模型加载失败")
                self.model_status_label.setStyleSheet("color: #ef4444;")  # 红色 

    def create_button_style(self, bg_color):
        """创建按钮样式表
        
        Args:
            bg_color: 背景颜色（CSS颜色值）
            
        Returns:
            按钮样式表字符串
        """
        return f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: 500;
                padding: 8px 12px;
            }}
            
            QPushButton:hover {{
                background-color: {self._adjust_color(bg_color, -10)};
            }}
            
            QPushButton:disabled {{
                background-color: #d1d5db;
                color: #9ca3af;
            }}
        """

    def _adjust_color(self, hex_color, factor):
        """调整颜色亮度
        
        Args:
            hex_color: 十六进制颜色值
            factor: 调整因子（正数增亮，负数变暗）
            
        Returns:
            调整后的十六进制颜色值
        """
        # 如果颜色值以#开头，去掉#
        if hex_color.startswith('#'):
            hex_color = hex_color[1:]
        
        # 解析RGB值
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        # 调整亮度
        r = max(0, min(255, r + factor))
        g = max(0, min(255, g + factor))
        b = max(0, min(255, b + factor))
        
        # 转回十六进制
        return f"#{r:02x}{g:02x}{b:02x}"

    def on_camera_selection_changed(self, index):
        """处理摄像头下拉列表选择变化"""
        selected_data = self.camera_combo.itemData(index)
        is_rtsp = (selected_data == "rtsp")
        self.rtsp_url_input.setVisible(is_rtsp)

        # 对于物理摄像头，可以考虑禁用或隐藏分辨率/帧率输入（如果它们在MMPosePage中）
        # 目前分辨率和帧率在 toggle_camera 中从 spinbox 获取，可以考虑禁用它们
        if hasattr(self, 'width_spinbox') and hasattr(self, 'height_spinbox'):
            self.width_spinbox.setEnabled(not is_rtsp)
            self.height_spinbox.setEnabled(not is_rtsp)
        if hasattr(self, 'fps_spinbox'):
            self.fps_spinbox.setEnabled(not is_rtsp) 