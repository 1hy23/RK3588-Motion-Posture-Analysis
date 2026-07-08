#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
设置页面
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFrame, QCheckBox, QComboBox,
                             QSlider, QLineEdit, QFormLayout, QGroupBox,
                             QScrollArea, QSizePolicy)
from PyQt6.QtGui import QPixmap, QIcon, QFont
from PyQt6.QtCore import Qt, QSize, QTimer

from ...utils.camera_manager import CameraManager


class SettingsGroup(QGroupBox):
    """设置分组"""

    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #edf0f5;
                border-radius: 10px;
                margin-top: 1.5ex;
                padding-top: 14px;
                background-color: white;
                color: #18191c;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                color: #18191c;
                font-size: 16px;
            }
        """)


class SettingsPage(QWidget):
    """设置页面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        # 创建摄像头管理器但不立即扫描
        self.camera_manager = CameraManager(self)
        self.camera_manager.camera_list_ready.connect(self.update_camera_list)
        
        # 初始化UI
        self.init_ui()
        
        # 页面显示后再开始异步扫描摄像头
        QTimer.singleShot(500, self.camera_manager.scan_cameras_async)

    def init_ui(self):
        """初始化UI"""
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 页面标题
        title_label = QLabel("设置")
        title_label.setStyleSheet("""
            QLabel {
                color: #18191c;
                font-size: 26px;
                font-weight: bold;
                margin-bottom: 10px;
            }
        """)
        main_layout.addWidget(title_label)

        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: #FFFFFF;
                border: none;
                border-radius: 10px;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 8px;
                margin: 0px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #c0c0c0;
                min-height: 30px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #a0a0a0;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                border: none;
                background: #f0f0f0;
                height: 8px;
                margin: 0px;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal {
                background: #c0c0c0;
                min-width: 30px;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #a0a0a0;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)

        # 创建内容容器
        content_widget = QWidget()
        content_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        content_widget.setStyleSheet("background-color: #FFFFFF; border-radius: 10px;")
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(15, 15, 15, 15)
        content_layout.setSpacing(20)

        # 相机设置组
        camera_group = SettingsGroup("相机设置")
        camera_layout = QFormLayout(camera_group)
        camera_layout.setSpacing(15)
        camera_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        camera_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        camera_layout.setContentsMargins(25, 30, 25, 30)
        
        # 设置字体大小
        font = QFont()
        font.setPointSize(14)  # 增大字号
        camera_group.setFont(font)

        # 相机选择
        self.camera_combo = QComboBox()
        self.camera_combo.setObjectName("camera_combo")
        
        # RTSP URL 输入框
        self.rtsp_url_input = QLineEdit()
        self.rtsp_url_input.setPlaceholderText("请输入 RTSP URL (例如 rtsp://...)")
        self.rtsp_url_input.hide() # 默认隐藏
        self.rtsp_url_input.setStyleSheet(self.camera_combo.styleSheet().replace("QComboBox", "QLineEdit")) # 借用样式

        # 为 RTSP 输入框创建一个隐藏的标签 (占位符，用于布局)
        self.rtsp_label = QLabel("")
        self.rtsp_label.hide()

        # 初始状态下添加一个"正在检测摄像头..."选项
        self.camera_combo.addItem("正在检测摄像头...", -1)
            
        self.camera_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #edf0f5;
                border-radius: 6px;
                padding: 8px 12px;
                min-width: 220px;
                background-color: #ffffff;
                color: #18191c;
                font-size: 14px;
            }
            
            QComboBox:hover {
                border-color: #4361EE;
            }
            
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 28px;
                border-left-width: 0px;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
            }
            
            QComboBox QAbstractItemView {
                border: 1px solid #edf0f5;
                border-radius: 6px;
                padding: 6px;
                selection-background-color: #EBF5FF;
                selection-color: #4361EE;
                background-color: white;
                color: #18191c;
            }
        """)
        camera_layout.addRow(QLabel("<span style='font-size:14px; color:#18191c;'>相机设备:</span>"), self.camera_combo)
        # 直接添加标签和输入框
        camera_layout.addRow(self.rtsp_label, self.rtsp_url_input)

        # 分辨率设置
        resolution_combo = QComboBox()
        resolution_combo.addItems(["640x480", "1280x720", "1920x1080"])
        resolution_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #edf0f5;
                border-radius: 6px;
                padding: 8px 12px;
                min-width: 220px;
                background-color: #ffffff;
                color: #18191c;
                font-size: 14px;
            }
            
            QComboBox:hover {
                border-color: #4361EE;
            }
            
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 28px;
                border-left-width: 0px;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
            }
            
            QComboBox QAbstractItemView {
                border: 1px solid #edf0f5;
                border-radius: 6px;
                padding: 6px;
                selection-background-color: #EBF5FF;
                selection-color: #4361EE;
                background-color: white;
                color: #18191c;
            }
        """)
        camera_layout.addRow(QLabel("<span style='font-size:14px; color:#18191c;'>分辨率:</span>"), resolution_combo)

        # 帧率设置
        fps_combo = QComboBox()
        fps_combo.addItems(["15 FPS", "30 FPS", "60 FPS"])
        fps_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #edf0f5;
                border-radius: 6px;
                padding: 8px 12px;
                min-width: 220px;
                background-color: #ffffff;
                color: #18191c;
                font-size: 14px;
            }
            
            QComboBox:hover {
                border-color: #4361EE;
            }
            
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 28px;
                border-left-width: 0px;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
            }
            
            QComboBox QAbstractItemView {
                border: 1px solid #edf0f5;
                border-radius: 6px;
                padding: 6px;
                selection-background-color: #EBF5FF;
                selection-color: #4361EE;
                background-color: white;
                color: #18191c;
            }
        """)
        camera_layout.addRow(QLabel("<span style='font-size:14px; color:#18191c;'>帧率:</span>"), fps_combo)

        content_layout.addWidget(camera_group)

        # 检测设置组
        detection_group = SettingsGroup("姿态检测设置")
        detection_layout = QFormLayout(detection_group)
        detection_layout.setSpacing(15)
        detection_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        detection_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        detection_layout.setContentsMargins(25, 30, 25, 30)
        
        # 设置字体大小
        detection_group.setFont(font)

        # 检测模式
        mode_combo = QComboBox()
        mode_combo.addItems(["精确模式", "快速模式", "平衡模式"])
        mode_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #edf0f5;
                border-radius: 6px;
                padding: 8px 12px;
                min-width: 220px;
                background-color: #ffffff;
                color: #18191c;
                font-size: 14px;
            }
            
            QComboBox:hover {
                border-color: #4361EE;
            }
            
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 28px;
                border-left-width: 0px;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
            }
            
            QComboBox QAbstractItemView {
                border: 1px solid #edf0f5;
                border-radius: 6px;
                padding: 6px;
                selection-background-color: #EBF5FF;
                selection-color: #4361EE;
                background-color: white;
                color: #18191c;
            }
        """)
        detection_layout.addRow(QLabel("<span style='font-size:14px; color:#18191c;'>检测模式:</span>"), mode_combo)

        # 置信度阈值
        confidence_slider = QSlider(Qt.Orientation.Horizontal)
        confidence_slider.setMinimum(0)
        confidence_slider.setMaximum(100)
        confidence_slider.setValue(50)
        confidence_slider.setFixedWidth(220)
        confidence_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #edf0f5;
                height: 8px;
                background: #f8f9fa;
                margin: 2px 0;
                border-radius: 4px;
            }
            
            QSlider::handle:horizontal {
                background: #4361EE;
                border: 1px solid #4361EE;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            
            QSlider::handle:horizontal:hover {
                background: #3A56D4;
                border: 1px solid #3A56D4;
            }
        """)

        conf_layout = QHBoxLayout()
        conf_layout.addWidget(confidence_slider)
        conf_value = QLabel("50%")
        conf_value.setMinimumWidth(40)
        conf_value.setStyleSheet("color: #18191c; font-size: 14px;")
        conf_layout.addWidget(conf_value)

        # 连接滑块值变化信号
        confidence_slider.valueChanged.connect(
            lambda v: conf_value.setText(f"{v}%"))

        detection_layout.addRow(QLabel("<span style='font-size:14px; color:#18191c;'>置信度阈值:</span>"), conf_layout)

        # 显示骨架
        skeleton_check = QCheckBox("显示检测骨架")
        skeleton_check.setChecked(True)
        skeleton_check.setStyleSheet("""
            QCheckBox {
                spacing: 6px;
                color: #18191c;
                font-size: 14px;
            }
            
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            
            QCheckBox::indicator:unchecked {
                border: 2px solid #edf0f5;
                background-color: white;
                border-radius: 3px;
            }
            
            QCheckBox::indicator:checked {
                border: 2px solid #4361EE;
                background-color: #4361EE;
                border-radius: 3px;
            }
            
            QCheckBox::indicator:hover {
                border-color: #3A6EC3;
            }
        """)
        detection_layout.addRow("", skeleton_check)

        # 显示关键点
        keypoints_check = QCheckBox("显示关键点")
        keypoints_check.setChecked(True)
        keypoints_check.setStyleSheet("""
            QCheckBox {
                spacing: 6px;
                color: #18191c;
                font-size: 14px;
            }
            
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            
            QCheckBox::indicator:unchecked {
                border: 2px solid #edf0f5;
                background-color: white;
                border-radius: 3px;
            }
            
            QCheckBox::indicator:checked {
                border: 2px solid #4361EE;
                background-color: #4361EE;
                border-radius: 3px;
            }
            
            QCheckBox::indicator:hover {
                border-color: #3A6EC3;
            }
        """)
        detection_layout.addRow("", keypoints_check)

        content_layout.addWidget(detection_group)

        # 界面设置组
        ui_group = SettingsGroup("界面设置")
        ui_layout = QFormLayout(ui_group)
        ui_layout.setSpacing(15)
        ui_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        ui_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        ui_layout.setContentsMargins(25, 30, 25, 30)
        
        # 设置字体大小
        ui_group.setFont(font)

        # 主题选择
        theme_combo = QComboBox()
        theme_combo.addItems(["浅色", "深色", "系统默认"])
        theme_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #edf0f5;
                border-radius: 6px;
                padding: 8px 12px;
                min-width: 220px;
                background-color: #ffffff;
                color: #18191c;
                font-size: 14px;
            }
            
            QComboBox:hover {
                border-color: #4361EE;
            }
            
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 28px;
                border-left-width: 0px;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
            }
            
            QComboBox QAbstractItemView {
                border: 1px solid #edf0f5;
                border-radius: 6px;
                padding: 6px;
                selection-background-color: #EBF5FF;
                selection-color: #4361EE;
                background-color: white;
                color: #18191c;
            }
        """)
        ui_layout.addRow(QLabel("<span style='font-size:14px; color:#18191c;'>主题:</span>"), theme_combo)

        # 语言选择
        language_combo = QComboBox()
        language_combo.addItems(["简体中文", "English", "日本语"])
        language_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #edf0f5;
                border-radius: 6px;
                padding: 8px 12px;
                min-width: 220px;
                background-color: #ffffff;
                color: #18191c;
                font-size: 14px;
            }
            
            QComboBox:hover {
                border-color: #4361EE;
            }
            
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 28px;
                border-left-width: 0px;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
            }
            
            QComboBox QAbstractItemView {
                border: 1px solid #edf0f5;
                border-radius: 6px;
                padding: 6px;
                selection-background-color: #EBF5FF;
                selection-color: #4361EE;
                background-color: white;
                color: #18191c;
            }
        """)
        ui_layout.addRow(QLabel("<span style='font-size:14px; color:#18191c;'>语言:</span>"), language_combo)
        
        content_layout.addWidget(ui_group)
        
        # 保存按钮区域
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 20, 0, 0)
        
        save_btn = QPushButton("保存设置")
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #4361EE;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 500;
                min-width: 100px;
            }
            
            QPushButton:hover {
                background-color: #3A56D4;
            }
            
            QPushButton:pressed {
                background-color: #2A46C4;
            }
        """)
        
        reset_btn = QPushButton("恢复默认")
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #f6f7f8;
                color: #18191c;
                border: 1px solid #edf0f5;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 400;
                min-width: 100px;
                margin-right: 12px;
            }
            
            QPushButton:hover {
                background-color: #edf0f5;
            }
            
            QPushButton:pressed {
                background-color: #e3e5e7;
            }
        """)
        
        button_layout.addWidget(reset_btn)
        button_layout.addWidget(save_btn)
        button_layout.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        content_layout.addLayout(button_layout)
        
        # 添加间距
        content_layout.addStretch()

        # 清空缓存按钮
        clear_cache_btn = QPushButton("清空缓存")
        clear_cache_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_cache_btn.setStyleSheet("""
            QPushButton {
                background-color: #4361EE;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 18px;
                font-size: 14px;
                font-weight: 500;
                min-width: 100px;
            }
            
            QPushButton:hover {
                background-color: #3A56D4;
            }
            
            QPushButton:pressed {
                background-color: #2A46C4;
            }
        """)
        content_layout.addWidget(clear_cache_btn)
        
        # 设置滚动区域的内容
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)

        # 连接信号
        self.camera_combo.currentIndexChanged.connect(self.on_camera_selection_changed)

    def update_camera_list(self, available_cameras):
        """更新摄像头列表"""
        current_selection_data = self.camera_combo.currentData()
        self.camera_combo.clear()

        # 添加 RTSP 选项
        self.camera_combo.addItem("RTSP 视频流", "rtsp")

        if available_cameras:
            for camera_id in available_cameras:
                self.camera_combo.addItem(f"摄像头 {camera_id}", camera_id)
        else:
            # 即使没有物理摄像头，也保留RTSP选项
            if self.camera_combo.count() == 1: # 只有RTSP选项
                 # 可以选择添加一个提示，或者什么都不做
                 pass

        # 尝试恢复之前的选择
        index_to_select = self.camera_combo.findData(current_selection_data)
        if index_to_select != -1:
            self.camera_combo.setCurrentIndex(index_to_select)
        else:
            # 默认选择第一个可用设备（可能是RTSP或摄像头0）
            if self.camera_combo.count() > 0:
                self.camera_combo.setCurrentIndex(0)
        
        # 手动触发一次检查，以确保UI状态正确
        self.on_camera_selection_changed(self.camera_combo.currentIndex())

    def on_camera_selection_changed(self, index):
        """处理摄像头下拉列表选择变化"""
        selected_data = self.camera_combo.itemData(index)
        is_rtsp = (selected_data == "rtsp")

        # 直接控制标签和输入框的可见性
        self.rtsp_label.setVisible(is_rtsp) # 保持标签隐藏，或根据需要显示
        self.rtsp_url_input.setVisible(is_rtsp)

        # 可选：根据是否为RTSP禁用/启用分辨率和帧率控件
        # self.resolution_combo.setEnabled(not is_rtsp)
        # self.fps_combo.setEnabled(not is_rtsp)

        # 注意：设置页面通常不直接启动摄像头，而是保存配置
        # 所以这里不直接调用 self.camera_manager.set_camera_id()
        # 需要在保存设置的逻辑中获取正确的值
