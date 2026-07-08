#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主页组件
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFrame, QGridLayout, QSpacerItem,
                             QSizePolicy)
from PyQt6.QtGui import QPixmap, QIcon, QFont
from PyQt6.QtCore import Qt, QSize


class FeatureCard(QFrame):
    """功能卡片"""

    def __init__(self, title, description, icon_path=None, parent=None):
        super().__init__(parent)
        self.title = title
        self.description = description
        self.icon_path = icon_path
        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 10px;
                border: 1px solid #edf0f5;
            }
            
            QFrame:hover {
                background-color: #f6f7f8;
                border-color: #4361EE;
            }
            
            QLabel {
                background-color: transparent;
                border: none;
            }
            
            QLabel:hover {
                background-color: transparent;
            }
        """)
        self.setMinimumHeight(150)

        # 主布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # 标题
        title_label = QLabel(self.title)
        title_label.setStyleSheet("""
            QLabel {
                color: #18191c;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        layout.addWidget(title_label)

        # 描述
        desc_label = QLabel(self.description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("""
            QLabel {
                color: #61666d;
                font-size: 14px;
                line-height: 1.5;
            }
        """)
        layout.addWidget(desc_label)

        # 设置弹性空间
        layout.addStretch()


class HomePage(QWidget):
    """主页"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        # 主布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(24)

        # 欢迎区域
        welcome_frame = QFrame()
        welcome_frame.setStyleSheet("""
            QFrame {
                background-color: #f6f7f8;
                border-radius: 10px;
                border: none;
            }
        """)
        welcome_layout = QVBoxLayout(welcome_frame)
        welcome_layout.setContentsMargins(32, 32, 32, 32)
        welcome_layout.setSpacing(15)

        # 欢迎标题
        welcome_title = QLabel("欢迎使用健身姿态辅助系统")
        welcome_title.setStyleSheet("""
            QLabel {
                color: #18191c;
                font-size: 24px;
                font-weight: bold;
            }
        """)
        welcome_layout.addWidget(welcome_title)

        # 欢迎描述
        welcome_desc = QLabel(
            "这是一个基于人工智能的健身姿态辅助系统，可以帮助您实时检测和分析健身姿势，提高训练效果，减少运动伤害。")
        welcome_desc.setWordWrap(True)
        welcome_desc.setStyleSheet("""
            QLabel {
                color: #61666d;
                font-size: 15px;
                line-height: 1.5;
            }
        """)
        welcome_layout.addWidget(welcome_desc)

        # 开始按钮
        start_btn = QPushButton("立即开始")
        start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4361EE;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 15px;
                font-weight: 500;
                margin-top: 10px;
                width: 120px;
            }
            
            QPushButton:hover {
                background-color: #3A56D4;
            }
            
            QPushButton:pressed {
                background-color: #2E4BBD;
            }
        """)
        start_btn.setFixedWidth(120)
        welcome_layout.addWidget(start_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(welcome_frame)

        # 功能区域标题
        features_title = QLabel("核心功能")
        features_title.setStyleSheet("""
            QLabel {
                color: #18191c;
                font-size: 20px;
                font-weight: bold;
                margin-top: 10px;
                padding-left: 5px;
            }
        """)
        layout.addWidget(features_title)

        # 功能卡片网格
        grid_layout = QGridLayout()
        grid_layout.setSpacing(20)

        # 添加功能卡片
        card1 = FeatureCard(
            "实时姿态检测",
            "使用摄像头实时检测您的健身姿势，提供即时反馈"
        )

        card2 = FeatureCard(
            "训练计划",
            "根据您的健身水平和目标，提供个性化的训练计划"
        )

        card3 = FeatureCard(
            "姿态分析",
            "详细分析您的健身姿势，找出需要改进的地方"
        )

        card4 = FeatureCard(
            "进度追踪",
            "记录您的健身进度，帮助您持续改进"
        )

        # 添加到网格
        grid_layout.addWidget(card1, 0, 0)
        grid_layout.addWidget(card2, 0, 1)
        grid_layout.addWidget(card3, 1, 0)
        grid_layout.addWidget(card4, 1, 1)

        layout.addLayout(grid_layout)

        # 添加占位符
        layout.addStretch()
