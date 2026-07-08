#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
健身姿态估计软件主窗口
"""

import os
import logging
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QStackedWidget, QFileDialog, QMessageBox)
from PyQt6.QtGui import QIcon, QPixmap, QImage
from PyQt6.QtCore import Qt, QSize, QTimer
import numpy as np

from .widgets.sidebar import Sidebar
from .widgets.home_page import HomePage
from .widgets.workout_page import WorkoutPage
from .widgets.analysis_page import AnalysisPage
from .widgets.mmpose_page import MMPosePage
from .widgets.settings_page import SettingsPage

# 获取logger
logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """主窗口类"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        # 设置窗口属性
        self.setWindowTitle("姿态健身助手")
        self.setMinimumSize(1200, 850)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f2f5;
            }
            QWidget {
                font-family: 'Microsoft YaHei UI', 'PingFang SC', sans-serif;
                color: #18191c;
            }
            QPushButton {
                border-radius: 4px;
            }
            QLabel {
                color: #18191c;
            }
            QScrollBar:vertical {
                border: none;
                background: #f1f3f5;
                width: 8px;
                border-radius: 4px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #c9ccd0;
                min-height: 30px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #aaaaaa;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)

        # 创建中心部件
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 侧边栏
        self.sidebar = Sidebar(self)
        main_layout.addWidget(self.sidebar)

        # 内容区域
        self.content_widget = QWidget()
        self.content_widget.setObjectName("contentWidget")
        self.content_widget.setStyleSheet("""
            QWidget#contentWidget {
                background-color: #ffffff;
                border-radius: 10px;
                margin: 12px 12px 12px 0px;
            }
        """)
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(24, 24, 24, 24)
        self.content_layout.setSpacing(20)
        main_layout.addWidget(self.content_widget)

        # 设置内容区域占比
        main_layout.setStretch(0, 1)  # 侧边栏
        main_layout.setStretch(1, 5)  # 内容区域

        # 创建页面堆栈
        self.stack = QStackedWidget()
        self.stack.setStyleSheet("""
            QStackedWidget {
                border: none;
                background-color: transparent;
            }
        """)
        self.content_layout.addWidget(self.stack)

        # 添加各个页面
        self.home_page = HomePage()
        self.workout_page = None  # 懒加载
        self.analysis_page = AnalysisPage()
        self.mmpose_page = None   # 懒加载
        self.settings_page = SettingsPage()

        self.stack.addWidget(self.home_page)
        self.stack.addWidget(QWidget())  # 占位符 - 后续由 workout_page 替换
        self.stack.addWidget(self.analysis_page)
        self.stack.addWidget(QWidget())  # 占位符 - 后续由 mmpose_page 替换
        self.stack.addWidget(self.settings_page)

        # 连接信号
        self.sidebar.page_changed.connect(self.change_page)

    def closeEvent(self, event):
        """处理窗口关闭事件，确保线程被正确停止"""
        print("MainWindow: 收到关闭事件，开始清理...")
        
        # 清理 WorkoutPage 资源 (如果已加载)
        if self.workout_page is not None:
            print("MainWindow: 正在清理 WorkoutPage...")
            try:
                self.workout_page.cleanup() # 调用 WorkoutPage 的清理方法
                print("MainWindow: WorkoutPage 清理完成")
            except Exception as e:
                print(f"MainWindow: 清理 WorkoutPage 时出错: {e}")

        # 清理 MMPosePage 资源 (如果已加载且需要清理)
        # (如果 MMPosePage 也有后台线程，也需要类似处理)
        # if self.mmpose_page is not None and hasattr(self.mmpose_page, 'cleanup'):
        #     print("MainWindow: 正在清理 MMPosePage...")
        #     try:
        #         self.mmpose_page.cleanup()
        #         print("MainWindow: MMPosePage 清理完成")
        #     except Exception as e:
        #         print(f"MainWindow: 清理 MMPosePage 时出错: {e}")
        
        print("MainWindow: 所有清理完成，接受关闭事件")
        event.accept() # 接受关闭事件，允许窗口关闭
        # 如果需要强制退出，可以使用 sys.exit(0)
        # import sys
        # sys.exit(0)

    def change_page(self, index):
        """切换页面"""
        try:
            # 延迟加载页面
            if index == 1 and self.workout_page is None:
                # 显示加载中提示
                loading_widget = QWidget()
                loading_layout = QVBoxLayout(loading_widget)
                loading_label = QLabel("正在加载训练页面，请稍候...")
                loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                loading_label.setStyleSheet("font-size: 16px; color: #61666d;")
                loading_layout.addWidget(loading_label)
                
                # 替换占位符
                old_widget = self.stack.widget(1)
                self.stack.removeWidget(old_widget)
                self.stack.insertWidget(1, loading_widget)
                self.stack.setCurrentIndex(1)
                
                # 使用延迟加载避免界面卡顿
                QTimer.singleShot(100, self.load_workout_page)
                return
                
            elif index == 3 and self.mmpose_page is None:
                # 显示加载中提示
                loading_widget = QWidget()
                loading_layout = QVBoxLayout(loading_widget)
                loading_label = QLabel("正在加载姿态检测页面，请稍候...")
                loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                loading_label.setStyleSheet("font-size: 16px; color: #61666d;")
                loading_layout.addWidget(loading_label)
                
                # 替换占位符
                old_widget = self.stack.widget(3)
                self.stack.removeWidget(old_widget)
                self.stack.insertWidget(3, loading_widget)
                self.stack.setCurrentIndex(3)
                
                # 使用延迟加载避免界面卡顿
                QTimer.singleShot(100, self.load_mmpose_page)
                return
            
            # 直接切换已加载的页面
            self.stack.setCurrentIndex(index)
            
            # 如果切换到的是MMPose页面，重置其视图
            if index == 3 and self.mmpose_page is not None:
                # 检查video_widget是否存在并清除 (使用display_frame)
                if hasattr(self.mmpose_page, 'video_widget') and self.mmpose_page.video_widget is not None:
                    self.mmpose_page.video_widget.display_frame(QImage()) # 传递空QImage
                # 检查pose_3d_viewer是否存在并重置
                if hasattr(self.mmpose_page, 'pose_3d_viewer') and self.mmpose_page.pose_3d_viewer is not None:
                    self.mmpose_page.pose_3d_viewer.update_keypoints(np.zeros((17, 3), dtype=np.float32))
                    self.mmpose_page.pose_3d_viewer.reset_view()
            
        except Exception as e:
            logger.error(f"切换页面失败: {str(e)}")
            QMessageBox.warning(self, "错误", f"切换页面时发生错误:\n{str(e)}")

    def load_workout_page(self):
        """加载训练页面"""
        try:
            if self.workout_page is None:
                # 创建页面
                self.workout_page = WorkoutPage()
                # 替换占位符
                old_widget = self.stack.widget(1)
                self.stack.removeWidget(old_widget)
                self.stack.insertWidget(1, self.workout_page)
                self.stack.setCurrentIndex(1)
        except Exception as e:
            logger.error(f"加载训练页面失败: {str(e)}")
            QMessageBox.warning(self, "错误", f"加载训练页面失败:\n{str(e)}")

    def load_mmpose_page(self):
        """加载MMPose姿态检测页面"""
        try:
            if self.mmpose_page is None:
                # 创建页面
                self.mmpose_page = MMPosePage()
                # 替换占位符
                old_widget = self.stack.widget(3)
                self.stack.removeWidget(old_widget)
                self.stack.insertWidget(3, self.mmpose_page)
                self.stack.setCurrentIndex(3)
        except Exception as e:
            logger.error(f"加载姿态检测页面失败: {str(e)}")
            QMessageBox.warning(self, "错误", f"加载姿态检测页面失败:\n{str(e)}")
