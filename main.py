#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
健身姿态估计软件主程序
"""

import sys
import os
import traceback
import time
import logging
import warnings
from PyQt6.QtWidgets import (QApplication, QMessageBox, QSplashScreen, 
                            QProgressBar, QLabel, QVBoxLayout, QWidget, QFrame)
from PyQt6.QtGui import QPixmap, QFont, QColor, QIcon
from PyQt6.QtCore import Qt, QTimer, QSize, pyqtSignal

# 导入日志系统
from src.utils.logger import setup_logging, filter_warnings

# 全局变量
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# 设置日志系统
logger = setup_logging(APP_DIR, logging.INFO)

class SplashScreen(QWidget):
    """自定义启动画面，包含进度条和状态信息"""
    
    # 完成信号
    finished = pyqtSignal()
    def __init__(self):
        super().__init__()
        
        # 设置无边框窗口
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # 设置固定大小
        self.setFixedSize(600, 400)
        
        # 主布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 创建内容框架
        self.container = QFrame()
        self.container.setObjectName("container")
        self.container.setStyleSheet("""
            #container {
                background-color: #ffffff;
                border-radius: 20px;
                border: 1px solid #e2e8f0;
            }
        """)
        
        # 内容布局
        content_layout = QVBoxLayout(self.container)
        content_layout.setContentsMargins(30, 30, 30, 30)
        content_layout.setSpacing(15)
        
        # 应用图标
        app_icon_label = QLabel()
        app_icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/icon.png")
        if os.path.exists(app_icon_path):
            app_icon = QPixmap(app_icon_path)
            # 设置窗口图标
            self.setWindowIcon(QIcon(app_icon_path))
        
        app_icon_label.setPixmap(app_icon.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio))
        app_icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(app_icon_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # 应用名称
        app_name = QLabel("姿态估计健身助手")
        app_name.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #111827;
        """)
        app_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(app_name)
        
        # 版本信息
        version_label = QLabel("Version 1.0.0")
        version_label.setStyleSheet("""
            font-size: 14px;
            color: #6b7280;
        """)
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(version_label)
        
        # 添加空间
        content_layout.addSpacing(30)
        
        # 状态信息
        self.status_label = QLabel("正在启动应用...")
        self.status_label.setStyleSheet("""
            font-size: 15px;
            color: #374151;
        """)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(self.status_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #e2e8f0;
                border-radius: 4px;
                border: none;
            }
            QProgressBar::chunk {
                background-color: #4361ee;
                border-radius: 4px;
            }
        """)
        content_layout.addWidget(self.progress_bar)
        
        # 详细进度信息
        self.detail_label = QLabel("初始化...")
        self.detail_label.setStyleSheet("""
            font-size: 13px;
            color: #6b7280;
        """)
        self.detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(self.detail_label)
        
        # 添加底部空间
        content_layout.addStretch()
        
        # 版权信息
        copyright_label = QLabel("© 2025 姿态估计健身助手")
        copyright_label.setStyleSheet("color: #9ca3af; font-size: 12px;")
        copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(copyright_label)
        
        # 将容器添加到主布局
        layout.addWidget(self.container)
        
        # 居中显示
        self.center()
        
        # 初始化加载项
        self.loading_steps = [
            ("初始化应用程序...", 10),
            ("检查系统环境...", 15),
            ("连接模型管理器...", 25),
            ("正在加载检测模型...", 40),
            ("正在加载姿态估计模型...", 60),
            ("初始化用户界面...", 80),
            ("准备完成...", 95),
            ("启动完成!", 100)
        ]
        
        self.current_step = 0
    
    def center(self):
        """将窗口居中显示"""
        screen_geometry = QApplication.primaryScreen().geometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)
    
    def update_progress(self, step=None, progress=None, detail=None):
        """更新进度"""
        if step is not None:
            self.current_step = step
            if step < len(self.loading_steps):
                message, value = self.loading_steps[step]
                self.status_label.setText(message)
                self.progress_bar.setValue(value)
        
        if progress is not None:
            self.progress_bar.setValue(progress)
            
        if detail is not None:
            self.detail_label.setText(detail)
            
        # 刷新界面
        QApplication.processEvents()
        
    def next_step(self, detail=None):
        """进入下一个加载步骤"""
        self.current_step += 1
        if self.current_step < len(self.loading_steps):
            message, value = self.loading_steps[self.current_step]
            self.status_label.setText(message)
            self.progress_bar.setValue(value)
            
            if detail:
                self.detail_label.setText(detail)
            
            # 刷新界面
            QApplication.processEvents()
        elif self.current_step == len(self.loading_steps):
            self.status_label.setText("启动完成!")
            self.progress_bar.setValue(100)
            self.detail_label.setText("正在打开主界面...")
            QApplication.processEvents()
            
            # 发送完成信号
            self.finished.emit()

# 全局异常处理
def exception_hook(exctype, value, tb):
    """捕获未处理的异常"""
    error_msg = ''.join(traceback.format_exception(exctype, value, tb))
    logger.error(f"未捕获的异常:\n{error_msg}")
    if QApplication.instance():
        QMessageBox.critical(None, "程序错误",
                           f"发生了未预期的错误:\n{str(value)}\n\n请联系开发人员处理此问题。")
    sys.__excepthook__(exctype, value, tb)


def main():
    """主函数"""
    # 设置全局异常钩子
    sys.excepthook = exception_hook
    
    try:
        # 创建应用
        app = QApplication(sys.argv)
        
        # 设置应用程序图标
        icon_path = os.path.join(APP_DIR, "resources", "icon.png")
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))
        
        # 记录启动信息
        logger.info("姿态估计健身助手 - 启动中")
        
        # 启用警告过滤器
        warning_filter = filter_warnings()
        warning_filter.__enter__()
        
        # 创建启动画面
        splash = SplashScreen()
        splash.show()
        app.processEvents()
        
        # 延迟导入以避免循环依赖
        splash.update_progress(0, None, "正在初始化系统...")
        logger.info("初始化系统")
        # time.sleep(0.3) # 移除模拟延迟
        
        splash.next_step("检查系统环境...") # 保留非模型加载的步骤
        logger.info("检查系统环境")
        # time.sleep(0.3) # 移除模拟延迟

        # 导入必要的模块 (可以提前导入，除非有特定原因)
        from src.ui.main_window import MainWindow
        from src.utils.model_manager import ModelManager

        splash.next_step("连接模型管理器...") # 保留
        logger.info("连接模型管理器")
        model_manager = ModelManager()
        # time.sleep(0.3) # 移除模拟延迟

        # 连接模型加载进度信号到启动画面
        # 注意：SplashScreen.update_progress 可能需要调整以接受 (int, str) 参数
        # 检查 update_progress 定义: def update_progress(self, step=None, progress=None, detail=None):
        # 需要一个适配器或修改 update_progress
        # 简单的适配器:
        def update_splash_progress(progress_val, detail_msg):
            splash.update_progress(progress=progress_val, detail=detail_msg)
        model_manager.load_progress.connect(update_splash_progress)

        # --- 异步加载处理 --- 
        # 创建主窗口实例 (可以在加载前创建，但不要显示)
        logger.info("初始化用户界面...")
        window = MainWindow() # 创建但不显示

        # 定义模型加载完成后的处理函数
        def on_model_loaded(success):
            logger.info(f"模型加载完成信号接收: success={success}")
            if success:
                logger.info("模型加载成功，准备显示主窗口")
                splash.update_progress(progress=100, detail="加载完成，正在启动...") # 更新最终状态
                # 使用 QTimer.singleShot 稍微延迟显示，确保启动画面更新
                QTimer.singleShot(100, lambda: show_main_window(window, splash))
            else:
                logger.error("模型加载失败，无法启动主程序")
                splash.update_progress(progress=100, detail="模型加载失败!") # 更新失败状态
                QMessageBox.critical(splash, "启动错误", "模型加载失败，应用程序无法启动。\n请检查模型文件或日志获取详细信息。")
                # 关闭启动画面和应用
                splash.close()
                app.quit()

        # 连接模型加载完成信号
        model_manager.model_loaded.connect(on_model_loaded)
        
        # 触发异步模型加载
        logger.info("开始异步加载模型...")
        load_initiated = model_manager.load_model()
        if not load_initiated:
            # 如果 load_model 返回 True (已加载) 或 False (正在加载)，这里应该处理
            # 如果返回 True (已加载), model_loaded 信号会被立即发出 (可能需要调整ModelManager确保这一点)
            # 如果返回 False (正在加载), 则等待信号
            # 如果返回 False (启动失败), model_loaded 信号也会发出 False
            # 此处的逻辑主要是处理 load_model 内部立即失败的情况（虽然我们期望错误通过信号传递）
            if not model_manager.is_loading(): # 添加一个 is_loading 状态判断（可选）
                 logger.error("未能启动模型加载过程。")
                 QMessageBox.critical(splash, "启动错误", "无法启动模型加载过程。")
                 splash.close()
                 app.quit()
                 return 1 # 退出 main

        # --- 移除旧的启动逻辑 ---
        # splash.next_step("正在加载检测模型...") # 由信号驱动
        # logger.info("加载模型")
        # model_load_success = model_manager.load_model() # 改为异步触发
        # if not model_load_success:
        #     logger.warning("模型加载失败，部分功能可能无法正常使用")
        # time.sleep(0.3) # 移除

        # splash.next_step("初始化用户界面...") # 提前
        # logger.info("初始化用户界面")

        # 创建主窗口 (提前)
        # window = MainWindow()

        # 连接完成信号 (移除，由 model_loaded 信号处理)
        # splash.finished.connect(lambda: show_main_window(window, splash))

        # 添加超时处理 (移除)
        # QTimer.singleShot(5000, lambda: show_main_window(window, splash))

        # 完成启动 (移除，由信号驱动)
        # splash.next_step("准备完成...")
        # time.sleep(0.3)
        # splash.next_step("启动完成!")

        # 主事件循环将由 app.exec() 启动，等待模型加载完成信号
        logger.info("启动主事件循环，等待模型加载完成...")
        return app.exec()

    except Exception as e:
        logger.error(f"启动失败: {str(e)}")
        if QApplication.instance():
            QMessageBox.critical(None, "启动错误", 
                               f"应用程序启动失败:\n{str(e)}\n\n请检查系统环境并重试。")
        return 1

def show_main_window(window, splash):
    """显示主窗口"""
    try:
        window.show()
        splash.close()
        logger.info("应用程序启动完成")
    except Exception as e:
        logger.error(f"显示主窗口失败: {str(e)}")
        if QApplication.instance():
            QMessageBox.critical(None, "错误", 
                               f"无法显示主窗口:\n{str(e)}\n\n请重启应用程序。")


if __name__ == "__main__":
    sys.exit(main())
