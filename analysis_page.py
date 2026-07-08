#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
姿态分析页面
"""

import os
import cv2
import numpy as np
import threading
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QPushButton, QFrame, QTabWidget, QFileDialog,
                           QGridLayout, QScrollArea, QProgressBar, QSplitter,
                           QMessageBox, QApplication, QSizePolicy)
from PyQt6.QtGui import QPixmap, QIcon, QFont, QImage, QPalette, QColor
from PyQt6.QtCore import Qt, QSize, pyqtSignal, pyqtSlot, QThread, QTimer

from .pose_3d_viewer import Pose3DViewer
from ...utils.mmpose_detector import MMPoseDetector
from ...utils.model_manager import ModelManager


class VideoProcessThread(QThread):
    """视频处理线程"""
    
    # 自定义信号
    frame_processed = pyqtSignal(QImage)           # 处理后的帧
    pose_detected = pyqtSignal(np.ndarray)         # 检测到的姿态
    progress_updated = pyqtSignal(float)           # 进度更新
    processing_finished = pyqtSignal()             # 处理完成
    processing_error = pyqtSignal(str)             # 处理错误
    
    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.is_running = False
        
        # 使用全局模型实例
        self.model_manager = ModelManager()
        self.mmpose_detector = self.model_manager.get_detector()
        # 设置为视频模式
        self.mmpose_detector.is_webcam = False
        
    def run(self):
        """线程执行函数"""
        self.is_running = True
        
        try:
            # 确保模型已加载
            if not self.model_manager.is_loaded():
                self.model_manager.load_model()
                
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.processing_error.emit("无法打开视频文件")
                return
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = 0
            
            while self.is_running and current_frame < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 处理帧
                result_frame, keypoints = self.mmpose_detector.process_image(frame)
                
                if result_frame is not None:
                    # 转换为QImage
                    height, width, channel = result_frame.shape
                    bytes_per_line = 3 * width
                    q_img = QImage(result_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
                    
                    # 发送信号
                    self.frame_processed.emit(q_img)
                
                # 发送关键点
                if keypoints is not None:
                    self.pose_detected.emit(keypoints)
                
                # 更新进度
                current_frame += 1
                progress = current_frame / total_frames * 100
                self.progress_updated.emit(progress)
            
            cap.release()
            
            # 发送处理完成信号
            self.processing_finished.emit()
            
        except Exception as e:
            # 发送错误信号
            self.processing_error.emit(str(e))
        
        finally:
            self.is_running = False
    
    def stop(self):
        """停止线程"""
        self.is_running = False
        self.wait()


class AnalysisResultWidget(QFrame):
    """分析结果组件"""
    
    def __init__(self, title, score, recommendations, parent=None):
        super().__init__(parent)
        self.title = title
        self.score = score
        self.recommendations = recommendations
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 14px;
                border: 1px solid #edf0f5;
                padding: 20px;
            }
        """)
        
        # 主布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(14)
        
        # 标题
        title_label = QLabel(self.title)
        title_label.setStyleSheet("""
            QLabel {
                color: #212529;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        layout.addWidget(title_label)
        
        # 分数
        score_layout = QHBoxLayout()
        score_label = QLabel("姿态评分:")
        score_label.setStyleSheet("font-weight: 600; color: #495057;")
        score_layout.addWidget(score_label)
        
        progress = QProgressBar()
        progress.setValue(self.score)
        progress.setMinimumWidth(180)
        progress.setMaximumWidth(250)
        
        # 根据分数设置颜色
        if self.score < 50:
            color = "#f64e60"  # 红色
        elif self.score < 80:
            color = "#ffb822"  # 黄色
        else:
            color = "#0bb783"  # 绿色
            
        progress.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 7px;
                text-align: center;
                background-color: #f8f9fa;
                height: 14px;
            }}
            
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 7px;
            }}
        """)
        
        score_layout.addWidget(progress)
        score_value = QLabel(f"{self.score}%")
        score_value.setStyleSheet(f"color: {color}; font-weight: bold;")
        score_layout.addWidget(score_value)
        score_layout.addStretch()
        
        layout.addLayout(score_layout)
        
        # 建议
        rec_label = QLabel("改进建议:")
        rec_label.setStyleSheet("font-weight: 600; color: #495057; margin-top: 12px;")
        layout.addWidget(rec_label)
        
        for rec in self.recommendations:
            item = QLabel(f"• {rec}")
            item.setWordWrap(True)
            item.setStyleSheet("color: #6c757d; font-size: 14px; margin: 3px 0;")
            layout.addWidget(item)
            
        layout.addStretch()


class VideoWidget(QLabel):
    """视频显示组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)  # 增大最小尺寸
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #ffffff;
                border-radius: 14px;
                border: 1px solid #edf0f5;
            }
        """)
        
        # 设置默认显示的文本
        self.setText("请选择视频或图片")
        self.setFont(QFont("Arial", 15))
    
    def display_frame(self, qimage):
        """显示帧"""
        if qimage:
            # 缩放图像以适应组件
            scaled_image = qimage.scaled(self.size(), 
                                        Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
            self.setPixmap(QPixmap.fromImage(scaled_image))


class AnalysisPage(QWidget):
    """姿态分析页面"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("analysisPage")
        
        # 初始化成员变量
        self.pose_3d_viewer = None  # 3D查看器
        
        # 使用全局模型实例
        self.model_manager = ModelManager()
        self.mmpose_detector = self.model_manager.get_detector()
        self.mmpose_detector.is_webcam = False  # 不是实时摄像头模式
        self.model_loaded = self.model_manager.is_loaded()
        
        # 检测线程
        self.video_thread = None
        self.current_file_path = None
        
        # 初始化UI
        self.init_ui()
        
        # 连接模型加载信号
        self.model_manager.model_loaded.connect(self.on_model_loaded)
        
    @pyqtSlot(bool)
    def on_model_loaded(self, success):
        """模型加载完成的回调"""
        self.model_loaded = success
        
    def init_pose_detector(self):
        """初始化姿态检测器"""
        try:
            if not self.model_loaded:
                return self.model_manager.load_model()
            return True
        except Exception as e:
            print(f"初始化姿态检测器时出错: {str(e)}")
            return False
    
    def init_ui(self):
        """初始化UI"""
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: #f8f9fa;
                border: none;
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
        content_widget.setObjectName("contentWidget")
        content_widget.setStyleSheet("""
            #contentWidget {
                background-color: #f8f9fa;
            }
        """)
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(24, 24, 24, 24)
        content_layout.setSpacing(24)
        
        # 页面标题
        title_label = QLabel("姿态分析")
        title_label.setStyleSheet("""
            QLabel {
                color: #212529;
                font-size: 28px;
                font-weight: bold;
                margin-bottom: 4px;
            }
        """)
        content_layout.addWidget(title_label)
        
        # 上传区域
        upload_frame = QFrame()
        upload_frame.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 16px;
                border: 1px solid #edf0f5;
            }
        """)
        upload_layout = QVBoxLayout(upload_frame)
        upload_layout.setContentsMargins(40, 50, 40, 50)
        upload_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 增加上传图标
        upload_icon_label = QLabel()
        upload_icon = QPixmap("icons/upload.png") # 假设有这个图标，如果没有可以去掉这部分
        if not upload_icon.isNull():
            upload_icon_label.setPixmap(upload_icon.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            upload_icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            upload_layout.addWidget(upload_icon_label, alignment=Qt.AlignmentFlag.AlignCenter)
            
        upload_label = QLabel("上传视频或图片进行姿态分析")
        upload_label.setStyleSheet("""
            QLabel {
                color: #495057;
                font-size: 20px;
                font-weight: 500;
                margin-top: 12px;
                border: none;
            }
        """)
        upload_layout.addWidget(upload_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        upload_sublabel = QLabel("支持MP4、AVI、MOV视频或JPG、PNG图片")
        upload_sublabel.setStyleSheet("""
            QLabel {
                color: #6c757d;
                font-size: 14px;
                margin-bottom: 20px;
                border: none;
            }
        """)
        upload_layout.addWidget(upload_sublabel, alignment=Qt.AlignmentFlag.AlignCenter)
        
        upload_btn = QPushButton("选择文件")
        upload_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #4361EE;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 14px 28px;
                font-size: 16px;
                font-weight: 500;
                margin-top: 18px;
            }
            
            QPushButton:hover {
                background-color: #3A56D4;
            }
            
            QPushButton:pressed {
                background-color: #2E4BBD;
            }
        """)
        upload_layout.addWidget(upload_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        
        content_layout.addWidget(upload_frame)
        
        # 分析内容区域
        self.analysis_container = QFrame()
        self.analysis_container.setObjectName("analysisContainer")
        self.analysis_container.setStyleSheet("""
            #analysisContainer {
                background-color: #ffffff;
                border-radius: 16px;
                border: 1px solid #edf0f5;
            }
        """)
        analysis_layout = QVBoxLayout(self.analysis_container)
        analysis_layout.setContentsMargins(24, 24, 24, 24)
        analysis_layout.setSpacing(24)
        
        # 内容标题
        content_title = QLabel("分析结果")
        content_title.setStyleSheet("""
            QLabel {
                color: #212529;
                font-size: 22px;
                font-weight: bold;
            }
        """)
        analysis_layout.addWidget(content_title)
        
        # 创建分割器，左侧显示视频，右侧显示3D模型和分析结果
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background: #edf0f5;
            }
        """)
        
        # 左侧视频区域
        left_frame = QFrame()
        left_frame.setObjectName("leftFrame")
        left_frame.setStyleSheet("""
            #leftFrame {
                background-color: #ffffff;
                border-radius: 14px;
                border: 1px solid #edf0f5;
            }
        """)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(16)
        
        # 视频标题
        video_title = QLabel("视频预览")
        video_title.setStyleSheet("""
            QLabel {
                color: #111827;
                font-size: 18px;
                font-weight: 600;
            }
        """)
        left_layout.addWidget(video_title)
        
        # 视频显示区域
        self.video_widget = VideoWidget()
        left_layout.addWidget(self.video_widget, 1)
        
        # 视频控制区域
        video_controls = QHBoxLayout()
        video_controls.setSpacing(12)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 7px;
                text-align: center;
                background-color: #f8f9fa;
                height: 14px;
                font-size: 12px;
            }
            
            QProgressBar::chunk {
                background-color: #4361EE;
                border-radius: 7px;
            }
        """)
        video_controls.addWidget(self.progress_bar)
        
        left_layout.addLayout(video_controls)
        
        # 右侧3D模型和分析结果区域
        right_frame = QFrame()
        right_frame.setObjectName("rightFrame")
        right_frame.setStyleSheet("""
            #rightFrame {
                background-color: #ffffff;
                border-radius: 14px;
                border: 1px solid #edf0f5;
            }
        """)
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(16)
        
        # 3D模型标题
        model_title = QLabel("3D姿态模型")
        model_title.setStyleSheet("""
            QLabel {
                color: #111827;
                font-size: 18px;
                font-weight: 600;
            }
        """)
        right_layout.addWidget(model_title)
        
        # 3D模型显示区域
        self.pose_3d_viewer = Pose3DViewer()
        self.pose_3d_viewer.setMinimumHeight(360) # 增大高度以更好地展示
        self.pose_3d_viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout.addWidget(self.pose_3d_viewer, 1)
        
        # 3D模型控制按钮行
        model_controls = QHBoxLayout()
        model_controls.setSpacing(12)
        
        # 自动旋转按钮
        self.auto_rotate_btn = QPushButton("自动旋转")
        self.auto_rotate_btn.setCheckable(True)
        self.auto_rotate_btn.setChecked(True)
        self.auto_rotate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_rotate_btn.setStyleSheet("""
            QPushButton {
                background-color: #f97316;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 12px 18px;
                font-size: 14px;
                font-weight: 500;
            }
            
            QPushButton:hover {
                background-color: #ea580c;
            }
            
            QPushButton:checked {
                background-color: #ea580c;
            }
        """)
        self.auto_rotate_btn.clicked.connect(
            lambda checked: self.pose_3d_viewer.toggle_auto_rotate(checked))
        model_controls.addWidget(self.auto_rotate_btn)
        
        # 重置视图按钮
        reset_view_btn = QPushButton("重置视图")
        reset_view_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reset_view_btn.setStyleSheet("""
            QPushButton {
                background-color: #64748b;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 12px 18px;
                font-size: 14px;
                font-weight: 500;
            }
            
            QPushButton:hover {
                background-color: #475569;
            }
        """)
        reset_view_btn.clicked.connect(self.pose_3d_viewer.reset_view)
        model_controls.addWidget(reset_view_btn)
        
        right_layout.addLayout(model_controls)
        
        # 添加组件到分割器
        splitter.addWidget(left_frame)
        splitter.addWidget(right_frame)
        
        # 设置初始大小比例
        splitter.setSizes([500, 500])
        
        analysis_layout.addWidget(splitter)
        
        # 创建标签页
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                border-radius: 10px;
                background-color: white;
            }
            
            QTabBar::tab {
                background-color: #f8f9fa;
                border: none;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                padding: 12px 24px;
                margin-right: 4px;
                color: #495057;
                font-size: 15px;
            }
            
            QTabBar::tab:selected {
                background-color: white;
                color: #4361EE;
                font-weight: 500;
                border-bottom: 2px solid #4361EE;
            }
            
            QTabBar::tab:hover:!selected {
                background-color: #e9ecef;
            }
        """)
        
        # 总体分析标签页
        overall_tab = QWidget()
        overall_layout = QVBoxLayout(overall_tab)
        overall_layout.setContentsMargins(0, 20, 0, 0)
        overall_layout.setSpacing(16)
        
        overall_result = AnalysisResultWidget(
            "总体姿态评估",
            78,
            [
                "整体姿势良好，但注意保持脊柱中立位置",
                "下肢对称性较好，但右侧膝盖略有内扣",
                "加强核心肌群的稳定性，以改善整体姿态",
            ]
        )
        
        overall_layout.addWidget(overall_result)
        overall_layout.addStretch()
        
        # 详细分析标签页
        detail_tab = QWidget()
        detail_layout = QVBoxLayout(detail_tab)
        detail_layout.setContentsMargins(0, 20, 0, 0)
        detail_layout.setSpacing(16)
        
        # 添加身体部位的详细分析
        body_parts_analysis = [
            {"title": "上肢姿态", "score": 85, "recs": ["肩膀位置良好", "注意避免肘部过度伸展"]},
            {"title": "下肢姿态", "score": 70, "recs": ["膝盖应与脚尖方向一致", "加强髋部稳定性"]},
            {"title": "躯干姿态", "score": 80, "recs": ["保持腰部中立位置", "避免脊柱过度弯曲"]}
        ]
        
        for analysis in body_parts_analysis:
            detail_result = AnalysisResultWidget(
                analysis["title"],
                analysis["score"],
                analysis["recs"]
            )
            detail_layout.addWidget(detail_result)
            
        detail_layout.addStretch()
        
        # 添加标签页
        tab_widget.addTab(overall_tab, "总体分析")
        tab_widget.addTab(detail_tab, "详细分析")
        
        analysis_layout.addWidget(tab_widget)
        
        content_layout.addWidget(self.analysis_container)
        content_layout.addStretch()
        
        # 设置滚动区域的内容
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        
        # 当没有上传文件时，隐藏分析结果部分
        self.analysis_container.setVisible(False)
        
        # 连接上传按钮信号
        upload_btn.clicked.connect(self.select_file)
    
    def select_file(self):
        """选择文件进行分析"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, 
            "选择视频或图片", 
            "", 
            "媒体文件 (*.mp4 *.avi *.mov *.jpg *.jpeg *.png)"
        )
        
        if file_path:
            self.current_file_path = file_path
            self.analyze_file(file_path)
    
    def analyze_file(self, file_path):
        """分析文件
        
        Args:
            file_path: 文件路径
        """
        # 确保模型已加载
        if not self.model_loaded:
            # 显示加载提示
            loading_dialog = QMessageBox(self)
            loading_dialog.setWindowTitle("加载模型")
            loading_dialog.setText("正在加载姿态检测模型，请稍候...")
            loading_dialog.setStandardButtons(QMessageBox.StandardButton.NoButton)
            loading_dialog.setIcon(QMessageBox.Icon.Information)
            
            # 显示加载对话框（非模态）
            loading_dialog.show()
            QApplication.processEvents()  # 确保UI更新
            
            try:
                # 加载模型
                success = self.model_manager.load_model()
                self.model_loaded = success
                
                # 关闭加载对话框
                loading_dialog.close()
                
                if not success:
                    QMessageBox.critical(self, "错误", "模型加载失败，请检查模型文件")
                    return
            except Exception as e:
                loading_dialog.close()
                QMessageBox.critical(self, "错误", f"加载模型时出错: {str(e)}")
                return
        
        # 判断文件类型
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # 显示分析结果区域
        self.analysis_container.setVisible(True)
        
        # 重置进度条
        self.progress_bar.setValue(0)
        
        # 改变进度条颜色为蓝色表示正在处理
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 7px;
                text-align: center;
                background-color: #f8f9fa;
                height: 14px;
                font-size: 12px;
            }
            
            QProgressBar::chunk {
                background-color: #4361EE;
                border-radius: 7px;
            }
        """)
        
        # 处理视频
        if file_ext in ['.mp4', '.avi', '.mov']:
            # 创建视频处理线程
            self.video_thread = VideoProcessThread(file_path)
            
            # 连接信号
            self.video_thread.frame_processed.connect(self.video_widget.display_frame)
            self.video_thread.pose_detected.connect(self.pose_3d_viewer.update_keypoints)
            self.video_thread.progress_updated.connect(self.progress_bar.setValue)
            self.video_thread.processing_finished.connect(self.on_processing_finished)
            self.video_thread.processing_error.connect(self.on_processing_error)
            
            # 启动线程
            self.video_thread.start()
            
        # 处理图片
        elif file_ext in ['.jpg', '.jpeg', '.png']:
            try:
                # 检查文件是否存在
                if not os.path.exists(file_path):
                    self.on_processing_error(f"文件不存在: {file_path}")
                    return
                
                # 检查文件是否可读
                if not os.access(file_path, os.R_OK):
                    self.on_processing_error(f"无法读取文件: {file_path}")
                    return
                
                # 读取图片并转换为cv2格式
                image = cv2.imread(file_path)
                if image is None:
                    self.on_processing_error(f"无法处理图片: {file_path}")
                    return
                
                # 使用MMPose检测器处理图片
                result_frame, keypoints = self.mmpose_detector.process_image(image)
                
                if result_frame is not None:
                    # 转换为QImage
                    height, width, channel = result_frame.shape
                    bytes_per_line = 3 * width
                    q_img = QImage(result_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
                    
                    # 显示图片
                    self.video_widget.display_frame(q_img)
                    
                    # 更新3D模型
                    if keypoints is not None:
                        self.pose_3d_viewer.update_keypoints(keypoints)
                    
                    # 设置进度条为100%
                    self.progress_bar.setValue(100)
                    
                    # 处理完成回调
                    self.on_processing_finished()
                else:
                    self.on_processing_error("无法处理图片")
            except Exception as e:
                self.on_processing_error(f"处理图片时出错: {str(e)}")
        else:
            self.on_processing_error("不支持的文件格式")
    
    def on_processing_finished(self):
        """处理完成回调"""
        # 重置进度条样式
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 7px;
                text-align: center;
                background-color: #f8f9fa;
                height: 14px;
                font-size: 12px;
            }
            
            QProgressBar::chunk {
                background-color: #28a745;
                border-radius: 7px;
            }
        """)
        
        # 模拟分析结果 (实际项目中应该基于关键点数据计算)
        # 这里仅作演示
    
    def on_processing_error(self, error_msg):
        """处理错误回调
        
        Args:
            error_msg: 错误信息
        """
        QMessageBox.critical(self, "处理错误", error_msg)
        
        # 设置进度条样式
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 7px;
                text-align: center;
                background-color: #f8f9fa;
                height: 14px;
                font-size: 12px;
            }
            
            QProgressBar::chunk {
                background-color: #dc3545;
                border-radius: 7px;
            }
        """)
    
    def show_demo_results(self, result_label, tab_widget):
        """显示演示结果（实际项目中应该由真实分析替代）"""
        result_label.setVisible(True)
        tab_widget.setVisible(True)
        
    def closeEvent(self, event):
        """关闭事件，确保线程被正确停止"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        
        if hasattr(self, 'load_timer') and self.load_timer.isActive():
            self.load_timer.stop()
        
        if hasattr(self, 'mmpose_detector'):
            self.mmpose_detector.stop()
        
        event.accept()
