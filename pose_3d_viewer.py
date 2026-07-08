#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
3D姿态查看器 - 将检测到的关键点显示为3D模型
"""

import numpy as np
import os
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtCore import Qt, QTimer
from OpenGL.GL import *
from OpenGL.GLU import *
import math
from PyQt6.QtCore import QUrl, QFileInfo
import trimesh
from PIL import Image, ImageDraw


class Human3DModel:
    """人体3D模型类，存储骨架连接和颜色信息"""

    def __init__(self):
        # 定义人体骨骼连接关系 (按照COCO标准，并添加额外连接)
        self.connections = [
            # 面部
            (0, 1),   # 鼻子到左眼
            (0, 2),   # 鼻子到右眼
            (1, 2),   # 左眼到右眼 - 新增
            (1, 3),   # 左眼到左耳
            (2, 4),   # 右眼到右耳
            (3, 5),   # 左耳到左肩 - 新增
            (4, 6),   # 右耳到右肩 - 新增
            
            # 躯干
            (5, 6),   # 左肩到右肩
            (5, 11),  # 左肩到左髋
            (6, 12),  # 右肩到右髋
            (11, 12), # 左髋到右髋
            
            # 左臂
            (5, 7),   # 左肩到左肘
            (7, 9),   # 左肘到左手腕
            
            # 右臂
            (6, 8),   # 右肩到右肘
            (8, 10),  # 右肘到右手腕
            
            # 左腿
            (11, 13),  # 左髋到左膝
            (13, 15),  # 左膝到左踝
            
            # 右腿
            (12, 14),  # 右髋到右膝
            (14, 16)   # 右膝到右踝
        ]

        # 定义身体部位分组（按COCO标准分组并加入新连接）
        self.body_groups = {
            'face': [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4)],  # 面部连接，增加左眼到右眼
            'face_shoulder': [(3, 5), (4, 6)],                  # 耳朵到肩膀的连接
            'torso': [(5, 6), (5, 11), (6, 12), (11, 12)],     # 躯干连接
            'left_arm': [(5, 7), (7, 9)],                       # 左臂连接
            'right_arm': [(6, 8), (8, 10)],                     # 右臂连接
            'left_leg': [(11, 13), (13, 15)],                   # 左腿连接
            'right_leg': [(12, 14), (14, 16)]                   # 右腿连接
        }

        # 优化配色方案 - 更柔和、现代的色彩
        self.primary_color = (0.0, 0.0, 0.0)     # 黑色作为主色调
        self.secondary_color = (0.3, 0.3, 0.3)   # 深灰色作为次色调
        self.accent_color = (0.1, 0.6, 0.9)      # 蓝色作为强调色

        # 关节点分组
        self.joint_groups = {
            'face': [0, 1, 2, 3, 4],           # 面部关节
            'torso': [5, 6, 11, 12],           # 躯干关节
            'left_arm': [7, 9],                # 左臂关节
            'right_arm': [8, 10],              # 右臂关节
            'left_leg': [13, 15],              # 左腿关节
            'right_leg': [14, 16]              # 右腿关节
        }

        # 关节点颜色 - 更柔和的配色
        self.joint_colors = {
            'face': (0.2, 0.7, 1.0, 1.0),      # 天蓝色 - 面部
            'torso': (0.3, 0.3, 0.3, 1.0),     # 深灰色 - 躯干
            'left_arm': (1.0, 0.4, 0.4, 1.0),  # 淡红色 - 左臂
            'right_arm': (1.0, 0.4, 0.4, 1.0), # 淡红色 - 右臂
            'left_leg': (0.3, 0.9, 0.3, 1.0),  # 浅绿色 - 左腿
            'right_leg': (0.3, 0.9, 0.3, 1.0)  # 浅绿色 - 右腿
        }

        # 关节点大小 - 火柴人风格更小的关节点
        self.joint_sizes = {
            0: 0.04,  # 鼻子稍大
            # 其他关节点默认值更小
        }

        # 骨骼颜色 - 优化配色方案
        self.bone_colors = {}
        
        # 面部骨骼 - 天蓝色
        face_color = (0.2, 0.7, 1.0, 1.0)
        for connection in self.body_groups['face']:
            self.bone_colors[connection] = face_color
            
        # 耳朵到肩膀的连接 - 淡紫色
        face_shoulder_color = (0.6, 0.5, 0.8, 1.0)
        for connection in self.body_groups['face_shoulder']:
            self.bone_colors[connection] = face_shoulder_color
            
        # 躯干骨骼 - 深灰色
        torso_color = (0.3, 0.3, 0.3, 1.0)
        for connection in self.body_groups['torso']:
            self.bone_colors[connection] = torso_color
            
        # 左臂骨骼 - 淡红色
        left_arm_color = (1.0, 0.4, 0.4, 1.0)
        for connection in self.body_groups['left_arm']:
            self.bone_colors[connection] = left_arm_color
            
        # 右臂骨骼 - 淡红色
        right_arm_color = (1.0, 0.4, 0.4, 1.0)
        for connection in self.body_groups['right_arm']:
            self.bone_colors[connection] = right_arm_color
            
        # 左腿骨骼 - 浅绿色
        left_leg_color = (0.3, 0.9, 0.3, 1.0)
        for connection in self.body_groups['left_leg']:
            self.bone_colors[connection] = left_leg_color
            
        # 右腿骨骼 - 浅绿色
        right_leg_color = (0.3, 0.9, 0.3, 1.0)
        for connection in self.body_groups['right_leg']:
            self.bone_colors[connection] = right_leg_color

        # 骨骼粗细 - 不同部位使用不同粗细增强可读性
        self.bone_thickness = {
            'face': 2.5,           # 面部骨骼较细
            'face_shoulder': 3.0,  # 耳朵到肩膀的连接
            'torso': 4.5,          # 躯干骨骼较粗
            'left_arm': 4.0,       # 左臂骨骼中等
            'right_arm': 4.0,      # 右臂骨骼中等
            'left_leg': 4.0,       # 左腿骨骼中等
            'right_leg': 4.0       # 右腿骨骼中等
        }
        
        # 默认粗细
        self.default_thickness = 4.0

        # 轨迹参数 - 禁用轨迹效果
        self.motion_trail_enabled = False


class Pose3DViewer(QOpenGLWidget):
    """3D姿态查看器组件"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # 设置OpenGL格式
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # 初始化变量
        self.keypoints = np.zeros((17, 3), dtype=np.float32)  # 17个关键点，每个有x, y, z坐标
        self.model = Human3DModel()  # 加载人体模型的连接关系
        self.has_data = False  # 是否有关键点数据
        
        # 视图变换参数 - 设置默认正面朝向
        self.x_rotation = 0.0
        self.y_rotation = 180.0  # 初始Y轴旋转180度，使模型正面朝向观察者
        self.z_rotation = 0.0    # 初始Z轴旋转为0
        self.offset = [0.0, 2.0, -5.0]  # x, y, z偏移，y轴上移0.5使模型全身可见
        
        # 目标旋转角度 - 与当前角度保持一致
        self.target_x_rotation = 0.0
        self.target_y_rotation = 180.0
        self.target_z_rotation = 0.0
        
        # 自动旋转相关属性
        self.auto_rotate = True
        self.auto_rotate_speed = 0.5
        self.auto_rotate_direction = 1
        self.auto_rotate_angle_limit = [-15, 15]  # X轴旋转角度限制
        self.auto_rotate_x_direction = 1
        self.auto_rotate_paused = False
        self.user_interacting = False
        self.rotation_transition_speed = 0.1
        
        # 设置默认视角
        self.reset_view()
        
        # 自动旋转设置
        self.rotation_timer = QTimer(self)
        self.rotation_timer.timeout.connect(self.rotate_model)
        self.rotation_timer.start(50)  # 每50毫秒更新一次
        
        # 交互超时设置 - 用户交互后暂停自动旋转
        self.interaction_timeout = QTimer(self)
        self.interaction_timeout.setSingleShot(True)
        self.interaction_timeout.timeout.connect(self.resume_auto_rotation)
        
        # 鼠标拖动控制
        self.last_pos = None
        self.drag_sensitivity = 0.5  # 拖动灵敏度
        
        # 启用鼠标滚轮缩放
        self.setMouseTracking(True)
        self.zoom_level = -5.0
        self.min_zoom = -12.0  # 增大缩放范围
        self.max_zoom = -1.0   # 增大缩放范围
        self.model_scale = 1.5  # 增大默认模型缩放因子
        self.min_model_scale = 0.5  # 增大最小缩放
        self.max_model_scale = 4.0  # 增大最大缩放
        
        # 运动轨迹
        self.previous_keypoints = []
        
        # 设置连续更新
        self.setAutoFillBackground(False)
        
        # 检查依赖包
        self.check_dependencies()

    def check_dependencies(self):
        """检查必要的依赖包"""
        try:
            # 仅检查是否能够导入PIL用于纹理创建
            from PIL import Image, ImageDraw
        except ImportError:
            print("未安装PIL/Pillow库，尝试安装...")
            self.install_dependency("pillow")
            
        try:
            # 尝试导入numpy（应该已经安装，因为它是PyQt6的依赖项）
            import numpy
        except ImportError:
            print("未安装numpy库，尝试安装...")
            self.install_dependency("numpy")
    
    def install_dependency(self, package_name):
        """安装依赖包
        
        Args:
            package_name: 包名
        """
        try:
            import subprocess
            import sys
            
            # 使用pip安装包
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"成功安装 {package_name}")
            return True
        except Exception as e:
            print(f"安装 {package_name} 失败: {str(e)}")
            return False

    def initializeGL(self):
        """初始化OpenGL"""
        # 设置背景颜色为纯白色，符合简约火柴人风格
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glEnable(GL_DEPTH_TEST)  # 启用深度测试
        glEnable(GL_BLEND)       # 启用混合
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # 启用点的抗锯齿
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

        # 启用线的抗锯齿
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # 启用多重采样抗锯齿（如果支持）
        try:
            glEnable(GL_MULTISAMPLE)
        except:
            pass

        # 火柴人风格不需要光照，禁用光照效果
        glDisable(GL_LIGHTING)
        glDisable(GL_LIGHT0)

        # 初始化动态效果所需的变量
        self.previous_keypoints = []

    def resizeGL(self, width, height):
        """调整OpenGL视口"""
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = width / height if height != 0 else 1.0
        gluPerspective(45, aspect, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def paintGL(self):
        """OpenGL绘制函数"""
        # 清除颜色缓冲区和深度缓冲区
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # 设置透视投影
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # 设置视野角度、长宽比、近裁剪面和远裁剪面
        gluPerspective(45, self.width() / max(1, self.height()), 0.1, 100.0)
        
        # 设置模型视图矩阵
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # 应用视图变换 (偏移和旋转)
        glTranslatef(self.offset[0], self.offset[1], self.offset[2] + self.zoom_level)
        
        glRotatef(self.x_rotation, 1.0, 0.0, 0.0)  # 绕X轴旋转
        glRotatef(self.y_rotation, 0.0, 1.0, 0.0)  # 绕Y轴旋转
        glRotatef(self.z_rotation, 0.0, 0.0, 1.0)  # 绕Z轴旋转
        
        # 应用用户定义的模型缩放因子
        glScalef(self.model_scale, self.model_scale, self.model_scale)
        
        # 绘制参考网格
        self.draw_reference_grid()
        
        # 如果有关键点数据，则绘制骨架
        if self.has_data:
            # 绘制骨架 - 先绘制骨骼
            self.draw_skeleton()
            # 再绘制关节点，覆盖在骨骼末端
            self.draw_joints()
        else:
            # 无数据时绘制默认姿态
            self.draw_default_tpose()

    def draw_reference_grid(self):
        """增强的参考网格"""
        light_state = glIsEnabled(GL_LIGHTING)
        if light_state:
            glDisable(GL_LIGHTING)

        try:
            grid_size = 5.0  # 稍微扩大网格范围
            num_lines = 10   # 增加网格线数量
            grid_y = -3.5  # 网格所在高度保持不变

            # 设置网格线颜色和宽度
            glColor4f(0.9, 0.9, 0.9, 0.7) # 更淡的灰色，稍高透明度
            glLineWidth(0.5)             # 更细的线条

            glBegin(GL_LINES)
            step = grid_size * 2 / num_lines
            for i in range(num_lines + 1):
                pos = -grid_size + i * step
                # 平行于Z轴的线
                glVertex3f(pos, grid_y, -grid_size)
                glVertex3f(pos, grid_y, grid_size)
                # 平行于X轴的线
                glVertex3f(-grid_size, grid_y, pos)
                glVertex3f(grid_size, grid_y, pos)
            glEnd()

            # 绘制中心十字线 (稍粗，稍深)
            glColor4f(0.8, 0.8, 0.8, 0.8)
            glLineWidth(0.8)
            glBegin(GL_LINES)
            # X轴中心线
            glVertex3f(-grid_size, grid_y, 0)
            glVertex3f(grid_size, grid_y, 0)
            # Z轴中心线
            glVertex3f(0, grid_y, -grid_size)
            glVertex3f(0, grid_y, grid_size)
            glEnd()

        finally:
            if light_state:
                glEnable(GL_LIGHTING)

    def draw_joints(self):
        """绘制关节点 - 按COCO标准显示关键点"""
        for i in range(len(self.keypoints)):
            if np.all(self.keypoints[i] != 0):  # 只绘制有效的关键点
                # 确定关节点类型和颜色
                joint_group = None
                for group, indices in self.model.joint_groups.items():
                    if i in indices:
                        joint_group = group
                        break

                # 获取关节点颜色
                color = self.model.joint_colors.get(joint_group, (0.7, 0.7, 0.7, 0.9))
                
                # 根据关节点重要程度调整大小
                # 所有关键点都显示，但大小不同
                if i == 0:  # 鼻子
                    size = 0.04
                elif i in [3, 4, 9, 10, 15, 16]:  # 耳朵、手腕、脚踝
                    size = 0.03
                else:  # 其他关节点
                    size = 0.02
                
                self.draw_joint_point(self.keypoints[i], size, color)

    def draw_joint_point(self, position, size, color):
        """绘制现代风格的关节点，带有更柔和的光晕效果

        Args:
            position: 位置坐标 [x, y, z]
            size: 关节点大小
            color: 颜色 (r, g, b, a)
        """
        glPushAttrib(GL_ALL_ATTRIB_BITS)

        try:
            glDisable(GL_LIGHTING)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_POINT_SMOOTH)
            glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

            # 绘制更柔和的外部光晕 (多层叠加)
            num_glow_layers = 4
            max_glow_size_factor = 5.0 # 光晕最大尺寸因子
            base_alpha = 0.15 # 基础透明度

            for i in range(num_glow_layers):
                # 尺寸从大到小，透明度从低到高
                lerp_factor = (i / (num_glow_layers - 1)) if num_glow_layers > 1 else 1
                glow_size_factor = max_glow_size_factor * (1.0 - lerp_factor * 0.7) # 非线性减小
                alpha = base_alpha * (1.0 - lerp_factor * 0.8) * (1.0 / (i + 1)) # 快速衰减

                glow_size = size * glow_size_factor
                glPointSize(glow_size * 20) # 调整点大小的缩放因子
                glBegin(GL_POINTS)
                glColor4f(color[0], color[1], color[2], alpha)
                glVertex3f(position[0], position[1], position[2])
                glEnd()

            # 绘制主要关节点 (稍微调亮)
            main_point_size_factor = 1.0
            glPointSize(size * main_point_size_factor * 15) # 调整点大小的缩放因子
            glBegin(GL_POINTS)
            # 稍微提亮主点颜色
            enhanced_color = (min(color[0] * 1.1, 1.0), min(color[1] * 1.1, 1.0),
                             min(color[2] * 1.1, 1.0), color[3] * 0.95) # 主点透明度稍降，让光晕更明显
            glColor4f(*enhanced_color)
            glVertex3f(position[0], position[1], position[2])
            glEnd()

            # 绘制更小、更亮的高光
            highlight_size_factor = 0.4
            glPointSize(size * highlight_size_factor * 15) # 调整点大小的缩放因子
            glBegin(GL_POINTS)
            glColor4f(1.0, 1.0, 1.0, 0.7) # 高光颜色和透明度
            glVertex3f(position[0], position[1], position[2])
            glEnd()

        finally:
            glPopAttrib()

    def draw_sphere(self, position, radius, color):
        """绘制球体 - 此方法保留但不再使用，改为使用draw_joint_point"""
        # 保留原方法以兼容性，但实际使用draw_joint_point
        self.draw_joint_point(position, radius, color)

    def draw_skeleton(self):
        """绘制骨架连接 - 为不同部位使用对应粗细"""
        for connection in self.model.connections:
            p1, p2 = connection
            # 如果两个关键点都是有效的（不是零向量）
            if np.any(self.keypoints[p1]) and np.any(self.keypoints[p2]):
                # 获取骨骼颜色
                color = self.model.bone_colors.get(
                    connection, (0.0, 0.0, 0.0, 1.0))  # 默认黑色
                
                # 获取骨骼所属的分组
                bone_group = None
                for group, connections in self.model.body_groups.items():
                    if connection in connections:
                        bone_group = group
                        break
                
                # 获取相应的线宽
                line_width = self.model.bone_thickness.get(
                    bone_group, self.model.default_thickness)

                # 计算骨骼线段的方向和长度
                start = self.keypoints[p1]
                end = self.keypoints[p2]

                # 绘制骨骼线段
                self.draw_bone(start, end, color, line_width)

    def draw_bone(self, start, end, color, line_width=4.0):
        """绘制骨骼线条，支持渐变色和可选轮廓"""
        # 检查start和end是否有效
        if not (isinstance(start, (list, np.ndarray)) and len(start) == 3 and
                isinstance(end, (list, np.ndarray)) and len(end) == 3):
            # print(f"无效的骨骼端点: start={start}, end={end}") # 调试用
            return

        glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT | GL_LINE_BIT)

        try:
            # 检查start和end是否有效 (numpy数组检查)
            if not (np.all(np.isfinite(start)) and np.all(np.isfinite(end))):
                # print(f"包含非有限值的骨骼端点: start={start}, end={end}") # 调试用
                return # 如果包含NaN或Inf，则不绘制

            direction = end - start
            length = np.linalg.norm(direction)
            if length == 0:
                return

            gradient_start = (min(color[0]*1.1, 1.0), min(color[1]*1.1, 1.0),
                             min(color[2]*1.1, 1.0), color[3])
            gradient_end = (color[0]*0.9, color[1]*0.9, color[2]*0.9, color[3])

            # 绘制主骨骼线
            glLineWidth(line_width)
            glBegin(GL_LINES)
            glColor4f(*gradient_start)
            glVertex3fv(start) # 使用glVertex3fv传递数组
            glColor4f(*gradient_end)
            glVertex3fv(end)
            glEnd()

            # 绘制圆头端点 (使用 GL_POINTS)
            # 点的大小应该与线宽匹配，glPointSize是以像素为单位，需要估算
            # 注意：glPointSize可能因驱动和抗锯齿设置而异，效果可能不完美
            glPointSize(line_width) # 设置点大小等于线宽
            glBegin(GL_POINTS)
            # 绘制起点圆头
            glColor4f(*gradient_start)
            glVertex3fv(start)
            # 绘制终点圆头
            glColor4f(*gradient_end)
            glVertex3fv(end)
            glEnd()

            # 可选：绘制细微的轮廓线 (如果需要)
            glLineWidth(line_width + 1.0)
            glColor4f(1.0, 1.0, 1.0, 0.2) # 降低轮廓透明度
            glBegin(GL_LINES)
            glVertex3fv(start)
            glVertex3fv(end)
            glEnd()

        finally:
            glPopAttrib()

    def draw_default_tpose(self):
        """绘制默认T姿势骨架 - 更自然的站姿"""
        # 创建一个自然的站姿作为默认显示
        default_keypoints = np.zeros((17, 3), dtype=np.float32)

        # 面部 - 更自然的位置和深度
        default_keypoints[0] = [0.0, 1.0, 0.1]       # 鼻子（稍微向前）
        default_keypoints[1] = [-0.15, 1.1, 0.0]     # 左眼
        default_keypoints[2] = [0.15, 1.1, 0.0]      # 右眼
        default_keypoints[3] = [-0.3, 1.0, -0.1]     # 左耳（稍微向后）
        default_keypoints[4] = [0.3, 1.0, -0.1]      # 右耳（稍微向后）

        # 上身 - 自然放松的姿势
        default_keypoints[5] = [-0.7, 0.5, 0.0]     # 左肩（稍微降低）
        default_keypoints[6] = [0.7, 0.5, 0.0]      # 右肩
        default_keypoints[7] = [-1.4, 0.0, 0.1]     # 左肘（稍微向前）
        default_keypoints[8] = [1.4, 0.0, 0.1]      # 右肘
        default_keypoints[9] = [-2.0, -0.3, 0.0]    # 左手腕（自然下垂）
        default_keypoints[10] = [2.0, -0.3, 0.0]    # 右手腕

        # 躯干 - 自然的站姿
        default_keypoints[11] = [-0.4, -1.2, 0.0]   # 左髋（更窄的站姿）
        default_keypoints[12] = [0.4, -1.2, 0.0]    # 右髋
        default_keypoints[13] = [-0.5, -2.4, 0.1]   # 左膝（稍微向前）
        default_keypoints[14] = [0.5, -2.4, 0.1]    # 右膝
        default_keypoints[15] = [-0.4, -3.6, 0.0]   # 左踝
        default_keypoints[16] = [0.4, -3.6, 0.0]    # 右踝

        # 临时保存当前关键点
        temp_keypoints = self.keypoints.copy()
        temp_has_data = self.has_data

        # 使用默认关键点
        self.keypoints = default_keypoints
        self.has_data = True

        # 绘制骨架
        self.draw_skeleton()
        
        # 绘制关节点
        self.draw_joints()

        # 恢复原始关键点
        self.keypoints = temp_keypoints
        self.has_data = temp_has_data

    def update_keypoints(self, keypoints):
        """更新关键点数据

        Args:
            keypoints: 关键点数组 (可能是2D或3D)
        """
        # Check if keypoints data is valid (not None, not empty, and not all zeros)
        if keypoints is None or keypoints.size == 0 or np.all(keypoints == 0):
            # If data is invalid or represents no detection
            if self.has_data: # Only update if state changes from having data to no data
                self.has_data = False
                # Clear previous keypoints to avoid drawing old trails if enabled
                if self.model.motion_trail_enabled:
                    self.previous_keypoints.clear()
                self.update() # Trigger redraw to show default pose
            return # Exit the function early

        # --- If we reach here, the data is considered valid ---
        self.has_data = True # Set flag indicating valid data is present

        # 保存先前的关键点用于运动轨迹
        if self.model.motion_trail_enabled:
            # Copy the *current* state before updating self.keypoints
            # Ensure we copy the *previous* valid state if needed, but the current logic copies before overwrite
            current_copy = self.keypoints.copy()
            self.previous_keypoints.insert(0, current_copy)
            # 限制轨迹长度
            while len(self.previous_keypoints) > self.model.motion_trail_length:
                self.previous_keypoints.pop()

        # 确保关键点是3D的
        if keypoints.shape[1] == 2:  # 如果是2D关键点 (x,y)
            # 转换为3D，设置z坐标为0
            keypoints_3d = np.zeros((keypoints.shape[0], 3), dtype=np.float32)
            
            # 先规范化关键点坐标到合适的显示范围
            # 计算边界框
            valid_points = keypoints[np.any(keypoints != 0, axis=1)]
            if len(valid_points) > 0:
                min_x, min_y = np.min(valid_points, axis=0)
                max_x, max_y = np.max(valid_points, axis=0)
                
                # 计算缩放比例和中心点偏移
                width = max_x - min_x
                height = max_y - min_y
                scale = 3.0 / max(width, height) if max(width, height) > 0 else 1.0
                
                # 计算中心点
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                
                # 对所有有效点进行缩放和居中
                for i in range(len(keypoints)):
                    if np.any(keypoints[i] != 0):
                        # 将点相对于中心点进行缩放和位移，翻转X轴修复左右颠倒问题
                        keypoints_3d[i, 0] = -1.0 * (keypoints[i, 0] - center_x) * scale  # x (翻转以修复左右问题)
                        keypoints_3d[i, 1] = -(keypoints[i, 1] - center_y) * scale + 0.5  # y (反转y轴方向，使朝上为正，并向上偏移)
            
            self.keypoints = keypoints_3d
        else:
            # 已经是3D关键点，同样需要规范化
            keypoints_3d = np.zeros((keypoints.shape[0], 3), dtype=np.float32)
            
            # 处理3D关键点
            valid_points = keypoints[np.any(keypoints != 0, axis=1)]
            if len(valid_points) > 0:
                # 计算中心点和缩放比例
                min_vals = np.min(valid_points, axis=0)
                max_vals = np.max(valid_points, axis=0)
                
                center = (min_vals + max_vals) / 2
                # Calculate scale factor robustly, avoiding division by zero or near-zero
                scale_factors = max_vals - min_vals
                valid_scale_factors = scale_factors[scale_factors > 1e-6] # Use a small epsilon
                if valid_scale_factors.size > 0:
                     scale = 3.0 / np.max(valid_scale_factors)
                else:
                     scale = 1.0 # No scaling needed if points are coincident or dimensions are zero

                # 对所有有效点进行缩放和居中
                for i in range(len(keypoints)):
                    if np.any(keypoints[i] != 0):
                        keypoints_3d[i] = (keypoints[i] - center) * scale
                        keypoints_3d[i, 0] = -keypoints_3d[i, 0]  # 翻转x轴以修复左右问题
                        keypoints_3d[i, 1] = -keypoints_3d[i, 1] + 0.5  # 反转y轴并向上偏移
            
            self.keypoints = keypoints_3d
            
        # 确保视图正确朝向并居中
        # 重置旋转角度使人体正面朝向屏幕
        self.target_y_rotation = 180  # 使人体正面朝向屏幕
        self.target_x_rotation = 0
        self.target_z_rotation = 0
        
        # 确保模型在视图中居中显示
        self.offset = np.array([0.0, 0.0, -5.0])
        
        # 触发重绘
        self.update()

    def update_pose(self, keypoints):
        """update_pose方法的别名，用于兼容现有代码
        
        Args:
            keypoints: 关键点数组 (可能是2D或3D)
        """
        return self.update_keypoints(keypoints)

    def rotate_model(self):
        """平滑自动旋转模型"""
        if self.auto_rotate and not self.user_interacting and not self.auto_rotate_paused:
            # Y轴旋转 - 持续旋转，速度降低以防止初始化时旋转过快
            self.target_y_rotation = (self.target_y_rotation + self.auto_rotate_speed * self.auto_rotate_direction * 0.4) % 360
            
            # X轴小范围摆动 - 使视角更生动
            if (self.target_x_rotation >= self.auto_rotate_angle_limit[1] and self.auto_rotate_x_direction > 0) or \
               (self.target_x_rotation <= self.auto_rotate_angle_limit[0] and self.auto_rotate_x_direction < 0):
                self.auto_rotate_x_direction *= -1  # 反向
            
            self.target_x_rotation += 0.1 * self.auto_rotate_x_direction  # 降低X轴摆动速度
        
        # 平滑过渡到目标角度，降低旋转过渡速度
        transition_speed = 0.03  # 降低过渡速度
        self.x_rotation += (self.target_x_rotation - self.x_rotation) * transition_speed
        self.y_rotation += (self.target_y_rotation - self.y_rotation) * transition_speed
        self.z_rotation += (self.target_z_rotation - self.z_rotation) * transition_speed
        
        # 触发重绘
        self.update()

    def toggle_auto_rotate(self, enabled):
        """切换自动旋转

        Args:
            enabled: 是否启用自动旋转
        """
        self.auto_rotate = enabled
        self.auto_rotate_paused = not enabled
        
        if not enabled:
            # 重置旋转目标
            self.target_x_rotation = 0
            self.target_y_rotation = 0
            self.target_z_rotation = 0

    def pause_auto_rotation(self):
        """暂停自动旋转，用于用户交互时"""
        self.user_interacting = True
        self.auto_rotate_paused = True
        
    def resume_auto_rotation(self):
        """恢复自动旋转，用户交互结束后延迟调用"""
        self.user_interacting = False
        if self.auto_rotate:
            self.auto_rotate_paused = False

    def mousePressEvent(self, event):
        """鼠标按下事件"""
        self.last_pos = event.position()
        self.pause_auto_rotation()

    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        # 设置计时器，2秒后恢复自动旋转
        self.interaction_timeout.start(2000)

    def mouseMoveEvent(self, event):
        """鼠标移动事件 - 增强拖动体验"""
        if self.last_pos is None:
            self.last_pos = event.position()
            return
            
        dx = event.position().x() - self.last_pos.x()
        dy = event.position().y() - self.last_pos.y()

        if event.buttons() & Qt.MouseButton.LeftButton:
            # 直接设置目标角度，让rotate_model函数处理平滑过渡
            self.target_y_rotation = (self.target_y_rotation + dx * self.drag_sensitivity) % 360
            self.target_x_rotation = max(-90, min(90, self.target_x_rotation + dy * self.drag_sensitivity))
            
            # 延长交互超时
            self.interaction_timeout.start(2000)

        self.last_pos = event.position()

    def wheelEvent(self, event):
        """鼠标滚轮事件 - 实现缩放"""
        delta = event.angleDelta().y() / 120  # 获取滚轮增量
        
        # 调整视图缩放
        new_zoom = self.zoom_level + delta * 0.5
        self.zoom_level = max(self.min_zoom, min(self.max_zoom, new_zoom))
        
        # 更新偏移
        self.offset[2] = self.zoom_level
        
        # 暂停自动旋转
        self.pause_auto_rotation()
        
        # 设置计时器，2秒后恢复自动旋转
        self.interaction_timeout.start(2000)
        
        # 更新视图
        self.update()

    def reset_view(self):
        """重置视图到默认正面朝向"""
        # 设置目标角度为默认正面朝向
        self.target_x_rotation = 0.0
        self.target_y_rotation = 180.0  # 保持Y轴旋转180度，使模型正面朝向
        self.target_z_rotation = 0.0
        
        # 直接设置当前角度，无需过渡动画
        self.x_rotation = self.target_x_rotation
        self.y_rotation = self.target_y_rotation
        self.z_rotation = self.target_z_rotation
        
        # 重置缩放
        self.zoom_level = -5.0
        self.offset[2] = self.zoom_level
        
        # 重置模型缩放
        self.model_scale = 1.5
        
        # 更新视图
        self.update()

    def toggle_model_mode(self, use_3d_model=None):
        """切换模型显示模式 (这个函数保留但内部逻辑修改为只返回False)
        
        Args:
            use_3d_model: 是否使用3D模型 (True/False)，如果为None则切换当前状态
            
        Returns:
            bool: 当前的模型模式
        """
        # 始终返回False（不使用3D模型模式）
        return False

    def render_text(self, text, x, y, z, color=(0.0, 0.0, 0.0, 1.0)):
        """渲染文本"""
        try:
            # 保存当前矩阵
            glPushMatrix()
            
            try:
                # 移动到文本位置
                glTranslatef(x, y, z)
                
                # 旋转使文本始终面向用户
                glRotatef(-self.y_rotation, 0.0, 1.0, 0.0)
                glRotatef(-self.x_rotation, 1.0, 0.0, 0.0)
                
                # 设置文本颜色
                glColor4f(*color)
                
                # 设置文本大小 - 使用缩放
                text_scale = 0.001
                glScalef(text_scale, text_scale, text_scale)
                
                # 使用Qt的QPainter绘制文本
                self.renderText(0, 0, 0, text)
            finally:
                # 恢复矩阵
                glPopMatrix()
        except Exception as e:
            # Qt6的OpenGL不支持直接renderText，忽略错误
            pass
