#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
UI组件初始化文件
"""

from .sidebar import Sidebar
from .home_page import HomePage
from .workout_page import WorkoutPage
from .analysis_page import AnalysisPage
from .settings_page import SettingsPage
from .pose_3d_viewer import Pose3DViewer, Human3DModel
from .mmpose_page import MMPosePage

__all__ = [
    'Sidebar',
    'HomePage',
    'WorkoutPage',
    'AnalysisPage',
    'SettingsPage',
    'Pose3DViewer',
    'Human3DModel',
    'MMPosePage'
]
