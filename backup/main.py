import sys
import os
import json
import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timedelta
import csv
import traceback

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QTextEdit, QLineEdit, QGroupBox, QGridLayout,
                             QSpacerItem, QSizePolicy, QCheckBox, QFormLayout, QScrollArea,
                             QDialog, QListWidget, QListWidgetItem, QSplitter, QMenu, QAction,
                             QMessageBox, QTableWidget, QTableWidgetItem, QTabWidget,
                             QProgressBar, QHeaderView)
from PyQt5.QtGui import (QImage, QPixmap, QFont, QColor, QMovie, QIcon, QPainter, QPen)
from PyQt5.QtCore import (Qt, QTimer, pyqtSignal, QThread, qVersion, QPoint, QSize)

# 导入自定义模块
from models import FaceRecognitionModels
from database import FaceRecognitionDatabase
from camera import FaceRecognitionCamera
from utils import FaceRecognitionUtils
from config import FaceRecognitionConfig

class FaceRecognitionSystem(QMainWindow):
    """智能人脸识别系统主类"""
    def __init__(self):
        super().__init__()
        # 初始化配置
        self.config = FaceRecognitionConfig()
        # 初始化模型状态（必须在init_ui之前）
        self.model_status = "未检查"
        # 初始化UI（必须在update_log之前）
        self.init_ui()
        # 初始化数据结构
        self.init_data_structures()
        # 初始化工具类
        self.utils = FaceRecognitionUtils(self)
        # 初始化数据库
        self.database = FaceRecognitionDatabase(self)
        # 初始化模型
        self.models = FaceRecognitionModels(self)
        # 初始化摄像头
        self.camera = FaceRecognitionCamera(self)
        # 加载人脸数据库
        self.load_face_database()
        # 系统启动信息
        self.update_log("系统启动成功")
        self.update_status("就绪")

    def init_data_structures(self):
        """初始化数据结构"""
        self.face_database = {}  # 人脸数据库: {name: {'features': [], 'images': [], 'info': {}}}
        self.current_frame = None
        self.current_faces = []
        self.tracking_data = defaultdict(dict)
        self.recognition_history = []
        self.attendance_records = {}
        # 新增：人脸识别稳定性相关
        self.recognition_results = {}  # {face_id: {'name': '', 'confidence': 0, 'history': deque()}}
        self.stable_recognition = {}  # 稳定的识别结果
        self.fixed_recognition = {}  # 固定的识别结果
        # 签退人脸识别相关（修复需求3）
        self.checkout_recognition = {}  # {name: {'required': True, 'recognized': False, 'confidence': 0}}
        # 实时状态
        self.is_camera_running = False
        self.is_recognizing = False
        self.is_recording = False
        self.is_attendance_running = False
        # 统计信息
        self.total_recognitions = 0
        self.total_attendance = 0
        self.total_users = 0
        # 考勤摄像头相关
        self.attendance_camera = None
        self.attendance_viewfinder = None
        # 修复问题1和2：初始化定时器
        self.recognition_timer = QTimer()
        self.recognition_timer.timeout.connect(self.perform_recognition)
        self.recognition_timer.setInterval(100)  # 10fps
        
        # 修复问题3：添加信息数据更新定时器（每10秒更新一次）
        self.info_update_timer = QTimer()
        self.info_update_timer.timeout.connect(self.update_info_data)
        self.info_update_timer.setInterval(10000)  # 10秒

    def init_ui(self):
        """初始化UI界面"""
        self.setWindowTitle("智能人脸识别系统 - 最终修复版")
        self.setGeometry(100, 100, 1300, 850)
        self.setMinimumSize(1100, 750)

        # 设置主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # 顶部状态栏
        self.create_status_bar(main_layout)

        # 主标签页
        self.tabs = QTabWidget()
        self.create_tabs()
        main_layout.addWidget(self.tabs)

        # 底部日志区域
        self.create_log_area(main_layout)

        # 应用样式
        self.apply_styles()

    def create_status_bar(self, parent_layout):
        """创建状态栏"""
        status_bar = QWidget()
        status_bar.setFixedHeight(40)
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(10, 5, 10, 5)

        # 系统信息
        self.mode_label = QLabel("模式: 人脸识别")
        self.status_label = QLabel("状态: 就绪")
        self.camera_status_label = QLabel("摄像头: 关闭")
        self.model_status_label = QLabel(f"模型状态: {self.model_status}")

        # 设置初始样式
        self.camera_status_label.setStyleSheet("color: red;")

        # 设置模型状态颜色
        if self.model_status == "完整":
            self.model_status_label.setStyleSheet("color: green;")
        elif self.model_status == "部分完整":
            self.model_status_label.setStyleSheet("color: orange;")
        else:
            self.model_status_label.setStyleSheet("color: red;")

        # 新增：统计信息
        self.stats_label = QLabel("用户数: 0 | 识别次数: 0 | 考勤次数: 0")

        status_layout.addWidget(self.mode_label)
        status_layout.addSpacing(20)
        status_layout.addWidget(self.status_label)
        status_layout.addSpacing(20)
        status_layout.addWidget(self.camera_status_label)
        status_layout.addSpacing(20)
        status_layout.addWidget(self.model_status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.stats_label)

        parent_layout.addWidget(status_bar)

    def create_tabs(self):
        """创建标签页"""
        # 人脸识别标签页
        self.create_recognition_tab()
        # 人脸录入标签页
        self.create_enrollment_tab()
        # 数据管理标签页 - 增强版
        self.create_enhanced_data_management_tab()
        # 考勤管理标签页
        self.create_attendance_tab_with_camera()
        # 系统设置标签页
        self.create_settings_tab()

        # 标签页切换信号
        self.tabs.currentChanged.connect(self.on_tab_changed)

    def create_recognition_tab(self):
        """创建人脸识别标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 摄像头显示区域
        self.video_label = QWidget()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setLayout(QVBoxLayout())
        self.video_label.layout().setAlignment(Qt.AlignCenter)

        # 默认显示文本
        default_text = QLabel("摄像头未启动")
        default_text.setStyleSheet("color: #aaa; font-size: 16px;")
        default_text.setAlignment(Qt.AlignCenter)
        self.video_label.layout().addWidget(default_text)

        # 控制按钮区域
        control_layout = QHBoxLayout()
        self.camera_btn = QPushButton("启动摄像头")
        self.camera_btn.clicked.connect(self.toggle_camera)
        self.start_recognition_btn = QPushButton("开始识别")
        self.start_recognition_btn.clicked.connect(self.start_recognition)
        self.stop_recognition_btn = QPushButton("停止识别")
        self.stop_recognition_btn.clicked.connect(self.stop_recognition)
        self.stop_recognition_btn.setEnabled(False)

        # 新增：识别稳定性控制
        self.stability_control = QCheckBox("启用识别稳定化")
        self.stability_control.setChecked(True)
        # 新增：识别结果固定控制
        self.fix_result_control = QCheckBox("识别稳定后固定结果")
        self.fix_result_control.setChecked(True)

        control_layout.addWidget(self.camera_btn)
        control_layout.addWidget(self.start_recognition_btn)
        control_layout.addWidget(self.stop_recognition_btn)
        control_layout.addWidget(self.stability_control)
        control_layout.addWidget(self.fix_result_control)

        # 文件选择按钮
        file_layout = QHBoxLayout()
        self.image_btn = QPushButton("选择图片")
        self.image_btn.clicked.connect(self.select_image)
        self.video_btn = QPushButton("选择视频")
        self.video_btn.clicked.connect(self.select_video)
        file_layout.addWidget(self.image_btn)
        file_layout.addWidget(self.video_btn)

        # 识别结果显示 - 增强版
        result_group = QGroupBox("识别结果")
        result_layout = QGridLayout(result_group)

        # 主要识别结果
        self.result_label = QLabel("等待识别...")
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.confidence_label = QLabel("置信度: -")
        self.confidence_label.setStyleSheet("font-size: 14px;")
        # 稳定性指示
        self.stability_label = QLabel("稳定性: -")
        self.stability_label.setStyleSheet("font-size: 14px;")
        # 固定状态指示
        self.fixed_label = QLabel("状态: 实时")
        self.fixed_label.setStyleSheet("font-size: 14px; color: blue;")

        # 详细信息
        self.age_gender_label = QLabel("年龄/性别: -")
        self.emotion_label = QLabel("情绪: -")
        self.mask_label = QLabel("口罩: -")

        # 布局排列
        result_layout.addWidget(self.result_label, 0, 0, 1, 2)
        result_layout.addWidget(self.confidence_label, 1, 0)
        result_layout.addWidget(self.stability_label, 1, 1)
        result_layout.addWidget(self.fixed_label, 2, 0)
        result_layout.addWidget(self.age_gender_label, 3, 0)
        result_layout.addWidget(self.emotion_label, 3, 1)
        result_layout.addWidget(self.mask_label, 4, 0)

        # 布局组装
        layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        layout.addLayout(control_layout)
        layout.addLayout(file_layout)
        layout.addWidget(result_group)

        self.tabs.addTab(tab, "人脸识别")

    def create_enrollment_tab(self):
        """创建人脸录入标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 人员信息表单
        form_group = QGroupBox("人员信息")
        form_layout = QFormLayout(form_group)
        self.enroll_name = QLineEdit()
        self.enroll_age = QLineEdit()
        self.enroll_gender = QLineEdit()
        self.enroll_department = QLineEdit()

        form_layout.addRow("姓名:", self.enroll_name)
        form_layout.addRow("年龄:", self.enroll_age)
        form_layout.addRow("性别:", self.enroll_gender)
        form_layout.addRow("部门:", self.enroll_department)

        # 录入方式选择
        method_layout = QHBoxLayout()
        self.enroll_camera_btn = QPushButton("启动摄像头录入")
        self.enroll_camera_btn.clicked.connect(self.toggle_enroll_camera)
        self.enroll_image_btn = QPushButton("选择图片录入")
        self.enroll_image_btn.clicked.connect(self.select_enroll_image)
        self.batch_enroll_btn = QPushButton("批量导入照片")
        self.batch_enroll_btn.clicked.connect(self.batch_enroll_images)

        method_layout.addWidget(self.enroll_camera_btn)
        method_layout.addWidget(self.enroll_image_btn)
        method_layout.addWidget(self.batch_enroll_btn)

        # 预览和照片列表区域
        preview_splitter = QSplitter(Qt.Horizontal)

        # 摄像头预览区域
        self.enroll_preview = QWidget()
        self.enroll_preview.setFixedSize(320, 240)
        self.enroll_preview.setLayout(QVBoxLayout())
        self.enroll_preview.layout().setAlignment(Qt.AlignCenter)

        # 默认预览文本
        default_preview = QLabel("预览区域")
        default_preview.setStyleSheet("color: #aaa; font-size: 14px;")
        default_preview.setAlignment(Qt.AlignCenter)
        self.enroll_preview.layout().addWidget(default_preview)

        # 照片列表区域
        self.photos_list = QListWidget()
        self.photos_list.setViewMode(QListWidget.IconMode)
        self.photos_list.setIconSize(QSize(60, 60))
        self.photos_list.setResizeMode(QListWidget.Adjust)
        self.photos_list.setGridSize(QSize(70, 70))
        self.photos_list.itemSelectionChanged.connect(self.on_photo_selection_changed)

        preview_splitter.addWidget(self.enroll_preview)
        preview_splitter.addWidget(self.photos_list)
        preview_splitter.setSizes([320, 320])

        # 操作按钮
        action_layout = QHBoxLayout()
        self.capture_btn = QPushButton("捕获人脸")
        self.capture_btn.clicked.connect(self.capture_face)
        self.capture_btn.setEnabled(False)
        self.delete_photo_btn = QPushButton("删除选中照片")
        self.delete_photo_btn.clicked.connect(self.delete_selected_photo)
        self.delete_photo_btn.setEnabled(False)
        self.save_btn = QPushButton("保存信息")
        self.save_btn.clicked.connect(self.save_enrollment)

        action_layout.addWidget(self.capture_btn)
        action_layout.addWidget(self.delete_photo_btn)
        action_layout.addWidget(self.save_btn)

        # 自动保存选项（修复需求2）
        auto_save_layout = QHBoxLayout()
        self.auto_save_checkbox = QCheckBox("照片录入后自动保存")
        self.auto_save_checkbox.setChecked(self.config.get('auto_save_after_enrollment', True))
        auto_save_layout.addWidget(self.auto_save_checkbox)
        auto_save_layout.addStretch()

        # 状态信息
        self.enrollment_status = QLabel("状态：请填写人员信息并选择录入方式")

        # 布局组装
        layout.addWidget(form_group)
        layout.addLayout(method_layout)
        layout.addWidget(preview_splitter)
        layout.addLayout(action_layout)
        layout.addLayout(auto_save_layout)
        layout.addWidget(self.enrollment_status)

        self.tabs.addTab(tab, "人脸录入")

    def create_enhanced_data_management_tab(self):
        """创建增强版数据管理标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 数据表格
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(6)
        self.data_table.setHorizontalHeaderLabels(['姓名', '年龄', '性别', '部门', '照片数量', '创建时间'])
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.data_table.verticalHeader().setVisible(False)
        self.data_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.data_table.selectionModel().selectionChanged.connect(self.on_data_table_selection_changed)

        # 操作按钮
        button_layout = QHBoxLayout()
        self.view_photos_btn = QPushButton("查看照片")
        self.view_photos_btn.clicked.connect(self.view_user_photos)
        self.view_photos_btn.setEnabled(False)
        self.edit_btn = QPushButton("编辑信息")
        self.edit_btn.clicked.connect(self.edit_user_info)
        self.edit_btn.setEnabled(False)
        self.delete_btn = QPushButton("删除选中")
        self.delete_btn.clicked.connect(self.delete_selected)
        self.delete_btn.setEnabled(False)

        button_layout.addWidget(self.view_photos_btn)
        button_layout.addWidget(self.edit_btn)
        button_layout.addWidget(self.delete_btn)

        # 数据导入导出
        import_export_layout = QHBoxLayout()
        self.import_btn = QPushButton("导入数据")
        self.import_btn.clicked.connect(self.import_data)
        self.export_btn = QPushButton("导出数据")
        self.export_btn.clicked.connect(self.export_data)
        self.refresh_btn = QPushButton("刷新数据")
        self.refresh_btn.clicked.connect(self.refresh_data)

        import_export_layout.addWidget(self.import_btn)
        import_export_layout.addWidget(self.export_btn)
        import_export_layout.addWidget(self.refresh_btn)

        # 布局组装
        layout.addWidget(self.data_table)
        layout.addLayout(button_layout)
        layout.addLayout(import_export_layout)

        self.tabs.addTab(tab, "数据管理")

    def create_attendance_tab_with_camera(self):
        """创建带摄像头的考勤管理标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 考勤摄像头区域
        attendance_camera_group = QGroupBox("考勤摄像头")
        attendance_camera_layout = QVBoxLayout(attendance_camera_group)
        
        # 摄像头显示
        self.attendance_camera_label = QWidget()
        self.attendance_camera_label.setFixedSize(320, 240)
        self.attendance_camera_label.setLayout(QVBoxLayout())
        self.attendance_camera_label.layout().setAlignment(Qt.AlignCenter)
        
        # 默认文本
        default_camera_text = QLabel("考勤摄像头已停止")
        default_camera_text.setStyleSheet("color: #aaa; font-size: 14px;")
        default_camera_text.setAlignment(Qt.AlignCenter)
        self.attendance_camera_label.layout().addWidget(default_camera_text)
        
        # 摄像头控制按钮
        camera_control_layout = QHBoxLayout()
        self.start_attendance_camera_btn = QPushButton("启动考勤摄像头")
        self.start_attendance_camera_btn.clicked.connect(self.start_attendance_camera)
        self.stop_attendance_camera_btn = QPushButton("停止考勤摄像头")
        self.stop_attendance_camera_btn.clicked.connect(self.stop_attendance_camera)
        self.stop_attendance_camera_btn.setEnabled(False)
        
        camera_control_layout.addWidget(self.start_attendance_camera_btn)
        camera_control_layout.addWidget(self.stop_attendance_camera_btn)
        
        attendance_camera_layout.addWidget(self.attendance_camera_label)
        attendance_camera_layout.addLayout(camera_control_layout)

        # 考勤控制
        control_group = QGroupBox("考勤控制")
        control_layout = QHBoxLayout(control_group)
        self.start_attendance_btn = QPushButton("开始考勤")
        self.start_attendance_btn.clicked.connect(self.start_attendance)
        self.stop_attendance_btn = QPushButton("停止考勤")
        self.stop_attendance_btn.clicked.connect(self.stop_attendance)
        self.stop_attendance_btn.setEnabled(False)
        
        # 签退人脸识别控制（修复需求3）
        self.checkout_recognition_checkbox = QCheckBox("签退需要人脸识别确认")
        self.checkout_recognition_checkbox.setChecked(self.config.get('checkout_recognition_required', True))
        
        control_layout.addWidget(self.start_attendance_btn)
        control_layout.addWidget(self.stop_attendance_btn)
        control_layout.addWidget(self.checkout_recognition_checkbox)

        # 考勤记录表格
        self.attendance_table = QTableWidget()
        self.attendance_table.setColumnCount(5)
        self.attendance_table.setHorizontalHeaderLabels(['姓名', '签到时间', '签退时间', '状态', '位置'])
        self.attendance_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.attendance_table.verticalHeader().setVisible(False)

        # 布局组装
        layout.addWidget(attendance_camera_group)
        layout.addWidget(control_group)
        layout.addWidget(self.attendance_table)

        self.tabs.addTab(tab, "考勤管理")

    def create_settings_tab(self):
        """创建系统设置标签页"""
        tab = QWidget()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)

        # 模型配置
        model_group = QGroupBox("模型配置")
        model_form = QFormLayout(model_group)
        self.shape_predictor_edit = QLineEdit(self.config.get('shape_predictor_path', 'models/shape_predictor_68_face_landmarks.dat'))
        self.face_recognizer_edit = QLineEdit(self.config.get('face_recognition_model_path', 'models/dlib_face_recognition_resnet_model_v1.dat'))
        self.use_local_checkbox = QCheckBox("仅使用本地模型")
        self.use_local_checkbox.setChecked(self.config.get('use_local_models_only', True))

        model_form.addRow("特征点预测器路径:", self.shape_predictor_edit)
        model_form.addRow("人脸识别模型路径:", self.face_recognizer_edit)
        model_form.addRow(self.use_local_checkbox)

        # MySQL数据库配置
        mysql_group = QGroupBox("MySQL数据库配置")
        mysql_form = QFormLayout(mysql_group)
        self.mysql_host_edit = QLineEdit(self.config.get('mysql_host', 'localhost'))
        self.mysql_port_edit = QLineEdit(str(self.config.get('mysql_port', 3306)))
        self.mysql_user_edit = QLineEdit(self.config.get('mysql_user', 'root'))
        self.mysql_password_edit = QLineEdit(self.config.get('mysql_password', '123456'))
        self.mysql_database_edit = QLineEdit(self.config.get('mysql_database', 'smart_attendance'))

        mysql_form.addRow("主机地址:", self.mysql_host_edit)
        mysql_form.addRow("端口:", self.mysql_port_edit)
        mysql_form.addRow("用户名:", self.mysql_user_edit)
        mysql_form.addRow("密码:", self.mysql_password_edit)
        mysql_form.addRow("数据库名:", self.mysql_database_edit)

        # 识别设置
        recognition_group = QGroupBox("识别设置")
        recognition_form = QFormLayout(recognition_group)
        self.threshold_edit = QLineEdit(str(self.config.get('threshold', 0.4)))
        self.stability_threshold_edit = QLineEdit(str(self.config.get('recognition_stability_threshold', 0.15)))
        self.fix_threshold_edit = QLineEdit(str(self.config.get('recognition_fix_threshold', 0.8)))
        self.min_stable_frames_edit = QLineEdit(str(self.config.get('min_stable_frames', 3)))
        self.min_face_size_edit = QLineEdit(str(self.config.get('min_face_size', 100)))

        recognition_form.addRow("识别阈值 (0.0-1.0):", self.threshold_edit)
        recognition_form.addRow("稳定性阈值 (0.0-1.0):", self.stability_threshold_edit)
        recognition_form.addRow("结果固定阈值 (0.0-1.0):", self.fix_threshold_edit)
        recognition_form.addRow("最少稳定帧数:", self.min_stable_frames_edit)
        recognition_form.addRow("最小人脸尺寸 (像素):", self.min_face_size_edit)

        # 系统设置
        system_group = QGroupBox("系统设置")
        system_form = QFormLayout(system_group)
        self.api_port_edit = QLineEdit(str(self.config.get('api_port', 5000)))
        self.camera_index_edit = QLineEdit(str(self.config.get('camera_index', 0)))
        self.face_images_per_person_edit = QLineEdit(str(self.config.get('face_images_per_person', 5)))
        
        # 修复需求2：自动保存设置
        self.auto_save_checkbox_setting = QCheckBox("人脸录入后自动保存")
        self.auto_save_checkbox_setting.setChecked(self.config.get('auto_save_after_enrollment', True))
        
        # 修复需求3：签退人脸识别设置
        self.checkout_recognition_checkbox_setting = QCheckBox("签退需要人脸识别确认")
        self.checkout_recognition_checkbox_setting.setChecked(self.config.get('checkout_recognition_required', True))

        system_form.addRow("API端口:", self.api_port_edit)
        system_form.addRow("摄像头索引:", self.camera_index_edit)
        system_form.addRow("每人最多照片数:", self.face_images_per_person_edit)
        system_form.addRow(self.auto_save_checkbox_setting)
        system_form.addRow(self.checkout_recognition_checkbox_setting)

        # 保存按钮
        save_layout = QHBoxLayout()
        self.save_settings_btn = QPushButton("保存设置")
        self.save_settings_btn.clicked.connect(lambda: self.config.save_settings(self))
        save_layout.addWidget(self.save_settings_btn)
        save_layout.addStretch()

        # 布局组装
        settings_layout.addWidget(model_group)
        settings_layout.addWidget(mysql_group)
        settings_layout.addWidget(recognition_group)
        settings_layout.addWidget(system_group)
        settings_layout.addLayout(save_layout)
        settings_layout.addStretch()

        scroll_area.setWidget(settings_widget)
        main_layout = QVBoxLayout(tab)
        main_layout.addWidget(scroll_area)

        self.tabs.addTab(tab, "系统设置")

    def apply_styles(self):
        """应用样式"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 4px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
                color: #333333;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QLineEdit {
                padding: 4px;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
            QLabel {
                color: #333333;
            }
            QTableWidget {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 4px;
            }
            QTableWidget::item:selected {
                background-color: #4CAF50;
                color: white;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 4px;
                border: 1px solid #cccccc;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 4px;
            }
        """)

    def update_status(self, status):
        """更新状态显示"""
        self.status_label.setText(f"状态: {status}")

    def update_log(self, message):
        """更新日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}\n"
        self.log_edit.insertPlainText(log_message)
        self.log_edit.moveCursor(self.log_edit.textCursor().End)

    def update_stats(self):
        """更新统计信息"""
        self.stats_label.setText(f"用户数: {self.total_users} | 识别次数: {self.total_recognitions} | 考勤次数: {self.total_attendance}")

    def create_log_area(self, parent_layout):
        """创建日志区域"""
        log_group = QGroupBox("系统日志")
        log_layout = QVBoxLayout(log_group)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFixedHeight(120)
        log_layout.addWidget(self.log_edit)
        parent_layout.addWidget(log_group)

    def load_face_database(self):
        """加载人脸数据库"""
        self.database.load_face_database()

    def toggle_camera(self):
        """切换摄像头状态"""
        self.camera.toggle_camera()

    def stop_camera(self):
        """停止摄像头"""
        self.camera.stop_camera()

    def select_image(self):
        """选择图片文件"""
        self.utils.select_image()

    def select_video(self):
        """选择视频文件"""
        self.utils.select_video()

    def display_image_from_pil(self, pil_image):
        """从PIL图像显示"""
        self.utils.display_image_from_pil(pil_image)

    def display_image(self, qimage):
        """显示图像"""
        self.utils.display_image(qimage)

    def start_recognition(self):
        """开始人脸识别"""
        self.models.start_recognition()
        # 修复问题3：开始识别时启动信息更新定时器
        if not self.info_update_timer.isActive():
            self.info_update_timer.start()
            self.update_log("信息数据更新定时器已启动（每10秒更新一次）")

    def stop_recognition(self):
        """停止人脸识别"""
        self.models.stop_recognition()
        # 修复问题3：停止识别时停止信息更新定时器
        if self.info_update_timer.isActive():
            self.info_update_timer.stop()
            self.update_log("信息数据更新定时器已停止")

    def recognize_face_from_image(self, image):
        """从图像识别人脸"""
        return self.models.recognize_face_from_image(image)

    def toggle_enroll_camera(self):
        """切换录入摄像头"""
        self.camera.toggle_enroll_camera()

    def start_enroll_camera(self):
        """启动录入摄像头"""
        self.camera.start_enroll_camera()

    def stop_enroll_camera(self):
        """停止录入摄像头"""
        self.camera.stop_enroll_camera()

    def select_enroll_image(self):
        """选择录入图片 - 修复需求1：确保照片能正确添加和保存"""
        self.utils.select_enroll_image()

    def display_enroll_image(self, pil_image):
        """显示录入图像"""
        self.utils.display_enroll_image(pil_image)

    def detect_face_for_enrollment(self, image):
        """检测录入人脸"""
        return self.models.detect_face_for_enrollment(image)

    def capture_face(self):
        """捕获人脸"""
        self.utils.capture_face()

    def add_photo_to_list(self, photo_path):
        """添加照片到列表显示 - 修复需求1的关键"""
        self.utils.add_photo_to_list(photo_path)

    def on_photo_selection_changed(self):
        """照片选择变化处理"""
        self.utils.on_photo_selection_changed()

    def delete_selected_photo(self):
        """删除选中的照片"""
        self.utils.delete_selected_photo()

    def batch_enroll_images(self):
        """批量导入照片"""
        self.utils.batch_enroll_images()

    def save_enrollment(self):
        """保存录入信息"""
        self.database.save_enrollment()

    def auto_save_enrollment(self):
        """自动保存录入信息 - 修复需求2"""
        self.database.auto_save_enrollment()

    def save_to_database(self, name, age, gender, department, photo_paths):
        """保存到数据库"""
        self.database.save_to_database(name, age, gender, department, photo_paths)

    def perform_recognition(self):
        """执行人脸识别"""
        self.models.perform_recognition()

    def start_attendance_camera(self):
        """启动考勤摄像头"""
        self.camera.start_attendance_camera()

    def stop_attendance_camera(self):
        """停止考勤摄像头"""
        self.camera.stop_attendance_camera()

    def on_data_table_selection_changed(self):
        """数据表格选择变化处理"""
        self.utils.on_data_table_selection_changed()

    def view_user_photos(self):
        """查看用户照片"""
        self.utils.view_user_photos()

    def edit_user_info(self):
        """编辑用户信息"""
        self.utils.edit_user_info()

    def import_data(self):
        """导入数据"""
        self.database.import_data()

    def refresh_data(self):
        """刷新数据表格"""
        self.database.refresh_data()

    def delete_selected(self):
        """删除选中行"""
        try:
            selected_rows = self.data_table.selectionModel().selectedRows()
            if not selected_rows:
                QMessageBox.warning(self, "警告", "请选择要删除的用户")
                return

            reply = QMessageBox.question(self, "确认", "确定要删除选中的用户吗？",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return

            deleted_count = 0
            for row in selected_rows:
                name = self.data_table.item(row.row(), 0).text()
                if self.database.delete_face(name):
                    deleted_count += 1

            self.refresh_data()
            QMessageBox.information(self, "成功", f"成功删除 {deleted_count} 个用户")
        except Exception as e:
            self.update_log(f"删除用户失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"删除用户失败: {str(e)}")

    def delete_face(self, name):
        """删除指定人脸"""
        self.database.delete_face(name)

    def export_data(self):
        """导出数据"""
        self.database.export_data()

    def start_attendance(self):
        """开始考勤"""
        try:
            self.is_attendance_running = True
            self.start_attendance_btn.setEnabled(False)
            self.stop_attendance_btn.setEnabled(True)
            self.update_status("考勤中")
            self.update_log("开始考勤监控")
            # 启动考勤摄像头
            self.start_attendance_camera()
        except Exception as e:
            self.update_log(f"启动考勤失败: {str(e)}")

    def stop_attendance(self):
        """停止考勤"""
        try:
            self.is_attendance_running = False
            self.start_attendance_btn.setEnabled(True)
            self.stop_attendance_btn.setEnabled(False)
            self.update_status("就绪")
            self.update_log("停止考勤监控")
            # 停止考勤摄像头
            self.stop_attendance_camera()
        except Exception as e:
            self.update_log(f"停止考勤失败: {str(e)}")

    def update_attendance(self):
        """更新考勤信息"""
        if not self.is_attendance_running:
            return
        try:
            # 简化实现：模拟考勤检测
            pass
        except Exception as e:
            self.update_log(f"更新考勤失败: {str(e)}")

    def on_tab_changed(self, index):
        """标签页切换处理"""
        tab_name = self.tabs.tabText(index)
        self.mode_label.setText(f"模式: {tab_name}")

    def update_info_data(self):
        """更新信息数据（每10秒更新一次）- 修复问题3"""
        try:
            if not self.is_recognizing:
                return
                
            self.update_log("正在更新信息数据...")
            
            # 1. 更新统计信息
            self.update_stats()
            
            # 2. 刷新识别历史记录
            if hasattr(self, 'recognition_history') and self.recognition_history:
                recent_history = self.recognition_history[-5:]  # 获取最近5条记录
                history_str = "最近识别: " + ", ".join([f"{item['name']}({item['confidence']:.2f})" 
                                                      for item in recent_history])
                self.update_log(history_str)
            
            # 3. 检查并更新数据库连接状态
            if hasattr(self.database, 'db_conn') and self.database.db_conn:
                try:
                    self.database.db_cursor.execute("SELECT 1")
                    db_status = "正常"
                except Exception as e:
                    db_status = f"异常: {str(e)}"
                self.update_log(f"数据库连接状态: {db_status}")
            
            # 4. 更新用户数量信息
            current_users = len(self.face_database)
            self.update_log(f"当前用户数量: {current_users}")
            
            # 5. 更新摄像头状态信息
            camera_status = "开启" if self.is_camera_running else "关闭"
            self.update_log(f"摄像头状态: {camera_status}")
            
            # 6. 更新识别模式信息
            recognition_mode = "稳定化" if self.stability_control.isChecked() else "实时"
            self.update_log(f"识别模式: {recognition_mode}")
            
            self.update_log("信息数据更新完成")
            
        except Exception as e:
            self.update_log(f"更新信息数据失败: {str(e)}")

    def update_all_stats(self):
        """更新所有统计信息"""
        try:
            # 更新界面统计显示
            self.update_stats()
            
            # 更新考勤统计
            if hasattr(self.database, 'get_attendance_records'):
                today = datetime.now().strftime('%Y-%m-%d')
                attendance_records = self.database.get_attendance_records(today)
                self.total_attendance = len(attendance_records)
                self.update_log(f"今日考勤记录: {self.total_attendance} 条")
                
        except Exception as e:
            self.update_log(f"更新统计信息失败: {str(e)}")

    def initiate_checkout(self, name):
        """初始化签退流程（修复需求3）"""
        self.database.initiate_checkout(name)

    def perform_checkout_recognition(self):
        """执行签退人脸识别（修复需求3）"""
        self.models.perform_checkout_recognition()

    def complete_checkout(self, name, confidence):
        """完成签退（修复需求3）"""
        self.database.complete_checkout(name, confidence)

    def closeEvent(self, event):
        """关闭事件处理"""
        try:
            # 停止所有服务
            self.stop_recognition()
            self.stop_camera()
            self.stop_attendance()
            self.stop_api_service()
            self.stop_attendance_camera()
            
            # 停止所有定时器
            if self.recognition_timer.isActive():
                self.recognition_timer.stop()
            if self.info_update_timer.isActive():
                self.info_update_timer.stop()
            
            # 保存数据
            self.database.save_face_database()
            
            # 关闭数据库连接
            if hasattr(self.database, 'db_conn') and self.database.db_conn:
                self.database.db_conn.close()
                
            self.update_log("系统关闭成功")
            event.accept()
        except Exception as e:
            self.update_log(f"关闭系统时出错: {str(e)}")
            event.accept()

    def start_api_service(self):
        """启动API服务"""
        # 简化实现
        self.update_log("API服务启动功能开发中")

    def stop_api_service(self):
        """停止API服务"""
        # 简化实现
        self.update_log("API服务停止功能开发中")

def main():
    """主函数"""
    try:
        app = QApplication(sys.argv)
        window = FaceRecognitionSystem()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"系统启动失败: {str(e)}")
        sys.exit(1)

def reset_fixed_recognition(self):
    """重置固定识别结果"""
    self.fixed_recognition = {}
    self.stable_recognition = {}
    self.recognition_results = {}
    self.update_log("已重置固定识别结果")
    self.result_label.setText("等待识别...")
    self.confidence_label.setText("置信度: -")
    self.stability_label.setText("稳定性: -")
    self.fixed_label.setText("状态: 实时")
    self.fixed_label.setStyleSheet("font-size: 14px; color: blue;")

# 在 create_recognition_tab 方法中的控制按钮区域添加重置按钮：
def create_recognition_tab(self):
    """创建人脸识别标签页"""
    tab = QWidget()
    layout = QVBoxLayout(tab)

    # 摄像头显示区域
    self.video_label = QWidget()
    self.video_label.setFixedSize(640, 480)
    self.video_label.setLayout(QVBoxLayout())
    self.video_label.layout().setAlignment(Qt.AlignCenter)

    # 默认显示文本
    default_text = QLabel("摄像头未启动")
    default_text.setStyleSheet("color: #aaa; font-size: 16px;")
    default_text.setAlignment(Qt.AlignCenter)
    self.video_label.layout().addWidget(default_text)

    # 控制按钮区域
    control_layout = QHBoxLayout()
    self.camera_btn = QPushButton("启动摄像头")
    self.camera_btn.clicked.connect(self.toggle_camera)
    self.start_recognition_btn = QPushButton("开始识别")
    self.start_recognition_btn.clicked.connect(self.start_recognition)
    self.stop_recognition_btn = QPushButton("停止识别")
    self.stop_recognition_btn.clicked.connect(self.stop_recognition)
    self.stop_recognition_btn.setEnabled(False)

    # 新增：重置识别按钮
    self.reset_btn = QPushButton("重置识别")
    self.reset_btn.clicked.connect(self.reset_fixed_recognition)

    # 新增：识别稳定性控制
    self.stability_control = QCheckBox("启用识别稳定化")
    self.stability_control.setChecked(True)
    # 新增：识别结果固定控制
    self.fix_result_control = QCheckBox("识别稳定后固定结果")
    self.fix_result_control.setChecked(True)

    control_layout.addWidget(self.camera_btn)
    control_layout.addWidget(self.start_recognition_btn)
    control_layout.addWidget(self.stop_recognition_btn)
    control_layout.addWidget(self.reset_btn)  # 添加重置按钮
    control_layout.addWidget(self.stability_control)
    control_layout.addWidget(self.fix_result_control)



if __name__ == "__main__":
    main()