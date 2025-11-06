# 智能人脸识别系统 - 修复版
# 修复了QStatusBar导入问题

import os
import sys
import json
import time
import datetime
import csv
import numpy as np
from PIL import Image, ImageQt
import dlib
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QLineEdit, QTextEdit, 
                             QFileDialog, QListWidget, QListWidgetItem, QMessageBox, 
                             QProgressBar, QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox,
                             QFormLayout, QTableWidget, QTableWidgetItem, QHeaderView, QStatusBar)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtMultimedia import QCamera, QCameraImageCapture
from PyQt5.QtMultimediaWidgets import QCameraViewfinder
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import mysql.connector
from mysql.connector import Error

class FaceRecognitionSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能人脸识别考勤系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 加载配置
        self.config = self.load_config()
        
        # 初始化数据库连接
        self.db_connection = None
        self.init_database()
        
        # 初始化人脸识别模型
        self.face_detector = None
        self.shape_predictor = None
        self.face_recognizer = None
        self.init_models()
        
        # 人脸特征数据
        self.face_features = {}
        self.load_face_features()
        
        # 摄像头相关
        self.camera = None
        self.camera_viewfinder = None
        self.image_capture = None
        self.is_camera_running = False
        self.is_recognizing = False
        self.capture_timer = None
        
        # 考勤相关
        self.attendance_records = []
        self.load_attendance_records()
        
        # UI初始化
        self.init_ui()
        
        # 状态更新
        self.update_status()
    
    def load_config(self):
        """加载配置文件"""
        config_path = "config.json"
        default_config = {
            "database_path": "face_database",
            "features_file": "face_features.csv",
            "log_file": "recognition_log.csv",
            "attendance_file": "attendance.csv",
            "model_path": "models",
            "shape_predictor_path": "models/shape_predictor_68_face_landmarks.dat",
            "face_recognition_model_path": "models/dlib_face_recognition_resnet_model_v1.dat",
            "threshold": 0.4,
            "max_faces": 100,
            "api_port": 5000,
            "camera_index": 0,
            "use_local_models_only": True,
            # MySQL数据库配置
            "mysql_host": "localhost",
            "mysql_database": "smart_attendance",
            "mysql_user": "root",
            "mysql_password": "123456"
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 合并默认配置和文件配置
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            except Exception as e:
                print(f"加载配置文件失败: {e}")
                return default_config
        else:
            # 保存默认配置
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            return default_config
    
    def init_database(self):
        """初始化MySQL数据库连接"""
        try:
            self.db_connection = mysql.connector.connect(
                host=self.config["mysql_host"],
                database=self.config["mysql_database"],
                user=self.config["mysql_user"],
                password=self.config["mysql_password"]
            )
            
            if self.db_connection.is_connected():
                db_Info = self.db_connection.get_server_info()
                print(f"数据库连接成功: MySQL Server version {db_Info}")
                
                # 创建必要的表
                self.create_tables()
                return True
                
        except Error as e:
            print(f"数据库连接错误: {e}")
            # 如果MySQL连接失败，尝试创建数据库
            try:
                connection = mysql.connector.connect(
                    host=self.config["mysql_host"],
                    user=self.config["mysql_user"],
                    password=self.config["mysql_password"]
                )
                
                if connection.is_connected():
                    cursor = connection.cursor()
                    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.config['mysql_database']}")
                    print(f"数据库 {self.config['mysql_database']} 创建成功")
                    connection.close()
                    
                    # 重新连接
                    self.db_connection = mysql.connector.connect(
                        host=self.config["mysql_host"],
                        database=self.config["mysql_database"],
                        user=self.config["mysql_user"],
                        password=self.config["mysql_password"]
                    )
                    self.create_tables()
                    return True
                    
            except Error as e2:
                print(f"创建数据库失败: {e2}")
                QMessageBox.critical(self, "数据库错误", f"无法连接到MySQL数据库: {e2}")
                return False
    
    def create_tables(self):
        """创建数据库表"""
        if not self.db_connection or not self.db_connection.is_connected():
            return
        
        try:
            cursor = self.db_connection.cursor()
            
            # 创建用户表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    age INT,
                    gender VARCHAR(20),
                    department VARCHAR(100),
                    face_encoding TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY unique_name (name)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')
            
            # 创建考勤记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    check_in_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    check_out_time TIMESTAMP NULL,
                    status VARCHAR(20) DEFAULT 'present',
                    location VARCHAR(100),
                    temperature DECIMAL(5,2),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')
            
            # 创建识别日志表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recognition_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NULL,
                    recognition_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence DECIMAL(10,6),
                    age_prediction INT,
                    gender_prediction VARCHAR(20),
                    emotion_prediction VARCHAR(20),
                    mask_detection VARCHAR(20),
                    image_path VARCHAR(255),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')
            
            self.db_connection.commit()
            print("数据库表创建成功")
            
        except Error as e:
            print(f"创建表失败: {e}")
            self.db_connection.rollback()
    
    def init_models(self):
        """初始化人脸识别模型"""
        try:
            # 初始化人脸检测器
            self.face_detector = dlib.get_frontal_face_detector()
            print("人脸检测器初始化成功")
            
            # 初始化形状预测器
            if os.path.exists(self.config["shape_predictor_path"]):
                self.shape_predictor = dlib.shape_predictor(self.config["shape_predictor_path"])
                print("形状预测器初始化成功")
            else:
                print(f"形状预测器模型文件不存在: {self.config['shape_predictor_path']}")
            
            # 初始化人脸识别模型
            if os.path.exists(self.config["face_recognition_model_path"]):
                self.face_recognizer = dlib.face_recognition_model_v1(self.config["face_recognition_model_path"])
                print("人脸识别模型初始化成功")
            else:
                print(f"人脸识别模型文件不存在: {self.config['face_recognition_model_path']}")
                
        except Exception as e:
            print(f"模型初始化失败: {e}")
            QMessageBox.critical(self, "模型错误", f"人脸识别模型初始化失败: {e}")
    
    def load_face_features(self):
        """从MySQL数据库加载人脸特征"""
        self.face_features = {}
        if not self.db_connection or not self.db_connection.is_connected():
            return
        
        try:
            cursor = self.db_connection.cursor(dictionary=True)
            cursor.execute("SELECT id, name, face_encoding FROM users")
            
            for user in cursor.fetchall():
                try:
                    # 将字符串编码转换为numpy数组
                    encoding = np.array(json.loads(user["face_encoding"]), dtype=np.float64)
                    self.face_features[user["id"]] = {
                        "name": user["name"],
                        "encoding": encoding
                    }
                except Exception as e:
                    print(f"加载用户 {user['name']} 的人脸特征失败: {e}")
            
            print(f"加载了 {len(self.face_features)} 个人脸特征")
            
        except Error as e:
            print(f"加载人脸特征失败: {e}")
    
    def load_attendance_records(self):
        """加载考勤记录"""
        self.attendance_records = []
        if not self.db_connection or not self.db_connection.is_connected():
            return
        
        try:
            cursor = self.db_connection.cursor(dictionary=True)
            cursor.execute('''
                SELECT a.*, u.name FROM attendance a
                LEFT JOIN users u ON a.user_id = u.id
                ORDER BY a.check_in_time DESC
            ''')
            
            self.attendance_records = cursor.fetchall()
            print(f"加载了 {len(self.attendance_records)} 条考勤记录")
            
        except Error as e:
            print(f"加载考勤记录失败: {e}")
    
    def init_ui(self):
        """初始化UI界面"""
        # 创建主Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        
        # 创建各个标签页
        self.create_recognition_tab()
        self.create_enrollment_tab()
        self.create_data_management_tab()
        self.create_attendance_tab()
        self.create_settings_tab()
        
        main_layout.addWidget(self.tab_widget)
        
        # 创建状态栏 - 修复导入问题
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
    
    def create_recognition_tab(self):
        """创建人脸识别标签页"""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # 左侧：摄像头显示
        left_layout = QVBoxLayout()
        
        # 摄像头显示区域
        self.camera_display = QLabel()
        self.camera_display.setFixedSize(640, 480)
        self.camera_display.setStyleSheet("background-color: #2c3e50;")
        self.camera_display.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.camera_display)
        
        # 摄像头控制按钮
        camera_controls = QHBoxLayout()
        self.start_camera_btn = QPushButton("启动摄像头")
        self.start_camera_btn.clicked.connect(self.start_camera)
        camera_controls.addWidget(self.start_camera_btn)
        
        self.stop_camera_btn = QPushButton("停止摄像头")
        self.stop_camera_btn.clicked.connect(self.stop_camera)
        self.stop_camera_btn.setEnabled(False)
        camera_controls.addWidget(self.stop_camera_btn)
        
        self.start_recognition_btn = QPushButton("开始识别")
        self.start_recognition_btn.clicked.connect(self.start_recognition)
        self.start_recognition_btn.setEnabled(False)
        camera_controls.addWidget(self.start_recognition_btn)
        
        self.stop_recognition_btn = QPushButton("停止识别")
        self.stop_recognition_btn.clicked.connect(self.stop_recognition)
        self.stop_recognition_btn.setEnabled(False)
        camera_controls.addWidget(self.stop_recognition_btn)
        
        left_layout.addLayout(camera_controls)
        
        # 右侧：识别结果和控制
        right_layout = QVBoxLayout()
        
        # 识别结果显示
        self.recognition_result = QTextEdit()
        self.recognition_result.setReadOnly(True)
        self.recognition_result.setFixedHeight(200)
        right_layout.addWidget(QLabel("识别结果:"))
        right_layout.addWidget(self.recognition_result)
        
        # 图片识别
        image_recognition_layout = QVBoxLayout()
        image_recognition_layout.addWidget(QLabel("图片识别:"))
        
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setPlaceholderText("选择图片文件...")
        image_recognition_layout.addWidget(self.image_path_edit)
        
        image_buttons = QHBoxLayout()
        self.browse_image_btn = QPushButton("浏览图片")
        self.browse_image_btn.clicked.connect(self.browse_image)
        image_buttons.addWidget(self.browse_image_btn)
        
        self.recognize_image_btn = QPushButton("识别图片")
        self.recognize_image_btn.clicked.connect(self.recognize_image)
        image_buttons.addWidget(self.recognize_image_btn)
        
        image_recognition_layout.addLayout(image_buttons)
        right_layout.addLayout(image_recognition_layout)
        
        # 识别统计
        stats_layout = QVBoxLayout()
        stats_layout.addWidget(QLabel("识别统计:"))
        
        self.stats_table = QTableWidget(4, 2)
        self.stats_table.setHorizontalHeaderLabels(["指标", "数值"])
        self.stats_table.setItem(0, 0, QTableWidgetItem("总识别次数"))
        self.stats_table.setItem(1, 0, QTableWidgetItem("成功识别次数"))
        self.stats_table.setItem(2, 0, QTableWidgetItem("识别准确率"))
        self.stats_table.setItem(3, 0, QTableWidgetItem("当前在线人数"))
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        stats_layout.addWidget(self.stats_table)
        
        right_layout.addLayout(stats_layout)
        
        layout.addLayout(left_layout)
        layout.addLayout(right_layout)
        
        self.tab_widget.addTab(tab, "人脸识别")
    
    def create_enrollment_tab(self):
        """创建人脸录入标签页"""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # 左侧：用户信息录入
        left_layout = QVBoxLayout()
        
        # 用户信息表单
        form_layout = QFormLayout()
        
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("请输入姓名")
        form_layout.addRow("姓名:", self.name_edit)
        
        self.age_spin = QSpinBox()
        self.age_spin.setRange(1, 120)
        self.age_spin.setValue(25)
        form_layout.addRow("年龄:", self.age_spin)
        
        self.gender_edit = QLineEdit()
        self.gender_edit.setPlaceholderText("请输入性别")
        form_layout.addRow("性别:", self.gender_edit)
        
        self.department_edit = QLineEdit()
        self.department_edit.setPlaceholderText("请输入部门")
        form_layout.addRow("部门:", self.department_edit)
        
        left_layout.addLayout(form_layout)
        
        # 摄像头录入区域
        camera_group = QGroupBox("摄像头录入")
        camera_group_layout = QVBoxLayout(camera_group)
        
        # 摄像头预览区域 - 修复预览问题
        self.enrollment_camera_display = QLabel()
        self.enrollment_camera_display.setFixedSize(320, 240)
        self.enrollment_camera_display.setStyleSheet("background-color: #2c3e50;")
        self.enrollment_camera_display.setAlignment(Qt.AlignCenter)
        camera_group_layout.addWidget(self.enrollment_camera_display)
        
        # 摄像头控制按钮
        camera_buttons = QHBoxLayout()
        self.start_enrollment_camera_btn = QPushButton("启动摄像头")
        self.start_enrollment_camera_btn.clicked.connect(self.start_enrollment_camera)
        camera_buttons.addWidget(self.start_enrollment_camera_btn)
        
        self.capture_face_btn = QPushButton("捕获人脸")
        self.capture_face_btn.clicked.connect(self.capture_face)
        self.capture_face_btn.setEnabled(False)
        camera_buttons.addWidget(self.capture_face_btn)
        
        self.stop_enrollment_camera_btn = QPushButton("停止摄像头")
        self.stop_enrollment_camera_btn.clicked.connect(self.stop_enrollment_camera)
        self.stop_enrollment_camera_btn.setEnabled(False)
        camera_buttons.addWidget(self.stop_enrollment_camera_btn)
        
        camera_group_layout.addLayout(camera_buttons)
        left_layout.addWidget(camera_group)
        
        # 图片录入
        image_group = QGroupBox("图片录入")
        image_group_layout = QVBoxLayout(image_group)
        
        self.enrollment_image_path = QLineEdit()
        self.enrollment_image_path.setPlaceholderText("选择包含人脸的图片...")
        image_group_layout.addWidget(self.enrollment_image_path)
        
        image_buttons = QHBoxLayout()
        self.browse_enrollment_image_btn = QPushButton("浏览图片")
        self.browse_enrollment_image_btn.clicked.connect(self.browse_enrollment_image)
        image_buttons.addWidget(self.browse_enrollment_image_btn)
        
        self.process_image_btn = QPushButton("处理图片")
        self.process_image_btn.clicked.connect(self.process_enrollment_image)
        image_buttons.addWidget(self.process_image_btn)
        
        image_group_layout.addLayout(image_buttons)
        left_layout.addWidget(image_group)
        
        # 保存按钮
        self.save_user_btn = QPushButton("保存用户信息")
        self.save_user_btn.clicked.connect(self.save_user)
        self.save_user_btn.setEnabled(False)
        left_layout.addWidget(self.save_user_btn)
        
        # 右侧：预览和状态
        right_layout = QVBoxLayout()
        
        # 人脸预览
        self.face_preview = QLabel()
        self.face_preview.setFixedSize(200, 200)
        self.face_preview.setStyleSheet("background-color: #34495e;")
        self.face_preview.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(QLabel("人脸预览:"))
        right_layout.addWidget(self.face_preview)
        
        # 特征提取状态
        self.feature_status = QLabel("请先捕获或选择人脸图片")
        self.feature_status.setStyleSheet("color: #7f8c8d;")
        right_layout.addWidget(self.feature_status)
        
        # 进度条
        self.enrollment_progress = QProgressBar()
        self.enrollment_progress.setVisible(False)
        right_layout.addWidget(self.enrollment_progress)
        
        # 消息显示
        self.enrollment_message = QTextEdit()
        self.enrollment_message.setReadOnly(True)
        self.enrollment_message.setFixedHeight(150)
        right_layout.addWidget(QLabel("操作日志:"))
        right_layout.addWidget(self.enrollment_message)
        
        layout.addLayout(left_layout)
        layout.addLayout(right_layout)
        
        self.tab_widget.addTab(tab, "人脸录入")
    
    def create_data_management_tab(self):
        """创建数据管理标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 用户列表
        self.user_list = QListWidget()
        self.user_list.setFixedHeight(300)
        self.user_list.itemClicked.connect(self.on_user_selected)
        layout.addWidget(QLabel("用户列表:"))
        layout.addWidget(self.user_list)
        
        # 用户详情
        detail_layout = QHBoxLayout()
        
        # 左侧：用户信息
        info_layout = QVBoxLayout()
        self.selected_user_info = QTextEdit()
        self.selected_user_info.setReadOnly(True)
        self.selected_user_info.setFixedHeight(150)
        info_layout.addWidget(QLabel("用户信息:"))
        info_layout.addWidget(self.selected_user_info)
        
        # 右侧：人脸特征可视化
        feature_layout = QVBoxLayout()
        self.feature_visualization = QLabel()
        self.feature_visualization.setFixedSize(200, 200)
        self.feature_visualization.setStyleSheet("background-color: #ecf0f1;")
        self.feature_visualization.setAlignment(Qt.AlignCenter)
        feature_layout.addWidget(QLabel("人脸特征:"))
        feature_layout.addWidget(self.feature_visualization)
        
        detail_layout.addLayout(info_layout)
        detail_layout.addLayout(feature_layout)
        layout.addLayout(detail_layout)
        
        # 操作按钮
        button_layout = QHBoxLayout()
        self.update_user_btn = QPushButton("更新选中用户")
        self.update_user_btn.clicked.connect(self.update_user)
        self.update_user_btn.setEnabled(False)
        button_layout.addWidget(self.update_user_btn)
        
        self.delete_user_btn = QPushButton("删除选中用户")
        self.delete_user_btn.clicked.connect(self.delete_user)
        self.delete_user_btn.setEnabled(False)
        button_layout.addWidget(self.delete_user_btn)
        
        self.export_data_btn = QPushButton("导出用户数据")
        self.export_data_btn.clicked.connect(self.export_user_data)
        button_layout.addWidget(self.export_data_btn)
        
        self.refresh_list_btn = QPushButton("刷新列表")
        self.refresh_list_btn.clicked.connect(self.refresh_user_list)
        button_layout.addWidget(self.refresh_list_btn)
        
        layout.addLayout(button_layout)
        
        self.tab_widget.addTab(tab, "数据管理")
        
        # 初始化用户列表
        self.refresh_user_list()
    
    def create_attendance_tab(self):
        """创建考勤管理标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 考勤控制
        control_layout = QHBoxLayout()
        
        self.start_attendance_btn = QPushButton("开始考勤")
        self.start_attendance_btn.clicked.connect(self.start_attendance)
        control_layout.addWidget(self.start_attendance_btn)
        
        self.stop_attendance_btn = QPushButton("停止考勤")
        self.stop_attendance_btn.clicked.connect(self.stop_attendance)
        self.stop_attendance_btn.setEnabled(False)
        control_layout.addWidget(self.stop_attendance_btn)
        
        self.generate_report_btn = QPushButton("生成考勤报表")
        self.generate_report_btn.clicked.connect(self.generate_attendance_report)
        control_layout.addWidget(self.generate_report_btn)
        
        self.export_attendance_btn = QPushButton("导出考勤数据")
        self.export_attendance_btn.clicked.connect(self.export_attendance_data)
        control_layout.addWidget(self.export_attendance_btn)
        
        layout.addLayout(control_layout)
        
        # 考勤状态显示
        self.attendance_status = QLabel("考勤系统未启动")
        self.attendance_status.setStyleSheet("font-size: 14px; color: #7f8c8d; margin: 10px 0;")
        layout.addWidget(self.attendance_status)
        
        # 考勤摄像头显示 - 新增：自动打开摄像头界面
        self.attendance_camera_display = QLabel()
        self.attendance_camera_display.setFixedSize(640, 480)
        self.attendance_camera_display.setStyleSheet("background-color: #2c3e50;")
        self.attendance_camera_display.setAlignment(Qt.AlignCenter)
        self.attendance_camera_display.setVisible(False)  # 默认隐藏
        layout.addWidget(self.attendance_camera_display)
        
        # 考勤记录表格
        self.attendance_table = QTableWidget()
        self.attendance_table.setColumnCount(6)
        self.attendance_table.setHorizontalHeaderLabels(["姓名", "签到时间", "签退时间", "状态", "位置", "温度"])
        self.attendance_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(QLabel("考勤记录:"))
        layout.addWidget(self.attendance_table)
        
        self.tab_widget.addTab(tab, "考勤管理")
        
        # 初始化考勤记录
        self.refresh_attendance_table()
    
    def create_settings_tab(self):
        """创建系统设置标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 识别参数设置
        recognition_group = QGroupBox("识别参数设置")
        recognition_layout = QFormLayout(recognition_group)
        
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(self.config["threshold"])
        recognition_layout.addRow("识别阈值:", self.threshold_spin)
        
        self.max_faces_spin = QSpinBox()
        self.max_faces_spin.setRange(10, 1000)
        self.max_faces_spin.setValue(self.config["max_faces"])
        recognition_layout.addRow("最大人脸数量:", self.max_faces_spin)
        
        self.camera_index_spin = QSpinBox()
        self.camera_index_spin.setRange(0, 10)
        self.camera_index_spin.setValue(self.config["camera_index"])
        recognition_layout.addRow("摄像头索引:", self.camera_index_spin)
        
        layout.addWidget(recognition_group)
        
        # MySQL数据库设置
        mysql_group = QGroupBox("MySQL数据库设置")
        mysql_layout = QFormLayout(mysql_group)
        
        self.mysql_host_edit = QLineEdit(self.config["mysql_host"])
        mysql_layout.addRow("数据库主机:", self.mysql_host_edit)
        
        self.mysql_db_edit = QLineEdit(self.config["mysql_database"])
        mysql_layout.addRow("数据库名:", self.mysql_db_edit)
        
        self.mysql_user_edit = QLineEdit(self.config["mysql_user"])
        mysql_layout.addRow("用户名:", self.mysql_user_edit)
        
        self.mysql_password_edit = QLineEdit(self.config["mysql_password"])
        self.mysql_password_edit.setEchoMode(QLineEdit.Password)
        mysql_layout.addRow("密码:", self.mysql_password_edit)
        
        test_mysql_btn = QPushButton("测试数据库连接")
        test_mysql_btn.clicked.connect(self.test_mysql_connection)
        mysql_layout.addRow(test_mysql_btn)
        
        layout.addWidget(mysql_group)
        
        # 模型设置
        model_group = QGroupBox("模型设置")
        model_layout = QFormLayout(model_group)
        
        self.shape_predictor_edit = QLineEdit(self.config["shape_predictor_path"])
        model_layout.addRow("特征点预测器路径:", self.shape_predictor_edit)
        
        self.face_recognizer_edit = QLineEdit(self.config["face_recognition_model_path"])
        model_layout.addRow("人脸识别模型路径:", self.face_recognizer_edit)
        
        self.use_local_models_check = QCheckBox("仅使用本地模型")
        self.use_local_models_check.setChecked(self.config["use_local_models_only"])
        model_layout.addRow(self.use_local_models_check)
        
        layout.addWidget(model_group)
        
        # 保存按钮
        save_btn = QPushButton("保存设置")
        save_btn.clicked.connect(self.save_settings)
        layout.addWidget(save_btn)
        
        self.tab_widget.addTab(tab, "系统设置")
    
    # 摄像头相关方法
    def start_camera(self):
        """启动摄像头"""
        try:
            self.camera = QCamera()
            self.camera.setCaptureMode(QCamera.CaptureStillImage)
            
            self.image_capture = QCameraImageCapture(self.camera)
            self.image_capture.imageCaptured.connect(self.process_camera_image)
            
            # 设置摄像头视图
            self.camera_viewfinder = QCameraViewfinder()
            self.camera_viewfinder.setFixedSize(640, 480)
            
            # 将视图转换为QPixmap显示在QLabel上
            self.camera.setViewfinder(self.camera_viewfinder)
            self.camera.start()
            
            self.is_camera_running = True
            self.start_camera_btn.setEnabled(False)
            self.stop_camera_btn.setEnabled(True)
            self.start_recognition_btn.setEnabled(True)
            
            self.status_bar.showMessage("摄像头启动成功")
            
        except Exception as e:
            print(f"启动摄像头失败: {e}")
            QMessageBox.critical(self, "摄像头错误", f"启动摄像头失败: {e}")
    
    def stop_camera(self):
        """停止摄像头"""
        if self.camera:
            self.camera.stop()
            self.is_camera_running = False
            self.is_recognizing = False
            
            if self.capture_timer:
                self.capture_timer.stop()
            
            self.start_camera_btn.setEnabled(True)
            self.stop_camera_btn.setEnabled(False)
            self.start_recognition_btn.setEnabled(False)
            self.stop_recognition_btn.setEnabled(False)
            
            self.camera_display.clear()
            self.camera_display.setText("摄像头已关闭")
            self.status_bar.showMessage("摄像头已停止")
    
    def start_recognition(self):
        """开始人脸识别"""
        if not self.is_camera_running:
            QMessageBox.warning(self, "警告", "请先启动摄像头")
            return
        
        self.is_recognizing = True
        self.start_recognition_btn.setEnabled(False)
        self.stop_recognition_btn.setEnabled(True)
        
        # 创建定时器进行定时识别
        self.capture_timer = QTimer()
        self.capture_timer.timeout.connect(self.capture_and_recognize)
        self.capture_timer.start(1000)  # 每秒识别一次
        
        self.status_bar.showMessage("人脸识别中...")
    
    def stop_recognition(self):
        """停止人脸识别"""
        self.is_recognizing = False
        if self.capture_timer:
            self.capture_timer.stop()
        
        self.start_recognition_btn.setEnabled(True)
        self.stop_recognition_btn.setEnabled(False)
        self.status_bar.showMessage("人脸识别已停止")
    
    def capture_and_recognize(self):
        """捕获并识别人脸"""
        if self.image_capture:
            self.image_capture.capture()
    
    def process_camera_image(self, request_id, image):
        """处理摄像头捕获的图像"""
        # 显示图像
        pixmap = QPixmap.fromImage(image)
        self.camera_display.setPixmap(pixmap.scaled(self.camera_display.size(), Qt.KeepAspectRatio))
        
        # 如果正在识别，进行人脸识别
        if self.is_recognizing:
            self.recognize_face_from_image(image)
    
    def recognize_face_from_image(self, image):
        """从图像中识别人脸"""
        try:
            # 转换为PIL图像
            pil_image = ImageQt.fromqimage(image).convert('RGB')
            numpy_image = np.array(pil_image)
            
            # 检测人脸
            faces = self.face_detector(numpy_image)
            
            if len(faces) == 0:
                self.recognition_result.append(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 未检测到人脸")
                return
            
            for face in faces:
                # 提取人脸特征
                shape = self.shape_predictor(numpy_image, face)
                face_descriptor = self.face_recognizer.compute_face_descriptor(numpy_image, shape)
                face_encoding = np.array(face_descriptor)
                
                # 对比已知人脸
                min_distance = float('inf')
                recognized_name = "未知人脸"
                recognized_id = None
                
                for user_id, user_data in self.face_features.items():
                    distance = np.linalg.norm(face_encoding - user_data["encoding"])
                    if distance < min_distance and distance < self.config["threshold"]:
                        min_distance = distance
                        recognized_name = user_data["name"]
                        recognized_id = user_id
                
                # 记录识别结果
                result_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                result_text = f"{result_time} - 识别到: {recognized_name} (置信度: {1-min_distance:.2f})"
                self.recognition_result.append(result_text)
                self.recognition_result.verticalScrollBar().setValue(self.recognition_result.verticalScrollBar().maximum())
                
                # 保存识别日志到数据库
                self.save_recognition_log(recognized_id, min_distance)
                
                # 如果是已知用户，更新考勤状态
                if recognized_id:
                    self.update_attendance_status(recognized_id)
                
        except Exception as e:
            print(f"人脸识别失败: {e}")
            self.recognition_result.append(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 识别失败: {str(e)}")
    
    # 人脸录入相关方法
    def start_enrollment_camera(self):
        """启动录入摄像头 - 修复预览问题"""
        try:
            # 创建专门用于录入的摄像头实例
            self.enrollment_camera = QCamera()
            self.enrollment_camera.setCaptureMode(QCamera.CaptureStillImage)
            
            self.enrollment_image_capture = QCameraImageCapture(self.enrollment_camera)
            self.enrollment_image_capture.imageCaptured.connect(self.process_enrollment_camera_image)
            
            # 设置摄像头视图
            self.enrollment_camera_viewfinder = QCameraViewfinder()
            self.enrollment_camera_viewfinder.setFixedSize(320, 240)
            
            # 启动摄像头
            self.enrollment_camera.start()
            
            # 创建定时器来更新预览
            self.enrollment_preview_timer = QTimer()
            self.enrollment_preview_timer.timeout.connect(self.update_enrollment_preview)
            self.enrollment_preview_timer.start(30)  # 30ms更新一次
            
            self.start_enrollment_camera_btn.setEnabled(False)
            self.capture_face_btn.setEnabled(True)
            self.stop_enrollment_camera_btn.setEnabled(True)
            
            self.feature_status.setText("摄像头启动成功，请面对摄像头")
            self.enrollment_message.append(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 录入摄像头启动成功")
            
        except Exception as e:
            print(f"启动录入摄像头失败: {e}")
            QMessageBox.critical(self, "摄像头错误", f"启动录入摄像头失败: {e}")
            self.feature_status.setText(f"摄像头启动失败: {str(e)}")
    
    def update_enrollment_preview(self):
        """更新录入摄像头预览 - 修复预览问题"""
        if hasattr(self, 'enrollment_camera') and self.enrollment_camera and self.enrollment_camera.state() == QCamera.ActiveState:
            # 使用grab()方法获取当前帧
            viewfinder_image = self.enrollment_camera_viewfinder.grab()
            if not viewfinder_image.isNull():
                self.enrollment_camera_display.setPixmap(
                    viewfinder_image.scaled(self.enrollment_camera_display.size(), Qt.KeepAspectRatio)
                )
    
    def process_enrollment_camera_image(self, request_id, image):
        """处理录入摄像头捕获的图像"""
        self.captured_enrollment_image = image
    
    def capture_face(self):
        """捕获人脸"""
        if hasattr(self, 'enrollment_image_capture') and self.enrollment_image_capture:
            self.enrollment_image_capture.capture()
            
            # 显示捕获的人脸
            if hasattr(self, 'captured_enrollment_image'):
                pixmap = QPixmap.fromImage(self.captured_enrollment_image)
                self.face_preview.setPixmap(pixmap.scaled(self.face_preview.size(), Qt.KeepAspectRatio))
                
                # 提取人脸特征
                self.extract_face_features(self.captured_enrollment_image)
    
    def stop_enrollment_camera(self):
        """停止录入摄像头"""
        if hasattr(self, 'enrollment_camera') and self.enrollment_camera:
            self.enrollment_camera.stop()
            
            if hasattr(self, 'enrollment_preview_timer') and self.enrollment_preview_timer:
                self.enrollment_preview_timer.stop()
            
            self.start_enrollment_camera_btn.setEnabled(True)
            self.capture_face_btn.setEnabled(False)
            self.stop_enrollment_camera_btn.setEnabled(False)
            
            self.enrollment_camera_display.clear()
            self.enrollment_camera_display.setText("摄像头已关闭")
            self.feature_status.setText("摄像头已关闭")
            self.enrollment_message.append(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 录入摄像头已停止")
    
    def extract_face_features(self, image):
        """提取人脸特征"""
        try:
            self.enrollment_progress.setVisible(True)
            self.enrollment_progress.setValue(0)
            
            # 转换为PIL图像
            pil_image = ImageQt.fromqimage(image).convert('RGB')
            numpy_image = np.array(pil_image)
            
            self.enrollment_progress.setValue(30)
            
            # 检测人脸
            faces = self.face_detector(numpy_image)
            
            if len(faces) == 0:
                self.feature_status.setText("未检测到人脸，请重新捕获")
                self.save_user_btn.setEnabled(False)
                self.enrollment_progress.setVisible(False)
                return
            
            if len(faces) > 1:
                self.feature_status.setText("检测到多张人脸，请确保只有一张人脸在画面中")
                self.save_user_btn.setEnabled(False)
                self.enrollment_progress.setVisible(False)
                return
            
            self.enrollment_progress.setValue(60)
            
            # 提取人脸特征
            face = faces[0]
            shape = self.shape_predictor(numpy_image, face)
            face_descriptor = self.face_recognizer.compute_face_descriptor(numpy_image, shape)
            self.current_face_encoding = np.array(face_descriptor)
            
            self.enrollment_progress.setValue(100)
            
            self.feature_status.setText("人脸特征提取成功，可以保存用户信息")
            self.save_user_btn.setEnabled(True)
            self.enrollment_message.append(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 人脸特征提取成功")
            
            self.enrollment_progress.setVisible(False)
            
        except Exception as e:
            print(f"提取人脸特征失败: {e}")
            self.feature_status.setText(f"特征提取失败: {str(e)}")
            self.save_user_btn.setEnabled(False)
            self.enrollment_progress.setVisible(False)
    
    def save_user(self):
        """保存用户信息到MySQL数据库"""
        if not hasattr(self, 'current_face_encoding'):
            QMessageBox.warning(self, "警告", "请先捕获人脸并提取特征")
            return
        
        name = self.name_edit.text().strip()
        age = self.age_spin.value()
        gender = self.gender_edit.text().strip()
        department = self.department_edit.text().strip()
        
        if not name:
            QMessageBox.warning(self, "警告", "请输入姓名")
            return
        
        try:
            if not self.db_connection or not self.db_connection.is_connected():
                QMessageBox.critical(self, "数据库错误", "数据库连接已断开")
                return
            
            cursor = self.db_connection.cursor()
            
            # 检查用户是否已存在
            cursor.execute("SELECT id FROM users WHERE name = %s", (name,))
            existing_user = cursor.fetchone()
            
            if existing_user:
                # 更新现有用户
                cursor.execute('''
                    UPDATE users 
                    SET age = %s, gender = %s, department = %s, face_encoding = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE name = %s
                ''', (age, gender, department, json.dumps(self.current_face_encoding.tolist()), name))
                message = f"用户 {name} 更新成功"
            else:
                # 创建新用户
                cursor.execute('''
                    INSERT INTO users (name, age, gender, department, face_encoding)
                    VALUES (%s, %s, %s, %s, %s)
                ''', (name, age, gender, department, json.dumps(self.current_face_encoding.tolist())))
                message = f"用户 {name} 创建成功"
            
            self.db_connection.commit()
            
            # 更新本地人脸特征缓存
            self.load_face_features()
            
            # 刷新用户列表
            self.refresh_user_list()
            
            QMessageBox.information(self, "成功", message)
            self.enrollment_message.append(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")
            
            # 清空表单
            self.name_edit.clear()
            self.age_spin.setValue(25)
            self.gender_edit.clear()
            self.department_edit.clear()
            self.face_preview.clear()
            if hasattr(self, 'current_face_encoding'):
                delattr(self, 'current_face_encoding')
            self.save_user_btn.setEnabled(False)
            self.feature_status.setText("请重新录入用户信息")
            
        except Error as e:
            print(f"保存用户失败: {e}")
            QMessageBox.critical(self, "数据库错误", f"保存用户失败: {e}")
            self.db_connection.rollback()
    
    # 考勤管理相关方法
    def start_attendance(self):
        """开始考勤 - 自动打开摄像头界面"""
        try:
            self.attendance_status.setText("考勤系统启动中...")
            
            # 自动启动考勤摄像头
            self.attendance_camera = QCamera()
            self.attendance_camera.setCaptureMode(QCamera.CaptureStillImage)
            
            self.attendance_image_capture = QCameraImageCapture(self.attendance_camera)
            self.attendance_image_capture.imageCaptured.connect(self.process_attendance_camera_image)
            
            # 设置摄像头视图
            self.attendance_camera_viewfinder = QCameraViewfinder()
            self.attendance_camera_viewfinder.setFixedSize(640, 480)
            
            # 启动摄像头
            self.attendance_camera.start()
            
            # 创建预览定时器
            self.attendance_preview_timer = QTimer()
            self.attendance_preview_timer.timeout.connect(self.update_attendance_preview)
            self.attendance_preview_timer.start(30)
            
            # 创建识别定时器
            self.attendance_recognition_timer = QTimer()
            self.attendance_recognition_timer.timeout.connect(self.capture_and_recognize_attendance)
            self.attendance_recognition_timer.start(2000)  # 每2秒识别一次
            
            self.is_attendance_running = True
            self.start_attendance_btn.setEnabled(False)
            self.stop_attendance_btn.setEnabled(True)
            
            # 显示摄像头界面
            self.attendance_camera_display.setVisible(True)
            self.attendance_status.setText("考勤系统运行中...正在识别人脸")
            
            if hasattr(self, 'enrollment_message'):
                self.enrollment_message.append(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 考勤系统启动成功")
            
        except Exception as e:
            print(f"启动考勤系统失败: {e}")
            QMessageBox.critical(self, "考勤错误", f"启动考勤系统失败: {e}")
            self.attendance_status.setText(f"考勤系统启动失败: {str(e)}")
    
    def update_attendance_preview(self):
        """更新考勤摄像头预览"""
        if hasattr(self, 'attendance_camera') and self.attendance_camera and self.attendance_camera.state() == QCamera.ActiveState:
            viewfinder_image = self.attendance_camera_viewfinder.grab()
            if not viewfinder_image.isNull():
                self.attendance_camera_display.setPixmap(
                    viewfinder_image.scaled(self.attendance_camera_display.size(), Qt.KeepAspectRatio)
                )
    
    def process_attendance_camera_image(self, request_id, image):
        """处理考勤摄像头图像"""
        self.captured_attendance_image = image
    
    def capture_and_recognize_attendance(self):
        """考勤人脸识别"""
        if hasattr(self, 'attendance_image_capture') and self.attendance_image_capture:
            self.attendance_image_capture.capture()
            
            if hasattr(self, 'captured_attendance_image'):
                self.recognize_face_for_attendance(self.captured_attendance_image)
    
    def recognize_face_for_attendance(self, image):
        """为考勤识别人脸"""
        try:
            # 转换为PIL图像
            pil_image = ImageQt.fromqimage(image).convert('RGB')
            numpy_image = np.array(pil_image)
            
            # 检测人脸
            faces = self.face_detector(numpy_image)
            
            if len(faces) == 0:
                return
            
            for face in faces:
                # 提取人脸特征
                shape = self.shape_predictor(numpy_image, face)
                face_descriptor = self.face_recognizer.compute_face_descriptor(numpy_image, shape)
                face_encoding = np.array(face_descriptor)
                
                # 对比已知人脸
                min_distance = float('inf')
                recognized_name = "未知人脸"
                recognized_id = None
                
                for user_id, user_data in self.face_features.items():
                    distance = np.linalg.norm(face_encoding - user_data["encoding"])
                    if distance < min_distance and distance < self.config["threshold"]:
                        min_distance = distance
                        recognized_name = user_data["name"]
                        recognized_id = user_id
                
                # 如果识别到已知用户，记录考勤
                if recognized_id:
                    self.record_attendance(recognized_id, recognized_name)
                
        except Exception as e:
            print(f"考勤人脸识别失败: {e}")
    
    def record_attendance(self, user_id, user_name):
        """记录考勤"""
        try:
            if not self.db_connection or not self.db_connection.is_connected():
                return
            
            cursor = self.db_connection.cursor(dictionary=True)
            
            # 检查今天是否已经签到
            today = datetime.date.today()
            cursor.execute('''
                SELECT * FROM attendance 
                WHERE user_id = %s AND DATE(check_in_time) = %s
                ORDER BY check_in_time DESC LIMIT 1
            ''', (user_id, today))
            
            last_attendance = cursor.fetchone()
            
            current_time = datetime.datetime.now()
            
            if not last_attendance:
                # 新签到
                cursor.execute('''
                    INSERT INTO attendance (user_id, check_in_time, status, location)
                    VALUES (%s, %s, %s, %s)
                ''', (user_id, current_time, 'present', '办公室'))
                
                message = f"{user_name} 签到成功"
                self.attendance_status.setText(f"{datetime.datetime.now().strftime('%H:%M:%S')} - {message}")
                
            elif not last_attendance['check_out_time'] and (current_time - last_attendance['check_in_time']).total_seconds() > 3600:
                # 签退（至少间隔1小时）
                cursor.execute('''
                    UPDATE attendance 
                    SET check_out_time = %s, status = %s
                    WHERE id = %s
                ''', (current_time, 'completed', last_attendance['id']))
                
                message = f"{user_name} 签退成功"
                self.attendance_status.setText(f"{datetime.datetime.now().strftime('%H:%M:%S')} - {message}")
            
            else:
                # 已经签到且未到签退时间
                return
            
            self.db_connection.commit()
            self.refresh_attendance_table()
            if hasattr(self, 'enrollment_message'):
                self.enrollment_message.append(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")
            
        except Error as e:
            print(f"记录考勤失败: {e}")
            self.db_connection.rollback()
    
    def stop_attendance(self):
        """停止考勤"""
        if hasattr(self, 'attendance_camera') and self.attendance_camera:
            self.attendance_camera.stop()
            
        if hasattr(self, 'attendance_preview_timer') and self.attendance_preview_timer:
            self.attendance_preview_timer.stop()
            
        if hasattr(self, 'attendance_recognition_timer') and self.attendance_recognition_timer:
            self.attendance_recognition_timer.stop()
            
        self.is_attendance_running = False
        self.start_attendance_btn.setEnabled(True)
        self.stop_attendance_btn.setEnabled(False)
        
        # 隐藏摄像头界面
        self.attendance_camera_display.setVisible(False)
        self.attendance_status.setText("考勤系统已停止")
        
        if hasattr(self, 'enrollment_message'):
            self.enrollment_message.append(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 考勤系统已停止")
    
    # 数据库操作方法
    def save_recognition_log(self, user_id, confidence):
        """保存识别日志"""
        try:
            if not self.db_connection or not self.db_connection.is_connected():
                return
            
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO recognition_logs (user_id, confidence)
                VALUES (%s, %s)
            ''', (user_id, 1 - confidence))
            
            self.db_connection.commit()
            
        except Error as e:
            print(f"保存识别日志失败: {e}")
            self.db_connection.rollback()
    
    def update_attendance_status(self, user_id):
        """更新考勤状态"""
        try:
            if not self.db_connection or not self.db_connection.is_connected():
                return
            
            cursor = self.db_connection.cursor(dictionary=True)
            
            # 检查今天是否已经签到
            today = datetime.date.today()
            cursor.execute('''
                SELECT * FROM attendance 
                WHERE user_id = %s AND DATE(check_in_time) = %s
                ORDER BY check_in_time DESC LIMIT 1
            ''', (user_id, today))
            
            last_attendance = cursor.fetchone()
            
            if not last_attendance:
                # 自动签到
                current_time = datetime.datetime.now()
                cursor.execute('''
                    INSERT INTO attendance (user_id, check_in_time, status, location)
                    VALUES (%s, %s, %s, %s)
                ''', (user_id, current_time, 'present', '办公室'))
                
                self.db_connection.commit()
                self.refresh_attendance_table()
                
        except Error as e:
            print(f"更新考勤状态失败: {e}")
            self.db_connection.rollback()
    
    # UI刷新方法
    def refresh_user_list(self):
        """刷新用户列表"""
        self.user_list.clear()
        if not self.db_connection or not self.db_connection.is_connected():
            return
        
        try:
            cursor = self.db_connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users ORDER BY name")
            
            for user in cursor.fetchall():
                item = QListWidgetItem(f"{user['name']} - {user['department']}")
                item.setData(Qt.UserRole, user['id'])
                self.user_list.addItem(item)
                
        except Error as e:
            print(f"刷新用户列表失败: {e}")
    
    def refresh_attendance_table(self):
        """刷新考勤表格"""
        self.attendance_table.setRowCount(0)
        self.load_attendance_records()
        
        for record in self.attendance_records:
            row_position = self.attendance_table.rowCount()
            self.attendance_table.insertRow(row_position)
            
            self.attendance_table.setItem(row_position, 0, QTableWidgetItem(record['name'] or '未知用户'))
            self.attendance_table.setItem(row_position, 1, QTableWidgetItem(record['check_in_time'].strftime('%Y-%m-%d %H:%M:%S')))
            self.attendance_table.setItem(row_position, 2, QTableWidgetItem(record['check_out_time'].strftime('%Y-%m-%d %H:%M:%S') if record['check_out_time'] else ''))
            self.attendance_table.setItem(row_position, 3, QTableWidgetItem(record['status']))
            self.attendance_table.setItem(row_position, 4, QTableWidgetItem(record['location'] or ''))
            self.attendance_table.setItem(row_position, 5, QTableWidgetItem(str(record['temperature']) if record['temperature'] else ''))
    
    def update_status(self):
        """更新系统状态"""
        model_status = "正常" if self.face_detector and self.shape_predictor and self.face_recognizer else "异常"
        db_status = "已连接" if (self.db_connection and self.db_connection.is_connected()) else "未连接"
        user_count = len(self.face_features)
        
        status_text = f"模型状态: {model_status} | 数据库: {db_status} | 用户数量: {user_count}"
        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage(status_text)
    
    # 其他辅助方法
    def browse_image(self):
        """浏览图片文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片文件", "", "Image Files (*.jpg *.jpeg *.png *.bmp)")
        if file_path:
            self.image_path_edit.setText(file_path)
    
    def browse_enrollment_image(self):
        """浏览录入图片文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择人脸图片", "", "Image Files (*.jpg *.jpeg *.png *.bmp)")
        if file_path:
            self.enrollment_image_path.setText(file_path)
    
    def recognize_image(self):
        """识别图片中的人脸"""
        file_path = self.image_path_edit.text().strip()
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "警告", "请选择有效的图片文件")
            return
        
        try:
            # 加载图片
            image = Image.open(file_path).convert('RGB')
            numpy_image = np.array(image)
            
            # 显示图片
            qimage = ImageQt.ImageQt(image)
            pixmap = QPixmap.fromImage(qimage)
            self.camera_display.setPixmap(pixmap.scaled(self.camera_display.size(), Qt.KeepAspectRatio))
            
            # 识别人脸
            faces = self.face_detector(numpy_image)
            
            if len(faces) == 0:
                self.recognition_result.append(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 图片中未检测到人脸")
                return
            
            self.recognition_result.append(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 图片中检测到 {len(faces)} 张人脸")
            
            for i, face in enumerate(faces):
                # 提取人脸特征
                shape = self.shape_predictor(numpy_image, face)
                face_descriptor = self.face_recognizer.compute_face_descriptor(numpy_image, shape)
                face_encoding = np.array(face_descriptor)
                
                # 对比已知人脸
                min_distance = float('inf')
                recognized_name = "未知人脸"
                recognized_id = None
                
                for user_id, user_data in self.face_features.items():
                    distance = np.linalg.norm(face_encoding - user_data["encoding"])
                    if distance < min_distance and distance < self.config["threshold"]:
                        min_distance = distance
                        recognized_name = user_data["name"]
                        recognized_id = user_id
                
                result_text = f"  人脸 {i+1}: {recognized_name} (置信度: {1-min_distance:.2f})"
                self.recognition_result.append(result_text)
                
        except Exception as e:
            print(f"图片识别失败: {e}")
            QMessageBox.critical(self, "识别错误", f"图片识别失败: {e}")
    
    def process_enrollment_image(self):
        """处理录入图片"""
        file_path = self.enrollment_image_path.text().strip()
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "警告", "请选择有效的图片文件")
            return
        
        try:
            # 加载图片
            image = Image.open(file_path).convert('RGB')
            qimage = ImageQt.ImageQt(image)
            
            # 显示图片预览
            pixmap = QPixmap.fromImage(qimage)
            self.face_preview.setPixmap(pixmap.scaled(self.face_preview.size(), Qt.KeepAspectRatio))
            
            # 提取人脸特征
            numpy_image = np.array(image)
            self.extract_face_features_from_numpy(numpy_image)
            
        except Exception as e:
            print(f"处理录入图片失败: {e}")
            QMessageBox.critical(self, "处理错误", f"处理录入图片失败: {e}")
    
    def extract_face_features_from_numpy(self, numpy_image):
        """从numpy数组提取人脸特征"""
        try:
            self.enrollment_progress.setVisible(True)
            self.enrollment_progress.setValue(0)
            
            # 检测人脸
            faces = self.face_detector(numpy_image)
            self.enrollment_progress.setValue(30)
            
            if len(faces) == 0:
                self.feature_status.setText("图片中未检测到人脸")
                self.save_user_btn.setEnabled(False)
                self.enrollment_progress.setVisible(False)
                return
            
            if len(faces) > 1:
                self.feature_status.setText("图片中检测到多张人脸，请选择只包含一张人脸的图片")
                self.save_user_btn.setEnabled(False)
                self.enrollment_progress.setVisible(False)
                return
            
            self.enrollment_progress.setValue(60)
            
            # 提取人脸特征
            face = faces[0]
            shape = self.shape_predictor(numpy_image, face)
            face_descriptor = self.face_recognizer.compute_face_descriptor(numpy_image, shape)
            self.current_face_encoding = np.array(face_descriptor)
            
            self.enrollment_progress.setValue(100)
            
            self.feature_status.setText("人脸特征提取成功，可以保存用户信息")
            self.save_user_btn.setEnabled(True)
            self.enrollment_message.append(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 从图片提取人脸特征成功")
            
            self.enrollment_progress.setVisible(False)
            
        except Exception as e:
            print(f"从图片提取人脸特征失败: {e}")
            self.feature_status.setText(f"特征提取失败: {str(e)}")
            self.save_user_btn.setEnabled(False)
            self.enrollment_progress.setVisible(False)
    
    def on_user_selected(self, item):
        """用户选择事件"""
        user_id = item.data(Qt.UserRole)
        if user_id in self.face_features:
            user_data = self.face_features[user_id]
            info_text = f"姓名: {user_data['name']}\n"
            
            # 从数据库获取更多信息
            if self.db_connection and self.db_connection.is_connected():
                try:
                    cursor = self.db_connection.cursor(dictionary=True)
                    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
                    user_info = cursor.fetchone()
                    
                    if user_info:
                        info_text += f"年龄: {user_info['age']}\n"
                        info_text += f"性别: {user_info['gender']}\n"
                        info_text += f"部门: {user_info['department']}\n"
                        info_text += f"创建时间: {user_info['created_at'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                        info_text += f"更新时间: {user_info['updated_at'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                        
                except Error as e:
                    print(f"获取用户信息失败: {e}")
            
            self.selected_user_info.setText(info_text)
            
            # 启用编辑按钮
            self.update_user_btn.setEnabled(True)
            self.delete_user_btn.setEnabled(True)
    
    def update_user(self):
        """更新用户信息"""
        selected_item = self.user_list.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "警告", "请先选择要更新的用户")
            return
        
        user_id = selected_item.data(Qt.UserRole)
        # 这里可以实现用户信息更新功能
        QMessageBox.information(self, "提示", "用户更新功能将在未来版本中实现")
    
    def delete_user(self):
        """删除用户"""
        selected_item = self.user_list.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "警告", "请先选择要删除的用户")
            return
        
        user_id = selected_item.data(Qt.UserRole)
        user_name = selected_item.text().split(" - ")[0]
        
        reply = QMessageBox.question(self, "确认删除", f"确定要删除用户 {user_name} 吗？", 
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            try:
                if not self.db_connection or not self.db_connection.is_connected():
                    QMessageBox.critical(self, "数据库错误", "数据库连接已断开")
                    return
                
                cursor = self.db_connection.cursor()
                cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
                self.db_connection.commit()
                
                # 更新本地缓存
                if user_id in self.face_features:
                    del self.face_features[user_id]
                
                # 刷新列表
                self.refresh_user_list()
                self.selected_user_info.clear()
                self.update_user_btn.setEnabled(False)
                self.delete_user_btn.setEnabled(False)
                
                QMessageBox.information(self, "成功", f"用户 {user_name} 删除成功")
                
            except Error as e:
                print(f"删除用户失败: {e}")
                QMessageBox.critical(self, "数据库错误", f"删除用户失败: {e}")
                self.db_connection.rollback()
    
    def export_user_data(self):
        """导出用户数据"""
        file_path, _ = QFileDialog.getSaveFileName(self, "导出用户数据", "user_data.csv", "CSV Files (*.csv)")
        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['ID', '姓名', '年龄', '性别', '部门', '创建时间'])
                    
                    if self.db_connection and self.db_connection.is_connected():
                        cursor = self.db_connection.cursor(dictionary=True)
                        cursor.execute("SELECT * FROM users")
                        
                        for user in cursor.fetchall():
                            writer.writerow([
                                user['id'],
                                user['name'],
                                user['age'],
                                user['gender'],
                                user['department'],
                                user['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                            ])
                
                QMessageBox.information(self, "成功", f"用户数据已导出到 {file_path}")
                
            except Exception as e:
                print(f"导出用户数据失败: {e}")
                QMessageBox.critical(self, "导出错误", f"导出用户数据失败: {e}")
    
    def generate_attendance_report(self):
        """生成考勤报表"""
        # 这里可以实现考勤报表生成功能
        QMessageBox.information(self, "提示", "考勤报表生成功能将在未来版本中实现")
    
    def export_attendance_data(self):
        """导出考勤数据"""
        file_path, _ = QFileDialog.getSaveFileName(self, "导出考勤数据", "attendance_data.csv", "CSV Files (*.csv)")
        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['ID', '姓名', '签到时间', '签退时间', '状态', '位置'])
                    
                    for record in self.attendance_records:
                        writer.writerow([
                            record['id'],
                            record['name'] or '未知用户',
                            record['check_in_time'].strftime('%Y-%m-%d %H:%M:%S'),
                            record['check_out_time'].strftime('%Y-%m-%d %H:%M:%S') if record['check_out_time'] else '',
                            record['status'],
                            record['location'] or ''
                        ])
                
                QMessageBox.information(self, "成功", f"考勤数据已导出到 {file_path}")
                
            except Exception as e:
                print(f"导出考勤数据失败: {e}")
                QMessageBox.critical(self, "导出错误", f"导出考勤数据失败: {e}")
    
    def test_mysql_connection(self):
        """测试MySQL连接"""
        host = self.mysql_host_edit.text().strip()
        database = self.mysql_db_edit.text().strip()
        user = self.mysql_user_edit.text().strip()
        password = self.mysql_password_edit.text().strip()
        
        try:
            connection = mysql.connector.connect(
                host=host,
                database=database,
                user=user,
                password=password
            )
            
            if connection.is_connected():
                QMessageBox.information(self, "测试成功", "MySQL数据库连接成功")
                connection.close()
            else:
                QMessageBox.warning(self, "测试失败", "MySQL数据库连接失败")
                
        except Error as e:
            QMessageBox.critical(self, "连接错误", f"MySQL连接失败: {e}")
    
    def save_settings(self):
        """保存系统设置"""
        # 更新配置
        self.config["threshold"] = self.threshold_spin.value()
        self.config["max_faces"] = self.max_faces_spin.value()
        self.config["camera_index"] = self.camera_index_spin.value()
        self.config["mysql_host"] = self.mysql_host_edit.text().strip()
        self.config["mysql_database"] = self.mysql_db_edit.text().strip()
        self.config["mysql_user"] = self.mysql_user_edit.text().strip()
        self.config["mysql_password"] = self.mysql_password_edit.text().strip()
        self.config["shape_predictor_path"] = self.shape_predictor_edit.text().strip()
        self.config["face_recognition_model_path"] = self.face_recognizer_edit.text().strip()
        self.config["use_local_models_only"] = self.use_local_models_check.isChecked()
        
        # 保存配置文件
        try:
            with open("config.json", 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            
            # 重新初始化数据库连接
            if self.db_connection and self.db_connection.is_connected():
                self.db_connection.close()
            self.init_database()
            
            # 重新加载人脸特征
            self.load_face_features()
            
            QMessageBox.information(self, "保存成功", "系统设置已保存并生效")
            
        except Exception as e:
            print(f"保存设置失败: {e}")
            QMessageBox.critical(self, "保存错误", f"保存系统设置失败: {e}")
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 停止所有摄像头
        self.stop_camera()
        if hasattr(self, 'enrollment_camera') and self.enrollment_camera:
            self.enrollment_camera.stop()
        if hasattr(self, 'attendance_camera') and self.attendance_camera:
            self.attendance_camera.stop()
        
        # 关闭数据库连接
        if self.db_connection and self.db_connection.is_connected():
            self.db_connection.close()
        
        event.accept()

def main():
    """主函数"""
    app = QApplication(sys.argv)
    window = FaceRecognitionSystem()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()