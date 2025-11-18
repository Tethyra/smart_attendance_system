import os
from PyQt5.QtMultimedia import QCamera, QCameraInfo
from PyQt5.QtMultimediaWidgets import QCameraViewfinder
from PyQt5.QtWidgets import QLabel, QMessageBox
from PyQt5.QtCore import Qt, QTimer

class FaceRecognitionCamera:
    """摄像头管理类"""
    def __init__(self, parent):
        self.parent = parent
        self.config = parent.config
        self.camera = None
        self.current_camera_index = self.config.get('camera_index', 0)
        self.viewfinder = None
        self.enroll_viewfinder = None
        self.attendance_camera = None
        self.attendance_viewfinder = None
        self.init_camera_and_timers()

    def init_camera_and_timers(self):
        """初始化摄像头和定时器"""
        # 定时器
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_frame)
        self.camera_timer.setInterval(30)  # 33fps
        
        # 修复问题1和2：使用主窗口的定时器而不是创建新的
        self.recognition_timer = self.parent.recognition_timer
        
        self.enrollment_timer = QTimer()
        self.enrollment_timer.timeout.connect(self.update_enrollment_frame)
        self.enrollment_timer.setInterval(30)
        
        self.attendance_timer = QTimer()
        self.attendance_timer.timeout.connect(self.parent.update_attendance)
        self.attendance_timer.setInterval(1000)
        
        # 考勤摄像头定时器
        self.attendance_camera_timer = QTimer()
        self.attendance_camera_timer.timeout.connect(self.update_attendance_camera_frame)
        self.attendance_camera_timer.setInterval(30)
        
        # 签退人脸识别定时器（修复需求3）
        self.checkout_recognition_timer = QTimer()
        self.checkout_recognition_timer.timeout.connect(self.parent.perform_checkout_recognition)
        self.checkout_recognition_timer.setInterval(500)  # 2fps
        
        # 新增：统计更新定时器
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.parent.update_all_stats)
        self.stats_timer.setInterval(5000)  # 每5秒更新一次统计

    def update_camera_frame(self):
        """更新摄像头帧"""
        # 简化实现：摄像头预览已经通过QCameraViewfinder处理
        pass

    def update_enrollment_frame(self):
        """更新录入摄像头帧"""
        # 简化实现：摄像头预览已经通过QCameraViewfinder处理
        pass

    def update_attendance_camera_frame(self):
        """更新考勤摄像头帧"""
        # 简化实现：考勤摄像头预览处理
        pass

    def toggle_camera(self):
        """切换摄像头状态"""
        if not self.parent.is_camera_running:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        """启动摄像头"""
        try:
            # 使用PyQt5的QCamera替代OpenCV
            # 获取可用摄像头
            cameras = QCameraInfo.availableCameras()
            if not cameras:
                QMessageBox.warning(self.parent, "警告", "未找到可用摄像头")
                return

            # 创建摄像头对象
            self.camera = QCamera(cameras[self.current_camera_index])
            
            # 创建取景器
            self.viewfinder = QCameraViewfinder()
            self.viewfinder.setFixedSize(640, 480)
            
            # 清空视频标签布局
            layout = self.parent.video_label.layout()
            if layout:
                for i in reversed(range(layout.count())):
                    widget = layout.itemAt(i).widget()
                    if widget:
                        widget.deleteLater()
            
            # 添加取景器到视频标签
            layout.addWidget(self.viewfinder)
            self.camera.setViewfinder(self.viewfinder)
            
            # 启动摄像头
            self.camera.start()
            self.parent.is_camera_running = True
            
            # 更新界面
            self.parent.camera_btn.setText("停止摄像头")
            self.parent.camera_status_label.setText("摄像头: 开启")
            self.parent.camera_status_label.setStyleSheet("color: green;")
            self.parent.update_log("摄像头启动成功")
            
        except Exception as e:
            self.parent.update_log(f"摄像头启动失败: {str(e)}")
            QMessageBox.critical(self.parent, "错误", f"摄像头启动失败: {str(e)}")

    def stop_camera(self):
        """停止摄像头"""
        try:
            if self.camera:
                self.camera.stop()
                self.camera.deleteLater()
                self.camera = None
            self.parent.is_camera_running = False
            
            # 恢复默认显示
            layout = self.parent.video_label.layout()
            if layout:
                for i in reversed(range(layout.count())):
                    widget = layout.itemAt(i).widget()
                    if widget:
                        widget.deleteLater()
            
            default_text = QLabel("摄像头未启动")
            default_text.setStyleSheet("color: #aaa; font-size: 16px;")
            default_text.setAlignment(Qt.AlignCenter)
            layout.addWidget(default_text)
            layout.setAlignment(Qt.AlignCenter)
            
            # 更新录入预览区域
            if hasattr(self.parent, 'enroll_preview'):
                enroll_layout = self.parent.enroll_preview.layout()
                if enroll_layout:
                    for i in reversed(range(enroll_layout.count())):
                        widget = enroll_layout.itemAt(i).widget()
                        if widget:
                            widget.deleteLater()
                default_preview = QLabel("预览区域")
                default_preview.setStyleSheet("color: #aaa; font-size: 14px;")
                default_preview.setAlignment(Qt.AlignCenter)
                enroll_layout.addWidget(default_preview)
                enroll_layout.setAlignment(Qt.AlignCenter)
            
            # 更新界面
            self.parent.camera_btn.setText("启动摄像头")
            self.parent.camera_status_label.setText("摄像头: 关闭")
            self.parent.camera_status_label.setStyleSheet("color: red;")
            
            # 更新录入界面
            if hasattr(self.parent, 'enroll_camera_btn'):
                self.parent.enroll_camera_btn.setText("启动摄像头录入")
            if hasattr(self.parent, 'capture_btn'):
                self.parent.capture_btn.setEnabled(False)
            if hasattr(self.parent, 'enrollment_status'):
                self.parent.enrollment_status.setText("摄像头已停止")
                
            self.parent.update_log("摄像头停止成功")
            
        except Exception as e:
            self.parent.update_log(f"摄像头停止失败: {str(e)}")

    def toggle_enroll_camera(self):
        """切换录入摄像头"""
        if not self.parent.is_camera_running:
            self.start_enroll_camera()
        else:
            self.stop_enroll_camera()

    def start_enroll_camera(self):
        """启动录入摄像头"""
        try:
            # 获取可用摄像头
            cameras = QCameraInfo.availableCameras()
            if not cameras:
                QMessageBox.warning(self.parent, "警告", "未找到可用摄像头")
                return

            # 创建摄像头对象
            self.camera = QCamera(cameras[self.current_camera_index])
            
            # 创建录入取景器
            self.enroll_viewfinder = QCameraViewfinder()
            self.enroll_viewfinder.setFixedSize(320, 240)
            
            # 清空录入预览布局
            enroll_layout = self.parent.enroll_preview.layout()
            if enroll_layout:
                for i in reversed(range(enroll_layout.count())):
                    widget = enroll_layout.itemAt(i).widget()
                    if widget:
                        widget.deleteLater()
            
            # 添加取景器到录入预览区域
            enroll_layout.addWidget(self.enroll_viewfinder)
            self.camera.setViewfinder(self.enroll_viewfinder)
            
            # 启动摄像头
            self.camera.start()
            self.parent.is_camera_running = True
            
            # 更新录入界面
            self.parent.enroll_camera_btn.setText("停止摄像头")
            self.parent.capture_btn.setEnabled(True)
            self.parent.enrollment_status.setText("摄像头已启动，请面对摄像头")
            
            # 更新主摄像头状态
            self.parent.camera_status_label.setText("摄像头: 开启")
            self.parent.camera_status_label.setStyleSheet("color: green;")
            
            self.parent.update_log("录入摄像头启动成功")
            
        except Exception as e:
            self.parent.update_log(f"录入摄像头启动失败: {str(e)}")

    def stop_enroll_camera(self):
        """停止录入摄像头"""
        self.stop_camera()

    def start_attendance_camera(self):
        """启动考勤摄像头"""
        try:
            # 获取可用摄像头
            cameras = QCameraInfo.availableCameras()
            if not cameras:
                QMessageBox.warning(self.parent, "警告", "未找到可用摄像头")
                return

            # 创建摄像头对象（使用第二个摄像头，如果有的话）
            camera_index = self.config.get('camera_index', 0)
            if len(cameras) > 1:
                camera_index = 1  # 使用第二个摄像头
            
            self.attendance_camera = QCamera(cameras[camera_index])
            
            # 创建取景器
            self.attendance_viewfinder = QCameraViewfinder()
            self.attendance_viewfinder.setFixedSize(320, 240)
            
            # 清空考勤摄像头布局
            layout = self.parent.attendance_camera_label.layout()
            if layout:
                for i in reversed(range(layout.count())):
                    widget = layout.itemAt(i).widget()
                    if widget:
                        widget.deleteLater()
            
            # 添加取景器到考勤摄像头区域
            layout.addWidget(self.attendance_viewfinder)
            self.attendance_camera.setViewfinder(self.attendance_viewfinder)
            
            # 启动摄像头
            self.attendance_camera.start()
            
            # 更新界面
            self.parent.start_attendance_camera_btn.setEnabled(False)
            self.parent.stop_attendance_camera_btn.setEnabled(True)
            self.parent.update_log("考勤摄像头启动成功")
            
            # 启动考勤摄像头定时器
            self.attendance_camera_timer.start()
            
        except Exception as e:
            self.parent.update_log(f"考勤摄像头启动失败: {str(e)}")
            QMessageBox.critical(self.parent, "错误", f"考勤摄像头启动失败: {str(e)}")

    def stop_attendance_camera(self):
        """停止考勤摄像头"""
        try:
            if self.attendance_camera:
                self.attendance_camera.stop()
                self.attendance_camera.deleteLater()
                self.attendance_camera = None
            
            # 停止定时器
            self.attendance_camera_timer.stop()
            
            # 恢复默认显示
            layout = self.parent.attendance_camera_label.layout()
            if layout:
                for i in reversed(range(layout.count())):
                    widget = layout.itemAt(i).widget()
                    if widget:
                        widget.deleteLater()
            
            default_text = QLabel("考勤摄像头已停止")
            default_text.setStyleSheet("color: #aaa; font-size: 14px;")
            default_text.setAlignment(Qt.AlignCenter)
            layout.addWidget(default_text)
            layout.setAlignment(Qt.AlignCenter)
            
            # 更新界面
            self.parent.start_attendance_camera_btn.setEnabled(True)
            self.parent.stop_attendance_camera_btn.setEnabled(False)
            self.parent.update_log("考勤摄像头停止成功")
            
        except Exception as e:
            self.parent.update_log(f"考勤摄像头停止失败: {str(e)}")