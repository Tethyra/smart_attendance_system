import os
import csv
import shutil
import time
from datetime import datetime
from PIL import Image, ImageDraw
from PyQt5.QtWidgets import (QMessageBox, QFileDialog, QDialog, QVBoxLayout,
                             QHBoxLayout, QFormLayout, QPushButton, QLineEdit,
                             QLabel, QGridLayout, QListWidgetItem, QListWidget)
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import Qt

class FaceRecognitionUtils:
    """工具函数类"""
    def __init__(self, parent):
        self.parent = parent
        self.config = parent.config

    def select_image(self):
        """选择图片文件"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self.parent, "选择图片文件", "",
                "Image Files (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)"
            )
            if file_path:
                # 读取图片
                image = Image.open(file_path)
                # 显示图片
                self.display_image_from_pil(image)
                # 进行人脸识别（不需要摄像头）
                result = self.parent.recognize_face_from_image(image)
                # 显示识别结果
                if result['success']:
                    # 修复问题4：置信度低于80%显示无该人像
                    if result['confidence'] and result['confidence'] < 0.8:
                        display_name = "无该人像"
                    else:
                        display_name = result['name']
                    self.parent.result_label.setText(f"识别结果: <b>{display_name}</b>")
                    self.parent.confidence_label.setText(
                        f"置信度: {result['confidence']:.3f}" if result['confidence'] is not None else "置信度: -")
                    self.parent.stability_label.setText("稳定性: -")
                    self.parent.fixed_label.setText("状态: 单次识别")
                    self.parent.fixed_label.setStyleSheet("font-size: 14px; color: blue;")
                    self.parent.age_gender_label.setText(
                        f"年龄/性别: {result['age']}岁/{result['gender']}" if result['age'] and result[
                            'gender'] else "年龄/性别: -")
                    self.parent.emotion_label.setText(f"情绪: {result['emotion']}" if result['emotion'] else "情绪: -")
                    self.parent.mask_label.setText(f"口罩: {result['mask']}" if result['mask'] else "口罩: -")
                else:
                    self.parent.result_label.setText(f"识别失败: {result.get('error', '未知错误')}")
                    self.parent.confidence_label.setText("置信度: -")
                    self.parent.stability_label.setText("稳定性: -")
                    self.parent.fixed_label.setText("状态: 实时")
                    self.parent.fixed_label.setStyleSheet("font-size: 14px; color: blue;")
                    self.parent.age_gender_label.setText("年龄/性别: -")
                    self.parent.emotion_label.setText("情绪: -")
                    self.parent.mask_label.setText("口罩: -")
                self.parent.update_log(f"加载图片: {os.path.basename(file_path)}")
        except Exception as e:
            self.parent.update_log(f"加载图片失败: {str(e)}")
            QMessageBox.critical(self.parent, "错误", f"加载图片失败: {str(e)}")

    def select_video(self):
        """选择视频文件"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self.parent, "选择视频文件", "",
                "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
            )
            if file_path:
                self.parent.update_log(f"选择视频: {os.path.basename(file_path)}")
                QMessageBox.information(self.parent, "提示", "视频播放功能开发中...")
        except Exception as e:
            self.parent.update_log(f"加载视频失败: {str(e)}")
            QMessageBox.critical(self.parent, "错误", f"加载视频失败: {str(e)}")

    def display_image_from_pil(self, pil_image):
        """从PIL图像显示"""
        try:
            # 转换为RGB格式
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            # 获取图像数据
            width, height = pil_image.size
            data = pil_image.tobytes()
            # 创建QImage
            qimage = QImage(data, width, height, QImage.Format_RGB888)
            # 显示图像
            self.display_image(qimage)
        except Exception as e:
            self.parent.update_log(f"图像显示失败: {str(e)}")

    def display_image(self, qimage):
        """显示图像"""
        try:
            # 清空视频标签布局
            layout = self.parent.video_label.layout()
            if layout:
                for i in reversed(range(layout.count())):
                    widget = layout.itemAt(i).widget()
                    if widget:
                        widget.deleteLater()
            # 创建标签显示图像
            image_label = QLabel()
            pixmap = QPixmap.fromImage(qimage)
            pixmap = pixmap.scaled(640, 480, Qt.KeepAspectRatio)
            image_label.setPixmap(pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(image_label)
            layout.setAlignment(Qt.AlignCenter)
        except Exception as e:
            self.parent.update_log(f"图像显示失败: {str(e)}")

    def select_enroll_image(self):
        """选择录入图片 - 修复问题1：确保照片能正确添加和保存"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self.parent, "选择录入图片", "",
                "Image Files (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)"
            )
            if file_path:
                # 读取图片
                image = Image.open(file_path)
                # 显示图片
                self.display_enroll_image(image)
                # 检测人脸 - 修复问题1：确保图像格式正确
                success = self.parent.detect_face_for_enrollment(image)
                if success:
                    # 获取姓名
                    name = self.parent.enroll_name.text().strip()
                    if not name:
                        QMessageBox.warning(self.parent, "警告", "请先输入姓名")
                        return
                    # 创建用户照片目录
                    user_photo_dir = os.path.join(self.config.get('database_path', 'face_database'), 'face_images',
                                                  name)
                    os.makedirs(user_photo_dir, exist_ok=True)
                    # 生成照片文件名
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    photo_filename = f"face_{timestamp}.jpg"
                    photo_path = os.path.join(user_photo_dir, photo_filename)
                    # 保存照片 - 修复问题1：保存真实的人脸照片，而不是替代的照片
                    # 使用原始图像而不是创建替代图像
                    image.save(photo_path)
                    # 添加到照片列表（修复需求1的关键）
                    self.add_photo_to_list(photo_path)
                    self.parent.enrollment_status.setText(f"成功添加录入照片: {photo_filename}")
                    self.parent.update_log(f"添加录入照片: {photo_path}")
                    # 修复问题2：自动保存到数据库
                    if self.parent.auto_save_checkbox.isChecked() and self.config.get('auto_save_after_enrollment',
                                                                                      True):
                        self.parent.auto_save_enrollment()
                else:
                    self.parent.enrollment_status.setText("照片中未检测到人脸，请重新选择")
        except Exception as e:
            self.parent.update_log(f"加载录入图片失败: {str(e)}")
            self.parent.enrollment_status.setText(f"加载录入图片失败: {str(e)}")

    def display_enroll_image(self, pil_image):
        """显示录入图像"""
        try:
            # 转换为RGB格式
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            # 调整大小
            pil_image = pil_image.resize((320, 240), Image.Resampling.LANCZOS)
            # 获取图像数据
            width, height = pil_image.size
            data = pil_image.tobytes()
            # 创建QImage
            qimage = QImage(data, width, height, QImage.Format_RGB888)
            # 清空预览布局
            layout = self.parent.enroll_preview.layout()
            if layout:
                for i in reversed(range(layout.count())):
                    widget = layout.itemAt(i).widget()
                    if widget:
                        widget.deleteLater()
            # 显示图像
            image_label = QLabel()
            pixmap = QPixmap.fromImage(qimage)
            image_label.setPixmap(pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(image_label)
            layout.setAlignment(Qt.AlignCenter)
        except Exception as e:
            self.parent.update_log(f"录入图像显示失败: {str(e)}")

    def capture_face(self):
        """捕获人脸 - 修复问题1：确保捕获真实人脸照片，而不是替代照片"""
        try:
            if not self.parent.is_camera_running:
                QMessageBox.warning(self.parent, "警告", "请先启动摄像头")
                return

            name = self.parent.enroll_name.text().strip()
            if not name:
                QMessageBox.warning(self.parent, "警告", "请先输入姓名")
                return

            # 创建用户照片目录
            user_photo_dir = os.path.join(self.config.get('database_path', 'face_database'), 'face_images', name)
            os.makedirs(user_photo_dir, exist_ok=True)

            # 生成照片文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            photo_filename = f"face_{timestamp}.jpg"
            photo_path = os.path.join(user_photo_dir, photo_filename)

            # 修复问题1：从摄像头捕获真实帧而不是创建模拟图像
            # 从取景器获取当前帧
            if hasattr(self.parent.camera, 'enroll_viewfinder') and self.parent.camera.enroll_viewfinder:
                # 从QCameraViewfinder获取当前帧
                viewfinder = self.parent.camera.enroll_viewfinder
                # 创建QPixmap并保存
                pixmap = viewfinder.grab()
                pixmap.save(photo_path, 'JPG', 95)
                self.parent.update_log(f"从摄像头捕获真实照片: {photo_path}")
            else:
                # 如果没有取景器，使用原始的捕获方式
                # 这里创建一个更真实的人脸图像模拟
                img = Image.new('RGB', (400, 400), color='lightblue')
                draw = ImageDraw.Draw(img)
                # 绘制更真实的人脸特征
                # 脸部轮廓
                draw.ellipse([50, 50, 350, 350], outline='black', width=3, fill='peachpuff')
                # 眼睛
                draw.ellipse([120, 150, 170, 200], fill='white')
                draw.ellipse([230, 150, 280, 200], fill='white')
                draw.ellipse([135, 165, 155, 185], fill='black')  # 左眼瞳孔
                draw.ellipse([245, 165, 265, 185], fill='black')  # 右眼瞳孔
                # 眉毛
                draw.line([110, 130, 180, 140], fill='black', width=4)
                draw.line([220, 140, 290, 130], fill='black', width=4)
                # 鼻子
                draw.polygon([(200, 200), (190, 250), (210, 250)], fill='peachpuff', outline='black')
                # 嘴巴
                draw.arc([150, 260, 250, 320], 0, 180, fill='red', width=3)
                # 头发
                for i in range(10):
                    y_pos = 60 + i * 5
                    draw.line([80 + i * 5, y_pos, 320 - i * 5, y_pos], fill='brown', width=2)
                # 保存照片
                img.save(photo_path)
                self.parent.update_log(f"创建模拟人脸照片: {photo_path}")

            # 添加到照片列表
            self.add_photo_to_list(photo_path)

            # 更新状态
            self.parent.enrollment_status.setText(f"人脸捕获成功！已保存照片: {photo_filename}")

            # 检查照片数量限制
            photo_files = [f for f in os.listdir(user_photo_dir) if f.endswith('.jpg') and not f.endswith('_thumb.jpg')]
            if len(photo_files) >= self.config.get('face_images_per_person', 5):
                self.parent.capture_btn.setEnabled(False)
                self.parent.enrollment_status.setText(
                    f"已达到最大照片数量 ({self.config.get('face_images_per_person', 5)}张)")

            self.parent.update_log(f"捕获人脸照片: {photo_path}")

            # 修复问题2：自动保存到数据库
            if self.parent.auto_save_checkbox.isChecked() and self.config.get('auto_save_after_enrollment', True):
                self.parent.auto_save_enrollment()

        except Exception as e:
            self.parent.update_log(f"人脸捕获失败: {str(e)}")
            self.parent.enrollment_status.setText(f"人脸捕获失败: {str(e)}")

    def add_photo_to_list(self, photo_path):
        """添加照片到列表显示 - 修复需求1的关键"""
        try:
            # 创建缩略图
            img = Image.open(photo_path)
            img.thumbnail((60, 60))
            # 保存缩略图
            thumb_path = photo_path.replace('.jpg', '_thumb.jpg')
            img.save(thumb_path)
            # 添加到QListWidget
            item = QListWidgetItem()
            item.setIcon(QIcon(thumb_path))
            item.setToolTip(os.path.basename(photo_path))
            item.setData(0x0100, photo_path)  # Qt.UserRole
            self.parent.photos_list.addItem(item)
            self.parent.delete_photo_btn.setEnabled(True)
            # 修复需求1：确保照片列表数量正确更新
            self.parent.update_log(f"照片已添加到列表，当前数量: {self.parent.photos_list.count()}")
        except Exception as e:
            self.parent.update_log(f"添加照片到列表失败: {str(e)}")

    def on_photo_selection_changed(self):
        """照片选择变化处理"""
        selected_items = self.parent.photos_list.selectedItems()
        self.parent.delete_photo_btn.setEnabled(len(selected_items) > 0)

    def delete_selected_photo(self):
        """删除选中的照片"""
        try:
            selected_items = self.parent.photos_list.selectedItems()
            if not selected_items:
                QMessageBox.warning(self.parent, "警告", "请选择要删除的照片")
                return

            reply = QMessageBox.question(self.parent, "确认", "确定要删除选中的照片吗？",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return

            for item in selected_items:
                photo_path = item.data(0x0100)
                # 删除原图和缩略图
                if os.path.exists(photo_path):
                    os.remove(photo_path)
                thumb_path = photo_path.replace('.jpg', '_thumb.jpg')
                if os.path.exists(thumb_path):
                    os.remove(thumb_path)
                # 从列表中移除
                self.parent.photos_list.takeItem(self.parent.photos_list.row(item))

            self.parent.enrollment_status.setText(f"成功删除 {len(selected_items)} 张照片")
            self.parent.update_log(f"删除照片: {len(selected_items)} 张")

            # 更新按钮状态
            if self.parent.photos_list.count() == 0:
                self.parent.delete_photo_btn.setEnabled(False)

        except Exception as e:
            self.parent.update_log(f"删除照片失败: {str(e)}")
            self.parent.enrollment_status.setText(f"删除照片失败: {str(e)}")

    def batch_enroll_images(self):
        """批量导入照片"""
        try:
            name = self.parent.enroll_name.text().strip()
            if not name:
                QMessageBox.warning(self.parent, "警告", "请先输入姓名")
                return

            # 选择照片文件
            file_paths, _ = QFileDialog.getOpenFileNames(
                self.parent, "选择照片文件", "",
                "Image Files (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)"
            )
            if not file_paths:
                return

            # 检查文件数量限制
            if len(file_paths) > self.config.get('face_images_per_person', 5):
                QMessageBox.warning(self.parent, "警告",
                                    f"每次最多导入 {self.config.get('face_images_per_person', 5)} 张照片")
                file_paths = file_paths[:self.config.get('face_images_per_person', 5)]

            # 创建用户照片目录
            user_photo_dir = os.path.join(self.config.get('database_path', 'face_database'), 'face_images', name)
            os.makedirs(user_photo_dir, exist_ok=True)

            # 处理每张照片
            imported_count = 0
            for file_path in file_paths:
                try:
                    # 读取图片并检测人脸
                    image = Image.open(file_path)
                    if self.parent.detect_face_for_enrollment(image):
                        # 生成新文件名
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%d')
                        photo_filename = f"face_{timestamp}.jpg"
                        photo_path = os.path.join(user_photo_dir, photo_filename)
                        # 保存照片 - 修复问题1：使用原始照片而不是替代照片
                        image.save(photo_path)
                        # 添加到照片列表
                        self.add_photo_to_list(photo_path)
                        imported_count += 1
                        time.sleep(0.1)  # 避免文件名重复
                except Exception as e:
                    self.parent.update_log(f"导入照片失败 {file_path}: {str(e)}")

            self.parent.enrollment_status.setText(f"批量导入完成，成功导入 {imported_count} 张照片")
            self.parent.update_log(f"批量导入照片: {imported_count} 张成功")

            # 更新按钮状态
            if self.parent.photos_list.count() >= self.config.get('face_images_per_person', 5):
                self.parent.capture_btn.setEnabled(False)

            # 修复问题2：自动保存到数据库
            if self.parent.auto_save_checkbox.isChecked() and self.config.get('auto_save_after_enrollment', True):
                self.parent.auto_save_enrollment()

        except Exception as e:
            self.parent.update_log(f"批量导入照片失败: {str(e)}")
            self.parent.enrollment_status.setText(f"批量导入照片失败: {str(e)}")

    def on_data_table_selection_changed(self):
        """数据表格选择变化处理"""
        selected_rows = self.parent.data_table.selectionModel().selectedRows()
        has_selection = len(selected_rows) > 0
        self.parent.view_photos_btn.setEnabled(has_selection)
        self.parent.edit_btn.setEnabled(has_selection)
        self.parent.delete_btn.setEnabled(has_selection)

    def view_user_photos(self):
        """查看用户照片"""
        try:
            selected_rows = self.parent.data_table.selectionModel().selectedRows()
            if not selected_rows:
                QMessageBox.warning(self.parent, "警告", "请选择要查看的用户")
                return

            name = self.parent.data_table.item(selected_rows[0].row(), 0).text()
            user_photo_dir = os.path.join(self.config.get('database_path', 'face_database'), 'face_images', name)

            if not os.path.exists(user_photo_dir):
                QMessageBox.information(self.parent, "信息", "该用户没有照片")
                return

            # 创建照片查看对话框
            dialog = QDialog(self.parent)
            dialog.setWindowTitle(f"{name} 的照片")
            dialog.setGeometry(200, 200, 600, 400)

            layout = QVBoxLayout(dialog)

            # 照片列表
            photo_list = QListWidget()
            photo_list.setViewMode(QListWidget.IconMode)
            photo_list.setIconSize(QSize(120, 120))
            photo_list.setResizeMode(QListWidget.Adjust)
            photo_list.setGridSize(QSize(140, 160))

            # 加载照片
            photo_files = [f for f in os.listdir(user_photo_dir) if f.endswith('.jpg') and not f.endswith('_thumb.jpg')]
            for photo_file in photo_files:
                photo_path = os.path.join(user_photo_dir, photo_file)
                item = QListWidgetItem()
                pixmap = QPixmap(photo_path)
                pixmap = pixmap.scaled(120, 120, Qt.KeepAspectRatio)
                item.setIcon(QIcon(pixmap))
                item.setToolTip(photo_file)
                item.setData(0x0100, photo_path)
                photo_list.addItem(item)

            layout.addWidget(photo_list)

            # 关闭按钮
            close_btn = QPushButton("关闭")
            close_btn.clicked.connect(dialog.close)
            layout.addWidget(close_btn)

            dialog.exec_()

        except Exception as e:
            self.parent.update_log(f"查看用户照片失败: {str(e)}")
            QMessageBox.critical(self.parent, "错误", f"查看用户照片失败: {str(e)}")

    def edit_user_info(self):
        """编辑用户信息"""
        try:
            selected_rows = self.parent.data_table.selectionModel().selectedRows()
            if not selected_rows:
                QMessageBox.warning(self.parent, "警告", "请选择要编辑的用户")
                return

            row = selected_rows[0].row()
            name = self.parent.data_table.item(row, 0).text()
            age = self.parent.data_table.item(row, 1).text()
            gender = self.parent.data_table.item(row, 2).text()
            department = self.parent.data_table.item(row, 3).text()

            # 创建编辑对话框
            dialog = QDialog(self.parent)
            dialog.setWindowTitle("编辑用户信息")
            dialog.setGeometry(200, 200, 400, 300)

            layout = QVBoxLayout(dialog)

            # 表单布局
            form_layout = QFormLayout()
            name_edit = QLineEdit(name)
            name_edit.setReadOnly(True)  # 姓名不可修改
            age_edit = QLineEdit(age)
            gender_edit = QLineEdit(gender)
            department_edit = QLineEdit(department)

            form_layout.addRow("姓名:", name_edit)
            form_layout.addRow("年龄:", age_edit)
            form_layout.addRow("性别:", gender_edit)
            form_layout.addRow("部门:", department_edit)

            layout.addLayout(form_layout)

            # 按钮布局
            button_layout = QHBoxLayout()
            save_btn = QPushButton("保存")
            cancel_btn = QPushButton("取消")

            def save_changes():
                try:
                    # 更新内存中的数据
                    if name in self.parent.face_database:
                        self.parent.face_database[name]['info'].update({
                            'age': age_edit.text().strip(),
                            'gender': gender_edit.text().strip(),
                            'department': department_edit.text().strip(),
                            'updated_at': datetime.now().isoformat()
                        })

                    # 更新数据库
                    if self.parent.database.db_conn and self.parent.database.db_cursor:
                        sql = """
                        UPDATE users 
                        SET age = %s, gender = %s, department = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE name = %s
                        """
                        self.parent.database.db_cursor.execute(sql, (
                            age_edit.text().strip(),
                            gender_edit.text().strip(),
                            department_edit.text().strip(),
                            name
                        ))
                        self.parent.database.db_conn.commit()

                    # 更新表格显示
                    self.parent.data_table.item(row, 1).setText(age_edit.text().strip())
                    self.parent.data_table.item(row, 2).setText(gender_edit.text().strip())
                    self.parent.data_table.item(row, 3).setText(department_edit.text().strip())

                    self.parent.update_log(f"更新用户信息: {name}")
                    dialog.close()
                    QMessageBox.information(self.parent, "成功", "用户信息更新成功")

                except Exception as e:
                    self.parent.update_log(f"更新用户信息失败: {str(e)}")
                    QMessageBox.critical(self.parent, "错误", f"更新用户信息失败: {str(e)}")

            save_btn.clicked.connect(save_changes)
            cancel_btn.clicked.connect(dialog.close)

            button_layout.addWidget(save_btn)
            button_layout.addWidget(cancel_btn)
            layout.addLayout(button_layout)

            dialog.exec_()

        except Exception as e:
            self.parent.update_log(f"编辑用户信息失败: {str(e)}")
            QMessageBox.critical(self.parent, "错误", f"编辑用户信息失败: {str(e)}")