import os
import csv
import pymysql
from pymysql import OperationalError
from datetime import datetime
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtCore import Qt


class FaceRecognitionDatabase:
    """数据库管理类"""

    def __init__(self, parent):
        self.parent = parent
        self.config = parent.config
        self.db_conn = None
        self.db_cursor = None
        self.init_directories()
        self.init_database()

    def init_directories(self):
        """初始化目录"""
        # 创建必要的目录
        directories = [
            self.config.get('database_path', 'face_database'),
            self.config.get('model_path', 'models'),
            'logs',
            os.path.join(self.config.get('database_path', 'face_database'), 'face_images')  # 人脸照片目录
        ]
        for dir_path in directories:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                self.parent.update_log(f"创建目录: {dir_path}")

        # 初始化数据文件
        data_files = {
            'features_file': os.path.join(self.config.get('database_path', 'face_database'),
                                          self.config.get('features_file', 'face_features.csv')),
            'log_file': os.path.join('logs', self.config.get('log_file', 'recognition_log.csv')),
            'attendance_file': os.path.join('logs', self.config.get('attendance_file', 'attendance.csv'))
        }
        for file_name, file_path in data_files.items():
            if not os.path.exists(file_path):
                with open(file_path, 'w', encoding='utf-8', newline='') as f:
                    if file_name == 'features_file':
                        writer = csv.writer(f)
                        writer.writerow(['name', 'timestamp'] + [f'feature_{i}' for i in range(128)])
                    elif file_name == 'log_file':
                        writer = csv.writer(f)
                        writer.writerow(['timestamp', 'name', 'confidence', 'status', 'method'])
                    elif file_name == 'attendance_file':
                        writer = csv.writer(f)
                        writer.writerow(['timestamp', 'name', 'status', 'location'])
                self.parent.update_log(f"创建数据文件: {file_path}")

    def init_database(self):
        """初始化MySQL数据库"""
        self.db_conn = None
        self.db_cursor = None
        try:
            # 连接MySQL数据库
            self.db_conn = pymysql.connect(
                host=self.config.get('mysql_host', 'localhost'),
                port=self.config.get('mysql_port', 3306),
                user=self.config.get('mysql_user', 'root'),
                password=self.config.get('mysql_password', '123456'),
                database=self.config.get('mysql_database', 'smart_attendance'),
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            self.db_cursor = self.db_conn.cursor()

            # 创建数据库（如果不存在）
            self.db_cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS {self.config.get('mysql_database', 'smart_attendance')}")
            self.db_cursor.execute(f"USE {self.config.get('mysql_database', 'smart_attendance')}")

            # 创建用户表
            self.db_cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTO_INCREMENT,
                    name VARCHAR(100) UNIQUE NOT NULL,
                    age INTEGER,
                    gender VARCHAR(20),
                    department VARCHAR(100),
                    face_encoding TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

            # 创建人脸识别记录表
            self.db_cursor.execute('''
                CREATE TABLE IF NOT EXISTS recognition_logs (
                    id INTEGER PRIMARY KEY AUTO_INCREMENT,
                    user_id INTEGER,
                    recognition_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence DECIMAL(10,6),
                    age_prediction INTEGER,
                    gender_prediction VARCHAR(20),
                    emotion_prediction VARCHAR(20),
                    mask_detection VARCHAR(20),
                    image_path VARCHAR(255),
                    FOREIGN KEY (user_id) REFERENCES users (id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

            # 创建考勤表
            self.db_cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTO_INCREMENT,
                    user_id INTEGER,
                    check_in_time TIMESTAMP,
                    check_out_time TIMESTAMP,
                    status VARCHAR(20),
                    location VARCHAR(100),
                    temperature DECIMAL(5,2),
                    checkout_recognition_confidence DECIMAL(10,6),
                    checkin_recognition_confidence DECIMAL(10,6),
                    FOREIGN KEY (user_id) REFERENCES users (id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

            # 创建人脸照片表
            self.db_cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_images (
                    id INTEGER PRIMARY KEY AUTO_INCREMENT,
                    user_id INTEGER,
                    image_path TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_primary TINYINT(1),
                    FOREIGN KEY (user_id) REFERENCES users (id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            ''')

            self.db_conn.commit()
            self.parent.update_log("MySQL数据库初始化成功")

        except OperationalError as e:
            self.parent.update_log(f"MySQL数据库连接失败: {str(e)}")
            self.parent.update_log("请检查MySQL服务是否启动，用户名密码是否正确")
            self.parent.update_log("如果MySQL服务未安装，系统将使用文件存储模式")
        except Exception as e:
            self.parent.update_log(f"数据库初始化失败: {str(e)}")

    def load_face_database(self):
        """加载人脸数据库"""
        try:
            # 从CSV文件加载特征
            features_file = os.path.join(self.config.get('database_path', 'face_database'),
                                         self.config.get('features_file', 'face_features.csv'))
            if os.path.exists(features_file):
                with open(features_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        name = row['name']
                        if name not in self.parent.face_database:
                            self.parent.face_database[name] = {
                                'features': [],
                                'images': [],
                                'info': {}
                            }
                        # 提取特征向量
                        features = []
                        for i in range(128):
                            feature_key = f'feature_{i}'
                            if feature_key in row:
                                features.append(float(row[feature_key]))
                        if features:
                            self.parent.face_database[name]['features'].append(features)

            # 从MySQL数据库加载用户信息
            if self.db_conn and self.db_cursor:
                try:
                    self.db_cursor.execute("SELECT * FROM users")
                    users = self.db_cursor.fetchall()
                    for user in users:
                        name = user['name']
                        if name not in self.parent.face_database:
                            self.parent.face_database[name] = {
                                'features': [],
                                'images': [],
                                'info': {}
                            }
                        # 更新用户信息
                        self.parent.face_database[name]['info'].update({
                            'age': str(user['age']) if user['age'] is not None else '',
                            'gender': user['gender'] if user['gender'] is not None else '',
                            'department': user['department'] if user['department'] is not None else '',
                            'created_at': user['created_at'].isoformat() if user['created_at'] is not None else '',
                            'updated_at': user['updated_at'].isoformat() if user['updated_at'] is not None else ''
                        })
                        # 加载用户照片
                        self.db_cursor.execute("SELECT image_path FROM face_images WHERE user_id = %s", (user['id'],))
                        image_paths = self.db_cursor.fetchall()
                        self.parent.face_database[name]['images'] = [img['image_path'] for img in image_paths]

                except Exception as e:
                    self.parent.update_log(f"从MySQL加载用户信息失败: {str(e)}")

            # 从文件系统加载照片路径
            face_images_dir = os.path.join(self.config.get('database_path', 'face_database'), 'face_images')
            if os.path.exists(face_images_dir):
                for name in os.listdir(face_images_dir):
                    user_dir = os.path.join(face_images_dir, name)
                    if os.path.isdir(user_dir):
                        if name not in self.parent.face_database:
                            self.parent.face_database[name] = {
                                'features': [],
                                'images': [],
                                'info': {}
                            }
                        # 获取所有照片文件
                        photo_files = [f for f in os.listdir(user_dir) if
                                       f.endswith('.jpg') and not f.endswith('_thumb.jpg')]
                        photo_paths = [os.path.join(user_dir, f) for f in photo_files]
                        self.parent.face_database[name]['images'] = photo_paths

            # 更新统计信息
            self.parent.total_users = len(self.parent.face_database)
            self.parent.update_log(f"人脸数据库加载完成，共 {self.parent.total_users} 个人脸")
            self.parent.update_stats()

        except Exception as e:
            self.parent.update_log(f"加载人脸数据库失败: {str(e)}")

    def save_face_database(self):
        """保存人脸数据库"""
        try:
            features_file = os.path.join(self.config.get('database_path', 'face_database'),
                                         self.config.get('features_file', 'face_features.csv'))
            with open(features_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['name', 'timestamp'] + [f'feature_{i}' for i in range(128)])
                for name, data in self.parent.face_database.items():
                    for features in data['features']:
                        row = [name, datetime.now().isoformat()]
                        row.extend(features)
                        writer.writerow(row)
            self.parent.update_log("人脸数据库保存完成")

        except Exception as e:
            self.parent.update_log(f"保存人脸数据库失败: {str(e)}")

    def save_enrollment(self):
        """保存录入信息"""
        try:
            # 获取录入信息
            name = self.parent.enroll_name.text().strip()
            age = self.parent.enroll_age.text().strip()
            gender = self.parent.enroll_gender.text().strip()
            department = self.parent.enroll_department.text().strip()

            if not name:
                QMessageBox.warning(self.parent, "警告", "请输入姓名")
                return

            # 检查是否有照片
            if self.parent.photos_list.count() == 0:
                QMessageBox.warning(self.parent, "警告", "请至少录入一张人脸照片")
                self.parent.update_log(f"保存失败：照片列表为空，数量: {self.parent.photos_list.count()}")
                return

            # 检查是否已存在
            user_exists = name in self.parent.face_database

            # 添加到数据库
            if name not in self.parent.face_database:
                self.parent.face_database[name] = {
                    'features': [],
                    'images': [],
                    'info': {}
                }

            # 收集照片路径
            photo_paths = []
            for i in range(self.parent.photos_list.count()):
                item = self.parent.photos_list.item(i)
                photo_path = item.data(0x0100)  # Qt.UserRole
                photo_paths.append(photo_path)

            self.parent.face_database[name]['info'] = {
                'age': age,
                'gender': gender,
                'department': department,
                'created_at': datetime.now().isoformat()
            }
            self.parent.face_database[name]['images'] = photo_paths

            # 保存到MySQL数据库
            self.save_to_database(name, age, gender, department, photo_paths)

            # 保存到文件
            self.save_face_database()

            # 更新数据表格
            self.refresh_data()

            # 清空表单和照片列表
            self.parent.enroll_name.clear()
            self.parent.enroll_age.clear()
            self.parent.enroll_gender.clear()
            self.parent.enroll_department.clear()
            self.parent.photos_list.clear()
            self.parent.delete_photo_btn.setEnabled(False)
            self.parent.capture_btn.setEnabled(False)

            # 更新状态
            action = "更新" if user_exists else "录入"
            self.parent.enrollment_status.setText(f"{action}成功：{name}")
            self.parent.update_log(f"人脸{action}成功：{name}")
            QMessageBox.information(self.parent, "成功", f"人脸{action}成功：{name}")

            # 更新统计
            self.parent.total_users = len(self.parent.face_database)
            self.parent.update_stats()

        except Exception as e:
            self.parent.update_log(f"保存录入信息失败: {str(e)}")
            QMessageBox.critical(self.parent, "错误", f"保存录入信息失败: {str(e)}")

    def auto_save_enrollment(self):
        """自动保存录入信息"""
        try:
            # 获取录入信息
            name = self.parent.enroll_name.text().strip()
            age = self.parent.enroll_age.text().strip()
            gender = self.parent.enroll_gender.text().strip()
            department = self.parent.enroll_department.text().strip()

            if not name:
                self.parent.update_log("自动保存失败：姓名为空")
                return

            # 检查是否有照片
            if self.parent.photos_list.count() == 0:
                self.parent.update_log("自动保存失败：没有录入照片")
                return

            # 检查是否已存在
            user_exists = name in self.parent.face_database

            # 添加到数据库
            if name not in self.parent.face_database:
                self.parent.face_database[name] = {
                    'features': [],
                    'images': [],
                    'info': {}
                }

            # 收集照片路径
            photo_paths = []
            for i in range(self.parent.photos_list.count()):
                item = self.parent.photos_list.item(i)
                photo_path = item.data(0x0100)  # Qt.UserRole
                photo_paths.append(photo_path)

            self.parent.face_database[name]['info'] = {
                'age': age,
                'gender': gender,
                'department': department,
                'created_at': datetime.now().isoformat()
            }
            self.parent.face_database[name]['images'] = photo_paths

            # 保存到MySQL数据库
            self.save_to_database(name, age, gender, department, photo_paths)

            # 保存到文件
            self.save_face_database()

            # 更新数据表格
            self.refresh_data()

            # 更新状态
            action = "更新" if user_exists else "录入"
            self.parent.enrollment_status.setText(f"自动{action}成功：{name}")
            self.parent.update_log(f"自动人脸{action}成功：{name}")

            # 更新统计
            self.parent.total_users = len(self.parent.face_database)
            self.parent.update_stats()

        except Exception as e:
            self.parent.update_log(f"自动保存录入信息失败: {str(e)}")

    def save_to_database(self, name, age, gender, department, photo_paths):
        """保存到数据库"""
        try:
            if not self.db_conn or not self.db_cursor:
                self.parent.update_log("MySQL连接未建立，跳过数据库保存")
                return

            # 转换年龄为整数
            age_int = int(age) if age.isdigit() else None

            # 检查用户是否存在
            self.db_cursor.execute("SELECT id FROM users WHERE name = %s", (name,))
            user = self.db_cursor.fetchone()

            if user:
                # 更新现有用户
                user_id = user['id']
                sql = """
                UPDATE users 
                SET age = %s, gender = %s, department = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """
                self.db_cursor.execute(sql, (age_int, gender, department, user_id))
                self.parent.update_log(f"更新用户信息: {name}")

                # 删除旧照片记录
                self.db_cursor.execute("DELETE FROM face_images WHERE user_id = %s", (user_id,))
                self.parent.update_log(f"删除用户旧照片记录: {name}")

            else:
                # 插入新用户
                sql = """
                INSERT INTO users (name, age, gender, department)
                VALUES (%s, %s, %s, %s)
                """
                self.db_cursor.execute(sql, (name, age_int, gender, department))
                user_id = self.db_cursor.lastrowid
                self.parent.update_log(f"插入新用户: {name} (ID: {user_id})")

            # 插入照片记录
            for i, photo_path in enumerate(photo_paths):
                sql = """
                INSERT INTO face_images (user_id, image_path, is_primary)
                VALUES (%s, %s, %s)
                """
                is_primary = 1 if i == 0 else 0  # 第一张设为主照片
                self.db_cursor.execute(sql, (user_id, photo_path, is_primary))

            self.db_conn.commit()
            self.parent.update_log(f"保存用户照片记录: {len(photo_paths)} 张")

        except Exception as e:
            self.parent.update_log(f"保存到数据库失败: {str(e)}")
            if self.db_conn:
                self.db_conn.rollback()

    def refresh_data(self):
        """刷新数据表格"""
        try:
            # 清空表格
            self.parent.data_table.setRowCount(0)

            # 从数据库重新加载数据
            self.load_face_database()

            # 重新填充表格
            row = 0
            for name, data in self.parent.face_database.items():
                user_info = data.get('info', {})
                age = user_info.get('age', '')
                gender = user_info.get('gender', '')
                department = user_info.get('department', '')
                photo_count = len(data.get('images', []))

                # 添加新行
                self.parent.data_table.insertRow(row)

                # 设置单元格内容
                name_item = QTableWidgetItem(name)
                age_item = QTableWidgetItem(age)
                gender_item = QTableWidgetItem(gender)
                department_item = QTableWidgetItem(department)
                photos_item = QTableWidgetItem(str(photo_count))

                # 设置单元格属性
                for item in [name_item, age_item, gender_item, department_item, photos_item]:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                # 添加到表格
                self.parent.data_table.setItem(row, 0, name_item)
                self.parent.data_table.setItem(row, 1, age_item)
                self.parent.data_table.setItem(row, 2, gender_item)
                self.parent.data_table.setItem(row, 3, department_item)
                self.parent.data_table.setItem(row, 4, photos_item)

                row += 1

            # 调整列宽
            self.parent.data_table.resizeColumnsToContents()
            self.parent.update_log("数据表格刷新完成")

        except Exception as e:
            self.parent.update_log(f"刷新数据表格失败: {str(e)}")

    def delete_face(self, name):
        """删除指定人脸"""
        try:
            if name in self.parent.face_database:
                # 从内存中删除
                del self.parent.face_database[name]

                # 从MySQL数据库删除
                if self.db_conn and self.db_cursor:
                    try:
                        # 获取用户ID
                        self.db_cursor.execute("SELECT id FROM users WHERE name = %s", (name,))
                        user = self.db_cursor.fetchone()
                        if user:
                            user_id = user['id']

                            # 删除照片记录
                            self.db_cursor.execute("DELETE FROM face_images WHERE user_id = %s", (user_id,))

                            # 删除用户记录
                            self.db_cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))

                            self.db_conn.commit()
                            self.parent.update_log(f"从数据库删除用户: {name}")

                    except Exception as e:
                        self.parent.update_log(f"从数据库删除用户失败: {str(e)}")
                        if self.db_conn:
                            self.db_conn.rollback()

                # 从文件系统删除照片
                face_images_dir = os.path.join(self.config.get('database_path', 'face_database'), 'face_images', name)
                if os.path.exists(face_images_dir):
                    import shutil
                    shutil.rmtree(face_images_dir)
                    self.parent.update_log(f"从文件系统删除用户照片: {face_images_dir}")

                # 保存数据库文件
                self.save_face_database()

                # 更新统计
                self.parent.total_users = len(self.parent.face_database)
                self.parent.update_stats()

                return True
            return False

        except Exception as e:
            self.parent.update_log(f"删除人脸失败: {str(e)}")
            return False

    def import_data(self):
        """导入数据"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self.parent, "导入CSV文件", "", "CSV Files (*.csv);;All Files (*)")

            if file_path:
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    imported_count = 0

                    for row in reader:
                        name = row.get('name', '').strip()
                        age = row.get('age', '').strip()
                        gender = row.get('gender', '').strip()
                        department = row.get('department', '').strip()

                        if name:
                            if name not in self.parent.face_database:
                                self.parent.face_database[name] = {
                                    'features': [],
                                    'images': [],
                                    'info': {}
                                }

                            self.parent.face_database[name]['info'] = {
                                'age': age,
                                'gender': gender,
                                'department': department,
                                'created_at': datetime.now().isoformat()
                            }

                            imported_count += 1

                    # 保存到数据库
                    self.save_face_database()
                    self.refresh_data()

                    QMessageBox.information(self.parent, "成功", f"成功导入 {imported_count} 条记录")
                    self.parent.update_log(f"导入数据成功: {imported_count} 条记录")

        except Exception as e:
            self.parent.update_log(f"导入数据失败: {str(e)}")
            QMessageBox.critical(self.parent, "错误", f"导入数据失败: {str(e)}")

    def export_data(self):
        """导出数据"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self.parent, "导出CSV文件", "", "CSV Files (*.csv);;All Files (*)")

            if file_path:
                with open(file_path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['name', 'age', 'gender', 'department', 'photo_count', 'created_at'])

                    for name, data in self.parent.face_database.items():
                        user_info = data.get('info', {})
                        writer.writerow([
                            name,
                            user_info.get('age', ''),
                            user_info.get('gender', ''),
                            user_info.get('department', ''),
                            len(data.get('images', [])),
                            user_info.get('created_at', '')
                        ])

                QMessageBox.information(self.parent, "成功", "数据导出完成")
                self.parent.update_log(f"导出数据成功: {file_path}")

        except Exception as e:
            self.parent.update_log(f"导出数据失败: {str(e)}")
            QMessageBox.critical(self.parent, "错误", f"导出数据失败: {str(e)}")

    # 考勤相关方法
    def check_in(self, name, location='默认位置'):
        """签到"""
        try:
            if not self.db_conn or not self.db_cursor:
                return False, "数据库连接失败"

            # 获取用户ID
            self.db_cursor.execute("SELECT id FROM users WHERE name = %s", (name,))
            user = self.db_cursor.fetchone()
            if not user:
                return False, "用户不存在"

            user_id = user['id']

            # 检查今天是否已经签到
            today = datetime.now().date()
            self.db_cursor.execute("""
                SELECT id, check_in_time, check_out_time FROM attendance 
                WHERE user_id = %s AND DATE(check_in_time) = %s
            """, (user_id, today))
            existing_record = self.db_cursor.fetchone()

            if existing_record:
                if existing_record['check_out_time']:
                    return False, "今日已签退，无法再次签到"
                else:
                    return False, "今日已签到，请先签退"

            # 插入签到记录
            current_time = datetime.now()
            sql = """
            INSERT INTO attendance (user_id, check_in_time, status, location, checkin_recognition_confidence)
            VALUES (%s, %s, %s, %s, %s)
            """
            self.db_cursor.execute(sql, (user_id, current_time, '已签到', location, 0.95))
            self.db_conn.commit()

            self.parent.update_log(f"签到成功: {name}")
            return True, "签到成功"

        except Exception as e:
            self.parent.update_log(f"签到失败: {str(e)}")
            if self.db_conn:
                self.db_conn.rollback()
            return False, str(e)

    def check_out(self, name, confidence=0.95):
        """签退"""
        try:
            if not self.db_conn or not self.db_cursor:
                return False, "数据库连接失败"

            # 获取用户ID
            self.db_cursor.execute("SELECT id FROM users WHERE name = %s", (name,))
            user = self.db_cursor.fetchone()
            if not user:
                return False, "用户不存在"

            user_id = user['id']

            # 检查今天是否有签到记录
            today = datetime.now().date()
            self.db_cursor.execute("""
                SELECT id, check_in_time, check_out_time FROM attendance 
                WHERE user_id = %s AND DATE(check_in_time) = %s
            """, (user_id, today))
            existing_record = self.db_cursor.fetchone()

            if not existing_record:
                return False, "今日未签到，无法签退"

            if existing_record['check_out_time']:
                return False, "今日已签退"

            # 更新签退记录
            current_time = datetime.now()
            sql = """
            UPDATE attendance 
            SET check_out_time = %s, status = %s, checkout_recognition_confidence = %s
            WHERE id = %s
            """
            self.db_cursor.execute(sql, (current_time, '已签退', confidence, existing_record['id']))
            self.db_conn.commit()

            self.parent.update_log(f"签退成功: {name}")
            return True, "签退成功"

        except Exception as e:
            self.parent.update_log(f"签退失败: {str(e)}")
            if self.db_conn:
                self.db_conn.rollback()
            return False, str(e)

    def get_attendance_records(self, date=None):
        """获取考勤记录"""
        try:
            if not self.db_conn or not self.db_cursor:
                return []

            if date:
                self.db_cursor.execute("""
                    SELECT a.*, u.name FROM attendance a
                    JOIN users u ON a.user_id = u.id
                    WHERE DATE(a.check_in_time) = %s
                    ORDER BY a.check_in_time DESC
                """, (date,))
            else:
                self.db_cursor.execute("""
                    SELECT a.*, u.name FROM attendance a
                    JOIN users u ON a.user_id = u.id
                    ORDER BY a.check_in_time DESC
                """)

            return self.db_cursor.fetchall()

        except Exception as e:
            self.parent.update_log(f"获取考勤记录失败: {str(e)}")
            return []

    def get_user_attendance(self, name, date=None):
        """获取用户考勤记录"""
        try:
            if not self.db_conn or not self.db_cursor:
                return None

            # 获取用户ID
            self.db_cursor.execute("SELECT id FROM users WHERE name = %s", (name,))
            user = self.db_cursor.fetchone()
            if not user:
                return None

            user_id = user['id']

            if date:
                # 查询指定日期的记录
                self.db_cursor.execute("""
                    SELECT * FROM attendance 
                    WHERE user_id = %s AND DATE(check_in_time) = %s
                    ORDER BY check_in_time DESC LIMIT 1
                """, (user_id, date))
            else:
                # 查询最新记录
                self.db_cursor.execute("""
                    SELECT * FROM attendance 
                    WHERE user_id = %s
                    ORDER BY check_in_time DESC LIMIT 1
                """, (user_id,))

            return self.db_cursor.fetchone()

        except Exception as e:
            self.parent.update_log(f"获取用户考勤记录失败: {str(e)}")
            return None