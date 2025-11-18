
import os
import sys
import uuid
import logging
import numpy as np
from datetime import datetime, timedelta
from PIL import Image, ImageOps
import face_recognition

# 尝试导入tkinter，如果失败则提供备选方案
try:
    from tkinter import filedialog

    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    logging.warning("tkinter模块未找到，将使用命令行方式选择文件")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('attendance_system.log'),
        logging.StreamHandler()
    ]
)

# 系统配置
CONFIG = {
    'RECOGNITION_THRESHOLD': 0.8,  # 80%置信度阈值
    'SIMILARITY_THRESHOLD': 0.7,  # 人脸相似度阈值（重复检测）
    'MIN_FACE_SIZE': 100,  # 最小人脸尺寸
    'MAX_FACE_SIZE': 800,  # 最大人脸尺寸
    'IMAGE_QUALITY': 90,  # 图片保存质量
    'TEMP_DIR': 'temp_images',  # 临时文件目录
    'MAX_REGISTRATION_IMAGES': 5,  # 每人最大注册照片数
}

# 创建临时目录
os.makedirs(CONFIG['TEMP_DIR'], exist_ok=True)


class FaceAttendanceFixer:
    def __init__(self, db_connection):
        self.db = db_connection
        self.setup_database()

    def setup_database(self):
        """设置数据库，确保表结构正确"""
        try:
            # 检查必要字段是否存在
            self._ensure_table_columns()
            logging.info("数据库结构检查完成")
        except Exception as e:
            logging.error(f"数据库设置失败: {str(e)}")

    def _ensure_table_columns(self):
        """确保表结构正确"""
        # 检查attendance表字段
        try:
            cursor = self.db.cursor()
            cursor.execute("DESCRIBE attendance")
            columns = [col[0] for col in cursor.fetchall()]

            # 检查并添加缺失字段
            if 'checkin_recognition_confidence' not in columns:
                cursor.execute("""
                    ALTER TABLE attendance 
                    ADD COLUMN checkin_recognition_confidence DECIMAL(10, 6) NULL COMMENT '签到识别置信度'
                """)

            if 'checkout_recognition_confidence' not in columns:
                cursor.execute("""
                    ALTER TABLE attendance 
                    ADD COLUMN checkout_recognition_confidence DECIMAL(10, 6) NULL COMMENT '签出识别置信度'
                """)

            self.db.commit()
        except Exception as e:
            logging.warning(f"数据库字段检查失败: {str(e)}")

    def select_image_files(self, file_paths=None):
        """
        修复的图片选择功能
        如果提供file_paths参数，则直接使用这些路径
        否则尝试使用GUI或命令行选择文件
        """
        try:
            # 如果直接提供了文件路径，则使用它们
            if file_paths and isinstance(file_paths, list) and len(file_paths) > 0:
                logging.info(f"使用提供的文件路径，共 {len(file_paths)} 个文件")
                return self._process_image_files(file_paths)

            # 如果没有提供文件路径，则尝试选择文件
            if TKINTER_AVAILABLE:
                return self._select_images_gui()
            else:
                return self._select_images_commandline()

        except Exception as e:
            logging.error(f"文件选择失败: {str(e)}")
            return []

    def _select_images_gui(self):
        """使用GUI选择图片文件"""
        try:
            # 使用更通用的文件选择对话框
            file_paths = filedialog.askopenfilenames(
                title="选择人脸照片",
                filetypes=[
                    ("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                    ("All Files", "*.*")
                ]
            )

            if not file_paths:
                logging.info("未选择任何文件")
                return []

            return self._process_image_files(file_paths)

        except Exception as e:
            logging.error(f"GUI文件选择失败: {str(e)}")
            # 如果GUI失败，回退到命令行方式
            return self._select_images_commandline()

    def _select_images_commandline(self):
        """使用命令行选择图片文件"""
        try:
            print("请输入图片文件路径（多个路径用空格分隔）:")
            input_paths = input().strip()

            if not input_paths:
                logging.info("未输入任何文件路径")
                return []

            file_paths = input_paths.split()
            return self._process_image_files(file_paths)

        except Exception as e:
            logging.error(f"命令行文件选择失败: {str(e)}")
            return []

    def _process_image_files(self, file_paths):
        """处理选中的图片文件"""
        valid_images = []

        for file_path in file_paths:
            try:
                # 检查文件是否存在
                if not os.path.exists(file_path):
                    logging.warning(f"文件不存在: {file_path}")
                    continue

                # 验证图片格式
                with Image.open(file_path) as img:
                    logging.info(f"处理图片: {file_path}, 格式: {img.mode}")

                    # 转换为RGB格式（如果不是）
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        logging.info(f"转换为RGB格式")

                    # 调整图片大小（保持比例）
                    img = self.resize_image_keep_ratio(img, CONFIG['MAX_FACE_SIZE'], CONFIG['MAX_FACE_SIZE'])

                    # 保存临时文件
                    temp_filename = f"temp_{uuid.uuid4()}.jpg"
                    temp_path = os.path.join(CONFIG['TEMP_DIR'], temp_filename)
                    img.save(temp_path, 'JPEG', quality=CONFIG['IMAGE_QUALITY'])

                    valid_images.append(temp_path)
                    logging.info(f"图片处理成功: {temp_path}")

            except Exception as e:
                logging.warning(f"图片处理失败 {file_path}: {str(e)}")
                continue

        logging.info(f"成功处理 {len(valid_images)} 张图片")
        return valid_images

    def resize_image_keep_ratio(self, image, max_width, max_height):
        """按比例调整图片大小"""
        try:
            ratio = min(max_width / image.width, max_height / image.height)
            new_width = int(image.width * ratio)
            new_height = int(image.height * ratio)

            # 使用高质量重采样
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logging.info(f"图片大小调整: {image.width}x{image.height} -> {new_width}x{new_height}")
            return resized_image
        except Exception as e:
            logging.error(f"图片调整失败: {str(e)}")
            return image

    def preprocess_image(self, image_path):
        """图片预处理"""
        try:
            with Image.open(image_path) as img:
                # 转换为RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # 转换为灰度图用于人脸检测
                gray = img.convert('L')

                # 直方图均衡化增强对比度
                gray = ImageOps.equalize(gray)

                # 转换为numpy数组
                np_image = np.array(gray)

                logging.info(f"图片预处理完成: {image_path}")
                return np_image

        except Exception as e:
            logging.error(f"图片预处理失败 {image_path}: {str(e)}")
            raise

    def detect_faces(self, image_path):
        """人脸检测"""
        try:
            # 读取图片
            image = face_recognition.load_image_file(image_path)

            # 人脸检测
            face_locations = face_recognition.face_locations(image)

            if not face_locations:
                logging.warning(f"未检测到人脸: {image_path}")
                return []

            logging.info(f"检测到 {len(face_locations)} 个人脸: {image_path}")

            # 过滤小脸
            filtered_locations = []
            for (top, right, bottom, left) in face_locations:
                face_width = right - left
                face_height = bottom - top

                if face_width >= CONFIG['MIN_FACE_SIZE'] and face_height >= CONFIG['MIN_FACE_SIZE']:
                    filtered_locations.append((top, right, bottom, left))
                    logging.info(f"有效人脸尺寸: {face_width}x{face_height}")
                else:
                    logging.warning(f"人脸尺寸过小: {face_width}x{face_height} < {CONFIG['MIN_FACE_SIZE']}")

            return filtered_locations

        except Exception as e:
            logging.error(f"人脸检测失败 {image_path}: {str(e)}")
            return []

    def extract_face_encoding(self, image_path, face_location=None):
        """提取人脸特征"""
        try:
            image = face_recognition.load_image_file(image_path)

            if face_location:
                # 指定人脸位置提取特征
                encodings = face_recognition.face_encodings(image, [face_location])
            else:
                # 自动检测人脸并提取特征
                encodings = face_recognition.face_encodings(image)

            if not encodings:
                logging.warning(f"无法提取人脸特征: {image_path}")
                return None

            logging.info(f"人脸特征提取成功: {image_path}")
            return encodings[0]

        except Exception as e:
            logging.error(f"人脸特征提取失败 {image_path}: {str(e)}")
            return None

    def load_known_faces(self):
        """加载已知人脸特征"""
        try:
            cursor = self.db.cursor()
            cursor.execute("""
                SELECT u.id, u.face_encoding 
                FROM users u 
                WHERE u.face_encoding IS NOT NULL AND u.status = 1
            """)

            results = cursor.fetchall()

            known_encodings = []
            known_user_ids = []

            for user_id, face_encoding_str in results:
                try:
                    # 转换特征字符串为numpy数组
                    face_encoding = np.fromstring(face_encoding_str, dtype=np.float64, sep=',')
                    known_encodings.append(face_encoding)
                    known_user_ids.append(user_id)
                except Exception as e:
                    logging.warning(f"用户 {user_id} 特征解析失败: {str(e)}")
                    continue

            logging.info(f"加载了 {len(known_encodings)} 个已知人脸特征")
            return known_encodings, known_user_ids

        except Exception as e:
            logging.error(f"加载已知人脸失败: {str(e)}")
            return [], []

    def check_duplicate_face(self, new_face_encoding):
        """检查是否为重复人脸"""
        try:
            # 加载已注册的人脸特征
            known_encodings, known_user_ids = self.load_known_faces()

            if not known_encodings:
                logging.info("无已注册用户，不是重复")
                return False, None, 0.0

            # 计算与所有已知人脸的相似度
            distances = face_recognition.face_distance(known_encodings, new_face_encoding)
            min_distance = min(distances)
            similarity = 1 - min_distance
            best_match_index = np.argmin(distances)

            logging.info(f"最佳匹配相似度: {similarity:.4f}, 阈值: {CONFIG['SIMILARITY_THRESHOLD']}")

            # 检查是否超过相似度阈值
            if similarity >= CONFIG['SIMILARITY_THRESHOLD']:
                duplicate_user_id = known_user_ids[best_match_index]
                logging.warning(f"检测到重复人脸，用户ID: {duplicate_user_id}, 相似度: {similarity:.4f}")
                return True, duplicate_user_id, similarity
            else:
                return False, None, similarity

        except Exception as e:
            logging.error(f"重复人脸检查失败: {str(e)}")
            return False, None, 0.0

    def recognize_face_with_confidence(self, image_path):
        """带置信度过滤的人脸识别"""
        try:
            # 人脸检测
            faces = self.detect_faces(image_path)
            if not faces:
                return None, "未检测到人脸", 0.0

            # 提取特征
            face_encoding = self.extract_face_encoding(image_path, faces[0])
            if face_encoding is None:
                return None, "人脸特征提取失败", 0.0

            # 与数据库中的人脸比对
            known_encodings, known_user_ids = self.load_known_faces()

            if not known_encodings:
                return None, "暂无注册用户", 0.0

            # 计算相似度
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            min_distance = min(distances)
            confidence = 1 - min_distance
            best_match_index = np.argmin(distances)

            logging.info(f"识别结果 - 置信度: {confidence:.4f}, 阈值: {CONFIG['RECOGNITION_THRESHOLD']}")

            # 应用置信度阈值
            if confidence >= CONFIG['RECOGNITION_THRESHOLD']:
                user_id = known_user_ids[best_match_index]
                return user_id, f"识别成功", confidence
            else:
                return None, f"识别失败，置信度不足", confidence

        except Exception as e:
            logging.error(f"人脸识别失败: {str(e)}")
            return None, str(e), 0.0

    def register_user_with_duplicate_check(self, name, age, gender, department, face_images):
        """带重复检查的用户注册"""
        try:
            # 验证输入
            if not all([name, face_images]):
                logging.error("注册失败：姓名和人脸照片不能为空")
                return False, "姓名和人脸照片不能为空"

            if len(face_images) > CONFIG['MAX_REGISTRATION_IMAGES']:
                msg = f"每人最多注册{CONFIG['MAX_REGISTRATION_IMAGES']}张照片"
                logging.error(f"注册失败：{msg}")
                return False, msg

            # 处理人脸照片
            face_encodings = []
            valid_image_paths = []

            for img_path in face_images:
                try:
                    # 人脸检测
                    faces = self.detect_faces(img_path)
                    if not faces:
                        logging.warning(f"跳过无脸图片: {img_path}")
                        continue

                    # 提取特征
                    encoding = self.extract_face_encoding(img_path, faces[0])
                    if encoding is None:
                        logging.warning(f"特征提取失败: {img_path}")
                        continue

                    face_encodings.append(encoding)
                    valid_image_paths.append(img_path)

                except Exception as e:
                    logging.warning(f"图片处理失败 {img_path}: {str(e)}")
                    continue

            if not face_encodings:
                logging.error("注册失败：未成功处理任何人脸照片")
                return False, "未成功处理任何人脸照片"

            # 检查是否为重复用户
            duplicate_found = False
            duplicate_user_id = None
            max_similarity = 0.0

            for encoding in face_encodings:
                is_duplicate, user_id, similarity = self.check_duplicate_face(encoding)
                if is_duplicate:
                    duplicate_found = True
                    duplicate_user_id = user_id
                    max_similarity = max(max_similarity, similarity)
                    break

            if duplicate_found:
                # 获取重复用户信息
                user_info = self.get_user_info(duplicate_user_id)
                msg = f"该人脸已注册，用户: {user_info.get('name', '未知用户')} (相似度: {max_similarity:.2%})"
                logging.error(f"注册失败：{msg}")
                return False, msg

            # 计算平均特征（多图注册）
            avg_encoding = np.mean(face_encodings, axis=0)
            encoding_str = ','.join(map(str, avg_encoding))

            # 保存用户信息
            cursor = self.db.cursor()
            current_time = datetime.now()

            # 插入用户记录
            cursor.execute("""
                INSERT INTO users (name, age, gender, department, face_encoding, 
                                 created_at, updated_at, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 1)
            """, (name, age, gender, department, encoding_str, current_time, current_time))

            user_id = cursor.lastrowid
            logging.info(f"用户创建成功，ID: {user_id}")

            # 保存人脸图片
            for i, img_path in enumerate(valid_image_paths):
                is_primary = 1 if i == 0 else 0  # 第一张设为主照片

                # 生成保存路径
                img_filename = f"user_{user_id}_{uuid.uuid4()}.jpg"
                save_dir = os.path.join("face_images", str(user_id))
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, img_filename)

                # 复制图片
                with open(img_path, 'rb') as src_file:
                    with open(save_path, 'wb') as dst_file:
                        dst_file.write(src_file.read())

                # 插入图片记录
                cursor.execute("""
                    INSERT INTO face_images (user_id, image_path, timestamp, is_primary)
                    VALUES (%s, %s, %s, %s)
                """, (user_id, save_path, current_time, is_primary))

                logging.info(f"人脸图片保存成功: {save_path}")

            self.db.commit()
            logging.info(f"用户注册完成，ID: {user_id}")
            return True, f"用户注册成功，ID: {user_id}"

        except Exception as e:
            self.db.rollback()
            logging.error(f"用户注册失败: {str(e)}", exc_info=True)
            return False, str(e)

    def get_user_info(self, user_id):
        """获取用户信息"""
        try:
            cursor = self.db.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            return cursor.fetchone() or {}
        except Exception as e:
            logging.error(f"获取用户信息失败: {str(e)}")
            return {}

    def save_attendance_record(self, user_id, check_type, confidence):
        """保存考勤记录，包含置信度"""
        try:
            cursor = self.db.cursor()
            current_time = datetime.now()

            if check_type == 'checkin':
                # 签到记录
                cursor.execute("""
                    INSERT INTO attendance (user_id, check_in_time, status, 
                                         checkin_recognition_confidence)
                    VALUES (%s, %s, 'checked_in', %s)
                """, (user_id, current_time, confidence))

                record_id = cursor.lastrowid
                logging.info(f"签到记录保存成功，ID: {record_id}, 置信度: {confidence:.4f}")

            elif check_type == 'checkout':
                # 签出记录（更新现有签到记录）
                cursor.execute("""
                    UPDATE attendance 
                    SET check_out_time = %s, status = 'checked_out',
                        checkout_recognition_confidence = %s
                    WHERE user_id = %s AND check_out_time IS NULL
                    ORDER BY check_in_time DESC LIMIT 1
                """, (current_time, confidence, user_id))

                if cursor.rowcount > 0:
                    logging.info(f"签出记录更新成功，置信度: {confidence:.4f}")
                else:
                    logging.warning(f"未找到待签出的签到记录: 用户 {user_id}")
                    return False, "未找到待签出的签到记录"

            self.db.commit()
            return True, "考勤记录保存成功"

        except Exception as e:
            self.db.rollback()
            logging.error(f"考勤记录保存失败: {str(e)}")
            return False, str(e)

    def generate_attendance_report(self, date=None):
        """生成考勤报表，过滤低置信度记录"""
        try:
            if date is None:
                date = datetime.now().date()

            start_time = datetime.combine(date, datetime.min.time())
            end_time = datetime.combine(date, datetime.max.time())

            logging.info(f"生成考勤报表: {date}")

            cursor = self.db.cursor(dictionary=True)
            cursor.execute("""
                SELECT u.id as user_id, u.name, u.department,
                       a.check_in_time, a.check_out_time, a.status,
                       a.checkin_recognition_confidence, a.checkout_recognition_confidence
                FROM attendance a
                JOIN users u ON a.user_id = u.id
                WHERE a.check_in_time BETWEEN %s AND %s
                AND (a.checkin_recognition_confidence >= %s OR a.checkin_recognition_confidence IS NULL)
                ORDER BY a.check_in_time DESC
            """, (start_time, end_time, CONFIG['RECOGNITION_THRESHOLD']))

            results = cursor.fetchall()
            logging.info(f"查询到 {len(results)} 条考勤记录")

            # 处理报表数据
            report_data = []
            for row in results:
                # 计算工作时长
                work_hours = 0
                if row['check_out_time']:
                    time_diff = row['check_out_time'] - row['check_in_time']
                    work_hours = time_diff.total_seconds() / 3600

                report_data.append({
                    'user_id': row['user_id'],
                    'name': row['name'],
                    'department': row['department'],
                    'check_in_time': row['check_in_time'].strftime('%Y-%m-%d %H:%M:%S') if row['check_in_time'] else '',
                    'check_out_time': row['check_out_time'].strftime('%Y-%m-%d %H:%M:%S') if row[
                        'check_out_time'] else '',
                    'status': row['status'],
                    'work_hours': f"{work_hours:.2f}" if work_hours > 0 else '0.00',
                    'checkin_confidence': f"{row['checkin_recognition_confidence']:.2%}" if row[
                        'checkin_recognition_confidence'] else "N/A",
                    'checkout_confidence': f"{row['checkout_recognition_confidence']:.2%}" if row[
                        'checkout_recognition_confidence'] else "N/A"
                })

            # 统计信息
            total_records = len(report_data)
            checked_in = len([r for r in report_data if r['status'] in ['checked_in', 'checked_out']])
            checked_out = len([r for r in report_data if r['status'] == 'checked_out'])
            attendance_rate = (checked_in / total_records * 100) if total_records > 0 else 0

            summary = {
                'date': date.strftime('%Y-%m-%d'),
                'total_records': total_records,
                'checked_in': checked_in,
                'checked_out': checked_out,
                'attendance_rate': f"{attendance_rate:.1f}%"
            }

            logging.info(f"考勤报表生成完成 - {summary}")
            return report_data, summary

        except Exception as e:
            logging.error(f"生成考勤报表失败: {str(e)}")
            raise

    def cleanup_temp_files(self):
        """清理临时文件"""
        try:
            temp_dir = CONFIG['TEMP_DIR']
            if os.path.exists(temp_dir):
                for filename in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            logging.info(f"清理临时文件: {file_path}")
                    except Exception as e:
                        logging.warning(f"临时文件清理失败 {file_path}: {str(e)}")
        except Exception as e:
            logging.error(f"临时文件清理失败: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # 这里需要数据库连接
    # fixer = FaceAttendanceFixer(db_connection)

    # 示例：测试图片选择功能（直接提供文件路径）
    # test_images = ["/path/to/test1.jpg", "/path/to/test2.jpg"]
    # processed_images = fixer.select_image_files(test_images)
    # print(f"处理了 {len(processed_images)} 张图片")

    logging.info("人脸考勤系统修复模块加载完成")
    if not TKINTER_AVAILABLE:
        logging.warning("注意：tkinter模块未安装，将使用命令行或直接文件路径方式操作")