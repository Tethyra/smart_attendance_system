import os
import json


class FaceRecognitionConfig:
    """配置文件管理类"""

    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        """加载配置文件"""
        default_config = {
            'database_path': 'face_database',
            'features_file': 'face_features.csv',
            'log_file': 'recognition_log.csv',
            'attendance_file': 'attendance.csv',
            'model_path': 'models',
            'shape_predictor_path': 'models/shape_predictor_68_face_landmarks.dat',
            'face_recognition_model_path': 'models/dlib_face_recognition_resnet_model_v1.dat',
            'threshold': 0.4,
            'max_faces': 100,
            'api_port': 5000,
            'camera_index': 0,
            'use_local_models_only': True,

            # MySQL数据库配置
            'mysql_host': 'localhost',
            'mysql_port': 3306,
            'mysql_user': 'root',
            'mysql_password': '123456',
            'mysql_database': 'smart_attendance',

            # 优化识别稳定性参数
            'recognition_stability_threshold': 0.3,  # 提高稳定性阈值，从0.15提高到0.3
            'max_recognition_history': 3,  # 减少历史记录数量，从5降到3
            'face_images_per_person': 5,  # 每人最多保存照片数量
            'min_face_size': 100,  # 最小人脸尺寸
            'recognition_fix_threshold': 0.6,  # 降低固定阈值，从0.8降到0.6，更容易固定结果
            'min_stable_frames': 2,  # 减少最少稳定帧数，从3降到2
            'auto_save_after_enrollment': True,  # 修复需求2：人脸录入后自动保存
            'checkout_recognition_required': True,  # 修复需求3：签退需要人脸识别确认

            # 新增：人脸丢失检测参数
            'face_lost_threshold': 5,  # 连续多少帧未检测到人脸算作丢失
            'recognition_pause_enabled': True  # 启用暂停识别功能
        }

        try:
            if os.path.exists('config.json'):
                with open('config.json', 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # 合并配置
                    for key, value in user_config.items():
                        if key in default_config:
                            default_config[key] = value
        except Exception as e:
            print(f"加载配置文件失败: {str(e)}")

        return default_config

    def get(self, key, default=None):
        """获取配置值"""
        return self.config.get(key, default)

    def __getitem__(self, key):
        """获取配置值"""
        return self.config[key]

    def __setitem__(self, key, value):
        """设置配置值"""
        self.config[key] = value

    def save_settings(self, parent):
        """保存设置"""
        try:
            # 更新配置
            self.config['shape_predictor_path'] = parent.shape_predictor_edit.text()
            self.config['face_recognition_model_path'] = parent.face_recognizer_edit.text()
            self.config['use_local_models_only'] = parent.use_local_checkbox.isChecked()

            # MySQL配置
            self.config['mysql_host'] = parent.mysql_host_edit.text()
            self.config['mysql_port'] = int(parent.mysql_port_edit.text())
            self.config['mysql_user'] = parent.mysql_user_edit.text()
            self.config['mysql_password'] = parent.mysql_password_edit.text()
            self.config['mysql_database'] = parent.mysql_database_edit.text()

            # 识别配置
            self.config['threshold'] = float(parent.threshold_edit.text())
            self.config['recognition_stability_threshold'] = float(parent.stability_threshold_edit.text())
            self.config['recognition_fix_threshold'] = float(parent.fix_threshold_edit.text())
            self.config['min_stable_frames'] = int(parent.min_stable_frames_edit.text())
            self.config['min_face_size'] = int(parent.min_face_size_edit.text())

            # 系统配置
            self.config['api_port'] = int(parent.api_port_edit.text())
            self.config['camera_index'] = int(parent.camera_index_edit.text())
            self.config['face_images_per_person'] = int(parent.face_images_per_person_edit.text())

            # 修复需求2：保存自动保存设置
            self.config['auto_save_after_enrollment'] = parent.auto_save_checkbox_setting.isChecked()
            parent.auto_save_checkbox.setChecked(self.config['auto_save_after_enrollment'])

            # 修复需求3：保存签退人脸识别设置
            self.config['checkout_recognition_required'] = parent.checkout_recognition_checkbox_setting.isChecked()
            parent.checkout_recognition_checkbox.setChecked(self.config['checkout_recognition_required'])

            # 保存到文件
            with open('config.json', 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)

            # 重新初始化模型
            parent.models.init_models()

            # 重新连接数据库
            parent.database.init_database()

            parent.update_log("设置保存成功")

        except Exception as e:
            parent.update_log(f"保存设置失败: {str(e)}")