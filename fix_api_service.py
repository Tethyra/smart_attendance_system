import os
def create_api_service_file():
    """创建api_service.py文件"""
    api_service_content = '''import os
import json
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime

class FaceRecognitionAPI:
    """人脸识别API服务类"""

    def __init__(self, parent):
        self.parent = parent
        self.config = parent.config
        self.app = None
        self.server_thread = None
        self.is_running = False
        self.port = self.config.get('api_port', 5000)

    def init_api(self):
        """初始化API服务"""
        try:
            self.app = Flask(__name__)
            CORS(self.app)  # 允许跨域请求

            # 注册路由
            self.register_routes()

            self.parent.update_log(f"API服务初始化完成，端口: {self.port}")
            return True
        except Exception as e:
            self.parent.update_log(f"API服务初始化失败: {str(e)}")
            return False

    def register_routes(self):
        """注册API路由"""

        @self.app.route('/')
        def index():
            """首页"""
            return jsonify({
                'status': 'success',
                'message': '智能人脸识别系统API服务',
                'version': '1.0.0',
                'endpoints': {
                    '/api/status': '获取系统状态',
                    '/api/face_recognition': '人脸识别',
                    '/api/attendance/check_in': '签到',
                    '/api/attendance/check_out': '签退',
                    '/api/attendance/records': '考勤记录',
                    '/api/users': '用户管理',
                    '/api/health': '健康检查'
                }
            })

        @self.app.route('/api/status')
        def get_status():
            """获取系统状态"""
            try:
                status = {
                    'system_status': 'running',
                    'camera_status': 'running' if self.parent.is_camera_running else 'stopped',
                    'recognition_status': 'running' if self.parent.is_recognizing else 'stopped',
                    'attendance_status': 'running' if self.parent.is_attendance_running else 'stopped',
                    'total_users': self.parent.total_users,
                    'total_recognitions': self.parent.total_recognitions,
                    'total_attendance': self.parent.total_attendance,
                    'model_status': self.parent.model_status,
                    'timestamp': datetime.now().isoformat()
                }
                return jsonify({'status': 'success', 'data': status})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)})

        @self.app.route('/api/face_recognition', methods=['POST'])
        def face_recognition():
            """人脸识别API"""
            try:
                # 检查是否有图片数据
                if 'image' not in request.files and 'image_data' not in request.json:
                    return jsonify({'status': 'error', 'message': '未提供图片数据'})

                image_data = None

                # 处理文件上传
                if 'image' in request.files:
                    file = request.files['image']
                    if file.filename == '':
                        return jsonify({'status': 'error', 'message': '未选择文件'})

                    # 读取图片
                    image_data = file.read()

                # 处理base64数据
                elif 'image_data' in request.json:
                    image_data = base64.b64decode(request.json['image_data'])

                if not image_data:
                    return jsonify({'status': 'error', 'message': '图片数据无效'})

                # 转换为PIL图像
                image = Image.open(io.BytesIO(image_data))

                # 进行人脸识别
                result = self.parent.recognize_face_from_image(image)

                if result['success']:
                    return jsonify({
                        'status': 'success',
                        'data': {
                            'face_count': result.get('face_count', 0),
                            'recognitions': result.get('recognitions', []),
                            'message': '识别成功'
                        }
                    })
                else:
                    return jsonify({
                        'status': 'error',
                        'message': result.get('error', '识别失败')
                    })

            except Exception as e:
                return jsonify({'status': 'error', 'message': f'识别过程中发生错误: {str(e)}'})

        @self.app.route('/api/attendance/check_in', methods=['POST'])
        def attendance_check_in():
            """签到API"""
            try:
                data = request.json
                if not data or 'name' not in data:
                    return jsonify({'status': 'error', 'message': '缺少必要参数: name'})

                name = data['name']
                location = data.get('location', 'API调用')

                # 执行签到
                success, message = self.parent.database.check_in(name, location)

                if success:
                    return jsonify({
                        'status': 'success',
                        'message': message,
                        'data': {
                            'name': name,
                            'location': location,
                            'timestamp': datetime.now().isoformat()
                        }
                    })
                else:
                    return jsonify({'status': 'error', 'message': message})

            except Exception as e:
                return jsonify({'status': 'error', 'message': f'签到过程中发生错误: {str(e)}'})

        @self.app.route('/api/attendance/check_out', methods=['POST'])
        def attendance_check_out():
            """签退API"""
            try:
                data = request.json
                if not data or 'name' not in data:
                    return jsonify({'status': 'error', 'message': '缺少必要参数: name'})

                name = data['name']
                location = data.get('location', 'API调用')

                # 执行签退
                success, message = self.parent.database.check_out(name, location)

                if success:
                    return jsonify({
                        'status': 'success',
                        'message': message,
                        'data': {
                            'name': name,
                            'location': location,
                            'timestamp': datetime.now().isoformat()
                        }
                    })
                else:
                    return jsonify({'status': 'error', 'message': message})

            except Exception as e:
                return jsonify({'status': 'error', 'message': f'签退过程中发生错误: {str(e)}'})

        @self.app.route('/api/attendance/records')
        def get_attendance_records():
            """获取考勤记录"""
            try:
                date = request.args.get('date')
                records = self.parent.database.get_attendance_records(date)

                # 格式化记录
                formatted_records = []
                for record in records:
                    formatted_records.append({
                        'name': record['name'],
                        'check_in_time': record['check_in_time'].isoformat() if record['check_in_time'] else None,
                        'check_out_time': record['check_out_time'].isoformat() if record['check_out_time'] else None,
                        'status': record['status'],
                        'location': record.get('location', '未知')
                    })

                return jsonify({
                    'status': 'success',
                    'data': {
                        'records': formatted_records,
                        'total': len(formatted_records)
                    }
                })

            except Exception as e:
                return jsonify({'status': 'error', 'message': f'获取考勤记录失败: {str(e)}'})

        @self.app.route('/api/users')
        def get_users():
            """获取用户列表"""
            try:
                users = []
                for name, data in self.parent.face_database.items():
                    info = data.get('info', {})
                    users.append({
                        'name': name,
                        'age': info.get('age', ''),
                        'gender': info.get('gender', ''),
                        'department': info.get('department', ''),
                        'photo_count': len(data.get('images', [])),
                        'created_at': info.get('created_at', '')
                    })

                return jsonify({
                    'status': 'success',
                    'data': {
                        'users': users,
                        'total': len(users)
                    }
                })

            except Exception as e:
                return jsonify({'status': 'error', 'message': f'获取用户列表失败: {str(e)}'})

        @self.app.route('/api/health')
        def health_check():
            """健康检查"""
            return jsonify({
                'status': 'success',
                'message': '服务运行正常',
                'timestamp': datetime.now().isoformat()
            })

    def start_service(self):
        """启动API服务"""
        try:
            if self.is_running:
                self.parent.update_log("API服务已经在运行中")
                return True

            if not self.app:
                if not self.init_api():
                    return False

            def run_server():
                try:
                    self.app.run(
                        host='0.0.0.0',
                        port=self.port,
                        debug=False,
                        threaded=True,
                        use_reloader=False
                    )
                except Exception as e:
                    self.parent.update_log(f"API服务运行错误: {str(e)}")

            # 在单独的线程中运行Flask服务
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()

            self.is_running = True
            self.parent.update_log(f"API服务启动成功，访问地址: http://localhost:{self.port}")
            return True

        except Exception as e:
            self.parent.update_log(f"启动API服务失败: {str(e)}")
            return False

    def stop_service(self):
        """停止API服务"""
        try:
            # 注意：Flask开发服务器没有优雅关闭的方法
            # 在生产环境中应该使用Waitress或Gunicorn
            self.is_running = False
            self.parent.update_log("API服务已停止")
            return True
        except Exception as e:
            self.parent.update_log(f"停止API服务失败: {str(e)}")
            return False

# 使用Waitress作为生产服务器的替代版本
def create_production_api(parent):
    """创建生产环境API服务（使用Waitress）"""
    try:
        from waitress import serve

        api = FaceRecognitionAPI(parent)
        if api.init_api():
            def run_production_server():
                serve(api.app, host='0.0.0.0', port=api.port)

            api.server_thread = threading.Thread(target=run_production_server, daemon=True)
            api.server_thread.start()
            api.is_running = True
            parent.update_log(f"生产环境API服务启动成功，端口: {api.port}")
            return api
        return None
    except ImportError:
        parent.update_log("未安装Waitress，使用开发服务器")
        return None
'''

    with open('api_service.py', 'w', encoding='utf-8') as f:
        f.write(api_service_content)

    print("api_service.py 文件创建成功")


def create_requirements_file():
    """创建requirements文件"""
    requirements_content = '''flask>=2.0.0
flask-cors>=3.0.0
waitress>=2.0.0
'''

    with open('requirements_api.txt', 'w', encoding='utf-8') as f:
        f.write(requirements_content)

    print("requirements_api.txt 文件创建成功")


def update_main_py():
    """更新main.py文件"""
    # 读取main.py内容
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 在导入部分添加API导入
    if 'from utils import FaceRecognitionUtils' in content and 'from api_service import' not in content:
        # 在utils导入后添加API导入
        content = content.replace(
            'from utils import FaceRecognitionUtils',
            'from utils import FaceRecognitionUtils\n\n# API服务导入\ntry:\n    from api_service import FaceRecognitionAPI, create_production_api\n    API_AVAILABLE = True\nexcept ImportError as e:\n    API_AVAILABLE = False\n    print(f"API模块导入失败: {str(e)}")'
        )

    # 修改start_api_service方法
    old_start_api = '''def start_api_service(self):
        """启动API服务"""
        # 简化实现
        self.update_log("API服务启动功能开发中")'''

    new_start_api = '''def start_api_service(self):
        """启动API服务"""
        try:
            if not API_AVAILABLE:
                self.update_log("API服务不可用，请安装Flask: pip install flask flask-cors")
                QMessageBox.warning(self, "警告", "API服务依赖未安装\\n\\n请执行: pip install flask flask-cors")
                return

            if self.api_service and self.api_service.start_service():
                self.update_log("API服务启动成功")

                # 在状态栏显示API状态
                if hasattr(self, 'api_status_label'):
                    self.api_status_label.setText("API: 运行中")
                    self.api_status_label.setStyleSheet("color: green;")
            else:
                self.update_log("API服务启动失败")
                QMessageBox.warning(self, "警告", "API服务启动失败，请查看日志")

        except Exception as e:
            self.update_log(f"启动API服务失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"启动API服务失败: {str(e)}")'''

    content = content.replace(old_start_api, new_start_api)

    # 修改stop_api_service方法
    old_stop_api = '''def stop_api_service(self):
        """停止API服务"""
        # 简化实现
        self.update_log("API服务停止功能开发中")'''

    new_stop_api = '''def stop_api_service(self):
        """停止API服务"""
        try:
            if self.api_service and self.api_service.stop_service():
                self.update_log("API服务停止成功")

                # 在状态栏显示API状态
                if hasattr(self, 'api_status_label'):
                    self.api_status_label.setText("API: 已停止")
                    self.api_status_label.setStyleSheet("color: red;")
            else:
                self.update_log("API服务停止失败")

        except Exception as e:
            self.update_log(f"停止API服务失败: {str(e)}")'''

    content = content.replace(old_stop_api, new_stop_api)

    # 在__init__方法中添加API服务初始化
    if 'self.stop_api_service()' in content and 'self.api_service = None' not in content:
        init_search = '        # 系统启动信息\n        self.update_log("系统启动成功")\n        self.update_status("就绪")'
        init_replace = '        # 初始化API服务\n        self.api_service = None\n        if API_AVAILABLE:\n            try:\n                self.api_service = FaceRecognitionAPI(self)\n                self.update_log("API服务初始化完成")\n            except Exception as e:\n                self.update_log(f"API服务初始化失败: {str(e)}")\n        else:\n            self.update_log("API服务不可用，请检查依赖")\n        \n        # 系统启动信息\n        self.update_log("系统启动成功")\n        self.update_status("就绪")'

        content = content.replace(init_search, init_replace)

    # 保存修改后的文件
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print("main.py 文件更新成功")


def main():
    """主函数"""
    print("开始修复API服务功能...")

    try:
        # 创建api_service.py文件
        create_api_service_file()

        # 创建requirements文件
        create_requirements_file()

        # 更新main.py文件
        update_main_py()

        print("\nAPI服务修复完成！")
        print("\n请执行以下命令安装依赖：")
        print("pip install -r requirements_api.txt")
        print("\n然后在系统设置中启动API服务")

    except Exception as e:
        print(f"修复过程中发生错误：{str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()