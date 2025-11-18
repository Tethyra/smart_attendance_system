# api_service.py
import os
import json
import threading
import cv2
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime

# 尝试导入Flask相关模块
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS

    FLASK_AVAILABLE = True
except ImportError as e:
    FLASK_AVAILABLE = False
    print(f"Flask导入失败: {str(e)}")

# 尝试导入Waitress
try:
    from waitress import serve

    WAITRESS_AVAILABLE = True
except ImportError:
    WAITRESS_AVAILABLE = False


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
            if not FLASK_AVAILABLE:
                self.parent.update_log("Flask未安装，无法启动API服务")
                return False

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
        if not self.app:
            return

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
                    '/api/health': '健康检查',
                    '/api_test': 'API测试页面'
                }
            })

        @self.app.route('/api_test')
        def api_test_page():
            """API测试页面"""
            try:
                # 读取HTML文件内容
                html_file_path = os.path.join(os.path.dirname(__file__), 'api_test.html')
                if os.path.exists(html_file_path):
                    with open(html_file_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    return html_content
                else:
                    # 如果文件不存在，返回一个简单的测试页面
                    return f'''
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>智能人脸识别系统 - API测试</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 40px; }}
                            .container {{ max-width: 800px; margin: 0 auto; }}
                            .card {{ border: 1px solid #ddd; padding: 20px; margin: 10px 0; border-radius: 5px; }}
                            button {{ padding: 10px 15px; margin: 5px; background: #4CAF50; color: white; border: none; border-radius: 3px; cursor: pointer; }}
                            button:hover {{ background: #45a049; }}
                            .response {{ background: #f5f5f5; padding: 10px; margin-top: 10px; border-radius: 3px; }}
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <h1>智能人脸识别系统 - API测试页面</h1>
                            <p>API服务运行正常，端口: {self.port}</p>

                            <div class="card">
                                <h3>系统状态</h3>
                                <button onclick="healthCheck()">健康检查</button>
                                <button onclick="statusCheck()">系统状态</button>
                                <div class="response" id="statusResponse">等待响应...</div>
                            </div>

                            <div class="card">
                                <h3>说明</h3>
                                <p>完整的API测试页面请确保api_test.html文件存在于系统目录中。</p>
                                <p>当前API基础URL: http://localhost:{self.port}</p>
                            </div>
                        </div>

                        <script>
                            const API_BASE_URL = 'http://localhost:{self.port}';

                            async function healthCheck() {{
                                try {{
                                    const response = await fetch(API_BASE_URL + '/api/health');
                                    const data = await response.json();
                                    document.getElementById('statusResponse').textContent = JSON.stringify(data, null, 2);
                                }} catch (error) {{
                                    document.getElementById('statusResponse').textContent = '错误: ' + error.message;
                                }}
                            }}

                            async function statusCheck() {{
                                try {{
                                    const response = await fetch(API_BASE_URL + '/api/status');
                                    const data = await response.json();
                                    document.getElementById('statusResponse').textContent = JSON.stringify(data, null, 2);
                                }} catch (error) {{
                                    document.getElementById('statusResponse').textContent = '错误: ' + error.message;
                                }}
                            }}
                        </script>
                    </body>
                    </html>
                    '''
            except Exception as e:
                return f"测试页面加载失败: {str(e)}"

        @self.app.route('/api/status')
        def get_status():
            """获取系统状态"""
            try:
                status = {
                    'system_status': 'running',
                    'camera_status': 'running' if getattr(self.parent, 'is_camera_running', False) else 'stopped',
                    'recognition_status': 'running' if getattr(self.parent, 'is_recognizing', False) else 'stopped',
                    'attendance_status': 'running' if getattr(self.parent, 'is_attendance_running',
                                                              False) else 'stopped',
                    'total_users': getattr(self.parent, 'total_users', 0),
                    'total_recognitions': getattr(self.parent, 'total_recognitions', 0),
                    'total_attendance': getattr(self.parent, 'total_attendance', 0),
                    'model_status': getattr(self.parent, 'model_status', '未知'),
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
                if 'image' not in request.files and ('image_data' not in request.json if request.json else True):
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
                elif request.json and 'image_data' in request.json:
                    image_data = base64.b64decode(request.json['image_data'])

                if not image_data:
                    return jsonify({'status': 'error', 'message': '图片数据无效'})

                # 转换为PIL图像
                image = Image.open(io.BytesIO(image_data))

                # 进行人脸识别
                result = self.parent.recognize_face_from_image(image)

                if result['success']:
                    # 构建识别结果
                    recognitions = []
                    if 'descriptors' in result:
                        for i, descriptor in enumerate(result['descriptors']):
                            recognitions.append({
                                'face_id': i,
                                'descriptor': descriptor.tolist() if hasattr(descriptor, 'tolist') else descriptor
                            })

                    return jsonify({
                        'status': 'success',
                        'data': {
                            'face_count': result.get('face_count', 0),
                            'recognitions': recognitions,
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
                face_database = getattr(self.parent, 'face_database', {})
                for name, data in face_database.items():
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

            if not FLASK_AVAILABLE:
                self.parent.update_log("Flask未安装，无法启动API服务")
                return False

            if not self.app:
                if not self.init_api():
                    return False

            def run_server():
                try:
                    if WAITRESS_AVAILABLE:
                        # 使用Waitress生产服务器
                        serve(self.app, host='0.0.0.0', port=self.port, threads=4)
                    else:
                        # 使用Flask开发服务器
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
            self.parent.update_log(f"API测试页面: http://localhost:{self.port}/api_test")
            return True

        except Exception as e:
            self.parent.update_log(f"启动API服务失败: {str(e)}")
            return False

    def stop_service(self):
        """停止API服务"""
        try:
            # 注意：Flask开发服务器没有优雅关闭的方法
            # 这里主要是标记状态为停止
            self.is_running = False
            self.parent.update_log("API服务已停止")
            return True
        except Exception as e:
            self.parent.update_log(f"停止API服务失败: {str(e)}")
            return False


def create_production_api(parent):
    """创建生产环境API服务（使用Waitress）"""
    try:
        if not WAITRESS_AVAILABLE:
            parent.update_log("未安装Waitress，使用开发服务器")
            return None

        api = FaceRecognitionAPI(parent)
        if api.init_api():
            def run_production_server():
                serve(api.app, host='0.0.0.0', port=api.port, threads=4)

            api.server_thread = threading.Thread(target=run_production_server, daemon=True)
            api.server_thread.start()
            api.is_running = True
            parent.update_log(f"生产环境API服务启动成功，端口: {api.port}")
            return api
        return None
    except Exception as e:
        parent.update_log(f"创建生产环境API服务失败: {str(e)}")
        return None