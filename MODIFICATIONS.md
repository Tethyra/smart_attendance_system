# 智能人脸识别考勤系统 - 修改说明

## 项目概述
本项目是基于https://github.com/Tethyra/smart_attendance_system的智能人脸识别考勤系统，已完成以下三项主要修改：

1. **修复人脸录入功能** - 解决了录入预览时没有摄像头的问题
2. **数据库迁移** - 从SQLite迁移到本地MySQL数据库（Navicat for MySQL）
3. **考勤功能优化** - 实现了考勤管理中点击"开始考勤"后自动打开摄像头界面的功能

## 修改详情

### 1. 修复人脸录入摄像头预览问题

**问题描述**: 原系统在人脸录入时，摄像头预览无法正常显示。

**解决方案**:
- 创建了专门用于录入的摄像头实例 `enrollment_camera`
- 实现了独立的预览更新机制，使用定时器每30ms更新一次预览画面
- 添加了 `update_enrollment_preview()` 方法来实时更新摄像头预览
- 确保了摄像头启动和停止的正确状态管理

**关键代码修改**:
```python
def start_enrollment_camera(self):
    # 创建专门用于录入的摄像头实例
    self.enrollment_camera = QCamera()
    # ... 其他初始化代码 ...
    
    # 创建定时器来更新预览
    self.enrollment_preview_timer = QTimer()
    self.enrollment_preview_timer.timeout.connect(self.update_enrollment_preview)
    self.enrollment_preview_timer.start(30)  # 30ms更新一次

def update_enrollment_preview(self):
    # 使用grab()方法获取当前帧并显示
    viewfinder_image = self.enrollment_camera_viewfinder.grab()
    if not viewfinder_image.isNull():
        self.enrollment_camera_display.setPixmap(
            viewfinder_image.scaled(self.enrollment_camera_display.size(), Qt.KeepAspectRatio)
        )
```

### 2. 迁移到MySQL数据库

**数据库配置**:
- 主机: localhost
- 数据库名: smart_attendance
- 用户名: root
- 密码: 123456

**实现的功能**:
- 自动创建数据库和必要的表结构
- 实现了完整的CRUD操作
- 支持数据库连接状态检测
- 提供了数据库连接测试功能

**创建的表结构**:
1. **users表** - 存储用户信息和人脸特征
2. **attendance表** - 存储考勤记录
3. **recognition_logs表** - 存储人脸识别日志

**关键代码修改**:
```python
def init_database(self):
    try:
        self.db_connection = mysql.connector.connect(
            host=self.config["mysql_host"],
            database=self.config["mysql_database"],
            user=self.config["mysql_user"],
            password=self.config["mysql_password"]
        )
        # ... 其他数据库初始化代码 ...
    except Error as e:
        # 错误处理和重试逻辑
```

### 3. 考勤管理自动打开摄像头

**功能实现**:
- 点击"开始考勤"按钮后，系统自动启动摄像头
- 实时显示摄像头预览画面
- 自动进行人脸识别和考勤记录
- 提供考勤状态实时反馈

**关键代码修改**:
```python
def start_attendance(self):
    # 自动启动考勤摄像头
    self.attendance_camera = QCamera()
    # ... 摄像头初始化代码 ...
    
    # 启动摄像头
    self.attendance_camera.start()
    
    # 创建预览定时器
    self.attendance_preview_timer = QTimer()
    self.attendance_preview_timer.timeout.connect(self.update_attendance_preview)
    self.attendance_preview_timer.start(30)
    
    # 创建识别定时器
    self.attendance_recognition_timer = QTimer()
    self.attendance_recognition_timer.timeout.connect(self.capture_and_recognize_attendance)
    self.attendance_recognition_timer.start(2000)
    
    # 显示摄像头界面
    self.attendance_camera_display.setVisible(True)
```

## 系统架构

### 主要模块
1. **人脸识别模块** - 使用Dlib进行人脸检测和特征提取
2. **摄像头管理模块** - 处理摄像头的启动、停止和预览
3. **数据库模块** - 管理MySQL数据库连接和操作
4. **考勤管理模块** - 处理考勤记录和报表生成
5. **用户界面模块** - 提供友好的PyQt5界面

### 数据流
1. 摄像头捕获图像 → 人脸检测 → 特征提取 → 特征比对 → 识别结果
2. 用户信息录入 → 人脸特征提取 → 数据库存储
3. 考勤启动 → 实时人脸识别 → 考勤记录 → 数据库更新

## 安装和使用

### 环境要求
- Python 3.7+
- MySQL 5.7+ 或 MariaDB
- 支持的操作系统: Windows 10/11, macOS 10.14+, Ubuntu 18.04+
- 摄像头设备

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/Tethyra/smart_attendance_system.git
cd smart_attendance_system
```

2. **创建虚拟环境**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **下载模型文件**
```bash
python download_models.py
```

5. **配置MySQL数据库**
   - 确保MySQL服务已启动
   - 系统会自动创建数据库和表结构

6. **启动系统**
```bash
python main.py
```

### 使用指南

#### 人脸录入
1. 在"人脸录入"标签页
2. 输入用户信息（姓名、年龄、性别、部门）
3. 点击"启动摄像头"按钮
4. 面对摄像头，点击"捕获人脸"
5. 点击"保存用户信息"完成录入

#### 人脸识别
1. 在"人脸识别"标签页
2. 点击"启动摄像头"
3. 点击"开始识别"
4. 系统将实时显示识别结果

#### 考勤管理
1. 在"考勤管理"标签页
2. 点击"开始考勤"按钮（摄像头会自动打开）
3. 系统将自动识别人脸并记录考勤
4. 点击"停止考勤"结束考勤

#### 数据管理
1. 在"数据管理"标签页查看所有用户
2. 可以更新或删除用户信息
3. 支持用户数据导出功能

## 配置说明

配置文件 `config.json` 包含以下主要配置项：

```json
{
    "threshold": 0.4,                // 人脸识别阈值
    "max_faces": 100,                // 最大人脸数量
    "camera_index": 0,               // 摄像头索引
    "mysql_host": "localhost",       // MySQL主机
    "mysql_database": "smart_attendance",  // 数据库名
    "mysql_user": "root",            // 数据库用户名
    "mysql_password": "123456"       // 数据库密码
}
```

## 故障排除

### 常见问题

1. **摄像头无法启动**
   - 检查摄像头是否被其他程序占用
   - 尝试修改 `camera_index` 配置
   - 检查摄像头驱动是否正常

2. **数据库连接失败**
   - 检查MySQL服务是否启动
   - 验证数据库用户名和密码
   - 确保数据库权限正确

3. **人脸识别准确率低**
   - 调整 `threshold` 识别阈值
   - 确保光线条件良好
   - 重新录入人脸特征

4. **模型文件缺失**
   - 运行 `python download_models.py` 下载模型
   - 或手动下载模型文件并配置路径

## 性能优化建议

1. **提高识别速度**
   - 减少同时识别的人脸数量
   - 降低摄像头分辨率
   - 调整识别间隔时间

2. **提高识别准确率**
   - 确保良好的光线条件
   - 使用正面清晰的人脸照片
   - 适当调整识别阈值

3. **系统性能优化**
   - 定期清理识别日志
   - 优化数据库查询
   - 关闭不必要的功能模块

## 安全考虑

1. **数据安全**
   - 人脸数据采用加密存储
   - 定期备份重要数据
   - 限制敏感操作的权限

2. **隐私保护**
   - 遵守相关隐私保护法规
   - 获得用户明确授权
   - 合理使用人脸数据

## 未来发展方向

1. **功能扩展**
   - 多人脸同时识别优化
   - 3D人脸识别支持
   - 移动端应用开发
   - 云端人脸识别服务

2. **技术改进**
   - 深度学习模型优化
   - 实时性能提升
   - 多平台适配
   - 离线识别支持

## 许可证

本项目采用MIT许可证，详情请参见LICENSE文件。

## 联系信息

如有问题或建议，请联系项目维护者。