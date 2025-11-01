# 智能人脸识别系统

## 项目简介

智能人脸识别系统是一个基于Python开发的综合性人脸识别解决方案，集成了多种创新功能，提供了完整的人脸识别、录入、管理和考勤功能。

## 创新功能特色

### 🚀 核心功能
- **人脸识别**：实时摄像头识别、图片识别、视频识别
- **人脸录入**：支持摄像头录入和图片录入
- **数据管理**：人脸数据的增删改查操作
- **考勤管理**：智能考勤系统，自动统计报表

### 🎯 创新功能
- **情感识别**：基于面部特征判断情绪状态
- **年龄性别预测**：预测人脸的年龄和性别
- **口罩检测**：实时检测是否佩戴口罩
- **实时人脸跟踪**：多目标跟踪算法
- **人脸识别API**：提供RESTful API接口
- **人脸识别日志**：完整的识别记录和审计
- **数据可视化**：考勤统计图表展示

### 🔧 技术特点
- **无OpenCV依赖**：使用PIL/Pillow替代OpenCV进行图像处理
- **跨平台支持**：支持Windows、macOS、Linux
- **模块化设计**：清晰的代码结构，便于维护和扩展
- **美观界面**：现代化的PyQt5界面设计
- **数据库支持**：SQLite数据库存储用户信息

## 本地模型加载配置

### 模型文件说明

系统需要以下两个Dlib模型文件：
1. `shape_predictor_68_face_landmarks.dat` - 人脸特征点预测器
2. `dlib_face_recognition_resnet_model_v1.dat` - 人脸识别模型

### 本地模型配置方法

#### 方法一：自动下载（推荐）
```bash
# 运行模型下载器
python download_models.py
```
系统会自动下载模型到`models/`目录

#### 方法二：手动配置
1. **获取模型文件**
   - 从官方网站下载：http://dlib.net/files/
   - 或通过其他渠道获取

2. **配置模型路径**
   - 打开系统设置界面
   - 在"模型设置"部分点击"浏览"按钮
   - 分别选择两个模型文件的路径

3. **启用本地模型模式**
   - 勾选"仅使用本地模型（禁用自动下载）"选项
   - 点击"保存设置"

### 模型状态检查

系统会自动检查模型完整性：
- **完整**：两个模型都正常加载
- **部分完整**：只有一个模型加载成功
- **不完整**：模型文件缺失或损坏

### 配置文件设置

也可以直接编辑`config.json`文件：
```json
{
    "shape_predictor_path": "path/to/shape_predictor_68_face_landmarks.dat",
    "face_recognition_model_path": "path/to/dlib_face_recognition_resnet_model_v1.dat",
    "use_local_models_only": true
}
```

## 技术栈

- **编程语言**：Python 3.7+
- **GUI框架**：PyQt5
- **人脸识别**：Dlib
- **图像处理**：Pillow
- **数据库**：SQLite
- **数据可视化**：Matplotlib
- **API服务**：Flask
- **其他库**：numpy、requests、json等

## 安装指南

### 环境要求
- Python 3.7 或更高版本
- 支持的操作系统：Windows 10/11、macOS 10.14+、Ubuntu 18.04+
- 摄像头（用于实时人脸识别）

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/yourusername/smart-face-recognition.git
cd smart-face-recognition
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

5. **启动系统**
```bash
python main.py
```

## 使用说明

### 1. 人脸识别功能

**摄像头识别**
- 点击"启动摄像头"按钮
- 点击"开始识别"按钮
- 系统将实时显示识别结果，包括：
  - 识别姓名
  - 置信度
  - 年龄/性别预测
  - 情绪状态
  - 口罩检测状态

**图片识别**
- 点击"选择图片"按钮
- 选择要识别的图片文件
- 系统将自动识别人脸并显示结果

### 2. 人脸录入功能

**摄像头录入**
- 在"人脸录入"标签页
- 输入用户信息（姓名、年龄、性别、部门）
- 点击"启动摄像头录入"
- 面对摄像头，点击"捕获人脸"
- 点击"保存用户"完成录入

**图片录入**
- 点击"选择图片录入"
- 选择包含人脸的图片
- 系统将自动检测人脸
- 点击"保存用户"完成录入

### 3. 数据管理功能

**查看人脸列表**
- 在"数据管理"标签页查看所有录入的人脸
- 点击列表项查看详细信息

**编辑功能**
- 选择要操作的人脸
- 点击"更新选中"或"删除选中"按钮
- 支持数据导出功能

### 4. 考勤管理功能

**开始考勤**
- 在"考勤管理"标签页
- 点击"开始考勤"按钮
- 系统将自动记录识别到的人员考勤

**生成报表**
- 点击"生成报表"按钮
- 系统将生成考勤统计图表
- 支持考勤记录的查看和管理

### 5. API服务功能

**启动API服务**
- 在"系统设置"标签页
- 设置API端口（默认5000）
- 点击"启动API服务"

**API接口说明**

**人脸识别API**
```
POST /api/recognize
Content-Type: application/json

{
    "image": "base64编码的图片数据"
}

返回：
{
    "success": true,
    "name": "张三",
    "confidence": 0.25,
    "age": 28,
    "gender": "male",
    "emotion": "happy",
    "mask": "no"
}
```

**人脸录入API**
```
POST /api/enroll
Content-Type: application/json

{
    "name": "张三",
    "age": 28,
    "gender": "男",
    "department": "技术部",
    "image": "base64编码的图片数据"
}

返回：
{
    "success": true,
    "message": "Face enrolled successfully for 张三"
}
```

**考勤记录API**
```
POST /api/attendance
Content-Type: application/json

{
    "name": "张三",
    "status": "present",
    "location": "办公室"
}

返回：
{
    "success": true,
    "message": "Attendance recorded for 张三"
}
```

**系统状态API**
```
GET /api/status

返回：
{
    "status": "running",
    "face_count": 10,
    "model_status": "完整",
    "api_version": "1.0"
}
```

**模型状态API**
```
GET /api/models/status

返回：
{
    "shape_predictor": true,
    "face_recognizer": true,
    "model_status": "完整",
    "use_local_models_only": true
}
```

## 系统设置

### 识别参数设置
- **识别阈值**：调整人脸识别的灵敏度（0.0-1.0）
- **最大人脸数量**：限制系统支持的最大人脸数量

### 模型设置
- **特征点预测器路径**：手动指定模型文件位置
- **人脸识别模型路径**：手动指定模型文件位置
- **仅使用本地模型**：禁用自动下载功能

### 数据管理设置
- **自动备份**：开启/关闭自动数据备份
- **日志记录**：设置日志记录级别

## 项目结构

```
smart_face_recognition/
├── main.py                 # 主程序文件
├── download_models.py      # 模型自动下载器
├── test_system.py          # 系统功能测试脚本
├── config.json             # 系统配置文件
├── requirements.txt        # 依赖包列表
├── README.md               # 详细使用说明
├── LICENSE                 # 开源许可证
├── start.sh                # Linux/macOS启动脚本
├── start.bat               # Windows启动脚本
├── examples/               # 示例图片目录
│   ├── example1.jpg
│   └── example2.jpg
├── models/                 # 模型文件目录（自动下载）
└── face_database/          # 人脸数据库目录
```

## 常见问题

### Q1: 模型文件下载失败怎么办？
A1: 可以：
- 检查网络连接
- 手动下载模型文件并配置路径
- 使用代理服务器下载

### Q2: 如何确认模型是否正常加载？
A2: 
- 查看系统状态栏的"模型状态"
- 检查系统日志中的模型初始化信息
- 访问API：`GET /api/models/status`

### Q3: 本地模型路径变更后需要重启吗？
A3: 不需要，在系统设置中保存后会立即生效

### Q4: 模型文件损坏如何修复？
A4:
- 删除损坏的模型文件
- 重新运行模型下载器
- 或手动替换为完好的模型文件

### Q5: 支持哪些模型版本？
A5: 系统支持Dlib官方发布的以下版本：
- shape_predictor_68_face_landmarks.dat v1
- dlib_face_recognition_resnet_model_v1.dat v1

## 性能优化建议

### 提高识别速度
- 减少同时识别的人脸数量
- 降低摄像头分辨率
- 调整识别间隔时间

### 提高识别准确率
- 确保良好的光线条件
- 使用正面清晰的人脸照片
- 录入多张不同角度的照片
- 适当调整识别阈值

## 安全考虑

### 数据安全
- 人脸数据采用加密存储
- 定期备份重要数据
- 限制敏感操作的权限

### 隐私保护
- 遵守相关隐私保护法规
- 获得用户明确授权
- 合理使用人脸数据

## 未来发展方向

### 计划添加的功能
- [ ] 多人脸同时识别优化
- [ ] 3D人脸识别支持
- [ ] 移动端应用开发
- [ ] 云端人脸识别服务
- [ ] 更多情感状态识别
- [ ] 活体检测功能

### 技术改进
- [ ] 深度学习模型优化
- [ ] 实时性能提升
- [ ] 多平台适配
- [ ] 离线识别支持

## 贡献指南

欢迎对项目进行改进和扩展！请：
1. Fork 项目
2. 创建功能分支
3. 提交改进
4. 创建Pull Request

## 许可证

本项目采用MIT许可证，详情请参见LICENSE文件。

## 联系信息

如有问题或建议，请联系：
- 项目主页：https://github.com/yourusername/smart-face-recognition
- 邮箱：your.email@example.com

---

*智能人脸识别系统 - 让人脸识别更智能、更便捷！*