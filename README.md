# 智能人脸识别系统

## 项目简介

智能人脸识别系统是一个基于 Python 开发的完整解决方案，集成了人脸检测、识别、录入管理和考勤功能。系统采用现代化的图形界面，支持实时摄像头识别、图片 / 视频文件识别，并提供完整的 API 服务接口。

## 功能特点

### 🎯 核心功能



* **实时人脸识别**：支持摄像头实时人脸检测和识别

* **人脸录入管理**：便捷的人脸信息录入和用户管理

* **智能考勤系统**：完整的签到、签退和考勤记录管理

* **API 服务接口**：提供 RESTful API 支持第三方集成

### 🚀 技术优势



* **识别稳定性优化**：智能算法减少误判，提高识别准确率

* **识别结果固定**：支持识别结果固定，避免频繁变化

* **多平台兼容**：支持 Windows、Linux、macOS 等操作系统

* **模块化设计**：清晰的代码结构，易于维护和扩展

### 🔧 修复工具



* 人脸丢失数据修复

* 信息数据更新和修复

* API 服务故障修复

* 识别结果数据修复

## 技术架构



```
┌─────────────────────────────────────────────────────────┐

│                  智能人脸识别系统                         │

├─────────────┬─────────────┬─────────────┬───────────────┤

│   界面层    │   业务层    │   算法层    │    数据层     │

│  (PyQt5)    │ (Python)    │  (dlib)     │  (SQLite)     │

├─────────────┼─────────────┼─────────────┼───────────────┤

│ 主界面      │ 人脸识别    │ 人脸检测    │ 用户信息      │

│ 录入界面    │ 考勤管理    │ 特征提取    │ 人脸特征      │

│ 考勤界面    │ API服务    │ 特征匹配    │ 考勤记录      │

│ 设置界面    │ 系统管理    │ 稳定性优化  │ 系统配置      │

└─────────────┴─────────────┴─────────────┴───────────────┘
```

## 快速开始

### 环境要求



* Python 3.7+

* 至少 4GB 内存

* 支持摄像头的设备（可选）

### 安装依赖



```
\# 安装基础依赖

pip install -r requirements.txt

\# 如果需要API服务支持

pip install flask flask-cors waitress
```

### 模型文件准备

系统需要以下模型文件（放置在 models 目录下）：



* `shape_predictor_68_face_landmarks.dat`：人脸特征点预测器

* `dlib_face_recognition_resnet_model_v1.dat`：人脸识别模型

**自动下载脚本**：



```
python download\_models.py
```

### 启动系统



```
\# 启动主程序

python main.py

\# 启动API服务（可选）

python api\_service.py
```

## 使用指南

### 1. 人脸识别功能

#### 实时摄像头识别



1. 点击 "启动摄像头" 按钮

2. 点击 "开始识别" 按钮

3. 系统将实时显示识别结果

4. 可启用 "识别稳定化" 和 "结果固定" 功能

#### 图片 / 视频识别



1. 点击 "选择图片" 或 "选择视频" 按钮

2. 选择要识别的文件

3. 系统将自动进行识别并显示结果

### 2. 人脸录入管理



1. 切换到 "人脸录入" 标签页

2. 填写用户基本信息（姓名、年龄、性别、部门）

3. 点击 "捕获人脸" 按钮拍摄照片

4. 可拍摄多张照片提高识别准确率

5. 点击 "保存信息" 完成录入

### 3. 考勤系统



1. 切换到 "考勤系统" 标签页

2. 启动考勤摄像头

3. 点击 "开始考勤" 按钮

4. 员工可进行签到和签退操作

5. 系统自动记录考勤信息

### 4. API 服务使用

API 服务启动后，可通过以下地址访问：



* 基础 URL：`http://localhost:5000`

* 测试页面：`http://localhost:5000/api_test`

主要 API 接口：



* `GET /api/status` - 获取系统状态

* `POST /api/face_recognition` - 人脸识别

* `POST /api/attendance/check_in` - 签到

* `POST /api/attendance/check_out` - 签退

* `GET /api/attendance/records` - 获取考勤记录

## 配置说明

系统配置文件：`config.py`

主要配置项：



* `api_port`：API 服务端口（默认 5000）

* `recognition_threshold`：识别阈值

* `stability_window`：稳定性窗口大小

* `camera_index`：摄像头索引

* `model_paths`：模型文件路径

## 修复工具使用

### 人脸丢失数据修复



```
python modify\_face\_lost.py

python modify\_face\_lost\_v2.py

python update\_face\_lost.py
```

### 信息数据修复



```
python fix\_missing\_info.py

python update\_user\_info.py

python modify\_age\_gender.py
```

### API 服务修复



```
python fix\_api\_service.py
```

### 识别结果修复



```
python fix\_fixed\_recognition\_save.py

python modify\_fixed\_recognition.py
```

### 综合修复



```
python face\_attendance\_fixer.py

python diagnose\_models.py

python apply\_modifications.py
```

## 项目结构



```
├── main.py                 # 主程序入口

├── models.py               # 人脸识别模型

├── camera.py               # 摄像头管理

├── database.py             # 数据库操作

├── utils.py                # 工具函数

├── config.py               # 系统配置

├── api\_service.py          # API服务

├── requirements.txt        # 依赖列表

├── README.md               # 项目说明

├── models/                 # 模型文件目录

├── data/                   # 数据文件目录

├── logs/                   # 日志文件目录

├── fix\_\*.py                # 修复工具

├── modify\_\*.py             # 修改工具

├── update\_\*.py             # 更新工具

└── test\_api.py             # API测试
```

## 常见问题

### Q1: 摄像头无法启动

**解决方案**：



* 检查摄像头是否被其他程序占用

* 修改 config.py 中的 camera\_index 配置

* 确保摄像头驱动正常安装

### Q2: 人脸识别准确率低

**解决方案**：



* 确保光线充足

* 录入时拍摄多角度照片

* 调整 recognition\_threshold 参数

* 检查模型文件是否完整

### Q3: API 服务无法启动

**解决方案**：



* 检查 Flask 是否安装

* 检查端口是否被占用

* 查看日志文件获取详细错误信息

### Q4: 系统运行缓慢

**解决方案**：



* 关闭不必要的程序释放内存

* 降低摄像头分辨率

* 禁用识别稳定化功能

## 性能优化

### 识别速度优化



* 使用较小的摄像头分辨率

* 减少识别频率

* 关闭不必要的功能

### 识别准确率优化



* 录入多张不同角度的照片

* 调整识别阈值

* 确保良好的光线条件

## 安全考虑



1. **数据安全**：所有敏感数据加密存储

2. **访问控制**：建议在生产环境中添加用户认证

3. **网络安全**：API 服务建议使用 HTTPS

4. **隐私保护**：遵循相关隐私保护法规

## 扩展建议



1. **多摄像头支持**：增加多摄像头同时监控

2. **云端集成**：支持云端人脸识别服务

3. **移动端支持**：开发移动端应用

4. **AI 功能增强**：添加情绪识别、年龄预测等功能

5. **报表系统**：完善的考勤统计和分析报表

## 许可证

本项目采用 MIT 许可证，详情请参考 LICENSE 文件。

## 联系方式

如有问题或建议，请通过以下方式联系：



* 项目地址：[GitHub Repository](https://github.com/yourusername/face-recognition-system)

* 邮箱：your.email@example.com



***

**注意**：本系统仅供学习和研究使用，请在合法合规的前提下使用。

> （注：文档部分内容可能由 AI 生成）