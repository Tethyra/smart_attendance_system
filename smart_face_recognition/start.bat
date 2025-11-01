@echo off
echo ======================================
echo   智能人脸识别系统启动脚本
echo ======================================
echo.

:: 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: Python未安装
    echo 请先安装Python 3.7或更高版本
    pause
    exit /b 1
)

:: 检查Python版本
for /f "tokens=2 delims= " %%a in ('python --version') do set PYTHON_VERSION=%%a
for /f "tokens=1 delims=." %%a in ("%PYTHON_VERSION%") do set PYTHON_MAJOR=%%a
for /f "tokens=2 delims=." %%a in ("%PYTHON_VERSION%") do set PYTHON_MINOR=%%a

if %PYTHON_MAJOR% lss 3 (
    echo 错误: Python版本过低
    echo 需要Python 3.7或更高版本，当前版本: %PYTHON_VERSION%
    pause
    exit /b 1
)

if %PYTHON_MINOR% lss 7 (
    echo 错误: Python版本过低
    echo 需要Python 3.7或更高版本，当前版本: %PYTHON_VERSION%
    pause
    exit /b 1
)

echo Python版本检查通过: %PYTHON_VERSION%
echo.

:: 检查虚拟环境
if not exist "venv" (
    echo 未找到虚拟环境，正在创建...
    python -m venv venv
    
    :: 激活虚拟环境
    call venv\Scripts\activate.bat
    
    :: 安装依赖
    echo 正在安装依赖包...
    pip install -r requirements.txt
    
    :: 下载模型
    echo 正在下载模型文件...
    python download_models.py
) else (
    :: 激活虚拟环境
    echo 激活虚拟环境...
    call venv\Scripts\activate.bat
)

:: 检查模型文件
set MODEL_MISSING=0
if not exist "models\shape_predictor_68_face_landmarks.dat" (
    echo 警告: 模型文件缺失: shape_predictor_68_face_landmarks.dat
    set MODEL_MISSING=1
)
if not exist "models\dlib_face_recognition_resnet_model_v1.dat" (
    echo 警告: 模型文件缺失: dlib_face_recognition_resnet_model_v1.dat
    set MODEL_MISSING=1
)

if %MODEL_MISSING% equ 1 (
    echo 正在重新下载模型文件...
    python download_models.py
)

:: 启动系统
echo 正在启动智能人脸识别系统...
python main.py

echo 系统已退出
pause