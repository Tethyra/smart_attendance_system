#!/bin/bash
# 智能人脸识别系统启动脚本

echo "======================================"
echo "  智能人脸识别系统启动脚本"
echo "======================================"

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: Python3 未安装"
    echo "请先安装Python3 (3.7或更高版本)"
    exit 1
fi

# 检查Python版本
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ $PYTHON_MAJOR -lt 3 ] || [ $PYTHON_MINOR -lt 7 ]; then
    echo "错误: Python版本过低"
    echo "需要Python 3.7或更高版本，当前版本: $PYTHON_VERSION"
    exit 1
fi

echo "Python版本检查通过: $PYTHON_VERSION"

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "未找到虚拟环境，正在创建..."
    python3 -m venv venv
    
    # 激活虚拟环境
    if [ "$(uname)" == "Darwin" ] || [ "$(uname)" == "Linux" ]; then
        source venv/bin/activate
    else
        venv\Scripts\activate
    fi
    
    # 安装依赖
    echo "正在安装依赖包..."
    pip install -r requirements.txt
    
    # 下载模型
    echo "正在下载模型文件..."
    python download_models.py
else
    # 激活虚拟环境
    echo "激活虚拟环境..."
    if [ "$(uname)" == "Darwin" ] || [ "$(uname)" == "Linux" ]; then
        source venv/bin/activate
    else
        venv\Scripts\activate
    fi
fi

# 检查模型文件
MODEL_FILES=("models/shape_predictor_68_face_landmarks.dat" "models/dlib_face_recognition_resnet_model_v1.dat")
MODEL_MISSING=0

for model in "${MODEL_FILES[@]}"; do
    if [ ! -f "$model" ]; then
        echo "警告: 模型文件缺失: $model"
        MODEL_MISSING=1
    fi
done

if [ $MODEL_MISSING -eq 1 ]; then
    echo "正在重新下载模型文件..."
    python download_models.py
fi

# 启动系统
echo "正在启动智能人脸识别系统..."
python main.py

echo "系统已退出"