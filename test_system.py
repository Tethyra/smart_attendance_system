#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能人脸识别系统 - 功能测试脚本
用于验证系统的基本功能是否正常工作
"""

import os
import sys
import subprocess
import time

def test_system():
    """测试系统基本功能"""
    print("智能人脸识别系统功能测试")
    print("========================")
    
    # 检查Python环境
    print("\n1. 检查Python环境...")
    try:
        python_version = sys.version.split()[0]
        print(f"   Python版本: {python_version}")
        if float(python_version[:3]) < 3.7:
            print("   警告: Python版本建议3.7或以上")
    except Exception as e:
        print(f"   错误: {e}")
    
    # 检查依赖包
    print("\n2. 检查依赖包...")
    required_packages = [
        "PyQt5", "dlib", "Pillow", "numpy", 
        "PyMySQL", "Flask", "opencv-python"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ✗ {package}")
    
    if missing_packages:
        print(f"\n   缺少依赖包: {', '.join(missing_packages)}")
        print("   请运行: pip install -r requirements.txt")
    
    # 检查模型文件目录
    print("\n3. 检查模型文件...")
    model_dir = "models"
    if os.path.exists(model_dir):
        print(f"   ✓ 模型目录存在: {model_dir}")
        
        model_files = [
            "shape_predictor_68_face_landmarks.dat",
            "dlib_face_recognition_resnet_model_v1.dat"
        ]
        
        for model_file in model_files:
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path) / (1024 * 1024)
                print(f"   ✓ {model_file} ({file_size:.1f}MB)")
            else:
                print(f"   ✗ {model_file} (缺失)")
    else:
        print(f"   ✗ 模型目录不存在: {model_dir}")
        print("   请创建models目录并放入模型文件")
    
    # 检查数据库配置
    print("\n4. 检查数据库配置...")
    config_file = "config.json.example"
    if os.path.exists(config_file):
        print(f"   ✓ 配置文件存在: {config_file}")
        # 这里可以添加更多数据库连接测试
    else:
        print(f"   ✗ 配置文件不存在: {config_file}")
    
    # 检查启动文件
    print("\n5. 检查启动文件...")
    start_files = ["main_fixed.py"]
    for start_file in start_files:
        if os.path.exists(start_file):
            print(f"   ✓ {start_file}")
        else:
            print(f"   ✗ {start_file}")
    
    # 检查目录结构
    print("\n6. 检查目录结构...")
    required_dirs = ["face_database", "logs"]
    for req_dir in required_dirs:
        if not os.path.exists(req_dir):
            print(f"   创建目录: {req_dir}")
            os.makedirs(req_dir, exist_ok=True)
        else:
            print(f"   ✓ {req_dir}")
    
    print("\n测试完成！")
    print("\n启动系统方法:")
    print("   Windows: 双击 start.bat")
    print("   Linux/macOS: 运行 ./start.sh 或 python main_fixed.py")
    print("\n注意事项:")
    print("   1. 确保MySQL服务已启动")
    print("   2. 确保摄像头可以正常使用")
    print("   3. 确保模型文件完整")

if __name__ == "__main__":
    test_system()