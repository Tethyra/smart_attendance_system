#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型诊断工具
用于检查人脸识别系统的模型文件状态
"""

import os
import json
import sys

def check_model_file(path, min_size_mb=100):
    """检查模型文件"""
    result = {
        'exists': False,
        'size_ok': False,
        'size_mb': 0,
        'message': ''
    }
    
    if not os.path.exists(path):
        result['message'] = f"文件不存在: {path}"
        return result
    
    result['exists'] = True
    
    # 检查文件大小
    file_size = os.path.getsize(path)
    result['size_mb'] = file_size / (1024 * 1024)
    
    if result['size_mb'] >= min_size_mb:
        result['size_ok'] = True
        result['message'] = f"文件正常 (大小: {result['size_mb']:.1f}MB)"
    else:
        result['message'] = f"文件过小 (大小: {result['size_mb']:.1f}MB，建议至少{min_size_mb}MB)"
    
    return result

def check_config():
    """检查配置文件"""
    config_path = 'config.json'
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ 配置文件加载成功")
        return config
    except Exception as e:
        print(f"❌ 配置文件解析失败: {str(e)}")
        return None

def main():
    """主函数"""
    print("=" * 60)
    print("智能人脸识别系统 - 模型诊断工具")
    print("=" * 60)
    
    # 检查配置文件
    config = check_config()
    
    if not config:
        print("\n⚠️ 使用默认配置进行检查...")
        config = {
            'shape_predictor_path': 'models/shape_predictor_68_face_landmarks.dat',
            'face_recognition_model_path': 'models/dlib_face_recognition_resnet_model_v1.dat',
            'use_local_models_only': True
        }
    
    print(f"\n📁 模型文件检查:")
    print("-" * 60)
    
    # 检查特征点预测器
    print(f"\n1. 特征点预测器: {config['shape_predictor_path']}")
    predictor_result = check_model_file(config['shape_predictor_path'])
    status = "✅" if predictor_result['exists'] and predictor_result['size_ok'] else "❌"
    print(f"   {status} {predictor_result['message']}")
    
    # 检查人脸识别模型
    print(f"\n2. 人脸识别模型: {config['face_recognition_model_path']}")
    recognition_result = check_model_file(config['face_recognition_model_path'])
    status = "✅" if recognition_result['exists'] and recognition_result['size_ok'] else "❌"
    print(f"   {status} {recognition_result['message']}")
    
    print("\n" + "-" * 60)
    
    # 总结
    all_ok = (predictor_result['exists'] and predictor_result['size_ok'] and
              recognition_result['exists'] and recognition_result['size_ok'])
    
    if all_ok:
        print("✅ 所有模型文件检查通过！")
        print("🎯 系统可以正常运行")
    else:
        print("❌ 模型文件检查失败")
        print("\n💡 建议解决方案:")
        
        if not predictor_result['exists'] or not recognition_result['exists']:
            print("   1. 运行模型下载器: python download_models.py")
            print("   2. 或手动下载模型文件并配置路径")
        
        if not predictor_result['size_ok'] or not recognition_result['size_ok']:
            print("   3. 检查模型文件是否完整，可能需要重新下载")
        
        print("\n📋 模型下载地址:")
        print("   - shape_predictor_68_face_landmarks.dat: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("   - dlib_face_recognition_resnet_model_v1.dat: http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
    
    print(f"\n⚙️ 配置信息:")
    print(f"   仅使用本地模型: {'是' if config.get('use_local_models_only', True) else '否'}")
    print(f"   识别阈值: {config.get('threshold', 0.4)}")
    print(f"   API端口: {config.get('api_port', 5000)}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()