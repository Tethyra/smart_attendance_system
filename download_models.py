#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型下载器
自动下载人脸识别所需的Dlib模型文件
"""

import os
import sys
import requests
import json
import progressbar

def load_config():
    """加载配置文件"""
    config_path = "config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "model_path": "models",
        "shape_predictor_path": "models/shape_predictor_68_face_landmarks.dat",
        "face_recognition_model_path": "models/dlib_face_recognition_resnet_model_v1.dat"
    }

def download_file(url, save_path):
    """下载文件并显示进度"""
    try:
        # 创建目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print(f"正在下载: {url}")
        print(f"保存路径: {save_path}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1KB
        
        progress_bar = progressbar.ProgressBar(maxval=total_size, 
                                              widgets=[progressbar.Bar('=', '[', ']'), 
                                                       ' ', progressbar.Percentage()])
        progress_bar.start()
        
        with open(save_path, 'wb') as f:
            for data in response.iter_content(block_size):
                if data:
                    f.write(data)
                    progress_bar.update(f.tell())
        
        progress_bar.finish()
        print(f"下载完成: {save_path}")
        return True
        
    except Exception as e:
        print(f"下载失败: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("智能人脸识别系统 - 模型下载器")
    print("=" * 50)
    
    config = load_config()
    
    # 模型文件URL
    models = [
        {
            "name": "shape_predictor_68_face_landmarks.dat",
            "url": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
            "save_path": config["shape_predictor_path"]
        },
        {
            "name": "dlib_face_recognition_resnet_model_v1.dat",
            "url": "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2",
            "save_path": config["face_recognition_model_path"]
        }
    ]
    
    # 检查是否已存在模型文件
    existing_models = []
    missing_models = []
    
    for model in models:
        if os.path.exists(model["save_path"]):
            existing_models.append(model["name"])
        else:
            missing_models.append(model)
    
    if existing_models:
        print(f"\n已存在的模型: {', '.join(existing_models)}")
    
    if not missing_models:
        print("\n所有模型文件都已存在，无需下载")
        return
    
    print(f"\n需要下载的模型: {', '.join([m['name'] for m in missing_models])}")
    print("\n注意: 模型文件较大，下载时间可能较长，请耐心等待...")
    
    # 开始下载
    for model in missing_models:
        print(f"\n下载模型: {model['name']}")
        success = download_file(model["url"], model["save_path"])
        if not success:
            print(f"模型 {model['name']} 下载失败，请检查网络连接后重试")
            sys.exit(1)
    
    print("\n" + "=" * 50)
    print("所有模型下载完成！")
    print("您现在可以启动人脸识别系统了。")
    print("=" * 50)

if __name__ == "__main__":
    main()