#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型下载器
自动下载人脸识别所需的Dlib模型文件
"""

import os
import sys
import requests
import zipfile
import shutil

def download_file(url, filename):
    """下载文件"""
    try:
        print(f"正在下载: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"下载完成: {filename}")
        return True
        
    except Exception as e:
        print(f"下载失败: {str(e)}")
        return False

def download_models(model_dir='models'):
    """下载所有需要的模型"""
    
    # 创建模型目录
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 模型下载地址
    models = [
        {
            'name': 'shape_predictor_68_face_landmarks.dat',
            'url': 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2',
            'compressed': True
        },
        {
            'name': 'dlib_face_recognition_resnet_model_v1.dat',
            'url': 'http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2',
            'compressed': True
        }
    ]
    
    # 下载并解压模型
    for model in models:
        model_path = os.path.join(model_dir, model['name'])
        
        if os.path.exists(model_path):
            print(f"模型已存在: {model['name']}")
            continue
        
        # 下载压缩文件
        compressed_path = model_path + '.bz2'
        if download_file(model['url'], compressed_path):
            # 解压文件
            try:
                import bz2
                print(f"正在解压: {compressed_path}")
                
                with bz2.BZ2File(compressed_path, 'rb') as f_in:
                    with open(model_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # 删除压缩文件
                os.remove(compressed_path)
                print(f"解压完成: {model['name']}")
                
            except Exception as e:
                print(f"解压失败: {str(e)}")
                if os.path.exists(compressed_path):
                    os.remove(compressed_path)

def download_sample_images(data_dir='database_path'):
    """下载示例图片"""
    
    # 创建数据目录
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # 示例图片下载地址（这里使用一些公开的示例图片）
    sample_images = [
        {
            'name': 'sample1.jpg',
            'url': 'https://randomuser.me/api/portraits/women/44.jpg'
        },
        {
            'name': 'sample2.jpg',
            'url': 'https://randomuser.me/api/portraits/men/32.jpg'
        },
        {
            'name': 'sample3.jpg',
            'url': 'https://randomuser.me/api/portraits/women/68.jpg'
        }
    ]
    
    for img in sample_images:
        img_path = os.path.join(data_dir, img['name'])
        
        if os.path.exists(img_path):
            print(f"示例图片已存在: {img['name']}")
            continue
        
        download_file(img['url'], img_path)

def main():
    """主函数"""
    print("=" * 50)
    print("智能人脸识别系统 - 模型下载器")
    print("=" * 50)
    
    print("\n1. 正在下载Dlib模型文件...")
    download_models()
    
    print("\n2. 正在下载示例图片...")
    download_sample_images()
    
    print("\n" + "=" * 50)
    print("下载完成！")
    print("=" * 50)

if __name__ == "__main__":
    main()