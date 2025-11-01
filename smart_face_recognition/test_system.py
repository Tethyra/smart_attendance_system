#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
系统功能测试脚本
验证智能人脸识别系统的核心功能
"""

import os
import sys
import json
import requests
import base64
from PIL import Image
from io import BytesIO

class FaceRecognitionTester:
    def __init__(self):
        self.base_url = "http://localhost:5000/api"
        self.test_image_path = "test_image.jpg"
        
    def download_test_image(self):
        """下载测试图片"""
        try:
            print("正在下载测试图片...")
            url = "https://randomuser.me/api/portraits/women/44.jpg"
            response = requests.get(url)
            
            with open(self.test_image_path, 'wb') as f:
                f.write(response.content)
            
            print(f"测试图片已保存: {self.test_image_path}")
            return True
            
        except Exception as e:
            print(f"下载测试图片失败: {str(e)}")
            return False
    
    def image_to_base64(self, image_path):
        """将图片转换为base64编码"""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            print(f"图片转换失败: {str(e)}")
            return None
    
    def test_api_status(self):
        """测试API服务状态"""
        try:
            print("\n=== 测试API服务状态 ===")
            response = requests.get(f"{self.base_url}/status")
            response.raise_for_status()
            
            data = response.json()
            print(f"API状态: {data['status']}")
            print(f"人脸数量: {data['face_count']}")
            print(f"API版本: {data['api_version']}")
            print("API服务正常运行")
            return True
            
        except Exception as e:
            print(f"API服务测试失败: {str(e)}")
            return False
    
    def test_face_recognition(self):
        """测试人脸识别功能"""
        try:
            print("\n=== 测试人脸识别功能 ===")
            
            # 确保测试图片存在
            if not os.path.exists(self.test_image_path):
                if not self.download_test_image():
                    return False
            
            # 转换图片为base64
            base64_image = self.image_to_base64(self.test_image_path)
            if not base64_image:
                return False
            
            # 调用API
            payload = {
                "image": base64_image
            }
            
            response = requests.post(
                f"{self.base_url}/recognize",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            response.raise_for_status()
            result = response.json()
            
            print(f"识别结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            if result.get('success'):
                print("人脸识别功能测试成功")
                return True
            else:
                print(f"人脸识别功能测试失败: {result.get('error')}")
                return False
                
        except Exception as e:
            print(f"人脸识别测试失败: {str(e)}")
            return False
    
    def test_face_enrollment(self):
        """测试人脸录入功能"""
        try:
            print("\n=== 测试人脸录入功能 ===")
            
            # 确保测试图片存在
            if not os.path.exists(self.test_image_path):
                if not self.download_test_image():
                    return False
            
            # 转换图片为base64
            base64_image = self.image_to_base64(self.test_image_path)
            if not base64_image:
                return False
            
            # 调用API
            payload = {
                "name": "测试用户",
                "age": 25,
                "gender": "女",
                "department": "测试部",
                "image": base64_image
            }
            
            response = requests.post(
                f"{self.base_url}/enroll",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            response.raise_for_status()
            result = response.json()
            
            print(f"录入结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            if result.get('success'):
                print("人脸录入功能测试成功")
                return True
            else:
                print(f"人脸录入功能测试失败: {result.get('error')}")
                return False
                
        except Exception as e:
            print(f"人脸录入测试失败: {str(e)}")
            return False
    
    def test_attendance(self):
        """测试考勤功能"""
        try:
            print("\n=== 测试考勤功能 ===")
            
            # 调用API
            payload = {
                "name": "测试用户",
                "status": "present",
                "location": "测试办公室"
            }
            
            response = requests.post(
                f"{self.base_url}/attendance",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            response.raise_for_status()
            result = response.json()
            
            print(f"考勤结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            if result.get('success'):
                print("考勤功能测试成功")
                return True
            else:
                print(f"考勤功能测试失败: {result.get('error')}")
                return False
                
        except Exception as e:
            print(f"考勤测试失败: {str(e)}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("智能人脸识别系统功能测试")
        print("=" * 60)
        
        results = []
        
        # 测试API状态
        results.append(("API服务状态", self.test_api_status()))
        
        # 测试人脸识别
        results.append(("人脸识别功能", self.test_face_recognition()))
        
        # 测试人脸录入
        results.append(("人脸录入功能", self.test_face_enrollment()))
        
        # 测试考勤
        results.append(("考勤功能", self.test_attendance()))
        
        # 清理测试文件
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
        
        # 显示测试总结
        print("\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)
        
        passed = 0
        failed = 0
        
        for test_name, result in results:
            status = "✓ 成功" if result else "✗ 失败"
            print(f"{test_name}: {status}")
            if result:
                passed += 1
            else:
                failed += 1
        
        print(f"\n总计: {passed} 项成功, {failed} 项失败")
        
        if failed == 0:
            print("所有测试通过！系统功能正常。")
            return True
        else:
            print("部分测试失败，请检查系统配置。")
            return False

if __name__ == "__main__":
    tester = FaceRecognitionTester()
    
    # 检查API服务是否启动
    try:
        response = requests.get(f"{tester.base_url}/status", timeout=5)
    except requests.exceptions.ConnectionError:
        print("错误: API服务未启动或无法连接")
        print("请先启动主程序并确保API服务已开启")
        sys.exit(1)
    
    tester.run_all_tests()