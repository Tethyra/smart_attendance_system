#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API服务测试脚本
"""

import requests
import json
import sys

def test_api_health():
    """测试API健康状态"""
    try:
        response = requests.get('http://localhost:5000/api/health', timeout=5)
        if response.status_code == 200:
            print("✓ API健康检查: 正常")
            return True
        else:
            print(f"✗ API健康检查: 失败 (状态码: {response.status_code})")
            return False
    except Exception as e:
        print(f"✗ API健康检查: 无法连接 ({e})")
        return False

def test_api_status():
    """测试API状态"""
    try:
        response = requests.get('http://localhost:5000/api/status', timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                print("✓ API状态检查: 正常")
                print(f"  系统状态: {data['data']}")
                return True
            else:
                print(f"✗ API状态检查: 失败 ({data.get('message', '未知错误')})")
                return False
        else:
            print(f"✗ API状态检查: 失败 (状态码: {response.status_code})")
            return False
    except Exception as e:
        print(f"✗ API状态检查: 无法连接 ({e})")
        return False

def test_api_users():
    """测试用户列表API"""
    try:
        response = requests.get('http://localhost:5000/api/users', timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                print(f"✓ 用户列表API: 正常 (共 {data['data']['total']} 个用户)")
                return True
            else:
                print(f"✗ 用户列表API: 失败 ({data.get('message', '未知错误')})")
                return False
        else:
            print(f"✗ 用户列表API: 失败 (状态码: {response.status_code})")
            return False
    except Exception as e:
        print(f"✗ 用户列表API: 无法连接 ({e})")
        return False

def main():
    """主测试函数"""
    print("开始测试API服务...")
    print("-" * 50)

    # 测试健康检查
    health_ok = test_api_health()

    # 测试状态检查
    status_ok = test_api_status()

    # 测试用户列表
    users_ok = test_api_users()

    print("-" * 50)

    if health_ok and status_ok and users_ok:
        print("🎉 所有API测试通过！")
        print("\nAPI服务运行正常，可以访问以下地址：")
        print("http://localhost:5000")
        print("http://localhost:5000/api/status")
        print("http://localhost:5000/api/users")
    else:
        print("❌ 部分API测试失败")
        print("请检查：")
        print("1. API服务是否已启动")
        print("2. 端口5000是否被占用")
        print("3. 防火墙设置")

    print("\n提示：您可以使用 api_test.html 文件进行更详细的测试")

if __name__ == "__main__":
    main()
