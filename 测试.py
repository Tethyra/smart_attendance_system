#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
插入示例用户数据到智能人脸识别考勤系统
"""

import mysql.connector
from mysql.connector import Error
import json
import numpy as np


def connect_to_database():
    """连接到MySQL数据库"""
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='smart_attendance',
            user='root',
            password='123456'
        )

        if connection.is_connected():
            db_Info = connection.get_server_info()
            print(f"成功连接到MySQL数据库，版本: {db_Info}")
            return connection

    except Error as e:
        print(f"数据库连接错误: {e}")
        return None


def generate_sample_face_encoding():
    """生成示例人脸特征编码"""
    # 生成一个128维的随机数组作为示例特征编码
    # 实际应用中这应该是通过Dlib提取的真实人脸特征
    face_encoding = np.random.rand(128).tolist()
    return json.dumps(face_encoding)


def insert_sample_user(connection):
    """插入示例用户数据"""
    if not connection or not connection.is_connected():
        print("数据库连接失败")
        return False

    try:
        cursor = connection.cursor()

        # 准备用户数据
        user_data = {
            'name': '张三',
            'age': 28,
            'gender': '男',
            'department': '技术部',
            'face_encoding': generate_sample_face_encoding()
        }

        # 检查用户是否已存在
        cursor.execute("SELECT id FROM users WHERE name = %s", (user_data['name'],))
        existing_user = cursor.fetchone()

        if existing_user:
            print(f"用户 {user_data['name']} 已存在，ID: {existing_user[0]}")
            return False

        # 插入新用户
        cursor.execute('''
            INSERT INTO users (name, age, gender, department, face_encoding)
            VALUES (%s, %s, %s, %s, %s)
        ''', (
            user_data['name'],
            user_data['age'],
            user_data['gender'],
            user_data['department'],
            user_data['face_encoding']
        ))

        connection.commit()

        # 获取插入的用户ID
        user_id = cursor.lastrowid
        print(f"成功插入用户: {user_data['name']}")
        print(f"用户ID: {user_id}")
        print(f"年龄: {user_data['age']}")
        print(f"性别: {user_data['gender']}")
        print(f"部门: {user_data['department']}")

        # 同时插入一条示例考勤记录
        insert_sample_attendance(connection, user_id, user_data['name'])

        return True

    except Error as e:
        print(f"插入数据失败: {e}")
        connection.rollback()
        return False
    finally:
        if cursor:
            cursor.close()


def insert_sample_attendance(connection, user_id, user_name):
    """插入示例考勤记录"""
    try:
        cursor = connection.cursor()

        # 插入今日签到记录
        cursor.execute('''
            INSERT INTO attendance (user_id, check_in_time, status, location)
            VALUES (%s, CURRENT_TIMESTAMP, %s, %s)
        ''', (user_id, 'present', '办公室'))

        connection.commit()
        print(f"成功插入 {user_name} 的考勤记录")

    except Error as e:
        print(f"插入考勤记录失败: {e}")
        connection.rollback()


def insert_sample_recognition_log(connection, user_id, user_name):
    """插入示例识别日志"""
    try:
        cursor = connection.cursor()

        # 插入识别日志
        cursor.execute('''
            INSERT INTO recognition_logs (user_id, confidence, age_prediction, gender_prediction, emotion_prediction, mask_detection)
            VALUES (%s, %s, %s, %s, %s, %s)
        ''', (user_id, 0.95, 28, 'male', 'happy', 'no'))

        connection.commit()
        print(f"成功插入 {user_name} 的识别日志")

    except Error as e:
        print(f"插入识别日志失败: {e}")
        connection.rollback()


def main():
    """主函数"""
    print("=" * 60)
    print("智能人脸识别考勤系统 - 插入示例数据")
    print("=" * 60)

    # 连接数据库
    connection = connect_to_database()

    if connection:
        try:
            # 插入示例用户
            success = insert_sample_user(connection)

            if success:
                print("\n" + "=" * 60)
                print("数据插入完成！")
                print("您现在可以在系统中看到张三的信息了。")
                print("=" * 60)
            else:
                print("\n数据插入失败或用户已存在")

        finally:
            # 关闭数据库连接
            if connection.is_connected():
                connection.close()
                print("数据库连接已关闭")


if __name__ == "__main__":
    main()