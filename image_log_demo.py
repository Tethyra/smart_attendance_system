#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人脸识别日志图片存储示例
演示如何在recognition_logs表中存储和管理识别图片
"""

import os
import cv2
import mysql.connector
from mysql.connector import Error
import datetime
import numpy as np


class FaceRecognitionLogger:
    def __init__(self):
        self.image_dir = "recognition_images"
        self.init_image_directory()

    def init_image_directory(self):
        """初始化图片存储目录"""
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
            print(f"创建图片存储目录: {self.image_dir}")

    def connect_to_database(self):
        """连接到MySQL数据库"""
        try:
            connection = mysql.connector.connect(
                host='localhost',
                database='smart_attendance',
                user='root',
                password='123456'
            )

            if connection.is_connected():
                return connection
        except Error as e:
            print(f"数据库连接错误: {e}")
            return None

    def generate_image_filename(self, user_name=None):
        """生成图片文件名"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if user_name:
            return f"{timestamp}_{user_name}.jpg"
        else:
            return f"{timestamp}_unknown.jpg"

    def save_recognition_image(self, image, user_name=None):
        """保存识别图片"""
        try:
            # 生成文件名
            filename = self.generate_image_filename(user_name)
            filepath = os.path.join(self.image_dir, filename)

            # 保存图片
            if isinstance(image, np.ndarray):
                # 如果是numpy数组（OpenCV格式）
                cv2.imwrite(filepath, image)
            else:
                # 如果是PIL图像
                image.save(filepath)

            print(f"图片保存成功: {filepath}")
            return filepath

        except Exception as e:
            print(f"保存图片失败: {e}")
            return None

    def insert_recognition_log_with_image(self, user_id, confidence, image,
                                          age_prediction=None, gender_prediction=None,
                                          emotion_prediction=None, mask_detection=None):
        """插入带图片的识别日志"""
        # 保存图片
        user_name = self.get_user_name(user_id) if user_id else None
        image_path = self.save_recognition_image(image, user_name)

        if not image_path:
            return False

        # 插入数据库记录
        connection = self.connect_to_database()
        if not connection:
            return False

        try:
            cursor = connection.cursor()

            cursor.execute('''
                INSERT INTO recognition_logs 
                (user_id, confidence, age_prediction, gender_prediction, 
                 emotion_prediction, mask_detection, image_path)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (
                user_id, confidence, age_prediction, gender_prediction,
                emotion_prediction, mask_detection, image_path
            ))

            connection.commit()
            log_id = cursor.lastrowid
            print(f"识别日志插入成功，ID: {log_id}")

            return True

        except Error as e:
            print(f"插入识别日志失败: {e}")
            connection.rollback()
            return False
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

    def get_user_name(self, user_id):
        """根据用户ID获取用户名"""
        connection = self.connect_to_database()
        if not connection:
            return None

        try:
            cursor = connection.cursor()
            cursor.execute("SELECT name FROM users WHERE id = %s", (user_id,))
            result = cursor.fetchone()
            return result[0] if result else None
        except Error as e:
            print(f"获取用户名失败: {e}")
            return None
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

    def get_recognition_logs_with_images(self, limit=10):
        """获取带图片路径的识别日志"""
        connection = self.connect_to_database()
        if not connection:
            return []

        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute('''
                SELECT rl.*, u.name as user_name 
                FROM recognition_logs rl
                LEFT JOIN users u ON rl.user_id = u.id
                ORDER BY rl.recognition_time DESC
                LIMIT %s
            ''', (limit,))

            return cursor.fetchall()

        except Error as e:
            print(f"获取识别日志失败: {e}")
            return []
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

    def display_recognition_log_image(self, log_id):
        """显示指定日志ID的图片"""
        logs = self.get_recognition_logs_with_images()
        for log in logs:
            if log['id'] == log_id:
                image_path = log['image_path']
                if os.path.exists(image_path):
                    image = cv2.imread(image_path)
                    cv2.imshow(f"Recognition Log - ID: {log_id}", image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print(f"图片文件不存在: {image_path}")
                return
        print(f"未找到ID为 {log_id} 的识别日志")


def main():
    """示例演示"""
    logger = FaceRecognitionLogger()

    print("=" * 60)
    print("人脸识别日志图片存储示例")
    print("=" * 60)

    # 示例1: 生成一张模拟的人脸识别图片
    print("\n1. 生成模拟人脸识别图片...")
    # 创建一张黑色背景的图片
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # 在图片上添加文字
    cv2.putText(image, "Face Recognition Test", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(image, "Zhang San", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 示例2: 插入带图片的识别日志
    print("\n2. 插入带图片的识别日志...")
    success = logger.insert_recognition_log_with_image(
        user_id=1,  # 张三的ID
        confidence=0.95,
        image=image,
        age_prediction=28,
        gender_prediction='male',
        emotion_prediction='happy',
        mask_detection='no'
    )

    if success:
        # 示例3: 获取最新的识别日志
        print("\n3. 获取最新的识别日志...")
        logs = logger.get_recognition_logs_with_images(limit=3)

        for i, log in enumerate(logs, 1):
            print(f"\n日志 {i}:")
            print(f"  ID: {log['id']}")
            print(f"  用户: {log['user_name'] or '未知用户'}")
            print(f"  时间: {log['recognition_time']}")
            print(f"  置信度: {log['confidence']:.3f}")
            print(f"  年龄预测: {log['age_prediction']}")
            print(f"  性别预测: {log['gender_prediction']}")
            print(f"  图片路径: {log['image_path']}")

            # 检查图片文件是否存在
            if log['image_path'] and os.path.exists(log['image_path']):
                print(f"  图片状态: 存在")
            else:
                print(f"  图片状态: 不存在")

    print("\n" + "=" * 60)
    print("示例演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()