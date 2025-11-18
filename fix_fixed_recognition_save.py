#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复固定识别结果的保存部分，确保包含年龄、性别和部门信息
"""

import fileinput
import sys

def fix_high_confidence_save():
    """修复高置信度固定结果的保存"""
    print("正在修复高置信度固定结果的保存...")
    
    # 使用fileinput直接修改文件
    for line in fileinput.input('models.py', inplace=True):
        # 找到高置信度固定的位置
        if 'if display_confidence > 0.85 and display_name != "无该人像":' in line:
            print(line, end='')
            # 读取下几行直到找到固定代码
            while True:
                next_line = fileinput.readline()
                if 'self.parent.fixed_recognition[face_id] = {' in next_line:
                    # 跳过原来的固定代码
                    while '}' not in next_line:
                        next_line = fileinput.readline()
                    # 添加新的包含完整信息的固定代码
                    print("                # 保存完整的用户信息到固定结果中")
                    print("                self.parent.fixed_recognition[face_id] = {")
                    print("                    'name': display_name,")
                    print("                    'confidence': display_confidence,")
                    print("                    'age': age,")
                    print("                    'gender': gender,")
                    print("                    'department': department,")
                    print("                    'fixed_at': datetime.now()")
                    print("                }")
                    break
                else:
                    print(next_line, end='')
        else:
            print(line, end='')

def fix_stability_save():
    """修复稳定性固定结果的保存"""
    print("正在修复稳定性固定结果的保存...")
    
    # 使用fileinput直接修改文件
    for line in fileinput.input('models.py', inplace=True):
        # 找到稳定性固定的位置
        if 'if stability > self.config.get(\'recognition_fix_threshold\', 0.6):' in line:
            print(line, end='')
            # 读取下几行直到找到固定代码
            while True:
                next_line = fileinput.readline()
                if 'self.parent.fixed_recognition[face_id] = {' in next_line:
                    # 跳过原来的固定代码
                    while '}' not in next_line:
                        next_line = fileinput.readline()
                    # 添加新的包含完整信息的固定代码
                    print("                        # 保存完整的用户信息到固定结果中")
                    print("                        self.parent.fixed_recognition[face_id] = {")
                    print("                            'name': most_common_name,")
                    print("                            'confidence': total_confidence / len(history),")
                    print("                            'age': age,")
                    print("                            'gender': gender,")
                    print("                            'department': department,")
                    print("                            'fixed_at': datetime.now()")
                    print("                        }")
                    break
                else:
                    print(next_line, end='')
        else:
            print(line, end='')

def main():
    """主函数"""
    print("开始修复固定识别结果的保存部分...")
    
    try:
        # 修复高置信度固定结果的保存
        fix_high_confidence_save()
        
        # 修复稳定性固定结果的保存
        fix_stability_save()
        
        print("固定识别结果保存部分修复完成！")
        
    except Exception as e:
        print(f"修复过程中发生错误：{str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()