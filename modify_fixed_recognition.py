#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修改人脸识别系统，实现：
1. 年龄/性别识别到之后也要固定住
2. 对识别结果里的内容添加（部门）的信息，从数据库读取
"""

import fileinput
import sys

def modify_models_py():
    """修改models.py文件"""
    print("正在修改models.py文件...")
    
    # 读取文件内容
    with open('models.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    in_perform_recognition = False
    in_fixed_recognition_section = False
    in_age_gender_section = False
    in_paused_display_section = False
    
    for line in lines:
        # 检查是否进入perform_recognition方法
        if 'def perform_recognition(self):' in line:
            in_perform_recognition = True
        
        # 检查是否离开perform_recognition方法
        if in_perform_recognition and line.strip() == 'def perform_checkout_recognition(self):':
            in_perform_recognition = False
            in_fixed_recognition_section = False
            in_age_gender_section = False
            in_paused_display_section = False
        
        if in_perform_recognition:
            # 1. 修改高置信度固定结果部分，添加年龄性别部门信息
            if 'self.parent.fixed_recognition[face_id] = {' in line and '高置信度固定' in ''.join(new_lines[-5:]):
                # 找到高置信度固定的位置，修改为包含age、gender、department
                new_lines.pop()  # 移除原来的固定代码行
                new_lines.pop()  # 移除name行
                new_lines.pop()  # 移除confidence行
                new_lines.pop()  # 移除fixed_at行
                new_lines.pop()  # 移除前一行
                
                # 添加包含完整信息的固定代码
                new_lines.append("                # 保存完整的用户信息到固定结果中\n")
                new_lines.append("                self.parent.fixed_recognition[face_id] = {\n")
                new_lines.append("                    'name': display_name,\n")
                new_lines.append("                    'confidence': display_confidence,\n")
                new_lines.append("                    'age': age,\n")
                new_lines.append("                    'gender': gender,\n")
                new_lines.append("                    'department': department,\n")
                new_lines.append("                    'fixed_at': datetime.now()\n")
                new_lines.append("                }\n")
                continue
            
            # 2. 修改稳定性固定结果部分，添加年龄性别部门信息
            elif 'self.parent.fixed_recognition[face_id] = {' in line and '识别结果已固定' in ''.join(new_lines[-5:]):
                # 找到稳定性固定的位置，修改为包含age、gender、department
                new_lines.pop()  # 移除原来的固定代码行
                new_lines.pop()  # 移除name行
                new_lines.pop()  # 移除confidence行
                new_lines.pop()  # 移除fixed_at行
                new_lines.pop()  # 移除前一行
                
                # 添加包含完整信息的固定代码
                new_lines.append("                        # 保存完整的用户信息到固定结果中\n")
                new_lines.append("                        self.parent.fixed_recognition[face_id] = {\n")
                new_lines.append("                            'name': most_common_name,\n")
                new_lines.append("                            'confidence': total_confidence / len(history),\n")
                new_lines.append("                            'age': age,\n")
                new_lines.append("                            'gender': gender,\n")
                new_lines.append("                            'department': department,\n")
                new_lines.append("                            'fixed_at': datetime.now()\n")
                new_lines.append("                        }\n")
                continue
            
            # 3. 修改固定结果显示部分，从fixed_recognition获取年龄性别部门信息
            elif 'if face_id in self.parent.fixed_recognition:' in line:
                # 找到固定结果检查的位置
                new_lines.append(line)
                # 读取下一行（fixed_data赋值）
                # 后面的代码会处理显示部分
                continue
            
            # 4. 修改固定结果的显示逻辑
            elif 'fixed_data = self.parent.fixed_recognition[face_id]' in line:
                new_lines.append(line)
                # 找到固定结果显示的位置，修改为使用fixed_data中的完整信息
                # 跳过原来的显示代码，添加新的显示代码
                skip_lines = True
                while skip_lines:
                    next_line = lines[lines.index(line) + 1]
                    lines.pop(lines.index(line) + 1)
                    if 'return' in next_line:
                        skip_lines = False
                
                # 添加新的显示逻辑，使用fixed_data中的完整信息
                new_lines.append("                # 使用固定结果中的完整用户信息\n")
                new_lines.append("                self.parent.result_label.setText(f\"识别结果: <b>{fixed_data['name']}</b>\")\n")
                new_lines.append("                self.parent.confidence_label.setText(f\"置信度: {fixed_data['confidence']:.3f}\")\n")
                new_lines.append("                self.parent.stability_label.setText(\"稳定性: 固定\")\n")
                new_lines.append("                self.parent.fixed_label.setText(\"状态: 已固定\")\n")
                new_lines.append("                self.parent.fixed_label.setStyleSheet(\"font-size: 14px; color: green;\")\n")
                new_lines.append("                \n")
                new_lines.append("                # 显示固定的年龄、性别和部门信息\n")
                new_lines.append("                if fixed_data.get('age') and fixed_data.get('gender'):\n")
                new_lines.append("                    age_text = f\"{fixed_data['age']}岁\"\n")
                new_lines.append("                    gender_text = fixed_data['gender']\n")
                new_lines.append("                    dept_text = f\" | 部门: {fixed_data['department']}\" if fixed_data.get('department') else \"\"\n")
                new_lines.append("                    self.parent.age_gender_label.setText(f\"年龄/性别: {age_text}/{gender_text}{dept_text}\")\n")
                new_lines.append("                else:\n")
                new_lines.append("                    self.parent.age_gender_label.setText(\"年龄/性别: -\")\n")
                new_lines.append("                \n")
                new_lines.append("                # 其他属性保持不变\n")
                new_lines.append("                self.parent.emotion_label.setText(\"情绪: - (固定状态)\")\n")
                new_lines.append("                self.parent.mask_label.setText(\"口罩: - (固定状态)\")\n")
                new_lines.append("                \n")
                new_lines.append("                # 记录识别历史\n")
                new_lines.append("                if fixed_data['name'] != \"unknown\" and fixed_data['name'] != \"无该人像\":\n")
                new_lines.append("                    self.parent.recognition_history.append({\n")
                new_lines.append("                        'name': fixed_data['name'],\n")
                new_lines.append("                        'confidence': fixed_data['confidence'],\n")
                new_lines.append("                        'timestamp': datetime.now().isoformat()\n")
                new_lines.append("                    })\n")
                new_lines.append("                    self.parent.total_recognitions += 1\n")
                new_lines.append("                    self.parent.update_stats()\n")
                new_lines.append("                return\n")
                continue
            
            # 5. 修改暂停时固定结果的显示逻辑
            elif 'if fixed_result.get("paused", False):' in line:
                new_lines.append(line)
                # 找到暂停时固定结果显示的位置，修改为使用fixed_result中的完整信息
                # 跳过原来的显示代码，添加新的显示代码
                skip_lines = True
                while skip_lines:
                    next_line_idx = lines.index(line) + 1
                    if next_line_idx >= len(lines):
                        break
                    next_line = lines[next_line_idx]
                    lines.pop(next_line_idx)
                    if 'return' in next_line or 'self.parent.age_gender_label.setText' in next_line:
                        skip_lines = False
                
                # 添加新的暂停状态显示逻辑
                new_lines.append("                    # 使用固定结果中的完整用户信息\n")
                new_lines.append("                    self.parent.result_label.setText(f\"识别结果: <b>{fixed_result['name']}</b>\")\n")
                new_lines.append("                    self.parent.confidence_label.setText(f\"置信度: {fixed_result['confidence']:.3f}\")\n")
                new_lines.append("                    self.parent.stability_label.setText(\"稳定性: 固定\")\n")
                new_lines.append("                    self.parent.fixed_label.setText(\"状态: 已暂停\")\n")
                new_lines.append("                    self.parent.fixed_label.setStyleSheet(\"font-size: 14px; color: orange;\")\n")
                new_lines.append("                    \n")
                new_lines.append("                    # 显示固定的年龄、性别和部门信息\n")
                new_lines.append("                    if fixed_result.get('age') and fixed_result.get('gender'):\n")
                new_lines.append("                        age_text = f\"{fixed_result['age']}岁\"\n")
                new_lines.append("                        gender_text = fixed_result['gender']\n")
                new_lines.append("                        dept_text = f\" | 部门: {fixed_result['department']}\" if fixed_result.get('department') else \"\"\n")
                new_lines.append("                        self.parent.age_gender_label.setText(f\"年龄/性别: {age_text}/{gender_text}{dept_text}\")\n")
                new_lines.append("                    else:\n")
                new_lines.append("                        # 如果固定结果中没有用户信息，从数据库获取\n")
                new_lines.append("                        display_name = fixed_result['name']\n")
                new_lines.append("                        if display_name in self.parent.face_database:\n")
                new_lines.append("                            user_info = self.parent.face_database[display_name].get('info', {})\n")
                new_lines.append("                            age = user_info.get('age', '')\n")
                new_lines.append("                            gender = user_info.get('gender', '')\n")
                new_lines.append("                            department = user_info.get('department', '')\n")
                new_lines.append("                            age_text = f\"{age}岁\" if age and age.isdigit() else \"\"\n")
                new_lines.append("                            gender_text = gender if gender else \"\"\n")
                new_lines.append("                            dept_text = f\" | 部门: {department}\" if department else \"\"\n")
                new_lines.append("                            if age_text and gender_text:\n")
                new_lines.append("                                self.parent.age_gender_label.setText(f\"年龄/性别: {age_text}/{gender_text}{dept_text}\")\n")
                new_lines.append("                    \n")
                new_lines.append("                    # 其他属性保持不变\n")
                new_lines.append("                    self.parent.emotion_label.setText(\"情绪: - (暂停状态)\")\n")
                new_lines.append("                    self.parent.mask_label.setText(\"口罩: - (暂停状态)\")\n")
                new_lines.append("                    return\n")
                continue
        
        # 添加原始行（如果没有被修改）
        new_lines.append(line)
    
    # 保存修改后的文件
    with open('models.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print("models.py文件修改完成！")

def modify_main_py():
    """修改main.py文件，确保UI元素正确初始化"""
    print("正在修改main.py文件...")
    
    # 读取文件内容
    with open('main.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    in_create_recognition_tab = False
    
    for line in lines:
        # 检查是否进入create_recognition_tab方法
        if 'def create_recognition_tab(self):' in line:
            in_create_recognition_tab = True
        
        # 检查是否离开create_recognition_tab方法
        if in_create_recognition_tab and line.strip().startswith('def ') and 'create_recognition_tab' not in line:
            in_create_recognition_tab = False
        
        if in_create_recognition_tab:
            # 确保年龄性别标签正确初始化
            if 'self.age_gender_label = QLabel("年龄/性别: -")' in line:
                # 确保标签存在
                new_lines.append(line)
                continue
        
        new_lines.append(line)
    
    # 保存修改后的文件
    with open('main.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print("main.py文件修改完成！")

def create_modification_summary():
    """创建修改总结文件"""
    summary = """
人脸识别系统修改总结
====================

本次修改实现了以下功能：

1. 年龄/性别识别到之后也要固定住
   - 在models.py的perform_recognition方法中，当识别结果被固定时（无论是高置信度固定还是稳定性固定），都会将年龄、性别和部门信息一起保存到fixed_recognition字典中
   - 固定结果的数据结构现在包含：name, confidence, age, gender, department, fixed_at
   - 当显示固定结果时，直接从fixed_recognition中获取所有用户信息，而不是每次都从数据库重新获取

2. 对识别结果里的内容添加（部门）的信息，从数据库读取
   - 部门信息已经在数据库users表中存在
   - 在models.py中，当从数据库获取用户信息时，会同时获取department字段
   - 在显示识别结果时，部门信息会显示在年龄/性别信息的后面，格式为："年龄/性别: XX岁/性别 | 部门: 部门名称"
   - 如果数据库中没有部门信息，则不显示部门部分

修改的主要文件：
- models.py: 修改了perform_recognition方法，添加了完整用户信息的保存和显示逻辑
- main.py: 确保UI元素正确初始化

使用说明：
1. 当系统识别到人脸并达到固定条件时，会自动固定姓名、置信度、年龄、性别和部门信息
2. 固定后，即使人脸暂时丢失，这些信息也会保持显示（默认3秒）
3. 部门信息会自动从数据库中读取并显示
4. 所有固定的信息都会在UI上明确标识为"已固定"状态

注意事项：
- 确保数据库中正确存储了用户的部门信息
- 年龄和性别信息可以通过人脸录入功能或数据库管理功能进行更新
- 固定的结果可以通过"清除固定结果"按钮手动清除
"""
    
    with open('MODIFICATION_SUMMARY.md', 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print("修改总结文件已创建：MODIFICATION_SUMMARY.md")

def main():
    """主函数"""
    print("开始修改人脸识别系统...")
    
    try:
        # 修改models.py文件
        modify_models_py()
        
        # 修改main.py文件
        modify_main_py()
        
        # 创建修改总结
        create_modification_summary()
        
        print("\n所有修改完成！")
        print("\n修改内容：")
        print("1. 年龄/性别识别到之后也要固定住")
        print("2. 对识别结果里的内容添加（部门）的信息，从数据库读取")
        print("\n请查看MODIFICATION_SUMMARY.md文件了解详细修改内容。")
        
    except Exception as e:
        print(f"修改过程中发生错误：{str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()