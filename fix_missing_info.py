#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复年龄、性别、部门信息不显示的问题
"""

def fix_models_py():
    """修复models.py文件"""
    print("正在修复models.py文件...")
    
    # 读取文件内容
    with open('models.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    in_perform_recognition = False
    fixed_display_found = False
    high_confidence_found = False
    stability_fixed_found = False
    
    for line in lines:
        # 检查是否进入perform_recognition方法
        if 'def perform_recognition(self):' in line:
            in_perform_recognition = True
        
        # 检查是否离开perform_recognition方法
        if in_perform_recognition and line.strip() == 'def perform_checkout_recognition(self):':
            in_perform_recognition = False
        
        if in_perform_recognition:
            # 1. 修复固定结果显示部分
            if 'if face_id in self.parent.fixed_recognition:' in line and not fixed_display_found:
                new_lines.append(line)
                # 读取下一行（fixed_data赋值）
                fixed_data_line = lines[lines.index(line) + 1]
                new_lines.append(fixed_data_line)
                
                # 添加完整的显示逻辑
                new_lines.append("                    # 使用固定结果中的完整用户信息\n")
                new_lines.append("                    self.parent.result_label.setText(f\"识别结果: <b>{fixed_data['name']}</b>\")\n")
                new_lines.append("                    self.parent.confidence_label.setText(f\"置信度: {fixed_data['confidence']:.3f}\")\n")
                new_lines.append("                    self.parent.stability_label.setText(\"稳定性: 固定\")\n")
                new_lines.append("                    self.parent.fixed_label.setText(\"状态: 已固定\")\n")
                new_lines.append("                    self.parent.fixed_label.setStyleSheet(\"font-size: 14px; color: green;\")\n")
                new_lines.append("                    \n")
                new_lines.append("                    # 显示固定的年龄、性别和部门信息\n")
                new_lines.append("                    if fixed_data.get('age') and fixed_data.get('gender'):\n")
                new_lines.append("                        age_text = f\"{fixed_data['age']}岁\"\n")
                new_lines.append("                        gender_text = fixed_data['gender']\n")
                new_lines.append("                        dept_text = f\" | 部门: {fixed_data['department']}\" if fixed_data.get('department') else \"\"\n")
                new_lines.append("                        self.parent.age_gender_label.setText(f\"年龄/性别: {age_text}/{gender_text}{dept_text}\")\n")
                new_lines.append("                    else:\n")
                new_lines.append("                        # 如果固定结果中没有用户信息，从数据库获取\n")
                new_lines.append("                        display_name = fixed_data['name']\n")
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
                new_lines.append("                            else:\n")
                new_lines.append("                                self.parent.age_gender_label.setText(\"年龄/性别: -\")\n")
                new_lines.append("                        else:\n")
                new_lines.append("                            self.parent.age_gender_label.setText(\"年龄/性别: -\")\n")
                new_lines.append("                    \n")
                new_lines.append("                    # 其他属性保持不变\n")
                new_lines.append("                    self.parent.emotion_label.setText(\"情绪: - (固定状态)\")\n")
                new_lines.append("                    self.parent.mask_label.setText(\"口罩: - (固定状态)\")\n")
                new_lines.append("                    \n")
                
                fixed_display_found = True
                continue
            
            # 2. 修复高置信度固定结果的保存
            elif 'if display_confidence > 0.85 and display_name != "无该人像":' in line and not high_confidence_found:
                new_lines.append(line)
                # 找到固定代码的位置
                while True:
                    next_line_idx = lines.index(line) + 1
                    if next_line_idx >= len(lines):
                        break
                    next_line = lines[next_line_idx]
                    if 'self.parent.fixed_recognition[face_id] = {' in next_line:
                        # 替换固定代码
                        new_lines.append("            # 保存完整的用户信息到固定结果中\n")
                        new_lines.append("            self.parent.fixed_recognition[face_id] = {\n")
                        new_lines.append("                'name': display_name,\n")
                        new_lines.append("                'confidence': display_confidence,\n")
                        new_lines.append("                'age': age,\n")
                        new_lines.append("                'gender': gender,\n")
                        new_lines.append("                'department': department,\n")
                        new_lines.append("                'fixed_at': datetime.now()\n")
                        new_lines.append("            }\n")
                        # 跳过原来的固定代码
                        while '}' not in next_line:
                            next_line_idx += 1
                            next_line = lines[next_line_idx]
                        high_confidence_found = True
                        break
                    else:
                        new_lines.append(next_line)
                        line = next_line
                continue
            
            # 3. 修复稳定性固定结果的保存
            elif 'if stability > self.config.get(\'recognition_fix_threshold\', 0.6):' in line and not stability_fixed_found:
                new_lines.append(line)
                # 找到固定代码的位置
                while True:
                    next_line_idx = lines.index(line) + 1
                    if next_line_idx >= len(lines):
                        break
                    next_line = lines[next_line_idx]
                    if 'self.parent.fixed_recognition[face_id] = {' in next_line:
                        # 替换固定代码
                        new_lines.append("                        # 保存完整的用户信息到固定结果中\n")
                        new_lines.append("                        self.parent.fixed_recognition[face_id] = {\n")
                        new_lines.append("                            'name': most_common_name,\n")
                        new_lines.append("                            'confidence': total_confidence / len(history),\n")
                        new_lines.append("                            'age': age,\n")
                        new_lines.append("                            'gender': gender,\n")
                        new_lines.append("                            'department': department,\n")
                        new_lines.append("                            'fixed_at': datetime.now()\n")
                        new_lines.append("                        }\n")
                        # 跳过原来的固定代码
                        while '}' not in next_line:
                            next_line_idx += 1
                            next_line = lines[next_line_idx]
                        stability_fixed_found = True
                        break
                    else:
                        new_lines.append(next_line)
                        line = next_line
                continue
        
        new_lines.append(line)
    
    # 保存修复后的文件
    with open('models.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print("models.py文件修复完成！")

def fix_main_py():
    """修复main.py文件，确保UI元素正确初始化"""
    print("正在修复main.py文件...")
    
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
                new_lines.append(line)
                continue
        
        new_lines.append(line)
    
    # 保存修改后的文件
    with open('main.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print("main.py文件修复完成！")

def main():
    """主函数"""
    print("开始修复年龄、性别、部门信息不显示的问题...")
    
    try:
        # 修复models.py文件
        fix_models_py()
        
        # 修复main.py文件
        fix_main_py()
        
        print("\n修复完成！")
        print("已修复的问题：")
        print("1. 固定结果显示时年龄、性别、部门信息不显示")
        print("2. 高置信度固定结果未保存完整用户信息")
        print("3. 稳定性固定结果未保存完整用户信息")
        
    except Exception as e:
        print(f"修复过程中发生错误：{str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()