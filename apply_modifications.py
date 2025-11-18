#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接应用所有必要的修改
"""

def modify_models():
    """修改models.py文件"""
    print("正在修改models.py文件...")
    
    # 读取文件内容
    with open('models.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. 修改固定结果显示部分
    content = content.replace(
        "if face_id in self.parent.fixed_recognition:\n"
        "                    fixed_data = self.parent.fixed_recognition[face_id]\n"
        "                    self.parent.result_label.setText(f\"识别结果: <b>{fixed_data['name']}</b>\")\n"
        "                    self.parent.confidence_label.setText(f\"置信度: {fixed_data['confidence']:.3f}\")\n"
        "                    self.parent.stability_label.setText(\"稳定性: 固定\")\n"
        "                    self.parent.fixed_label.                    if display_name != \"无该人像\" and display_name in self.parent.face_database:",
        "if face_id in self.parent.fixed_recognition:\n"
        "                    fixed_data = self.parent.fixed_recognition[face_id]\n"
        "                    # 使用固定结果中的完整用户信息\n"
        "                    self.parent.result_label.setText(f\"识别结果: <b>{fixed_data['name']}</b>\")\n"
        "                    self.parent.confidence_label.setText(f\"置信度: {fixed_data['confidence']:.3f}\")\n"
        "                    self.parent.stability_label.setText(\"稳定性: 固定\")\n"
        "                    self.parent.fixed_label.setText(\"状态: 已固定\")\n"
        "                    self.parent.fixed_label.setStyleSheet(\"font-size: 14px; color: green;\")\n"
        "                    \n"
        "                    # 显示固定的年龄、性别和部门信息\n"
        "                    if fixed_data.get('age') and fixed_data.get('gender'):\n"
        "                        age_text = f\"{fixed_data['age']}岁\"\n"
        "                        gender_text = fixed_data['gender']\n"
        "                        dept_text = f\" | 部门: {fixed_data['department']}\" if fixed_data.get('department') else \"\"\n"
        "                        self.parent.age_gender_label.setText(f\"年龄/性别: {age_text}/{gender_text}{dept_text}\")\n"
        "                    else:\n"
        "                        self.parent.age_gender_label.setText(\"年龄/性别: -\")\n"
        "                    \n"
        "                    # 其他属性保持不变\n"
        "                    self.parent.emotion_label.setText(\"情绪: - (固定状态)\")\n"
        "                    self.parent.mask_label.setText(\"口罩: - (固定状态)\")\n"
        "                    \n"
        "                    # 记录识别历史\n"
        "                    if fixed_data['name'] != \"unknown\" and fixed_data['name'] != \"无该人像\":\n"
        "                        self.parent.recognition_history.append({\n"
        "                            'name': fixed_data['name'],\n"
        "                            'confidence': fixed_data['confidence'],\n"
        "                            'timestamp': datetime.now().isoformat()\n"
        "                        })\n"
        "                        self.parent.total_recognitions += 1\n"
        "                        self.parent.update_stats()\n"
        "                    return\n"
        "                    if display_name != \"无该人像\" and display_name in self.parent.face_database:"
    )
    
    # 2. 修改高置信度固定结果的保存
    content = content.replace(
        "            self.parent.fixed_recognition[face_id] = {\n"
        "                'name': display_name,\n"
        "                'confidence': display_confidence,\n"
        "                'fixed_at': datetime.now()\n"
        "            }",
        "            # 保存完整的用户信息到固定结果中\n"
        "            self.parent.fixed_recognition[face_id] = {\n"
        "                'name': display_name,\n"
        "                'confidence': display_confidence,\n"
        "                'age': age,\n"
        "                'gender': gender,\n"
        "                'department': department,\n"
        "                'fixed_at': datetime.now()\n"
        "            }"
    )
    
    # 3. 修改稳定性固定结果的保存
    content = content.replace(
        "                        self.parent.fixed_recognition[face_id] = {\n"
        "                            'name': most_common_name,\n"
        "                            'confidence': total_confidence / len(history),\n"
        "                            'fixed_at': datetime.now()\n"
        "                        }",
        "                        # 保存完整的用户信息到固定结果中\n"
        "                        self.parent.fixed_recognition[face_id] = {\n"
        "                            'name': most_common_name,\n"
        "                            'confidence': total_confidence / len(history),\n"
        "                            'age': age,\n"
        "                            'gender': gender,\n"
        "                            'department': department,\n"
        "                            'fixed_at': datetime.now()\n"
        "                        }"
    )
    
    # 4. 修改暂停时固定结果的显示
    content = content.replace(
        "                    # 如果是暂停时固定的结果，保持不变\n"
        "                    self.parent.result_label.setText(f\"识别结果: <b>{fixed_result['name']}</b>\")\n"
        "                    self.parent.confidence_label.setText(f\"置信度: {fixed_result['confidence']:.3f}\")\n"
        "                    self.parent.sta\n"
        "bility_label.setText(\"稳定性: 固定\")\n"
        "                    self.parent.fixed_label.setText(\"状态: 已暂停\")\n"
        "                    self.parent.fixed_label.setStyleSheet(\"font-size: 14px; color: orange;\")\n"
        "                    # 从数据库获取用户信息显示\n",
        "                    # 使用固定结果中的完整用户信息\n"
        "                    self.parent.result_label.setText(f\"识别结果: <b>{fixed_result['name']}</b>\")\n"
        "                    self.parent.confidence_label.setText(f\"置信度: {fixed_result['confidence']:.3f}\")\n"
        "                    self.parent.stability_label.setText(\"稳定性: 固定\")\n"
        "                    self.parent.fixed_label.setText(\"状态: 已暂停\")\n"
        "                    self.parent.fixed_label.setStyleSheet(\"font-size: 14px; color: orange;\")\n"
        "                    \n"
        "                    # 显示固定的年龄、性别和部门信息\n"
        "                    if fixed_result.get('age') and fixed_result.get('gender'):\n"
        "                        age_text = f\"{fixed_result['age']}岁\"\n"
        "                        gender_text = fixed_result['gender']\n"
        "                        dept_text = f\" | 部门: {fixed_result['department']}\" if fixed_result.get('department') else \"\"\n"
        "                        self.parent.age_gender_label.setText(f\"年龄/性别: {age_text}/{gender_text}{dept_text}\")\n"
        "                    else:\n"
        "                        # 如果固定结果中没有用户信息，从数据库获取\n"
        "                        display_name = fixed_result['name']\n"
        "                        if display_name in self.parent.face_database:\n"
        "                            user_info = self.parent.face_database[display_name].get('info', {})\n"
        "                            age = user_info.get('age', '')\n"
        "                            gender = user_info.get('gender', '')\n"
        "                            department = user_info.get('department', '')\n"
        "                            age_text = f\"{age}岁\" if age and age.isdigit() else \"\"\n"
        "                            gender_text = gender if gender else \"\"\n"
        "                            dept_text = f\" | 部门: {department}\" if department else \"\"\n"
        "                            if age_text and gender_text:\n"
        "                                self.parent.age_gender_label.setText(f\"年龄/性别: {age_text}/{gender_text}{dept_text}\")\n"
        "                    \n"
        "                    # 其他属性保持不变\n"
        "                    self.parent.emotion_label.setText(\"情绪: - (暂停状态)\")\n"
        "                    self.parent.mask_label.setText(\"口罩: - (暂停状态)\")\n"
        "                    return\n"
        "                    # 如果不是暂停固定的结果，延迟清除"
    )
    
    # 保存修改后的文件
    with open('models.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("models.py文件修改完成！")

def main():
    """主函数"""
    print("开始应用修改...")
    
    try:
        # 修改models.py文件
        modify_models()
        
        print("所有修改应用完成！")
        
    except Exception as e:
        print(f"修改过程中发生错误：{str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()