# 读取models.py文件
with open('models.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 查找目标位置
target_start = None
for i, line in enumerate(lines):
    if '        else:' in line and i + 1 < len(lines) and '            # 未检测到人脸时，清除固定结果' in lines[i+1]:
        target_start = i
        break

if target_start is not None:
    # 找到目标位置，开始替换
    new_lines = []
    
    # 添加前面的行
    new_lines.extend(lines[:target_start])
    
    # 添加新的代码
    new_code = '''        else:
            # 修复问题1：优化人脸丢失处理，减少误清除
            # 未检测到人脸时，延迟清除固定结果，提高稳定性
            face_id = 0
            if face_id in self.parent.fixed_recognition:
                # 检查固定结果是否是暂停时固定的
                fixed_result = self.parent.fixed_recognition[face_id]
                if fixed_result.get('paused', False):
                    # 如果是暂停时固定的结果，保持不变
                    self.parent.result_label.setText(f"识别结果: <b>{fixed_result['name']}</b>")
                    self.parent.confidence_label.setText(f"置信度: {fixed_result['confidence']:.3f}")
                    self.parent.stability_label.setText("稳定性: 固定")
                    self.parent.fixed_label.setText("状态: 已暂停")
                    self.parent.fixed_label.setStyleSheet("font-size: 14px; color: orange;")
                    
                    # 从数据库获取用户信息显示
                    display_name = fixed_result['name']
                    if display_name in self.parent.face_database:
                        user_info = self.parent.face_database[display_name].get('info', {})
                        age = user_info.get('age', '')
                        gender = user_info.get('gender', '')
                        department = user_info.get('department', '')
                        age_text = f"{age}岁" if age and age.isdigit() else ""
                        gender_text = gender if gender else ""
                        dept_text = f" | 部门: {department}" if department else ""
                        if age_text and gender_text:
                            self.parent.age_gender_label.setText(f"年龄/性别: {age_text}/{gender_text}{dept_text}")
                    return
                
                # 如果不是暂停固定的结果，延迟清除
                fixed_name = fixed_result['name']
                fixed_time = fixed_result.get('fixed_at', datetime.now())
                time_since_fixed = (datetime.now() - fixed_time).total_seconds()
                
                # 固定结果保持3秒后再清除，提高稳定性
                if time_since_fixed < 3:
                    self.parent.result_label.setText(f"识别结果: <b>{fixed_name}</b>")
                    self.parent.confidence_label.setText(f"置信度: {fixed_result['confidence']:.3f}")
                    self.parent.stability_label.setText("稳定性: 固定")
                    self.parent.fixed_label.setText("状态: 已固定")
                    self.parent.fixed_label.setStyleSheet("font-size: 14px; color: green;")
                    return
                
                # 超过3秒未检测到人脸，清除固定结果
                self.parent.update_log(f"人脸丢失超过3秒，清除固定结果: {fixed_name}")
                del self.parent.fixed_recognition[face_id]
            
            # 未检测到人脸
            self.parent.result_label.setText("未检测到人脸")
            self.parent.confidence_label.setText("置信度: -")
            self.parent.stability_label.setText("稳定性: -")
            self.parent.fixed_label.setText("状态: 实时")
            self.parent.fixed_label.setStyleSheet("font-size: 14px; color: blue;")
            self.parent.age_gender_label.setText("年龄/性别: -")
            self.parent.emotion_label.setText("情绪: -")
            self.parent.mask_label.setText("口罩: -")'''
    
    new_lines.append(new_code)
    
    # 找到结束位置，跳过原有的代码
    target_end = target_start
    while target_end < len(lines) and not lines[target_end].strip().startswith('except Exception as e:'):
        target_end += 1
    
    # 添加后面的行
    new_lines.extend(lines[target_end:])
    
    # 写回文件
    with open('models.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print("人脸丢失处理逻辑修改完成")
else:
    print("未找到目标代码位置")