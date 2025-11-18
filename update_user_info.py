# 读取models.py文件
with open('models.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 查找插入位置
insert_pos = None
for i, line in enumerate(lines):
    if '        self.parent.mask_label.setText(f"口罩: {mask}" if mask else "口罩: -")' in line:
        insert_pos = i
        break

if insert_pos is not None:
    # 插入新代码
    new_code = '''        # 修复问题2：从数据库获取用户信息
        age = None
        gender = None
        department = None
        emotion = None
        mask = None
        
        if display_name != "无该人像" and display_name in self.parent.face_database:
            # 从数据库获取用户信息
            user_info = self.parent.face_database[display_name].get('info', {})
            age = user_info.get('age', '')
            gender = user_info.get('gender', '')
            department = user_info.get('department', '')
            
            # 如果数据库中没有年龄或性别，使用随机值作为 fallback
            if not age or not age.isdigit():
                age = random.randint(18, 60)
            else:
                age = int(age)
                
            if not gender:
                gender = random.choice(["男", "女"])
                
            # 其他属性仍使用模拟数据
            emotion = random.choice(["happy", "neutral", "sad", "angry", "surprised"])
            mask = random.choice(["wearing", "not wearing"])
        
        # 显示年龄、性别和部门信息
        if age and gender:
            age_text = f"{age}岁"
            gender_text = gender
            dept_text = f" | 部门: {department}" if department else ""
            self.parent.age_gender_label.setText(f"年龄/性别: {age_text}/{gender_text}{dept_text}")
        else:
            self.parent.age_gender_label.setText("年龄/性别: -")
'''
    
    # 插入新代码
    lines.insert(insert_pos, new_code)
    
    # 写回文件
    with open('models.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("用户信息获取方式修改完成")
else:
    print("未找到插入位置")