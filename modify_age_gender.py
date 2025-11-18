import fileinput

# 修改年龄性别部门信息的获取方式
replace_lines = []
in_target_section = False

for line in fileinput.input('models.py', inplace=True):
    if '        # 模拟其他属性检测' in line:
        in_target_section = True
        # 开始替换
        print('        # 修复问题2：从数据库获取用户信息')
        print('        age = None')
        print('        gender = None')
        print('        department = None')
        print('        emotion = None')
        print('        mask = None')
        print('        ')
        print('        if display_name != "无该人像" and display_name in self.parent.face_database:')
        print('            # 从数据库获取用户信息')
        print('            user_info = self.parent.face_database[display_name].get(\'info\', {})')
        print('            age = user_info.get(\'age\', \'\')')
        print('            gender = user_info.get(\'gender\', \'\')')
        print('            department = user_info.get(\'department\', \'\')')
        print('            ')
        print('            # 如果数据库中没有年龄或性别，使用随机值作为 fallback')
        print('            if not age or not age.isdigit():')
        print('                age = random.randint(18, 60)')
        print('            else:')
        print('                age = int(age)')
        print('                ')
        print('            if not gender:')
        print('                gender = random.choice(["男", "女"])')
        print('                ')
        print('            # 其他属性仍使用模拟数据')
        print('            emotion = random.choice(["happy", "neutral", "sad", "angry", "surprised"])')
        print('            mask = random.choice(["wearing", "not wearing"])')
        print('        ')
        print('        # 显示年龄、性别和部门信息')
        print('        if age and gender:')
        print('            age_text = f"{age}岁"')
        print('            gender_text = gender')
        print('            dept_text = f" | 部门: {department}" if department else ""')
        print('            self.parent.age_gender_label.setText(f"年龄/性别: {age_text}/{gender_text}{dept_text}")')
        print('        else:')
        print('            self.parent.age_gender_label.setText("年龄/性别: -")')
    elif in_target_section and 'self.parent.emotion_label.setText' in line:
        in_target_section = False
        print(line, end='')
    elif not in_target_section:
        print(line, end='')

print("年龄性别部门信息获取方式修改完成")