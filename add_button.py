import os

os.chdir(r'c:\NCHC_DATA\flydata')

# 讀取文件
with open('demo_with_real_data.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f'總共 {len(lines)} 行')

# 找到清除按鈕的行
found = False
for i, line in enumerate(lines):
    if '清除所有結果' in line:
        print(f'找到清除按鈕在第 {i+1} 行: {repr(line)}')
        # 在下一行插入新按鈕
        new_button = '                <button class="btn" onclick="showAllFlights()" id="showAllBtn" disabled style="opacity: 0.5; background: #2196f3;">👁️ 顯示所有航班</button>\n'
        lines.insert(i + 1, new_button)
        found = True
        print('新按鈕已插入')
        break

if not found:
    print('沒有找到清除按鈕')

# 寫回文件
with open('demo_with_real_data.html', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('文件已更新')
