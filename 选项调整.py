import pandas as pd
import re

# 读取csv文件
df = pd.read_csv('C:/Users/admin/Desktop/选项异常CFBenchmark整选择题-20250506.csv')  # 请将此处替换为实际的csv文件名

# 定义正则表达式模式
pattern = r'^[A-D]\.\s*'

# 对每一列进行处理
columns_to_process = ['选项A内容', '选项B内容', '选项C内容', '选项D内容']  # 根据实际列名调整
for col in columns_to_process:
    df[col] = df[col].str.replace(pattern, '', regex=True)

# 保存修改后的csv文件
df.to_csv('修改后的选择题.csv', index=False)  # 请将此处替换为想要保存的文件名