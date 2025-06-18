import pandas as pd
import random
import json
import os

# 设置随机种子
random.seed(42)

# 文件路径
file_path = 'C:/Users/admin/Desktop/CFBenchmark整合选择题-20250506.csv'

# 读取 CSV 文件
df = pd.read_csv(file_path, encoding='utf-8-sig')

# 存储抽取结果
extracted_data = {}

# 排除风险控制类型，对其他类型进行处理
types_to_exclude = ['指标分析-风险控制']
for type_ in df['类型'].unique():
    if type_ not in types_to_exclude:
        type_df = df[df['类型'] == type_]
        # 打乱顺序
        shuffled_df = type_df.sample(frac=1).reset_index(drop=True)
        # 随机抽取 50 个
        sampled_df = shuffled_df[:50]
        unique_ids = sampled_df['唯一ID'].tolist()
        extracted_data[type_] = unique_ids
        # 保存为 CSV 文件
        sampled_df.to_csv(f'{type_}_sampled.csv', index=False, encoding='utf-8-sig', lineterminator='\n')

# 保存为 JSON 文件
with open('extracted_data.json', 'w', encoding='utf-8') as f:
    json.dump(extracted_data, f, ensure_ascii=False, indent=4)

# 合并 CSV 文件
csv_files = [f for f in os.listdir('.') if f.endswith('_sampled.csv')]
combined_df = pd.concat([pd.read_csv(f, encoding='utf-8-sig') for f in csv_files], ignore_index=True)

# 保存合并后的 CSV 文件
combined_df.to_csv('选择题合并的csv', index=False, encoding='utf-8-sig', lineterminator='\n')

# 删除中间生成的 CSV 文件（可选）
for f in csv_files:
    os.remove(f)
