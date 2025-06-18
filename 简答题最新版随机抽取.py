import pandas as pd
import random
import json
import os

# 设置随机种子
random.seed(42)

# 文件路径
file_path = 'C:/Users/admin/Desktop/CFBenchmark整合简答题-20250506.csv'

# 读取CSV文件
df = pd.read_csv(file_path)

# 存储抽取结果
extracted_data = {}

# 设定不抽取的类型和抽取数量
types_to_exclude = ['合规-法律法规', '指标分析-投资收益', '指标分析-账户分析']
sampling_numbers = {
    '基础知识-期货基础': 20
}

for type_ in df['类型'].unique():
    if type_ in types_to_exclude:
        continue
    num_to_sample = sampling_numbers.get(type_, 50)
    type_df = df[df['类型'] == type_]
    # 打乱顺序
    shuffled_df = type_df.sample(frac=1).reset_index(drop=True)
    # 随机抽取指定数量
    sampled_df = shuffled_df[:num_to_sample]
    unique_ids = sampled_df['唯一ID'].tolist()
    extracted_data[type_] = unique_ids
    # 保存为CSV文件
    sampled_df.to_csv(f'{type_}_sampled.csv', index=False)

# 保存为JSON文件
with open('extracted_data.json', 'w', encoding='utf-8') as f:
    json.dump(extracted_data, f, ensure_ascii=False, indent=4)

#合并CSV文件（如果有需要）
csv_files = [f for f in os.listdir('.') if f.endswith('_sampled.csv')]
combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
combined_df.to_csv('简答题合并的csv', index=False)

#删除中间生成的CSV文件（可选）
for f in csv_files:
    os.remove(f)