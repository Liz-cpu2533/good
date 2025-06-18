import os
import pandas as pd
import re

# 原文件夹路径
source_folder = r'C:\Users\admin\Desktop\选择题训练集0512'

# 定义输出文件路径
output_all = os.path.join(os.getcwd(), 'new_cot_answers_merged.csv')
output_matched = os.path.join(os.getcwd(), 'new_cot_answers_matched_merged.csv')
output_unmatched = os.path.join(os.getcwd(), 'new_cot_answers_unmatched_merged.csv')

# 初始化数据框列表
all_dfs = []
matched_dfs = []
unmatched_dfs = []

# 正则表达式模式 - 适应实际文件名格式
pattern_matched = re.compile(r'new_cot_answers\d+_matched\.csv')
pattern_unmatched = re.compile(r'new_cot_answers\d+_unmatched\.csv')
pattern_original = re.compile(r'new_cot_answers_\d+\.csv')  # 原始文件有下划线

# 遍历原文件夹中的所有CSV文件
for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(root, file)
            try:
                # 读取CSV文件
                df = pd.read_csv(file_path)

                # 使用正则表达式精确匹配
                if pattern_matched.match(file):
                    matched_dfs.append(df)
                elif pattern_unmatched.match(file):
                    unmatched_dfs.append(df)
                elif pattern_original.match(file):
                    all_dfs.append(df)
                else:
                    print(f"跳过未匹配的文件: {file}")
            except Exception as e:
                print(f"Error reading {file}: {e}")

# 合并数据框并保存为CSV
if all_dfs:
    pd.concat(all_dfs, ignore_index=True).to_csv(output_all, index=False)
    print(f"已合并 {len(all_dfs)} 个原始类型的文件到 {output_all}")

if matched_dfs:
    pd.concat(matched_dfs, ignore_index=True).to_csv(output_matched, index=False)
    print(f"已合并 {len(matched_dfs)} 个 matched 类型的文件到 {output_matched}")

if unmatched_dfs:
    pd.concat(unmatched_dfs, ignore_index=True).to_csv(output_unmatched, index=False)
    print(f"已合并 {len(unmatched_dfs)} 个 unmatched 类型的文件到 {output_unmatched}")