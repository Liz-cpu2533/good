import pandas as pd

file_path = 'C:/Users/admin/Desktop/new_cot_answers_016.csv'
# 尝试指定编码读取，这里假设是utf - 8，可按需调整
df = pd.read_csv(file_path, encoding='utf - 8')

# 统一数据类型为字符串
df['答案'] = df['答案'].astype(str)
df['输出结果'] = df['输出结果'].astype(str)

# 去除前后空白字符
df['答案'] = df['答案'].str.strip()
df['输出结果'] = df['输出结果'].str.strip()

# 去除换行符等不可见字符
df['答案'] = df['答案'].str.replace('\n', '').str.replace('\t', '')
df['输出结果'] = df['输出结果'].str.replace('\n', '').str.replace('\t', '')

# 处理缺失值
df = df.dropna(subset=['答案', '输出结果'])

# 筛选出答案和输出结果一致的行
matched_df = df[df['答案'] == df['输出结果']]
# 筛选出答案和输出结果不一致的行
unmatched_df = df[df['答案'] != df['输出结果']]

original_count = len(df)
filtered_count = len(matched_df)

print(f"原始数据行数: {original_count}")
print(f"筛选后保留行数（一致的）: {filtered_count}")
print(f"筛选后保留行数（不一致的）: {len(unmatched_df)}")

# 保存一致的数据到一个CSV文件
new_file_path_matched = 'C:/Users/admin/Desktop/new_cot_answers016_matched.csv'
matched_df.to_csv(new_file_path_matched, index=False)
print(f"已成功将一致的数据保存至: {new_file_path_matched}")

# 保存不一致的数据到另一个CSV文件
new_file_path_unmatched = 'C:/Users/admin/Desktop/new_cot_answers016_unmatched.csv'
unmatched_df.to_csv(new_file_path_unmatched, index=False)
print(f"已成功将不一致的数据保存至: {new_file_path_unmatched}")