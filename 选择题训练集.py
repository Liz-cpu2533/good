import pandas as pd
import json

# 读取csv文件
csv_df = pd.read_csv('C:/Users/admin/Desktop/CFBenchmark整合--选择题.csv')  # 请替换为实际csv文件名

# 读取json文件
with open('C:/Users/admin/Desktop/CFBenchmark整合选择题-20250506  test  id.json', 'r', encoding='utf-8') as f:
    json_data = json.load(f)
    all_id_list = []
    for key, value in json_data.items():
        if isinstance(value, list):
            all_id_list.extend(value)

# 筛选出csv中不包含json中ID的行
result_df = csv_df[~csv_df['唯一ID'].isin(all_id_list)]

# 保存结果为新的csv文件
result_df.to_csv('去重后的简答题训练集.csv', index=False)