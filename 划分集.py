import os
import random
import json

# 定义文件路径
file_path = 'C:/Users/admin/PycharmProjects/PythonProject/12-指标分析-账户分析-简答题.json'

# 检查文件是否存在
if os.path.exists(file_path):
    if os.path.isfile(file_path):
        try:
            # 读取 JSON 文件
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # 确保数据是列表形式
            if isinstance(data, list):
                # 随机打乱顺序
                random.shuffle(data)

                # 抽取 15 个作为测试集
                test_set = data[:3]
                # 剩下的作为训练集
                train_set = data[3:]

                # 保存测试集和训练集到 JSON 文件
                with open('12-指标分析-账户分析-简答题test_set.json', 'w', encoding='utf-8') as test_file:
                    json.dump(test_set, test_file, ensure_ascii=False, indent=4)

                with open('12-指标分析-账户分析-简答题train_set.json', 'w', encoding='utf-8') as train_file:
                    json.dump(train_set, train_file, ensure_ascii=False, indent=4)

                print("测试集和训练集已保存到 12-指标分析-账户分析-简答题test_set.json 和 12-指标分析-账户分析-简答题train_set.json 文件中。")
            else:
                print("JSON 文件内容不是列表形式，请检查文件结构。")
        except json.JSONDecodeError:
            print("无法解析 JSON 文件，请检查文件格式是否正确。")
    else:
        print(f"{file_path} 是一个文件夹，不是文件。")
else:
    print(f"文件 {file_path} 不存在，请检查文件路径是否正确。")