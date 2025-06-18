import csv
import json
import jieba

# 定义 CSV 文件路径和 JSON 文件路径
csv_file_path = 'C:/Users/admin/Desktop/简答题训练集.csv'
json_file_path = '简答题实体识别有误.json'

# 用于存储符合条件的唯一 ID
unique_ids = []

# 打开 CSV 文件
with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    # 遍历每一行
    for row in reader:
        # 检查类型是否为实体识别
        if row.get('类型') == '实体识别-实体识别':
            question = row.get('问题', '')
            answer = row.get('原答案', '')
            unique_id = row.get('唯一ID')
            if unique_id == "STSB-JD-01741":
                a=1
            # 对问题和答案进行分词
            question_words = set(jieba.lcut(question))
            answer_words = jieba.lcut(answer)

            # 检查答案中的词是否都在问题中出现
            if not all(word in question_words for word in answer_words):
                unique_ids.append(unique_id)

# 将唯一 ID 保存到 JSON 文件中
result = {"唯一ID": unique_ids}
with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
    json.dump(result, jsonfile, ensure_ascii=False, indent=4)

print(f"符合条件的唯一 ID 已保存到 {json_file_path}")