import csv
import json
import re
import chardet


def formula_to_latex(text):
    """将文本中的公式转换为 LaTeX 格式"""
    # 简单示例：假设公式用 $公式内容$ 表示，这里仅做简单提取
    pattern = r'\$(.*?)\$'
    return re.sub(pattern, r'\\(\1\\)', text)


def remove_option_label(option):
    """去除选项内容开头的标识（如 A.、A、等）"""
    return option.lstrip('ABCDE.、 ').lstrip()


data = []
csv_file_path = 'C:/Users/admin/Desktop/CFBenchmark整合--选择题.csv'
with open(csv_file_path, 'rb') as f:
    raw_data = f.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']

with open(csv_file_path, 'r', encoding=encoding) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # 跳过空行
        if all(not value.strip() for value in row.values()):
            continue

        # 处理答案字段，去除逗号、顿号和空格
        answer = row['答案'].replace(',', '').replace(' ', '').replace('、', '')

        # 修改 id 的生成逻辑
        item = {
            "id": f"{row['类型']}-{row['序号(统计用）']}",
            "uid": row['唯一ID'],
            "question": formula_to_latex(row['问题']),
            "A": formula_to_latex(remove_option_label(row['选项A内容'])),
            "B": formula_to_latex(remove_option_label(row['选项B内容'])),
            "C": formula_to_latex(remove_option_label(row['选项C内容'])),
            "D": formula_to_latex(remove_option_label(row['选项D内容'])),
            "answer": answer,
            "cot": formula_to_latex(row['思考过程.思考过程']),
            "output": formula_to_latex(row['思考过程.输出结果']),
            "check": formula_to_latex(row['检查']),
            "result": formula_to_latex(row['检查结论']).replace('\n', ' ')
        }

        if row['选项E内容'].strip():
            item['E'] = formula_to_latex(remove_option_label(row['选项E内容']))

        # 在删除前检查键是否存在
        if 'E' in item and not row['选项E内容'].strip():
            del item['E']

        data.append(item)

# 保存为 JSON 文件
json_file_path = '选择题测试集.json'
with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
    json.dump(data, jsonfile, ensure_ascii=False, indent=4)