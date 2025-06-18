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
csv_file_path = 'C:/Users/admin/Desktop/选择题.csv'
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

        item = {
            "id": f"合规-法律法规-选择题-{row['序号']}",
            "question": formula_to_latex(row['问题']),
            "A": formula_to_latex(remove_option_label(row['选项A'])),
            "B": formula_to_latex(remove_option_label(row['选项B'])),
            "C": formula_to_latex(remove_option_label(row['选项C'])),
            "D": formula_to_latex(remove_option_label(row['选项D'])),
            "answer": answer,
            "cot": formula_to_latex(row['思考过程.思考过程']),
            "output": formula_to_latex(row['思考过程.输出结果']),
            "check": formula_to_latex(row['检查']),
            "result": formula_to_latex(row['检查结论']).replace('\n', ' ')
        }

        if row['选项E'].strip():
            item['E'] = formula_to_latex(remove_option_label(row['选项E']))

        # 在删除前检查键是否存在
        if 'E' in item and not row['选项E'].strip():
            del item['E']

        data.append(item)

# 保存为 JSON 文件
json_file_path = '3-合规-法律法规-选择题-未修订版.json'
with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
    json.dump(data, jsonfile, ensure_ascii=False, indent=4)
