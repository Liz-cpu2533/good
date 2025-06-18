import csv
import json
import re
import chardet


def convert_to_latex(s):
    """将字符串中的公式转换为 LaTeX 格式"""
    return re.sub(r'\$(.*?)\$', r'\\\(\1\\\)', s)


def parse_criteria(integration_str):
    """解析整合列生成评分标准字典"""
    criteria = {}
    pattern = r'"criteri[uo]m(\d+)"\s*:\s*{\s*"content"\s*:\s*"([^"]+)"\s*,\s*"score"\s*:\s*([0-9.]+)\s*}'
    matches = re.findall(pattern, integration_str)

    for match in matches:
        index = int(match[0])
        criteria[f"criterium{index}"] = {
            "content": match[1],
            "score": float(match[2])
        }

    # 补全缺失的评分标准
    for i in range(1, 6):
        key = f"criterium{i}"
        if key not in criteria:
            criteria[key] = {
                "content": "未定义",
                "score": 0.0
            }

    return criteria


# 定义 CSV 文件路径和输出的 JSON 文件路径
csv_file_path = 'C:/Users/admin/Desktop/jaindatiti.csv'
json_file_path = '简答题训练集.json'

# 用于存储处理后的数据
new_data = []

# 以二进制模式打开文件以检测编码
with open(csv_file_path, 'rb') as f:
    raw_data = f.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']

# 使用检测到的编码打开文件
with open(csv_file_path, 'r', encoding=encoding) as csvfile:
    # 先读取第一行确定表头
    headers = next(csv.reader(csvfile))

    # 验证必要列存在
    required_columns = ['类型', '序号(统计用）', '问题', '原答案', '整合.思考过程', '检查', '检查结论', '整合']
    missing_columns = [col for col in required_columns if col not in headers]

    if missing_columns:
        print(f"CSV文件缺少必要列: {missing_columns}")
        exit(1)

    # 重新定位到文件开头
    csvfile.seek(0)
    reader = csv.DictReader(csvfile)

    for row in reader:
        # 跳过空行
        if all(not value.strip() for value in row.values()):
            continue

        # 解析整合列
        try:
            criteria_data = parse_criteria(row['整合'])
        except Exception as e:
            print(f"解析整合列失败（行号：{reader.line_num}）: {str(e)}")
            continue

        # 构建新的 JSON 对象
        new_obj = {
            "id": f"{row['类型']}-简答题-{row['序号(统计用）']}",
            "question": convert_to_latex(row['问题']),
            **criteria_data,
            "answer": convert_to_latex(row['原答案']),
            "cot": convert_to_latex(row['整合.思考过程']),
            "check": convert_to_latex(row['检查']),
            "result": convert_to_latex(row['检查结论'])
        }

        new_data.append(new_obj)

# 将处理后的数据保存为 JSON 文件
with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
    json.dump(new_data, jsonfile, ensure_ascii=False, indent=4, separators=(',', ': '))

print(f"转换完成，已生成 {len(new_data)} 条数据到 {json_file_path}")