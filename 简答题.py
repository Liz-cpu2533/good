import pandas as pd
import json


def convert_to_latex(text):
    return text


def process_csv_to_json(csv_file_path, json_file_path):
    try:
        df = pd.read_csv(csv_file_path)
        json_data = []

        for index, row in df.iterrows():
            criterium_text = row['要点.输出结果']
            if pd.notna(criterium_text):
                # 去除文本两端可能存在的多余字符，如换行符等
                criterium_text = criterium_text.strip()
                # 检查是否以json格式的大括号开头和结尾，若不是则尝试修正
                if not criterium_text.startswith('{') or not criterium_text.endswith('}'):
                    if criterium_text.count('{') == 1 and criterium_text.count('}') == 1:
                        start_index = criterium_text.find('{')
                        end_index = criterium_text.rfind('}')
                        criterium_text = criterium_text[start_index: end_index + 1]
                    else:
                        print(f"第{index}行要点.输出结果格式严重错误，无法修正，内容为: {criterium_text}")
                        continue
                try:
                    criterium_json = json.loads(criterium_text)
                    criterium_dict = {}
                    for i in range(1, 6):
                        criterium_key = f"criterium{i}"
                        if criterium_key in criterium_json:
                            criterium_dict[criterium_key] = {
                                "content": criterium_json[criterium_key]["content"],
                                "score": criterium_json[criterium_key]["score"]
                            }
                    item = {
                        "id": f"{row['类型']}-简答题-{row['序号(统计用）']}",
                        "uid": row['唯一ID'],
                        "question": convert_to_latex(row['问题']),
                        "answer": convert_to_latex(row['原答案']),
                        "cot": convert_to_latex(row['思考过程.思考过程'])
                    }
                    item.update(criterium_dict)
                    json_data.append(item)
                except json.JSONDecodeError as e:
                    print(f"第{index}行要点.输出结果解析失败，错误信息: {e}，内容为: {criterium_text}")
            else:
                continue

        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        print(f"文件处理完成，已保存为{json_file_path}")

    except FileNotFoundError:
        print(f"错误：未找到指定的CSV文件，请检查路径: {csv_file_path}")
    except Exception as e:
        print(f"发生未知错误: {e}")


if __name__ == "__main__":
    csv_file_path = 'C:/Users/admin/Desktop/简答题.csv'
    json_file_path = 'C:/Users/admin/Desktop/简答题.json'
    process_csv_to_json(csv_file_path, json_file_path)