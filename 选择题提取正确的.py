import re
import csv
import os


def keep_alphabets(input_string):
    result = "".join(filter(str.isalpha, input_string))
    return result


def extract_choice(response: str) -> str:
    """
    Always return a choice, even cannot match by regex,
    to ensure fair comparison to other models.
    """
    if response == "":
        return ""
    # 多选
    pattern = r"[ABCDE]+(?:[、,][ABCDE]+)*"
    m = re.search(pattern, response, re.M)
    if m:
        answer = m.group()
        answer = keep_alphabets(answer)
        return answer
    choices = ["A", "B", "C", "D", "E"]
    # 1. Single match
    patterns = [
        (r"答案(选项)?(是|为)：? ?([ABCDE])", 3),
        (r"答案(是|为)选项 ?([ABCDE])", 2),
        (r"故?选择?：? ?([ABCDE])", 1),
        (r"([ABCDE]) ?选?项(是|为)?正确", 1),
        (r"正确的?选项(是|为) ?([ABCDE])", 2),
        (r"答案(应该)?(是|为)([ABCDE])", 3),
        (r"选项 ?([ABCDE]) ?(是|为)?正确", 1),
        (r"选择答案 ?([ABCDE])", 1),
        (r"答案?：?([ABCDE])", 1),
        (r"([ABCDE])(选?项)?是?符合题意", 1),
        (r"答案选项：? ?([ABCDE])", 1),  # chatglm
        (r"答案(选项)?为(.*?)([ABCDE])", 3),  # chatgpt
        (r"选项([ABCDE])是最恰当的", 1),
        (r"选项([ABCDE]).*最恰当", 1),
        (r"选项([ABCDE]).*最能恰当", 1),
        (r"选项([ABCDE]).*最能", 1),
        (r"最恰当.*是选项([ABCDE])", 1),
        (r"correct answer is.*([ABCDE])", 1),
    ]
    for pattern, idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            answer = m.group(idx)
            assert answer in choices
            return answer

    # 2. Recursive match
    patterns = [
        (r"([ABCDE])(.*?)当选", 1),
        (r"([ABCDE])(.*?)正确", 1),
    ]
    for pattern, idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            while m:
                answer = m.group(idx)
                m = re.search(pattern, m.group(0)[1:], re.M)
            assert answer in choices
            return answer

    # 3. Weak single match
    patterns = [
        (r"[^不]是：? ?([ABCDE])", 1),
    ]
    for pattern, idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            answer = m.group(idx)
            assert answer in choices
            return answer

    # 4. Check the only mentioned choices
    pattern = r"^[^ABCDE]*([ABCDE])[^ABCDE]*$"
    m = re.match(pattern, response)
    if m:
        answer = m.group(1)
        assert answer in choices
        return answer

    # 5. Check the only mentioned choices in the start of the sentence
    m = re.match(pattern, response[:4])
    if m:
        answer = m.group(1)
        assert answer in choices
        return answer

    m = re.match(pattern, response[:2])
    if m:
        answer = m.group(1)
        assert answer in choices
        return answer

    return ""


def process_csv_with_extracted_answers(input_file):
    """读取CSV文件，提取答案，并将结果写入新列"""
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        fieldnames = reader.fieldnames

        # 添加新列名
        if '输出结果' not in fieldnames:
            fieldnames.append('输出结果')

    # 处理并添加结果
    for row in rows:
        answer_text = row.get('answer', '')
        extracted = extract_choice(answer_text)
        row['输出结果'] = extracted

    # 写入带新列的文件
    temp_file = input_file + '.temp'
    with open(temp_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 替换原文件
    os.replace(temp_file, input_file)
    print(f"已成功处理 {len(rows)} 条记录，并添加'输出结果'列到 {input_file}")


if __name__ == "__main__":
    csv_file_path = "C:/Users/admin/Desktop/new_cot_answers_016.csv"  # 替换为实际的csv文件路径
    process_csv_with_extracted_answers(csv_file_path)

