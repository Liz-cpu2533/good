import re
from docx import Document


def count_clauses(file_path):
    doc = Document(file_path)
    clause_pattern = re.compile(
        r'^(?P<制度名称>[^条款]+?)条款\s+(\d+):\s+(?P<条款内容>.*)$',
        re.IGNORECASE | re.DOTALL
    )
    clause_counts = {}

    for para in doc.paragraphs:
        text = para.text.strip()
        match = clause_pattern.match(text)
        if match:
            # 提取制度名称（去除可能的冗余后缀）
            institution = re.sub(r'\s+管理办法|细则|制度|操作规程$', '', match.group(1).strip())
            institution = institution.strip()  # 去除首尾空格

            # 处理特殊情况：如“交易者适当性管理办法条款”可能包含多余空格
            if "交易者适当性管理办" in institution:
                institution = "交易者适当性管理办法"

            clause_number = match.group(2)
            full_title = f"{institution}条款 {clause_number}"

            # 排除非条款内容（如包含“主要条款”的干扰项）
            if "主要条款" in text or "前款所称" in text:
                continue

            # 统计计数
            if full_title in clause_counts:
                clause_counts[full_title] += 1
            else:
                clause_counts[full_title] = 1

    # 整理结果（按制度名称分组）
    result = {}
    for title, count in clause_counts.items():
        # 提取制度名称（去除“条款 X”部分）
        institution = re.match(r'^(.*?)条款', title).group(1)
        result[institution] = result.get(institution, 0) + count

    return result




# 替换为你的实际文件路径
file_path = "C:/Users/admin/Desktop/合规内容提取.docx"
results = count_clauses(file_path)
for clause_type, count in results.items():
    print(f"{clause_type}条款: {count}条")