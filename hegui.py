from docx import Document
import re

# 读取.docx文件
doc = Document("C:/Users/admin/Desktop/合规问题.docx")
content = "\n".join([para.text for para in doc.paragraphs])  # 提取所有段落文本

# 分割问题（处理编号和可能的换行/空格）
questions = re.split(r'\n(\d+\.\s)', content)
questions = [q.strip() for q in questions if q.strip()]  # 过滤空字符串

# 重新编号并整理
new_questions = []
for i, q in enumerate(questions, 1):
    # 处理可能残留的编号（例如：如果原问题包含旧编号，这里清除）
    q = re.sub(r'^\d+\.\s*', '', q)  # 移除可能存在的前置编号
    new_questions.append(f"{i}. {q}")

# 输出结果，每个问题占一行
for q in new_questions:
    print(q)

# （可选）如果需要保存到文件
with open("reordered_questions.txt", "w", encoding="utf-8") as f:
    for q in new_questions:
        f.write(q + "\n")