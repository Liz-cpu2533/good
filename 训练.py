import asyncio
import csv
import os
from volcenginesdkarkruntime import AsyncArk


# --------------------- 修复1：构造符合考试要求的提示词 ---------------------
def construct_question_prompt(row):
    """根据CSV行数据生成合规的考试提示词"""
    prompt = f"""你现在正在参加期货行业的从业资格考试，题目和解析如下：
# 问题
{row['问题']}

# 选项
A: {row['选项 A']}
B: {row['选项 B']}
C: {row['选项 C']}
D: {row['选项 D']}

# 已知解析
{row.get('解析', '无')}

请按以下步骤回答：
1. 先进行详细思考分析，但不要提前透露答案
2. 最后用以下格式给出答案：
答案：选 [选项] 
分析：[详细分析]"""

    return [{
        "role": "user",
        "content": prompt
    }]


# --------------------- 修复2：增强结果解析功能 ---------------------
def parse_response(response_text):
    """解析模型返回的答案"""
    try:
        # 提取答案部分
        answer_part = response_text.split("答案：选 ")[1].split("\n")[0].strip()
        analysis_part = response_text.split("分析：")[1].strip()

        # 处理多选情况
        if "选" in answer_part:
            return answer_part.replace("选 ", ""), analysis_part
        return answer_part, analysis_part
    except:
        return "解析失败", response_text


# --------------------- 修改后的worker函数 ---------------------
async def ark_worker(model_name, client, model_id, worker_id, task_queue):
    results = []
    for data in task_queue:
        try:
            # 构造合规提示词
            messages = construct_question_prompt(data)

            # 调用模型
            completion = await client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.1  # 降低随机性保证答案稳定
            )

            # 解析结果
            full_response = completion.choices[0].message.content
            answer, analysis = parse_response(full_response)

            # 记录思考过程（仅限r1模型）
            if "deepseek-r1" in model_name:
                data["thinking"] = getattr(
                    completion.choices[0].message,
                    "reasoning_content",
                    "无思考过程"
                )

            # 保存结果
            data.update({
                "模型答案": answer,
                "分析": analysis,
                "完整响应": full_response
            })

        except Exception as e:
            data.update({
                "模型答案": f"错误：{str(e)[:50]}",
                "分析": "",
                "thinking": ""
            })

        results.append(data)
    return results


# --------------------- CSV处理增强版 ---------------------
async def process_exam_questions(
        input_csv="选择题训练集.csv",
        output_csv="结果.csv",
        model_name="deepseek-r1",
        model_id="ep-bi-20250424171837-4z2xm",
        concurrent=5
):
    # 读取CSV文件（处理各种编码问题）
    try:
        with open(input_csv, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            dataset = list(reader)
    except:
        with open(input_csv, encoding='gbk') as f:
            reader = csv.DictReader(f)
            dataset = list(reader)

    # 分发任务
    client = AsyncArk(api_key=os.getenv("ARK_API_KEY"))

    # 分割任务队列
    chunk_size = len(dataset) // concurrent + 1
    queues = [dataset[i * chunk_size:(i + 1) * chunk_size] for i in range(concurrent)]

    # 运行任务
    tasks = [ark_worker(model_name, client, model_id, i, q) for i, q in enumerate(queues)]
    results = await asyncio.gather(*tasks)

    # 合并结果
    all_results = []
    for r in results:
        all_results.extend(r)

    # 写入结果（保留原始字段）
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        fieldnames = reader.fieldnames + ["模型答案", "分析", "thinking"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)


if __name__ == "__main__":
    # 配置参数
    config = {
        "input_csv": "选择题训练集.csv",
        "output_csv": "带答案的训练集.csv",
        "model_name": "deepseek-r1",
        "model_id": "ep-bi-20250424171837-4z2xm",
        "concurrent": 5
    }

    # 执行处理
    asyncio.run(process_exam_questions(**config))