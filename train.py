import asyncio
import csv
import os
import time
from datetime import datetime
from volcenginesdkarkruntime import AsyncArk


# --------------------- 修复1：正确构造单轮对话 ---------------------
def construct_input(question):
    """将单个问题包装成对话格式"""
    return [{
        "role": "user",
        "content": f"请先仔细思考，然后给出专业解答。问题：{question}"
    }]


# --------------------- 修复2：增强错误处理 ---------------------
async def ark_worker(model_name, client, model_id, worker_id, task_queue):
    results = []
    for i, data in enumerate(task_queue):
        try:
            # 添加重试机制（简易版）
            for attempt in range(3):
                try:
                    completion = await client.chat.completions.create(  # 修正API调用
                        model=model_id,
                        messages=construct_input(data["question"]),
                        temperature=0.3,
                        request_timeout=30
                    )
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(1)

            # 提取回答内容
            response = completion.choices[0].message
            data["answer"] = response.content

            # 提取思考过程（仅限deepseek-r1）
            if "deepseek-r1" in model_name:
                data["thinking"] = getattr(response, "reasoning_content", "")

        except Exception as e:
            print(f"Worker {worker_id} 任务失败: {str(e)[:50]}...")
            data.update({"answer": "", "thinking": f"错误: {type(e).__name__}"})

        results.append(data)
    return results


# --------------------- 新增功能：CSV处理 ---------------------
async def process_csv(model_name, model_id, input_path, output_path, concurrent=5):
    # 读取CSV文件
    with open('C:/Users/admin/Desktop/选择题训练集.csv', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        dataset = list(reader)

    # 分发任务
    client = AsyncArk(api_key=os.getenv("ARK_API_KEY"))  # 从环境变量读取密钥
    results = await ark_batch_query(model_name, model_id, dataset, concurrent)

    # 写入结果
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = reader.fieldnames + ["answer", "thinking"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


# --------------------- 原有函数优化 ---------------------
async def ark_batch_query(model_name, model_id, dataset, concurrent=5):
    queues = [[] for _ in range(concurrent)]
    for i, data in enumerate(dataset):
        queues[i % concurrent].append(data)

    client = AsyncArk(api_key=os.getenv("ARK_API_KEY"))
    tasks = [ark_worker(model_name, client, model_id, i, q) for i, q in enumerate(queues)]

    return await asyncio.gather(*tasks)


if __name__ == "__main__":
    # --------------------- 安全配置 ---------------------
    from dotenv import load_dotenv

    load_dotenv()  # 从.env文件加载配置

    # --------------------- 参数配置 ---------------------
    config = {
        "model_name": "deepseek-r1",
        "model_id": "ep-bi-20250424171837-4z2xm",  # 替换你的真实端点ID
        "input_csv": "questions.csv",
        "output_csv": "answers.csv",
        "concurrent": 5  # 建议5-10之间
    }

    # --------------------- 执行处理 ---------------------
    start_time = time.perf_counter()
    asyncio.run(process_csv(**config))
    print(f"处理完成，总耗时：{time.perf_counter() - start_time:.2f}秒")