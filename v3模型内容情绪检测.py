import asyncio
import csv
import os
import time
from datetime import datetime
from volcenginesdkarkruntime import AsyncArk


def construct_input(question):
    """将单个问题包装成对话格式"""
    return [{
        "role": "user",
        "content": question
    }]


async def ark_worker(model_name, client, model_id, worker_id, task_queue, is_force_think=False):
    print(f"Worker {worker_id} is starting.")
    results = []
    for i, data in enumerate(task_queue):
        print(f"Worker {worker_id} task {i} is running.")

        # 只处理类型为"情绪检测-内容情绪检测"的行
        if data.get("类型") != "情绪检测-内容情绪检测":
            data.update({"新答案": "", "新解析": ""})
            results.append(data)
            continue

        # 构造prompt
        final_prompt = f'你是一个内容情绪识别助手，请你分析给定内容的情绪并判断属于【积极/中性/消极】的哪一类，请给出正确选项:\n{data["问题"]}\n选项A:{data["选项A内容"]}\n选项B:{data["选项B内容"]}\n选项C:{data["选项C内容"]}\n给出思考过程,输出结果要说出正确答案,比如:选B，输出结果不要只说答案也要有一定的分析'

        if is_force_think:
            final_prompt = '任何输出都要有思考过程，输出内容必须以 "<think>\n\n嗯" 开头。仔细揣摩用户意图，在思考过程之后，提供逻辑清晰且内容完整的回答\n\n' + final_prompt

        try:
            completion = await client.batch_chat.completions.create(
                model=model_id,
                messages=construct_input(final_prompt),
            )

            if "deepseek-r1" in model_name:
                cot = completion.choices[0].message.reasoning_content
                if cot is None:
                    cot = ""
                data["新解析"] = cot
            answer = completion.choices[0].message.content
            data["新答案"] = answer
            results.append(data)
            print(f"Worker {worker_id} task {i} is completed.")
        except Exception as e:
            print(f"Worker {worker_id} task {i} failed with error: {e}")
            data.update({"新答案": "", "新解析": ""})
            results.append(data)

    print(f"Worker {worker_id} is completed.")
    return results


async def ark_batch_query(model_name, model_id, dataset, concurrent=2, is_force_think=False):
    start = datetime.now()
    queues = [[] for _ in range(concurrent)]
    for i, data in enumerate(dataset):
        queues[i % concurrent].append(data)

    async with AsyncArk(timeout=3600) as client:
        tasks = [ark_worker(model_name, client, model_id, i, queues[i], is_force_think) for i in range(concurrent)]
        result = await asyncio.gather(*tasks)

    end = datetime.now()
    print(f"Total time: {end - start}, Total task: {len(dataset)}")
    all_results = []
    for r in result:
        all_results.extend(r)
    return all_results


async def process_csv(model_name, model_id, input_path, output_path, concurrent=5, is_force_think=False):
    # 读取CSV文件
    with open(input_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        dataset = list(reader)
        fieldnames = reader.fieldnames + ["新答案", "新解析"]  # 新增两列

    # 处理数据并保持顺序
    batchsize = 1000
    total_batches = (len(dataset) // batchsize) + 1
    all_results = []

    for i in range(total_batches):
        batch_start = i * batchsize
        batch_end = (i + 1) * batchsize
        current_batch = dataset[batch_start:batch_end]

        # 分发任务
        results = await ark_batch_query(model_name, model_id, current_batch, concurrent, is_force_think)
        all_results.extend(results)

        # 实时保存进度
        temp_output_path = output_path.replace('.csv', f'_temp_{i:03d}.csv')
        with open(temp_output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    # 合并所有批次结果到最终文件
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # 删除临时文件
    for i in range(total_batches):
        temp_output_path = output_path.replace('.csv', f'_temp_{i:03d}.csv')
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)


if __name__ == "__main__":
    # --------------------- 安全配置 ---------------------
    from dotenv import load_dotenv

    load_dotenv()  # 从.env文件加载配置
    os.environ["ARK_API_KEY"] = "f46d245b-7dbe-48e7-ac8a-4c19740cd14a"

    # --------------------- 参数配置 ---------------------
    config = {
        "model_name": "deepseek-v3",
        "model_id": "ep-bi-20250418153651-h8dqv",  # 替换你的真实端点ID
        "input_path": "C:/Users/admin/Desktop/去重后的选择题训练集.csv",
        "output_path": "修改内容情绪检测后的选择题训练集.csv",
        "concurrent": 50,  # 建议5-10之间 100
        "is_force_think": True
    }

    # --------------------- 执行处理 ---------------------
    start_time = time.perf_counter()
    asyncio.run(process_csv(**config))
    print(f"处理完成，总耗时：{time.perf_counter() - start_time:.2f}秒")