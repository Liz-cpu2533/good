import asyncio
import csv
import os
import time
from datetime import datetime
from volcenginesdkarkruntime import AsyncArk

# Authentication
# 1.If you authorize your endpoint using an API key, you can set your api key to environment variable "ARK_API_KEY"
# or specify api key by Ark(api_key="${YOUR_API_KEY}").
# Note: If you use an API key, this API key will not be refreshed.
# To prevent the API from expiring and failing after some time, choose an API key with no expiration date.

# 2.If you authorize your endpoint with Volcengine Identity and Access Management（IAM), set your api key to environment variable "VOLC_ACCESSKEY", "VOLC_SECRETKEY"
# or specify ak&sk by Ark(ak="${YOUR_AK}", sk="${YOUR_SK}").
# To get your ak&sk, please refer to this document(https://www.volcengine.com/docs/6291/65568)
# For more information，please check this document（https://www.volcengine.com/docs/82379/1263279）

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
        
        # 假设 data 是包含问题、选项和答案的字典
        if data["选项A内容"] == "正确":
           final_prompt = f'你现在正在参加期货行业的从业资格考试,你已经知道解析：\n{data["解析"]}\n请你回答问题：{data["问题"]}\n选项A:{data["选项A内容"]}\n选项B:{data["选项B内容"]}\n答案:{data["答案"]}\n思考过程不要提前透露已知答案,输出结果要说出正确答案,比如:选B，输出结果不要只说答案也要有一定的分析'
        elif data["选项C内容"] in ["消极", "不确定"] and data['选项D内容'].strip()=="":
            # if  data['选项D内容'].strip()=="":
            #     a = 1
            final_prompt = f'你现在正在参加期货行业的从业资格考试,你已经知道解析：\n{data["解析"]}\n请你回答问题：{data["问题"]}\n选项A:{data["选项A内容"]}\n选项B:{data["选项B内容"]}\n选项C:{data["选项C内容"]}\n答案:{data["答案"]}\n思考过程不要提前透露已知答案,输出结果要说出正确答案,比如:选B，输出结果不要只说答案也要有一定的分析'
        else:
            # if data["选项C内容"] in ["消极", "不确定"]:
            #     a = 1
            final_prompt = f'你现在正在参加期货行业的从业资格考试,你已经知道解析：\n{data["解析"]}\n请你回答问题：{data["问题"]}\n选项A:{data["选项A内容"]}\n选项B:{data["选项B内容"]}\n选项C:{data["选项C内容"]}\n选项D:{data["选项D内容"]}\n答案:{data["答案"]}\n思考过程不要提前透露已知答案,输出结果要说出正确答案,比如:选B,或者多选,选AB，输出结果不要只说答案也要有一定的分析'
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
                data["cot"] = cot
            answer = completion.choices[0].message.content
            data["answer"] = answer
            results.append(data)
            print(f"Worker {worker_id} task {i} is completed.")
        except Exception as e:
            print(f"Worker {worker_id} task {i} failed with error: {e}")
            data.update({"answer": "", "cot": ""})
            results.append(data)
        
    print(f"Worker {worker_id} is completed.")
    return results

async def ark_batch_query(model_name, model_id, dataset, concurrent=2, is_force_think=False):
    """
    输入： dataset是一个数组，每个元素是一个dict，必须包含prompt字段
    输出： 返回和输入相同长度的数组，大模型回复在content字段，如果是r1，会有额外的cot字段存思考过程
    """
    start = datetime.now()
    queues = [[] for _ in range(concurrent)]
    for i, data in enumerate(dataset):
        queues[i % concurrent].append(data)

    # 创建任务列表
    async with AsyncArk(timeout=3600*24) as client:
        tasks = [ark_worker(model_name, client, model_id, i, queues[i], is_force_think) for i in range(concurrent)]
        # 等待所有任务完成
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

        dataset = dataset[:10]

    import random
    random.seed(2025)
    random.shuffle(dataset)

    batchsize = 1000
    for i in range(len(dataset)//batchsize+1):
        ######## 跑最后一个 
        if i < len(dataset)//batchsize:
            continue

        
        ################
        # 分发任务
        results = await ark_batch_query(model_name, model_id, dataset[i*batchsize:(i+1)*batchsize], concurrent, is_force_think)

        # 写入结果
        with open(output_path.replace('.csv', f"_{i:03d}.csv"), 'w', newline='', encoding='utf-8') as f:
            fieldnames = reader.fieldnames + ["answer", "cot"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

if __name__ == "__main__":
    # --------------------- 安全配置 ---------------------
    from dotenv import load_dotenv

    load_dotenv()  # 从.env文件加载配置
    os.environ["ARK_API_KEY"] = "f46d245b-7dbe-48e7-ac8a-4c19740cd14a"

    # --------------------- 参数配置 ---------------------
    config = {
        "model_name": "deepseek-r1",
        "model_id": "ep-bi-20250424171837-4z2xm",  # 替换你的真实端点ID
        "input_path": "raw_data/clustered_deduplicated_by_type.csv",
        "output_path": "clustered_deduplicated_by_type_answers.csv",
        "concurrent": 50,  # 建议5-10之间 100
        "is_force_think": True
    }

    # --------------------- 执行处理 ---------------------
    start_time = time.perf_counter()
    asyncio.run(process_csv(**config))
    print(f"处理完成，总耗时：{time.perf_counter() - start_time:.2f}秒")