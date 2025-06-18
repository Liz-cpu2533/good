


import asyncio
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


def construct_input(input_list):
    api_input = list()
    for i, item in enumerate(input_list):
        if i % 2 == 0:
            api_input.append({
                "role": "user",
                "content": item
            })
        else:
            api_input.append({
                "role": "assistant",
                "content": item
            })
    return api_input

async def ark_worker(model_name, client, model_id, worker_id, task_queue, is_force_think=False, output_key="output", prompt_key="prompt"):
    result = []
    for i, data in enumerate(task_queue):
        print(f"Worker {worker_id}: task {i+1}/{len(task_queue)} is running...")
        final_prompt = data[prompt_key]
        if is_force_think:
            final_prompt = '任何输出都要有思考过程，输出内容必须以 "<think>\n\n嗯" 开头。仔细揣摩用户意图，在思考过程之后，提供逻辑清晰且内容完整的回答\n\n' + final_prompt
        try:
            completion = await client.batch_chat.completions.create(
                model=model_id,
                messages=construct_input([final_prompt]),
            )
            if "deepseek-r1" in model_name:
                cot = completion.choices[0].message.reasoning_content
                if cot is None:
                    cot = ""
                data["cot"] = cot
            answer = completion.choices[0].message.content
            data[output_key] = answer
            print(f"Worker {worker_id} task {i+1}/{len(task_queue)} is completed.")
            result.append(data)
        except Exception as e:
            print(f"Worker {worker_id} task {i+1}/{len(task_queue)} failed with error: {e}")
    print(f"Worker {worker_id} {len(task_queue)} tasks are completed.")
    return result


async def ark_batch_query(model_name, model_id, dataset, concurrent=2, is_force_think=False, output_key="output", prompt_key="prompt"):
    """
    输入： dataset是一个数组，每个元素是一个dict，必须包含prompt字段
    输出： 返回和输入相同长度的数组，大模型回复在content字段，如果是r1，会有额外的cot字段存思考过程
    """
    start = datetime.now()
    queues = []
    for i in range(concurrent):
        queues.append([])
    for i, data in enumerate(dataset):
        queues[i % concurrent].append(data)

    # 创建任务列表
    async with AsyncArk(timeout=3600*24) as client:
        tasks = [ark_worker(model_name, client, model_id, i, queues[i], is_force_think, output_key, prompt_key) for i in range(concurrent)]
        # 等待所有任务完成
        result = await asyncio.gather(*tasks)
    end = datetime.now()
    print(f"Total time: {end - start}, Total task: {len(dataset)}")
    all_results = []
    for r in result:
        all_results.extend(r)
    return all_results


if __name__ == "__main__":
    import os
    os.environ["ARK_API_KEY"] = "f46d245b-7dbe-48e7-ac8a-4c19740cd14a"
    model_name = "deepseek-r1"
    dataset = [
        {'prompt': "介绍一下 Python 中的异步编程"},
        {'prompt': "分析一下大模型对A股有哪些利好"}
    ]
    # 并发数量
    concurrent = 2
    start = time.perf_counter()
    if model_name == "deepseek-v3":
        v3_id = "ep-bi-20250418153651-h8dqv"
        dataset = asyncio.run(ark_batch_query(model_name, v3_id, dataset, concurrent))
    elif model_name == "deepseek-r1":
        r1_id = "ep-bi-20250424171837-4z2xm"
        dataset = asyncio.run(ark_batch_query(model_name, r1_id, dataset, concurrent))
    else:
        print("还不支持的模型")
        exit(1)
    end = time.perf_counter()
    print("async总耗时: ", end - start)
    print(dataset)


