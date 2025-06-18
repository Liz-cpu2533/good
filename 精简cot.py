import asyncio
import csv
import os
import time
from datetime import datetime
from volcenginesdkarkruntime import AsyncArk


def construct_input(prompt):
    """构建对话格式输入"""
    return [{"role": "user", "content": prompt}]


async def truncate_cot_worker(client, model_id, data):
    """单条数据的cot缩写处理"""
    original_cot = data.get("cot", "")
    if not original_cot:
        return data

    # 生成精简提示词
    prompt = (
        f"请将以下思考过程精简到500字以内，保持关键逻辑和结论完整：\n"
        f"{original_cot}\n\n"
        "精简后的内容："
    )

    try:
        completion = await client.batch_chat.completions.create(
            model=model_id,
            messages=construct_input(prompt),
            max_tokens=512  # 限制模型输出长度
        )
        data["cot"] = completion.choices[0].message.content.strip()
        return data
    except Exception as e:
        print(f"处理失败: {e}")
        return data


async def batch_truncate_cot(model_name, model_id, dataset, concurrent=5):
    """批量处理cot缩写"""
    async with AsyncArk(timeout=3600) as client:
        # 拆分任务队列
        queues = [[] for _ in range(concurrent)]
        for i, data in enumerate(dataset):
            queues[i % concurrent].append(data)

        # 创建异步任务
        tasks = [
            asyncio.create_task(
                asyncio.gather(*[truncate_cot_worker(client, model_id, item) for item in queue])
            )
            for queue in queues
        ]

        # 等待所有任务完成
        results = []
        for task in tqdm(asyncio.as_completed(tasks), total=len(queues), desc="处理批次"):
            results.extend(await task)

    return results


async def process_csv(model_name, model_id, input_path, output_path, concurrent=5):
    # 读取CSV文件
    with open(input_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        dataset = list(reader)

    print(f"读取数据完成，总记录数: {len(dataset)}")

    # 分批处理（每批1000条）
    batch_size = 1000
    total_batches = (len(dataset) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(dataset))
        batch_data = dataset[start_idx:end_idx]

        print(f"处理第 {batch_idx + 1}/{total_batches} 批，处理数量: {len(batch_data)}")
        processed_batch = await batch_truncate_cot(model_name, model_id, batch_data, concurrent)

        # 写入结果
        output_filename = output_path if batch_idx == total_batches - 1 else f"{output_path.split('.csv')[0]}_{batch_idx + 1}.csv"
        with open(output_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
            if batch_idx == 0:
                writer.writeheader()
            writer.writerows(processed_batch)

    print(f"所有批次处理完成，结果保存至: {output_path}")


if __name__ == "__main__":
    # 加载环境变量
    from dotenv import load_dotenv

    load_dotenv()
    os.environ["ARK_API_KEY"] = "f46d245b-7dbe-48e7-ac8a-4c19740cd14a"  # 替换为你的API Key

    # 配置参数
    config = {
        "model_name": "deepseek-v3",
        "model_id": "ep-20250402192514-btf77",  # 替换为你的模型端点ID
        "input_path": "raw_data/clustered_deduplicated_by_type.csv",  # 输入文件路径
        "output_path": "clustered_deduplicated_by_type_trimmed.csv",  # 输出文件路径
        "concurrent": 5  # 并发数，建议根据API限制调整（通常5-10）
    }

    # 执行处理
    start_time = time.perf_counter()
    asyncio.run(process_csv(**config))
    print(f"处理完成，总耗时：{time.perf_counter() - start_time:.2f}秒")