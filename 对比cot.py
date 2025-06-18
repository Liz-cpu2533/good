import asyncio
import csv
import os
import time
from datetime import datetime
from shutil import copy2
from volcenginesdkarkruntime import AsyncArk
from tqdm import tqdm

def construct_contrast_prompt(original_cot, short_cot):
    """构建对比分析的提示词"""
    return (
        f"请对比以下两份思考过程：\n"
        f"原始版本（cot）：{original_cot}\n"
        f"精简版本（short_cot）：{short_cot}\n\n"
        "判断精简版本是否包含了原始版本的所有重要信息，是否有遗漏或关键信息缺失。"
        "请严格按照以下格式输出：\n"
        "结论：[一致/不一致]\n"
        "理由：[简要说明差异点]"
    )

async def check_coverage_worker(client, model_id, data):
    """单条数据的覆盖检查"""
    original_cot = data.get("cot", "")
    short_cot = data.get("short_cot", "")
    
    if not original_cot or not short_cot:
        data["result"] = "数据缺失"
        data["reason"] = "原始或精简COT为空"
        return data
    
    prompt = construct_contrast_prompt(original_cot, short_cot)
    
    try:
        # 调用DeepSeek-V3模型进行对比分析
        completion = await client.batch_chat.completions.create(
            model=model_id,
            messages=[[{"role": "user", "content": prompt}]],
            max_tokens=512
        )
        
        response = completion.choices[0].message.content.strip()
        
        # 解析模型输出
        conclusion = "不一致"
        reason = "未解析到有效结论"
        
        for line in response.split("\n"):
            if "结论：" in line:
                conclusion = line.split("：")[-1].strip()
            if "理由：" in line:
                reason = line.split("：")[-1].strip()
        
        data["result"] = conclusion
        data["reason"] = reason
        return data
    
    except Exception as e:
        print(f"处理失败 (ID: {data.get('id', '未知')}): {e}")
        data["result"] = "处理失败"
        data["reason"] = str(e)
        return data

async def batch_check_coverage(model_id, dataset, concurrent=5):
    """批量检查覆盖情况"""
    async with AsyncArk(timeout=3600) as client:
        tasks = []
        for data in dataset:
            tasks.append(check_coverage_worker(client, model_id, data))
        
        results = []
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="对比进度"):
            results.append(await future)
    
    return results

async def process_csv(model_id, input_path, concurrent=5):
    # 读取CSV文件
    with open(input_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        dataset = list(reader)
    
    print(f"读取数据完成，总记录数: {len(dataset)}")
    
    # 创建备份文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{os.path.splitext(input_path)[0]}_{timestamp}.bak"
    copy2(input_path, backup_path)
    print(f"已创建备份文件: {backup_path}")
    
    # 仅处理前10条数据进行测试
    test_dataset = dataset[:10]
    print(f"将仅处理前10条数据进行测试...")
    
    # 处理数据
    processed_data = await batch_check_coverage(model_id, test_dataset, concurrent)
    
    # 更新列名，添加result和reason
    fieldnames = reader.fieldnames + ["result", "reason"]
    
    # 写入结果回原文件
    with open(input_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed_data)
    
    print(f"测试完成，结果已保存至: {input_path}")
    print(f"注意：原始数据已备份至 {backup_path}")

if __name__ == "__main__":
    # 设置API Key
    os.environ["ARK_API_KEY"] = "f46d245b-7dbe-48e7-ac8a-4c19740cd14a"
    
    # 配置参数
    config = {
        "model_id": "deepseek-v3",  # 使用DeepSeek-V3模型
        "input_path": "raw_data/clustered_deduplicated_by_type.csv",  # 输入文件路径
        "concurrent": 5  # 并发请求数
    }

    # 执行处理
    start_time = time.perf_counter()
    asyncio.run(process_csv(**config))
    print(f"处理完成，总耗时：{time.perf_counter() - start_time:.2f}秒")