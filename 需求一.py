import asyncio
import pandas as pd
import time
import logging
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Any, Optional
from volcenginesdkarkruntime import AsyncArk

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def construct_input(prompt: str) -> List[Dict[str, str]]:
    """构建符合火山引擎API要求的输入格式"""
    return [{"role": "user", "content": prompt}]

class QuestionAnswerProcessor:
    def __init__(self, api_key: str, model_id: str, concurrent: int = 5):
        self.api_key = api_key
        self.model_id = model_id
        self.concurrent = concurrent
        self.timeout = 3600  # API请求超时时间，单位秒
    
    async def process_csv(self, input_file: str, output_file: str, batch_size: int = 10, test_mode: bool = False):
        """读取CSV文件并处理问题"""
        try:
            df = pd.read_csv(input_file)
            logger.info(f"成功读取CSV文件，共{len(df)}行数据")
            
            # 检查必要的列是否存在
            required_columns = ['唯一ID', '问题', '选项A内容', '选项B内容', '选项C内容', '选项D内容', '答案']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"CSV文件缺少必要的列: {', '.join(missing_columns)}")
            
            # 初始化新列
            df['模型答案'] = None
            df['模型解析'] = None
            
            # 测试模式下只处理部分数据
            if test_mode:
                test_size = min(10, len(df))
                logger.info(f"测试模式：仅处理前{test_size}行数据")
                df = df.head(test_size)
            
            # 分批处理数据
            total_batches = (len(df) + batch_size - 1) // batch_size
            
            async with AsyncArk(timeout=self.timeout) as client:
                for batch_idx in tqdm(range(total_batches), desc="处理批次"):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(df))
                    batch_df = df.iloc[start_idx:end_idx].copy()
                    
                    # 准备批次数据
                    batch_data = []
                    for idx, row in batch_df.iterrows():
                        question = row['问题']
                        options = {
                            'A': row['选项A内容'],
                            'B': row['选项B内容'],
                            'C': row['选项C内容'],
                            'D': row['选项D内容']
                        }
                        batch_data.append({
                            'index': idx,
                            'question': question,
                            'options': options
                        })
                    
                    # 异步处理批次数据
                    processed_data = await self._process_batch_async(client, batch_data)
                    
                    # 更新DataFrame
                    for data in processed_data:
                        idx = data['index']
                        df.loc[idx, '模型答案'] = data.get('answer', '')
                        df.loc[idx, '模型解析'] = data.get('explanation', '')
                    
                    # 每处理一个批次就保存一次
                    df.to_csv(output_file, index=False)
                    logger.info(f"已保存批次 {batch_idx+1}/{total_batches} 到 {output_file}")
            
            logger.info(f"处理完成，结果已保存到 {output_file}")
            return df
            
        except Exception as e:
            logger.error(f"处理CSV文件时出错: {str(e)}")
            raise
    
    async def _process_batch_async(self, client: AsyncArk, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """异步处理一个批次的问题"""
        # 创建协程列表
        tasks = []
        for data in batch_data:
            tasks.append(self._process_single_async(client, data))
        
        # 使用asyncio.gather并发执行，并添加进度显示
        results = []
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="处理进度", leave=False):
            results.append(await future)
        
        return results
    
    async def _process_single_async(self, client: AsyncArk, data: Dict[str, Any]) -> Dict[str, Any]:
        """异步处理单个问题"""
        index = data['index']
        question = data['question']
        options = data['options']
        
        # 构建提示词
        prompt = self._build_prompt(question, options)
        
        # 调用API
        retry_times = 3
        for attempt in range(retry_times):
            try:
                # 调用火山引擎SDK的异步接口
                completion = await client.batch_chat.completions.create(
                    model=self.model_id,
                    messages=construct_input(prompt),
                    max_tokens=512,
                    temperature=0.2
                )
                
                # 解析响应
                response_text = completion.choices[0].message.content
                parsed_response = self._parse_response(response_text)
                
                return {
                    'index': index,
                    'answer': parsed_response['answer'],
                    'explanation': parsed_response['explanation']
                }
            except Exception as e:
                if attempt == retry_times - 1:
                    logger.error(f"处理索引 {index} 失败，已达到最大重试次数: {str(e)}")
                    return {
                        'index': index,
                        'answer': '',
                        'explanation': f'调用API失败: {str(e)}'
                    }
                else:
                    logger.warning(f"处理索引 {index} 失败，尝试重试 ({attempt+1}/{retry_times}): {str(e)}")
                    await asyncio.sleep(2)  # 重试前等待2秒
    
    def _build_prompt(self, question: str, options: Dict[str, str]) -> str:
        """构建向模型提交的提示词"""
        options_text = "\n".join([f"{key}. {value}" for key, value in options.items()])
        
        prompt = f"""
问题: {question}
选项:
{options_text}

请回答正确选项（只能是A/B/C/D，可以是单选或多选），并给出理由。格式如下：
答案：A
理由：选项A正确的原因是...
"""
        return prompt.strip()
    
    def _parse_response(self, response: str) -> Dict[str, str]:
        """解析模型返回的结果"""
        answer = ""
        explanation = ""
        
        lines = response.strip().split('\n')
        for i, line in enumerate(lines):
            if line.lower().startswith('答案：') or line.lower().startswith('答案:'):
                answer_part = line.split('：', 1)[-1].split(':', 1)[-1].strip()
                # 提取A/B/C/D中的选项
                valid_options = [char for char in answer_part if char in 'ABCD']
                answer = '/'.join(valid_options)
                
                # 剩余行作为理由
                if i + 1 < len(lines):
                    explanation = '\n'.join(lines[i+1:]).strip()
                break
        
        if not answer:
            # 如果没有找到答案行，尝试从第一行提取
            first_line = lines[0].strip()
            valid_options = [char for char in first_line if char in 'ABCD']
            if valid_options:
                answer = '/'.join(valid_options)
                explanation = response.replace(first_line, '').strip()
            else:
                # 如果完全无法提取答案，返回原始响应作为解析
                explanation = response
        
        return {
            'answer': answer,
            'explanation': explanation
        }

if __name__ == "__main__":
    import os
    # 直接设置API Key
    os.environ["ARK_API_KEY"] = "xxx"
    
    # 配置参数
    config = {
        "api_key": os.environ["ARK_API_KEY"],
        "model_id": "xxx",
        "input_file": "raw_data/需求一测试.csv",  # 输入CSV文件路径
        "output_file": "output.csv",  # 输出CSV文件路径
        "batch_size": 5,  # 每批处理的行数
        "concurrent": 5,  # 并发请求数
        "test_mode": True  # 是否启用测试模式
    }
    
    # 创建处理器
    processor = QuestionAnswerProcessor(config["api_key"], config["model_id"], config["concurrent"])
    
    # 执行处理
    start_time = time.perf_counter()
    asyncio.run(processor.process_csv(
        config["input_file"],
        config["output_file"],
        config["batch_size"],
        config["test_mode"]
    ))
    print(f"处理完成，总耗时：{time.perf_counter() - start_time:.2f}秒")    
