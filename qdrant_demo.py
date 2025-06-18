import pandas as pd
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Qdrant配置
QDRANT_PATH = "local_qdrant2"  # 本地存储路径
COLLECTION_NAME = "question_embeddings"  # 向量集合名称
VECTOR_DIM = 2560  # 向量维度，根据您的模型调整

# OpenAI配置
OPENAI_API_KEY = "f46d245b-7dbe-48e7-ac8a-4c19740cd14a"
OPENAI_MODEL_ID = "ep-20250526165855-tdx4s"
OPENAI_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

def generate_embedding(text):
    """生成文本的向量表示"""
    if not text or not isinstance(text, str):
        return [0.0] * VECTOR_DIM  # 处理空文本或非字符串
    
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )
    
    try:
        logger.debug(f"正在生成嵌入向量: {text[:50]}...")
        resp = client.embeddings.create(
            model=OPENAI_MODEL_ID,
            input=[text],
            encoding_format="float"
        )
        return resp.data[0].embedding
    except Exception as e:
        logger.error(f"生成嵌入向量失败: {str(e)}")
        return [0.0] * VECTOR_DIM  # 出错时返回零向量

def create_qdrant_collection(client):
    """创建Qdrant集合"""
    if client.get_collection(collection_name=COLLECTION_NAME):
        logger.info(f"集合 '{COLLECTION_NAME}' 已存在，跳过创建")
        return
    
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
    )
    logger.info(f"成功创建集合 '{COLLECTION_NAME}'")

def upload_to_qdrant(df):
    """将CSV数据上传到Qdrant"""
    client = QdrantClient(path=QDRANT_PATH)
    create_qdrant_collection(client)
    
    batch_size = 100  # 每次上传的记录数
    total_records = len(df)
    points = []
    
    logger.info(f"开始上传数据到Qdrant，共 {total_records} 条记录")
    
    for idx, row in tqdm(df.iterrows(), total=total_records, desc="上传进度"):
        unique_id = row['唯一ID']
        question = row['问题']
        embedding = generate_embedding(question)
        
        points.append(
            PointStruct(
                id=unique_id,  # 使用CSV中的唯一ID作为点ID
                vector=embedding,
                payload={"问题": question, "唯一ID": unique_id}
            )
        )
        
        # 每batch_size条记录上传一次
        if len(points) >= batch_size or idx == total_records - 1:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            points = []  # 清空批次
    
    logger.info(f"数据上传完成，集合 '{COLLECTION_NAME}' 现在包含 {total_records} 个点")
    return client

def find_similar_questions(df):
    """查找每个问题的最相似问题"""
    client = QdrantClient(path=QDRANT_PATH)
    results = []
    
    logger.info("开始查找相似问题...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="查找进度"):
        question = row['问题']
        embedding = generate_embedding(question)
        
        # 查询最相似的问题（排除自身）
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedding,
            limit=2  # 获取前2个结果，第一个可能是自身
        )
        
        # 处理查询结果
        if not search_result:
            results.append((None, 0.0, None))
            continue
            
        # 获取最相似的问题（排除自身）
        best_match = None
        for hit in search_result:
            if hit.payload['唯一ID'] != row['唯一ID']:  # 确保不是自身
                best_match = hit
                break
        
        if best_match:
            results.append((
                best_match.payload['唯一ID'],
                best_match.score,
                best_match.payload['问题']
            ))
        else:
            results.append((None, 0.0, None))
    
    return results

def process_csv(input_file, output_file):
    """处理CSV文件并添加相似问题信息"""
    # 读取CSV文件
    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在: {input_file}")
        return
    
    df = pd.read_csv(input_file)
    
    # 检查必要的列
    required_columns = ['唯一ID', '问题']
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"CSV文件缺少必要的列: {col}")
            return
    
    logger.info(f"成功加载CSV文件，包含 {len(df)} 条记录")
    
    # 上传到Qdrant
    upload_to_qdrant(df)
    
    # 查找相似问题
    results = find_similar_questions(df)
    
    # 添加新列
    df['可能的重复ID'] = [r[0] for r in results]
    df['相似度'] = [r[1] for r in results]
    df['对应问题'] = [r[2] for r in results]
    
    # 保存结果
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    logger.info(f"处理完成，结果已保存到: {output_file}")

if __name__ == "__main__":
    INPUT_FILE = "raw_data/CFBenchmark整选择题-20250506.csv"  # 输入CSV文件路径
    OUTPUT_FILE = "output.csv"  # 输出CSV文件路径
    
    process_csv(INPUT_FILE, OUTPUT_FILE)