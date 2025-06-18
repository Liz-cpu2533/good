import pandas as pd
import numpy as np
import os
import json
import re
from chardet import detect

# 配置参数
RANDOM_SEED = 42
MIN_SAMPLE_SIZE = 100
SELECT_COUNT = 50
BASE_DIR = r"C:\Users\admin\Desktop"
INPUT_FILE = os.path.join(BASE_DIR, "CFBenchmark整合.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "抽样结果")
JSON_OUTPUT = os.path.join(BASE_DIR, "抽样结果.json")

def detect_encoding(file_path):
    """自动检测文件编码"""
    with open(file_path, 'rb') as f:
        return detect(f.read())['encoding']

def safe_filename(name):
    """生成安全文件名"""
    return re.sub(r'[\\/*?:"<>|]', '_', name)[:50]

def main():
    # 初始化环境
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(RANDOM_SEED)

    # 读取数据
    try:
        encoding = detect_encoding(INPUT_FILE)
        df = pd.read_csv(INPUT_FILE, encoding=encoding)
    except Exception as e:
        print(f"文件读取失败: {str(e)}")
        return

    # 验证数据列
    required_cols = ['唯一ID', '类型']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"缺失必要列: {missing}")
        print(f"现有列: {df.columns.tolist()}")
        return

    # 处理数据
    result = []
    for type_name, group in df.groupby('类型'):
        # 筛选数量≥100的类型
        if len(group) < MIN_SAMPLE_SIZE:
            continue

        # 打乱数据并抽样
        sampled = group.sample(n=SELECT_COUNT,
                              random_state=RANDOM_SEED,
                              replace=False)

        # 记录结果
        result.append({
            "类型": type_name,
            "数量": SELECT_COUNT,
            "唯一ID列表": sampled['唯一ID'].tolist()
        })

        # 保存CSV
        safe_name = safe_filename(type_name) + ".csv"
        output_path = os.path.join(OUTPUT_DIR, safe_name)
        sampled.to_csv(output_path,
                      index=False,
                      encoding='utf-8-sig')

    # 保存JSON结果
    with open(JSON_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"处理完成，共处理 {len(result)} 个类型")
    print(f"CSV文件保存至: {OUTPUT_DIR}")
    print(f"JSON结果保存至: {JSON_OUTPUT}")

if __name__ == "__main__":
    # 异常处理
    try:
        main()
    except PermissionError as e:
        print(f"权限错误: {str(e)}\n请关闭正在使用的CSV/Excel文件")
    except Exception as e:
        print(f"发生未预期错误: {str(e)}")