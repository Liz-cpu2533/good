import pandas as pd
import numpy as np
import os
import re
import json  # 新增json模块
from chardet import detect

# 设置随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def try_read_csv(path, encodings=['utf-8', 'gbk', 'gb18030', 'utf-16', 'latin1']):
    """尝试多种编码方式读取CSV"""
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise ValueError(f"无法用{encodings}解码文件，请检查文件编码或格式")


# 文件路径配置
config = {
    "input": r"C:\Users\admin\Desktop\CFBenchmark整合.csv",
    "output": r"C:\Users\admin\Desktop\output",
    "result_json": r"C:\Users\admin\Desktop\output\results.json"  # JSON文件路径
}

try:
    # ========== 1. 数据加载 ==========
    if not os.path.exists(config["input"]):
        raise FileNotFoundError(f"输入文件不存在: {config['input']}")

    data = try_read_csv(config["input"])
    print("✔ 文件加载成功")
    print(f"数据维度: {data.shape}")

    # ========== 2. 数据验证 ==========
    required_columns = ['唯一ID', '类型']
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        available = data.columns.tolist()
        raise ValueError(f"缺失必要列:\n需要: {required_columns}\n实际: {available}")

    # ========== 3. 数据处理 ==========
    filtered = data[~data['类型'].str.contains('风险控制|风控', na=False, regex=True)]
    print(f"\n✔ 过滤后数据量: {len(filtered)}/{len(data)}")

    # 创建输出目录
    os.makedirs(config["output"], exist_ok=True)

    # ========== 4. 分组抽样 ==========
    results = []
    for group_name, group_data in filtered.groupby('类型'):
        sample_size = min(50, len(group_data))
        sampled = group_data.sample(n=sample_size, random_state=RANDOM_SEED)

        # 构建结构化结果
        results.append({
            "类型": group_name,
            "数量": sample_size,
            "唯一ID列表": sampled['唯一ID'].tolist()
        })

        # 保存CSV文件（可选）
        safe_name = re.sub(r'[^\w\u4e00-\u9fa5]', '_', group_name)[:50]
        output_path = os.path.join(config["output"], f"{safe_name}.csv")
        sampled.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"  → 已保存: {safe_name}.csv ({sample_size}条)")

    # ========== 5. 保存结果为JSON ==========
    with open(config["result_json"], 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✔ 结果已保存为JSON文件: {config['result_json']}")

    # ========== 6. 控制台输出 ==========
    print("\n" + "=" * 50)
    print("抽样结果预览 (JSON格式):")
    print(json.dumps(results[:2], indent=2, ensure_ascii=False))  # 只显示前2类

    print("\n完整结果结构示例:")
    print({
        "类型": "示例类型",
        "数量": 50,
        "唯一ID列表": ["ID1", "ID2", "..."]
    })

except Exception as e:
    print("\n" + "!" * 50)
    print(f"处理失败: {type(e).__name__}")
    print(f"错误详情: {str(e)}")
    print("!" * 50)

    if 'data' in locals():
        print("\n数据前2行:")
        print(data.head(2).to_markdown(index=False))