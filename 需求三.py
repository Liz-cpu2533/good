import pandas as pd
import os
from openpyxl import load_workbook
from collections import defaultdict
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 类型编号映射表
TYPE_MAPPING = {
    "基础知识-期货基础": "QHJC",
    "合规-法律法规": "FLFG",
    "合规-投资者对话合规性": "TZHG",
    "合规-内部员工对话合规性": "YGHG",
    "品种研究-投资机会研究": "TZJH",
    "指标分析-估值定价": "GZDJ",
    "指标分析-投资分析": "TZFX",
    "指标分析-风险分析 ： FXFX",
    "情绪检测-内容情绪检测": "NRQX",
    "情绪检测-客户情绪检测": "KHQX",
    "实体识别-实体消岐": "STXQ",
    "实体识别-实体识别": "STSB"
}

# 问题类型映射表
QUESTION_TYPE_MAPPING = {
    "选择题": "XZ",
    "简答题": "JD"
}


def generate_new_id(type_name, question_type, counters):
    """生成新的唯一ID"""
    type_code = TYPE_MAPPING.get(type_name, "UNKNOWN")
    question_code = QUESTION_TYPE_MAPPING.get(question_type, "UNKNOWN")

    # 获取当前计数器值并自增
    counters[(type_code, question_code)] += 1
    counter_value = counters[(type_code, question_code)]

    # 格式化为四位数字
    counter_str = f"{counter_value:04d}"

    return f"{type_code}-{question_code}-{counter_str}"


def process_excel(input_file, output_file):
    """处理Excel文件，添加新列"""
    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在: {input_file}")
        return

    try:
        # 读取Excel文件
        xls = pd.ExcelFile(input_file)

        # 读取两个sheet
        df_choice = xls.parse("选择题")
        df_short = xls.parse("简答题")

        logger.info(f"成功加载数据: 选择题 {len(df_choice)} 条，简答题 {len(df_short)} 条")

        # 检查必要的列
        required_columns = ["类型", "唯一ID", "前述问题ID"]
        for sheet_name, df in [("选择题", df_choice), ("简答题", df_short)]:
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"Sheet '{sheet_name}' 缺少必要的列: {col}")
                    return

        # 生成唯一ID映射字典
        id_mapping = {}
        counters = defaultdict(int)  # 用于跟踪每个类型的计数器

        # 处理选择题
        for idx, row in df_choice.iterrows():
            type_name = row["类型"]
            new_id = generate_new_id(type_name, "选择题", counters)
            original_id = row["唯一ID"]
            id_mapping[original_id] = new_id

        # 处理简答题
        for idx, row in df_short.iterrows():
            type_name = row["类型"]
            new_id = generate_new_id(type_name, "简答题", counters)
            original_id = row["唯一ID"]
            id_mapping[original_id] = new_id

        logger.info(f"已生成 {len(id_mapping)} 个唯一ID")

        # 映射前述问题ID
        def map_previous_id(original_id):
            if pd.isna(original_id) or original_id == "nan":
                return "nan"
            return id_mapping.get(original_id, "nan")

        # 添加新列
        df_choice["最新唯一ID"] = [id_mapping.get(oid, "nan") for oid in df_choice["唯一ID"]]
        df_choice["最新前述问题ID"] = df_choice["前述问题ID"].apply(map_previous_id)

        df_short["最新唯一ID"] = [id_mapping.get(oid, "nan") for oid in df_short["唯一ID"]]
        df_short["最新前述问题ID"] = df_short["前述问题ID"].apply(map_previous_id)

        # 保存结果
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_choice.to_excel(writer, sheet_name="选择题", index=False)
            df_short.to_excel(writer, sheet_name="简答题", index=False)

        logger.info(f"处理完成，结果已保存到: {output_file}")

    except Exception as e:
        logger.error(f"处理过程中出错: {str(e)}")
        return


def main():
    input_file = "src/问题库.xlsx"  # 替换为实际文件路径
    output_file = "output/问题库_新版.xlsx"  # 替换为实际输出路径

    process_excel(input_file, output_file)


if __name__ == "__main__":
    main()