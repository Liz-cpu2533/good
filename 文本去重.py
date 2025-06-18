import pandas as pd
import numpy as np
import os
import re
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextDeduplicator:
    def __init__(self, config):
        """初始化文本去重器"""
        self.config = config
        self.df = None
        self.vectors = None

    def load_data(self, file_path):
        """加载CSV数据"""
        logger.info(f"加载数据: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        self.df = pd.read_csv(file_path)
        logger.info(f"数据加载完成，共 {len(self.df)} 条记录")
        return self

    def preprocess_text(self, text_column='text'):
        """文本预处理"""
        logger.info("开始文本预处理")

        def clean_text(text):
            if not isinstance(text, str):
                return ""
            text = re.sub(r'[^\w\s]', '', text)  # 移除标点符号
            text = re.sub(r'\s+', ' ', text).strip()  # 合并连续空格
            return text.lower()  # 转换为小写

        self.df[text_column] = self.df[text_column].apply(clean_text)
        logger.info("文本预处理完成")
        return self

    def generate_vectors(self, text_column='text'):
        """使用TF-IDF生成文本向量"""
        logger.info("开始生成文本向量")

        # 处理空文本
        non_empty_texts = self.df[text_column].fillna("")

        vectorizer = TfidfVectorizer(
            max_features=5000,  # 限制特征数量，控制内存使用
            ngram_range=(1, 2),  # 同时考虑单字和双字
        )

        self.vectors = vectorizer.fit_transform(non_empty_texts)
        logger.info(f"文本向量生成完成，维度: {self.vectors.shape}")
        return self

    def deduplicate_by_minibatch_kmeans(self, text_column='text', target_count=3000):
        """基于MiniBatchKMeans聚类进行去重"""
        logger.info(f"开始基于MiniBatchKMeans聚类去重 (目标数量={target_count})")

        # 检查向量是否已经生成
        if self.vectors is None:
            raise ValueError("请先生成向量再进行去重")

        # 估计聚类数量
        n_clusters = min(target_count, len(self.df))
        logger.info(f"设置聚类数量: {n_clusters}")

        # 执行MiniBatchKMeans聚类
        logger.info("开始MiniBatchKMeans聚类...")

        # 转换为密集矩阵进行聚类（注意内存使用）
        dense_vectors = self.vectors.toarray()

        minibatch_kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            init='k-means++',
            batch_size=1024,  # 可根据实际情况调整批次大小
            n_init=10,
            max_iter=300,
            random_state=42
        )

        labels = minibatch_kmeans.fit_predict(dense_vectors)
        self.df['cluster'] = labels
        logger.info(f"聚类完成，共生成 {len(np.unique(labels))} 个聚类")

        # 从每个聚类中选择代表性样本
        logger.info("开始从聚类中选择代表性样本...")
        keep_indices = []

        for cluster_id in tqdm(np.unique(labels), desc="处理聚类"):
            cluster_indices = np.where(labels == cluster_id)[0]

            # 对于每个聚类，选择第一个样本或最长的文本
            if len(cluster_indices) > 0:
                # 选择最长的文本作为代表性样本
                text_lengths = [len(self.df.iloc[idx][text_column]) for idx in cluster_indices]
                max_length_idx = cluster_indices[np.argmax(text_lengths)]
                keep_indices.append(max_length_idx)

                # 检查是否达到目标数量
                if len(keep_indices) >= target_count:
                    break

        # 保存去重结果
        self.df_deduplicated = self.df.iloc[keep_indices].copy()
        logger.info(f"去重完成，保留 {len(self.df_deduplicated)} 条记录")
        return self

    def save_results(self, output_file):
        """保存去重结果"""
        logger.info(f"保存去重结果到: {output_file}")
        self.df_deduplicated.to_csv(output_file, index=False)
        return self


# 主函数
def main():
    # 配置参数
    config = {
        'input_file': 'C:/Users/admin/Desktop/new_cot_answers_matched_merged.csv',  # 输入CSV文件
        'output_file': 'deduplicated_new_cot_answers_matched.csv',  # 输出CSV文件
        'text_column': '问题',  # 文本列名
        'target_count': 5000,  # 目标保留记录数
    }

    # 创建去重器实例
    deduplicator = TextDeduplicator(config)

    # 执行去重流程
    try:
        deduplicator.load_data(config['input_file']) \
            .preprocess_text(config['text_column']) \
            .generate_vectors(config['text_column']) \
            .deduplicate_by_minibatch_kmeans(config['text_column'], config['target_count']) \
            .save_results(config['output_file'])
        logger.info("文本去重流程全部完成")
    except Exception as e:
        logger.error(f"文本去重过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    main()