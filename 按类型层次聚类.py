from sklearn.metrics.pairwise import pairwise_distances
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import jieba
import jieba.posseg as pseg
from tqdm import tqdm
import logging
import re
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_stopwords(file_path: str) -> set:
    """
    加载停用词表
    :param file_path: 停用词文件路径
    :return: 停用词集合
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            stopwords = {line.strip() for line in f}
        return stopwords
    except FileNotFoundError:
        logger.warning(f"停用词文件未找到: {file_path}，将使用空停用词表")
        return set()


def preprocess_text(text: str, stopwords: set) -> str:
    """
    预处理文本，包括分词、去停用词、词性过滤等
    :param text: 原始文本
    :param stopwords: 停用词集合
    :return: 处理后的文本
    """
    if not isinstance(text, str):
        return ''

    # 去除非中文字符
    text = re.sub(r'[^\u4e00-\u9fa5]', ' ', text)

    # 分词并过滤词性和停用词
    words = pseg.cut(text)
    # 保留名词、动词、形容词，长度大于1且不在停用词表中的词
    filtered_words = [word.word for word in words
                      if word.flag.startswith(('n', 'v', 'a'))
                      and len(word.word) > 1
                      and word.word not in stopwords]

    return ' '.join(filtered_words)


def find_optimal_clusters(
        tfidf_matrix: np.ndarray,
        max_clusters: int = 20,
        batch_size: int = 1000
) -> int:
    """
    使用轮廓系数找到最优聚类数
    :param tfidf_matrix: TF-IDF特征矩阵
    :param max_clusters: 最大聚类数
    :param batch_size: MiniBatchKMeans的批处理大小
    :return: 最优聚类数
    """
    if tfidf_matrix.shape[0] < 10:  # 样本太少，不需要聚类
        return 1

    max_clusters = min(max_clusters, tfidf_matrix.shape[0] - 1)
    if max_clusters < 2:
        return 1

    best_score = -1
    best_k = 2

    for k in range(2, max_clusters + 1):
        try:
            kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=42)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)

            # 检查是否只有一个聚类
            if len(np.unique(cluster_labels)) < 2:
                continue

            silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)
            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_k = k
        except Exception as e:
            logger.warning(f"聚类计算k={k}时出错: {str(e)}")
            continue

    return best_k


def plot_clustering_metrics(
        tfidf_matrix: np.ndarray,
        max_clusters: int = 20,
        batch_size: int = 1000,
        output_path: Optional[str] = None
) -> None:
    """
    绘制不同聚类数的评估指标图
    :param tfidf_matrix: TF-IDF特征矩阵
    :param max_clusters: 最大聚类数
    :param batch_size: MiniBatchKMeans的批处理大小
    :param output_path: 图像保存路径，为None则显示图像
    """
    max_clusters = min(max_clusters, tfidf_matrix.shape[0] - 1)
    if max_clusters < 2:
        logger.warning("样本数量不足，无法进行聚类评估")
        return

    results = {'k': [], 'silhouette': [], 'calinski_harabasz': [], 'davies_bouldin': []}

    for k in range(2, max_clusters + 1):
        try:
            kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=42)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)

            # 检查是否只有一个聚类
            if len(np.unique(cluster_labels)) < 2:
                continue

            silhouette = silhouette_score(tfidf_matrix, cluster_labels)
            calinski = calinski_harabasz_score(tfidf_matrix.toarray(), cluster_labels)
            davies = davies_bouldin_score(tfidf_matrix.toarray(), cluster_labels)

            results['k'].append(k)
            results['silhouette'].append(silhouette)
            results['calinski_harabasz'].append(calinski)
            results['davies_bouldin'].append(davies)
        except Exception as e:
            logger.warning(f"计算聚类指标k={k}时出错: {str(e)}")
            continue

    if not results['k']:
        logger.warning("无法计算任何聚类指标")
        return

    # 绘制图表
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(results['k'], results['silhouette'], 'o-')
    plt.xlabel('聚类数')
    plt.ylabel('轮廓系数')
    plt.title('轮廓系数随聚类数变化')

    plt.subplot(1, 3, 2)
    plt.plot(results['k'], results['calinski_harabasz'], 'o-')
    plt.xlabel('聚类数')
    plt.ylabel('Calinski-Harabasz指数')
    plt.title('Calinski-Harabasz指数随聚类数变化')

    plt.subplot(1, 3, 3)
    plt.plot(results['k'], results['davies_bouldin'], 'o-')
    plt.xlabel('聚类数')
    plt.ylabel('Davies-Bouldin指数')
    plt.title('Davies-Bouldin指数随聚类数变化')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        logger.info(f"聚类评估图表已保存至: {output_path}")
    else:
        plt.show()


def process_group(
        group_df: pd.DataFrame,
        stopwords: set,
        max_samples: int = 500,
        max_clusters: int = 100,
        samples_per_cluster: int = 30,
        min_samples_per_cluster: int = 10
) -> pd.DataFrame:
    """
    处理单个分组（类型）的聚类和采样
    :param group_df: 分组的DataFrame
    :param stopwords: 停用词集合
    :param max_samples: 最大保留样本数
    :param max_clusters: 最大聚类数
    :param samples_per_cluster: 每个聚类保留样本数
    :param min_samples_per_cluster: 每个聚类最小保留样本数
    :return: 处理后的DataFrame
    """
    if len(group_df) <= max_samples:
        logger.info(f"样本数 {len(group_df)} 不超过最大限制 {max_samples}，直接返回")
        return group_df

    # 复制DataFrame以避免SettingWithCopyWarning
    group_df = group_df.copy()

    # 文本预处理
    logger.info(f"开始文本预处理，原始样本数: {len(group_df)}")
    group_df['processed_text'] = group_df['问题'].apply(lambda x: preprocess_text(x, stopwords))
    group_df = group_df[group_df['processed_text'] != '']

    if group_df.empty:
        logger.warning("预处理后没有有效样本，返回空DataFrame")
        return pd.DataFrame(columns=group_df.columns)

    logger.info(f"预处理后有效样本数: {len(group_df)}")

    # 提取TF-IDF特征
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(group_df['processed_text'])

    # 使用轮廓系数找到最优聚类数
    target_clusters = find_optimal_clusters(tfidf_matrix, max_clusters=max_clusters)
    logger.info(f"最优聚类数: {target_clusters}")

    if target_clusters < 2:
        logger.info(f"样本数不足或不需要聚类，直接采样 {max_samples} 个样本")
        return group_df.sample(n=min(max_samples, len(group_df)), random_state=42)

    # 聚类
    kmeans = MiniBatchKMeans(n_clusters=target_clusters, batch_size=1000, random_state=42)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)

    # 使用.loc避免SettingWithCopyWarning
    group_df.loc[:, 'cluster'] = cluster_labels

    # 计算距离并选择样本
    distances = pairwise_distances(tfidf_matrix, kmeans.cluster_centers_[cluster_labels], metric='cosine')
    group_df.loc[:, 'distance_to_center'] = distances.diagonal()

    # 从每个聚类中选择样本
    selected_indices = []
    logger.info(f"开始从 {target_clusters} 个聚类中选择样本")

    for cluster_id in range(target_clusters):
        cluster_df = group_df[group_df['cluster'] == cluster_id]
        cluster_size = len(cluster_df)

        if cluster_size == 0:
            logger.warning(f"聚类 {cluster_id} 为空，跳过")
            continue

        # 确定每个聚类要选择的样本数
        n_samples = min(
            max(min_samples_per_cluster, samples_per_cluster),
            cluster_size
        )

        # 选择离中心点最近的样本
        selected_indices.extend(cluster_df.sort_values('distance_to_center').index[:n_samples])
        logger.info(f"聚类 {cluster_id}: 大小={cluster_size}, 选择样本数={n_samples}")

    # 确保采样数不超过可用样本数
    filtered_df = group_df.loc[selected_indices]
    n_available = len(filtered_df)
    n_sample = min(max_samples, n_available)

    logger.info(f"从所有聚类中选择的总样本数: {n_available}, 最终将采样: {n_sample}")

    if n_available == 0:
        logger.warning("过滤后没有可用样本，返回空DataFrame")
        return pd.DataFrame(columns=group_df.columns)

    dedup_df = filtered_df.sample(n=n_sample, random_state=42)
    return dedup_df


def cluster_and_deduplicate_by_type(
        input_file: str,
        output_file: str,
        stopwords_path: str,
        max_samples_per_type: int = 500
) -> None:
    """
    按类型分组处理的主函数
    :param input_file: 输入文件路径
    :param output_file: 输出文件路径
    :param stopwords_path: 停用词路径
    :param max_samples_per_type: 每个类型最大保留样本数
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在: {input_file}")
        return

    # 加载数据和停用词
    try:
        df = pd.read_csv(input_file)
        logger.info(f"成功加载数据，总行数: {len(df)}")
    except Exception as e:
        logger.error(f"读取CSV文件失败: {str(e)}")
        return

    # 检查必要的列是否存在
    required_columns = ['问题', '类型']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"CSV文件缺少必要的列: {', '.join(missing_columns)}")
        return

    stopwords = load_stopwords(stopwords_path)

    # 按类型分组处理
    processed_dfs = []
    type_counts = df['类型'].value_counts()
    logger.info(f"数据包含 {len(type_counts)} 种类型")

    for group_name, group_df in tqdm(df.groupby('类型'), desc="处理分组"):
        logger.info(f"\n===== 处理类型: {group_name}, 原始样本数: {len(group_df)} =====")

        # 检查该类型的样本数是否超过限制
        if len(group_df) <= max_samples_per_type:
            logger.info(f"类型 {group_name} 的样本数 {len(group_df)} 未超过限制，直接保留")
            processed_dfs.append(group_df)
            continue

        processed_df = process_group(
            group_df,
            stopwords,
            max_samples=max_samples_per_type
        )

        logger.info(f"类型 {group_name}: 处理后保留样本数 {len(processed_df)}")
        processed_dfs.append(processed_df)

    # 合并结果
    if not processed_dfs:
        logger.warning("没有处理任何数据，输出空文件")
        pd.DataFrame().to_csv(output_file, index=False)
        return

    final_df = pd.concat(processed_dfs)

    # 保存结果
    try:
        final_df.to_csv(output_file, index=False)
        logger.info(f"\n===== 处理完成 =====")
        logger.info(f"总保留样本数: {len(final_df)}")
        logger.info(f"结果已保存至: {output_file}")
    except Exception as e:
        logger.error(f"保存结果失败: {str(e)}")


if __name__ == "__main__":
    input_csv = r"C:\Users\admin\Desktop\new_cot_answers_matched_merged.csv"
    output_csv = r"clustered_deduplicated_by_type.csv"
    stopwords_file = r"hit_stopwords.txt"

    cluster_and_deduplicate_by_type(
        input_csv,
        output_csv,
        stopwords_file,
        max_samples_per_type=500
    )