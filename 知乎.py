|-- conf
|   |-- __init__.py
|   |-- config.py
|   `-- logging_conf.py
|-- core
|   |-- __init__.py
|   |-- clustering_algorithm.py
|   |-- descend_dim.py
|   |-- model.py
|   |-- source_analysis.py
|   |-- statistic_analysis.py
|   |-- text_representation.py
|   `-- utils.py
|-- dataset
|-- db
|   |-- dict_new.txt
|   |-- hit_stopwords.txt
|   `-- zj_userdict.dict
import sys
import os
current_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = current_dir.replace('conf','')
sys.path.append(root_dir)

from conf.logging_conf import log_conf
dataset_dir = current_dir.replace('conf','dataset')
images_dir = current_dir.replace('conf','images')
model_dir = current_dir.replace('conf','model')
ptm_dir = current_dir.replace('conf','ptm')
log_dir = current_dir.replace('conf','log')
res_dir = current_dir.replace('conf','result')
db_dir = current_dir.replace('conf','db')
ptm_dir = root_dir.replace('CustomModule/large_scale_text_clustering','')

log_conf(os.path.join(log_dir,'statistic_analysis.log'))
import logging
from logging.handlers import RotatingFileHandler
import os

def log_conf(log_path):

    LOG_FORMAT = "%(asctime)s - %(filename)s - %(lineno)d -  %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    log_file_handler = RotatingFileHandler(filename=log_path, maxBytes=1024 * 1024*10,
                                           backupCount=1)
    # 设置日志打印格式
    formatter = logging.Formatter(LOG_FORMAT)
    log_file_handler.setFormatter(formatter)
    logging.getLogger('').addHandler(log_file_handler)
    logging.getLogger("transformers").setLevel(logging.INFO)
    logging.getLogger("jieba").setLevel(logging.WARNING)
 -*- coding:utf-8 -*-
'''
# 目标任务 :
参考 # https://blog.csdn.net/qq_37967241/article/details/116721067
'''
from sklearn.cluster import MiniBatchKMeans,KMeans
from sklearn.cluster import AgglomerativeClustering as aggl
from sklearn import metrics
import gensim
from tqdm import tqdm
import numpy as np
import codecs
import pandas as pd
from conf.config import *
import logging
import json
logging.getLogger().setLevel(logging.WARNING)

'''使用minibatch'''
def mini_cluster(tfidf_model,num_cluster,source_file,target_file):
    '''
    :param tfidf_model:  使用sklearn计算的tfidf模型
    :param num_cluster:  目标数目
    :param source_file:  原文本文件
    :param target_file:  聚类后文件
    :return:
    '''
    minikm_tfidf_lsa = MiniBatchKMeans(n_clusters=num_cluster,init='k-means++',init_size=1024*3,batch_size=1024).fit(tfidf_model)
    label = minikm_tfidf_lsa.labels_
    centroid = minikm_tfidf_lsa.cluster_centers_
    minikm_tfidf_lsa.fit_predict(tfidf_model)  # 每篇的类别
    np.savetxt(u'/Users/sunmengge/Desktop/scopus_content/aero_conf_cluster_center_600.txt', minikm_tfidf_lsa.cluster_centers_)  # 输出各个聚类簇的质心
    num_clusters = []
    cluster = minikm_tfidf_lsa.labels_tolist()
    result = codecs.open('','w')
    for label in cluster:
        result.write(str(label) + '\n')
    result.close()

    # 模型结果预测评价
    score_res = metrics.silhouette_score(tfidf_model,label,metric='euclidean')
    logging.info('聚类结果%s'%(str(score_res)))


'''使用hash特征进行聚类'''
def hash_cluster(hash_file,num_cluster,source_file,target_file):
    '''
    :param hash_file: hash指纹文件
    :param num_cluster: 聚类数目
    :param source_file:  # 原文本文件
    :param target_file:  # 聚类后文件
    :return:
    '''
    hash_path = os.path.join(dataset_dir,hash_file)
    df_data = pd.read_csv(hash_path,sep='\t',nrows=1000)
    simhash_list = df_data['simhash'].values.tolist()
    simhash_list = [[float(i)] for i in simhash_list]
    res = KMeans(n_clusters=num_cluster,init='k-means++',n_init=10).fit(simhash_list,y=None)  # 每行文本聚类的标签 [56,1]
    res_list = res.labels_.tolist()
    # print(res.labels_.tolist())
    target_file = os.path.join(res_dir,target_file)
    total_list = []
    for ind,raw in tqdm(enumerate(df_data['text'].values.tolist())):
        total_list.append([raw,res_list[ind]])
    df = pd.DataFrame(total_list,columns=['text','label'])
    df.to_csv(target_file,sep='\t',index=False)
    # df.to_excel(target_file,index=False)
    logging.info(('hash聚类完成'))

'''使用single_pass进行聚类,输入可以是tfidf,可以是文本向量'''

class single_pass():
    def __init__(self):
        ''''''
    def get_tfidf_similarity(self, cluster_cores, vector):
        max_value = 0
        max_index = -1
        for k, core in cluster_cores.items():
            similarity = gensim.matutils.cossim(vector, core)
            if similarity > max_value:
                max_value = similarity
                max_index = k
        return max_index, max_value
    def get_doc2vec_similarity(self, cluster_cores, vector):
        max_value = 0
        max_index = -1
        for k, core in cluster_cores.items():
            similarity = metrics.pairwise.cosine_similarity(vector.reshape(1, -1), core.reshape(1, -1))
            similarity = similarity[0, 0]
            if similarity > max_value:
                max_value = similarity
                max_index = k
        return max_index, max_value
    def tfidf_single_pass(self,corpus_vec,corpus_id_list,vocab,theta):
        '''
        :param corpus_vec:  向量文件gensim生成的向量
        :param corpus_id_list:  对应内容的id列表
        :param vocab:  词典 {int:str}
        :param theta:  阈值
        :return:  使用tfidf进行聚类
        '''
        clusters = {}
        cluster_cores = {}
        cluster_text = {}
        num_topic = 0
        cnt = 0
        for vector, text in tqdm(zip(corpus_vec, corpus_id_list), desc='聚类中...'):
            if num_topic == 0:
                clusters.setdefault(num_topic, []).append(vector)
                cluster_cores[num_topic] = vector
                cluster_text.setdefault(num_topic, []).append(text)
                num_topic += 1
            else:
                max_index, max_value = self.get_tfidf_similarity(cluster_cores, vector)
                if max_value > theta:
                    clusters[max_index].append(vector)
                    text_matrix = gensim.matutils.corpus2dense(clusters[max_index], num_terms=len(vocab),
                                                        num_docs=len(clusters[max_index])).T  # 稀疏转稠密
                    core = np.mean(text_matrix, axis=0)  # 更新簇中心
                    core = gensim.matutils.any2sparse(core)  # 将稠密向量core转为稀疏向量
                    cluster_cores[max_index] = core
                    cluster_text[max_index].append(text)
                else:  # 创建一个新簇
                    clusters.setdefault(num_topic, []).append(vector)
                    cluster_cores[num_topic] = vector
                    cluster_text.setdefault(num_topic, []).append(text)
                    num_topic += 1
            cnt += 1
            if cnt % 10000 == 0:
                logging.info('num_tops {}...'.format(num_topic)+'\t'+'processing {}...'.format(cnt))
        return clusters, cluster_text

    def doc2vec_single_pass(self, corpus_vec, corpus_id_list, theta):
        '''
        :param corpus_vec:
        :param corpus_id_list:
        :param theta:
        :return:  使用文本向量进行聚类
        '''
        clusters = {}
        cluster_cores = {}
        cluster_text = {}
        num_topic = 0
        cnt = 0
        for vector, text in tqdm(zip(corpus_vec, corpus_id_list), desc='聚类中...'):
            if num_topic == 0:
                clusters.setdefault(num_topic, []).append(vector)
                cluster_cores[num_topic] = vector
                cluster_text.setdefault(num_topic, []).append(text)
                num_topic += 1
            else:
                max_index, max_value = self.get_doc2vec_similarity(cluster_cores, vector)
                if max_value > theta:  #
                    clusters[max_index].append(vector)
                    core = np.mean(clusters[max_index], axis=0)  # 更新簇中心
                    cluster_cores[max_index] = core
                    cluster_text[max_index].append(text)
                else:  # 创建一个新簇  阈值太高很容易造成太多的新簇,时间太慢
                    clusters.setdefault(num_topic, []).append(vector)
                    cluster_cores[num_topic] = vector
                    cluster_text.setdefault(num_topic, []).append(text)
                    num_topic += 1
            cnt += 1
            if cnt % 10000 == 0:
                print('num_tops {}...'.format(num_topic), 'processing {}...'.format(cnt))
        return clusters, cluster_text

    def save_cluster(self, method, index2corpus, cluster_text, cluster_path):
        '''
        :param method:  使用tfidf还是文本向量
        :param index2corpus: {int:str}
        :param cluster_text:  每个聚类标签下对应的文本id {int:[id,id]}
        :param cluster_path:  结果保存路径
        :return:  保存聚类文件
        '''
        clusterTopic_list = sorted(cluster_text.items(), key=lambda x: len(x[1]), reverse=True)
        with open(cluster_path, 'w+', encoding='utf-8') as save_obj:
            for k in clusterTopic_list:
                data = dict()
                data["cluster_id"] = k[0]
                data["cluster_nums"] = len(k[1])
                data["cluster_docs"] = [{"doc_id": index, "doc_content": index2corpus.get(value)} for index, value in
                                        enumerate(k[1], start=1)]
                json_obj = json.dumps(data, ensure_ascii=False)
                save_obj.write(json_obj+'\n')
        logging.info('single_pass聚类完成')

'''使用SSE曲线判断最佳类别'''



if __name__ == '__main__':

    hash_cluster(hash_file='数据-hash.tsv',num_cluster=10,source_file='',target_file='数据-hash聚类结果.xlsx')

    pass
# -*- coding:utf-8 -*-
import sys
import os
current_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = current_dir.replace('core','')
sys.path.append(root_dir)

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import gensim
from collections import Counter
from conf.config import *
import logging


'''使用SVD降维,需要查看可解释比例,输入sklearn的tfidf'''
def svd_descend(tfidf):
    n_components_list = [64,128,256,512,1024,2048]
    explained_variance_final = 0
    for n_components in n_components_list:
        svd = TruncatedSVD(n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd,normalizer)
        tfidf_lsa = lsa.fit_transform(tfidf)
        explained_variance = svd.explained_variance_ratio_.sum()
        if explained_variance > 0.8:
            n_components_final = n_components
            break
        if explained_variance > explained_variance_final:
            n_components_final = n_components
            explained_variance_final = explained_variance
    svd = TruncatedSVD(n_components_final)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    tfidf_lsa = lsa.fit_transform(tfidf)
    explained_variance = svd.explained_variance_ratio_.sum()
    logging.info('解释方差比例%f'%explained_variance)
    logging.info('svd-lsa降维后维度%s'%(str(tfidf_lsa.shape)))
    return tfidf_lsa

'''使用lsi潜在语义分析,输入gensim的tfidf'''
def lsi_descend(tfidf,dictionary,num_topics):
    lsi_model = gensim.models.LsiModel(tfidf,id2word=dictionary,num_topics=num_topics)
    corpus_lsi = lsi_model[tfidf]
    return corpus_lsi

'''使用lda降维,输入gensim的tfidf'''
def lda_descend(tfidf,dictionary,num_topics):
    '''https://zhuanlan.zhihu.com/p/21364664'''
    lda_model = gensim.models.LdaModel(tfidf,id2word=dictionary,num_topics=num_topics)
    corpus_lda = lda_model[tfidf]
    return corpus_lda


'''测试脚本'''
def test():
    from core.statistic_analysis import sklearn_tfidf
    sklearn_tf_idf = sklearn_tfidf('数据分词列表.txt')
    tfidf_model = sklearn_tf_idf.get_tfidf()
    svd_descend(tfidf_model)
if __name__ == '__main__':

    test()
    pass
# -*- coding:utf-8 -*-
'''
---------------------------------
# 目标任务 :  通过节点树计算左右熵和互信息
---------------------------------
'''
from conf.config import *
import math
import pandas as pd


class Node(object):
    """
    建立字典树的节点
    """

    def __init__(self, char):
        self.char = char
        # 记录是否完成
        self.word_finish = False
        # 用来计数
        self.count = 0
        # 用来存放节点
        self.child = []
        # 方便计算 左右熵
        # 判断是否是后缀（标识后缀用的，也就是记录 b->c->a 变换后的标记）
        self.isback = False


class TrieNode(object):
    """
    建立前缀树，并且包含统计词频，计算左右熵，计算互信息的方法
    """

    def __init__(self, node, data=None, PMI_limit=20):
        """
        初始函数，data为外部词频数据集
        :param node:
        :param data:
        """
        self.root = Node(node)
        self.PMI_limit = PMI_limit
        if not data:
            return
        node = self.root
        for key, values in data.items():
            new_node = Node(key)
            new_node.count = int(values)
            new_node.word_finish = True
            node.child.append(new_node)

    def add(self, word):
        """
        添加节点，对于左熵计算时，这里采用了一个trick，用a->b<-c 来表示 cba
        具体实现是利用 self.isback 来进行判断
        :param word: ('机构',) 或 ('结构', '副框')
        :return:  相当于对 [a, b, c] a->b->c, [b, c, a] b->c->a
        """
        node = self.root
        # 正常加载
        for count, char in enumerate(word):
            found_in_child = False
            # 在节点中找字符
            for child in node.child:
                if char == child.char:
                    node = child
                    found_in_child = True
                    break

            # 顺序在节点后面添加节点。 a->b->c
            if not found_in_child:
                new_node = Node(char)
                node.child.append(new_node)
                node = new_node

            # 判断是否是最后一个节点，这个词每出现一次就+1
            if count == len(word) - 1:
                node.count += 1
                node.word_finish = True

        # 建立后缀表示
        length = len(word)
        node = self.root
        if length == 3:
            word = list(word)
            word[0], word[1], word[2] = word[1], word[2], word[0]

            for count, char in enumerate(word):
                found_in_child = False
                # 在节点中找字符（不是最后的后缀词）
                if count != length - 1:
                    for child in node.child:
                        if char == child.char:
                            node = child
                            found_in_child = True
                            break
                else:
                    # 由于初始化的 isback 都是 False， 所以在追加 word[2] 后缀肯定找不到
                    for child in node.child:
                        if char == child.char and child.isback:
                            node = child
                            found_in_child = True
                            break

                # 顺序在节点后面添加节点。 b->c->a
                if not found_in_child:
                    new_node = Node(char)
                    node.child.append(new_node)
                    node = new_node

                # 判断是否是最后一个节点，这个词每出现一次就+1
                if count == len(word) - 1:
                    node.count += 1
                    node.isback = True
                    node.word_finish = True

    def search_one(self):
        """
        计算互信息: 寻找一阶共现，并返回词概率
        :return:
        """
        result = {}
        node = self.root
        if not node.child:
            return False, 0

        # 计算 1 gram 总的出现次数
        total = 0
        for child in node.child:
            if child.word_finish is True:
                total += child.count
        # 计算 当前词 占整体的比例
        for child in node.child:
            if child.word_finish is True:
                result[child.char] = child.count / total
        # print(result)  # 每个词出现的概率
        # print(total)  # 数量非常大 160001512
        return result, total

    def search_bi(self):
        """
        计算互信息: 寻找二阶共现，并返回 log2( P(X,Y) / (P(X) * P(Y)) 和词概率
        :return:
        """
        result = {}
        node = self.root
        if not node.child:
            return False, 0

        total = 0
        # 1 gram 各词的占比，和 1 gram 的总次数
        one_dict, total_one = self.search_one()

        for child in node.child:
            for ch in child.child:
                if ch.word_finish is True:
                    total += ch.count
                    # print(ch.char, ch.count)
        # print(total)  # 183
        for child in node.child:
            for ch in child.child:
                if ch.word_finish is True:
                    # print(child.char,child.count,ch.char,ch.count,total)
                    # 互信息值越大，说明 a,b 两个词相关性越大  PMI是点互信息,只计算log值
                    PMI = math.log(max(ch.count, 1), 2) - math.log(total, 2) - \
                          math.log(one_dict[child.char], 2) - math.log(one_dict[ch.char], 2)
                    # 这里做了PMI阈值约束
                    # if PMI > self.PMI_limit:
                    # 例如: dict{ "a_b": (PMI, 出现概率), .. }
                    result[child.char + '_' + ch.char] = (PMI, ch.count / total)
        return result

    def search_left(self):
        """
        寻找左频次
        统计左熵， 并返回左熵 (bc - a 这个算的是 abc|bc 所以是左熵)
        :return:
        """
        result = {}
        node = self.root
        if not node.child:
            return False, 0

        for child in node.child:
            for cha in child.child:
                total = 0
                p = 0.0
                for ch in cha.child:
                    if ch.word_finish is True and ch.isback:
                        total += ch.count
                for ch in cha.child:
                    if ch.word_finish is True and ch.isback:
                        p += (ch.count / total) * math.log(ch.count / total, 2)
                # 计算的是信息熵
                result[child.char + cha.char] = -p
        return result

    def search_right(self):
        """
        寻找右频次
        统计右熵，并返回右熵 (ab - c 这个算的是 abc|ab 所以是右熵)
        :return:
        """
        result = {}
        node = self.root
        if not node.child:
            return False, 0

        for child in node.child:
            for cha in child.child:
                total = 0
                p = 0.0
                for ch in cha.child:
                    if ch.word_finish is True and not ch.isback:
                        total += ch.count
                for ch in cha.child:
                    if ch.word_finish is True and not ch.isback:
                        p += (ch.count / total) * math.log(ch.count / total, 2)
                # 计算的是信息熵
                result[child.char + cha.char] = -p
        return result

    def find_pmi(self):
        '''只计算点互信息,做为聚类判断使用'''
        bi = self.search_bi()
        result = {}
        for key, values in bi.items():
            result[key] = values[0]  # 只保留pmi
        result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

        return result

    def find_word(self, N):
        # 通过搜索得到互信息
        # 例如: dict{ "a_b": (PMI, 出现概率), .. }
        bi = self.search_bi()
        # 通过搜索得到左右熵
        left = self.search_left()
        right = self.search_right()
        result = {}
        for key, values in bi.items():
            d = "".join(key.split('_'))
            # 计算公式 score = PMI + min(左熵， 右熵) => 熵越小，说明越有序，这词再一起可能性更大！  PMI越大,越凝固,熵越小,越有序
            if len(key) != len(d) + 1:
                print(key, values, d)
                continue
            result[key] = (values[0] + min(left[d], right[d])) * values[1]
        # 按照 大到小倒序排列，value 值越大，说明是组合词的概率越大
        # result变成 => [('世界卫生_大会', 0.4380419441616299), ('蔡_英文', 0.28882968751888893) ..]
        result = sorted(result.items(), key=lambda x: x[1], reverse=True)  # 降序
        # print("result: ", result)
        # b = list(map(lambda x:[x[0],x[0].replace('_',''),x[1]],result))
        # df = pd.DataFrame(b, columns=['ngram名称', '合并名称', '分值'])
        # df.to_excel(newWord_path,index=0)
        dict_list = [result[0][0]]
        # print("dict_list: ", dict_list)
        add_word = {}
        new_word = "".join(dict_list[0].split('_'))
        # 获得概率
        add_word[new_word] = result[0][1]

        # 取前5个
        # [('蔡_英文', 0.28882968751888893), ('民进党_当局', 0.2247420989996931), ('陈时_中', 0.15996145099751344), ('九二_共识', 0.14723726297223602)]
        for d in result[1: N]:
            flag = True
            for tmp in dict_list:
                pre = tmp.split('_')[0]
                # 新出现单词后缀，在老词的前缀中 or 如果发现新词，出现在列表中; 则跳出循环
                # 前面的逻辑是： 如果A和B组合，那么B和C就不能组合(这个逻辑有点问题)，例如：`蔡_英文` 出现，那么 `英文_也` 这个不是新词
                # 疑惑: **后面的逻辑，这个是完全可能出现，毕竟没有重复**
                if d[0].split('_')[-1] == pre or "".join(tmp.split('_')) in "".join(d[0].split('_')):
                    flag = False
                    break
            if flag:
                new_word = "".join(d[0].split('_'))
                add_word[new_word] = d[1]
                dict_list.append(d[0])
        b = list(map(lambda x: [x[0], x[0].replace('_', ''), x[1]], result))
        df = pd.DataFrame(b, columns=['ngram名称', '合并名称', '分值'])
        # df.to_excel(newWord_path, index=0)
        return result, add_word


if __name__ == '__main__':
    pass
# -*- coding:utf-8 -*-
'''
---------
# 目标任务 :   数据统计分析
---------------------------------
'''
import sys
sys.path.append('/home/Algorithm_Frame/CustomModule/large_scale_text_clustering')
import os
import pandas as pd
import matplotlib.pyplot as plt
import jieba.posseg as pseg
from gensim import corpora, models, matutils
# import jieba
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from conf.config import *
from core.utils import *
import logging
import pickle
import json
import Levenshtein
from thefuzz.fuzz import QRatio
from thefuzz import process
from itertools import combinations, product

logging.getLogger().setLevel(logging.WARNING)

'''统计文本长度,词频'''
class text_statistic():
    def __init__(self,filename):
        self.filename = filename
        self.df_data = self.read_tsv()
        self.stop_words = self.read_stop()
    def read_tsv(self):
        df_data = pd.read_csv(os.path.join(dataset_dir,self.filename),sep='\t')
        logging.info('文件读取完成')
        return df_data

    def get_len_res(self):
        '''统计长度'''
        self.df_data['text_len'] = self.df_data['patrol_desc'].str.len()
        logging.info('数据量%d'%len(self.df_data['text_len']))
        self.df_data = self.df_data.loc[self.df_data['text_len']<80].copy()
        logging.info('保留频数多的数据,数据量%d'%len(self.df_data['text_len']))
        # print(self.df_data['text_len'].value_counts())
        self.df_data.text_len.plot(kind='hist',bins=100,color="steelblue",label="直方图")
        # self.df_data.text_len.plot(kind='kde',color="red",label="核密度图")
        plt.xlabel('文本长度')
        plt.title('文本长度分布图')
        plt.legend()
        plt.savefig(os.path.join(images_dir,self.filename.replace('.tsv','.png')))
        plt.show()
        logging.info('文本长度分布分析完成')

    def read_stop(self):
        '''加载停用词'''
        stop_words = [i.strip() for i in open(os.path.join(dataset_dir,'hit_stopwords.txt')).readlines()]
        logging.info('停用词加载完成')
        return set(stop_words)

    def jieba_seg(self,sentence_list):
        '''使用结巴分词,加载词性'''
        # jieba.enable_parallel(10)
        pos_list = set(['n','nz','a','v','vd','vn'])
        total_list = []
        # length = len(sentence_list)
        # batch_size = 32
        # for i in range(0,length,batch_size):
        #     start = i
        #     end = i+batch_size
        #     words_list = pseg.cut(sentence_list[start:end])
        #     print(words_list)
        #     for words in words_list:
        #         print(words)
        #         word_list = []
        #         for word, flag in words:
        #             if flag in pos_list and word not in self.stop_words and len(word) > 1:
        #                 word_list.append(word)
        #         total_list.append(word_list)
        for text in tqdm(sentence_list,desc='jieba分词中'):
            words = pseg.cut(text)
            word_list = []
            for word,flag in words:
                if flag in pos_list and word not in self.stop_words and len(word)>1:
                    word_list.append(word)
            total_list.append(word_list)
        # jieba.disable_parallel()
        logging.info('jieba分词完成')
        return total_list

    def generate_wordlist(self,mode='jieba'):
        '''生成分词列表'''
        self.df_data['text_len'] = self.df_data['patrol_desc'].str.len()
        self.df_data = self.df_data.loc[self.df_data['text_len']<80].copy()
        sentence_list = self.df_data['patrol_desc'].values.tolist()
        if mode == 'jieba':
            total_list = self.jieba_seg(sentence_list)
        with open(os.path.join(dataset_dir,self.filename.replace('.tsv','分词列表.txt')),'w') as fw:
            for word_list in total_list:
                fw.write(str(word_list)+'\n')
        logging.info('分词词表生成完成')

'''使用gensim计算tfidf'''
class gensim_tfidf():
    '''https://blog.csdn.net/iszhuangsha/article/details/85163685'''
    def __init__(self,seg_filename):
        self.filename = seg_filename
        self.words_list = self.read_seg()
    def read_seg(self):
        # sklearn 空格或其他符号分割
        words_list = [' '.join(eval(i)) for i in open(os.path.join(dataset_dir,self.filename)).readlines()]
        return words_list
    def get_tfidf(self):
        gensim_tfidf_path = os.path.join(dataset_dir,self.filename.replace('.txt','-gensim-tfidf.mm'))
        gensim_id_token_path = os.path.join(dataset_dir,self.filename.replace('.txt','-gensim-dictionary.json'))
        if os.path.exists(gensim_tfidf_path):
            dictionary = json.load(gensim_id_token_path)
            tfidf = corpora.MmCorpus(gensim_tfidf_path)
            return tfidf,dictionary
        else:
            dictionary = corpora.Dictionary(self.words_list)
            logging.info('gensim统计词典数量%d' % len(dictionary))
            corpus = [dictionary.doc2bow(text) for text in self.words_list]
            tfidf = models.TfidfModel(corpus)  # 得到单词的tf-idf值
            with open(gensim_id_token_path,'w') as fw:
                json.dump(dictionary.id2token,fw,indent=2,ensure_ascii=False)
            corpora.MmCorpus.serialize(gensim_tfidf_path,tfidf)
            return tfidf,dictionary

'''使用sklearn计算tfidf'''
class sklearn_tfidf():
    def __init__(self,seg_filename):
        self.filename = seg_filename
        self.words_list = self.read_seg()
    def read_seg(self):
        # sklearn 空格或其他符号分割
        words_list = [' '.join(eval(i)) for i in open(os.path.join(dataset_dir,self.filename)).readlines()]
        return words_list
    def get_tfidf(self):
        '''获取idf:出现该词的行数/总行数,词频默认一行中所有词都只出现一次(短文本),有log值'''

        sklearn_idf_path = os.path.join(dataset_dir,self.filename).replace('.txt','-sklearn-idf.tsv')
        sklearn_tfidf_path = os.path.join(dataset_dir,self.filename.replace('.txt','-sklearn-tfidf.pkl'))
        sklearn_vocabulary_path = os.path.join(dataset_dir,self.filename.replace('.txt','-sklearn-vocab.pkl'))

        if os.path.exists(sklearn_idf_path):
            logging.info('加载已有sklearn模型')
            tfidf = pickle.load(open(sklearn_tfidf_path,'rb'))
            vocab = pickle.load(open(sklearn_vocabulary_path,'rb'))
            logging.info('sklearn统计tfidf维度 %s' % (str(tfidf.shape)))
            return tfidf,vocab
        else:
            vectorizer = CountVectorizer(decode_error="replace")
            textvector = vectorizer.fit_transform(self.words_list)
            transformer = TfidfTransformer(norm=None)
            tfidf = transformer.fit_transform(textvector)  # 假设idf即为每个单词出现的概率
            with open(sklearn_vocabulary_path,'wb') as fw:
                pickle.dump(vectorizer.vocabulary_,fw)
            with open(sklearn_tfidf_path,'wb') as fw:
                pickle.dump(sklearn_tfidf_path,fw)
            df_word_idf = pd.DataFrame(list(zip(vectorizer.get_feature_names(), transformer.idf_)), columns=['单词', 'idf'])
            df_word_idf.to_csv(sklearn_idf_path,sep='\t',index=False)
            logging.info('sklearn统计tfidf维度 %s' % (str(tfidf.shape)))
            return tfidf,vectorizer.vocabulary_

'''计算pmi'''
def get_pmi(seg_filename,res_file,filename):
    '''计算点互信息'''
    from core.model import TrieNode
    root_name = os.path.join(db_dir,'root.pkl')
    stopwords = get_stopwords(os.path.join(db_dir,'hit_stopwords.txt'))
    if os.path.exists(root_name):  # 如果有新的语料一定要删除root.pkl
        root = load_model(root_name)
    else:
        dict_name = os.path.join(db_dir,'dict_new.txt')
        word_freq = load_dictionary(dict_name)
        root = TrieNode('*', word_freq)
        save_model(root, root_name)

    # 加载新的文章
    if os.path.exists(seg_filename):
        data = [eval(i) for i in open(seg_filename).readlines()][:100]
        # print(data)
        # 将新的文章插入到Root中
        print('------> 插入节点')
        for word_list in tqdm(data):
            # tmp 表示每一行自由组合后的结果（n gram）
            # tmp: [['它'], ['是'], ['小'], ['狗'], ['它', '是'], ['是', '小'], ['小', '狗'], ['它', '是', '小'], ['是', '小', '狗']]
            ngrams = generate_ngram(word_list, 2)
            for d in ngrams:
                root.add(d)
        print('------> 插入成功')
        result = root.find_pmi()
        fw = open(res_file, 'w')
        json.dump(result, fw, indent=2, ensure_ascii=False)
        fw.close()
        # print(json.dumps(result,indent=2,ensure_ascii=False))
    else:
        # 语料一定要清晰，去除乱码、空白符等
        # data = load_data(filename, stopwords)
        df_total,df_mapping = get_mapping_data(filename,stopwords,2)
        print('------> 插入节点')
        for ngrams in tqdm(df_total['n_gram列表'].values.tolist()):
            for d in ngrams:
                root.add(d)
        print('------> 插入成功')
        result = root.find_pmi()
        print('------> pmi计算完成')
        fw = open(res_file, 'w')
        json.dump(result, fw, indent=2, ensure_ascii=False)
        fw.close()
        # print(json.dumps(result,indent=2,ensure_ascii=False))
        # print(df_mapping)
        # print(result)
        res_list = []
        for ind,raw in df_mapping.iterrows():
            raw = dict(raw)
            ngram = raw['n_gram']
            if len(ngram) > 1:
                ngram = '_'.join(ngram)
            if ngram in result:
                pmi = result[ngram]
            else:
                pmi = 1
            raw['pmi'] = pmi
            res_list.append(list(raw.values()))
        df_data = pd.DataFrame(res_list,columns=['n_gram','原句','pmi'])

        # 根据pmi排序
        df_data.sort_values(by='pmi',inplace=True,ascending=False)
        # 保留第一个非重复值
        df_data.drop_duplicates(subset=['原句'],inplace=True)
        df_data.to_excel(filename.replace('.tsv','_关键词组合.xlsx'),index=False)

        # 后续可以根据提取包含想要的关键词的句子,以n_gram进行聚合

        # 根据高频词组合

'''基于字面的模糊匹配'''
def fuzz_similarity(df_data:pd.DataFrame,field_name):
    '''
    :param df_data: 去重DataFrame
    :param field_name:  去重字段
    :return: 返回DataFrame
    '''
    # 数量不超过1000效率较高

    def compute_similarity(i, j, scorer=QRatio):
        to_be_match_dict = j[j != i].to_dict()  # 不包含当前的任务名称i
        if len(to_be_match_dict) == 0:
            return ['', '', '']
        res = process.extractOne(i, to_be_match_dict, scorer=scorer)  # 返回三列
        return res
    df_data.drop_duplicates(subset=[field_name],inplace=True)
    df_data = df_data.sample(frac=1)
    df_match = df_data.copy(deep=True)
    # tqdm.pandas(desc='fuzz匹配处理中') progress_apply
    df_match['match'] = df_data[field_name].apply(compute_similarity, args=(df_data[field_name],))
    df_match['match_'+field_name] = df_match['match'].apply(lambda x: x[0])
    df_match['match_分数'] = df_match['match'].apply(lambda x: x[1])
    df_match.drop(['match'], axis=1,inplace=True)

    total_set = set()
    temp_set = set()
    for ind, raw in df_match.iterrows():
        score = raw['match_分数']
        text_a = raw[field_name]
        text_b = raw['match_'+field_name]
        if score > 85:  # 设置的阈值
            if text_a not in temp_set:
                total_set.add(text_a)  # 先把当前的数据放到结果集中
                temp_set.add(text_a)  # 将当前数据放到缓存中
                temp_set.add(text_b)  # 将已经匹配过的数据放到缓存中
        else:  # 相似度不高
            if text_a not in temp_set:  # 当前数据不在缓存数据集里
                total_set.add(text_a)  # 添加到结果集
    df_data_1 = pd.DataFrame(total_set, columns=[field_name])
    print('最终数量', len(total_set))
    df_match = pd.merge(df_data_1,df_match,on=field_name)

    return df_match

def hamming(a,b):
    '''len(a)==len(b)'''
    return Levenshtein.hamming(a,b)

'''基于hamming距离的统计去重'''
def remove_hamming(df_data:pd.DataFrame,field_name):
    '''
    :param df_data: 待去重的DataFrame
    :param filed_name: 待去重的字段
    :return: 返回去重后的DataFrame
    '''
    # 数量不超过10000效率较高
    df_match = df_data.copy(deep=True)
    df_match = df_match[[field_name]]  # 要去重的字段
    df_match.drop_duplicates(inplace=True)
    df_match['text_len'] = df_match[field_name].str.len()  # 计算文本长度
    df_match = df_match.loc[df_match['text_len'] > 5].copy()
    print('数据量', len(df_match))  # 333634
    df_match = df_match.sample(frac=1, random_state=22)   # 随机选择一定比例数据
    total_list = []
    for df_data_single in tqdm(df_match.groupby('text_len'), desc='df_match'):  # 按不同长度进行分组
        text_list = df_data_single[1][field_name].values.tolist()
        num = 0
        while len(text_list) != 1 and len(text_list) > num + 1:
            a = [text_list[num]]
            b = text_list[num + 1:]
            c = product(a, b)
            for i in c:
                score = hamming(i[0], i[1])
                if score <= 2 or score > 10:  # 设置过滤值大小
                    text_list.remove(i[1])
            num += 1
        total_list.extend(text_list)
    total_set = set(total_list)
    df_match.drop(columns=['text_len'], inplace=True)
    total_list = []
    for ind, raw in tqdm(df_match.iterrows()):
        raw = dict(raw)
        text_a = raw[field_name]
        if text_a in total_set:
            total_list.append(list(raw.values()))
    print('df_match 去重前数量', len(df_match))
    print('df_match 汉明去重后数量', len(total_list))
    df = pd.DataFrame(total_list, columns=df_match.columns.tolist())

    df_match = pd.merge(df,df_data,on=field_name)
    return df_match
    # df.to_excel(os.path.join('v7', '待清洗去重数据', '0524-%d-待标注.xlsx' % len(total_list)), index=False)

if __name__ == '__main__':
    # text_sta = text_statistic('数据.tsv')
    # text_sta.get_len_res()  # 文本长度分布
    # text_sta.generate_wordlist()

    # sklearn_tf_idf = sklearn_tfidf('数据分词列表.txt')
    # sklearn_tf_idf.get_tfidf()


    # 编辑距离过滤
    df_data = pd.DataFrame({"任务名称":["这是一个任务名称","这是两个任务名称的计算",'这是两个任务名称'],
                            "临时字段":["懂法守法","金佛文风的分配","金佛文风的分"],
                            "临时": ["dsofdhsfds",'dfosdhfdsfo','dshfodsh']})
    # print(df_data.head())
    # df = fuzz_similarity(df_data,"任务名称")
    # df = remove_hamming(df_data,"任务名称")
    # print(df.head())

    # get_pmi(os.path.join(dataset_dir,'数据分词列表.txt'),
    #         os.path.join(dataset_dir,'数据PMI结果.json'),
    #         '')
    get_pmi('',
            os.path.join(dataset_dir,'数据PMI结果.json'),
            os.path.join(dataset_dir,'数据.tsv'))
    #
    pass
# -*- coding:utf-8 -*-

from conf.config import *

from simhash import Simhash
from tqdm import tqdm
import torch
import pickle
from transformers import AlbertModel, BertTokenizer
import pandas as pd
import logging
import re

def filter_puntuation(string):
    text=re.sub('[\.\!\/_,-:;<>≦$%^*()+\"\']+|[+——！，。？、~@#￥%……&*（）]+',' ',string)
    text=re.sub('  ','',text,3)
    return text

def get_features(s,width):
    s = s.lower()
    s = re.sub(r'[^\w]+', '', s)
    return [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]

def read_tsv(text_path):
    df_data = pd.read_csv(text_path, sep='\t')
    df_data.rename(columns={'patrol_desc': 'text_a'}, inplace=True)
    return df_data

'''hash表示'''
def hash_vector(filename,seg_file,ngram=2,use_seg=True):
    logging.getLogger("simhash").setLevel(logging.WARNING)
    file_path = os.path.join(dataset_dir, filename)
    df_data = read_tsv(file_path)
    out_file = os.path.join(dataset_dir, filename).replace('.tsv', '-hash.tsv')
    text_list = df_data['text_a'].values.tolist()
    logging.info('数据总长度 %d'%len(text_list))
    total_list = []
    if use_seg: # 读取分词列表
        seg_file = os.path.join(dataset_dir,seg_file)
        fr = open(seg_file,'r').readlines()
        for id,seg,in tqdm(enumerate(fr),desc='分词hash处理'):
            simhash_ = Simhash(eval(seg)).value
            total_list.append([id,text_list[id],simhash_])
    else:
        for id,text in tqdm(enumerate(text_list),desc='原句hash处理'):
            simhash_ = Simhash(get_features(text,ngram)).value
            total_list.append([id,text,simhash_])
    df = pd.DataFrame(total_list,columns=['id','text','simhash'])
    df.to_csv(out_file,sep='\t',index=False)
    logging.info('simhash向量表征完成')
    return df


'''通过PTM-encode文本向量'''
def albert_create_vector(filename):
    '''使用自定义的预训练模型'''
    pretrained_path = os.path.join(ptm_dir, 'PretrainingModel/tf-albert')
    file_path = os.path.join(dataset_dir,filename)
    corpus_vector_pkl_path = os.path.join(dataset_dir, filename.replace('.tsv','-vector.pkl'))

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    model = AlbertModel.from_pretrained(pretrained_path, from_tf=True).to(device)
    logging.info('albert正常加载模型')
    df_data = read_tsv(file_path)
    text_list = df_data['text_a'].values.tolist()
    total_length = len(text_list)
    batchsize = 128
    max_length = 64
    vectors = []
    for i in tqdm(range(0,total_length,batchsize),desc='提取句向量'):
        start = i
        end = i+batchsize
        input_id_list = []
        for inputtext in text_list[start:end]:
            input_ids = tokenizer.encode(inputtext)
            input_id_list.append(input_ids)
        input_ids_padding = []
        for i in input_id_list:
            if max_length > len(i):
                input_ids_padding.append(i + [0] * (max_length - len(i)))
            else:
                input_ids_padding.append(i[:max_length])
        input_ids_padding = torch.tensor(input_ids_padding).to(device)
        with torch.no_grad():
            vector = model(input_ids_padding).pooler_output.cpu().detach().numpy()
            vectors.extend(vector)
    embeddign_cache = open(corpus_vector_pkl_path,'wb')
    pickle.dump({"vectors":vectors},embeddign_cache)
    embeddign_cache.close()
    logging.info('albert提取句向量完成,数量 %d'%len(vectors))


'''多进程生成大批量数据'''

class multi_vector():

    def __init__(self,filename):
        self.filename = filename
        df_data = read_tsv(os.path.join(dataset_dir,filename))
        self.text_list = df_data['text_a'].values.tolist()
    def single_process(self,inputtext_list,start_index,end_index):
        pretrained_path = os.path.join(ptm_dir, 'PretrainingModel/tf-albert')

        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = BertTokenizer.from_pretrained(pretrained_path)
        model = AlbertModel.from_pretrained(pretrained_path, from_tf=True).to(device)
        inputtext_list = inputtext_list[start_index:end_index]
        total_length = len(inputtext_list)
        batchsize = 128
        max_length = 64
        vectors = []
        for i in tqdm(range(0, total_length, batchsize), desc='提取句向量%d-%d' % (start_index, end_index)):
            start = i
            end = i + batchsize
            input_id_list = []
            for inputtext in inputtext_list[start:end]:
                input_ids = tokenizer.encode(inputtext)
                input_id_list.append(input_ids)
            input_ids_padding = []
            for i in input_id_list:
                if max_length > len(i):
                    input_ids_padding.append(i + [0] * (max_length - len(i)))
                else:
                    input_ids_padding.append(i[:max_length])
            input_ids_padding = torch.tensor(input_ids_padding).to(device)
            with torch.no_grad():
                vector = model(input_ids_padding).pooler_output.cpu().detach().numpy()
                vectors.extend(vector)
        return vectors
    def multi_process(self):
        import multiprocessing as mp
        from multiprocessing import Pool
        process_num = 5
        Pool(processes=process_num)
        total_length = len(self.text_list)
        # 根据进程数设置起始结束索引值
        per_num = total_length // process_num
        results = []
        for i in range(process_num):
            start = per_num * i
            end = per_num * (i + 1)
            res = mp.Process(target=self.single_process, args=(self.text_list, start, end))
            results.append(res)
            res.start()
        for task in results:
            task.join()
        vectors = []
        for i in results:
            vectors.extend(i.get())
        corpus_vector_pkl_path = os.path.join(dataset_dir, self.filename.replace('.tsv', '-vector.pkl'))
        embeddign_cache = open(corpus_vector_pkl_path, 'wb')
        pickle.dump({"vectors": vectors}, embeddign_cache)
        embeddign_cache.close()
        logging.info('多进程提取句向量完成,数量 %d' % len(vectors))


if __name__ == '__main__':

    # print(get_features('4号楼门',2))
    hash_vector('数据.tsv','',use_seg=False)

    pass
# -*- coding:utf-8 -*-

import logging
import pickle
from tqdm import tqdm
import jieba
import pandas as pd
import os
import jieba.posseg as pseg

def get_stopwords(stop_path):
    with open(stop_path, 'r',encoding='utf-8') as f:
        stopword = [line.strip() for line in f]
    return set(stopword)


def generate_ngram(input_list, n):
    '''
    :param input_list: 输入 ['机构', '结构', '框']
    :param n: n_gram
    :return: [('机构',), ('结构',), ('框',), ('机构', '结构'), ('结构', '框')]
    '''
    result = []
    for i in range(1, n+1):
        result.extend(zip(*[input_list[j:] for j in range(i)]))
    # logging.info('ngram数据生成完成')
    return result

def load_dictionary(filename):
    """
    加载外部词频记录,最好是垂直业务词典
    :param filename:
    :return:
    """
    word_freq = {}
    with open(filename, 'r',encoding='utf-8') as f:
        for line in f:
            try:
                line_list = line.strip().split(' ')
                # 规定最少词频
                if int(line_list[1]) >= 1:
                    word_freq[line_list[0]] = line_list[1]
            except IndexError as e:
                print(line)
                continue
    return word_freq

def save_model(model, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(model, fw)


def load_model(filename):
    with open(filename, 'rb') as fr:
        model = pickle.load(fr)
    return model


def load_data(filename, stopwords):
    """
    :param filename:
    :param stopwords:
    :return: 二维数组,[[句子1分词list], [句子2分词list],...,[句子n分词list]]
    """
    data = []
    with open(filename, 'r',encoding='utf-8') as f:
        for line in f:
            line_list = line.strip().split(' ')
            for i in line_list:
                word_list = [x for x in jieba.cut(i, cut_all=False) if x not in stopwords]
                data.append(word_list)
    return data


# 生成DataFrame数据
def get_mapping_data(filename,stopwords,n):
    '''
    filename: tsv数据(其他数据格式自定义处理)
    stopwords: 停用词集合
    :return: 文件1 ['原句','分词列表','n_gram列表'] 文件2['n_gram','原句']
    '''


    df_data = pd.read_csv(filename, sep='\t',nrows=10000)
    sentence_list = df_data['patrol_desc'].values.tolist()  # 自定义字段

    pos_list = set(['n','nz','a','v','vd','vn'])
    total_list = []
    map_list = []
    for ind,text in tqdm(enumerate(sentence_list),desc='jieba分词中'):
        words = pseg.cut(text)
        word_list = []
        for word,flag in words:
            if flag in pos_list and word not in stopwords and len(word)>1:
                word_list.append(word)
        n_gram_list = generate_ngram(word_list,n)
        total_list.append([text,word_list,n_gram_list])
        for i in n_gram_list:
            map_list.append([i,text])

    df_total = pd.DataFrame(total_list,columns=['原句','分词列表','n_gram列表'])
    df_total.to_excel(filename.replace('.tsv','_分词_ngram.xlsx'),index=False)

    df_mapping = pd.DataFrame(map_list,columns=['n_gram','原句'])
    df_mapping.to_excel(filename.replace('.tsv','_ngram映射.xlsx'))

    # jieba.disable_parallel()
    logging.info('jieba分词完成')
    return df_total,df_mapping

if __name__ == '__main__':
    pass
# 补充
# faiss, milvus向量聚类原理
# 背景
# 一个简单的特征相似度比对的例子
# 在多个CUDA设备上进行特征的相似度搜索
# Faiss环境准备
# Faiss的简单使用：Flat
# xb用来表示特征库，xq用来表示待查询的2048维向量。如果沿用上面的例子，则xb就是提前存储了7030个样本的特征的“数据库”，它的shape就是7030x2048——这样的“数据库”在Faiss中称作Index object
# IndexFlatL2:最简单的库:使用暴力L2搜索的数据库——也就是和特征库中的每个特征进行L2距离计算然后取出距离最近的那个 Index object的“训练”其实就是提前根据特征的分布进行聚类训练,IndexFlatL2并不在列 xq的batch很小，Index很小：CPU通常更快； xq的batch很小，Index很大：GPU通常更快； xq的batch很大，Index很小：随便； xq的batch很大，Index很大：GPU通常更快； GPU通常比CPU快5到10倍；
# 让Faiss使用更少的内存：PQ
# Product Quantizer,这是一种有损压缩，所以这种Index进行检索的返回值只是近似的准确 上面的centroid（最类中心的那个2048维向量），在Faiss中我们称之为code；上述的8组256个centroid，在Faiss中我们称之为8个code book；这8个code book可以表达256^8个值，还是很大的。
# 让Faiss进行更快的检索：IVF
# 两两特征比对更少的计算量；PQ顺带着做了； 只和特征库的一部分进行比对；和特征库的每一个特征进行比对，叫做穷举；只和部分特征进行比对，叫做IVF(倒排索引)； 因此，在Faiss中所有带有IVF的index，指的都是提前将数据库使用k-means聚类算法划分为多个partition，每个partition中包含有对应的feature vectors（这就是inverted file lists指向的），同时，每个partition还有对应的centroid，用来决定该partition是应该被穷举搜索，还是被忽略。
# 这个partition的概念，在Faiss中称之为Voronoi cells；选择某个Voronoi cells然后进行检索的动作，称之为“probe”；而在最近的“多少个”Voronoi cells中进行检索，这个“多少个”的数量称之为nprobe；在IVF Index object的属性中，就有nprobe这个属性 ==IndexIVFPQ的参数中，d代表特征的维度2048，nlist代表Voronoi cells的数量，m代表subquantizers的数量（给PQ用），8代表使用8bits表示一个特征的1个sub-vector==。
# 无论如何内存都不够用：Distributed index
# 将特征库分散加载到多个服务器的CUDA设备上
# 无论如何内存都不够用：On-disk storage
# 参考项目中的faiss/contrib/ondisk.py
# 当xq是pytorch的tensor时
# Faiss提供了一种临时的措施，可以直接读取Tensor底层的内存（Tensor必须是is_contiguous的），然后使用index的search_c() API来检索。关于在index中检索PyTorch Tensor的详细用法，
# 最后，如何选择一种Index呢？
# 我们已经见识过的关键字有Flat、IVF、PQ，那么如何选择一种Index来匹配我们的场景呢？
# 当需要绝对准确的结果时，使用Flat；比如IndexFlatL2 或者 IndexFlatIP；
# 如果内存完全够用富裕的不行，使用HNSW；如果一般够用，使用Flat；如果有点吃紧，使用PCARx,...,SQ8；如果非常吃紧，使用OPQx_y,...,PQx；
# 如果特征库小于1M个记录，使用"...,IVFx,..."==；如果在1M到10M之间，使用"...,IVF65536_HNSW32,..."==；如果在10M - 100M之间，使用"...,IVF262144_HNSW32,..."；如果在100M - 1B之间，使用"...,IVF1048576_HNSW32,..."。
# 用于训练的xb记录越多，聚类算法的训练效果越好，但训练需要花的时间也就越多，别忘了这一点。
# FAISS 教程
#
# IndexIVFFlat
#
# 先聚类再搜索，可以加快检索速度 先将xb中的数据进行聚类（聚类的数目是超参），nlist: 聚类的数目 nprobe: 在多少个聚类中进行搜索，默认为1, nprobe越大，结果越精确，但是速度越慢
# IndexIVFPQ
#
# 基于乘积量化（product quantizers）对存储向量进行压缩，节省存储空间 m：乘积量化中，将原来的向量维度平均分成多少份，d必须为m的整数倍 bits: 每个子向量用多少个bits表示
# Milvus 百万向量搜索（SIFT1B）
#
# 部分插入检索代码
# faiss
#
# def build_index(encoder_conf, index_file_name, title_list, para_list):
#     dual_encoder = rocketqa.load_model(**encoder_conf)
#     para_embs = dual_encoder.encode_para(para=para_list, title=title_list)
#     para_embs = np.array(list(para_embs))
#
#     print("Building index with Faiss...")
#     indexer = faiss.IndexFlatIP(768)
#     indexer.add(para_embs.astype('float32'))
#     faiss.write_index(indexer, index_file_name)
# class FaissTool():
#     """
#     Faiss index tools
#     """
#     def __init__(self, text_filename, index_filename):
#         self.engine = faiss.read_index(index_filename)
#         self.id2text = []
#         for line in open(text_filename):
#             self.id2text.append(line.strip())
#
#     def search(self, q_embs, topk=5):
#         res_dist, res_pid = self.engine.search(q_embs, topk)
#         result_list = []
#         for i in range(topk):
#             result_list.append(self.id2text[res_pid[0][i]])
#         return result_list
# milvus_util
#
# import sys
# sys.path.append('./')
#
# from milvus import *
# from utils.config import MILVUS_HOST, MILVUS_PORT, collection_param, index_type, index_param
# from utils.config import top_k, search_param
#
#
# class VecToMilvus:
#     def __init__(self):
#         self.client = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)
#
#     def has_collection(self, collection_name):
#         try:
#             status, ok = self.client.has_collection(collection_name)
#             return ok
#         except Exception as e:
#             print("Milvus has_table error:", e)
#
#     def creat_collection(self, collection_name):
#         try:
#             collection_param["collection_name"] = collection_name
#             status = self.client.create_collection(collection_param)
#             print(status)
#             return status
#         except Exception as e:
#             print("Milvus create collection error:", e)
#
#     def create_index(self, collection_name):
#         try:
#             status = self.client.create_index(collection_name, index_type, index_param)
#             print(status)
#             return status
#         except Exception as e:
#             print("Milvus create index error:", e)
#
#     def has_partition(self, collection_name, partition_tag):
#         try:
#             status, ok = self.client.has_partition(collection_name, partition_tag)
#             return ok
#         except Exception as e:
#             print("Milvus has partition error: ", e)
#
#     def delete_partition(self, collection_name, partition_tag):
#         try:
#             status = self.client.drop_collection(collection_name)
#             return status
#         except Exception as e:
#             print("Milvus has partition error: ", e)
#
#     def create_partition(self, collection_name, partition_tag):
#         try:
#             status = self.client.create_partition(collection_name, partition_tag)
#             print("create partition {} successfully".format(partition_tag))
#             return status
#         except Exception as e:
#             print("Milvus create partition error: ", e)
#
#     def insert(self, vectors, collection_name, ids=None, partition_tag=None):
#         try:
#             if not self.has_collection(collection_name):
#                 self.creat_collection(collection_name)
#                 self.create_index(collection_name)
#                 print("collection info: {}".format(self.client.get_collection_info(collection_name)[1]))
#             if (partition_tag is not None) and (not self.has_partition(collection_name, partition_tag)):
#                 self.create_partition(collection_name, partition_tag)
#             status, ids = self.client.insert(
#                 collection_name=collection_name, records=vectors, ids=ids, partition_tag=partition_tag
#             )
#             self.client.flush([collection_name])
#             print(
#                 "Insert {} entities, there are {} entities after insert data.".format(
#                     len(ids), self.client.count_entities(collection_name)[1]
#                 )
#             )
#             return status, ids
#         except Exception as e:
#             print("Milvus insert error:", e)
#
#
# class RecallByMilvus:
#     def __init__(self):
#         self.client = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)
#
#     def search(self, vectors, collection_name, partition_tag=None):
#         try:
#             status, results = self.client.search(
#                 collection_name=collection_name,
#                 query_records=vectors,
#                 top_k=top_k,
#                 params=search_param,
#                 partition_tag=partition_tag,
#             )
#             return status, results
#         except Exception as e:
#             print("Milvus recall error: ", e)
# config
#
# from milvus import MetricType, IndexType
#
# MILVUS_HOST = "10.0.98.28"
# MILVUS_PORT = 19530
#
# output_emb_size = 256
#
# collection_param = {
#     "dimension": output_emb_size if output_emb_size > 0 else 768,
#     "index_file_size": 256,
#     "metric_type": MetricType.L2,
# }
#
# index_type = IndexType.FLAT
# index_param = {"nlist": 1000}
#
# top_k = 5
# search_param = {"nprobe": 5}
#
# # collection_name = "label_text"
# collection_name = "serving_3_frame"
# partition_tag = "partition_2"
# vector_insert
#
# import argparse
#
# from tqdm import tqdm
# import numpy as np
#
# from milvus_util import VecToMilvus
# from config import collection_name, partition_tag
#
# # yapf: disable
# parser = argparse.ArgumentParser()
# parser.add_argument("--vector_path", type=str, default='./data/3/label_embedding.npy',
#     help="feature file path.")
#
# args = parser.parse_args()
# # yapf: enable
#
#
# def vector_insert(file_path):
#     embeddings = np.load(file_path)
#     print(embeddings.shape)
#     embedding_ids = [i for i in range(embeddings.shape[0])]
#     print(len(embedding_ids))
#     client = VecToMilvus()
#
#     if client.has_partition(collection_name, partition_tag):
#         client.delete_partition(collection_name, partition_tag)
#     data_size = len(embedding_ids)
#     batch_size = 50000
#     for i in tqdm(range(0, data_size, batch_size)):
#         cur_end = i + batch_size
#         if cur_end > data_size:
#             cur_end = data_size
#         batch_emb = embeddings[np.arange(i, cur_end)]
#         status, ids = client.insert(
#             collection_name=collection_name,
#             vectors=batch_emb.tolist(),
#             ids=embedding_ids[i : i + batch_size],
#             partition_tag=partition_tag,
#         )
#
#
# if __name__ == "__main__":
#     vector_insert(args.vector_path)