#!/usr/bin/env python
# coding: utf-8

# In[31]:


# 验证基础依赖（Anaconda默认已安装，无需额外安装）
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

print("✅ 所有基础依赖验证通过！")
print(f"numpy版本：{np.__version__}")
print(f"networkx版本：{nx.__version__}")


# In[64]:


import os
import re
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ---------------------- 1. 数据加载：训练集/预测集分离 ----------------------
def load_train_data(train_file_paths):
    """
    加载训练集（sample.positive/negative.txt）：标签固定，仅用于训练
    :param train_file_paths: 字典 {文件路径: 固定标签}
    :return: 训练数据 list[(review_id, text, label)], 训练集统计
    """
    train_data = []
    train_stats = {}
    auto_id = 10000  # 唯一ID起始值

    for file_path, fixed_label in train_file_paths.items():
        if not os.path.exists(file_path):
            print(f"⚠️ 训练文件 {os.path.basename(file_path)} 不存在，跳过")
            train_stats[file_path] = {"total": 0, "valid": 0}
            continue

        # 读取文件
        try:
            with open(file_path, "rb") as f:
                content = f.read().decode("utf-8", errors="ignore")
            print(f"\n✅ 读取训练文件：{os.path.basename(file_path)}")
        except Exception as e:
            print(f"❌ 读取 {os.path.basename(file_path)} 失败：{str(e)}")
            train_stats[file_path] = {"total": 0, "valid": 0}
            continue

        # 提取评论内容（仅匹配<review>标签）
        content_pattern = r'<review[^>]*>(.*?)</review>'
        content_matches = re.findall(content_pattern, content, re.DOTALL)
        valid_count = 0

        # 处理每条评论（标签固定为文件对应的正/负面）
        for idx, text in enumerate(content_matches):
            # 清理文本
            clean_text = re.sub(r'<[^>]+>', '', text.strip())
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            if len(clean_text) < 5:
                continue
            # 生成唯一ID
            review_id = auto_id + idx
            # 标签固定（positive=1，negative=0）
            train_data.append((review_id, clean_text, fixed_label))
            valid_count += 1

        # 记录统计
        train_stats[file_path] = {
            "total": len(content_matches),
            "valid": valid_count
        }
        label_desc = "正面(1)" if fixed_label == 1 else "负面(0)"
        print(f"🔧 {os.path.basename(file_path)} 提取：有效评论 {valid_count} 条 | 标签固定为 {label_desc}")

    # 训练集整体统计
    total_train = len(train_data)
    total_pos = sum([1 for _, _, lab in train_data if lab == 1])
    total_neg = total_train - total_pos
    print(f"\n📊 训练集最终统计：")
    print(f"总有效评论数：{total_train} 条 | 正面(1) {total_pos} 条 | 负面(0) {total_neg} 条")

    # 检查训练集有效性（必须包含正负两类）
    if total_pos == 0 or total_neg == 0:
        raise ValueError("❌ 训练集必须同时包含正面和负面评论！请检查sample.positive/negative.txt文件")

    return train_data, train_stats

def load_predict_data(predict_file_path):
    """
    加载预测集（test.en.txt）：无标签，仅用于预测
    :param predict_file_path: 预测文件路径
    :return: 预测数据 list[(review_id, text)], 预测集统计
    """
    predict_data = []
    auto_id = 20000  # 预测集ID起始值（与训练集区分）

    if not os.path.exists(predict_file_path):
        raise FileNotFoundError(f"❌ 预测文件 {predict_file_path} 不存在！")

    # 读取文件
    try:
        with open(predict_file_path, "rb") as f:
            content = f.read().decode("utf-8", errors="ignore")
        print(f"\n✅ 读取预测文件：{os.path.basename(predict_file_path)}")
    except Exception as e:
        raise ValueError(f"❌ 读取预测文件失败：{str(e)}")

    # 提取评论内容
    content_pattern = r'<review[^>]*>(.*?)</review>'
    content_matches = re.findall(content_pattern, content, re.DOTALL)
    valid_count = 0

    # 处理每条评论（无标签）
    for idx, text in enumerate(content_matches):
        clean_text = re.sub(r'<[^>]+>', '', text.strip())
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        if len(clean_text) < 5:
            continue
        review_id = auto_id + idx
        predict_data.append((review_id, clean_text))  # 无标签
        valid_count += 1

    # 统计
    predict_stats = {
        "total": len(content_matches),
        "valid": valid_count
    }
    print(f"🔧 {os.path.basename(predict_file_path)} 提取：有效评论 {valid_count} 条 | 无标签（待预测）")

    if valid_count == 0:
        raise ValueError("❌ 预测集无有效评论！请检查test.en.txt文件格式")

    return predict_data, predict_stats

# ---------------------- 2. GNN特征构建（适配训练/预测） ----------------------
def graph_convolution_aggregation(graph, node_features, alpha=0.8):
    """图卷积聚合：核心逻辑"""
    if len(graph.nodes) == 0 or node_features.shape[0] == 0:
        return np.array([])
    
    num_nodes = len(graph.nodes)
    new_features = np.zeros_like(node_features)
    graph_nodes = sorted(list(graph.nodes))
    
    for idx, node_id in enumerate(graph_nodes):
        self_feat = node_features[idx]
        neighbors = list(graph.neighbors(node_id))
        neighbor_indices = [graph_nodes.index(n) for n in neighbors if n in graph_nodes]
        
        if neighbor_indices:
            neighbor_feat = node_features[neighbor_indices].mean(axis=0)
        else:
            neighbor_feat = np.zeros_like(self_feat)
        
        new_features[idx] = alpha * self_feat + (1 - alpha) * neighbor_feat
    return new_features

def build_gnn_features(texts, tfidf_model=None, fit_tfidf=True, vocab_size=5000):
    """
    构建GNN特征
    :param texts: 文本列表
    :param tfidf_model: 训练好的TF-IDF模型（预测时传入）
    :param fit_tfidf: 是否训练TF-IDF（训练集=True，预测集=False）
    :return: GNN特征矩阵, TF-IDF模型（仅fit_tfidf=True时返回）
    """
    if not texts:
        raise ValueError("❌ 无文本数据，无法构建GNN特征")

    # TF-IDF特征
    if fit_tfidf:
        tfidf = TfidfVectorizer(max_features=vocab_size, stop_words='english', max_df=0.95)
        tfidf_feat = tfidf.fit_transform(texts).toarray()
    else:
        if tfidf_model is None:
            raise ValueError("❌ 预测时必须传入训练好的TF-IDF模型")
        tfidf_feat = tfidf_model.transform(texts).toarray()

    # 构建文档相似度图（分批计算，避免内存溢出）
    graph = nx.Graph()
    num_docs = len(texts)
    graph.add_nodes_from(range(num_docs))
    
    if num_docs > 0:
        from sklearn.metrics.pairwise import cosine_similarity
        batch_size = 100
        for i in range(0, num_docs, batch_size):
            end_idx = min(i + batch_size, num_docs)
            sim_batch = cosine_similarity(tfidf_feat[i:end_idx], tfidf_feat)
            # 仅保留高相似度边
            for j in range(end_idx - i):
                for k in range(j + 1, num_docs):
                    if sim_batch[j][k] > 0.3:
                        graph.add_edge(i + j, k, weight=sim_batch[j][k])

    # 图卷积聚合
    gnn_feat = graph_convolution_aggregation(graph, tfidf_feat)

    if fit_tfidf:
        return gnn_feat, tfidf
    else:
        return gnn_feat

# ---------------------- 3. 模型训练+评估指标（核心新增） ----------------------
def train_and_evaluate_model(train_gnn_feat, train_labels, val_size=0.2):
    """
    训练模型并输出评估指标（准确率、精确率、召回率、F1）
    :param train_gnn_feat: 训练集GNN特征
    :param train_labels: 训练集标签
    :param val_size: 验证集比例
    :return: 训练好的模型、TF-IDF模型
    """
    # 拆分训练集/验证集（用于评估）
    train_idx, val_idx = train_test_split(
        np.arange(len(train_gnn_feat)), 
        test_size=val_size, 
        random_state=42, 
        stratify=train_labels
    )
    train_feat_split = train_gnn_feat[train_idx]
    train_labels_split = train_labels[train_idx]
    val_feat_split = train_gnn_feat[val_idx]
    val_labels_split = train_labels[val_idx]

    # 训练模型
    print("\n🚀 开始训练GNN模型（仅使用sample.positive/negative.txt）...")
    model = LogisticRegression(max_iter=5000, class_weight='balanced', C=0.5)
    model.fit(train_feat_split, train_labels_split)
    print("✅ 模型训练完成！")

    # 验证集预测
    val_pred = model.predict(val_feat_split)

    # 计算评估指标
    accuracy = accuracy_score(val_labels_split, val_pred)
    precision = precision_score(val_labels_split, val_pred, zero_division=0)
    recall = recall_score(val_labels_split, val_pred, zero_division=0)
    f1 = f1_score(val_labels_split, val_pred, zero_division=0)
    conf_mat = confusion_matrix(val_labels_split, val_pred)

    # 输出评估报告
    print("\n" + "="*80)
    print("📊 模型训练评估报告（验证集）")
    print("="*80)
    print(f"验证集总量：{len(val_labels_split)} 条")
    print(f"准确率（Accuracy）：{accuracy:.4f}")
    print(f"精确率（Precision）：{precision:.4f}")
    print(f"召回率（Recall）：{recall:.4f}")
    print(f"F1分数（F1-Score）：{f1:.4f}")

    # 输出混淆矩阵
    print("\n混淆矩阵（行=真实标签，列=预测标签）：")
    print("                预测负面(0)      预测正面(1)")
    print(f"真实负面(0)      {conf_mat[0][0]:^12d}      {conf_mat[0][1]:^12d}")
    print(f"真实正面(1)      {conf_mat[1][0]:^12d}      {conf_mat[1][1]:^12d}")

    return model

# ---------------------- 4. 全流程执行 ----------------------
if __name__ == "__main__":
    # 配置文件路径（根据实际路径修改）
    TRAIN_FILES = {
        "sample.positive.txt": 1,  # 固定正面标签
        "sample.negative.txt": 0   # 固定负面标签
    }
    PREDICT_FILE = "test.en.txt"

    try:
        # 1. 加载训练集（仅sample.positive/negative.txt）
        train_data, train_stats = load_train_data(TRAIN_FILES)
        
        # 2. 加载预测集（仅test.en.txt，无标签）
        predict_data, predict_stats = load_predict_data(PREDICT_FILE)

        # 3. 提取训练集特征和标签
        train_texts = [x[1] for x in train_data]
        train_labels = np.array([x[2] for x in train_data])

        # 4. 构建训练集GNN特征（训练TF-IDF）
        train_gnn_feat, tfidf_model = build_gnn_features(
            train_texts, fit_tfidf=True, vocab_size=5000
        )

        # 5. 训练模型并输出评估指标（核心新增）
        model = train_and_evaluate_model(train_gnn_feat, train_labels, val_size=0.2)

        # 6. 构建预测集GNN特征（复用训练好的TF-IDF）
        predict_texts = [x[1] for x in predict_data]
        predict_gnn_feat = build_gnn_features(
            predict_texts, tfidf_model=tfidf_model, fit_tfidf=False
        )

        print("\n🎉 全流程完成！")
        print("- 训练集评估指标已输出（准确率/精确率/召回率/F1）")

    except Exception as e:
        print(f"\n❌ 执行失败：{str(e)}")


# In[ ]:




