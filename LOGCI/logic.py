#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install jieba scikit-learn numpy


# In[7]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
商品评价情感分类项目 - 完整代码
逻辑回归算法实现中文情感分类
修复BOM和XML格式问题
"""

import os
import re
import sys
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def remove_bom(content):
    """
    移除UTF-8 BOM字符
    """
    if content.startswith('\ufeff'):
        return content[1:]
    return content

def parse_xml_content(xml_content):
    """
    解析XML格式的评论数据
    支持两种格式:
    1. 测试标签格式: <review id="91" label="1">文本</review>
    2. 测试数据格式: <review id="91">文本</review>
    
    返回: (id列表, 文本列表, 标签列表[可选])
    """
    ids = []
    texts = []
    labels = []
    
    # 使用正则表达式匹配review标签
    # 匹配格式: <review id="数字" label="数字">文本</review> 或 <review id="数字">文本</review>
    pattern = r'<review\s+id="(\d+)"(?:\s+label="(\d+)")?>(.*?)</review>'
    matches = re.findall(pattern, xml_content, re.DOTALL)
    
    print(f"找到 {len(matches)} 个review标签")
    
    for match in matches:
        review_id = match[0]
        label = match[1]
        text = match[2].strip()
        
        # 清理文本中的换行符和多余空格
        text = re.sub(r'\s+', ' ', text)
        
        ids.append(int(review_id))
        texts.append(text)
        
        if label:  # 如果有标签
            labels.append(int(label))
    
    return ids, texts, labels if labels else None

def parse_training_file(file_path, is_negative=True):
    """
    解析训练文件，支持XML格式
    返回: 文本列表
    """
    texts = []
    
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return texts
    
    print(f"读取训练文件: {file_path}")
    
    try:
        # 读取整个文件内容
        with open(file_path, 'r', encoding='utf-8-sig') as f:  # 使用utf-8-sig自动处理BOM
            content = f.read()
        
        # 移除BOM字符（如果还有）
        content = remove_bom(content)
        
        # 检查内容是否为空
        if not content.strip():
            print(f"警告: 文件 {file_path} 为空")
            return texts
        
        # 检查是否为XML格式（包含<review>标签）
        if '<review' in content and '</review>' in content:
            print("检测到XML格式，使用XML解析器...")
            ids, file_texts, _ = parse_xml_content(content)
            texts.extend(file_texts)
            
            # 显示一些统计信息
            if ids:
                print(f"ID范围: {min(ids)} - {max(ids)}")
                # 检查是否有重复ID
                if len(ids) != len(set(ids)):
                    print(f"警告: 有重复的ID，共 {len(ids)} 条评论，{len(set(ids))} 个唯一ID")
        else:
            print("检测到纯文本格式，按行读取...")
            # 按行读取，跳过空行
            lines = content.split('\n')
            valid_lines = 0
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):  # 跳过注释行
                    # 清理文本
                    line = re.sub(r'\s+', ' ', line)
                    texts.append(line)
                    valid_lines += 1
            print(f"从 {file_path} 读取到 {valid_lines} 行有效数据")
    
    except UnicodeDecodeError:
        # 尝试用gbk编码读取
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                content = f.read()
            
            content = remove_bom(content)
            
            if '<review' in content and '</review>' in content:
                print("检测到XML格式(GBK编码)，使用XML解析器...")
                ids, file_texts, _ = parse_xml_content(content)
                texts.extend(file_texts)
                
                if ids:
                    print(f"ID范围: {min(ids)} - {max(ids)}")
            else:
                print("检测到纯文本格式(GBK编码)，按行读取...")
                lines = content.split('\n')
                valid_lines = 0
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        line = re.sub(r'\s+', ' ', line)
                        texts.append(line)
                        valid_lines += 1
                print(f"从 {file_path} 读取到 {valid_lines} 行有效数据")
        
        except Exception as e:
            print(f"读取文件 {file_path} 时出错(GBK): {e}")
            return texts
    
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return texts
    
    print(f"从 {file_path} 读取到 {len(texts)} 条评论")
    
    # 显示前3条评论作为示例
    if texts:
        print("示例评论（前3条）:")
        for i in range(min(3, len(texts))):
            preview = texts[i][:80] + "..." if len(texts[i]) > 80 else texts[i]
            print(f"  {i+1}. {preview}")
    
    return texts

def load_training_data():
    """
    加载训练数据
    支持XML格式和纯文本格式
    """
    neg_file = "sample.negative.txt"
    pos_file = "sample.positive.txt"
    
    texts = []
    labels = []
    
    # 检查文件是否存在
    if not os.path.exists(neg_file):
        print(f"错误: 找不到负面训练文件 {neg_file}")
        print("请确保文件在当前目录下")
        sys.exit(1)
    
    if not os.path.exists(pos_file):
        print(f"错误: 找不到正面训练文件 {pos_file}")
        print("请确保文件在当前目录下")
        sys.exit(1)
    
    print("=" * 60)
    print("加载训练数据")
    print("=" * 60)
    
    # 加载负面评论
    print(f"\n处理负面评论文件: {neg_file}")
    negative_texts = parse_training_file(neg_file, is_negative=True)
    
    # 检查读取的数据
    if len(negative_texts) > 6000:
        print(f"警告: 负面评论读取到 {len(negative_texts)} 条，可能过多！")
        print("可能原因:")
        print("1. 文件格式不是预期的XML格式")
        print("2. 文件包含空行或其他格式问题")
        print("3. 文件编码问题")
        print("建议检查文件前几行内容")
    
    texts.extend(negative_texts)
    labels.extend([0] * len(negative_texts))  # 负面标签为0
    
    # 加载正面评论
    print(f"\n处理正面评论文件: {pos_file}")
    positive_texts = parse_training_file(pos_file, is_negative=False)
    
    if len(positive_texts) > 6000:
        print(f"警告: 正面评论读取到 {len(positive_texts)} 条，可能过多！")
        print("建议检查文件格式")
    
    texts.extend(positive_texts)
    labels.extend([1] * len(positive_texts))  # 正面标签为1
    
    # 统计信息
    print("\n训练数据统计:")
    print(f"总评论数: {len(texts)}")
    print(f"负面评论数: {labels.count(0)}")
    print(f"正面评论数: {labels.count(1)}")
    
    return texts, labels

def load_test_data(test_file, has_labels=False):
    """
    加载测试数据
    支持XML格式: <review id="91">文本</review> 或 <review id="91" label="1">文本</review>
    """
    if not os.path.exists(test_file):
        print(f"错误: 找不到测试文件 {test_file}")
        return [], [], []
    
    print(f"加载测试数据: {test_file}")
    
    try:
        # 读取整个文件内容，使用utf-8-sig自动处理BOM
        with open(test_file, 'r', encoding='utf-8-sig') as f:
            content = f.read()
        
        # 移除BOM字符（如果还有）
        content = remove_bom(content)
        
    except UnicodeDecodeError:
        # 尝试用gbk编码读取
        try:
            with open(test_file, 'r', encoding='gbk') as f:
                content = f.read()
            content = remove_bom(content)
        except:
            print(f"无法读取文件 {test_file}，请检查编码")
            return [], [], []
    
    # 解析XML内容
    ids, texts, labels = parse_xml_content(content)
    
    if has_labels:
        if labels is None:
            print("警告: 文件应该有标签但没有找到标签")
            return ids, texts, []
        print(f"找到 {len(ids)} 条评论，{len(labels)} 个标签")
    else:
        print(f"找到 {len(ids)} 条评论")
    
    return ids, texts, labels

def preprocess_chinese_text(text):
    """
    中文文本预处理函数
    输入: 原始文本
    输出: 预处理后的文本
    """
    if not isinstance(text, str):
        return ""
    
    # 1. 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 2. 去除特殊字符和数字（保留中文和英文字母）
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z\s]', '', text)
    
    # 3. 分词
    words = jieba.lcut(text)
    
    # 4. 去除停用词
    stop_words = set([
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', 
        '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', 
        '你', '会', '着', '没有', '看', '好', '自己', '这', '这个'
    ])
    
    # 只保留非停用词且长度大于1的词
    words = [word for word in words if word not in stop_words and len(word) > 1]
    
    return ' '.join(words)

def save_predictions(ids, predictions, texts, output_file="predictions.txt"):
    """
    保存预测结果为XML格式
    格式: <review id="数字" label="0或1">评论文本</review>
    """
    print(f"\n保存预测结果到 {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for review_id, pred, text in zip(ids, predictions, texts):
            label_str = "1" if pred == 1 else "0"
            f.write(f'<review id="{review_id}" label="{label_str}">\n')
            f.write(f'{text}\n')
            f.write('</review>\n')
    
    print(f"预测结果已保存到 {output_file}")

def evaluate_model(true_labels, pred_labels):
    """
    评估模型性能
    返回precision, recall, F1-score
    """
    if len(true_labels) == 0 or len(pred_labels) == 0:
        print("无法评估: 没有可用的真实标签")
        return 0, 0, 0
    
    precision = precision_score(true_labels, pred_labels, average='binary')
    recall = recall_score(true_labels, pred_labels, average='binary')
    f1 = f1_score(true_labels, pred_labels, average='binary')
    
    return precision, recall, f1

def main():
    """
    主函数 - 执行完整的训练和预测流程
    """
    print("=" * 60)
    print("商品评价情感分类系统")
    print("基于逻辑回归的中文情感分类")
    print("=" * 60)
    
    # ============ 步骤1: 数据加载和预处理 ============
    print("\n步骤1: 数据加载和预处理")
    print("-" * 40)
    
    # 加载训练数据
    train_texts, train_labels = load_training_data()
    
    if len(train_texts) == 0:
        print("错误: 训练数据为空!")
        return
    
    if len(train_texts) != len(train_labels):
        print(f"错误: 文本数量({len(train_texts)})和标签数量({len(train_labels)})不匹配!")
        return
    
    # 检查数据量是否合理
    expected_count = 10000  # 假设正负面各5000
    actual_count = len(train_texts)
    if actual_count > expected_count * 1.2:  # 超过预期20%
        print(f"警告: 读取到 {actual_count} 条评论，可能过多!")
        print("建议检查训练文件格式是否正确")
        print("按回车键继续，或按Ctrl+C中止...")
        try:
            input()
        except:
            print("程序中止")
            return
    
    # 预处理训练文本
    print("\n预处理训练文本...")
    preprocessed_train_texts = []
    processed_count = 0
    
    for i, text in enumerate(train_texts):
        preprocessed_text = preprocess_chinese_text(text)
        preprocessed_train_texts.append(preprocessed_text)
        processed_count += 1
        
        # 显示进度
        if processed_count % 1000 == 0:
            print(f"已预处理 {processed_count}/{len(train_texts)} 条评论")
        
        # 打印前3个样本的预处理结果
        if i < 3:
            print(f"样本 {i+1}:")
            print(f"  原始: {text[:50]}...")
            print(f"  预处理: {preprocessed_text[:50]}...")
            print()
    
    print(f"预处理完成，共处理 {len(preprocessed_train_texts)} 条评论")
    
    # ============ 步骤2: 特征提取 ============
    print("\n步骤2: 特征提取 (TF-IDF)")
    print("-" * 40)
    
    # 创建TF-IDF向量化器
    vectorizer = TfidfVectorizer(
        max_features=5000,  # 限制特征数量
        token_pattern=r'(?u)\b\w+\b'
    )
    
    # 转换训练数据
    print("转换训练数据为TF-IDF特征...")
    X_train = vectorizer.fit_transform(preprocessed_train_texts)
    y_train = np.array(train_labels)
    
    print(f"特征维度: {X_train.shape}")
    
    # ============ 步骤3: 模型训练 ============
    print("\n步骤3: 训练逻辑回归模型")
    print("-" * 40)
    
    # 创建逻辑回归模型
    model = LogisticRegression(
        C=1.0,               # 正则化强度
        max_iter=1000,       # 最大迭代次数
        random_state=42,     # 随机种子
        solver='liblinear'   # 优化算法
    )
    
    # 训练模型
    print("训练模型中...")
    model.fit(X_train, y_train)
    
    print("训练完成!")
    print(f"模型准确率（训练集）: {model.score(X_train, y_train):.4f}")
    
    # ============ 步骤4: 加载和预处理测试数据 ============
    print("\n步骤4: 处理测试数据")
    print("-" * 40)
    
    # 检查测试文件是否存在
    test_file = "test.cn.txt"
    test_label_file = "test.label.cn.txt"
    
    # 优先使用带标签的测试文件进行评估
    if os.path.exists(test_label_file):
        print("找到带标签的测试文件，用于模型评估")
        test_ids, test_texts, true_labels = load_test_data(test_label_file, has_labels=True)
    elif os.path.exists(test_file):
        print("找到测试文件（不带标签），用于预测")
        test_ids, test_texts, true_labels = load_test_data(test_file, has_labels=False)
    else:
        print("未找到测试文件，跳过测试阶段")
        test_ids, test_texts, true_labels = [], [], []
    
    if len(test_texts) == 0:
        print("测试数据为空，跳过测试阶段")
        # 保存模型供后续使用
        joblib.dump(model, "sentiment_model.pkl")
        joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
        print("\n模型已保存，可用于后续预测")
        return
    
    # 预处理测试文本
    print("预处理测试文本...")
    preprocessed_test_texts = []
    for text in test_texts:
        preprocessed_test_texts.append(preprocess_chinese_text(text))
    
    # 转换测试数据
    X_test = vectorizer.transform(preprocessed_test_texts)
    
    # ============ 步骤5: 预测 ============
    print("\n步骤5: 预测测试集情感")
    print("-" * 40)
    
    # 预测
    predictions = model.predict(X_test)
    
    # 统计预测结果
    num_positive = sum(predictions == 1)
    num_negative = sum(predictions == 0)
    
    print(f"预测统计:")
    print(f"  正面评论: {num_positive} ({num_positive/len(predictions)*100:.1f}%)")
    print(f"  负面评论: {num_negative} ({num_negative/len(predictions)*100:.1f}%)")
    
    # ============ 步骤6: 评估模型 (如果有真实标签) ============
    if true_labels and len(true_labels) == len(predictions):
        print("\n步骤6: 模型评估")
        print("-" * 40)
        
        # 计算评估指标
        precision, recall, f1 = evaluate_model(true_labels, predictions)
        
        print(f"评估结果:")
        print(f"  Precision (精确率): {precision:.4f}")
        print(f"  Recall (召回率): {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        # 详细分类报告
        print("\n详细分类报告:")
        print(classification_report(true_labels, predictions, 
                                  target_names=['negative', 'positive']))
        
        # 混淆矩阵
        print("混淆矩阵:")
        cm = confusion_matrix(true_labels, predictions)
        print(f"       预测负面  预测正面")
        print(f"真实负面  {cm[0,0]:8d}  {cm[0,1]:8d}")
        print(f"真实正面  {cm[1,0]:8d}  {cm[1,1]:8d}")
        
        # 计算准确率
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        print(f"\n准确率: {accuracy:.4f}")
        
        # 保存评估指标
        with open("evaluation_results.txt", 'w', encoding='utf-8') as f:
            f.write(f"模型评估结果\n")
            f.write(f"==============\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1-Score: {f1:.4f}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"\n混淆矩阵:\n")
            f.write(f"预测负面  预测正面\n")
            f.write(f"真实负面  {cm[0,0]:8d}  {cm[0,1]:8d}\n")
            f.write(f"真实正面  {cm[1,0]:8d}  {cm[1,1]:8d}\n")
        
        print(f"\n评估结果已保存到 evaluation_results.txt")
    
    # ============ 步骤7: 保存结果 ============
    print("\n步骤7: 保存结果")
    print("-" * 40)
    
    # 保存预测结果
    save_predictions(test_ids, predictions, test_texts, "predictions.txt")
    
    # 保存模型和向量化器
    try:
        joblib.dump(model, "sentiment_model.pkl")
        joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
        print(f"模型已保存到 sentiment_model.pkl")
        print(f"向量化器已保存到 tfidf_vectorizer.pkl")
    except Exception as e:
        print(f"保存模型时出错: {e}")
    
    # ============ 步骤8: 显示预测示例 ============
    print("\n步骤8: 预测示例")
    print("-" * 40)
    
    # 显示前5个预测结果
    print("前5个预测结果:")
    for i in range(min(5, len(test_texts))):
        pred_label = "正面" if predictions[i] == 1 else "负面"
        
        # 如果有真实标签，显示对比
        if true_labels and len(true_labels) > i:
            true_label = "正面" if true_labels[i] == 1 else "负面"
            correct = "✓" if predictions[i] == true_labels[i] else "✗"
            print(f"{i+1}. ID: {test_ids[i]} | 预测: {pred_label} | 实际: {true_label} {correct}")
        else:
            print(f"{i+1}. ID: {test_ids[i]} | 预测: {pred_label}")
        
        # 显示部分文本
        text_preview = test_texts[i][:50] + "..." if len(test_texts[i]) > 50 else test_texts[i]
        print(f"   文本: {text_preview}")
        print()
    
    print("=" * 60)
    print("程序执行完成!")
    print("=" * 60)
    
    # 显示生成的文件
    print("\n生成的文件:")
    print("  - predictions.txt: 测试集的预测结果")
    if true_labels:
        print("  - evaluation_results.txt: 模型评估结果")
    print("  - sentiment_model.pkl: 训练好的模型")
    print("  - tfidf_vectorizer.pkl: TF-IDF向量化器")
    
    print("\n使用方法:")
    print("  1. 确保训练文件是XML格式: <review id=\"数字\">评论内容</review>")
    print("  2. 将评论文本放入test.cn.txt（XML格式）")
    print("  3. 运行本程序进行预测")
    print("  4. 查看predictions.txt获取预测结果")

def check_file_format():
    """
    检查文件格式，帮助诊断问题
    """
    print("检查文件格式...")
    
    files_to_check = ["sample.negative.txt", "sample.positive.txt"]
    
    for file_name in files_to_check:
        if os.path.exists(file_name):
            print(f"\n检查文件: {file_name}")
            try:
                # 读取前几行
                with open(file_name, 'r', encoding='utf-8-sig') as f:
                    lines = []
                    for i in range(5):
                        line = f.readline()
                        if not line:
                            break
                        lines.append(line.strip())
                
                print(f"前5行内容:")
                for i, line in enumerate(lines):
                    print(f"  行{i+1}: {line[:100]}{'...' if len(line) > 100 else ''}")
                
                # 检查格式
                if lines and lines[0].startswith('<review'):
                    print("  格式: XML格式")
                else:
                    print("  格式: 纯文本格式")
                    
            except Exception as e:
                print(f"  读取文件时出错: {e}")

if __name__ == "__main__":
    # 检查依赖库
    try:
        import sklearn
        import jieba
    except ImportError as e:
        print(f"错误: 缺少必要的依赖库 - {e}")
        print("请运行以下命令安装依赖:")
        print("pip install scikit-learn jieba numpy")
        sys.exit(1)
    
    # 首先检查文件格式
    check_file_format()
    
    # 运行主程序
    main()


# In[ ]:




