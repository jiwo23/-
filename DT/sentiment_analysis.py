# -*- coding: utf-8 -*-

import re
import os
os.chdir(r"D:\qq\WORK1\Lesson06")
print("cwd =", os.getcwd())
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import jieba

def parse_review_file(file_path, label_is_present=False):
    """
    解析评论文件，提取评论文本和标签。
    """
    # In Python 3, open defaults to 'utf-8' on many systems, but it's safer to be explicit.
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 使用正则表达式匹配<review>标签内的内容
    # S flags makes . match newlines
    reviews_xml = re.findall(r'<review id=".*?"(.*?)>(.*?)</review>', content, re.S)
    
    texts = []
    labels = []

    for review_match in reviews_xml:
        text = review_match[1].strip()
        if not text:
            continue
        texts.append(text)
        if label_is_present:
            # 提取label属性
            label_match = re.search(r'label="(\d)"', review_match[0])
            if label_match:
                labels.append(int(label_match.group(1)))
    return texts, labels

def load_data(train_path, test_path):
    """
    加载训练和测试数据。
    """
    # 加载训练数据
    pos_train_file = os.path.join(train_path, 'cn_sample_data', 'sample.positive.txt')
    neg_train_file = os.path.join(train_path, 'cn_sample_data', 'sample.negative.txt')
    
    pos_texts, _ = parse_review_file(pos_train_file)
    neg_texts, _ = parse_review_file(neg_train_file)

    X_train = pos_texts + neg_texts
    y_train = [1] * len(pos_texts) + [0] * len(neg_texts)

    # 加载测试数据和标签
    test_file = os.path.join(test_path, 'Sentiment Classification with Deep Learning', 'test.label.cn.txt')
    X_test, y_test = parse_review_file(test_file, label_is_present=True)

    return X_train, y_train, X_test, y_test

def chinese_word_cut(text):
    """
    使用jieba进行中文分词。
    """
    return " ".join(jieba.cut(text))

def main():
    """
    主函数，执行情感分析流程。
    """
    # 数据路径
    train_path = 'train/evaltask2_sample_data'
    test_path = 'test mark'

    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data(train_path, test_path)
    print("Data loaded. Train samples: {}, Test samples: {}".format(len(X_train), len(X_test)))

    print("\nSegmenting text...")
    X_train_cut = [chinese_word_cut(text) for text in X_train]
    X_test_cut = [chinese_word_cut(text) for text in X_test]
    print("Segmentation complete.")

    print("\nVectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train_cut)
    X_test_tfidf = vectorizer.transform(X_test_cut)
    print("Vectorization complete.")

    print("\nTraining Decision Tree model...")
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train_tfidf, y_train)
    print("Model training complete.")

    print("\nPredicting on test set...")
    y_pred = clf.predict(X_test_tfidf)
    print("Prediction complete.")

    print("\n--- Product Sentiment Analysis Report ---")
    print("\nClassification Report:\n")
    # In Python 3, strings are unicode by default.
    report = classification_report(y_test, y_pred, target_names=['负面评价', '正面评价'])
    print(report)
    print("---------------------------------------\n")

if __name__ == '__main__':
    main()
