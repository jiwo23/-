#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install jieba pandas numpy scikit-learn matplotlib seaborn')


# In[18]:


# ==================== 1. 导入必要的库 ====================
import pandas as pd
import numpy as np
import jieba
import re
import os
import warnings
warnings.filterwarnings('ignore')

# 机器学习相关库
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# 可视化库
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

print("所有必要的库已导入")
print("=" * 60)

# ==================== 2. XML文件解析函数 ====================
def parse_xml_file(file_path):
    """
    解析XML格式的评论文件
    格式: <review id="10">评论内容</review>
    返回: 评论列表
    """
    print(f"解析XML文件: {file_path}")
    reviews = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # 尝试使用GBK编码
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                content = f.read()
        except:
            print(f"  错误：无法读取文件 {file_path}，请检查编码格式")
            return []
    
    # 使用正则表达式提取所有review标签内的内容
    # 匹配模式: <review id="数字">内容</review>
    pattern = r'<review[^>]*>(.*?)</review>'
    matches = re.findall(pattern, content, re.DOTALL)  # re.DOTALL使.匹配包括换行符在内的所有字符
    
    for i, match in enumerate(matches):
        # 清理内容：去除前后空白，合并换行符
        text = match.strip()
        # 将换行符替换为空格
        text = re.sub(r'\s+', ' ', text)
        reviews.append(text)
    
    print(f"  找到 {len(reviews)} 条评论")
    
    # 显示前3条评论作为示例
    if len(reviews) >= 3:
        print("  前3条评论示例:")
        for i in range(min(3, len(reviews))):
            print(f"    评论{i+1}: {reviews[i][:50]}...")
    
    return reviews

def parse_xml_label_file(file_path):
    """
    解析XML格式的标签文件
    格式: <review id="0"  label="0">
    返回: 标签列表 (0或1)
    """
    print(f"解析XML标签文件: {file_path}")
    labels = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # 尝试使用GBK编码
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                content = f.read()
        except:
            print(f"  错误：无法读取文件 {file_path}，请检查编码格式")
            return []
    
    # 使用正则表达式提取所有review标签的label属性
    # 匹配模式: <review id="数字" label="数字">
    pattern = r'<review[^>]*label\s*=\s*"(\d+)"[^>]*>'
    matches = re.findall(pattern, content)
    
    for i, match in enumerate(matches):
        try:
            label = int(match)
            if label not in [0, 1]:
                label = label % 2  # 如果不是0或1，取模2
                print(f"  警告：第{i+1}个标签值 {match} 不是0或1，已转换为 {label}")
            labels.append(label)
        except ValueError:
            # 如果转换失败，设为默认值0
            labels.append(0)
            print(f"  错误：第{i+1}个标签无法转换，设为默认值0")
    
    # 如果没有找到label属性，尝试其他方法
    if len(labels) == 0:
        print("  未找到label属性，尝试其他提取方法...")
        # 提取所有review标签
        pattern = r'<review[^>]*>'
        matches = re.findall(pattern, content)
        
        for i, match in enumerate(matches):
            # 提取所有数字
            numbers = re.findall(r'\d+', match)
            if numbers:
                try:
                    label = int(numbers[0]) % 2
                    labels.append(label)
                except ValueError:
                    labels.append(0)
            else:
                labels.append(0)
    
    print(f"  找到 {len(labels)} 个标签")
    
    # 显示标签分布
    if labels:
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        print(f"  标签分布: 正面={pos_count} ({pos_count/len(labels):.2%}), 负面={neg_count} ({neg_count/len(labels):.2%})")
    
    return labels

def parse_plain_text_file(file_path):
    """
    解析纯文本文件（每行一条评论）
    返回: 评论列表
    """
    print(f"解析纯文本文件: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        # 尝试使用GBK编码
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                lines = f.readlines()
        except:
            print(f"  错误：无法读取文件 {file_path}，请检查编码格式")
            return []
    
    # 清理每行：去除空白字符
    reviews = [line.strip() for line in lines if line.strip()]
    
    print(f"  找到 {len(reviews)} 条评论")
    
    # 显示前3条评论作为示例
    if len(reviews) >= 3:
        print("  前3条评论示例:")
        for i in range(min(3, len(reviews))):
            print(f"    评论{i+1}: {reviews[i][:50]}...")
    
    return reviews

# ==================== 3. 数据加载函数 ====================
def load_data():
    """
    加载所有数据文件
    自动检测文件格式并选择正确的解析方法
    返回: 训练集和测试集的DataFrame
    """
    print("正在加载数据文件...")
    
    # 定义文件列表
    files = ['sample.positive.txt', 'sample.negative.txt', 'test.cn.txt', 'test.label.cn.txt']
    missing_files = []
    
    # 检查文件是否存在
    for file in files:
        if not os.path.exists(file):
            missing_files.append(file)
            print(f"  ✗ 未找到: {file}")
        else:
            print(f"  ✓ 找到: {file}")
    
    # 如果有文件缺失，提示用户
    if missing_files:
        print(f"\n错误：以下文件未找到: {missing_files}")
        return None, None
    
    try:
        # 检测文件格式并选择正确的解析方法
        def detect_file_format(file_path):
            """检测文件是XML格式还是纯文本格式"""
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    # 检查是否包含XML标签
                    if first_line.startswith('<review') and '>' in first_line:
                        return 'xml'
                    else:
                        # 检查整个文件是否有XML标签
                        f.seek(0)
                        content = f.read(1000)  # 读取前1000个字符
                        if '<review' in content and '</review>' in content:
                            return 'xml'
                        else:
                            return 'text'
            except UnicodeDecodeError:
                # 尝试GBK编码
                try:
                    with open(file_path, 'r', encoding='gbk') as f:
                        first_line = f.readline().strip()
                        if first_line.startswith('<review') and '>' in first_line:
                            return 'xml'
                        else:
                            f.seek(0)
                            content = f.read(1000)
                            if '<review' in content and '</review>' in content:
                                return 'xml'
                            else:
                                return 'text'
                except:
                    return 'text'  # 默认按文本处理
        
        print("\n检测文件格式...")
        
        # 1. 检测并读取正面训练集
        positive_format = detect_file_format('sample.positive.txt')
        print(f"  sample.positive.txt 格式: {positive_format}")
        
        if positive_format == 'xml':
            positive_reviews = parse_xml_file('sample.positive.txt')
        else:
            positive_reviews = parse_plain_text_file('sample.positive.txt')
        
        # 2. 检测并读取负面训练集
        negative_format = detect_file_format('sample.negative.txt')
        print(f"  sample.negative.txt 格式: {negative_format}")
        
        if negative_format == 'xml':
            negative_reviews = parse_xml_file('sample.negative.txt')
        else:
            negative_reviews = parse_plain_text_file('sample.negative.txt')
        
        # 3. 检测并读取测试集
        test_format = detect_file_format('test.cn.txt')
        print(f"  test.cn.txt 格式: {test_format}")
        
        if test_format == 'xml':
            test_reviews = parse_xml_file('test.cn.txt')
        else:
            test_reviews = parse_plain_text_file('test.cn.txt')
        
        # 4. 读取测试集标签 (总是按XML格式解析，因为您提到是XML格式)
        print(f"  test.label.cn.txt 格式: xml (强制)")
        test_labels = parse_xml_label_file('test.label.cn.txt')
        
        # 检查数据一致性
        if len(test_reviews) != len(test_labels):
            print(f"\n警告：测试集评论数({len(test_reviews)})与标签数({len(test_labels)})不匹配！")
            # 取较小值
            min_len = min(len(test_reviews), len(test_labels))
            test_reviews = test_reviews[:min_len]
            test_labels = test_labels[:min_len]
            print(f"  调整为 {min_len} 条测试数据")
        
        # 创建训练集DataFrame
        train_reviews = positive_reviews + negative_reviews
        train_labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)
        
        train_df = pd.DataFrame({
            'review': train_reviews,
            'label': train_labels
        })
        
        # 创建测试集DataFrame
        test_df = pd.DataFrame({
            'review': test_reviews,
            'label': test_labels
        })
        
        print(f"\n✓ 数据加载完成！")
        print(f"  训练集大小: {len(train_df)} (正面: {len(positive_reviews)}, 负面: {len(negative_reviews)})")
        print(f"  测试集大小: {len(test_df)}")
        
        # 显示标签分布
        print(f"\n📊 标签分布:")
        train_pos = train_df['label'].sum()
        train_neg = len(train_df) - train_pos
        print(f"  训练集 - 正面: {train_pos} ({train_pos/len(train_df):.2%}), 负面: {train_neg} ({train_neg/len(train_df):.2%})")
        
        test_pos = test_df['label'].sum()
        test_neg = len(test_df) - test_pos
        print(f"  测试集 - 正面: {test_pos} ({test_pos/len(test_df):.2%}), 负面: {test_neg} ({test_neg/len(test_df):.2%})")
        
        return train_df, test_df
        
    except Exception as e:
        print(f"\n错误：读取文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# 加载数据
print("\n" + "=" * 60)
print("数据加载")
print("=" * 60)
train_df, test_df = load_data()

# 如果数据加载失败，显示错误信息
if train_df is None or test_df is None:
    print("数据加载失败，请检查文件是否存在和格式是否正确")
    # 退出程序
    raise SystemExit("程序终止：数据加载失败")

# 显示数据前几行
print("\n训练集前5行:")
print(train_df.head())
print("\n测试集前5行:")
print(test_df.head())

# ==================== 4. 数据分析和可视化 ====================
print("\n" + "=" * 60)
print("数据集分析")
print("=" * 60)

# 训练集分析
print("\n📊 训练集统计:")
print(f"  总样本数: {len(train_df)}")
print(f"  正面样本数: {train_df['label'].sum()}")
print(f"  负面样本数: {len(train_df) - train_df['label'].sum()}")
print(f"  正面比例: {train_df['label'].mean():.2%}")
print(f"  负面比例: {(1 - train_df['label'].mean()):.2%}")

# 测试集分析
print("\n📊 测试集统计:")
print(f"  总样本数: {len(test_df)}")
print(f"  正面样本数: {test_df['label'].sum()}")
print(f"  负面样本数: {len(test_df) - test_df['label'].sum()}")
print(f"  正面比例: {test_df['label'].mean():.2%}")
print(f"  负面比例: {(1 - test_df['label'].mean()):.2%}")

# 文本长度分析
train_df['review_length'] = train_df['review'].apply(len)
test_df['review_length'] = test_df['review'].apply(len)

print("\n📊 文本长度统计:")
print("  训练集:")
print(f"    平均长度: {train_df['review_length'].mean():.1f} 字符")
print(f"    最小长度: {train_df['review_length'].min()} 字符")
print(f"    最大长度: {train_df['review_length'].max()} 字符")
print(f"    长度中位数: {train_df['review_length'].median()} 字符")
print("  测试集:")
print(f"    平均长度: {test_df['review_length'].mean():.1f} 字符")
print(f"    最小长度: {test_df['review_length'].min()} 字符")
print(f"    最大长度: {test_df['review_length'].max()} 字符")
print(f"    长度中位数: {test_df['review_length'].median()} 字符")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 训练集标签分布
ax1 = axes[0, 0]
train_counts = train_df['label'].value_counts()
ax1.bar(['负面 (0)', '正面 (1)'], train_counts.values, color=['#FF6B6B', '#4ECDC4'])
ax1.set_title('训练集标签分布', fontsize=14)
ax1.set_ylabel('样本数', fontsize=12)
for i, v in enumerate(train_counts.values):
    ax1.text(i, v + max(train_counts.values)*0.01, str(v), ha='center', fontsize=12)

# 2. 测试集标签分布
ax2 = axes[0, 1]
test_counts = test_df['label'].value_counts()
ax2.bar(['负面 (0)', '正面 (1)'], test_counts.values, color=['#FF6B6B', '#4ECDC4'])
ax2.set_title('测试集标签分布', fontsize=14)
ax2.set_ylabel('样本数', fontsize=12)
for i, v in enumerate(test_counts.values):
    ax2.text(i, v + max(test_counts.values)*0.01, str(v), ha='center', fontsize=12)

# 3. 训练集文本长度分布
ax3 = axes[1, 0]
ax3.hist(train_df['review_length'], bins=30, color='#45B7D1', alpha=0.7, edgecolor='black')
ax3.set_title('训练集文本长度分布', fontsize=14)
ax3.set_xlabel('文本长度 (字符数)', fontsize=12)
ax3.set_ylabel('频率', fontsize=12)
ax3.axvline(train_df['review_length'].mean(), color='red', linestyle='dashed', linewidth=1, label=f'平均: {train_df["review_length"].mean():.1f}')
ax3.legend()

# 4. 测试集文本长度分布
ax4 = axes[1, 1]
ax4.hist(test_df['review_length'], bins=30, color='#96CEB4', alpha=0.7, edgecolor='black')
ax4.set_title('测试集文本长度分布', fontsize=14)
ax4.set_xlabel('文本长度 (字符数)', fontsize=12)
ax4.set_ylabel('频率', fontsize=12)
ax4.axvline(test_df['review_length'].mean(), color='red', linestyle='dashed', linewidth=1, label=f'平均: {test_df["review_length"].mean():.1f}')
ax4.legend()

plt.suptitle('数据集分析', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# ==================== 5. 中文文本预处理 ====================
def chinese_preprocess(text):
    """中文文本预处理"""
    if not isinstance(text, str):
        return ""
    
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 去除URL链接
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # 去除特殊字符，保留中文、英文、数字
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
    
    # 去除多余空白字符
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def chinese_tokenize(text):
    """中文分词"""
    if not text:
        return []
    
    # 使用jieba进行分词
    tokens = jieba.lcut(text)
    
    # 过滤掉单字符（除非是中文单字）
    tokens = [token for token in tokens if len(token) > 1 or '\u4e00' <= token <= '\u9fff']
    
    return tokens

print("\n" + "=" * 60)
print("文本预处理和特征提取")
print("=" * 60)

print("正在进行文本预处理...")

# 应用文本预处理
train_df['cleaned_review'] = train_df['review'].apply(chinese_preprocess)
test_df['cleaned_review'] = test_df['review'].apply(chinese_preprocess)

# 显示预处理前后的示例
print("\n📝 预处理示例:")
if len(train_df) > 0:
    sample_idx = 0
    print("原始评论:", train_df['review'].iloc[sample_idx][:50] + "..." if len(train_df['review'].iloc[sample_idx]) > 50 else train_df['review'].iloc[sample_idx])
    print("预处理后:", train_df['cleaned_review'].iloc[sample_idx][:50] + "..." if len(train_df['cleaned_review'].iloc[sample_idx]) > 50 else train_df['cleaned_review'].iloc[sample_idx])

# ==================== 6. 特征提取 ====================
# 定义分词函数
def tokenizer(text):
    return chinese_tokenize(text)

print("\n正在提取TF-IDF特征...")

# 使用TF-IDF向量化
vectorizer = TfidfVectorizer(
    tokenizer=tokenizer,
    max_features=5000,  # 限制特征数量
    ngram_range=(1, 2),  # 使用单字和双字
    min_df=2,  # 最小文档频率
    max_df=0.9,  # 最大文档频率
    sublinear_tf=True  # 使用子线性TF缩放
)

# 转换训练集和测试集
X_train = vectorizer.fit_transform(train_df['cleaned_review'])
X_test = vectorizer.transform(test_df['cleaned_review'])

y_train = train_df['label'].values
y_test = test_df['label'].values

print(f"✓ 特征提取完成!")
print(f"  训练集特征形状: {X_train.shape}")
print(f"  测试集特征形状: {X_test.shape}")
print(f"  特征数量: {X_train.shape[1]}")

# ==================== 7. SVM模型训练和评估 ====================
print("\n" + "=" * 60)
print("SVM模型训练和评估")
print("=" * 60)

print("正在训练SVM模型...")

# 训练SVM模型
svm_model = SVC(
    kernel='linear',  # 使用线性核，适合文本分类
    C=1.0,  # 正则化参数
    probability=True,  # 启用概率预测
    random_state=42,  # 随机种子
    verbose=False  # 不显示训练过程
)

svm_model.fit(X_train, y_train)
print("✓ SVM模型训练完成!")

# 预测
y_pred = svm_model.predict(X_test)
y_pred_proba = svm_model.predict_proba(X_test)[:, 1]

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n📈 模型性能指标:")
print(f"  准确率 (Accuracy): {accuracy:.4f}")
print(f"  F1分数: {f1:.4f}")

print(f"\n📊 分类报告:")
print(classification_report(y_test, y_pred, target_names=['负面', '正面'], digits=4))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['负面', '正面'], 
            yticklabels=['负面', '正面'])
plt.title('SVM模型 - 混淆矩阵', fontsize=14)
plt.ylabel('真实标签', fontsize=12)
plt.xlabel('预测标签', fontsize=12)
plt.show()

# 显示一些预测示例
print("\n📋 预测示例（前10条测试数据）:")
print("-" * 80)

for i in range(min(10, len(y_test))):
    if len(test_df['review']) > i:
        review_preview = test_df['review'].iloc[i][:40] + "..." if len(test_df['review'].iloc[i]) > 40 else test_df['review'].iloc[i]
        true_label = "正面" if y_test[i] == 1 else "负面"
        pred_label = "正面" if y_pred[i] == 1 else "负面"
        prob = y_pred_proba[i] if i < len(y_pred_proba) else 0
        
        # 标记预测正确/错误
        if true_label == pred_label:
            marker = "✓"
            color = "\033[92m"  # 绿色
        else:
            marker = "✗"
            color = "\033[91m"  # 红色
        
        print(f"{color}{marker} 样本 {i+1}:")
        print(f"    评论: {review_preview}")
        print(f"    真实: {true_label} | 预测: {pred_label} | 正面概率: {prob:.4f}")
        print("\033[0m" + "-" * 80)

# ==================== 8. 特征重要性分析（修复版）====================
def analyze_feature_importance(model, vectorizer, top_n=20):
    """分析特征重要性"""
    print("\n" + "=" * 60)
    print("特征重要性分析")
    print("=" * 60)
    
    # 检查模型是否有系数属性
    if hasattr(model, 'coef_'):
        # 获取特征名称
        if hasattr(vectorizer, 'get_feature_names_out'):
            feature_names = vectorizer.get_feature_names_out()
        elif hasattr(vectorizer, 'get_feature_names'):
            feature_names = vectorizer.get_feature_names()
        else:
            print("无法获取特征名称")
            return
        
        # 获取系数
        coefficients = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        
        # 检查是否是稀疏矩阵，如果是则转换为密集数组
        from scipy.sparse import issparse
        if issparse(coefficients):
            coefficients = coefficients.toarray().flatten()
            print("  注意：系数是稀疏矩阵，已转换为密集数组")
        else:
            coefficients = coefficients.flatten()
        
        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': coefficients
        })
        
        # 按重要性排序
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        # 显示最重要的特征
        print(f"\n🔝 最重要的正面特征（对正面分类贡献最大，前{top_n}个）:")
        top_positive = feature_importance.head(top_n)
        for i, (idx, row) in enumerate(top_positive.iterrows(), 1):
            print(f"  {i:2d}. {row['feature']:20s} : {row['importance']:.6f}")
        
        print(f"\n🔻 最重要的负面特征（对负面分类贡献最大，前{top_n}个）:")
        top_negative = feature_importance.tail(top_n).iloc[::-1]
        for i, (idx, row) in enumerate(top_negative.iterrows(), 1):
            print(f"  {i:2d}. {row['feature']:20s} : {row['importance']:.6f}")
        
        # 可视化前10个正面和负面特征
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 正面特征
        top_pos = top_positive.head(10)
        axes[0].barh(range(len(top_pos)), top_pos['importance'].values)
        axes[0].set_yticks(range(len(top_pos)))
        axes[0].set_yticklabels(top_pos['feature'].values, fontsize=10)
        axes[0].invert_yaxis()
        axes[0].set_title('最重要的正面特征', fontsize=14)
        axes[0].set_xlabel('系数值', fontsize=12)
        
        # 负面特征
        top_neg = top_negative.head(10)
        axes[1].barh(range(len(top_neg)), top_neg['importance'].values)
        axes[1].set_yticks(range(len(top_neg)))
        axes[1].set_yticklabels(top_neg['feature'].values, fontsize=10)
        axes[1].invert_yaxis()
        axes[1].set_title('最重要的负面特征', fontsize=14)
        axes[1].set_xlabel('系数值', fontsize=12)
        
        plt.suptitle('特征重要性分析', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️  当前模型没有系数属性（可能使用了非线性核）")

# 运行特征重要性分析
analyze_feature_importance(svm_model, vectorizer, top_n=15)

# ==================== 9. 新评论预测 ====================
def predict_new_reviews(model, vectorizer):
    """预测新评论的情感"""
    print("\n" + "=" * 60)
    print("新评论预测")
    print("=" * 60)
    
    # 示例新评论
    new_reviews = [
        "这个产品真的太棒了！质量非常好，使用起来非常方便，强烈推荐！",
        "非常糟糕的购物体验，产品质量差，客服态度也不好，再也不会买了。",
        "还行吧，价格便宜，但质量一般，对得起这个价格。",
        "这是我买过的最好的商品之一，完全超出了我的期望，性价比超高！",
        "商品有瑕疵，包装也破损了，非常失望，不推荐购买。",
        "物流速度快，商品包装完好，使用效果很好，满意！",
        "与描述不符，实物质量很差，感觉被骗了。",
        "客服服务很好，及时解决问题，商品也不错。",
        "价格有点贵，但质量确实好，物有所值。",
        "根本不能用，完全是废品，要求退款！"
    ]
    
    # 预测函数
    def predict_single_review(review):
        """预测单个评论"""
        # 预处理
        cleaned_review = chinese_preprocess(review)
        
        # 转换为特征向量
        review_vector = vectorizer.transform([cleaned_review])
        
        # 预测
        prediction = model.predict(review_vector)[0]
        proba = model.predict_proba(review_vector)[0]
        
        return prediction, proba
    
    print("预测示例评论的情感:\n")
    
    for i, review in enumerate(new_reviews, 1):
        pred, proba = predict_single_review(review)
        sentiment = "正面" if pred == 1 else "负面"
        prob_positive = proba[1]
        
        # 根据置信度显示不同颜色
        if (sentiment == "正面" and prob_positive > 0.7) or (sentiment == "负面" and prob_positive < 0.3):
            color = "\033[92m"  # 高置信度用绿色
        elif (sentiment == "正面" and prob_positive > 0.6) or (sentiment == "负面" and prob_positive < 0.4):
            color = "\033[93m"  # 中等置信度用黄色
        else:
            color = "\033[91m"  # 低置信度用红色
        
        print(f"评论 {i}: {review[:50]}..." if len(review) > 50 else f"评论 {i}: {review}")
        print(f"{color}  预测情感: {sentiment}")
        print(f"  正面概率: {prob_positive:.4f} | 负面概率: {proba[0]:.4f}\033[0m")
        print("-" * 60)

# 运行新评论预测
predict_new_reviews(svm_model, vectorizer)

# ==================== 10. 项目总结 ====================
print("\n" + "=" * 60)
print("项目总结")
print("=" * 60)
print("✓ 数据加载: 成功读取4个数据文件（自动检测并解析XML格式）")
print("✓ 文本预处理: 完成中文分词和清洗")
print("✓ 特征提取: 使用TF-IDF提取了文本特征")
print("✓ 模型训练: 成功训练了SVM分类器")
print("✓ 模型评估: 在测试集上评估了模型性能")
print(f"\n🎉 项目完成！最终模型准确率: {accuracy:.4f}")
print("=" * 60)

# ==================== 11. 模型保存（可选） ====================
# 如果需要保存模型，可以取消以下代码的注释
'''
import joblib
joblib.dump(svm_model, 'svm_sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("\n模型已保存为 svm_sentiment_model.pkl")
print("向量化器已保存为 tfidf_vectorizer.pkl")
'''


# In[ ]:




