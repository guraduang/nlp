# 基于朴素贝叶斯的邮件分类系统

## 项目概述
本仓库实现了一个支持特征模式切换的邮件分类系统，采用机器学习算法对垃圾邮件和普通邮件进行自动识别。系统提供两种特征选择模式：
- 高频词特征选择
- TF-IDF加权特征选择

## 算法基础
### 多项式朴素贝叶斯分类器
采用基于贝叶斯定理的概率分类器，其核心公式为：

$$
P(y|x_1,...,x_n) = \frac{P(y)\prod_{i=1}^n P(x_i|y)}{P(x_1,...,x_n)}
$$

**特征独立性假设**：
假设各特征条件独立，即：
$$
P(x_i|y,x_1,...,x_{i-1},x_{i+1},...,x_n) = P(x_i|y)
$$

**邮件分类应用**：
- 计算邮件属于垃圾/普通类别的后验概率
- 比较两类概率进行决策：
  $$
  \hat{y} = \arg\max_y P(y)\prod_{i=1}^n P(x_i|y)
  $$

## 数据处理流程
### 预处理流程
1. ​**文本清洗**：
   ```python
   re.sub(r'[.【】0-9、——。，！~\*]', '', line)  # 过滤特殊符号
2. ​**中文分词**：
   ```python
   from jieba import cut
   list(cut(line))  # 使用结巴分词

3. ​**停用词过滤**：
   ```python
   filter(lambda word: len(word) > 1, line)  # 过滤单字词

## 特征构建过程
1. ​**高频词特征选择**：
实现方式：
    ```python
    Counter(chain(*all_words)).most_common(top_num)
2. TF-IDF加权特征
实现方式：
    ```python
    TfidfVectorizer(max_features=top_num)

# 高频词模式
python main.py --feature=frequency --top=100

# TF-IDF模式
python main.py --feature=tfidf --top=100