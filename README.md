# IJCAL_challenge_2nd 任务2

对应链接：https://magic-group-buaa.github.io/IJCAI25/index.html

## 任务定义 Task Definition

### 背景介绍 Introduction
突发事件级联分类任务旨在将社交网络结构分类为突发事件级联或正常事件。突发事件级联的高效分类，实现突发事件的早期检测，维护网络话语的完整性和公众感知。

The bursty event cascade classification task aims to classify social network structures as either the cascade of bursty events or normal events. Efficient classification of bursty event cascade enables the early detection of bursty events, safeguarding the integrity of online discourse and public perception.

### 任务类型 Task Type
- **分类类型**: 二分类任务 (Binary Classification)
- **输入**: 级联图网络结构 + 节点BERT特征
- **输出**: 事件类别 (0: 正常事件, 1: 突发事件)
- **评估指标**: F1分数 (F1-score)

### 数据集信息 Dataset Information
- **来源**: 微博数据 (Weibo Data)
- **训练集**: 13,599个级联样本 
- **测试集**: 无标签，需要预测
- **特征维度**: 768维BERT文本嵌入
- **网络规模**: 平均31个节点，5-499个节点范围

### 核心数据结构 Data Structure
```json
{
  "edge_index": [[0, 0, 0, ...], [1, 2, 3, ...]],  // 级联网络边索引 [2, num_edges]
  "x": {
    "0": [768维特征向量],  // 节点0的BERT特征
    "1": [768维特征向量],  // 节点1的BERT特征
    ...
  },
  "label": 1  // 事件标签 (0:正常, 1:突发)
}
```

### 关键特征差异 Key Feature Differences
根据数据统计分析，突发事件与正常事件的主要区别：
- **传播深度**: 突发事件平均深度3.93 vs 正常事件2.00
- **网络规模**: 突发事件平均34.10个节点 vs 正常事件29.27个节点
- **传播模式**: 突发事件具有更复杂的多层级传播结构

### 提交要求 Submission Requirements
- **格式**: .zip文件，包含source_code文件夹、result.json、introduction.pdf
- **结果格式**: [0, 1, 0, 1, 0, ..., 1, 0, 0] (对应test.json中每个样本的预测标签)
- **评估标准**: F1分数 (精确率和召回率的调和平均值)

## 设计方案 Solution Design


### 核心技术栈 Core Technologies

**图神经网络架构**:
- GCN (Graph Convolutional Network): 基础图卷积，捕获局部邻域信息
- GAT (Graph Attention Network): 注意力机制，识别关键传播路径  
- GraphSAGE: 图采样聚合，适应不同规模图结构
- GIN (Graph Isomorphism Network): 最强图表示能力

**关键技术组件**:
- 全局池化策略 (Global Pooling): Mean + Max 池化融合
- 时序感知建模 (Temporal Modeling): 考虑传播时间动态
- 多尺度特征融合 (Multi-scale Fusion): 局部+全局结构特征
- 类别不平衡处理 (Imbalanced Classification): Focal Loss + 加权损失

**核心建模思路**: 
重点建模不同事件类型的级联传播特征差异，特别是突发事件的深层传播模式和快速扩散特性。



### 方案一：GAT + 全局池化 + MLP

```
输入: (edge_index, node_features)
↓
GAT层1-3: 学习节点间注意力权重
↓
全局注意力池化: 得到图级表示
↓
MLP分类器: 二分类输出
```

### 方案二：多尺度GNN融合

```
输入: (edge_index, node_features)
↓
并行分支:
- GCN分支: 捕获局部结构
- GraphSAGE分支: 处理大图结构  
- 统计特征分支: 手工特征
↓
特征融合 → MLP分类器
```

### 方案三：时序+结构联合建模

```
输入: (edge_index, node_features, timestamps)
↓
时序编码 + 图结构编码
↓ 
时空注意力机制
↓
分类输出
```

我认为核心：核心应该去考虑建模不同事件的传播图的传播特征

## 技术创新点 Technical Innovations

### 1. 传播模式建模
- **深度感知**: 针对突发事件深度更大的特点(3.93 vs 2.00)，设计层次化注意力机制
- **扩散速度**: 建模突发事件的快速传播特征
- **结构差异**: 捕获星型vs树型vs链型等不同传播拓扑

### 2. 多模态特征融合
- **内容特征**: BERT文本嵌入的语义信息
- **结构特征**: 图拓扑统计特征(度分布、聚类系数等)
- **传播特征**: 时间序列、影响力传播路径

### 3. 自适应图学习
- **图结构增强**: 基于内容相似性补充缺失边
- **动态注意力**: 根据传播阶段调整节点重要性
- **多尺度聚合**: 不同感受野的特征融合

### 4. 鲁棒性设计
- **规模自适应**: 处理5-499节点的巨大规模差异
- **噪声容忍**: 对不完整级联和噪声数据的鲁棒性
- **泛化能力**: 跨时间、跨主题的模型泛化










