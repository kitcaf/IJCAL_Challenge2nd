# IJCAL_challenge_2nd 任务2

对应链接：https://magic-group-buaa.github.io/IJCAI25/index.html

## 设计方案

任务定义：二分类任务


### 技术

GCN (Graph Convolutional Network)、GraphSAGE、GAT (Graph Attention Network)、 池化策略、时序感知GNN、局部结构级别、手工特征



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










