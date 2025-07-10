# 级联图分类项目

## 项目简介

本项目实现了一个基于图神经网络的突发事件级联分类系统，用于识别社交网络中的突发事件传播模式。

## 项目结构

```
├── data/                    # 数据文件夹
│   ├── train.json          # 训练数据
│   └── test.json           # 测试数据
├── checkpoints/            # 模型检查点
├── results/                # 结果文件
├── logs/                   # 训练日志
├── dataset.py              # 数据加载模块
├── models.py               # 模型定义模块
├── trainer.py              # 训练和评估模块
├── config.py               # 配置文件
├── train.py                # 主训练脚本
├── evaluate.py             # 评估脚本
├── experiment.py           # 多模型实验脚本
└── README.md               # 说明文档
```

## 环境要求

```bash
torch>=1.12.0
torch-geometric>=2.3.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
```

## 安装依赖

```bash
pip install torch torch-geometric numpy pandas scikit-learn matplotlib
```

## 数据格式

输入数据为JSON格式，每个样本包含：
- `edge_index`: 级联网络的边索引，形状为[2, num_edges]
- `x`: 节点特征字典，每个节点对应768维BERT特征
- `label`: 事件标签（0: 正常事件, 1: 突发事件，测试集无此字段）

## 使用方法

### 1. 训练单个模型

```bash
# 使用GCN模型训练
python train.py --model GCN --epochs 100 --batch_size 32 --lr 0.001

# 使用GAT模型训练
python train.py --model GAT --epochs 100 --batch_size 32 --lr 0.001

# 支持的模型类型: GCN, GAT, SAGE, GIN
```

### 2. 评估已训练模型

```bash
# 评估训练好的模型
python evaluate.py --model_path checkpoints/GCN_best.pth --data_type test --save_predictions

# 评估验证集
python evaluate.py --model_path checkpoints/GCN_best.pth --data_type val
```

### 3. 多模型对比实验

```bash
# 运行所有模型的对比实验
python experiment.py
```

## 模型架构

支持四种图神经网络架构：

### GCN (Graph Convolutional Network)
- 基础图卷积网络
- 适合捕获局部邻域信息

### GAT (Graph Attention Network)  
- 图注意力网络
- 能识别关键传播路径
- 具有可解释性

### GraphSAGE
- 图采样聚合网络
- 适合大规模图处理
- 计算效率高

### GIN (Graph Isomorphism Network)
- 图同构网络
- 理论上具有最强表示能力
- 对图结构差异敏感

## 模型配置

所有模型采用统一架构：

```
输入层: 节点特征(768维) + 邻接矩阵
    ↓
图神经网络层1: 768 → 512 (ReLU + Dropout)
    ↓
图神经网络层2: 512 → 256 (ReLU + Dropout)
    ↓  
图神经网络层3: 256 → 128 (ReLU + Dropout)
    ↓
全局池化层: Graph-level表示 (Mean + Max)
    ↓
分类器: 256 → 64 → 2 (突发/正常)
```

## 训练特性

- **类别不平衡处理**: 支持加权损失函数和Focal Loss
- **早停机制**: 基于验证集F1分数，防止过拟合
- **学习率调度**: 自适应学习率调整
- **模型保存**: 自动保存最佳模型
- **详细日志**: 完整的训练和评估指标

## 输出文件

### 训练输出
- `checkpoints/{model_name}_best.pth`: 最佳模型权重
- `results/{model_name}_predictions.json`: 详细预测结果
- `results/{model_name}_result.json`: 比赛提交格式
- `results/{model_name}_history.json`: 训练历史

### 实验输出
- `results/experiment_results.json`: 多模型对比结果
- `results/experiment_comparison.csv`: 性能比较表格
- `results/experiment_best_result.json`: 最佳模型提交文件

## 配置说明

主要配置参数在 `config.py` 中：

```python
# 模型配置
MODEL_CONFIGS = {
    'input_dim': 768,           # BERT特征维度
    'hidden_dims': [512, 256, 128],  # 隐藏层维度
    'dropout': 0.3,             # Dropout比例
    'pooling_type': 'mean_max', # 池化策略
    # ...
}

# 训练配置  
TRAIN_CONFIG = {
    'learning_rate': 0.001,     # 学习率
    'epochs': 100,              # 最大训练轮数
    'patience': 15,             # 早停耐心值
    'batch_size': 32,           # 批处理大小
    # ...
}
```

## 数据统计

根据数据分析结果：
- 总样本数: 13,599
- 正常事件: 7,875 (57.91%)
- 突发事件: 5,724 (42.09%)
- 特征维度: 768
- 平均节点数: 31.31
- 平均边数: 30.31
- 突发事件平均深度更大 (3.93 vs 2.00)

## 性能指标

主要评估指标：
- **F1-score (Macro)**: 主要评估指标
- **准确率 (Accuracy)**
- **精确率 (Precision)**
- **召回率 (Recall)**

## 快速开始

1. 确保数据文件在正确位置
2. 安装依赖环境
3. 运行训练命令：

```bash
# 快速训练GCN模型
python train.py --model GCN --epochs 50

# 或运行完整实验
python experiment.py
```

## 注意事项

1. **GPU内存**: 大图可能需要较大GPU内存，可调整batch_size
2. **训练时间**: 完整训练可能需要较长时间，建议使用GPU
3. **随机种子**: 已设置固定随机种子保证结果可复现
4. **数据预处理**: 系统会自动处理节点特征对齐和图结构转换

## 扩展性

代码设计具有良好的模块化和扩展性：
- 轻松添加新的图神经网络模型
- 支持不同的池化策略
- 可配置的损失函数和优化器
- 灵活的数据预处理流程
