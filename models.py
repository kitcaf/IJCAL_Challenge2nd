"""
图神经网络模型模块
支持多种GNN架构：GCN, GAT, GraphSAGE, GIN等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv, GINConv,
    global_mean_pool, global_max_pool, global_add_pool
)
from typing import Optional


class GraphEncoder(nn.Module):
    """可配置的图编码器，支持多种GNN架构"""
    
    def __init__(self, 
                 input_dim: int = 768,
                 hidden_dims: list = [512, 256, 128],
                 model_type: str = 'GCN',
                 dropout: float = 0.3,
                 use_residual: bool = True):
        """
        初始化图编码器
        
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            model_type: 模型类型 ('GCN', 'GAT', 'SAGE', 'GIN')
            dropout: Dropout比例
            use_residual: 是否使用残差连接
        """
        super().__init__()
        
        self.model_type = model_type
        self.dropout = dropout
        self.use_residual = use_residual
        
        # 构建网络层
        dims = [input_dim] + hidden_dims
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            
            # 根据模型类型选择卷积层
            if model_type == 'GCN':
                conv = GCNConv(in_dim, out_dim)
            elif model_type == 'GAT':
                # GAT使用多头注意力
                heads = 4 if i < len(dims) - 2 else 1  # 最后一层使用单头
                conv = GATConv(in_dim, out_dim // heads, heads=heads, concat=(i < len(dims) - 2))
                if i < len(dims) - 2:  # 前面层concat多头，需要调整输出维度
                    out_dim = out_dim // heads * heads
            elif model_type == 'SAGE':
                conv = SAGEConv(in_dim, out_dim)
            elif model_type == 'GIN':
                # GIN需要MLP作为更新函数
                mlp = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim)
                )
                conv = GINConv(mlp)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            self.convs.append(conv)
            
            # 批归一化
            self.norms.append(nn.BatchNorm1d(out_dim))
        
        self.output_dim = hidden_dims[-1]
    
    def forward(self, x, edge_index, batch=None):
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            batch: 批次索引 [num_nodes]
        
        Returns:
            节点级特征 [num_nodes, output_dim]
        """
        # 保存输入用于残差连接
        residual = None
        
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # 保存残差
            if self.use_residual and i > 0 and x.size(-1) == self.output_dim:
                residual = x
            
            # 图卷积
            x = conv(x, edge_index)
            
            # 批归一化
            x = norm(x)
            
            # 激活函数（最后一层不使用）
            if i < len(self.convs) - 1:
                x = F.relu(x)
            
            # 残差连接
            if residual is not None and x.size() == residual.size():
                x = x + residual
            
            # Dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class GlobalPooling(nn.Module):
    """全局池化层，支持多种池化策略"""
    
    def __init__(self, pooling_type: str = 'mean_max'):
        """
        初始化全局池化
        
        Args:
            pooling_type: 池化类型 ('mean', 'max', 'add', 'mean_max')
        """
        super().__init__()
        self.pooling_type = pooling_type
    
    def forward(self, x, batch):
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, feature_dim]
            batch: 批次索引 [num_nodes]
        
        Returns:
            图级特征 [batch_size, feature_dim * pool_factor]
        """
        if self.pooling_type == 'mean':
            return global_mean_pool(x, batch)
        elif self.pooling_type == 'max':
            return global_max_pool(x, batch)
        elif self.pooling_type == 'add':
            return global_add_pool(x, batch)
        elif self.pooling_type == 'mean_max':
            # 拼接mean和max池化
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            return torch.cat([mean_pool, max_pool], dim=1)
        else:
            raise ValueError(f"不支持的池化类型: {self.pooling_type}")


class Classifier(nn.Module):
    """分类器模块"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 2, dropout: float = 0.5):
        """
        初始化分类器
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_classes: 分类数量
            dropout: Dropout比例
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        """前向传播"""
        return self.classifier(x)


class CascadeClassifier(nn.Module):
    """完整的级联分类模型"""
    
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dims: list = [512, 256, 128],
                 model_type: str = 'GCN',
                 pooling_type: str = 'mean_max',
                 num_classes: int = 2,
                 dropout: float = 0.3,
                 classifier_hidden: int = 64,
                 use_residual: bool = True):
        """
        初始化级联分类模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dims: 图编码器隐藏层维度
            model_type: 图神经网络类型
            pooling_type: 全局池化类型
            num_classes: 分类数量
            dropout: Dropout比例
            classifier_hidden: 分类器隐藏层维度
            use_residual: 是否使用残差连接
        """
        super().__init__()
        
        # 图编码器
        self.graph_encoder = GraphEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            model_type=model_type,
            dropout=dropout,
            use_residual=use_residual
        )
        
        # 全局池化
        self.global_pooling = GlobalPooling(pooling_type)
        
        # 计算池化后的特征维度
        pool_factor = 2 if pooling_type == 'mean_max' else 1
        pooled_dim = hidden_dims[-1] * pool_factor
        
        # 分类器
        self.classifier = Classifier(
            input_dim=pooled_dim,
            hidden_dim=classifier_hidden,
            num_classes=num_classes,
            dropout=dropout
        )
        
        self.model_info = {
            'model_type': model_type,
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'pooling_type': pooling_type,
            'num_classes': num_classes
        }
    
    def forward(self, data):
        """
        前向传播
        
        Args:
            data: PyG数据对象，包含x, edge_index, batch
        
        Returns:
            分类logits [batch_size, num_classes]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 图编码
        node_features = self.graph_encoder(x, edge_index, batch)
        
        # 全局池化得到图级特征
        graph_features = self.global_pooling(node_features, batch)
        
        # 分类
        logits = self.classifier(graph_features)
        
        return logits
    
    def get_model_info(self):
        """获取模型信息"""
        return self.model_info


def create_model(config: dict) -> CascadeClassifier:
    """
    根据配置创建模型
    
    Args:
        config: 模型配置字典
    
    Returns:
        CascadeClassifier模型实例
    """
    return CascadeClassifier(**config)
