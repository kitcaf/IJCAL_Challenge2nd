"""
数据集加载和预处理模块
处理JSON格式的级联图数据，转换为PyTorch Geometric格式
"""

import json
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional


class CascadeDataset(Dataset):
    """级联图数据集类"""
    
    def __init__(self, json_file: str, transform=None, pre_transform=None):
        """
        初始化数据集
        
        Args:
            json_file: JSON数据文件路径
            transform: 数据变换
            pre_transform: 预处理变换
        """
        super().__init__(None, transform, pre_transform)
        self.json_file = json_file
        self.data_list = self._load_data()
        
    def _load_data(self) -> List[Data]:
        """加载JSON数据并转换为PyG Data对象列表"""
        print(f"正在加载数据文件: {self.json_file}")
        
        with open(self.json_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        print(f"数据加载完成，共 {len(raw_data)} 条记录")
        
        data_list = []
        for i, item in enumerate(raw_data):
            if i % 1000 == 0:
                print(f"处理进度: {i}/{len(raw_data)}")
            
            # 转换为PyG Data对象
            data = self._convert_to_pyg_data(item)
            if data is not None:
                data_list.append(data)
        
        print(f"数据处理完成，有效样本: {len(data_list)}")
        return data_list
    
    def _convert_to_pyg_data(self, item: Dict) -> Optional[Data]:
        """将单个样本转换为PyG Data对象"""
        try:
            # 获取边索引
            edge_index = item.get('edge_index', [])
            if not edge_index or len(edge_index) != 2 or len(edge_index[0]) == 0:
                return None
            
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            
            # 获取节点特征
            x_dict = item.get('x', {})
            if not x_dict:
                return None
            
            # 构建节点特征矩阵
            node_features = self._build_node_features(x_dict, edge_index)
            
            # 获取标签（如果存在）
            label = item.get('label', None)
            y = torch.tensor([label], dtype=torch.long) if label is not None else None
            
            # 创建Data对象
            data = Data(
                x=node_features,
                edge_index=edge_index,
                y=y
            )
            
            return data
            
        except Exception as e:
            print(f"处理样本时出错: {e}")
            return None
    
    def _build_node_features(self, x_dict: Dict, edge_index: torch.Tensor) -> torch.Tensor:
        """构建节点特征矩阵"""
        # 获取所有节点ID
        all_nodes = set(edge_index[0].tolist() + edge_index[1].tolist())
        max_node_id = max(all_nodes)
        
        # 初始化特征矩阵
        feature_dim = len(next(iter(x_dict.values())))
        node_features = torch.zeros(max_node_id + 1, feature_dim, dtype=torch.float)
        
        # 填充已有的节点特征
        for node_id_str, features in x_dict.items():
            node_id = int(node_id_str)
            if node_id <= max_node_id:
                node_features[node_id] = torch.tensor(features, dtype=torch.float)
        
        return node_features
    
    def len(self) -> int:
        """返回数据集大小"""
        return len(self.data_list)
    
    def get(self, idx: int) -> Data:
        """获取单个样本"""
        return self.data_list[idx]


def create_data_loaders(train_file: str, test_file: str, 
                       batch_size: int = 32, 
                       val_split: float = 0.15,
                       random_seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证和测试数据加载器
    
    Args:
        train_file: 训练数据文件路径
        test_file: 测试数据文件路径
        batch_size: 批处理大小
        val_split: 验证集比例
        random_seed: 随机种子
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # 设置随机种子
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # 加载训练数据集
    train_dataset = CascadeDataset(train_file)
    
    # 分割训练集和验证集
    dataset_size = len(train_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # 加载测试数据集
    test_dataset = CascadeDataset(test_file)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"数据加载器创建完成:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本") 
    print(f"  测试集: {len(test_dataset)} 样本")
    
    return train_loader, val_loader, test_loader


def get_dataset_info(data_loader: DataLoader) -> Dict:
    """获取数据集基本信息"""
    sample_batch = next(iter(data_loader))
    
    info = {
        'num_features': sample_batch.x.size(1),
        'num_classes': len(torch.unique(sample_batch.y)) if sample_batch.y is not None else 2,
        'batch_size': sample_batch.batch.max().item() + 1 if hasattr(sample_batch, 'batch') else 1
    }
    
    return info
