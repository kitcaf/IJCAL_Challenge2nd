"""
配置文件
包含模型、训练和数据的各种配置参数
"""

import os

# 数据路径配置
DATA_CONFIG = {
    'train_file': 'data/train.json',
    'test_file': 'data/test.json',
    'batch_size': 32,
    'val_split': 0.15,
    'random_seed': 42,
    'num_workers': 0  # Windows系统建议设为0
}

# 模型配置
MODEL_CONFIGS = {
    'GCN': {
        'input_dim': 768,
        'hidden_dims': [512, 256, 128],
        'model_type': 'GCN',
        'pooling_type': 'mean_max',
        'num_classes': 2,
        'dropout': 0.3,
        'classifier_hidden': 64,
        'use_residual': True
    },
    
    'GAT': {
        'input_dim': 768,
        'hidden_dims': [512, 256, 128],
        'model_type': 'GAT',
        'pooling_type': 'mean_max',
        'num_classes': 2,
        'dropout': 0.3,
        'classifier_hidden': 64,
        'use_residual': True
    },
    
    'SAGE': {
        'input_dim': 768,
        'hidden_dims': [512, 256, 128],
        'model_type': 'SAGE',
        'pooling_type': 'mean_max',
        'num_classes': 2,
        'dropout': 0.3,
        'classifier_hidden': 64,
        'use_residual': True
    },
    
    'GIN': {
        'input_dim': 768,
        'hidden_dims': [512, 256, 128],
        'model_type': 'GIN',
        'pooling_type': 'mean_max',
        'num_classes': 2,
        'dropout': 0.3,
        'classifier_hidden': 64,
        'use_residual': True
    }
}

# 训练配置
TRAIN_CONFIG = {
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'epochs': 100,
    'patience': 15,
    'loss_type': 'ce',  # 'ce' 或 'focal'
    'class_weights': [1.0, 1.38],  # [正常事件权重, 突发事件权重]
    'device': 'cuda' if os.getenv('CUDA_VISIBLE_DEVICES') else 'cpu'
}

# 路径配置
PATH_CONFIG = {
    'checkpoint_dir': 'checkpoints',
    'log_dir': 'logs',
    'result_dir': 'results'
}

# 实验配置
EXPERIMENT_CONFIG = {
    'model_types': ['GCN', 'GAT', 'SAGE', 'GIN'],  # 要测试的模型类型
    'run_all_models': False,  # 是否运行所有模型
    'default_model': 'GCN',  # 默认模型
    'save_predictions': True,  # 是否保存预测结果
    'generate_submission': True  # 是否生成提交文件
}

# 获取完整配置
def get_config(model_type: str = 'GCN') -> dict:
    """
    获取指定模型类型的完整配置
    
    Args:
        model_type: 模型类型
    
    Returns:
        配置字典
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    config = {
        'data': DATA_CONFIG,
        'model': MODEL_CONFIGS[model_type],
        'train': TRAIN_CONFIG,
        'paths': PATH_CONFIG,
        'experiment': EXPERIMENT_CONFIG
    }
    
    return config

# 打印配置信息
def print_config(config: dict):
    """打印配置信息"""
    print("当前配置:")
    print("=" * 50)
    
    for section, params in config.items():
        print(f"\n[{section.upper()}]")
        for key, value in params.items():
            print(f"  {key}: {value}")

# 验证配置
def validate_config(config: dict) -> bool:
    """验证配置的有效性"""
    try:
        # 检查数据文件是否存在
        data_config = config['data']
        if not os.path.exists(data_config['train_file']):
            print(f"警告: 训练文件不存在 - {data_config['train_file']}")
            return False
        
        if not os.path.exists(data_config['test_file']):
            print(f"警告: 测试文件不存在 - {data_config['test_file']}")
            return False
        
        # 检查模型配置
        model_config = config['model']
        if model_config['model_type'] not in ['GCN', 'GAT', 'SAGE', 'GIN']:
            print(f"错误: 不支持的模型类型 - {model_config['model_type']}")
            return False
        
        # 创建必要的目录
        for path in config['paths'].values():
            os.makedirs(path, exist_ok=True)
        
        print("配置验证通过!")
        return True
        
    except Exception as e:
        print(f"配置验证失败: {e}")
        return False
