"""
多模型实验脚本
用于批量训练和比较不同的图神经网络模型
"""

import torch
import json
import os
import pandas as pd
from datetime import datetime

from dataset import create_data_loaders, get_dataset_info
from models import CascadeClassifier
from trainer import Trainer, print_detailed_metrics, load_model
from config import get_config, MODEL_CONFIGS


def run_single_experiment(model_type: str, config: dict, data_loaders: tuple, experiment_name: str = None):
    """
    运行单个模型实验
    
    Args:
        model_type: 模型类型
        config: 配置字典
        data_loaders: (train_loader, val_loader, test_loader)
        experiment_name: 实验名称
    
    Returns:
        实验结果字典
    """
    train_loader, val_loader, test_loader = data_loaders
    
    print(f"\n{'='*60}")
    print(f"开始实验: {model_type}")
    print(f"{'='*60}")
    
    # 创建模型
    model = CascadeClassifier(**config['model'])
    print(f"模型信息: {model.get_model_info()}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        device=config['train']['device'],
        learning_rate=config['train']['learning_rate'],
        weight_decay=config['train']['weight_decay'],
        loss_type=config['train']['loss_type'],
        class_weights=config['train']['class_weights']
    )
    
    # 训练
    start_time = datetime.now()
    
    if experiment_name:
        model_name = f"{experiment_name}_{model_type}"
    else:
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        model_name = f"experiment_{model_type}_{timestamp}"
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['train']['epochs'],
        patience=config['train']['patience'],
        save_dir=config['paths']['checkpoint_dir'],
        model_name=model_name
    )
    
    end_time = datetime.now()
    training_time = end_time - start_time
    
    # 加载最佳模型进行评估
    best_model_path = os.path.join(config['paths']['checkpoint_dir'], f'{model_name}_best.pth')
    best_model, checkpoint = load_model(CascadeClassifier, best_model_path, config['train']['device'])
    
    evaluator = Trainer(model=best_model, device=config['train']['device'])
    
    # 验证集评估
    val_metrics, val_preds, val_labels = evaluator.evaluate(val_loader)
    
    # 测试集预测
    test_preds, test_probs = evaluator.predict(test_loader)
    
    # 整理结果
    result = {
        'model_type': model_type,
        'model_name': model_name,
        'model_path': best_model_path,
        'training_time': str(training_time),
        'total_params': total_params,
        'best_val_f1': checkpoint['best_f1'],
        'val_metrics': val_metrics,
        'test_predictions': test_preds,
        'test_probabilities': test_probs,
        'config': config,
        'history': history
    }
    
    print(f"实验完成: {model_type}")
    print(f"最佳验证F1: {checkpoint['best_f1']:.4f}")
    print(f"训练时间: {training_time}")
    
    return result


def main():
    """主函数"""
    print("多模型实验开始...")
    
    # 实验配置
    experiment_name = f"multi_model_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_types = ['GCN', 'GAT', 'SAGE', 'GIN']
    
    # 基础配置
    base_config = get_config('GCN')
    base_config['train']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"实验名称: {experiment_name}")
    print(f"使用设备: {base_config['train']['device']}")
    print(f"测试模型: {model_types}")
    
    # 创建数据加载器（所有实验共用）
    print("\n准备数据...")
    data_loaders = create_data_loaders(
        train_file=base_config['data']['train_file'],
        test_file=base_config['data']['test_file'],
        batch_size=base_config['data']['batch_size'],
        val_split=base_config['data']['val_split'],
        random_seed=base_config['data']['random_seed']
    )
    
    dataset_info = get_dataset_info(data_loaders[0])
    print(f"数据集信息: {dataset_info}")
    
    # 运行实验
    all_results = []
    
    for model_type in model_types:
        try:
            # 获取模型特定配置
            config = get_config(model_type)
            config['train']['device'] = base_config['train']['device']
            
            # 运行实验
            result = run_single_experiment(
                model_type=model_type,
                config=config,
                data_loaders=data_loaders,
                experiment_name=experiment_name
            )
            
            all_results.append(result)
            
        except Exception as e:
            print(f"实验 {model_type} 失败: {e}")
            continue
    
    # 分析和比较结果
    print(f"\n{'='*60}")
    print("实验结果汇总")
    print(f"{'='*60}")
    
    # 创建结果表格
    comparison_data = []
    for result in all_results:
        comparison_data.append({
            '模型类型': result['model_type'],
            '验证F1': f"{result['best_val_f1']:.4f}",
            '验证准确率': f"{result['val_metrics']['accuracy']:.4f}",
            '验证精确率': f"{result['val_metrics']['precision']:.4f}",
            '验证召回率': f"{result['val_metrics']['recall']:.4f}",
            '参数量': f"{result['total_params']:,}",
            '训练时间': result['training_time']
        })
    
    df = pd.DataFrame(comparison_data)
    print("\n性能比较:")
    print(df.to_string(index=False))
    
    # 找出最佳模型
    best_result = max(all_results, key=lambda x: x['best_val_f1'])
    print(f"\n最佳模型: {best_result['model_type']}")
    print(f"最佳验证F1: {best_result['best_val_f1']:.4f}")
    
    # 保存结果
    results_dir = base_config['paths']['result_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存详细结果
    experiment_file = os.path.join(results_dir, f'{experiment_name}_results.json')
    with open(experiment_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 保存比较表格
    comparison_file = os.path.join(results_dir, f'{experiment_name}_comparison.csv')
    df.to_csv(comparison_file, index=False, encoding='utf-8-sig')
    
    # 保存最佳模型的提交文件
    best_submission_file = os.path.join(results_dir, f'{experiment_name}_best_result.json')
    with open(best_submission_file, 'w', encoding='utf-8') as f:
        json.dump(best_result['test_predictions'], f)
    
    print(f"\n实验结果已保存:")
    print(f"  详细结果: {experiment_file}")
    print(f"  比较表格: {comparison_file}")
    print(f"  最佳提交: {best_submission_file}")
    
    print(f"\n{'='*60}")
    print("所有实验完成!")


if __name__ == "__main__":
    main()
