"""
主训练脚本
用于训练和评估级联图分类模型
"""

import torch
import json
import os
import argparse
from datetime import datetime

# 导入自定义模块
from dataset import create_data_loaders, get_dataset_info
from models import CascadeClassifier
from trainer import Trainer, print_detailed_metrics, load_model
from config import get_config, print_config, validate_config


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='级联图分类训练')
    parser.add_argument('--model', type=str, default='GCN', 
                       choices=['GCN', 'GAT', 'SAGE', 'GIN'],
                       help='模型类型')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--device', type=str, default='auto', help='设备类型')
    parser.add_argument('--save_name', type=str, default=None, help='模型保存名称')
    
    args = parser.parse_args()
    
    # 获取配置
    config = get_config(args.model)
    
    # 更新配置
    if args.epochs != 100:
        config['train']['epochs'] = args.epochs
    if args.batch_size != 32:
        config['data']['batch_size'] = args.batch_size
    if args.lr != 0.001:
        config['train']['learning_rate'] = args.lr
    if args.device != 'auto':
        config['train']['device'] = args.device
    else:
        config['train']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 打印配置
    print_config(config)
    
    # 验证配置
    if not validate_config(config):
        return
    
    print(f"\n使用设备: {config['train']['device']}")
    print(f"PyTorch版本: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # 创建数据加载器
    print("\n" + "="*60)
    print("准备数据...")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_file=config['data']['train_file'],
        test_file=config['data']['test_file'],
        batch_size=config['data']['batch_size'],
        val_split=config['data']['val_split'],
        random_seed=config['data']['random_seed']
    )
    
    # 获取数据集信息
    dataset_info = get_dataset_info(train_loader)
    print(f"数据集信息: {dataset_info}")
    
    # 创建模型
    print("\n" + "="*60)
    print("创建模型...")
    
    model = CascadeClassifier(**config['model'])
    print(f"模型已创建: {model.get_model_info()}")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 创建训练器
    print("\n" + "="*60)
    print("初始化训练器...")
    
    trainer = Trainer(
        model=model,
        device=config['train']['device'],
        learning_rate=config['train']['learning_rate'],
        weight_decay=config['train']['weight_decay'],
        loss_type=config['train']['loss_type'],
        class_weights=config['train']['class_weights']
    )
    
    # 开始训练
    print("\n" + "="*60)
    print("开始训练...")
    
    start_time = datetime.now()
    
    # 设置模型保存名称
    if args.save_name:
        model_name = args.save_name
    else:
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        model_name = f"{args.model}_{timestamp}"
    
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
    print(f"训练完成! 总耗时: {training_time}")
    
    # 评估最佳模型
    print("\n" + "="*60)
    print("评估最佳模型...")
    
    # 加载最佳模型
    best_model_path = os.path.join(config['paths']['checkpoint_dir'], f'{model_name}_best.pth')
    best_model, checkpoint = load_model(CascadeClassifier, best_model_path, config['train']['device'])
    
    # 创建评估器
    evaluator = Trainer(
        model=best_model,
        device=config['train']['device']
    )
    
    # 验证集评估
    print("\n验证集评估:")
    val_metrics, val_preds, val_labels = evaluator.evaluate(val_loader)
    print_detailed_metrics(val_labels, val_preds)
    
    # 测试集预测（如果有标签）
    print("\n测试集预测:")
    test_preds, test_probs = evaluator.predict(test_loader)
    
    # 保存预测结果
    results_dir = config['paths']['result_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存预测结果
    predictions = {
        'model_type': args.model,
        'model_name': model_name,
        'predictions': test_preds,
        'probabilities': test_probs,
        'val_metrics': val_metrics,
        'config': config,
        'training_time': str(training_time)
    }
    
    pred_file = os.path.join(results_dir, f'{model_name}_predictions.json')
    with open(pred_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    print(f"预测结果已保存到: {pred_file}")
    
    # 生成提交文件（比赛格式）
    submission_file = os.path.join(results_dir, f'{model_name}_result.json')
    with open(submission_file, 'w', encoding='utf-8') as f:
        json.dump(test_preds, f)
    
    print(f"提交文件已保存到: {submission_file}")
    
    # 保存训练历史
    history_file = os.path.join(results_dir, f'{model_name}_history.json')
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    
    print(f"训练历史已保存到: {history_file}")
    
    print("\n" + "="*60)
    print("所有任务完成!")
    print(f"最佳验证F1分数: {checkpoint['best_f1']:.4f}")
    print(f"模型文件: {best_model_path}")
    print(f"预测结果: {pred_file}")
    print(f"提交文件: {submission_file}")


if __name__ == "__main__":
    main()
