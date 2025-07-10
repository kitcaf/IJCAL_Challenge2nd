"""
评估脚本
用于单独评估训练好的模型
"""

import torch
import json
import os
import argparse

from dataset import create_data_loaders
from models import CascadeClassifier
from trainer import Trainer, load_model, print_detailed_metrics
from config import get_config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模型评估')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--data_type', type=str, default='test', choices=['train', 'val', 'test'], help='评估数据类型')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--device', type=str, default='auto', help='设备类型')
    parser.add_argument('--save_predictions', action='store_true', help='是否保存预测结果')
    
    args = parser.parse_args()
    
    # 设备配置
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    
    # 加载模型
    print("加载模型...")
    model, checkpoint = load_model(CascadeClassifier, args.model_path, device)
    model_info = checkpoint['model_info']
    
    # 获取配置（基于模型类型）
    config = get_config(model_info['model_type'])
    config['data']['batch_size'] = args.batch_size
    
    # 创建数据加载器
    print("准备数据...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_file=config['data']['train_file'],
        test_file=config['data']['test_file'],
        batch_size=config['data']['batch_size'],
        val_split=config['data']['val_split'],
        random_seed=config['data']['random_seed']
    )
    
    # 选择要评估的数据
    if args.data_type == 'train':
        data_loader = train_loader
        data_name = '训练集'
    elif args.data_type == 'val':
        data_loader = val_loader
        data_name = '验证集'
    else:
        data_loader = test_loader
        data_name = '测试集'
    
    # 创建评估器
    evaluator = Trainer(model=model, device=device)
    
    # 进行评估
    print(f"\n评估{data_name}...")
    
    if args.data_type != 'test':  # 训练集和验证集有标签
        metrics, preds, labels = evaluator.evaluate(data_loader)
        print_detailed_metrics(labels, preds)
        
        # 保存结果
        if args.save_predictions:
            results = {
                'data_type': args.data_type,
                'metrics': metrics,
                'predictions': preds,
                'labels': labels,
                'model_path': args.model_path,
                'model_info': model_info
            }
            
            result_file = f"evaluation_{args.data_type}_{os.path.basename(args.model_path)}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"评估结果已保存到: {result_file}")
    
    else:  # 测试集没有标签，只做预测
        preds, probs = evaluator.predict(data_loader)
        print(f"测试集预测完成，共 {len(preds)} 个样本")
        print(f"预测分布: 正常事件={sum(1 for p in preds if p==0)}, 突发事件={sum(1 for p in preds if p==1)}")
        
        # 保存结果
        if args.save_predictions:
            results = {
                'data_type': args.data_type,
                'predictions': preds,
                'probabilities': probs,
                'model_path': args.model_path,
                'model_info': model_info
            }
            
            result_file = f"predictions_{args.data_type}_{os.path.basename(args.model_path)}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # 同时保存比赛提交格式
            submission_file = f"result_{os.path.basename(args.model_path)}.json"
            with open(submission_file, 'w', encoding='utf-8') as f:
                json.dump(preds, f)
            
            print(f"预测结果已保存到: {result_file}")
            print(f"提交文件已保存到: {submission_file}")


if __name__ == "__main__":
    main()
