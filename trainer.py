"""
训练和评估模块
包含训练循环、评估指标计算、模型保存加载等功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
import numpy as np
import time
import os
from typing import Dict, List, Tuple, Optional
import json


class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""
    
    def __init__(self, alpha: float = 0.58, gamma: float = 2.0, reduction: str = 'mean'):
        """
        初始化Focal Loss
        
        Args:
            alpha: 类别权重 (针对突发事件类别)
            gamma: 聚焦参数
            reduction: 降维方式
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # 计算alpha权重
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # 计算focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class Trainer:
    """训练器类"""
    
    def __init__(self, 
                 model,
                 device: str = 'cuda',
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 loss_type: str = 'ce',  # 'ce' or 'focal'
                 class_weights: Optional[List[float]] = None):
        """
        初始化训练器
        
        Args:
            model: 模型实例
            device: 设备类型
            learning_rate: 学习率
            weight_decay: 权重衰减
            loss_type: 损失函数类型
            class_weights: 类别权重
        """
        self.model = model.to(device)
        self.device = device
        
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max',  # 基于F1-score最大化
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        # 损失函数
        if loss_type == 'focal':
            self.criterion = FocalLoss()
        elif loss_type == 'ce':
            if class_weights is not None:
                weights = torch.tensor(class_weights, dtype=torch.float).to(device)
                self.criterion = nn.CrossEntropyLoss(weight=weights)
            else:
                self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"不支持的损失函数类型: {loss_type}")
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_f1': [],
            'val_loss': [],
            'val_f1': [],
            'val_accuracy': [],
            'learning_rates': []
        }
    
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in train_loader:
            batch = batch.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = self.criterion(outputs, batch.y)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return avg_loss, f1
    
    def evaluate(self, data_loader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch.y)
                
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
        
        # 计算各种指标
        metrics = {
            'loss': total_loss / len(data_loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_macro': f1_score(all_labels, all_preds, average='macro'),
            'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
            'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0)
        }
        
        return metrics, all_preds, all_labels
    
    def train(self, 
              train_loader, 
              val_loader, 
              epochs: int = 100,
              patience: int = 10,
              save_dir: str = 'checkpoints',
              model_name: str = 'cascade_classifier') -> Dict:
        """
        完整训练循环
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            patience: 早停耐心值
            save_dir: 模型保存目录
            model_name: 模型名称
        
        Returns:
            训练历史字典
        """
        os.makedirs(save_dir, exist_ok=True)
        best_f1 = 0
        patience_counter = 0
        
        print("开始训练...")
        print(f"模型信息: {self.model.get_model_info()}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 训练
            train_loss, train_f1 = self.train_epoch(train_loader)
            
            # 验证
            val_metrics, _, _ = self.evaluate(val_loader)
            val_loss = val_metrics['loss']
            val_f1 = val_metrics['f1_macro']
            val_acc = val_metrics['accuracy']
            
            # 学习率调度
            self.scheduler.step(val_f1)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_f1'].append(train_f1)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)
            self.history['val_accuracy'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # 打印进度
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s)")
            print(f"  Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Acc: {val_acc:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                
                # 保存模型
                model_path = os.path.join(save_dir, f'{model_name}_best.pth')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'best_f1': best_f1,
                    'model_info': self.model.get_model_info(),
                    'history': self.history
                }, model_path)
                
                print(f"  新的最佳模型已保存! F1: {best_f1:.4f}")
            else:
                patience_counter += 1
            
            # 早停检查
            if patience_counter >= patience:
                print(f"早停触发! 最佳F1: {best_f1:.4f}")
                break
            
            print("-" * 60)
        
        print("训练完成!")
        return self.history
    
    def predict(self, data_loader) -> Tuple[List[int], List[float]]:
        """预测"""
        self.model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                
                # 预测标签
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                
                # 预测概率
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs[:, 1].cpu().numpy())  # 突发事件概率
        
        return all_preds, all_probs


def load_model(model_class, checkpoint_path: str, device: str = 'cuda'):
    """
    加载训练好的模型
    
    Args:
        model_class: 模型类
        checkpoint_path: 检查点路径
        device: 设备类型
    
    Returns:
        加载的模型实例
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_info = checkpoint['model_info']
    
    # 创建模型
    model = model_class(**model_info)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"模型已从 {checkpoint_path} 加载")
    print(f"最佳F1分数: {checkpoint['best_f1']:.4f}")
    
    return model, checkpoint


def print_detailed_metrics(labels, preds, class_names=None):
    """打印详细的评估指标"""
    if class_names is None:
        class_names = ['正常事件', '突发事件']
    
    print("\n详细评估结果:")
    print("=" * 50)
    
    # 整体指标
    print(f"准确率 (Accuracy): {accuracy_score(labels, preds):.4f}")
    print(f"F1分数 (Macro): {f1_score(labels, preds, average='macro'):.4f}")
    print(f"F1分数 (Weighted): {f1_score(labels, preds, average='weighted'):.4f}")
    print(f"精确率 (Macro): {precision_score(labels, preds, average='macro', zero_division=0):.4f}")
    print(f"召回率 (Macro): {recall_score(labels, preds, average='macro', zero_division=0):.4f}")
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(labels, preds, target_names=class_names, zero_division=0))
