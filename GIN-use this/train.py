import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
import numpy as np
from tqdm import tqdm
from utils import evaluate_model

def train_gin_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, early_stopping_patience=20, target_scaler=None):
    """
    训练GIN模型
    
    参数:
        model: GIN模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        num_epochs: 训练轮数
        device: 计算设备
        early_stopping_patience: 早停耐心值
        target_scaler: 目标值归一化器
    
    返回:
        model: 训练好的模型
        history: 训练历史记录
    """
    model = model.to(device)
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'metrics': {}
    }
    
    # 早停设置
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # 训练循环
    for epoch in range(1, num_epochs + 1):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for data in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]"):
            data = data.to(device)
            optimizer.zero_grad()
            
            # 前向传播
            output = model(data)
            
            # 计算损失 - 修复维度不匹配问题
            batch_size = output.size(0)
            target = data.y.view(batch_size, -1)
            loss = F.mse_loss(output, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.num_graphs
        
        # 计算训练集平均损失
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]"):
                data = data.to(device)
                
                # 前向传播
                output = model(data)
                
                # 计算损失 - 修复维度不匹配问题
                batch_size = output.size(0)
                target = data.y.view(batch_size, -1)
                loss = F.mse_loss(output, target)
                val_loss += loss.item() * data.num_graphs
        
        # 计算验证集平均损失
        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        
        # 评估性能指标
        val_preds, val_targets, val_metrics = evaluate_model(model, val_loader, device, target_scaler)
        
        # 记录指标
        for metric, value in val_metrics.items():
            if metric not in history['metrics']:
                history['metrics'][metric] = []
            history['metrics'][metric].append(value)
        
        # 打印进度
        print(f"Epoch {epoch}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val MAE Avg: {val_metrics['mae_avg']:.4f}, "
              f"Val R2 Avg: {val_metrics['r2_avg']:.4f}")
        
        # 学习率调度
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"早停: 验证损失在 {early_stopping_patience} 个轮次内未改善.")
                break
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

def save_model(model, model_dir='models', model_name='gin_model.pt'):
    """
    保存训练好的模型
    
    参数:
        model: 训练好的模型
        model_dir: 模型保存目录
        model_name: 模型文件名
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")

def load_model(model, model_path, device):
    """
    加载预训练模型
    
    参数:
        model: 模型实例
        model_path: 模型文件路径
        device: 计算设备
    
    返回:
        model: 加载了权重的模型
    """
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"模型已从 {model_path} 加载")
    else:
        print(f"警告: 模型文件 {model_path} 不存在，使用初始化模型")
    
    return model 