import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from bayes_opt import BayesianOptimization
from dataset import load_and_split_data, inverse_normalize_targets
from model import GIN
from torch_geometric.loader import DataLoader
from utils import evaluate_model, plot_bayesian_optimization_results


def evaluate_hyperparams(input_dim, hidden_dim, num_layers, dropout, learning_rate, batch_size, pooling_method, weight_decay, train_dataset, val_dataset, device, target_scaler=None, num_epochs=50):
    """
    评估一组超参数的性能
    
    参数:
        input_dim: 输入特征维度（固定值）
        hidden_dim, num_layers, dropout, learning_rate, batch_size, pooling_method: 要评估的超参数
        train_dataset, val_dataset: 训练和验证数据集
        device: 计算设备
        target_scaler: 目标值归一化器
        num_epochs: 训练轮数
    
    返回:
        val_r2_avg: 验证集上的平均R2分数 (优化目标)
    """
    # 转换超参数为适当的类型
    hidden_dim = int(hidden_dim)
    num_layers = int(num_layers)
    batch_size = int(batch_size)
    
    # 选择池化方法
    if pooling_method < 0.33:
        pooling = 'mean'
    elif pooling_method < 0.67:
        pooling = 'sum'
    else:
        pooling = 'max'
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 创建模型
    model = GIN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=3, 
                num_layers=num_layers, dropout=dropout, pooling=pooling)
    model = model.to(device)
    
    # 创建优化器和学习率调度器
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 训练模型
    best_val_r2 = -float('inf')
    early_stop_counter = 0
    early_stop_patience = 10
    
    for epoch in range(1, num_epochs + 1):
        # 训练阶段
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            batch_size = output.size(0)
            target = data.y.view(batch_size, -1)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()
        
        # 验证阶段
        _, _, val_metrics = evaluate_model(model, val_loader, device, target_scaler)
        val_r2_avg = val_metrics['r2_avg']
        
        # 更新学习率
        scheduler.step(1 - val_r2_avg)  # 最小化1-R²
        
        # 早停检查
        if val_r2_avg > best_val_r2:
            best_val_r2 = val_r2_avg
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                break
    
    return best_val_r2

def optimize_hyperparams(train_dataset, val_dataset, device, input_dim, n_iterations=50, initial_points=10, results_dir='results', target_scaler=None):
    """
    使用贝叶斯优化寻找最佳超参数
    
    参数:
        train_dataset, val_dataset: 训练和验证数据集
        device: 计算设备
        input_dim: 输入特征维度
        n_iterations: 贝叶斯优化迭代次数
        initial_points: 初始随机探索点数量
        results_dir: 结果保存目录
        target_scaler: 目标值归一化器
    
    返回:
        best_params: 最佳超参数字典
        optimizer: 贝叶斯优化器
    """
    # 定义超参数空间
    pbounds = {
        'hidden_dim': (32, 256),           # 隐藏层维度
        'num_layers': (2, 5),              # GIN层数
        'dropout': (0.0, 0.5),             # Dropout率
        'learning_rate': (1e-4, 1e-2),     # 学习率
        'batch_size': (32, 128),           # 批量大小
        'pooling_method': (0, 1),          # 池化方法
        'weight_decay': (1e-6, 1e-3)       # 权重衰减
    }
    
    # 将评估函数包装为只接受要优化的参数
    def objective(**params):
        return evaluate_hyperparams(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            learning_rate=params['learning_rate'],
            batch_size=params['batch_size'],
            pooling_method=params['pooling_method'],
            weight_decay=params['weight_decay'],
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            target_scaler=target_scaler
        )
    
    # 创建贝叶斯优化器
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=42
    )
    
    # 执行优化
    optimizer.maximize(
        init_points=initial_points,
        n_iter=n_iterations
    )
    
    # 提取最佳参数
    best_params = optimizer.max['params']
    
    # 将pooling_method转换为具体的池化方法
    if best_params['pooling_method'] < 0.33:
        best_params['pooling'] = 'mean'
    elif best_params['pooling_method'] < 0.67:
        best_params['pooling'] = 'sum'
    else:
        best_params['pooling'] = 'max'
    
    # 转换整数参数
    best_params['hidden_dim'] = int(best_params['hidden_dim'])
    best_params['num_layers'] = int(best_params['num_layers'])
    best_params['batch_size'] = int(best_params['batch_size'])
    
    # 打印最佳参数
    print("最佳超参数:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"验证集最佳R2: {optimizer.max['target']:.4f}")
    
    # 保存优化结果
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, 'bayesian_optimization_results.txt')
    with open(result_file, 'w') as f:
        f.write("最佳超参数:\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
        f.write(f"验证集最佳R2: {optimizer.max['target']:.4f}\n\n")
        
        f.write("所有迭代结果:\n")
        for i, res in enumerate(optimizer.res):
            f.write(f"迭代 {i+1}:\n")
            for param, value in res['params'].items():
                f.write(f"  {param}: {value}\n")
            f.write(f"  R2: {res['target']:.4f}\n\n")
    
    # 可视化优化过程
    plot_bayesian_optimization_results(optimizer.res, save_dir=results_dir)
    
    return best_params, optimizer

def train_with_optimal_params(best_params, train_dataset, val_dataset, test_dataset, input_dim, device, output_dim=3, num_epochs=200, model_dir='models', target_scaler=None):
    """
    使用最佳超参数训练最终模型
    
    参数:
        best_params: 最佳超参数字典
        train_dataset, val_dataset, test_dataset: 数据集
        input_dim: 输入特征维度
        device: 计算设备
        output_dim: 输出维度
        num_epochs: 训练轮数
        model_dir: 模型保存目录
        target_scaler: 目标值归一化器
    
    返回:
        best_model: 训练好的最佳模型
        test_metrics: 测试集评估指标
    """
    from train import train_gin_model, save_model
    
    # 创建数据加载器
    batch_size = best_params['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 创建模型
    model = GIN(
        input_dim=input_dim,
        hidden_dim=best_params['hidden_dim'],
        output_dim=output_dim,
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout'],
        pooling=best_params['pooling']
    )
    model = model.to(device)
    
    # 创建优化器和学习率调度器
    optimizer = Adam(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # 训练模型
    best_model, history = train_gin_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device,
        early_stopping_patience=20,
        target_scaler=target_scaler
    )
    
    # 保存最佳模型
    save_model(best_model, model_dir=model_dir, model_name='gin_optimal_model.pt')
    
    # 在测试集上评估模型
    test_preds, test_targets, test_metrics = evaluate_model(best_model, test_loader, device, target_scaler)
    
    # 打印测试集结果
    print("\n测试集评估结果:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return best_model, test_metrics, test_preds, test_targets 