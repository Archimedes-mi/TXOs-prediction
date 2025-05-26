import os
import torch
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from dataset import load_and_split_data, load_prediction_data, inverse_normalize_targets
from model import GIN
from train import train_gin_model, save_model, load_model
from bayesian_opt import optimize_hyperparams, train_with_optimal_params
from gnn_explainer import explain_model_predictions
from utils import plot_training_curves, plot_prediction_scatter, evaluate_model, save_predictions_to_csv

def main(args):
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    train_dataset, val_dataset, test_dataset, train_df, val_df, test_df, target_scaler = load_and_split_data(
        args.data_file, 
        smiles_col=args.smiles_col,
        target_cols=args.target_cols,
        test_size=args.test_size,
        val_size=args.val_size,
        normalize=not args.no_normalize
    )
    
    # 确定输入维度
    if len(train_dataset) > 0:
        input_dim = train_dataset[0].x.shape[1]
    else:
        print("错误: 训练集为空！")
        return
    
    # 超参数优化
    if args.optimize:
        print("\n开始贝叶斯超参数优化...")
        best_params, optimizer = optimize_hyperparams(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            input_dim=input_dim,
            n_iterations=args.opt_iterations,
            initial_points=args.opt_initial_points,
            results_dir=args.output_dir,
            target_scaler=target_scaler
        )
        
        # 使用最佳超参数训练最终模型
        print("\n使用最佳超参数训练最终模型...")
        best_model, test_metrics, test_preds, test_targets = train_with_optimal_params(
            best_params=best_params,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            input_dim=input_dim,
            device=device,
            output_dim=len(args.target_cols),
            num_epochs=args.epochs,
            model_dir=args.model_dir,
            target_scaler=target_scaler
        )
    else:
        # 直接使用默认参数训练模型
        print("\n使用默认参数训练模型...")
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        # 创建模型
        model = GIN(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=len(args.target_cols),
            num_layers=args.num_layers,
            dropout=args.dropout,
            pooling=args.pooling
        )
        model = model.to(device)
        
        # 如果存在预训练的模型，则加载它
        if args.load_model and os.path.exists(os.path.join(args.model_dir, args.model_name)):
            model = load_model(model, os.path.join(args.model_dir, args.model_name), device)
        else:
            # 创建优化器和学习率调度器
            optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
            
            # 训练模型
            model, history = train_gin_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=args.epochs,
                device=device,
                target_scaler=target_scaler
            )
            
            # 保存模型
            save_model(model, model_dir=args.model_dir, model_name=args.model_name)
            
            # 绘制训练曲线
            plot_training_curves(
                train_losses=history['train_loss'],
                val_losses=history['val_loss'],
                metrics_history=history['metrics'],
                save_dir=args.output_dir
            )
        
        # 在测试集上评估模型
        test_preds, test_targets, test_metrics = evaluate_model(model, test_loader, device, target_scaler)
        print("\n测试集评估结果:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # 保存为best_model以便后续步骤使用
        best_model = model
    
    # 绘制预测散点图
    plot_prediction_scatter(
        y_true=test_targets,
        y_pred=test_preds,
        target_names=args.target_cols,
        save_dir=args.output_dir
    )
    
    # 模型解释 
    if args.explain:
        print("\n开始模型解释...")
        test_smiles = test_df[args.smiles_col].tolist()
        test_loader_single = DataLoader(test_dataset, batch_size=1, shuffle=False)
        explain_model_predictions(
            model=best_model,
            data_loader=test_loader_single,
            smiles_list=test_smiles,
            device=device,
            save_dir=args.output_dir,
            target_scaler=target_scaler
        )
    
    # 预测新分子
    if args.predict and os.path.exists(args.predict_file):
        print(f"\n预测文件 {args.predict_file} 中的分子...")
        prediction_dataset, pred_smiles, target_scaler = load_prediction_data(
            args.predict_file, 
            smiles_col=args.smiles_col,
            target_scaler=target_scaler
        )
        
        if len(prediction_dataset) > 0:
            # 创建预测数据加载器
            pred_loader = DataLoader(prediction_dataset, batch_size=args.batch_size)
            
            # 进行预测
            best_model.eval()
            predictions = []
            
            with torch.no_grad():
                for data in pred_loader:
                    data = data.to(device)
                    output = best_model(data)
                    predictions.append(output.cpu().numpy())
            
            # 合并批次结果
            if len(predictions) > 0:
                predictions = np.vstack(predictions)
                
                # 如果使用了归一化，将预测值转换回原始尺度
                if target_scaler is not None:
                    predictions = inverse_normalize_targets(predictions, target_scaler)
                
                # 将预测结果保存到CSV
                save_predictions_to_csv(
                    smiles_list=pred_smiles,
                    predictions=predictions,
                    filename=os.path.join(args.output_dir, 'predictions.csv'),
                    target_names=args.target_cols
                )
                print(f"预测结果已保存到 {os.path.join(args.output_dir, 'predictions.csv')}")
            else:
                print("未生成预测结果。")
        else:
            print(f"预测文件 {args.predict_file} 中没有有效的分子。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GIN分子性质预测')
    
    # 数据参数
    parser.add_argument('--data_file', type=str, default='database.csv', help='数据文件路径')
    parser.add_argument('--smiles_col', type=str, default='smiles', help='SMILES列名')
    parser.add_argument('--target_cols', type=str, nargs='+', default=['homo', 'lumo', 'energy'], help='目标列名')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.16, help='验证集比例')
    parser.add_argument('--no_normalize', action='store_true', help='禁用目标值归一化')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3, help='GIN层数')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout率')
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'sum', 'max'], help='图池化方法')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--no_cuda', action='store_true', help='禁用CUDA')
    
    # 超参数优化
    parser.add_argument('--optimize', action='store_true', help='执行贝叶斯超参数优化')
    parser.add_argument('--opt_iterations', type=int, default=60, help='贝叶斯优化迭代次数')
    parser.add_argument('--opt_initial_points', type=int, default=10, help='贝叶斯优化初始点数量')
    
    # 模型解释
    parser.add_argument('--explain', action='store_true', help='使用GNNExplainer解释模型')
    
    # 预测
    parser.add_argument('--predict', action='store_true', help='预测新分子')
    parser.add_argument('--predict_file', type=str, default='TXs.csv', help='包含要预测的分子的CSV文件')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='results', help='输出目录')
    parser.add_argument('--model_dir', type=str, default='models', help='模型保存目录')
    parser.add_argument('--model_name', type=str, default='gin_model.pt', help='模型文件名')
    parser.add_argument('--load_model', action='store_true', help='加载现有模型')
    
    args = parser.parse_args()
    
    # 检查目标列是否有效
    if not args.target_cols or len(args.target_cols) == 0:
        args.target_cols = ['homo', 'lumo', 'energy']
    
    # 运行主程序
    main(args) 