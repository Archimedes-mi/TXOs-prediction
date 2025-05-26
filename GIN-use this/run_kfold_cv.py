import os
import torch
import pandas as pd
import argparse
from model import GIN
from utils import kfold_cross_validation

def read_bayesian_optimization_results(params_file):
    """
    从贝叶斯优化结果文件中读取最佳超参数
    
    参数:
        params_file: 贝叶斯优化结果文件路径
    
    返回:
        best_params: 最佳超参数字典
    """
    best_params = {}
    
    if os.path.exists(params_file):
        print(f"从 {params_file} 读取最佳超参数...")
        
        with open(params_file, 'r') as f:
            lines = f.readlines()
            # 提取最佳超参数部分
            for i, line in enumerate(lines):
                if line.strip() == "最佳超参数:":
                    # 读取参数直到遇到空行
                    j = i + 1
                    while j < len(lines) and lines[j].strip() != "":
                        if ":" in lines[j]:
                            param, value = lines[j].strip().split(":", 1)
                            param = param.strip()
                            value = value.strip()
                            # 尝试转换为适当的类型
                            try:
                                if '.' in value:
                                    best_params[param] = float(value)
                                else:
                                    best_params[param] = int(value)
                            except ValueError:
                                best_params[param] = value
                        j += 1
                    break
        
        # 打印找到的参数
        if best_params:
            print("读取的最佳超参数:")
            for param, value in best_params.items():
                print(f"  {param}: {value}")
        else:
            print("警告: 无法从文件中读取到超参数")
    else:
        print(f"警告: 贝叶斯优化结果文件 {params_file} 不存在")
    
    return best_params

def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    df = pd.read_csv(args.data_file)
    print(f"加载了 {len(df)} 个分子")
    
    # 确定输入维度 
    # 注意：这里我们需要创建一个临时图来获取输入维度
    from utils import smiles_to_graph
    temp_graph = smiles_to_graph(df[args.smiles_col].iloc[0])
    if temp_graph is not None:
        input_dim = temp_graph.x.shape[1]
    else:
        print("错误: 无法确定输入维度！")
        return
    
    # 初始化超参数
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    dropout = args.dropout
    pooling = args.pooling
    batch_size = args.batch_size
    learning_rate = args.lr
    weight_decay = args.weight_decay
    
    # 如果指定了使用贝叶斯优化结果
    if args.use_bayesian_opt:
        # 读取贝叶斯优化的最佳超参数
        best_params = read_bayesian_optimization_results(args.bayes_results_file)
        
        # 使用最佳超参数覆盖默认值（如果存在）
        if 'hidden_dim' in best_params:
            hidden_dim = best_params['hidden_dim']
        if 'num_layers' in best_params:
            num_layers = best_params['num_layers']
        if 'dropout' in best_params:
            dropout = best_params['dropout']
        if 'pooling' in best_params:
            pooling = best_params['pooling']
        if 'batch_size' in best_params:
            batch_size = best_params['batch_size']
        if 'learning_rate' in best_params:
            learning_rate = best_params['learning_rate']
        if 'weight_decay' in best_params:
            weight_decay = best_params['weight_decay']
        
        print("\n使用贝叶斯优化的最佳超参数进行交叉验证...")
    else:
        print("\n使用命令行参数指定的超参数进行交叉验证...")
    
    # 打印将要使用的超参数
    print(f"超参数设置:")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  num_layers: {num_layers}")
    print(f"  dropout: {dropout}")
    print(f"  pooling: {pooling}")
    print(f"  batch_size: {batch_size}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  weight_decay: {weight_decay}")
    
    # 如果提供了预训练模型路径
    if args.pretrained_model_path:
        print(f"  使用预训练模型: {args.pretrained_model_path}")
    
    # 执行k折交叉验证
    print(f"\n开始执行{args.n_splits}折交叉验证...")
    results_df, metrics_dict = kfold_cross_validation(
        model_class=GIN,
        data_df=df,
        smiles_col=args.smiles_col,
        target_cols=args.target_cols,
        input_dim=input_dim,
        n_splits=args.n_splits,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        pooling=pooling,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=args.epochs,
        device=device,
        output_dir=args.output_dir,
        normalize=not args.no_normalize,
        pretrained_model_path=args.pretrained_model_path
    )
    
    # 打印最终结果
    print("\n交叉验证完成!")
    print(f"结果已保存到 {args.output_dir} 目录")
    print(f"- kfold_cross_validation_results.csv: 包含每一折的详细结果")
    print(f"- kfold_summary.txt: 包含汇总指标")
    print(f"- kfold_boxplots.png: 箱型图可视化")
    print(f"- kfold_comparison.png: 各折对比图")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GIN模型k折交叉验证')
    
    # 数据参数
    parser.add_argument('--data_file', type=str, default='database.csv', help='数据文件路径')
    parser.add_argument('--smiles_col', type=str, default='smiles', help='SMILES列名')
    parser.add_argument('--target_cols', type=str, nargs='+', default=['homo', 'lumo', 'energy'], help='目标列名')
    parser.add_argument('--no_normalize', action='store_true', help='禁用目标值归一化')
    
    # 交叉验证参数
    parser.add_argument('--n_splits', type=int, default=5, help='交叉验证折数，默认为5折')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=60, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3, help='GIN层数')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout率')
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'sum', 'max'], help='图池化方法')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--no_cuda', action='store_true', help='禁用CUDA')
    
    # 贝叶斯优化参数
    parser.add_argument('--use_bayesian_opt', action='store_true', help='使用贝叶斯优化结果的最佳超参数')
    parser.add_argument('--bayes_results_file', type=str, default='results/bayesian_optimization_results.txt', 
                       help='贝叶斯优化结果文件路径')
    
    # 预训练模型参数
    parser.add_argument('--pretrained_model_path', type=str, default='models/gin_optimal_model.pt', 
                      help='预训练模型路径，默认使用 models/gin_optimal_model.pt')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='results/kfold_cv', help='输出目录')
    
    args = parser.parse_args()
    
    # 检查目标列是否有效
    if not args.target_cols or len(args.target_cols) == 0:
        args.target_cols = ['homo', 'lumo', 'energy']
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行主程序
    main(args) 