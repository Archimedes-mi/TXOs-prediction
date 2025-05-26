import os
import sys
import warnings
import torch
import json

# 忽略PyTorch学习率调度器的警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

# 以"加载最佳模型"模式运行程序
if __name__ == "__main__":
    # 检查最佳超参数文件是否存在
    results_dir = 'results'
    params_file = os.path.join(results_dir, 'bayesian_optimization_results.txt')
    
    # 读取贝叶斯优化结果
    if os.path.exists(params_file):
        print(f"从 {params_file} 读取最佳超参数...")
        
        # 初始化参数字典
        best_params = {}
        
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
                            value = value.strip()
                            # 尝试转换为适当的类型
                            try:
                                # 尝试作为数字处理
                                if '.' in value:
                                    best_params[param] = float(value)
                                else:
                                    best_params[param] = int(value)
                            except ValueError:
                                # 如果不是数字，保持为字符串
                                best_params[param] = value
                        j += 1
                    break
        
        # 打印找到的参数
        print("读取的最佳超参数:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        # 构建命令并执行
        if 'hidden_dim' in best_params and 'num_layers' in best_params:
            command = f"python main.py --data_file database.csv --load_model --model_name gin_optimal_model.pt --explain"
            command += f" --hidden_dim {best_params.get('hidden_dim', 158)}"
            command += f" --num_layers {best_params.get('num_layers', 4)}"
            command += f" --dropout {best_params.get('dropout', 0.3)}"
            command += f" --pooling {best_params.get('pooling', 'mean')}"
            
            print(f"执行命令: {command}")
            os.system(command)
        else:
            print("错误: 无法从贝叶斯优化结果文件中提取必要的参数。")
            print("将使用默认参数运行...")
            command = f"python main.py --data_file database.csv --load_model --model_name gin_optimal_model.pt --explain"
            os.system(command)
    else:
        print(f"警告: 未找到贝叶斯优化结果文件 {params_file}")
        print("尝试直接使用保存的模型参数...")
        
        # 从模型中获取超参数
        model_path = os.path.join('models', 'gin_optimal_model.pt')
        if os.path.exists(model_path):
            # 尝试加载模型参数
            model_state = torch.load(model_path, map_location=torch.device('cpu'))
            
            # 检查隐藏层维度
            hidden_dim = 158  # 从错误消息中推断的值
            num_layers = 4    # 从错误消息中推断的值
            
            command = f"python main.py --data_file database.csv --load_model --model_name gin_optimal_model.pt --explain"
            command += f" --hidden_dim {hidden_dim} --num_layers {num_layers}"
            
            print(f"执行命令: {command}")
            os.system(command)
        else:
            print(f"错误: 未找到模型文件 {model_path}")
            sys.exit(1) 