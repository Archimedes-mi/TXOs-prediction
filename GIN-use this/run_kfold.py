import os
import sys
import warnings

# 忽略PyTorch学习率调度器的警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

# 以"仅训练模型"模式运行程序
if __name__ == "__main__":

    # 构建命令并执行
    command = f"python run_kfold_cv.py --use_bayesian_opt --pretrained_model_path models/gin_optimal_model.pt"
    print(f"执行命令: {command}")
    os.system(command) 

#依次使用这些命令
#python main.py --data_file database.csv --optimize