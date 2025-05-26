# 分子性质预测深度神经网络模型

本项目实现了一个基于深度神经网络的分子性质预测模型，用于从SMILES表示预测分子的HOMO、LUMO能级(eV)和能量(kcal/mol)。

## 特点

- 基于分子SMILES表示构建三目标深度神经网络
- 使用贝叶斯优化方法进行超参数优化
- 提供多种激活函数选择（Sigmoid、ReLU、Leaky ReLU、Tanh、ELU、SELU）
- 使用PyTorch实现，支持GPU加速
- 模型性能可视化和评估
- 可用于新分子的性质预测

## 环境要求

- Python 3.6+
- PyTorch 1.7+
- RDKit
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Optuna
- Joblib

## 安装依赖

```bash
pip install torch rdkit numpy pandas scikit-learn matplotlib seaborn optuna joblib
```

## 数据集

- `database.csv`: 包含分子SMILES及其对应的HOMO、LUMO和能量值
- `TXs.csv`: 需要预测性质的分子SMILES

## 使用方法

1. 确保数据集文件在正确位置
2. 运行主脚本进行模型训练和预测

```bash
python mol_nn_model.py
```

3. 结果将保存在以下位置：
   - 模型性能可视化图表: `results/`目录
   - 最佳模型: `results/best_model.pt`
   - 定标器: `results/feature_scaler.pkl`和`results/target_scaler.pkl`
   - 预测结果: `TXs_predicted.csv`

## 代码结构

- `smiles_to_features()`: 将SMILES转换为分子特征向量（Morgan指纹）
- `MolecularPropertyNN`: 定义三目标深度神经网络模型类
- `load_and_preprocess_data()`: 加载和预处理数据
- `train_model()`: 训练模型函数
- `evaluate_model()`: 评估模型性能
- `visualize_performance()`: 可视化模型性能
- `objective()`: Optuna贝叶斯优化目标函数
- `predict_with_model()`: 使用模型进行预测
- `main()`: 主函数

## 模型架构

该模型使用Morgan分子指纹作为输入特征，通过贝叶斯优化确定最佳网络结构和超参数。模型包括：

1. 共享特征提取层：从分子指纹中提取通用特征
2. 三个专用输出分支：分别预测HOMO、LUMO和能量

## 性能评估

模型性能通过以下指标进行评估：
- 平均绝对误差（MAE）
- 均方根误差（RMSE）
- 决定系数（R²）

性能可视化结果保存在`results/`目录中。 