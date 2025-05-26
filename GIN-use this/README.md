# GIN 分子性质预测

基于图同构网络(GIN)的多目标分子性质预测模型，用于预测分子的HOMO、LUMO和三重态激发能。

## 项目概述

本项目使用PyTorch Geometric (PyG)构建了基于图同构网络(GIN)的模型，从SMILES分子表示中学习预测HOMO、LUMO和三重态激发能值。

主要功能包括：

1. 分子图表示学习与特征提取
2. 多目标GIN模型构建与训练
3. 贝叶斯优化超参数调优
4. 模型性能评估与可视化
5. GNNExplainer模型解释
6. 新分子性质预测

## 环境配置

创建并激活Python环境后，安装所需依赖：

```bash
pip install -r requirements.txt
```

## 数据集

项目使用database.csv数据集，包含分子的SMILES表示及其HOMO、LUMO和能量值。

## 使用方法

### 模型训练

使用默认参数训练模型：

```bash
python main.py --data_file database.csv
```

### 超参数优化

执行贝叶斯优化搜索最佳超参数：

```bash
python main.py --data_file database.csv --optimize
```

### 模型解释

使用GNNExplainer解释模型预测：

```bash
python main.py --data_file database.csv --load_model --model_name gin_optimal_model.pt --explain
```

### 预测新分子

对新分子进行性质预测：

```bash
python main.py --load_model --predict --predict_file TXs.csv
```

### 完整流程

一次性执行完整的训练、优化、解释和预测流程：

```bash
python main.py --data_file database.csv --optimize --explain --predict --predict_file TXs.csv
```

## 主要组件

- `main.py`: 主程序入口
- `model.py`: GIN模型定义
- `dataset.py`: 数据集加载与处理
- `train.py`: 模型训练函数
- `bayesian_opt.py`: 贝叶斯优化功能
- `gnn_explainer.py`: 模型解释功能
- `utils.py`: 工具函数和可视化功能

## 结果输出

模型训练和评估的结果将保存在以下位置：

- 模型权重: `models/`目录
- 性能指标与可视化: `results/`目录
- 预测结果: `results/predictions.csv`
- 模型解释可视化: `results/`目录 