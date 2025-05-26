# 10折交叉验证使用说明

本项目已添加K折交叉验证功能，用于更全面地评估GIN模型的性能和稳定性。交叉验证方法相比简单的训练/验证/测试集分割更能客观评估模型性能，特别是在数据集较小时。

## 功能特点

1. 实现了标准的k折交叉验证（默认为10折）
2. 每一折内部使用验证集进行早停，防止过拟合
3. 生成详细的性能指标统计和可视化：
   - 所有折的详细指标（MAE、RMSE、R²）
   - 箱型图显示各个指标在不同折上的分布
   - 各个目标属性的单独指标分析
   - 详细的汇总统计（均值、标准差）
4. 支持使用贝叶斯优化的最佳超参数进行交叉验证

## 使用方法

### 基本用法

使用默认参数运行10折交叉验证：

```bash
python run_kfold_cv.py --data_file database.csv
```

### 结合贝叶斯优化结果

使用贝叶斯优化找到的最佳超参数进行交叉验证（推荐方式）：

```bash
python run_kfold_cv.py --data_file database.csv --use_bayesian_opt --bayes_results_file results/bayesian_optimization_results.txt
```

这种方法先通过贝叶斯优化找到最优超参数配置，再使用这组参数进行交叉验证，能够更准确地评估模型的泛化性能。

### 自定义参数

您可以自定义各种参数来控制交叉验证过程：

```bash
python run_kfold_cv.py \
  --data_file database.csv \
  --n_splits 10 \
  --hidden_dim 128 \
  --num_layers 3 \
  --dropout 0.3 \
  --pooling mean \
  --batch_size 64 \
  --epochs 100 \
  --lr 0.001 \
  --output_dir results/kfold_cv
```

### 参数说明

#### 数据参数
- `--data_file`: 数据文件路径（默认: database.csv）
- `--smiles_col`: SMILES列名（默认: smiles）
- `--target_cols`: 目标列名列表（默认: homo lumo energy）
- `--no_normalize`: 禁用目标值归一化

#### 交叉验证参数
- `--n_splits`: 交叉验证折数（默认: 10）

#### 模型参数
- `--hidden_dim`: 隐藏层维度（默认: 128）
- `--num_layers`: GIN层数（默认: 3）
- `--dropout`: Dropout率（默认: 0.3）
- `--pooling`: 图池化方法（默认: mean，可选: mean, sum, max）

#### 训练参数
- `--batch_size`: 批量大小（默认: 64）
- `--epochs`: 每一折的训练轮数（默认: 100）
- `--lr`: 学习率（默认: 0.001）
- `--weight_decay`: 权重衰减（默认: 1e-5）
- `--no_cuda`: 禁用CUDA

#### 贝叶斯优化参数
- `--use_bayesian_opt`: 使用贝叶斯优化结果的最佳超参数
- `--bayes_results_file`: 贝叶斯优化结果文件路径（默认: results/bayesian_optimization_results.txt）

#### 输出参数
- `--output_dir`: 输出目录（默认: results/kfold_cv）

## 输出文件

交叉验证完成后，以下文件将保存在指定的输出目录中：

1. `kfold_cross_validation_results.csv`: 包含每一折的详细指标结果
2. `kfold_summary.txt`: 包含汇总统计信息（均值±标准差）
3. `kfold_boxplots.png`: 箱型图显示各指标的分布
4. `kfold_comparison.png`: 各折间的性能比较图

## 如何解读结果

### 模型稳定性分析

- **标准差越小**表示模型在不同数据子集上的表现越稳定
- 箱型图中的**箱体高度**反映了指标的变异程度，箱体越短说明模型越稳定
- 如果某些折的性能显著低于其他折，可能表明数据集中存在特异的子集

### 性能评估

- 使用平均指标作为模型的整体性能评估
- 与单次分割的测试集结果相比，交叉验证能提供更全面、更客观的评估
- 各个目标属性的单独指标可以帮助分析模型在不同预测任务上的表现差异

## 实现细节

交叉验证的核心逻辑在`utils.py`中的`kfold_cross_validation`函数中实现。这个函数：

1. 将数据集分成k个子集（折）
2. 对每一折，使用k-1个子集作为训练数据，剩下的1个子集作为测试数据
3. 在训练数据中再划分出一部分作为验证集用于早停
4. 训练模型并在测试子集上评估性能
5. 汇总所有折的结果并生成可视化

通过这种方式，我们能够充分利用所有数据，得到更可靠的模型性能评估。

## 关于两阶段方法：贝叶斯优化 + 交叉验证

该实现使用了"两阶段"方法来优化和评估模型：

1. **第一阶段（贝叶斯优化）**：使用贝叶斯优化在固定的训练/验证集分割上寻找最佳超参数
2. **第二阶段（交叉验证）**：使用找到的最佳超参数进行k折交叉验证，全面评估模型性能和稳定性

这种方法比完全嵌套的交叉验证（每一折内部再进行超参数优化）计算效率更高，同时仍能有效评估模型的泛化能力。 