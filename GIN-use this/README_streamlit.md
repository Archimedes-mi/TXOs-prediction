# 分子性质预测与探索 Streamlit 应用

本项目是一个基于 Streamlit 的交互式应用程序，用于预测和探索分子的电子性质，包括 α-HOMO、β-LUMO 和 delta-E 值。

## 功能特点

1. **SMILES to Properties**：输入 SMILES 分子表示，预测其电子性质
2. **Properties to SMILES**：根据 delta-E 值范围筛选分子库中的分子
3. **Database Exploration**：探索分子数据库的统计特性和分布

## 安装说明

### 前提条件

- Python 3.8 或更高版本
- pip 包管理器

### 安装步骤

1. 克隆或下载本仓库：

```bash
git clone <repository-url>
cd <repository-folder>
```

2. 安装依赖项：

```bash
pip install -r requirements.txt
```

注意：由于 RDKit 和 PyTorch Geometric 的安装可能会比较复杂，建议使用 conda 环境：

```bash
conda create -n molecule-app python=3.8
conda activate molecule-app
conda install -c conda-forge rdkit
pip install torch==2.0.1
pip install torch-geometric==2.3.1
pip install -r requirements.txt
```

## 运行应用

安装完依赖后，只需运行以下命令启动 Streamlit 应用：

```bash
streamlit run app.py
```

应用将在浏览器中自动打开，默认地址是 http://localhost:8501

## 使用指南

### SMILES to Properties

1. 在侧边栏选择 "SMILES to Properties" 模式
2. 在输入框中输入有效的 SMILES 字符串
3. 应用将显示该分子的结构图及预测的 α-HOMO、β-LUMO 和 delta-E 值

### Properties to SMILES

1. 在侧边栏选择 "Properties to SMILES" 模式
2. 设置 delta-E 值的范围
3. 应用将显示符合条件的所有分子及其性质

### Database Exploration

1. 在侧边栏选择 "Database Exploration" 模式
2. 浏览数据库统计信息和分布图表
3. 使用高级筛选功能根据多条件筛选分子

## 文件结构

- `app.py`: Streamlit 应用程序主文件
- `models/gin_optimal_model.pt`: 优化后的 GIN 模型
- `results/predictions.csv`: 包含预测结果的数据集
- `requirements.txt`: 依赖项列表

## 技术细节

该应用使用：
- Streamlit 构建交互式界面
- PyTorch 和 PyTorch Geometric 加载和运行图神经网络模型
- RDKit 处理分子结构和可视化
- Pandas、Matplotlib 和 Seaborn 进行数据处理和可视化

## 问题排查

如果遇到 RDKit 相关错误，请确保通过 conda 正确安装：
```bash
conda install -c conda-forge rdkit
```

如果遇到 CUDA 相关错误，请确保安装了与您系统兼容的 PyTorch 版本。 