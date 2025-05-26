import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import smiles_to_graph

class MoleculeDataset(Dataset):
    """
    分子数据集类，用于从SMILES加载分子图
    """
    def __init__(self, root, df, smiles_col='smiles', target_cols=None, transform=None, pre_transform=None, target_scaler=None):
        """
        初始化分子数据集
        
        参数:
            root: 数据存储目录
            df: 包含分子信息的DataFrame
            smiles_col: SMILES列名
            target_cols: 目标属性列名列表
            transform: 数据转换函数
            pre_transform: 预处理转换函数
            target_scaler: 目标值归一化器（用于测试/验证集）
        """
        self.df = df
        self.smiles_col = smiles_col
        self.target_cols = target_cols if target_cols is not None else []
        self.target_scaler = target_scaler
        self.processed_data = []
        
        # 处理每个分子
        for i, row in self.df.iterrows():
            smiles = row[smiles_col]
            graph_data = smiles_to_graph(smiles)
            
            if graph_data is not None:
                # 添加标签
                if self.target_cols:
                    y = torch.tensor([row[col] for col in self.target_cols], dtype=torch.float)
                    graph_data.y = y
                
                self.processed_data.append(graph_data)
        
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return []
    
    def len(self):
        return len(self.processed_data)
    
    def get(self, idx):
        return self.processed_data[idx]

def normalize_targets(train_df, val_df, test_df, target_cols):
    """
    对目标值进行归一化处理
    
    参数:
        train_df, val_df, test_df: 训练、验证和测试集DataFrame
        target_cols: 目标属性列名列表
    
    返回:
        train_df, val_df, test_df: 目标值归一化后的DataFrame
        scaler: 归一化器，用于后续逆变换
    """
    # 创建归一化器
    scaler = StandardScaler()
    
    # 使用训练集拟合归一化器
    scaler.fit(train_df[target_cols].values)
    
    # 应用归一化
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df[target_cols] = scaler.transform(train_df[target_cols].values)
    val_df[target_cols] = scaler.transform(val_df[target_cols].values)
    test_df[target_cols] = scaler.transform(test_df[target_cols].values)
    
    print("目标值已归一化，标准差:")
    for i, col in enumerate(target_cols):
        print(f"  {col}: {scaler.scale_[i]:.6f}")
    
    return train_df, val_df, test_df, scaler

def inverse_normalize_targets(predictions, scaler, target_cols=None):
    """
    将归一化后的预测值转换回原始尺度
    
    参数:
        predictions: 模型的标准化预测值
        scaler: 用于归一化的StandardScaler
        target_cols: 目标属性列名列表（仅用于打印信息）
    
    返回:
        inverse_predictions: 原始尺度的预测值
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    
    # 逆变换
    inverse_predictions = scaler.inverse_transform(predictions)
    
    return inverse_predictions

def load_and_split_data(csv_file, smiles_col='smiles', target_cols=None, test_size=0.2, val_size=0.15, random_state=42, normalize=True):
    """
    加载CSV数据并分割为训练、验证和测试集
    
    参数:
        csv_file: CSV文件路径
        smiles_col: SMILES列名
        target_cols: 目标属性列名列表
        test_size: 测试集比例
        val_size: 验证集比例
        random_state: 随机种子
        normalize: 是否对目标值进行归一化
    
    返回:
        train_dataset, val_dataset, test_dataset: 训练、验证和测试数据集
        train_df, val_df, test_df: 相应的DataFrame
        target_scaler: 目标值归一化器（如果normalize=True）
    """
    # 加载数据
    df = pd.read_csv(csv_file)
    print(f"加载了 {len(df)} 个分子")
    
    # 设置默认目标列
    if target_cols is None:
        target_cols = ['homo', 'lumo', 'energy']
    
    # 分割数据
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df, val_df = train_test_split(train_df, test_size=val_size/(1-test_size), random_state=random_state)
    
    print(f"训练集: {len(train_df)} 个分子")
    print(f"验证集: {len(val_df)} 个分子")
    print(f"测试集: {len(test_df)} 个分子")
    
    # 对目标值进行归一化处理
    target_scaler = None
    if normalize:
        train_df, val_df, test_df, target_scaler = normalize_targets(
            train_df, val_df, test_df, target_cols
        )
    
    # 创建数据集
    train_dataset = MoleculeDataset(root='data/train', df=train_df, smiles_col=smiles_col, target_cols=target_cols)
    val_dataset = MoleculeDataset(root='data/val', df=val_df, smiles_col=smiles_col, target_cols=target_cols, target_scaler=target_scaler)
    test_dataset = MoleculeDataset(root='data/test', df=test_df, smiles_col=smiles_col, target_cols=target_cols, target_scaler=target_scaler)
    
    return train_dataset, val_dataset, test_dataset, train_df, val_df, test_df, target_scaler

def load_prediction_data(csv_file, smiles_col='smiles', target_scaler=None):
    """
    加载需要预测的数据
    
    参数:
        csv_file: CSV文件路径
        smiles_col: SMILES列名
        target_scaler: 目标值归一化器（用于后处理）
    
    返回:
        prediction_dataset: 预测数据集
        smiles_list: SMILES字符串列表
        target_scaler: 目标值归一化器
    """
    # 加载数据
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        # 创建一个空的DataFrame用于测试
        df = pd.DataFrame({smiles_col: []})
    
    # 创建数据集
    prediction_dataset = MoleculeDataset(root='data/predict', df=df, smiles_col=smiles_col, target_scaler=target_scaler)
    smiles_list = df[smiles_col].tolist()
    
    return prediction_dataset, smiles_list, target_scaler 