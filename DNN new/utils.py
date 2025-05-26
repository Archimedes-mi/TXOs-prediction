import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from torch_geometric.data import Data, Batch
import torch
import networkx as nx
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

# 设置蓝绿配色
COLORS = {
    'blue': '#1f77b4',
    'teal': '#39ac73',
    'dark_blue': '#035096',
    'cyan': '#40E0D0'
}

# 创建自定义蓝绿色配色方案
BLUE_GREEN_CMAP = LinearSegmentedColormap.from_list(
    'blue_green', [COLORS['dark_blue'], COLORS['blue'], COLORS['teal'], COLORS['cyan']]
)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

def set_plotting_style():
    """设置绘图样式"""
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.2)

def smiles_to_graph(smiles):
    """
    将SMILES字符串转换为分子图
    
    参数:
        smiles: SMILES字符串
    
    返回:
        PyG的Data对象，包含节点特征和边索引
    """
    # 解析SMILES字符串
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # 计算摩根指纹作为节点特征
    num_atoms = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        atom_features = []
        # 原子类型的one-hot编码
        atom_type = atom.GetAtomicNum()
        if atom_type == 6:  # 碳
            atom_features += [1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif atom_type == 7:  # 氮
            atom_features += [0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif atom_type == 8:  # 氧
            atom_features += [0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif atom_type == 9:  # 氟
            atom_features += [0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif atom_type == 16:  # 硫
            atom_features += [0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif atom_type == 17:  # 氯
            atom_features += [0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif atom_type == 35:  # 溴
            atom_features += [0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif atom_type == 53:  # 碘
            atom_features += [0, 0, 0, 0, 0, 0, 0, 1, 0]
        else:  # 其他
            atom_features += [0, 0, 0, 0, 0, 0, 0, 0, 1]
        
        # 添加其他原子特征
        atom_features.append(atom.GetFormalCharge())
        atom_features.append(atom.GetNumExplicitHs())
        atom_features.append(atom.GetIsAromatic())
        atom_features.append(atom.GetDegree())
        atom_features.append(atom.GetTotalNumHs())
        
        # Morgan指纹局部环境特征
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=32, fromAtoms=[atom.GetIdx()])
        morgan_features = [int(bit) for bit in morgan_fp.ToBitString()]
        atom_features.extend(morgan_features)
        
        features.append(atom_features)
    
    # 节点特征矩阵
    x = torch.tensor(features, dtype=torch.float)
    
    # 边索引（分子中的化学键）
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # 无向图需要添加两个方向的边
        edge_indices.append([i, j])
        edge_indices.append([j, i])
    
    if len(edge_indices) == 0:  # 处理没有边的情况
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    # 边特征（键类型）
    edge_attr = []
    for bond in mol.GetBonds():
        # 键类型特征
        bond_type = bond.GetBondType()
        if bond_type == Chem.rdchem.BondType.SINGLE:
            edge_attr.extend([[1, 0, 0, 0], [1, 0, 0, 0]])
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            edge_attr.extend([[0, 1, 0, 0], [0, 1, 0, 0]])
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            edge_attr.extend([[0, 0, 1, 0], [0, 0, 1, 0]])
        else:  # 芳香键或其他
            edge_attr.extend([[0, 0, 0, 1], [0, 0, 0, 1]])
    
    if len(edge_attr) == 0:
        edge_attr = torch.zeros((0, 4), dtype=torch.float)
    else:
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # 创建Data对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data

def mol2graph(mol_list):
    """
    将分子对象列表转换为分子图
    
    参数:
        mol_list: 分子对象列表（RDKit Mol对象）
    
    返回:
        PyG的Data对象，包含节点特征、边索引和分子的批处理信息
    """
    from torch_geometric.data import Batch, Data
    
    # 为每个分子创建图数据
    data_list = []
    for mol in mol_list:
        if mol is None:
            continue
        
        # 计算节点特征
        num_atoms = mol.GetNumAtoms()
        features = []
        for atom in mol.GetAtoms():
            atom_features = []
            # 原子类型的one-hot编码
            atom_type = atom.GetAtomicNum()
            if atom_type == 6:  # 碳
                atom_features += [1, 0, 0, 0, 0, 0, 0, 0, 0]
            elif atom_type == 7:  # 氮
                atom_features += [0, 1, 0, 0, 0, 0, 0, 0, 0]
            elif atom_type == 8:  # 氧
                atom_features += [0, 0, 1, 0, 0, 0, 0, 0, 0]
            elif atom_type == 9:  # 氟
                atom_features += [0, 0, 0, 1, 0, 0, 0, 0, 0]
            elif atom_type == 16:  # 硫
                atom_features += [0, 0, 0, 0, 1, 0, 0, 0, 0]
            elif atom_type == 17:  # 氯
                atom_features += [0, 0, 0, 0, 0, 1, 0, 0, 0]
            elif atom_type == 35:  # 溴
                atom_features += [0, 0, 0, 0, 0, 0, 1, 0, 0]
            elif atom_type == 53:  # 碘
                atom_features += [0, 0, 0, 0, 0, 0, 0, 1, 0]
            else:  # 其他
                atom_features += [0, 0, 0, 0, 0, 0, 0, 0, 1]
            
            # 添加其他原子特征
            atom_features.append(atom.GetFormalCharge())
            atom_features.append(atom.GetNumExplicitHs())
            atom_features.append(atom.GetIsAromatic())
            atom_features.append(atom.GetDegree())
            atom_features.append(atom.GetTotalNumHs())
            
            # Morgan指纹局部环境特征 - 添加更多特征以达到总计46个特征
            morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=32, fromAtoms=[atom.GetIdx()])
            morgan_features = [int(bit) for bit in morgan_fp.ToBitString()]
            atom_features.extend(morgan_features)
            
            features.append(atom_features)
        
        # 节点特征矩阵
        x = torch.tensor(features, dtype=torch.float)
        
        # 边索引（分子中的化学键）
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # 无向图需要添加两个方向的边
            edge_indices.append([i, j])
            edge_indices.append([j, i])
        
        if len(edge_indices) == 0:  # 处理没有边的情况
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        
        # 创建Data对象
        data = Data(x=x, edge_index=edge_index)
        data_list.append(data)
    
    # 如果列表为空，创建一个空的Batch
    if not data_list:
        return None
    
    # 将所有分子图合并成一个批处理
    batch = Batch.from_data_list(data_list)
    
    return batch

def plot_training_curves(train_losses, val_losses, metrics_history, save_dir='results'):
    """
    绘制训练曲线
    
    参数:
        train_losses: 训练损失历史
        val_losses: 验证损失历史
        metrics_history: 评估指标历史
        save_dir: 保存图表的目录
    """
    os.makedirs(save_dir, exist_ok=True)
    set_plotting_style()
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color=COLORS['blue'], linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color=COLORS['teal'], linewidth=2, linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制评估指标
    for metric in ['mae', 'rmse', 'r2']:
        plt.figure(figsize=(10, 6))
        for target in ['homo', 'lumo', 'energy']:
            key = f'{metric}_{target}'
            if key in metrics_history:
                plt.plot(metrics_history[key], 
                         label=f'{target.upper()}', 
                         linewidth=2,
                         color=COLORS['blue'] if target == 'homo' else 
                               COLORS['teal'] if target == 'lumo' else 
                               COLORS['cyan'])
        
        plt.xlabel('Epochs')
        plt.ylabel(metric.upper())
        plt.title(f'{metric.upper()} During Training')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{metric}_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_prediction_scatter(y_true, y_pred, target_names, save_dir='results'):
    """
    绘制真实值与预测值的散点图
    
    参数:
        y_true: 真实值 shape=(n_samples, n_targets)
        y_pred: 预测值 shape=(n_samples, n_targets)
        target_names: 目标名称列表
        save_dir: 保存图表的目录
    """
    os.makedirs(save_dir, exist_ok=True)
    set_plotting_style()
    
    # 创建3x1布局的图表
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, target in enumerate(target_names):
        ax = axes[i]
        
        # 计算评估指标
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        
        # 绘制散点图
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.7, 
                   color=COLORS['blue'] if i == 0 else 
                         COLORS['teal'] if i == 1 else 
                         COLORS['cyan'])
        
        # 绘制理想线 (y=x)
        min_val = min(np.min(y_true[:, i]), np.min(y_pred[:, i]))
        max_val = max(np.max(y_true[:, i]), np.max(y_pred[:, i]))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
        
        # 设置轴标签和标题
        ax.set_xlabel(f'True {target.upper()}')
        ax.set_ylabel(f'Predicted {target.upper()}')
        ax.set_title(f'{target.upper()} Prediction Performance')
        
        # 添加性能指标文本框
        ax.text(0.05, 0.95, f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 显示网格
        ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_bayesian_optimization_results(opt_results, save_dir='results'):
    """
    绘制贝叶斯优化结果
    
    参数:
        opt_results: 优化结果字典
        save_dir: 保存图表的目录
    """
    os.makedirs(save_dir, exist_ok=True)
    set_plotting_style()
    
    # 提取优化迭代数据
    # 检查opt_results的结构并相应地处理
    if isinstance(opt_results, list):
        # 如果是列表格式，直接处理列表中的每个结果
        iterations = list(range(1, len(opt_results) + 1))
        target_values = [-res['target'] for res in opt_results]  # 转换为正值（我们最小化的是负R2）
        param_values = {param: [res['params'][param] for res in opt_results] 
                      for param in opt_results[0]['params'].keys()} if opt_results else {}
    else:
        # 原始代码的处理方式
        iterations = list(range(1, len(opt_results['target']) + 1))
        target_values = [-x for x in opt_results['target']]  # 转换为正值（我们最小化的是负R2）
        param_values = {param: [res[param] for res in opt_results['params']] 
                    for param in opt_results['params'][0].keys()} if opt_results['params'] else {}
    
    # 绘制优化过程
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, target_values, marker='o', linestyle='-', color=COLORS['blue'], linewidth=2)
    plt.axhline(y=max(target_values), color=COLORS['teal'], linestyle='--', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Validation Performance (R2)')
    plt.title('Bayesian Optimization Progress')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'bayesian_optimization_progress.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制参数值变化
    if not param_values:
        print("没有参数值可以绘制")
        return
        
    param_names = list(param_values.keys())
    
    # 每个参数一个子图
    fig, axes = plt.subplots(len(param_names), 1, figsize=(10, 3 * len(param_names)))
    
    for i, param in enumerate(param_names):
        if len(param_names) > 1:
            ax = axes[i]
        else:
            ax = axes
            
        # 绘制参数值
        scatter = ax.scatter(iterations, param_values[param], c=target_values, 
                             cmap=BLUE_GREEN_CMAP, s=70, alpha=0.8)
        
        # 标记最佳值
        best_idx = np.argmax(target_values)
        ax.scatter([iterations[best_idx]], [param_values[param][best_idx]], 
                   s=120, facecolors='none', edgecolors='red', linewidths=2)
        
        # 设置轴标签
        ax.set_xlabel('Iteration')
        ax.set_ylabel(param)
        ax.set_title(f'Parameter: {param}')
        ax.grid(True, linestyle='--', alpha=0.5)
    
    # 添加颜色条
    plt.tight_layout()
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Validation Performance (R2)')
    
    plt.savefig(os.path.join(save_dir, 'bayesian_optimization_parameters.png'), dpi=300, bbox_inches='tight')
    plt.close()

def visualize_molecule_with_explanation(smiles, atom_weights, filename='molecule_explanation.png'):
    """
    可视化分子结构并根据GNNExplainer的结果标记重要的原子和键
    
    参数:
        smiles: SMILES字符串
        atom_weights: 原子重要性权重
        filename: 保存的文件名
    """
    import numpy as np
    from rdkit import Chem
    try:
        from rdkit.Chem import Draw
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        print("无法导入rdMolDraw2D，尝试使用替代方法...")
        # 使用替代方法 - 基本的分子可视化
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"无法解析SMILES: {smiles}")
            return
            
        # 添加原子索引
        for atom in mol.GetAtoms():
            atom.SetProp("atomNote", str(atom.GetIdx()))
            
        # 使用基本绘图功能
        img = Draw.MolToImage(mol, size=(500, 500))
        img.save(filename)
        print(f"分子解释可视化已保存到 {filename} (简化版)")
        return
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"无法解析SMILES: {smiles}")
        return
    
    # 创建分子绘图对象
    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    drawer.drawOptions().addAtomIndices = True
    
    # 设置原子高亮颜色
    atom_cols = {}
    bond_cols = {}
    
    # 计算每个原子的重要性（行总和）
    if len(atom_weights) > 0:
        if atom_weights.ndim > 1:
            # 对于多维数组，计算每个原子（每行）的重要性总和
            atom_importances = np.sum(atom_weights, axis=1)
        else:
            # 对于一维数组，直接使用
            atom_importances = atom_weights
        
        # 确保原子数量与分子匹配
        num_atoms = min(len(atom_importances), mol.GetNumAtoms())
        atom_importances = atom_importances[:num_atoms]
        
        # 标准化权重
        if num_atoms > 0 and np.max(atom_importances) > np.min(atom_importances):
            norm_weights = (atom_importances - np.min(atom_importances)) / (np.max(atom_importances) - np.min(atom_importances))
            
            # 为每个原子设置颜色
            for i, weight in enumerate(norm_weights):
                if i < mol.GetNumAtoms():
                    r, g, b = 0, int(255 * float(weight)), int(200 * (1 - float(weight)))
                    atom_cols[i] = (r, g, b)
    
    # 绘制分子
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=list(range(mol.GetNumAtoms())), 
                                        highlightAtomColors=atom_cols, highlightBonds=[], 
                                        highlightBondColors=bond_cols)
    drawer.FinishDrawing()
    
    # 保存图像
    with open(filename, 'wb') as f:
        f.write(drawer.GetDrawingText())
    
    print(f"分子解释可视化已保存到 {filename}")

def evaluate_model(model, data_loader, device, target_scaler=None):
    """
    评估模型性能
    
    参数:
        model: GIN模型
        data_loader: 数据加载器
        device: 计算设备
        target_scaler: 目标值归一化器，用于逆转换
    
    返回:
        predictions: 原始尺度的预测值
        targets: 原始尺度的真实值
        metrics: 评估指标字典
    """
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            
            # 确保data.y的形状与output匹配
            batch_size = output.size(0)
            target = data.y.view(batch_size, -1)
            
            predictions.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
    
    # 合并批次结果
    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    
    # 如果使用了归一化，将预测值和真实值转换回原始尺度
    if target_scaler is not None:
        from dataset import inverse_normalize_targets
        original_predictions = inverse_normalize_targets(predictions, target_scaler)
        original_targets = inverse_normalize_targets(targets, target_scaler)
    else:
        original_predictions = predictions
        original_targets = targets
    
    # 计算每个目标的评估指标
    metrics = {}
    target_names = ['homo', 'lumo', 'energy']
    
    for i, target in enumerate(target_names):
        if i < original_targets.shape[1]:  # 确保索引有效
            mae = mean_absolute_error(original_targets[:, i], original_predictions[:, i])
            rmse = np.sqrt(mean_squared_error(original_targets[:, i], original_predictions[:, i]))
            r2 = r2_score(original_targets[:, i], original_predictions[:, i])
            
            metrics[f'mae_{target}'] = mae
            metrics[f'rmse_{target}'] = rmse
            metrics[f'r2_{target}'] = r2
    
    # 计算平均指标
    metrics['mae_avg'] = np.mean([metrics[f'mae_{t}'] for t in target_names if f'mae_{t}' in metrics])
    metrics['rmse_avg'] = np.mean([metrics[f'rmse_{t}'] for t in target_names if f'rmse_{t}' in metrics])
    metrics['r2_avg'] = np.mean([metrics[f'r2_{t}'] for t in target_names if f'r2_{t}' in metrics])
    
    return original_predictions, original_targets, metrics

def save_predictions_to_csv(smiles_list, predictions, filename='predictions.csv', target_names=None):
    """
    将预测结果保存为CSV文件
    
    参数:
        smiles_list: SMILES字符串列表
        predictions: 预测值数组
        filename: 输出文件名
        target_names: 目标列名称列表
    """
    import pandas as pd
    
    # 设置默认目标名称
    if target_names is None:
        target_names = ['homo', 'lumo', 'energy']
    
    # 创建结果DataFrame
    results = {'smiles': smiles_list}
    
    # 添加每个目标的预测值
    for i, name in enumerate(target_names):
        if i < predictions.shape[1]:
            results[name] = predictions[:, i]
    
    # 保存为CSV
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"预测结果已保存到 {filename}")

def kfold_cross_validation(model_class, data_df, smiles_col, target_cols, input_dim, n_splits=5, 
                          hidden_dim=128, num_layers=3, dropout=0.3, pooling='mean',
                          batch_size=64, learning_rate=0.001, weight_decay=1e-5, num_epochs=100,
                          device='cuda', output_dir='results', normalize=True, pretrained_model_path=None):
    """
    执行k折交叉验证并可视化结果
    
    参数:
        model_class: 模型类
        data_df: 包含数据的DataFrame
        smiles_col: SMILES列名
        target_cols: 目标属性列名列表
        input_dim: 输入特征维度
        n_splits: 交叉验证折数
        hidden_dim: 隐藏层维度
        num_layers: GIN层数
        dropout: Dropout率
        pooling: 池化方法
        batch_size: 批量大小
        learning_rate: 学习率
        weight_decay: 权重衰减
        num_epochs: 训练轮数
        device: 计算设备
        output_dir: 输出目录
        normalize: 是否对目标值进行归一化
        pretrained_model_path: 预训练模型路径，如果提供，将在每一折加载此模型
    
    返回:
        results_df: 包含所有折结果的DataFrame
        metrics_dict: 包含各指标均值和标准差的字典
    """
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch_geometric.loader import DataLoader
    from dataset import MoleculeDataset, normalize_targets
    from train import train_gin_model, load_model
    
    # 创建K折交叉验证器
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 保存每一折的结果
    results = []
    fold_metrics = {
        'mae': [],
        'rmse': [],
        'r2': []
    }
    
    # 获取所有目标列的名称
    all_metrics = {}
    for target in target_cols:
        all_metrics[f'mae_{target}'] = []
        all_metrics[f'rmse_{target}'] = []
        all_metrics[f'r2_{target}'] = []
    
    # 每一折进行训练和评估
    for fold, (train_idx, test_idx) in enumerate(kf.split(data_df)):
        print(f"\n开始第 {fold+1}/{n_splits} 折交叉验证")
        
        # 获取这一折的训练集和测试集
        train_fold_df = data_df.iloc[train_idx].reset_index(drop=True)
        test_fold_df = data_df.iloc[test_idx].reset_index(drop=True)
        
        # 将训练集再分成训练集和验证集
        # 注意：我们在每一折内部再做一次分割，保留一部分作为验证集用于早停
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(train_fold_df, test_size=0.2, random_state=42)
        
        # 对目标值进行归一化处理
        target_scaler = None
        if normalize:
            train_df, val_df, test_fold_df, target_scaler = normalize_targets(
                train_df, val_df, test_fold_df, target_cols
            )
        
        # 创建数据集
        train_dataset = MoleculeDataset(root=f'data/fold_{fold+1}/train', 
                                        df=train_df, 
                                        smiles_col=smiles_col, 
                                        target_cols=target_cols)
        
        val_dataset = MoleculeDataset(root=f'data/fold_{fold+1}/val', 
                                      df=val_df, 
                                      smiles_col=smiles_col, 
                                      target_cols=target_cols,
                                      target_scaler=target_scaler)
        
        test_dataset = MoleculeDataset(root=f'data/fold_{fold+1}/test', 
                                       df=test_fold_df, 
                                       smiles_col=smiles_col, 
                                       target_cols=target_cols,
                                       target_scaler=target_scaler)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # 创建模型
        model = model_class(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=len(target_cols),
            num_layers=num_layers,
            dropout=dropout,
            pooling=pooling
        ).to(device)
        
        # 如果提供了预训练模型路径，则加载预训练模型
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            model = load_model(model, pretrained_model_path, device)
            print(f"已加载预训练模型: {pretrained_model_path}")
        
        # 创建优化器和学习率调度器
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        # 训练模型
        model, history = train_gin_model(
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
        
        # 在测试集上评估模型
        test_preds, test_targets, test_metrics = evaluate_model(model, test_loader, device, target_scaler)
        
        # 保存这一折的指标
        fold_metrics['mae'].append(test_metrics['mae_avg'])
        fold_metrics['rmse'].append(test_metrics['rmse_avg'])
        fold_metrics['r2'].append(test_metrics['r2_avg'])
        
        # 保存每个目标的单独指标
        for target in target_cols:
            all_metrics[f'mae_{target}'].append(test_metrics[f'mae_{target}'])
            all_metrics[f'rmse_{target}'].append(test_metrics[f'rmse_{target}'])
            all_metrics[f'r2_{target}'].append(test_metrics[f'r2_{target}'])
        
        # 打印这一折的结果
        print(f"第 {fold+1} 折结果:")
        print(f"  MAE: {test_metrics['mae_avg']:.4f}")
        print(f"  RMSE: {test_metrics['rmse_avg']:.4f}")
        print(f"  R2: {test_metrics['r2_avg']:.4f}")
        
        # 保存详细结果
        fold_result = {
            'fold': fold + 1,
            'mae_avg': test_metrics['mae_avg'],
            'rmse_avg': test_metrics['rmse_avg'],
            'r2_avg': test_metrics['r2_avg']
        }
        
        # 添加每个目标的单独指标
        for target in target_cols:
            fold_result[f'mae_{target}'] = test_metrics[f'mae_{target}']
            fold_result[f'rmse_{target}'] = test_metrics[f'rmse_{target}']
            fold_result[f'r2_{target}'] = test_metrics[f'r2_{target}']
        
        results.append(fold_result)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(results)
    
    # 计算汇总指标
    metrics_dict = {
        'mae_avg_mean': np.mean(fold_metrics['mae']),
        'mae_avg_std': np.std(fold_metrics['mae']),
        'rmse_avg_mean': np.mean(fold_metrics['rmse']),
        'rmse_avg_std': np.std(fold_metrics['rmse']),
        'r2_avg_mean': np.mean(fold_metrics['r2']),
        'r2_avg_std': np.std(fold_metrics['r2'])
    }
    
    # 为每个目标添加汇总指标
    for target in target_cols:
        metrics_dict[f'mae_{target}_mean'] = np.mean(all_metrics[f'mae_{target}'])
        metrics_dict[f'mae_{target}_std'] = np.std(all_metrics[f'mae_{target}'])
        metrics_dict[f'rmse_{target}_mean'] = np.mean(all_metrics[f'rmse_{target}'])
        metrics_dict[f'rmse_{target}_std'] = np.std(all_metrics[f'rmse_{target}'])
        metrics_dict[f'r2_{target}_mean'] = np.mean(all_metrics[f'r2_{target}'])
        metrics_dict[f'r2_{target}_std'] = np.std(all_metrics[f'r2_{target}'])
    
    # 打印汇总结果
    print(f"\n{n_splits}折交叉验证汇总结果:")
    print(f"整体性能指标:")
    print(f"MAE: {metrics_dict['mae_avg_mean']:.4f} ± {metrics_dict['mae_avg_std']:.4f}")
    print(f"RMSE: {metrics_dict['rmse_avg_mean']:.4f} ± {metrics_dict['rmse_avg_std']:.4f}")
    print(f"R2: {metrics_dict['r2_avg_mean']:.4f} ± {metrics_dict['r2_avg_std']:.4f}")
    
    # 打印每个目标属性的详细指标
    for target in target_cols:
        print(f"\n{target.upper()} 预测性能:")
        print(f"MAE: {metrics_dict[f'mae_{target}_mean']:.4f} ± {metrics_dict[f'mae_{target}_std']:.4f}")
        print(f"RMSE: {metrics_dict[f'rmse_{target}_mean']:.4f} ± {metrics_dict[f'rmse_{target}_std']:.4f}")
        print(f"R2: {metrics_dict[f'r2_{target}_mean']:.4f} ± {metrics_dict[f'r2_{target}_std']:.4f}")
    
    # 绘制误差分布箱型图
    plot_kfold_boxplot(results_df, target_cols, output_dir)
    
    # 保存结果到CSV
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, 'kfold_cross_validation_results.csv'), index=False)
    
    # 写入汇总结果到文本文件
    with open(os.path.join(output_dir, 'kfold_summary.txt'), 'w') as f:
        f.write(f"{n_splits}折交叉验证汇总结果:\n")
        f.write(f"整体性能指标:\n")
        f.write(f"MAE: {metrics_dict['mae_avg_mean']:.4f} ± {metrics_dict['mae_avg_std']:.4f}\n")
        f.write(f"RMSE: {metrics_dict['rmse_avg_mean']:.4f} ± {metrics_dict['rmse_avg_std']:.4f}\n")
        f.write(f"R2: {metrics_dict['r2_avg_mean']:.4f} ± {metrics_dict['r2_avg_std']:.4f}\n\n")
        
        # 写入每个目标属性的详细指标
        for target in target_cols:
            f.write(f"\n{target.upper()} 预测性能:\n")
            f.write(f"MAE: {metrics_dict[f'mae_{target}_mean']:.4f} ± {metrics_dict[f'mae_{target}_std']:.4f}\n")
            f.write(f"RMSE: {metrics_dict[f'rmse_{target}_mean']:.4f} ± {metrics_dict[f'rmse_{target}_std']:.4f}\n")
            f.write(f"R2: {metrics_dict[f'r2_{target}_mean']:.4f} ± {metrics_dict[f'r2_{target}_std']:.4f}\n")
    
    return results_df, metrics_dict

def plot_kfold_boxplot(results_df, target_cols, output_dir):
    """
    绘制k折交叉验证误差分布箱型图
    
    参数:
        results_df: 包含所有折结果的DataFrame
        target_cols: 目标属性列名列表
        output_dir: 输出目录
    """
    # 设置画布大小
    plt.figure(figsize=(15, 10))
    
    # 绘制总体指标的箱型图
    plt.subplot(2, 2, 1)
    sns.boxplot(data=results_df[['mae_avg', 'rmse_avg', 'r2_avg']], palette=[COLORS['blue'], COLORS['teal'], COLORS['cyan']])
    plt.title('Overall Performance Metrics Distribution')
    plt.ylabel('Metric Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 为每个目标属性创建各自的指标箱型图
    # MAE
    plt.subplot(2, 2, 2)
    mae_cols = [f'mae_{target}' for target in target_cols]
    mae_data = results_df[mae_cols]
    mae_data.columns = target_cols  # 简化列名以便显示
    sns.boxplot(data=mae_data, palette=[COLORS['blue'], COLORS['teal'], COLORS['cyan']])
    plt.title('MAE Distribution for Each Target Property')
    plt.ylabel('MAE')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # RMSE
    plt.subplot(2, 2, 3)
    rmse_cols = [f'rmse_{target}' for target in target_cols]
    rmse_data = results_df[rmse_cols]
    rmse_data.columns = target_cols
    sns.boxplot(data=rmse_data, palette=[COLORS['blue'], COLORS['teal'], COLORS['cyan']])
    plt.title('RMSE Distribution for Each Target Property')
    plt.ylabel('RMSE')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # R^2
    plt.subplot(2, 2, 4)
    r2_cols = [f'r2_{target}' for target in target_cols]
    r2_data = results_df[r2_cols]
    r2_data.columns = target_cols
    sns.boxplot(data=r2_data, palette=[COLORS['blue'], COLORS['teal'], COLORS['cyan']])
    plt.title('R2 Distribution for Each Target Property')
    plt.ylabel('R2')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kfold_boxplots.png'), dpi=300)
    plt.close()
    
    # 创建更详细的指标分布图
    plt.figure(figsize=(24, 6))
    
    # 所有折的性能比较
    metrics = ['mae_avg', 'rmse_avg', 'r2_avg']
    colors = [COLORS['blue'], COLORS['teal'], COLORS['cyan']]
    
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        bars = plt.bar(results_df['fold'], results_df[metric], color=colors[i], alpha=0.8)
        plt.axhline(y=results_df[metric].mean(), color=COLORS['dark_blue'], linestyle='-', alpha=0.7, 
                   label=f'Mean: {results_df[metric].mean():.4f}')
        plt.fill_between(range(1, len(results_df)+1), 
                        results_df[metric].mean() - results_df[metric].std(),
                        results_df[metric].mean() + results_df[metric].std(),
                        alpha=0.2, color=COLORS['dark_blue'], label=f'Std: {results_df[metric].std():.4f}')
        plt.title(f'Comparison of {metric} Across Folds')
        plt.xlabel('Fold Number')
        plt.ylabel(metric)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kfold_comparison.png'), dpi=300)
    plt.close()
    
    # 创建各目标属性的性能比较图表
    for target in target_cols:
        plt.figure(figsize=(18, 6))
        metrics = [f'mae_{target}', f'rmse_{target}', f'r2_{target}']
        titles = ['MAE', 'RMSE', 'R2']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            plt.subplot(1, 3, i+1)
            bars = plt.bar(results_df['fold'], results_df[metric], color=colors[i], alpha=0.8)
            plt.axhline(y=results_df[metric].mean(), color=COLORS['dark_blue'], linestyle='-', alpha=0.7,
                       label=f'Mean: {results_df[metric].mean():.4f}')
            plt.fill_between(range(1, len(results_df)+1),
                            results_df[metric].mean() - results_df[metric].std(),
                            results_df[metric].mean() + results_df[metric].std(),
                            alpha=0.2, color=COLORS['dark_blue'], label=f'Std: {results_df[metric].std():.4f}')
            plt.title(f'{target.upper()} - {title} Across Folds')
            plt.xlabel('Fold Number')
            plt.ylabel(title)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'kfold_{target}_comparison.png'), dpi=300)
        plt.close() 