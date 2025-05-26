import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch_geometric.explain.algorithm import GNNExplainer
from rdkit import Chem
try:
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
except ImportError:
    print("警告: 无法导入rdMolDraw2D，将使用替代可视化方法")
from utils import visualize_molecule_with_explanation
from model import AtomFeatureExtractor
import networkx as nx

class CustomGNNExplainer:
    """
    自定义GNN解释器，用于解释模型预测
    """
    def __init__(self, model, device, target_scaler=None):
        """
        初始化解释器
        
        参数:
            model: 已训练的GIN模型
            device: 计算设备
            target_scaler: 目标值归一化器
        """
        self.model = model.to(device)
        self.device = device
        self.target_scaler = target_scaler
        self.model.eval()
    
    def explain_molecule(self, data, target_idx=0, num_atoms=None):
        """
        解释分子的预测结果
        
        参数:
            data: 分子数据 (PyG Data对象)
            target_idx: 目标属性索引 (0=homo, 1=lumo, 2=energy)
            num_atoms: 要考虑的原子数量(如果为None则使用所有原子)
        
        返回:
            node_feat_mask: 节点特征的重要性掩码
            edge_mask: 边的重要性掩码
        """
        data = data.to(self.device)
        
        # 创建特征提取器
        feature_extractor = AtomFeatureExtractor(self.model, target_idx)
        feature_extractor = feature_extractor.to(self.device)
        
        # 手动实现GNNExplainer的核心功能，避免使用可能存在版本兼容问题的接口
        # 这是简化版实现，仅用于解决当前问题
        feature_extractor.eval()
        
        # 创建可训练的掩码
        num_nodes = data.x.size(0)
        num_features = data.x.size(1)
        num_edges = data.edge_index.size(1)
        
        node_feat_mask = torch.ones(num_features, requires_grad=True, device=self.device)
        edge_mask = torch.ones(num_edges, requires_grad=True, device=self.device)
        
        # 设置优化器
        optimizer = torch.optim.Adam([node_feat_mask, edge_mask], lr=0.01)
        
        # 训练循环
        for epoch in range(200):  # 固定训练200轮
            optimizer.zero_grad()
            
            # 应用掩码
            masked_x = data.x * node_feat_mask
            
            # 前向传播
            out = feature_extractor(masked_x, data.edge_index)
            loss = -torch.mean(out)  # 最大化输出
            
            # 正则化
            edge_mask_penalty = 0.001 * torch.sum(torch.abs(edge_mask))
            node_feat_mask_penalty = 0.001 * torch.sum(torch.abs(node_feat_mask))
            
            # 总损失
            loss = loss + edge_mask_penalty + node_feat_mask_penalty
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 标准化掩码
            with torch.no_grad():
                node_feat_mask.clamp_(0, 1)
                edge_mask.clamp_(0, 1)
        
        # 返回最终掩码
        node_feat_mask = node_feat_mask.detach().cpu().numpy()
        # 扩展节点特征掩码到每个节点
        node_feat_mask_expanded = np.tile(node_feat_mask, (num_nodes, 1))
        edge_mask = edge_mask.detach().cpu().numpy()
        
        return node_feat_mask_expanded, edge_mask
    
    def visualize_explanation(self, smiles, node_feat_mask, edge_mask, target_idx=0, save_dir='results'):
        """
        可视化解释结果
        
        参数:
            smiles: SMILES分子字符串
            node_feat_mask: 节点特征重要性
            edge_mask: 边重要性
            target_idx: 目标特性索引(0=homo, 1=lumo, 2=energy)
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取目标名称
        target_names = ['HOMO', 'LUMO', 'Energy']
        target_name = target_names[target_idx]
        
        # 保存可视化
        output_file = os.path.join(save_dir, f'molecule_explanation_{target_name}.png')
        visualize_molecule_with_explanation(smiles, node_feat_mask, output_file)
        
        # 解析分子
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"无法解析SMILES: {smiles}")
            return
        
        try:
            # 计算每个原子的重要性总和 - 确保是标量值
            atom_importances = np.sum(node_feat_mask, axis=1)
            
            # 创建一个图形列表
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            
            # 1. 绘制原子重要性条形图
            atoms = [atom.GetSymbol() + str(atom.GetIdx()) for atom in mol.GetAtoms()]
            
            # 确保数据长度一致
            min_len = min(len(atoms), len(atom_importances))
            atoms = atoms[:min_len]
            atom_importances = atom_importances[:min_len]
            
            importance_df = pd.DataFrame({'atom': atoms, 'importance': atom_importances})
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # 选择重要性最高的N个原子
            top_n = min(15, len(atoms))
            importance_df = importance_df.head(top_n)
            
            # 配色
            colors = plt.cm.viridis(importance_df['importance'] / importance_df['importance'].max())
            
            # 绘制条形图
            bars = axes[0].bar(importance_df['atom'], importance_df['importance'], color=colors)
            axes[0].set_xlabel('Atom')
            axes[0].set_ylabel('Importance')
            axes[0].set_title(f'Top {top_n} Important Atoms for {target_name} Prediction')
            axes[0].tick_params(axis='x', rotation=45)
            
            # 2. 绘制特征维度重要性热图
            # 计算每个特征维度的平均重要性
            feature_importances = np.mean(node_feat_mask, axis=0)
            
            # 分组特征维度
            feature_groups = {
                'Atom Type': list(range(5)),
                'Formal Charge': [5],
                'Explicit Hs': [6],
                'Aromaticity': [7],
                'Degree': [8],
                'Total Hs': [9],
                'Morgan FP': list(range(10, node_feat_mask.shape[1]))
            }
            
            group_importances = {}
            for group, indices in feature_groups.items():
                # 确保索引在数组范围内
                valid_indices = [i for i in indices if i < len(feature_importances)]
                if valid_indices:
                    group_importances[group] = float(np.mean(feature_importances[valid_indices]))
                else:
                    group_importances[group] = 0.0
            
            # 创建热图数据
            group_names = list(group_importances.keys())
            group_values = list(group_importances.values())
            
            # 归一化 - 确保是标量值
            max_value = max(group_values) if group_values else 1.0
            group_values = [float(val) / max_value for val in group_values]
            
            # 绘制热图
            im = axes[1].imshow([group_values], cmap='viridis', aspect='auto')
            axes[1].set_yticks([])
            axes[1].set_xticks(np.arange(len(group_names)))
            axes[1].set_xticklabels(group_names, rotation=45, ha='right')
            axes[1].set_title(f'Feature Group Importance for {target_name} Prediction')
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=axes[1], orientation='vertical', pad=0.05)
            cbar.set_label('Normalized Importance')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'feature_importance_{target_name}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"已完成 {target_name} 的特征重要性可视化")
        
        except Exception as e:
            print(f"创建可视化图表时出错: {str(e)}")
            print("继续执行，但可视化结果可能不完整")

def explain_model_predictions(model, data_loader, smiles_list, device, save_dir='results', target_scaler=None):
    """
    解释模型预测并可视化结果
    
    参数:
        model: 训练好的GIN模型
        data_loader: 数据加载器
        smiles_list: SMILES分子列表
        device: 计算设备
        save_dir: 保存目录
        target_scaler: 目标值归一化器
    """
    # 确保存在足够的分子用于解释
    if len(smiles_list) == 0:
        print("没有分子可以解释")
        return
    
    # 创建解释器
    explainer = CustomGNNExplainer(model, device, target_scaler)
    
    # 为每个目标属性解释模型(HOMO, LUMO, 能量)
    target_names = ['HOMO', 'LUMO', 'Energy']
    
    # 从数据加载器中获取单个分子
    for i, data in enumerate(data_loader):
        if i >= 3:  # 限制为3个分子以节省时间
            break
        
        try:
            # 获取当前分子的SMILES
            smiles = smiles_list[i]
            print(f"解释分子 {i+1}: {smiles}")
            
            # 为每个目标属性解释
            for target_idx, target_name in enumerate(target_names):
                # 获取解释
                node_feat_mask, edge_mask = explainer.explain_molecule(data, target_idx=target_idx)
                
                # 可视化解释
                explainer.visualize_explanation(
                    smiles, 
                    node_feat_mask, 
                    edge_mask, 
                    target_idx=target_idx, 
                    save_dir=save_dir
                )
                
                print(f"已完成 {target_name} 的解释可视化")
        
        except Exception as e:
            print(f"解释分子 {i+1} 时出错: {str(e)}")
            continue 