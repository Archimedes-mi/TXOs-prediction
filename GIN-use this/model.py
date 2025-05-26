import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool

class MLP(nn.Module):
    """
    多层感知机模块
    """
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # 检查批量大小，如果只有1个样本，则跳过BatchNorm或切换到评估模式
        if x.size(0) > 1:
            x = self.batch_norm(x)
        else:
            # 对单个样本使用实例归一化或者使用评估模式的BatchNorm
            batch_norm_mode = self.batch_norm.training
            self.batch_norm.eval()
            with torch.no_grad():
                x = self.batch_norm(x)
            self.batch_norm.training = batch_norm_mode
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class GIN(nn.Module):
    """
    图同构网络(GIN)模型
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5, pooling='mean'):
        """
        初始化GIN模型
        
        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（预测目标数量）
            num_layers: GIN层数
            dropout: Dropout概率
            pooling: 图池化方法 ('mean', 'sum', 'max')
        """
        super(GIN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        
        # GIN卷积层
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # 输入层
        mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv_layers.append(GINConv(mlp))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 隐藏层
        for i in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.conv_layers.append(GINConv(mlp))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 预测层
        self.pred_layer = MLP(hidden_dim * num_layers, hidden_dim, output_dim, dropout)
    
    def forward(self, data):
        """
        前向传播
        
        参数:
            data: PyG Data对象，包含x(节点特征)和edge_index(边索引)
        
        返回:
            output: shape=(batch_size, output_dim)的预测结果
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 如果batch未定义，假设只有一个图
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # 存储每层的输出
        h_list = []
        
        for i in range(self.num_layers):
            # 应用GIN卷积
            x = self.conv_layers[i](x, edge_index)
            
            # 处理BatchNorm - 如果批量大小为1，使用评估模式
            if x.size(0) > 1:
                x = self.batch_norms[i](x)
            else:
                # 保存当前训练状态
                batch_norm_mode = self.batch_norms[i].training
                self.batch_norms[i].eval()
                with torch.no_grad():
                    x = self.batch_norms[i](x)
                self.batch_norms[i].training = batch_norm_mode
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # 保存当前层的节点表示
            h_list.append(x)
        
        # 池化操作（将节点特征聚合为图特征）
        x_list = []
        for h in h_list:
            if self.pooling == 'sum':
                x_list.append(global_add_pool(h, batch))
            elif self.pooling == 'max':
                x_list.append(global_max_pool(h, batch))
            else:  # 默认使用mean pooling
                x_list.append(global_mean_pool(h, batch))
        
        # 连接所有层的图表示
        x = torch.cat(x_list, dim=1)
        
        # 预测层
        output = self.pred_layer(x)
        
        return output

class AtomFeatureExtractor(nn.Module):
    """
    原子特征提取器，用于GNNExplainer
    """
    def __init__(self, gin_model, target_idx=0):
        super(AtomFeatureExtractor, self).__init__()
        self.gin_model = gin_model
        self.target_idx = target_idx  # 要解释的目标索引(0=homo, 1=lumo, 2=energy)
    
    def forward(self, x, edge_index, batch=None, **kwargs):
        """
        为兼容GNNExplainer，使用明确的参数名
        """
        # 如果batch未定义，假设只有一个图
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # 存储每层的节点表示
        h_list = []
        
        for i in range(self.gin_model.num_layers):
            x = self.gin_model.conv_layers[i](x, edge_index)
            
            # 处理BatchNorm - 如果批量大小为1，使用评估模式
            if x.size(0) > 1:
                x = self.gin_model.batch_norms[i](x)
            else:
                # 保存当前训练状态
                batch_norm_mode = self.gin_model.batch_norms[i].training
                self.gin_model.batch_norms[i].eval()
                with torch.no_grad():
                    x = self.gin_model.batch_norms[i](x)
                self.gin_model.batch_norms[i].training = batch_norm_mode
                
            x = F.relu(x)
            h_list.append(x)
        
        # 为节点级别的解释返回最后一层的节点表示
        # 对于目标索引，通过返回每个节点特征总和的绝对值作为重要性指标
        output = h_list[-1]
        # 确保权重始终为正值
        return torch.abs(output) 