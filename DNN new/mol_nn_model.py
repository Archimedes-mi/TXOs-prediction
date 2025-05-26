import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED
from tqdm import tqdm
import optuna
import joblib
import warnings
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

# 设置随机种子以确保可重现性
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 检查是否可以使用CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 计算RDKit分子描述符
def calculate_rdkit_descriptors(mol):
    """
    计算RDKit所有可用的分子描述符
    
    参数:
        mol: RDKit分子对象
        
    返回:
        numpy.ndarray: 描述符向量
    """
    if mol is None:
        return None
    
    # 获取所有描述符的名称和函数
    descriptor_list = Descriptors._descList
    descriptor_names = [desc[0] for desc in descriptor_list]
    descriptor_funcs = [desc[1] for desc in descriptor_list]
    
    try:
        # 计算所有描述符
        desc_values = []
        for func in descriptor_funcs:
            try:
                value = func(mol)
                desc_values.append(value)
            except:
                # 如果计算特定描述符出错，填入0
                desc_values.append(0)
        
        # 额外添加QED药物性质评分
        try:
            qed_value = QED.qed(mol)
            desc_values.append(qed_value)
            descriptor_names.append('QED')
        except:
            desc_values.append(0)
            descriptor_names.append('QED')
        
        return np.array(desc_values, dtype=np.float32)
    except:
        # 如果描述符计算完全失败，返回None
        return None

# 存储描述符名称全局变量，仅在第一次调用时初始化
DESCRIPTOR_NAMES = None

# 从SMILES生成分子特征
def smiles_to_features(smiles, morgan_size=2048, use_rdkit_desc=True):
    """
    使用Morgan指纹和RDKit描述符将SMILES转换为特征向量
    
    参数:
        smiles (str): 分子的SMILES表示
        morgan_size (int): Morgan指纹的大小
        use_rdkit_desc (bool): 是否使用RDKit描述符
        
    返回:
        numpy.ndarray: 分子特征向量
    """
    global DESCRIPTOR_NAMES
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # 使用Morgan指纹
            morgan_fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=morgan_size))
            
            # 如果不使用RDKit描述符，仅返回Morgan指纹
            if not use_rdkit_desc:
                return morgan_fp
            
            # 计算RDKit描述符
            rdkit_desc = calculate_rdkit_descriptors(mol)
            
            # 如果描述符计算失败，仅返回Morgan指纹
            if rdkit_desc is None:
                print(f"RDKit描述符计算失败，仅使用Morgan指纹: {smiles}")
                return morgan_fp
            
            # 初始化描述符名称（仅第一次）
            if DESCRIPTOR_NAMES is None:
                descriptor_list = Descriptors._descList
                DESCRIPTOR_NAMES = [desc[0] for desc in descriptor_list] + ['QED']
                print(f"使用的RDKit描述符数量: {len(DESCRIPTOR_NAMES)}")
            
            # 合并Morgan指纹和RDKit描述符
            return np.hstack([morgan_fp, rdkit_desc])
        else:
            # 如果SMILES无法解析，返回零向量
            print(f"无法解析SMILES: {smiles}")
            if use_rdkit_desc:
                # 确保rdkit_desc_size初始化正确
                if DESCRIPTOR_NAMES is None:
                    # 使用简单分子初始化描述符名称
                    temp_mol = Chem.MolFromSmiles('C')
                    rdkit_desc = calculate_rdkit_descriptors(temp_mol)
                    descriptor_list = Descriptors._descList
                    DESCRIPTOR_NAMES = [desc[0] for desc in descriptor_list] + ['QED']
                    rdkit_desc_size = len(rdkit_desc)
                else:
                    rdkit_desc_size = len(DESCRIPTOR_NAMES)
                return np.zeros(morgan_size + rdkit_desc_size)
            else:
                return np.zeros(morgan_size)
    except Exception as e:
        print(f"处理SMILES时出错: {smiles}, 错误: {e}")
        if use_rdkit_desc:
            # 零向量大小为Morgan指纹大小加上RDKit描述符数量
            try:
                if DESCRIPTOR_NAMES is None:
                    # 使用简单分子初始化描述符名称
                    temp_mol = Chem.MolFromSmiles('C')
                    rdkit_desc = calculate_rdkit_descriptors(temp_mol)
                    descriptor_list = Descriptors._descList
                    DESCRIPTOR_NAMES = [desc[0] for desc in descriptor_list] + ['QED']
                    rdkit_desc_size = len(rdkit_desc)
                else:
                    rdkit_desc_size = len(DESCRIPTOR_NAMES)
                return np.zeros(morgan_size + rdkit_desc_size)
            except:
                # 如果无法确定描述符大小，仅返回Morgan指纹大小的零向量
                print(f"无法确定描述符大小，仅返回Morgan指纹大小的零向量")
                return np.zeros(morgan_size)
        else:
            return np.zeros(morgan_size)

# 计算一组SMILES的所有描述符并返回DataFrame
def calculate_dataset_descriptors(smiles_list, include_morgan=True, morgan_size=2048):
    """
    为一组SMILES计算所有描述符并返回DataFrame
    
    参数:
        smiles_list: SMILES字符串列表
        include_morgan: 是否包含Morgan指纹
        morgan_size: Morgan指纹大小
        
    返回:
        pd.DataFrame: 包含所有分子描述符的DataFrame
    """
    # 转换SMILES为RDKit分子对象
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    mols = [m for m in mols if m is not None]  # 过滤无效分子
    
    # 获取所有描述符的名称
    descriptor_list = Descriptors._descList
    descriptor_names = [desc[0] for desc in descriptor_list] + ['QED']
    
    # 计算所有分子的所有描述符
    all_descriptors = []
    
    for mol in tqdm(mols, desc="计算分子描述符"):
        # 计算RDKit描述符
        rdkit_desc = calculate_rdkit_descriptors(mol)
        
        if rdkit_desc is not None:
            # 如果包含Morgan指纹，计算并添加
            if include_morgan:
                morgan_fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=morgan_size))
                # 创建Morgan指纹的列名
                morgan_names = [f"Morgan_{i}" for i in range(morgan_size)]
                # 合并描述符
                all_features = np.hstack([rdkit_desc, morgan_fp])
                all_descriptors.append(all_features)
            else:
                all_descriptors.append(rdkit_desc)
        else:
            # 如果计算描述符失败，跳过该分子
            continue
    
    # 创建列名
    if include_morgan:
        column_names = descriptor_names + [f"Morgan_{i}" for i in range(morgan_size)]
    else:
        column_names = descriptor_names
    
    # 创建描述符数据框
    descriptor_df = pd.DataFrame(all_descriptors, columns=column_names)
    
    return descriptor_df

# 定义三目标深度神经网络模型
class MolecularPropertyNN(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout_rate, activation_fn):
        """
        分子性质预测的深度神经网络
        
        参数:
            input_size (int): 输入特征大小
            hidden_layers (list): 每个隐藏层的神经元数量
            dropout_rate (float): Dropout率
            activation_fn (str): 激活函数类型
        """
        super(MolecularPropertyNN, self).__init__()
        
        layers = []
        # 输入层
        prev_size = input_size
        
        # 获取激活函数
        activation_map = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'selu': nn.SELU()
        }
        
        # 构建隐藏层
        for h_size in hidden_layers:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(activation_map[activation_fn])
            layers.append(nn.Dropout(dropout_rate))
            prev_size = h_size
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 三个输出分支，分别预测homo、lumo和energy
        self.homo_head = nn.Linear(prev_size, 1)
        self.lumo_head = nn.Linear(prev_size, 1)
        self.energy_head = nn.Linear(prev_size, 1)
        
    def forward(self, x):
        """前向传播"""
        features = self.feature_extractor(x)
        homo = self.homo_head(features)
        lumo = self.lumo_head(features)
        energy = self.energy_head(features)
        
        return homo, lumo, energy

# 数据加载和预处理
def load_and_preprocess_data(csv_path, test_size=0.2, val_size=0.16, use_rdkit_desc=True, morgan_size=2048):
    """
    加载和预处理数据
    
    参数:
        csv_path (str): CSV文件路径
        test_size (float): 测试集比例
        val_size (float): 验证集比例
        use_rdkit_desc (bool): 是否使用RDKit描述符
        morgan_size (int): Morgan指纹的大小
        
    返回:
        tuple: (训练加载器，验证加载器，测试加载器，特征定标器，目标定标器字典)
    """
    # 加载数据
    df = pd.read_csv(csv_path)
    print(f"加载了 {len(df)} 条数据")
    
    # 提取特征和目标变量
    X = np.array([smiles_to_features(s, morgan_size=morgan_size, use_rdkit_desc=use_rdkit_desc) 
                  for s in df['smiles']])
    y = df[['homo', 'lumo', 'energy']].values
    
    # 划分数据集
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED
    )
    
    # 进一步划分训练集和验证集
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=SEED
    )
    
    # 特征标准化
    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(X_train)
    X_val = feature_scaler.transform(X_val)
    X_test = feature_scaler.transform(X_test)
    
    # 对每个目标变量单独进行标准化
    # 创建三个独立的标准化器
    homo_scaler = StandardScaler()
    lumo_scaler = StandardScaler()
    energy_scaler = StandardScaler()
    
    # 分别对每个目标变量进行标准化
    y_train_homo = homo_scaler.fit_transform(y_train[:, 0].reshape(-1, 1))
    y_train_lumo = lumo_scaler.fit_transform(y_train[:, 1].reshape(-1, 1))
    y_train_energy = energy_scaler.fit_transform(y_train[:, 2].reshape(-1, 1))
    
    y_val_homo = homo_scaler.transform(y_val[:, 0].reshape(-1, 1))
    y_val_lumo = lumo_scaler.transform(y_val[:, 1].reshape(-1, 1))
    y_val_energy = energy_scaler.transform(y_val[:, 2].reshape(-1, 1))
    
    y_test_homo = homo_scaler.transform(y_test[:, 0].reshape(-1, 1))
    y_test_lumo = lumo_scaler.transform(y_test[:, 1].reshape(-1, 1))
    y_test_energy = energy_scaler.transform(y_test[:, 2].reshape(-1, 1))
    
    # 将标准化后的结果合并
    y_train = np.hstack([y_train_homo, y_train_lumo, y_train_energy])
    y_val = np.hstack([y_val_homo, y_val_lumo, y_val_energy])
    y_test = np.hstack([y_test_homo, y_test_lumo, y_test_energy])
    
    # 创建一个字典来保存所有的定标器
    target_scalers = {
        'homo': homo_scaler,
        'lumo': lumo_scaler,
        'energy': energy_scaler
    }
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # 创建DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    return train_loader, val_loader, test_loader, feature_scaler, target_scalers

# 模型训练函数
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=100, early_stopping=10):
    """
    训练模型
    
    参数:
        model: PyTorch模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        criterion: 损失函数
        epochs (int): 训练轮数
        early_stopping (int): 早停的耐心值
        
    返回:
        tuple: (训练损失历史，验证损失历史)
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练模式
        model.train()
        train_loss = 0.0
        
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            homo_pred, lumo_pred, energy_pred = model(features)
            
            # 计算损失
            homo_loss = criterion(homo_pred, targets[:, 0:1])
            lumo_loss = criterion(lumo_pred, targets[:, 1:2])
            energy_loss = criterion(energy_pred, targets[:, 2:3])
            
            # 总损失为三个任务的损失和
            loss = homo_loss + lumo_loss + energy_loss
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证模式
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                
                homo_pred, lumo_pred, energy_pred = model(features)
                
                homo_loss = criterion(homo_pred, targets[:, 0:1])
                lumo_loss = criterion(lumo_pred, targets[:, 1:2])
                energy_loss = criterion(energy_pred, targets[:, 2:3])
                
                loss = homo_loss + lumo_loss + energy_loss
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 输出训练进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    return train_losses, val_losses

# 评估模型性能
def evaluate_model(model, test_loader, target_scalers):
    """
    评估模型性能
    
    参数:
        model: PyTorch模型
        test_loader: 测试数据加载器
        target_scalers: 目标变量定标器字典
        
    返回:
        dict: 包含各项性能指标的字典
    """
    model.eval()
    
    all_homo_preds = []
    all_lumo_preds = []
    all_energy_preds = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            
            homo_pred, lumo_pred, energy_pred = model(features)
            
            all_homo_preds.append(homo_pred.cpu().numpy())
            all_lumo_preds.append(lumo_pred.cpu().numpy())
            all_energy_preds.append(energy_pred.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # 合并批次结果
    all_homo_preds = np.vstack(all_homo_preds)
    all_lumo_preds = np.vstack(all_lumo_preds)
    all_energy_preds = np.vstack(all_energy_preds)
    all_targets = np.vstack(all_targets)
    
    # 分别反标准化每个目标变量的预测结果
    all_homo_preds_original = target_scalers['homo'].inverse_transform(all_homo_preds)
    all_lumo_preds_original = target_scalers['lumo'].inverse_transform(all_lumo_preds)
    all_energy_preds_original = target_scalers['energy'].inverse_transform(all_energy_preds)
    
    # 分别反标准化每个目标变量的真实值
    all_homo_targets_original = target_scalers['homo'].inverse_transform(all_targets[:, 0:1])
    all_lumo_targets_original = target_scalers['lumo'].inverse_transform(all_targets[:, 1:2])
    all_energy_targets_original = target_scalers['energy'].inverse_transform(all_targets[:, 2:3])
    
    # 合并预测值
    all_preds = np.hstack([all_homo_preds_original, all_lumo_preds_original, all_energy_preds_original])
    all_targets = np.hstack([all_homo_targets_original, all_lumo_targets_original, all_energy_targets_original])
    
    # 计算性能指标
    metrics = {}
    
    # 整体性能
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    metrics['overall'] = {'mae': mae, 'rmse': rmse, 'r2': r2}
    
    # 各属性性能
    properties = ['homo', 'lumo', 'energy']
    for i, prop in enumerate(properties):
        mae = mean_absolute_error(all_targets[:, i], all_preds[:, i])
        rmse = np.sqrt(mean_squared_error(all_targets[:, i], all_preds[:, i]))
        r2 = r2_score(all_targets[:, i], all_preds[:, i])
        metrics[prop] = {'mae': mae, 'rmse': rmse, 'r2': r2}
    
    return metrics, all_targets, all_preds

# 可视化模型性能
def visualize_performance(targets, predictions, metrics, save_path="results"):
    """
    可视化模型性能
    
    参数:
        targets: 真实值
        predictions: 预测值
        metrics: 性能指标
        save_path: 保存结果的路径
    """
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 设置蓝绿配色
    palette = sns.color_palette("mako", 3)
    
    # 绘制散点图
    properties = ['homo', 'lumo', 'energy']
    titles = ['HOMO (eV)', 'LUMO (eV)', 'Energy (kcal/mol)']
    
    plt.figure(figsize=(18, 6))
    for i, (prop, title) in enumerate(zip(properties, titles)):
        plt.subplot(1, 3, i+1)
        
        # 散点图
        plt.scatter(targets[:, i], predictions[:, i], alpha=0.7, color=palette[i])
        
        # 对角线
        min_val = min(targets[:, i].min(), predictions[:, i].min())
        max_val = max(targets[:, i].max(), predictions[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--')
        
        # 添加性能指标文本
        mae = metrics[prop]['mae']
        rmse = metrics[prop]['rmse']
        r2 = metrics[prop]['r2']
        
        plt.text(0.05, 0.95, f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}',
                transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.xlabel(f'True {title}')
        plt.ylabel(f'Predicted {title}')
        plt.title(f'{title} Prediction Performance')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, 'performance_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制性能指标条形图
    plt.figure(figsize=(15, 5))
    
    # MAE
    plt.subplot(1, 3, 1)
    mae_values = [metrics[prop]['mae'] for prop in properties]
    sns.barplot(x=properties, y=mae_values, palette=palette)
    plt.title('Mean Absolute Error (MAE)')
    plt.ylabel('MAE')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # RMSE
    plt.subplot(1, 3, 2)
    rmse_values = [metrics[prop]['rmse'] for prop in properties]
    sns.barplot(x=properties, y=rmse_values, palette=palette)
    plt.title('Root Mean Square Error (RMSE)')
    plt.ylabel('RMSE')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # R2
    plt.subplot(1, 3, 3)
    r2_values = [metrics[prop]['r2'] for prop in properties]
    sns.barplot(x=properties, y=r2_values, palette=palette)
    plt.title('Coefficient of Determination (R²)')
    plt.ylabel('R²')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 使用Optuna进行贝叶斯优化
def objective(trial):
    """
    Optuna优化目标函数
    
    参数:
        trial: Optuna试验对象
        
    返回:
        float: 验证损失
    """
    # 超参数搜索空间
    hidden_layers = []
    n_layers = trial.suggest_int('n_layers', 2, 5)
    for i in range(n_layers):
        hidden_layers.append(trial.suggest_int(f'hidden_units_{i}', 32, 256))
    
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    activation_fn = trial.suggest_categorical('activation_fn', 
                                              ['relu', 'leaky_relu', 'sigmoid', 'tanh', 'elu', 'selu'])
    
    # 创建模型
    model = MolecularPropertyNN(
        input_size=X_train_tensor.shape[1],
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        activation_fn=activation_fn
    ).to(device)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # 训练模型
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=50,
        early_stopping=5
    )
    
    return val_losses[-1]

# 使用最佳模型进行预测
def predict_with_model(model, smiles_list, feature_scaler, target_scalers, use_rdkit_desc=True, morgan_size=2048):
    """
    使用模型预测SMILES列表的属性
    
    参数:
        model: PyTorch模型
        smiles_list: SMILES字符串列表
        feature_scaler: 特征定标器
        target_scalers: 目标变量定标器字典
        use_rdkit_desc (bool): 是否使用RDKit描述符
        morgan_size (int): Morgan指纹的大小
        
    返回:
        tuple: (homo, lumo, energy) 预测值
    """
    model.eval()
    
    # 提取特征
    features = np.array([smiles_to_features(s, morgan_size=morgan_size, use_rdkit_desc=use_rdkit_desc) 
                         for s in smiles_list])
    
    # 标准化特征
    features = feature_scaler.transform(features)
    
    # 转换为张量
    features_tensor = torch.FloatTensor(features).to(device)
    
    # 预测
    with torch.no_grad():
        homo_pred, lumo_pred, energy_pred = model(features_tensor)
    
    # 分别反标准化每个属性
    homo_original = target_scalers['homo'].inverse_transform(homo_pred.cpu().numpy())
    lumo_original = target_scalers['lumo'].inverse_transform(lumo_pred.cpu().numpy())
    energy_original = target_scalers['energy'].inverse_transform(energy_pred.cpu().numpy())
    
    # 提取结果
    homo = homo_original.flatten()
    lumo = lumo_original.flatten()
    energy = energy_original.flatten()
    
    return homo, lumo, energy

# 特征选择函数
def select_features(X, y, variance_threshold=True, rf_importance=True, n_top_features=None):
    """
    进行特征选择
    
    参数:
        X: 特征矩阵
        y: 目标变量
        variance_threshold: 是否使用VarianceThreshold去除方差为0的特征
        rf_importance: 是否使用随机森林特征重要性
        n_top_features: 选择前n个重要特征，如果为None，保留所有非零重要性特征
        
    返回:
        tuple: (选择后的特征矩阵, 特征选择器列表, 特征重要性)
    """
    selectors = []
    feature_importance = None
    
    # 初始特征数量
    n_features_original = X.shape[1]
    print(f"原始特征数量: {n_features_original}")
    
    # 1. 方差阈值特征选择 - 去除方差为0的特征
    if variance_threshold:
        var_selector = VarianceThreshold(threshold=0)
        X = var_selector.fit_transform(X)
        selectors.append(var_selector)
        print(f"方差阈值后特征数量: {X.shape[1]}")
    
    # 2. 随机森林特征重要性选择
    if rf_importance:
        # 创建一个简单的随机森林回归器用于特征选择
        rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
        
        # 对于多目标问题，我们取每个目标变量的平均特征重要性
        importances = np.zeros(X.shape[1])
        
        # 每个目标变量分别计算特征重要性
        for target_idx in range(y.shape[1]):
            rf.fit(X, y[:, target_idx])
            importances += rf.feature_importances_
        
        # 平均特征重要性
        importances /= y.shape[1]
        feature_importance = importances
        
        # 选择前n个重要特征
        if n_top_features is not None and n_top_features < X.shape[1]:
            top_indices = np.argsort(importances)[::-1][:n_top_features]
            X = X[:, top_indices]
            # 记录选择的特征索引
            class FeatureSelector:
                def __init__(self, indices):
                    self.indices = indices
                def transform(self, X):
                    return X[:, self.indices]
                def get_support(self, indices=False):
                    if indices:
                        return self.indices
                    else:
                        mask = np.zeros(len(feature_importance), dtype=bool)
                        mask[self.indices] = True
                        return mask
            
            rf_selector = FeatureSelector(top_indices)
            selectors.append(rf_selector)
            print(f"随机森林特征选择后特征数量: {X.shape[1]}")
    
    return X, selectors, feature_importance

# 可视化特征重要性
def visualize_feature_importance(importance, file_path="results/feature_importance.png"):
    """
    可视化特征重要性
    
    参数:
        importance: 特征重要性数组
        file_path: 保存文件路径
    """
    global DESCRIPTOR_NAMES
    
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 只保留前10个特征以便于可视化
    n_features = min(10, len(importance))
    
    # 获取前n个重要特征的索引
    indices = np.argsort(importance)[::-1][:n_features]
    sorted_importance = importance[indices]
    
    # 生成特征名称
    feature_names = []
    
    # 确保DESCRIPTOR_NAMES已被初始化
    if DESCRIPTOR_NAMES is None:
        temp_mol = Chem.MolFromSmiles('C')
        rdkit_desc = calculate_rdkit_descriptors(temp_mol)
        descriptor_list = Descriptors._descList
        DESCRIPTOR_NAMES = [desc[0] for desc in descriptor_list] + ['QED']
    
    rdkit_desc_size = len(DESCRIPTOR_NAMES)
    morgan_size = len(importance) - rdkit_desc_size if len(importance) > rdkit_desc_size else 0
        
    # 为每个特征创建名称
    for idx in indices:
        if idx < rdkit_desc_size:
            # 如果是RDKit描述符
            feature_names.append(DESCRIPTOR_NAMES[idx])
        else:
            # 如果是Morgan指纹
            morgan_idx = idx - rdkit_desc_size
            feature_names.append(f"Morgan_{morgan_idx}")
    
    # 反转顺序，使最重要的特征在最上面
    sorted_importance = sorted_importance[::-1]
    feature_names = feature_names[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(n_features), sorted_importance, align='center')
    plt.yticks(range(n_features), feature_names)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.title('Top 10 Feature Importance')
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()
    
    print(f"特征重要性可视化已保存到 {file_path}")

# 交叉验证评估模型性能
def cross_validate_model(X, y, best_params, n_splits=5, use_feature_selection=True, use_rdkit_desc=True, morgan_size=2048):
    """
    使用交叉验证评估模型性能
    
    参数:
        X: 特征矩阵 (SMILES列表)
        y: 目标变量
        best_params: 优化后的最佳模型参数
        n_splits: 交叉验证折数
        use_feature_selection: 是否使用特征选择
        use_rdkit_desc: 是否使用RDKit描述符
        morgan_size: Morgan指纹大小
        
    返回:
        dict: 交叉验证结果
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    # 存储每一折的评估指标
    fold_metrics = {
        'homo_mae': [], 'homo_rmse': [], 'homo_r2': [],
        'lumo_mae': [], 'lumo_rmse': [], 'lumo_r2': [],
        'energy_mae': [], 'energy_rmse': [], 'energy_r2': [],
    }
    
    # 存储每一折的预测值和真实值
    all_predictions = []
    all_targets = []
    
    # 存储每一折的结果以用于绘图
    results = []
    
    # 首先提取所有SMILES的特征
    X_features = np.array([smiles_to_features(s, morgan_size=morgan_size, use_rdkit_desc=use_rdkit_desc) 
                         for s in X])
    
    print(f"交叉验证开始，共 {n_splits} 折...")
    
    # 从best_params中提取模型参数
    hidden_layers = []
    for i in range(best_params['n_layers']):
        hidden_layers.append(best_params[f'hidden_units_{i}'])
    
    dropout_rate = best_params['dropout_rate']
    learning_rate = best_params['learning_rate']
    activation_fn = best_params['activation_fn']
    
    # 交叉验证
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_features)):
        print(f"处理第 {fold+1}/{n_splits} 折...")
        
        # 分割训练集和验证集
        X_train, X_valid = X_features[train_idx], X_features[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        
        # 特征选择 (仅在训练集上进行)
        if use_feature_selection:
            X_train, selectors, feature_importance = select_features(X_train, y_train)
            
            # 将特征选择应用于验证集
            for selector in selectors:
                X_valid = selector.transform(X_valid)
            
            # 可视化特征重要性 (仅在第一折)
            if fold == 0 and feature_importance is not None:
                visualize_feature_importance(feature_importance)
        
        # 特征标准化
        feature_scaler = StandardScaler()
        X_train = feature_scaler.fit_transform(X_train)
        X_valid = feature_scaler.transform(X_valid)
        
        # 对每个目标变量单独进行标准化
        homo_scaler = StandardScaler()
        lumo_scaler = StandardScaler()
        energy_scaler = StandardScaler()
        
        y_train_homo = homo_scaler.fit_transform(y_train[:, 0].reshape(-1, 1))
        y_train_lumo = lumo_scaler.fit_transform(y_train[:, 1].reshape(-1, 1))
        y_train_energy = energy_scaler.fit_transform(y_train[:, 2].reshape(-1, 1))
        
        y_valid_homo = homo_scaler.transform(y_valid[:, 0].reshape(-1, 1))
        y_valid_lumo = lumo_scaler.transform(y_valid[:, 1].reshape(-1, 1))
        y_valid_energy = energy_scaler.transform(y_valid[:, 2].reshape(-1, 1))
        
        # 将标准化后的结果合并
        y_train_scaled = np.hstack([y_train_homo, y_train_lumo, y_train_energy])
        y_valid_scaled = np.hstack([y_valid_homo, y_valid_lumo, y_valid_energy])
        
        # 创建目标定标器字典
        target_scalers = {
            'homo': homo_scaler,
            'lumo': lumo_scaler,
            'energy': energy_scaler
        }
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train_scaled)
        X_valid_tensor = torch.FloatTensor(X_valid)
        y_valid_tensor = torch.FloatTensor(y_valid_scaled)
        
        # 创建DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
        
        # 创建模型，使用最佳超参数
        model = MolecularPropertyNN(
            input_size=X_train.shape[1],
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
            activation_fn=activation_fn
        ).to(device)
        
        # 训练模型
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        train_losses, val_losses = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=valid_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=100,  # 增加轮数以充分训练
            early_stopping=10
        )
        
        # 评估模型
        metrics, targets, predictions = evaluate_model(model, valid_loader, target_scalers)
        
        # 保存指标结果
        for prop in ['homo', 'lumo', 'energy']:
            fold_metrics[f'{prop}_mae'].append(metrics[prop]['mae'])
            fold_metrics[f'{prop}_rmse'].append(metrics[prop]['rmse'])
            fold_metrics[f'{prop}_r2'].append(metrics[prop]['r2'])
        
        # 保存预测结果
        all_predictions.append(predictions)
        all_targets.append(targets)
        
        # 添加当前折的结果到结果列表
        fold_result = {
            'fold': fold + 1,
            'mae_avg': np.mean([metrics[prop]['mae'] for prop in ['homo', 'lumo', 'energy']]),
            'rmse_avg': np.mean([metrics[prop]['rmse'] for prop in ['homo', 'lumo', 'energy']]),
            'r2_avg': np.mean([metrics[prop]['r2'] for prop in ['homo', 'lumo', 'energy']])
        }
        
        # 添加每个目标的单独指标
        for prop in ['homo', 'lumo', 'energy']:
            fold_result[f'mae_{prop}'] = metrics[prop]['mae']
            fold_result[f'rmse_{prop}'] = metrics[prop]['rmse']
            fold_result[f'r2_{prop}'] = metrics[prop]['r2']
        
        results.append(fold_result)
        
        print(f"第 {fold+1} 折 - HOMO MAE: {metrics['homo']['mae']:.4f}, LUMO MAE: {metrics['lumo']['mae']:.4f}, Energy MAE: {metrics['energy']['mae']:.4f}")
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(results)
    
    # 计算平均指标
    avg_metrics = {}
    for key in fold_metrics:
        avg_metrics[key] = np.mean(fold_metrics[key])
        print(f"平均 {key}: {avg_metrics[key]:.4f} ± {np.std(fold_metrics[key]):.4f}")
    
    # 可视化交叉验证结果
    visualize_cv_results(fold_metrics, save_path="results/cv_results.png")
    
    # 使用utils.py中的绘图函数绘制更详细的折间比较
    plot_kfold_boxplot(results_df, ['homo', 'lumo', 'energy'], 'results')
    
    # 保存交叉验证结果到CSV
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/kfold_cross_validation_results.csv', index=False)
    
    # 写入汇总结果到文本文件
    with open(os.path.join('results', 'kfold_summary.txt'), 'w') as f:
        f.write(f"{n_splits}折交叉验证汇总结果:\n")
        for prop in ['homo', 'lumo', 'energy']:
            f.write(f"\n{prop.upper()} 预测性能:\n")
            f.write(f"MAE: {avg_metrics[f'{prop}_mae']:.4f} ± {np.std(fold_metrics[f'{prop}_mae']):.4f}\n")
            f.write(f"RMSE: {avg_metrics[f'{prop}_rmse']:.4f} ± {np.std(fold_metrics[f'{prop}_rmse']):.4f}\n")
            f.write(f"R2: {avg_metrics[f'{prop}_r2']:.4f} ± {np.std(fold_metrics[f'{prop}_r2']):.4f}\n")
    
    return {
        'fold_metrics': fold_metrics,
        'avg_metrics': avg_metrics,
        'all_predictions': all_predictions,
        'all_targets': all_targets,
        'results_df': results_df
    }

# 可视化交叉验证结果
def visualize_cv_results(fold_metrics, save_path="results/cv_results.png"):
    """
    可视化交叉验证的结果
    
    参数:
        fold_metrics: 每一折的评估指标
        save_path: 保存文件路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 设置蓝绿配色
    colors = {
        'blue': '#1f77b4',
        'teal': '#39ac73',
        'dark_blue': '#035096',
        'cyan': '#40E0D0'
    }
    
    # 准备数据
    metrics_df = pd.DataFrame()
    for prop in ['homo', 'lumo', 'energy']:
        for metric in ['mae', 'rmse', 'r2']:
            key = f'{prop}_{metric}'
            for fold, value in enumerate(fold_metrics[key]):
                # 使用concat替代append
                metrics_df = pd.concat([metrics_df, pd.DataFrame({
                    'Property': [prop.upper()],
                    'Metric': [metric.upper()],
                    'Fold': [fold + 1],
                    'Value': [value]
                })], ignore_index=True)
    
    # 绘制箱型图
    plt.figure(figsize=(15, 10))
    
    # 分别为每个指标类型绘制子图
    for i, metric in enumerate(['MAE', 'RMSE', 'R2']):
        plt.subplot(1, 3, i+1)
        
        # 筛选当前指标的数据
        metric_data = metrics_df[metrics_df['Metric'] == metric]
        
        # 绘制箱型图
        sns.boxplot(x='Property', y='Value', data=metric_data, 
                   palette=[colors['blue'], colors['teal'], colors['cyan']])
        
        plt.title(f'Cross-Validation {metric}')
        plt.xlabel('Property')
        plt.ylabel(metric)
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # 绘制每一折的性能比较
    fold_count = len(fold_metrics['homo_mae'])
    
    # 为每种指标创建折线图
    for metric in ['mae', 'rmse', 'r2']:
        plt.figure(figsize=(12, 6))
        
        for i, prop in enumerate(['homo', 'lumo', 'energy']):
            plt.plot(range(1, fold_count + 1), fold_metrics[f'{prop}_{metric}'], 
                     marker='o', linestyle='-', linewidth=2,
                     label=f'{prop.upper()}',
                     color=colors['blue'] if prop == 'homo' else 
                           colors['teal'] if prop == 'lumo' else 
                           colors['cyan'])
        
        plt.xlabel('Fold')
        plt.ylabel(metric.upper())
        plt.title(f'{metric.upper()} for Each Fold')
        plt.xticks(range(1, fold_count + 1))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 保存图表
        metric_path = os.path.join(os.path.dirname(save_path), f'{metric}_fold_comparison.png')
        plt.savefig(metric_path, dpi=300)
        plt.close()
    
    print(f"交叉验证结果可视化已保存到 {os.path.dirname(save_path)} 目录")

def plot_kfold_boxplot(results_df, target_cols, output_dir):
    """
    绘制k折交叉验证误差分布箱型图
    
    参数:
        results_df: 包含所有折结果的DataFrame
        target_cols: 目标属性列名列表
        output_dir: 输出目录
    """
    # 设置蓝绿配色
    colors = {
        'blue': '#1f77b4',
        'teal': '#39ac73',
        'dark_blue': '#035096',
        'cyan': '#40E0D0',
        'purple': '#9370DB',
        'orange': '#FFA500'
    }
    
    # 设置画布大小
    plt.figure(figsize=(15, 10))
    
    # 绘制总体指标的箱型图
    plt.subplot(2, 2, 1)
    sns.boxplot(data=results_df[['mae_avg', 'rmse_avg', 'r2_avg']], 
                palette=[colors['blue'], colors['teal'], colors['cyan']])
    plt.title('Overall Performance Metrics Distribution')
    plt.ylabel('Metric Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 为每个目标属性创建各自的指标箱型图
    # MAE
    plt.subplot(2, 2, 2)
    mae_cols = [f'mae_{target}' for target in target_cols]
    mae_data = results_df[mae_cols]
    mae_data.columns = target_cols  # 简化列名以便显示
    sns.boxplot(data=mae_data, palette=[colors['blue'], colors['teal'], colors['cyan']])
    plt.title('MAE Distribution for Each Target Property')
    plt.ylabel('MAE')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # RMSE
    plt.subplot(2, 2, 3)
    rmse_cols = [f'rmse_{target}' for target in target_cols]
    rmse_data = results_df[rmse_cols]
    rmse_data.columns = target_cols
    sns.boxplot(data=rmse_data, palette=[colors['blue'], colors['teal'], colors['cyan']])
    plt.title('RMSE Distribution for Each Target Property')
    plt.ylabel('RMSE')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # R^2
    plt.subplot(2, 2, 4)
    r2_cols = [f'r2_{target}' for target in target_cols]
    r2_data = results_df[r2_cols]
    r2_data.columns = target_cols
    sns.boxplot(data=r2_data, palette=[colors['blue'], colors['teal'], colors['cyan']])
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
    color_list = [colors['blue'], colors['teal'], colors['cyan']]
    
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        bars = plt.bar(results_df['fold'], results_df[metric], color=color_list[i], alpha=0.8)
        plt.axhline(y=results_df[metric].mean(), color=colors['dark_blue'], linestyle='-', alpha=0.7, 
                   label=f'Mean: {results_df[metric].mean():.4f}')
        plt.fill_between(range(1, len(results_df)+1), 
                        results_df[metric].mean() - results_df[metric].std(),
                        results_df[metric].mean() + results_df[metric].std(),
                        alpha=0.2, color=colors['dark_blue'], label=f'Std: {results_df[metric].std():.4f}')
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
    
    # ===== 为每个目标属性创建单独的性能比较图表 =====
    # 为指标定义颜色映射
    metric_colors = {'mae': 'blue', 'rmse': 'teal', 'r2': 'cyan'}
    
    for target in target_cols:
        plt.figure(figsize=(18, 6))
        metrics = [f'mae_{target}', f'rmse_{target}', f'r2_{target}']
        titles = ['MAE', 'RMSE', 'R²']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            plt.subplot(1, 3, i+1)
            metric_type = metric.split('_')[0]  # 提取指标类型(mae, rmse, r2)
            bars = plt.bar(results_df['fold'], results_df[metric], 
                          color=colors[metric_colors[metric_type]], alpha=0.8)
            
            # 添加平均线和标准差范围
            mean_val = results_df[metric].mean()
            std_val = results_df[metric].std()
            plt.axhline(y=mean_val, color=colors['dark_blue'], linestyle='-', alpha=0.7,
                       label=f'Mean: {mean_val:.4f}')
            plt.fill_between(range(1, len(results_df)+1),
                            mean_val - std_val,
                            mean_val + std_val,
                            alpha=0.2, color=colors['dark_blue'], 
                            label=f'Std: {std_val:.4f}')
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom')
            
            plt.title(f'{target.upper()} - {title} Across Folds')
            plt.xlabel('Fold Number')
            plt.ylabel(title)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'kfold_{target}_comparison.png'), dpi=300)
        plt.close()
    
    
    # 创建每个折上三个属性的对比条形图
    metrics = ['mae', 'rmse', 'r2']
    titles = ['MAE', 'RMSE', 'R²']
    metric_colors = {'mae': 'blue', 'rmse': 'teal', 'r2': 'cyan'}
    
    for metric, title in zip(metrics, titles):
        plt.figure(figsize=(15, 8))
        
        # 设置分组条形图的参数
        n_folds = len(results_df)
        n_props = len(target_cols)
        width = 0.25  # 条形宽度
        
        # 设置x轴位置
        ind = np.arange(n_folds)  # 折的位置
        
        # 绘制每个属性的条形
        for i, prop in enumerate(target_cols):
            offset = (i - n_props/2 + 0.5) * width
            metric_key = f'{metric}_{prop}'
            
            bars = plt.bar(ind + offset, results_df[metric_key], width,
                          label=prop.upper(),
                          color=colors[metric_colors[metric]])
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', 
                        fontsize=9, rotation=0)
        
        # 设置x轴标签和标题
        plt.xlabel('Fold')
        plt.ylabel(title)
        plt.title(f'{title} Comparison by Property for Each Fold')
        plt.xticks(ind, [f'Fold {i}' for i in results_df['fold']])
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_property_comparison.png'), dpi=300)
        plt.close()
    
    print(f"交叉验证可视化分析结果已保存到 {output_dir} 目录")

# 主函数
def main():
    # 配置
    use_rdkit_desc = True
    morgan_size = 2048
    do_cross_validation = True
    
    # 加载数据
    df = pd.read_csv('database.csv')
    print(f"加载了 {len(df)} 条数据")
    
    # 加载和预处理数据进行最终模型训练
    global train_loader, val_loader, test_loader, X_train_tensor, feature_scaler, target_scalers
    train_loader, val_loader, test_loader, feature_scaler, target_scalers = load_and_preprocess_data(
        'database.csv', use_rdkit_desc=use_rdkit_desc, morgan_size=morgan_size
    )
    
    # 提取特征以进行特征选择
    X_all = []
    y_all = []
    for X_batch, y_batch in train_loader:
        X_all.append(X_batch.cpu().numpy())
        y_all.append(y_batch.cpu().numpy())
    
    X_all = np.vstack(X_all)
    y_all = np.vstack(y_all)
    
    # 特征选择
    print("进行特征选择...")
    X_selected, selectors, feature_importance = select_features(X_all, y_all)
    
    # 可视化特征重要性
    if feature_importance is not None:
        visualize_feature_importance(feature_importance)
    
    # 获取一个批次以获取输入大小
    for X_batch, _ in train_loader:
        X_train_tensor = X_batch
        break
    
    # 使用Optuna进行贝叶斯优化
    print("开始贝叶斯优化...")
    # 创建带有随机初始探索的采样器
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=10,  # 10次初始随机探索
        seed=SEED
    )
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=60)  # 总共60轮（10次随机+50次贝叶斯）
    
    # 显示最佳超参数
    best_params = study.best_params
    print("最佳超参数:", best_params)
    
    if do_cross_validation:
        # 执行交叉验证
        print("执行5折交叉验证...")
        X = df['smiles'].values
        y = df[['homo', 'lumo', 'energy']].values
        
        cv_results = cross_validate_model(
            X, y, best_params, n_splits=5,
            use_feature_selection=True,
            use_rdkit_desc=use_rdkit_desc,
            morgan_size=morgan_size
        )
        
        # 保存交叉验证结果
        joblib.dump(cv_results, 'results/cv_results.pkl')
        print("交叉验证完成，结果已保存")
    
    # 使用最佳超参数创建最终模型
    hidden_layers = []
    for i in range(best_params['n_layers']):
        hidden_layers.append(best_params[f'hidden_units_{i}'])
    
    best_model = MolecularPropertyNN(
        input_size=X_train_tensor.shape[1],
        hidden_layers=hidden_layers,
        dropout_rate=best_params['dropout_rate'],
        activation_fn=best_params['activation_fn']
    ).to(device)
    
    # 使用最佳超参数训练最终模型
    print("训练最终模型...")
    optimizer = optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])
    criterion = nn.MSELoss()
    
    train_losses, val_losses = train_model(
        model=best_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=200,
        early_stopping=20
    )
    
    # 评估最终模型
    print("评估最终模型...")
    metrics, targets, predictions = evaluate_model(best_model, test_loader, target_scalers)
    
    # 打印性能指标
    print("\n模型性能:")
    for prop in ['homo', 'lumo', 'energy']:
        print(f"\n{prop.upper()} 预测性能:")
        print(f"MAE: {metrics[prop]['mae']:.4f}")
        print(f"RMSE: {metrics[prop]['rmse']:.4f}")
        print(f"R2: {metrics[prop]['r2']:.4f}")
    
    # 可视化性能
    visualize_performance(targets, predictions, metrics)
    
    # 保存最佳模型和定标器
    torch.save(best_model.state_dict(), 'results/best_model.pt')
    joblib.dump(feature_scaler, 'results/feature_scaler.pkl')
    # 保存所有目标变量的定标器
    joblib.dump(target_scalers, 'results/target_scalers.pkl')
    # 保存特征选择器
    joblib.dump(selectors, 'results/feature_selectors.pkl')
    
    # 预测TXs.csv中的化合物
    print("\n预测TXs.csv中的化合物...")
    txs_df = pd.read_csv('TXs.csv')
    
    # 检查是否已有预测结果列
    for col in ['homo', 'lumo', 'energy']:
        if col not in txs_df.columns:
            txs_df[col] = np.nan
    
    # 预测
    homo_pred, lumo_pred, energy_pred = predict_with_model(
        best_model, txs_df['smiles'].tolist(), feature_scaler, target_scalers, 
        use_rdkit_desc=use_rdkit_desc, morgan_size=morgan_size
    )
    
    # 更新预测结果
    txs_df['homo'] = homo_pred
    txs_df['lumo'] = lumo_pred
    txs_df['energy'] = energy_pred
    
    # 保存预测结果
    txs_df.to_csv('TXs_predicted.csv', index=False)
    print("预测完成并保存到 TXs_predicted.csv")

if __name__ == "__main__":
    main() 
