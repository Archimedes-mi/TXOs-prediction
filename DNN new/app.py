import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from rdkit.Chem import AllChem, Descriptors, QED
import os

# 设置页面
st.set_page_config(
    page_title="Molecule Properties Predictor",
    page_icon="🧪",
    layout="wide"
)

# 从mol_nn_model-use this copy.py导入必要的函数和类
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

# 存储描述符名称全局变量，仅在第一次调用时初始化
DESCRIPTOR_NAMES = None

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

# 创建特征归一化处理器和目标归一化处理器
@st.cache_resource
def create_scalers():
    """创建特征和目标变量的缩放器"""
    # 检查是否存在保存的缩放器
    if os.path.exists('results/feature_scaler.pkl') and os.path.exists('results/target_scalers.pkl'):
        try:
            feature_scaler = joblib.load('results/feature_scaler.pkl')
            target_scalers = joblib.load('results/target_scalers.pkl')
            return feature_scaler, target_scalers
        except Exception as e:
            st.warning(f"无法加载保存的缩放器，将创建新的缩放器: {e}")
    
    # 如果没有保存的缩放器，从数据创建新的缩放器
    df = pd.read_csv('TXs_predicted.csv')
    
    # 创建目标变量缩放器
    target_cols = ['homo', 'lumo', 'energy']
    homo_scaler = StandardScaler()
    lumo_scaler = StandardScaler()
    energy_scaler = StandardScaler()
    
    # 拟合缩放器
    homo_scaler.fit(df['homo'].values.reshape(-1, 1))
    lumo_scaler.fit(df['lumo'].values.reshape(-1, 1))
    energy_scaler.fit(df['energy'].values.reshape(-1, 1))
    
    # 创建特征缩放器
    # 提取一个样本特征来确定大小
    sample_smiles = df['smiles'].iloc[0]
    sample_features = smiles_to_features(sample_smiles)
    feature_scaler = StandardScaler()
    feature_scaler.fit([sample_features])  # 简单拟合，实际应用中应该用更多样本
    
    # 创建目标缩放器字典
    target_scalers = {
        'homo': homo_scaler,
        'lumo': lumo_scaler,
        'energy': energy_scaler
    }
    
    # 保存缩放器以便将来使用
    try:
        os.makedirs('results', exist_ok=True)
        joblib.dump(feature_scaler, 'results/feature_scaler.pkl')
        joblib.dump(target_scalers, 'results/target_scalers.pkl')
    except Exception as e:
        st.warning(f"无法保存缩放器: {e}")
    
    return feature_scaler, target_scalers

# 加载预训练模型
@st.cache_resource
def load_model():
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 获取特征和目标缩放器
    feature_scaler, target_scalers = create_scalers()
    
    # 加载模型参数
    try:
        # 检查模型文件是否存在
        model_path = 'results/best_model.pt'
        if not os.path.exists(model_path):
            st.error(f"模型文件不存在: {model_path}")
            # 创建一个简单的模型作为替代
            best_params = {
                'n_layers': 3,
                'hidden_units_0': 128,
                'hidden_units_1': 64,
                'hidden_units_2': 32,
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'activation_fn': 'relu'
            }
            
            # 获取输入特征大小
            sample_smiles = 'C'
            sample_features = smiles_to_features(sample_smiles)
            input_size = len(sample_features)
            
            # 构建隐藏层配置
            hidden_layers = []
            for i in range(best_params['n_layers']):
                hidden_layers.append(best_params[f'hidden_units_{i}'])
            
            # 创建模型
            model = MolecularPropertyNN(
                input_size=input_size,
                hidden_layers=hidden_layers,
                dropout_rate=best_params['dropout_rate'],
                activation_fn=best_params['activation_fn']
            ).to(device)
            
            # 提示用户模型未经训练
            st.warning("使用未训练的模型，预测结果可能不准确。请确保模型文件存在。")
        else:
            # 根据错误信息匹配模型结构
            best_params = {
                'n_layers': 4,  # 修改为4层
                'hidden_units_0': 242,  # 修改为242
                'hidden_units_1': 171,  # 修改为171
                'hidden_units_2': 46,   # 修改为46
                'hidden_units_3': 187,  # 添加一个187神经元的层
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'activation_fn': 'relu'
            }
            
            # 获取输入特征大小
            sample_smiles = 'C'
            sample_features = smiles_to_features(sample_smiles)
            input_size = len(sample_features)
            
            # 构建隐藏层配置
            hidden_layers = []
            for i in range(best_params['n_layers']):
                hidden_layers.append(best_params[f'hidden_units_{i}'])
            
            # 创建模型
            model = MolecularPropertyNN(
                input_size=input_size,
                hidden_layers=hidden_layers,
                dropout_rate=best_params['dropout_rate'],
                activation_fn=best_params['activation_fn']
            ).to(device)
            
            # 加载模型权重
            model.load_state_dict(torch.load(model_path, map_location=device))
        
        model.eval()
        return model, feature_scaler, target_scalers, device
    except Exception as e:
        st.error(f"加载模型时出错: {str(e)}")
        return None, feature_scaler, target_scalers, device

# 从SMILES生成分子图
def generate_mol_img(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Draw.MolToImage(mol, size=(300, 300))
    except:
        return None

# 预测分子属性
def predict_properties(smiles, model, feature_scaler, target_scalers, device):
    try:
        # 提取特征
        features = np.array([smiles_to_features(smiles)])
        
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
        homo = homo_original.flatten()[0]
        lumo = lumo_original.flatten()[0]
        energy = energy_original.flatten()[0]
        
        return homo, lumo, energy
    except Exception as e:
        st.error(f"预测过程中发生错误: {str(e)}")
        return None, None, None

# 加载数据集
@st.cache_data
def load_data():
    # 尝试加载TXs_predicted.csv文件
    try:
        df = pd.read_csv('TXs_predicted.csv')
        return df
    except Exception as e:
        st.error(f"加载TXs_predicted.csv失败: {str(e)}")
        # 尝试加载results/predictions.csv作为备选
        try:
            df = pd.read_csv('results/predictions.csv')
            return df
        except Exception as e2:
            st.error(f"加载results/predictions.csv也失败: {str(e2)}")
            # 创建一个空的DataFrame作为备选
            return pd.DataFrame(columns=['no', 'smiles', 'homo', 'lumo', 'energy'])

# 主函数
def main():
    st.title("Molecule Properties Predictor")
    
    # 侧边栏导航
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a mode", 
                                   ["SMILES to Properties", 
                                    "Properties to SMILES", 
                                    "Database Exploration"])
    
    # 加载模型和数据
    model, feature_scaler, target_scalers, device = load_model()
    df = load_data()
    
    # 模式1: SMILES到属性预测
    if app_mode == "SMILES to Properties":
        st.header("SMILES to Properties")
        st.write("Input a SMILES code to predict its properties")
        
        # 用户输入
        smiles_input = st.text_input("Enter SMILES code:")
        
        if smiles_input:
            mol_img = generate_mol_img(smiles_input)
            
            if mol_img is None:
                st.error("Invalid SMILES code. Please check your input.")
            else:
                # 显示分子结构
                st.image(mol_img, caption="Molecular Structure")
                
                # 预测属性
                homo, lumo, energy = predict_properties(smiles_input, model, feature_scaler, target_scalers, device)
                
                if homo is not None:
                    # 创建三列显示结果
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(label="α-HOMO Energy (eV)", value=f"{homo:.4f}")
                    with col2:
                        st.metric(label="β-LUMO Energy (eV)", value=f"{lumo:.4f}")
                    with col3:
                        st.metric(label="delta-E (kcal/mol)", value=f"{energy:.4f}")
                else:
                    st.error("Failed to predict properties for this molecule.")
    
    # 模式2: 通过能量范围筛选分子
    elif app_mode == "Properties to SMILES":
        st.header("Properties to SMILES")
        st.write("Filter molecules by delta-E range")
        
        if df.empty:
            st.error("No data available. Please make sure TXs_predicted.csv is loaded correctly.")
            return
            
        # 获取能量的最小和最大值
        min_energy = float(df['energy'].min())
        max_energy = float(df['energy'].max())
        
        # 用户输入能量范围
        col1, col2 = st.columns(2)
        with col1:
            min_e = st.number_input("Minimum delta-E (kcal/mol)", 
                                  min_value=min_energy, 
                                  max_value=max_energy,
                                  value=min_energy)
        with col2:
            max_e = st.number_input("Maximum delta-E (kcal/mol)", 
                                  min_value=min_energy, 
                                  max_value=max_energy,
                                  value=max_energy)
        
        # 筛选分子
        filtered_df = df[(df['energy'] >= min_e) & (df['energy'] <= max_e)]
        
        if filtered_df.empty:
            st.warning("No molecules found in this energy range. Please modify the range.")
        else:
            st.success(f"Found {len(filtered_df)} molecules in the specified range.")
            
            # 显示结果
            for i, row in filtered_df.iterrows():
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    mol_img = generate_mol_img(row['smiles'])
                    if mol_img is not None:
                        st.image(mol_img, caption=f"Molecule {row['no']}")
                
                with col2:
                    st.markdown(f"**SMILES**: `{row['smiles']}`")
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric(label="α-HOMO Energy (eV)", value=f"{row['homo']:.4f}")
                    with m2:
                        st.metric(label="β-LUMO Energy (eV)", value=f"{row['lumo']:.4f}")
                    with m3:
                        st.metric(label="delta-E (kcal/mol)", value=f"{row['energy']:.4f}")
                
                st.divider()
    
    # 模式3: 数据库探索
    elif app_mode == "Database Exploration":
        st.header("Database Exploration")
        st.write("Explore the database statistics and distributions")
        
        if df.empty:
            st.error("No data available. Please make sure TXs_predicted.csv is loaded correctly.")
            return
            
        # 显示数据集基本信息
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Molecules", len(df))
        with col2:
            st.metric("Min delta-E (kcal/mol)", f"{df['energy'].min():.4f}")
        with col3:
            st.metric("Max delta-E (kcal/mol)", f"{df['energy'].max():.4f}")
        
        # 统计信息标签页
        tab1, tab2, tab3 = st.tabs(["Energy Distribution", "HOMO/LUMO Distribution", "Correlations"])
        
        with tab1:
            # 绘制能量分布直方图
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df['energy'], bins=30, kde=True, ax=ax)
            ax.set_title("Distribution of delta-E Values")
            ax.set_xlabel("delta-E (kcal/mol)")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            
        with tab2:
            # HOMO/LUMO分布
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            sns.histplot(df['homo'], bins=30, kde=True, ax=ax1)
            ax1.set_title("Distribution of α-HOMO Values")
            ax1.set_xlabel("α-HOMO (eV)")
            ax1.set_ylabel("Count")
            
            sns.histplot(df['lumo'], bins=30, kde=True, color='orange', ax=ax2)
            ax2.set_title("Distribution of β-LUMO Values")
            ax2.set_xlabel("β-LUMO (eV)")
            ax2.set_ylabel("Count")
            
            plt.tight_layout()
            st.pyplot(fig)
            
        with tab3:
            # 相关性散点图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            sns.scatterplot(x='homo', y='energy', data=df, ax=ax1, alpha=0.6)
            ax1.set_title("α-HOMO vs delta-E")
            ax1.set_xlabel("α-HOMO (eV)")
            ax1.set_ylabel("delta-E (kcal/mol)")
            
            sns.scatterplot(x='lumo', y='energy', data=df, ax=ax2, alpha=0.6, color='orange')
            ax2.set_title("β-LUMO vs delta-E")
            ax2.set_xlabel("β-LUMO (eV)")
            ax2.set_ylabel("delta-E (kcal/mol)")
            
            plt.tight_layout()
            st.pyplot(fig)
            
        # 高级筛选
        st.subheader("Advanced Filtering")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            min_homo = st.number_input("Min α-HOMO (eV)", 
                                     value=float(df['homo'].min()),
                                     step=0.1)
            max_homo = st.number_input("Max α-HOMO (eV)", 
                                     value=float(df['homo'].max()),
                                     step=0.1)
        
        with col2:
            min_lumo = st.number_input("Min β-LUMO (eV)", 
                                     value=float(df['lumo'].min()),
                                     step=0.1)
            max_lumo = st.number_input("Max β-LUMO (eV)", 
                                     value=float(df['lumo'].max()),
                                     step=0.1)
        
        with col3:
            min_energy = st.number_input("Min delta-E (kcal/mol)", 
                                       value=float(df['energy'].min()),
                                       step=1.0)
            max_energy = st.number_input("Max delta-E (kcal/mol)", 
                                       value=float(df['energy'].max()),
                                       step=1.0)
        
        # 应用筛选
        if st.button("Apply Filters"):
            filtered_df = df[
                (df['homo'] >= min_homo) & (df['homo'] <= max_homo) &
                (df['lumo'] >= min_lumo) & (df['lumo'] <= max_lumo) &
                (df['energy'] >= min_energy) & (df['energy'] <= max_energy)
            ]
            
            if filtered_df.empty:
                st.warning("No molecules match these criteria.")
            else:
                st.success(f"Found {len(filtered_df)} molecules matching your criteria.")
                st.dataframe(filtered_df)
                
                # 显示前5个分子结构
                if len(filtered_df) > 0:
                    st.subheader("Sample Molecules")
                    num_to_display = min(5, len(filtered_df))
                    cols = st.columns(num_to_display)
                    
                    for i, (col, (_, row)) in enumerate(zip(cols, filtered_df.head(num_to_display).iterrows())):
                        mol_img = generate_mol_img(row['smiles'])
                        if mol_img is not None:
                            col.image(mol_img, caption=f"Energy: {row['energy']:.2f}")

if __name__ == "__main__":
    main() 