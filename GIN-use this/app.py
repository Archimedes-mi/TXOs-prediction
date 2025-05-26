import streamlit as st
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import seaborn as sns
from model import GIN
from utils import mol2graph, smiles_to_graph
from sklearn.preprocessing import StandardScaler

# 设置页面
st.set_page_config(
    page_title="Molecule Properties Predictor",
    page_icon="🧪",
    layout="wide"
)

# 创建特征归一化处理器
@st.cache_resource
def create_target_scaler():
    # 从predictions.csv中获取训练好的数据来重建scaler
    df = pd.read_csv('results/predictions.csv')
    
    # 原始数据 - 从database.csv获取
    raw_df = pd.read_csv('database.csv')
    
    # 如果列名不一致，需要手动匹配
    target_cols = ['homo', 'lumo', 'energy']
    
    # 创建并拟合scaler
    scaler = StandardScaler()
    scaler.fit(raw_df[target_cols].values)
    
    return scaler

# 加载预训练模型
@st.cache_resource
def load_model():
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GIN(
        input_dim=46,   # 从12修改为46，与预训练模型匹配
        hidden_dim=100, # 从300修改为100，与预训练模型匹配
        output_dim=3,   # 保持不变，输出三个属性
        num_layers=3,   # 从5修改为3，与预训练模型匹配
        dropout=0.5,    
        pooling='mean'  
    ).to(device)
    model.load_state_dict(torch.load('models/gin_optimal_model.pt', map_location=device))
    model.eval()
    return model, device

# 逆变换归一化的预测结果
def inverse_normalize_targets(predictions, scaler):
    """
    将归一化后的预测值转换回原始尺度
    
    参数:
        predictions: 模型的标准化预测值
        scaler: 用于归一化的StandardScaler
    
    返回:
        inverse_predictions: 原始尺度的预测值
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    
    # 逆变换
    inverse_predictions = scaler.inverse_transform(predictions)
    
    return inverse_predictions

# 加载数据集
@st.cache_data
def load_data():
    # 加载predictions.csv文件
    df = pd.read_csv('results/predictions.csv')
    return df

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
def predict_properties(smiles, model, device):
    try:
        # 获取scaler
        scaler = create_target_scaler()
        
        # 直接使用smiles_to_graph函数，保持与训练时的数据处理方式一致
        graph = smiles_to_graph(smiles)
        if graph is None:
            return None, None, None
        
        # 将图数据转移到设备上
        graph = graph.to(device)
        
        # 添加批处理维度，因为这里只处理一个分子
        graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
        
        # 预测
        with torch.no_grad():
            pred = model(graph)
        
        # 应用逆变换恢复到原始尺度
        original_pred = inverse_normalize_targets(pred.cpu().numpy(), scaler)
        
        # 获取预测值
        homo = original_pred[0, 0]  # α-HOMO (eV)
        lumo = original_pred[0, 1]  # β-LUMO (eV)
        energy = original_pred[0, 2]  # delta-E (kcal/mol)
        
        return homo, lumo, energy
    except Exception as e:
        st.error(f"预测过程中发生错误: {str(e)}")
        return None, None, None

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
    model, device = load_model()
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
                homo, lumo, energy = predict_properties(smiles_input, model, device)
                
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
                        st.image(mol_img, caption=f"Molecule {i+1}")
                
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