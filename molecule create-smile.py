from rdkit import Chem 
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import random
from glob import glob
from rdkit.Chem.Draw import MolDrawing
import itertools
import os
from rdkit.Chem import rdmolops
from rdkit.Chem import Descriptors
import pandas as pd
from rdkit.Chem import rdmolfiles
import warnings
import numpy as np
from rdkit import RDLogger
import matplotlib.pyplot as plt  # 添加matplotlib导入
warnings.filterwarnings('ignore',message='not removing hydrogen atom with dummy atom neighbors')

# 定义光敏基团
cores_1 = [
    #'O=C1C2=C(C=CC=C2)SC3=CC=CC=C31'    # 分子1-噻吨酮
    'O=C1C2=C(C=CC=C2[*:1])SC3=CC=CC([*:2])=C31' ,
    'O=C1C2=C(C=CC([*:1])=C2)SC3=CC=C([*:2])C=C31' ,
    'O=C1C2=C(C=C([*:1])C=C2)SC3=CC([*:2])=CC=C31' , 
    'O=C1C2=C(C([*:1])=CC=C2)SC3=C([*:2])C=CC=C31',
    #'O=C1C2=C(C=C(C=CC=C3)C3=C2)SC4=CC5=C(C=CC=C5)C=C41'  #分子2-噻吨酮苯环共轭衍生
    'O=C1C2=C(C=C(C=CC=C3)C3=C2([*:1]))SC4=CC5=C(C=CC=C5)C([*:2])=C41',
    'O=C1C2=C(C=C(C=CC=C3([*:1]))C3=C2)SC4=CC5=C(C([*:2])=CC=C5)C=C41',
    'O=C1C2=C(SC3=CC4=C(C=C31)C=C([*:1])C=C4)C=C5C=CC([*:2])=CC5=C2',
    'O=C1C2=C(C=C(C=C([*:1])C=C3)C3=C2)SC4=CC5=C(C=CC([*:2])=C5)C=C41',
    'O=C1C2=C(C=C(C([*:1])=CC=C3)C3=C2)SC4=CC5=C(C=CC=C5[*:2])C=C41',
    'O=C1C2=C(C([*:1])=C(C=CC=C3)C3=C2)SC4=C([*:2])C5=C(C=CC=C5)C=C41'
    ]

R1 = ['[H][*:1]',   # 氢原子
      '[*:1]C',     # 甲基
      '[*:1]CC',    # 乙基
      '[*:1]CCC',   # 丙基
      '[*:1]CCCC',  # 丁基
      '[*:1]C(C)C',  # 异丙基
      '[*:1]C(C)(C)C',   # 叔丁基
      'O[*:1]',     # 羟基
      '[*:1]OC',    # 甲氧基
      '[*:1]OCC',   # 乙氧基
      '[*:1]N([H])[H]', # 氨基
      '[*:1]N(C)[H]', # 甲胺基
      '[*:1]N(C)C', # 二甲基胺基
      'F[*:1]',     # 氟原子
      'Cl[*:1]',    # 氯原子
      'Br[*:1]',    # 溴原子
      'I[*:1]',     # 碘原子
      '[*:1]C#N',   # 氰基
      '[*:1]C(C)=O', # 羰基
      '[*:1]C(F)(F)F',   # 三氟甲基
      '[*:1][N+]([O-])=O', # 硝基
      '[*:1]C1=CC=CC=C1',   # 苯基
      '[*:1]NC1=CC=CC=C1' # 氨基苯基
      ]

R2 = ['[H][*:2]',   # 氢原子
      '[*:2]C',     # 甲基
      '[*:2]CC',    # 乙基
      '[*:2]CCC',   # 丙基
      '[*:2]CCCC',  # 丁基
      '[*:2]C(C)C',  # 异丙基
      '[*:2]C(C)(C)C',   # 叔丁基
      'O[*:2]',     # 羟基
      '[*:2]OC',    # 甲氧基
      '[*:2]OCC',   # 乙氧基
      '[*:2]N([H])[H]', # 氨基
      '[*:2]N(C)[H]', # 甲胺基
      '[*:2]N(C)C', # 二甲基胺基
      'F[*:2]',     # 氟原子
      'Cl[*:2]',    # 氯原子
      'Br[*:2]',    # 溴原子
      'I[*:2]',     # 碘原子
      '[*:2]C#N',   # 氰基
      '[*:2]C(C)=O', # 羰基
      '[*:2]C(F)(F)F',   # 三氟甲基
      '[*:2][N+]([O-])=O', # 硝基
      '[*:2]C1=CC=CC=C1',   # 苯基
      '[*:2]NC1=CC=CC=C1' # 氨基苯基
      ]

results = []

for core in cores_1:
    # 把SMILES字符串转化为分子格式
    core_mol=Chem.MolFromSmiles(core)
    warnings.filterwarnings('ignore',message='not removing hydrogen atom with dummy atom neighbors')

    # 对光敏剂与取代基进行排列组合
    lists=[[core], R1, R2]
    combinations = list(itertools.product(*lists))

    # 对所有的排列组合，分子片段相同的部分进行连接从而组合成新分子
    for combination in combinations:
        smi = '.'.join(combination)
        # 把SMILES格式变成分子格式
        RDLogger.DisableLog('rdApp.*')
        mol = Chem.MolFromSmiles(smi)
        RDLogger.EnableLog('rdApp.*')
        warnings.filterwarnings('ignore',message='not removing hydrogen atom with dummy atom neighbors')
        mol_com = Chem.molzip(mol)
        mol_smi=Chem.MolToSmiles(mol_com)
        # 将合成的分子加入结果库中
        results.append(mol_smi)
    
# 将所有数据转换为DataFrame
df = pd.DataFrame(results, columns=['smiles'])  # 设置列名为'smiles'

# 保存到CSV文件
df.to_csv('molecules_creation.csv', index=False)

# 读取 CSV 文件
df = pd.read_csv('molecules_creation.csv')

# 去除重复的行
df_cleaned_unique = df.drop_duplicates()

# 保存清洗后的数据到新 CSV 文件
df_cleaned_unique.to_csv("cleaned_moleculars.csv", index=False)

# 获取去重后的SMILES字符串列表
unique_smiles = df_cleaned_unique['smiles'].tolist()  # 使用列名'smiles'访问
print(f'去除重复后共有{len(unique_smiles)}个光敏剂分子')

# 选择光敏剂骨架结构来统一显示
template = Chem.MolFromSmiles('O=C1C2=C(C=CC=C2)SC3=CC=CC=C31')
AllChem.Compute2DCoords(template)

# 从unique_smiles生成分子列表
unique_mols_list = []
for smis in unique_smiles:
    mol = Chem.MolFromSmiles(smis)
    if mol:  # 确保分子有效
        AllChem.GenerateDepictionMatching2DStructure(mol, template)
        unique_mols_list.append(mol)

print(f'成功生成了{len(unique_mols_list)}个有效的光敏剂分子')

# 生成TX系列编号
tx_series = [f'TX-{i+1}' for i in range(len(unique_mols_list))]

# 将TX编号与SMILES对应关系保存到CSV
tx_df = pd.DataFrame({
    'TX_ID': tx_series,
    'SMILES': unique_smiles[:len(unique_mols_list)]
})
tx_df.to_csv('molecule_tx_mapping.csv', index=False)
print(f'已将分子TX编号映射保存至 molecule_tx_mapping.csv')

'''
# 设置每个图显示的分子数量
mols_per_page = 25  # 5*5=25
rows_per_page = 5
cols_per_page = 5

# 计算需要多少页
total_pages = len(unique_mols_list) // mols_per_page
if len(unique_mols_list) % mols_per_page > 0:
    total_pages += 1

# 创建输出目录
if not os.path.exists('molecule_images'):
    os.makedirs('molecule_images')

# 生成多页图像
for page in range(total_pages):
    start_idx = page * mols_per_page
    end_idx = min((page + 1) * mols_per_page, len(unique_mols_list))
    page_mols = unique_mols_list[start_idx:end_idx]
    
    fig, axes = plt.subplots(rows_per_page, cols_per_page, figsize=(15, 15))
    axes = axes.flatten()
    
    # 为本页中的每个分子生成图像
    for i, mol in enumerate(page_mols):
        img = Draw.MolToImage(mol, size=(300, 300))
        axes[i].imshow(img)
        tx_id = tx_series[start_idx + i]
        axes[i].set_title(tx_id, fontsize=12, fontweight='bold')
        axes[i].axis('off')
    
    # 对于未填满的网格位置，隐藏坐标轴
    for i in range(len(page_mols), mols_per_page):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'molecule_images/molecules_page_{page+1}.png', dpi=300)
    plt.close()
    print(f'已生成第 {page+1}/{total_pages} 页分子结构图')

print('所有分子结构图已保存到 molecule_images 文件夹')
'''
# 生成分子图像并保存
print('Task finished!')