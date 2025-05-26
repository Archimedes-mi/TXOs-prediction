import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# 蓝绿配色，与utils.py一致
COLORS = {
    'blue': '#1f77b4',
    'teal': '#39ac73',
    'dark_blue': '#035096',
    'cyan': '#40E0D0'
}
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
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.2)

def main():
    set_plotting_style()
    # 读取数据
    df = pd.read_csv('database.csv')
    # 只保留有效列
    df = df[['homo', 'lumo', 'energy']]
    # 绘制分布
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # HOMO
    sns.histplot(df['homo'], bins=30, kde=True, color=COLORS['blue'], ax=axes[0])
    axes[0].set_title('Distribution of HOMO')
    axes[0].set_xlabel('HOMO (eV)')
    axes[0].set_ylabel('Count')
    
    # LUMO
    sns.histplot(df['lumo'], bins=30, kde=True, color=COLORS['teal'], ax=axes[1])
    axes[1].set_title('Distribution of LUMO')
    axes[1].set_xlabel('LUMO (eV)')
    axes[1].set_ylabel('Count')
    
    # Energy
    sns.histplot(df['energy'], bins=30, kde=True, color=COLORS['cyan'], ax=axes[2])
    axes[2].set_title('Distribution of Energy')
    axes[2].set_xlabel('Energy (kcal/mol)')
    axes[2].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('stat_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main() 