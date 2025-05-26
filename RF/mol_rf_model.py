import os
import numpy as np
import pandas as pd
# 设置matplotlib后端为Agg，避免线程相关错误
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import VarianceThreshold
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED
import optuna
import joblib
import warnings

# 禁用警告信息和RDKit日志
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

# 设置随机种子以确保可重现性
SEED = 42
np.random.seed(SEED)

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
        if len(y.shape) > 1 and y.shape[1] > 1:
            importances = np.zeros(X.shape[1])
            
            # 每个目标变量分别计算特征重要性
            for target_idx in range(y.shape[1]):
                rf.fit(X, y[:, target_idx])
                importances += rf.feature_importances_
            
            # 平均特征重要性
            importances /= y.shape[1]
        else:
            # 单目标变量
            rf.fit(X, y)
            importances = rf.feature_importances_
            
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
def visualize_feature_importance(importance, target_name, file_path="results/feature_importance_{}.png"):
    """
    可视化特征重要性
    
    参数:
        importance: 特征重要性数组
        target_name: 目标变量名称
        file_path: 保存文件路径模板
    """
    global DESCRIPTOR_NAMES
    
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path.format(target_name)), exist_ok=True)
    
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
            
    # 设置蓝绿配色
    color = '#39ac73'  # 绿松石色
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(n_features), sorted_importance, align='center', color=color)
    plt.yticks(range(n_features), feature_names)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.title(f'Top 10 Feature Importance for {target_name.upper()}')
    plt.gca().invert_yaxis()  # 反转y轴，使得重要性更大的特征在上方
    plt.tight_layout()
    plt.savefig(file_path.format(target_name), dpi=300)
    plt.close()
    
    print(f"特征重要性可视化已保存到 {file_path.format(target_name)}")

# 优化并训练随机森林模型
def optimize_rf_model(X, y, target_name, n_trials=60, n_startup_trials=10):
    """
    使用贝叶斯优化随机森林模型的超参数
    
    参数:
        X: 特征矩阵
        y: 目标变量
        target_name: 目标变量名称
        n_trials: 总试验次数
        n_startup_trials: 初始随机试验次数
        
    返回:
        最佳模型和最佳参数
    """
    print(f"\n开始优化{target_name}的随机森林模型...")
    
    # 定义优化目标函数
    def objective(trial):
        # 超参数搜索空间
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0)
        }
        
        # 创建随机森林模型
        rf = RandomForestRegressor(random_state=SEED, n_jobs=-1, **params)
        
        # 使用K折交叉验证评估模型
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        mae_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            mae_scores.append(mean_absolute_error(y_val, y_pred))
        
        return np.mean(mae_scores)
    
    # 创建带有随机初始探索的采样器
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=n_startup_trials,  # 初始随机探索次数
        seed=SEED
    )
    
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    
    # 获取最佳参数
    best_params = study.best_params
    print(f"\n{target_name}的最佳超参数:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # 使用最佳参数训练最终模型
    best_model = RandomForestRegressor(random_state=SEED, n_jobs=-1, **best_params)
    best_model.fit(X, y)
    
    return best_model, best_params

# 评估模型性能
def evaluate_model(model, X, y, target_name):
    """
    评估模型性能
    
    参数:
        model: 训练好的模型
        X: 特征矩阵
        y: 目标变量
        target_name: 目标变量名称
        
    返回:
        字典形式的评估指标
    """
    y_pred = model.predict(X)
    
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    print(f"\n{target_name}模型评估结果:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'y_true': y,
        'y_pred': y_pred
    }

# 可视化模型性能
def visualize_performance(metrics, target_name, save_path="results"):
    """
    可视化模型性能
    
    参数:
        metrics: 评估指标字典
        target_name: 目标变量名称
        save_path: 保存结果的路径
    """
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 设置蓝绿配色
    colors = {
        'blue': '#1f77b4',
        'teal': '#39ac73',
        'dark_blue': '#035096',
        'cyan': '#40E0D0'
    }
    
    # 根据目标变量名称选择颜色
    color = colors['blue'] if target_name == 'homo' else colors['teal'] if target_name == 'lumo' else colors['cyan']
    
    y_true = metrics['y_true']
    y_pred = metrics['y_pred']
    
    plt.figure(figsize=(10, 8))
    
    # 散点图
    plt.scatter(y_true, y_pred, alpha=0.7, color=color)
    
    # 对角线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    
    # 添加性能指标文本
    mae = metrics['mae']
    rmse = metrics['rmse']
    r2 = metrics['r2']
    
    plt.text(0.05, 0.95, f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}',
            transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 设置标题和轴标签
    titles = {'homo': 'HOMO (eV)', 'lumo': 'LUMO (eV)', 'energy': 'Energy (kcal/mol)'}
    plt.title(f'{titles.get(target_name, target_name.upper())} Prediction Performance')
    plt.xlabel(f'True {titles.get(target_name, target_name.upper())}')
    plt.ylabel(f'Predicted {titles.get(target_name, target_name.upper())}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(save_path, f'{target_name}_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"{target_name}性能可视化已保存到 {os.path.join(save_path, f'{target_name}_performance.png')}")

# 模型交叉验证
def cross_validate_model(X_all, y_all, target_name, best_params, n_splits=5):
    """
    使用交叉验证评估模型性能
    
    参数:
        X_all: 特征矩阵
        y_all: 目标变量
        target_name: 目标变量名称
        best_params: 最佳模型参数
        n_splits: 交叉验证折数
        
    返回:
        交叉验证结果
    """
    print(f"\n对{target_name}模型进行{n_splits}折交叉验证...")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    # 存储每一折的评估指标
    mae_scores = []
    rmse_scores = []
    r2_scores = []
    
    # 存储每一折的预测结果
    all_true = []
    all_pred = []
    
    # 存储交叉验证的结果
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_all)):
        print(f"处理第 {fold+1}/{n_splits} 折...")
        
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        
        # 创建并训练模型
        model = RandomForestRegressor(random_state=SEED, n_jobs=-1, **best_params)
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算性能指标
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # 存储结果
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        
        all_true.extend(y_test)
        all_pred.extend(y_pred)
        
        fold_results.append({
            'fold': fold + 1,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        })
        
        print(f"第 {fold+1} 折 - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    # 计算平均指标
    avg_mae = np.mean(mae_scores)
    avg_rmse = np.mean(rmse_scores)
    avg_r2 = np.mean(r2_scores)
    
    std_mae = np.std(mae_scores)
    std_rmse = np.std(rmse_scores)
    std_r2 = np.std(r2_scores)
    
    print(f"\n{n_splits}折交叉验证平均指标:")
    print(f"  平均 MAE: {avg_mae:.4f} ± {std_mae:.4f}")
    print(f"  平均 RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}")
    print(f"  平均 R²: {avg_r2:.4f} ± {std_r2:.4f}")
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(fold_results)
    
    # 保存交叉验证结果到CSV
    os.makedirs('results', exist_ok=True)
    results_df.to_csv(f'results/cv_results_{target_name}.csv', index=False)
    
    # 写入汇总结果到文本文件
    with open(os.path.join('results', f'cv_summary_{target_name}.txt'), 'w') as f:
        f.write(f"{n_splits}折交叉验证汇总结果:\n")
        f.write(f"\n{target_name.upper()} 预测性能:\n")
        f.write(f"MAE: {avg_mae:.4f} ± {std_mae:.4f}\n")
        f.write(f"RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}\n")
        f.write(f"R2: {avg_r2:.4f} ± {std_r2:.4f}\n")
    
    # 可视化交叉验证结果
    visualize_cv_results(results_df, target_name)
    
    return {
        'fold_metrics': results_df,
        'avg_metrics': {
            'mae': avg_mae,
            'rmse': avg_rmse,
            'r2': avg_r2
        },
        'std_metrics': {
            'mae': std_mae,
            'rmse': std_rmse,
            'r2': std_r2
        },
        'all_true': np.array(all_true),
        'all_pred': np.array(all_pred)
    }

# 可视化交叉验证结果
def visualize_cv_results(results_df, target_name):
    """
    可视化交叉验证的结果
    
    参数:
        results_df: 包含每一折结果的DataFrame
        target_name: 目标变量名称
    """
    # 确保目录存在
    os.makedirs('results', exist_ok=True)
    
    # 设置蓝绿配色
    colors = {
        'blue': '#1f77b4',
        'teal': '#39ac73',
        'dark_blue': '#035096',
        'cyan': '#40E0D0'
    }
    
    # 指标名称
    metrics = ['mae', 'rmse', 'r2']
    titles = ['Mean Absolute Error (MAE)', 'Root Mean Square Error (RMSE)', 'Coefficient of Determination (R²)']
    
    # 绘制每一折指标的条形图
    plt.figure(figsize=(18, 6))
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.subplot(1, 3, i+1)
        
        metric_color = 'blue' if metric == 'mae' else 'teal' if metric == 'rmse' else 'cyan'
        bars = plt.bar(results_df['fold'], results_df[metric], color=colors[metric_color], alpha=0.8)
        
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
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.title(f'{target_name.upper()} - {title}')
        plt.xlabel('Fold')
        plt.ylabel(metric.upper())
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/cv_metrics_{target_name}.png', dpi=300)
    plt.close()
    
    # 绘制折线图
    plt.figure(figsize=(12, 6))
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        metric_color = 'blue' if metric == 'mae' else 'teal' if metric == 'rmse' else 'cyan'
        plt.plot(results_df['fold'], results_df[metric], 
                 marker='o', linestyle='-', linewidth=2,
                 label=f'{metric.upper()}',
                 color=colors[metric_color])
    
    plt.xlabel('Fold')
    plt.ylabel('Metric Value')
    plt.title(f'{target_name.upper()} Metrics Across Folds')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/cv_line_{target_name}.png', dpi=300)
    plt.close()
    
    # 绘制箱型图
    plt.figure(figsize=(10, 6))
    
    # 转换数据格式以便于使用seaborn
    df_long = pd.melt(results_df, id_vars=['fold'], value_vars=['mae', 'rmse', 'r2'],
                      var_name='Metric', value_name='Value')
    
    sns.boxplot(x='Metric', y='Value', data=df_long, 
               palette=[colors['blue'], colors['teal'], colors['cyan']])
    
    plt.title(f'Cross-Validation Metrics Distribution for {target_name.upper()}')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'results/cv_boxplot_{target_name}.png', dpi=300)
    plt.close()
    
    print(f"{target_name}交叉验证结果可视化已保存")

# 可视化三个模型的性能比较
def visualize_combined_performance(metrics_dict):
    """
    可视化三个目标变量模型的性能比较
    
    参数:
        metrics_dict: 包含三个目标变量评估指标的字典
    """
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    # 设置蓝绿配色
    colors = {
        'blue': '#1f77b4',
        'teal': '#39ac73',
        'dark_blue': '#035096',
        'cyan': '#40E0D0'
    }
    
    # 准备数据
    target_names = list(metrics_dict.keys())
    metrics = ['mae', 'rmse', 'r2']
    titles = ['Mean Absolute Error (MAE)', 'Root Mean Square Error (RMSE)', 'Coefficient of Determination (R²)']
    
    # 绘制三个指标的条形图
    plt.figure(figsize=(15, 5))
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.subplot(1, 3, i+1)
        
        values = [metrics_dict[target][metric] for target in target_names]
        bars = plt.bar(target_names, values, color=[colors['blue'], colors['teal'], colors['cyan']])
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(values),
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.title(title)
        plt.xlabel('Property')
        plt.ylabel(metric.upper())
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('results/combined_performance_metrics.png', dpi=300)
    plt.close()
    
    print("三个模型的性能比较可视化已保存")

# 主函数
def main():
    """
    主函数：加载数据，训练模型，交叉验证，预测
    """
    print("开始随机森林分子属性预测...")
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    # 配置参数
    use_rdkit_desc = True
    morgan_size = 2048
    n_trials = 60
    n_startup_trials = 10
    target_properties = ['homo', 'lumo', 'energy']
    
    # 加载数据
    df = pd.read_csv('database.csv')
    print(f"加载了 {len(df)} 条数据")
    
    # 提取特征
    print("开始提取分子特征...")
    X_features = np.array([smiles_to_features(s, morgan_size=morgan_size, use_rdkit_desc=use_rdkit_desc) 
                         for s in df['smiles']])
    
    # 提取目标变量
    y_dict = {
        'homo': df['homo'].values,
        'lumo': df['lumo'].values,
        'energy': df['energy'].values
    }
    
    # 存储训练好的模型
    trained_models = {}
    feature_selectors = {}
    test_metrics = {}
    cv_results_all = {}
    
    # 分别对三个目标变量训练模型
    for target_name in target_properties:
        print(f"\n==== 开始处理 {target_name} 属性 ====")
        
        # 提取目标变量
        y = y_dict[target_name]
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=0.2, random_state=SEED
        )
        
        # 特征选择 (仅在训练集上进行)
        X_train_selected, selectors, feature_importance = select_features(X_train, y_train)
        
        # 可视化特征重要性
        if feature_importance is not None:
            visualize_feature_importance(feature_importance, target_name)
        
        # 将特征选择应用于测试集
        X_test_selected = X_test.copy()
        for selector in selectors:
            X_test_selected = selector.transform(X_test_selected)
        
        # 优化和训练模型
        best_model, best_params = optimize_rf_model(
            X_train_selected, y_train, target_name, 
            n_trials=n_trials, n_startup_trials=n_startup_trials
        )
        
        # 评估模型
        metrics = evaluate_model(best_model, X_test_selected, y_test, target_name)
        test_metrics[target_name] = {
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'r2': metrics['r2']
        }
        
        # 可视化性能
        visualize_performance(metrics, target_name)
        
        # 交叉验证
        cv_results = cross_validate_model(
            X_features, y, target_name, best_params, n_splits=5
        )
        cv_results_all[target_name] = cv_results
        
        # 存储模型和选择器
        trained_models[target_name] = best_model
        feature_selectors[target_name] = selectors
        
        # 保存模型和选择器
        joblib.dump(best_model, f'results/rf_model_{target_name}.pkl')
        joblib.dump(selectors, f'results/feature_selectors_{target_name}.pkl')
        
        print(f"{target_name}模型和选择器已保存")
    
    # 可视化三个模型的性能比较
    visualize_combined_performance(test_metrics)
    
    # 整合交叉验证结果并生成综合报告
    generate_comprehensive_report(cv_results_all, target_properties)
    
    # 预测TXs.csv中的化合物
    print("\n预测TXs.csv中的化合物...")
    
    # 加载TXs.csv
    txs_df = pd.read_csv('TXs.csv')
    
    # 提取特征
    txs_features = np.array([smiles_to_features(s, morgan_size=morgan_size, use_rdkit_desc=use_rdkit_desc) 
                           for s in txs_df['smiles']])
    
    # 预测每个目标变量
    for target_name in target_properties:
        # 应用特征选择
        X_txs = txs_features.copy()
        for selector in feature_selectors[target_name]:
            X_txs = selector.transform(X_txs)
        
        # 使用对应模型预测
        model = trained_models[target_name]
        predictions = model.predict(X_txs)
        
        # 将预测结果添加到DataFrame
        txs_df[target_name] = predictions
    
    # 保存预测结果
    txs_df.to_csv('TXs_rf_predicted.csv', index=False)
    print("预测结果已保存到 TXs_rf_predicted.csv")
    
    print("\n随机森林分子属性预测完成!")

# 生成综合报告
def generate_comprehensive_report(cv_results_all, target_properties):
    """
    整合三个目标属性的交叉验证结果，生成综合报告
    
    参数:
        cv_results_all: 包含三个属性交叉验证结果的字典
        target_properties: 目标属性列表
    """
    print("\n生成交叉验证综合报告...")
    
    # 创建综合结果DataFrame
    comprehensive_df = pd.DataFrame()
    
    # 为每个属性提取结果
    for target in target_properties:
        cv_results = cv_results_all[target]
        fold_metrics = cv_results['fold_metrics']
        
        # 重命名列名以包含属性名
        renamed_df = fold_metrics.copy()
        renamed_df.columns = ['fold'] + [f'{col}_{target}' for col in ['mae', 'rmse', 'r2']]
        
        # 合并到综合DataFrame
        if comprehensive_df.empty:
            comprehensive_df = renamed_df
        else:
            comprehensive_df = pd.merge(comprehensive_df, renamed_df, on='fold')
    
    # 计算每一折的平均指标
    comprehensive_df['mae_avg'] = comprehensive_df[[f'mae_{target}' for target in target_properties]].mean(axis=1)
    comprehensive_df['rmse_avg'] = comprehensive_df[[f'rmse_{target}' for target in target_properties]].mean(axis=1)
    comprehensive_df['r2_avg'] = comprehensive_df[[f'r2_{target}' for target in target_properties]].mean(axis=1)
    
    # 保存综合结果到CSV
    comprehensive_df.to_csv('results/comprehensive_cv_results.csv', index=False)
    
    # 生成综合报告文本文件
    with open(os.path.join('results', 'comprehensive_report.txt'), 'w') as f:
        f.write("5折交叉验证综合报告\n")
        f.write("====================\n\n")
        
        # 总体性能
        f.write("总体性能指标 (所有属性平均):\n")
        f.write(f"MAE: {comprehensive_df['mae_avg'].mean():.4f} ± {comprehensive_df['mae_avg'].std():.4f}\n")
        f.write(f"RMSE: {comprehensive_df['rmse_avg'].mean():.4f} ± {comprehensive_df['rmse_avg'].std():.4f}\n")
        f.write(f"R2: {comprehensive_df['r2_avg'].mean():.4f} ± {comprehensive_df['r2_avg'].std():.4f}\n\n")
        
        # 各属性性能
        f.write("各属性性能指标:\n")
        for target in target_properties:
            mae_mean = comprehensive_df[f'mae_{target}'].mean()
            mae_std = comprehensive_df[f'mae_{target}'].std()
            rmse_mean = comprehensive_df[f'rmse_{target}'].mean()
            rmse_std = comprehensive_df[f'rmse_{target}'].std()
            r2_mean = comprehensive_df[f'r2_{target}'].mean()
            r2_std = comprehensive_df[f'r2_{target}'].std()
            
            titles = {'homo': 'HOMO (eV)', 'lumo': 'LUMO (eV)', 'energy': 'Energy (kcal/mol)'}
            title = titles.get(target, target.upper())
            
            f.write(f"\n{title} 预测性能:\n")
            f.write(f"MAE: {mae_mean:.4f} ± {mae_std:.4f}\n")
            f.write(f"RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}\n")
            f.write(f"R2: {r2_mean:.4f} ± {r2_std:.4f}\n")
    
    # 使用箱型图可视化综合结果
    plot_comprehensive_boxplots(comprehensive_df, target_properties)
    
    # 创建折线图来比较三个属性在每一折上的性能
    plot_property_comparison_lines(comprehensive_df, target_properties)
    
    print("交叉验证综合报告已生成")

# 可视化综合箱型图
def plot_comprehensive_boxplots(results_df, target_properties):
    """
    使用箱型图可视化综合交叉验证结果
    
    参数:
        results_df: 综合结果DataFrame
        target_properties: 目标属性列表
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
    
    # 为每个指标类型创建箱型图
    # MAE
    plt.subplot(2, 2, 2)
    mae_cols = [f'mae_{target}' for target in target_properties]
    mae_data = results_df[mae_cols]
    mae_data.columns = target_properties  # 简化列名以便显示
    sns.boxplot(data=mae_data, palette=[colors['blue'], colors['teal'], colors['cyan']])
    plt.title('MAE Distribution for Each Target Property')
    plt.ylabel('MAE')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # RMSE
    plt.subplot(2, 2, 3)
    rmse_cols = [f'rmse_{target}' for target in target_properties]
    rmse_data = results_df[rmse_cols]
    rmse_data.columns = target_properties
    sns.boxplot(data=rmse_data, palette=[colors['blue'], colors['teal'], colors['cyan']])
    plt.title('RMSE Distribution for Each Target Property')
    plt.ylabel('RMSE')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # R^2
    plt.subplot(2, 2, 4)
    r2_cols = [f'r2_{target}' for target in target_properties]
    r2_data = results_df[r2_cols]
    r2_data.columns = target_properties
    sns.boxplot(data=r2_data, palette=[colors['blue'], colors['teal'], colors['cyan']])
    plt.title('R² Distribution for Each Target Property')
    plt.ylabel('R²')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig('results/comprehensive_boxplots.png', dpi=300)
    plt.close()
    
    print("综合箱型图分析已保存")

# 绘制属性比较折线图
def plot_property_comparison_lines(comprehensive_df, target_properties):
    """
    创建折线图来比较三个属性在每一折上的性能
    
    参数:
        comprehensive_df: 综合结果DataFrame
        target_properties: 目标属性列表
    """
    # 设置蓝绿配色
    colors = {
        'blue': '#1f77b4',
        'teal': '#39ac73',
        'dark_blue': '#035096',
        'cyan': '#40E0D0'
    }
    
    # 为每种指标创建折线图
    metrics = ['mae', 'rmse', 'r2']
    titles = ['Mean Absolute Error (MAE)', 'Root Mean Square Error (RMSE)', 'Coefficient of Determination (R²)']
    
    for metric, title in zip(metrics, titles):
        plt.figure(figsize=(12, 6))
        
        # 为每个属性绘制折线
        for target in target_properties:
            color_name = 'blue' if target == 'homo' else 'teal' if target == 'lumo' else 'cyan'
            
            plt.plot(comprehensive_df['fold'], comprehensive_df[f'{metric}_{target}'],
                     marker='o', linestyle='-', linewidth=2,
                     label=f'{target.upper()}',
                     color=colors[color_name])
        
        # 设置图表标题和标签
        plt.title(f'{title} Comparison Across Properties')
        plt.xlabel('Fold')
        plt.ylabel(title)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(f'results/{metric}_property_comparison.png', dpi=300)
        plt.close()
    
    print("属性比较折线图已保存")

if __name__ == "__main__":
    main() 
