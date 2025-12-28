"""
效果量計算模組
=============

提供各種效果量的計算函數：
- Cohen's d (t檢定效果量)
- Cramér's V (卡方檢定效果量)
- η² (ANOVA效果量)
- Odds Ratio (邏輯迴歸效果量)

參考標準：
- Cohen's d: small=0.2, medium=0.5, large=0.8
- Cramér's V: small=0.1, medium=0.3, large=0.5
- η²: small=0.01, medium=0.06, large=0.14
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy import stats


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    計算Cohen's d效果量
    
    Parameters:
    -----------
    group1, group2 : array-like
        兩組數據
        
    Returns:
    --------
    float : Cohen's d值
    
    解釋標準：
    - |d| < 0.2: 可忽略
    - 0.2 ≤ |d| < 0.5: 小效果
    - 0.5 ≤ |d| < 0.8: 中等效果
    - |d| ≥ 0.8: 大效果
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # 合併標準差（pooled standard deviation）
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    return d


def interpret_cohens_d(d: float) -> str:
    """解釋Cohen's d的大小"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "可忽略"
    elif abs_d < 0.5:
        return "小效果"
    elif abs_d < 0.8:
        return "中等效果"
    else:
        return "大效果"


def cramers_v(chi2: float, n: int, min_dim: int) -> float:
    """
    計算Cramér's V效果量（卡方檢定）
    
    Parameters:
    -----------
    chi2 : float
        卡方統計量
    n : int
        總樣本數
    min_dim : int
        min(行數-1, 列數-1)
        
    Returns:
    --------
    float : Cramér's V值
    
    解釋標準：
    - V < 0.1: 可忽略
    - 0.1 ≤ V < 0.3: 小效果
    - 0.3 ≤ V < 0.5: 中等效果
    - V ≥ 0.5: 大效果
    """
    v = np.sqrt(chi2 / (n * min_dim))
    return v


def interpret_cramers_v(v: float) -> str:
    """解釋Cramér's V的大小"""
    if v < 0.1:
        return "可忽略"
    elif v < 0.3:
        return "小效果"
    elif v < 0.5:
        return "中等效果"
    else:
        return "大效果"


def eta_squared(f_statistic: float, df_between: int, df_within: int) -> float:
    """
    計算η²效果量（ANOVA）
    
    Parameters:
    -----------
    f_statistic : float
        F統計量
    df_between : int
        組間自由度
    df_within : int
        組內自由度
        
    Returns:
    --------
    float : η²值
    
    解釋標準：
    - η² < 0.01: 可忽略
    - 0.01 ≤ η² < 0.06: 小效果
    - 0.06 ≤ η² < 0.14: 中等效果
    - η² ≥ 0.14: 大效果
    """
    eta2 = (f_statistic * df_between) / (f_statistic * df_between + df_within)
    return eta2


def interpret_eta_squared(eta2: float) -> str:
    """解釋η²的大小"""
    if eta2 < 0.01:
        return "可忽略"
    elif eta2 < 0.06:
        return "小效果"
    elif eta2 < 0.14:
        return "中等效果"
    else:
        return "大效果"


def odds_ratio_ci(odds_ratio: float, se: float, alpha: float = 0.05) -> Tuple[float, float]:
    """
    計算Odds Ratio的信賴區間
    
    Parameters:
    -----------
    odds_ratio : float
        勝算比
    se : float
        標準誤
    alpha : float
        顯著水準（預設0.05）
        
    Returns:
    --------
    tuple : (下界, 上界)
    """
    z = stats.norm.ppf(1 - alpha / 2)
    log_or = np.log(odds_ratio)
    
    lower = np.exp(log_or - z * se)
    upper = np.exp(log_or + z * se)
    
    return lower, upper


def interpret_odds_ratio(or_value: float) -> str:
    """
    解釋Odds Ratio的實質意義
    
    OR = 1: 無關聯
    OR > 1: 正向關聯
    OR < 1: 負向關聯
    """
    if 0.9 <= or_value <= 1.1:
        return "無實質關聯"
    elif or_value > 1.1:
        if or_value < 1.5:
            return "弱正向關聯"
        elif or_value < 3.0:
            return "中等正向關聯"
        else:
            return "強正向關聯"
    else:  # or_value < 0.9
        if or_value > 0.67:
            return "弱負向關聯"
        elif or_value > 0.33:
            return "中等負向關聯"
        else:
            return "強負向關聯"


def point_biserial_correlation(binary_var: np.ndarray, continuous_var: np.ndarray) -> Tuple[float, float]:
    """
    計算點二系列相關（Point-Biserial Correlation）
    用於二元變數與連續變數的關聯
    
    Parameters:
    -----------
    binary_var : array-like
        二元變數（0/1）
    continuous_var : array-like
        連續變數
        
    Returns:
    --------
    tuple : (相關係數, p值)
    
    解釋：與Pearson相關係數相同標準
    """
    r, p = stats.pointbiserialr(binary_var, continuous_var)
    return r, p


def interpret_correlation(r: float) -> str:
    """解釋相關係數的大小"""
    abs_r = abs(r)
    if abs_r < 0.1:
        return "可忽略"
    elif abs_r < 0.3:
        return "弱相關"
    elif abs_r < 0.5:
        return "中等相關"
    else:
        return "強相關"


def calculate_all_effect_sizes(data: pd.DataFrame,
                               group_var: str,
                               outcome_var: str,
                               test_type: str = 'auto') -> dict:
    """
    根據變數類型自動計算適當的效果量
    
    Parameters:
    -----------
    data : DataFrame
        資料
    group_var : str
        分組變數名稱
    outcome_var : str
        結果變數名稱
    test_type : str
        'auto', 't-test', 'chi-square', 'anova'
        
    Returns:
    --------
    dict : 效果量結果
    """
    results = {}
    
    # 自動判斷測試類型
    if test_type == 'auto':
        group_unique = data[group_var].nunique()
        outcome_unique = data[outcome_var].nunique()
        
        if outcome_unique == 2:
            test_type = 'chi-square'
        elif group_unique == 2:
            test_type = 't-test'
        else:
            test_type = 'anova'
    
    results['test_type'] = test_type
    
    # 根據測試類型計算效果量
    if test_type == 't-test':
        groups = data[group_var].unique()
        if len(groups) != 2:
            raise ValueError("t檢定需要正好2組")
        
        group1 = data[data[group_var] == groups[0]][outcome_var].values
        group2 = data[data[group_var] == groups[1]][outcome_var].values
        
        d = cohens_d(group1, group2)
        results['cohens_d'] = d
        results['interpretation'] = interpret_cohens_d(d)
        
    elif test_type == 'chi-square':
        contingency = pd.crosstab(data[group_var], data[outcome_var])
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        
        n = len(data)
        min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
        v = cramers_v(chi2, n, min_dim)
        
        results['cramers_v'] = v
        results['interpretation'] = interpret_cramers_v(v)
        results['chi2'] = chi2
        results['p_value'] = p
        
    elif test_type == 'anova':
        groups = [data[data[group_var] == g][outcome_var].values 
                 for g in data[group_var].unique()]
        f_stat, p = stats.f_oneway(*groups)
        
        df_between = len(groups) - 1
        df_within = len(data) - len(groups)
        
        eta2 = eta_squared(f_stat, df_between, df_within)
        results['eta_squared'] = eta2
        results['interpretation'] = interpret_eta_squared(eta2)
        results['f_statistic'] = f_stat
        results['p_value'] = p
    
    return results


def format_effect_size_report(effect_size_dict: dict) -> str:
    """
    格式化效果量報告
    
    Parameters:
    -----------
    effect_size_dict : dict
        calculate_all_effect_sizes的輸出
        
    Returns:
    --------
    str : 格式化的報告文字
    """
    test_type = effect_size_dict.get('test_type', 'unknown')
    
    if test_type == 't-test':
        d = effect_size_dict['cohens_d']
        interp = effect_size_dict['interpretation']
        return f"Cohen's d = {d:.3f} ({interp})"
        
    elif test_type == 'chi-square':
        v = effect_size_dict['cramers_v']
        interp = effect_size_dict['interpretation']
        chi2 = effect_size_dict['chi2']
        p = effect_size_dict['p_value']
        return f"Cramér's V = {v:.3f} ({interp}), χ² = {chi2:.2f}, p = {p:.4f}"
        
    elif test_type == 'anova':
        eta2 = effect_size_dict['eta_squared']
        interp = effect_size_dict['interpretation']
        f = effect_size_dict['f_statistic']
        p = effect_size_dict['p_value']
        return f"η² = {eta2:.3f} ({interp}), F = {f:.2f}, p = {p:.4f}"
    
    return "無法格式化效果量報告"


# 批次計算效果量
def batch_effect_sizes(data: pd.DataFrame,
                       group_var: str,
                       outcome_vars: list,
                       test_type: str = 'auto') -> pd.DataFrame:
    """
    批次計算多個結果變數的效果量
    
    Parameters:
    -----------
    data : DataFrame
        資料
    group_var : str
        分組變數
    outcome_vars : list
        結果變數列表
    test_type : str
        測試類型
        
    Returns:
    --------
    DataFrame : 效果量摘要表
    """
    results = []
    
    for var in outcome_vars:
        try:
            effect = calculate_all_effect_sizes(data, group_var, var, test_type)
            
            row = {
                'outcome_variable': var,
                'test_type': effect['test_type']
            }
            
            if 'cohens_d' in effect:
                row['effect_size'] = effect['cohens_d']
                row['effect_type'] = "Cohen's d"
            elif 'cramers_v' in effect:
                row['effect_size'] = effect['cramers_v']
                row['effect_type'] = "Cramér's V"
            elif 'eta_squared' in effect:
                row['effect_size'] = effect['eta_squared']
                row['effect_type'] = "η²"
            
            row['interpretation'] = effect.get('interpretation', '')
            
            if 'p_value' in effect:
                row['p_value'] = effect['p_value']
            
            results.append(row)
            
        except Exception as e:
            print(f"計算 {var} 的效果量時發生錯誤: {e}")
    
    return pd.DataFrame(results)