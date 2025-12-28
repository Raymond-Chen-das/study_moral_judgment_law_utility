"""
t檢定與單變量分析模組
==================

功能：
1. 獨立樣本t檢定
2. 點二系列相關
3. ANOVA單因子變異數分析
4. 批次單變量分析
5. 視覺化（箱型圖、小提琴圖）
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .effect_size import (cohens_d, interpret_cohens_d,
                         eta_squared, interpret_eta_squared,
                         point_biserial_correlation, interpret_correlation)


class UnivariateAnalysis:
    """單變量分析類別"""
    
    def __init__(self, alpha: float = 0.05):
        """
        Parameters:
        -----------
        alpha : float
            顯著水準
        """
        self.alpha = alpha
        self.results = []
    
    def independent_t_test(self,
                          data: pd.DataFrame,
                          outcome_var: str,
                          group_var: str,
                          equal_var: bool = True) -> Dict:
        """
        獨立樣本t檢定
        
        Parameters:
        -----------
        data : DataFrame
            資料
        outcome_var : str
            連續型結果變數
        group_var : str
            二元分組變數
        equal_var : bool
            是否假設變異數相等
            
        Returns:
        --------
        dict : 檢定結果
        """
        groups = data[group_var].unique()
        if len(groups) != 2:
            raise ValueError(f"{group_var} 必須恰好有2個類別")
        
        # 提取兩組數據
        group1 = data[data[group_var] == groups[0]][outcome_var].dropna()
        group2 = data[data[group_var] == groups[1]][outcome_var].dropna()
        
        # t檢定
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
        
        # 效果量
        d = cohens_d(group1.values, group2.values)
        d_interpretation = interpret_cohens_d(d)
        
        # 描述統計
        result = {
            'test': 't-test',
            'outcome_var': outcome_var,
            'group_var': group_var,
            'group1': groups[0],
            'group2': groups[1],
            'n1': len(group1),
            'n2': len(group2),
            'mean1': group1.mean(),
            'mean2': group2.mean(),
            'std1': group1.std(),
            'std2': group2.std(),
            'mean_diff': group1.mean() - group2.mean(),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': d,
            'effect_interpretation': d_interpretation,
            'significant': p_value < self.alpha
        }
        
        return result
    
    def point_biserial_test(self,
                           data: pd.DataFrame,
                           outcome_var: str,
                           continuous_var: str) -> Dict:
        """
        點二系列相關（二元結果 × 連續變數）
        
        Parameters:
        -----------
        data : DataFrame
            資料
        outcome_var : str
            二元結果變數（0/1）
        continuous_var : str
            連續型變數
            
        Returns:
        --------
        dict : 檢定結果
        """
        # 移除缺失值
        clean_data = data[[outcome_var, continuous_var]].dropna()
        
        # 點二系列相關
        r, p_value = point_biserial_correlation(
            clean_data[outcome_var].values,
            clean_data[continuous_var].values
        )
        
        r_interpretation = interpret_correlation(r)
        
        # 描述統計
        group0 = clean_data[clean_data[outcome_var] == 0][continuous_var]
        group1 = clean_data[clean_data[outcome_var] == 1][continuous_var]
        
        result = {
            'test': 'point-biserial',
            'outcome_var': outcome_var,
            'continuous_var': continuous_var,
            'n': len(clean_data),
            'r': r,
            'p_value': p_value,
            'correlation_interpretation': r_interpretation,
            'significant': p_value < self.alpha,
            'mean_when_0': group0.mean() if len(group0) > 0 else np.nan,
            'mean_when_1': group1.mean() if len(group1) > 0 else np.nan,
            'std_when_0': group0.std() if len(group0) > 0 else np.nan,
            'std_when_1': group1.std() if len(group1) > 0 else np.nan
        }
        
        return result
    
    def one_way_anova(self,
                     data: pd.DataFrame,
                     outcome_var: str,
                     group_var: str) -> Dict:
        """
        單因子變異數分析（ANOVA）
        
        Parameters:
        -----------
        data : DataFrame
            資料
        outcome_var : str
            連續型結果變數
        group_var : str
            分組變數（≥2組）
            
        Returns:
        --------
        dict : 檢定結果
        """
        groups = data[group_var].unique()
        
        # 提取各組數據
        group_data = [data[data[group_var] == g][outcome_var].dropna().values
                     for g in groups]
        
        # ANOVA
        f_stat, p_value = stats.f_oneway(*group_data)
        
        # 效果量
        df_between = len(groups) - 1
        df_within = len(data) - len(groups)
        eta2 = eta_squared(f_stat, df_between, df_within)
        eta2_interpretation = interpret_eta_squared(eta2)
        
        # 各組描述統計
        group_stats = []
        for g in groups:
            g_data = data[data[group_var] == g][outcome_var].dropna()
            group_stats.append({
                'group': g,
                'n': len(g_data),
                'mean': g_data.mean(),
                'std': g_data.std()
            })
        
        result = {
            'test': 'ANOVA',
            'outcome_var': outcome_var,
            'group_var': group_var,
            'n_groups': len(groups),
            'f_statistic': f_stat,
            'p_value': p_value,
            'df_between': df_between,
            'df_within': df_within,
            'eta_squared': eta2,
            'effect_interpretation': eta2_interpretation,
            'significant': p_value < self.alpha,
            'group_stats': group_stats
        }
        
        return result
    
    def batch_univariate_tests(self,
                              data: pd.DataFrame,
                              outcome_var: str,
                              test_vars: List[str],
                              var_types: Dict[str, str]) -> pd.DataFrame:
        """
        批次執行單變量檢定
        
        Parameters:
        -----------
        data : DataFrame
            資料
        outcome_var : str
            結果變數（二元：0/1）
        test_vars : list
            要測試的變數列表
        var_types : dict
            變數類型 {變數名: 'continuous' 或 'categorical'}
            
        Returns:
        --------
        DataFrame : 檢定結果摘要表
        """
        results = []
        
        for var in test_vars:
            var_type = var_types.get(var, 'unknown')
            
            try:
                if var_type == 'continuous':
                    # 點二系列相關
                    result = self.point_biserial_test(data, outcome_var, var)
                    
                    results.append({
                        '變數': var,
                        '類型': '連續',
                        '檢定方法': '點二系列相關',
                        '統計量': f"r = {result['r']:.3f}",
                        'p值': result['p_value'],
                        '效果量': result['r'],
                        '效果解釋': result['correlation_interpretation'],
                        '顯著': '✓' if result['significant'] else '',
                        '樣本數': result['n']
                    })
                    
                elif var_type == 'categorical':
                    # 檢查類別數
                    n_categories = data[var].nunique()
                    
                    if n_categories == 2:
                        # t檢定（反轉：以類別為分組）
                        result = self.independent_t_test(
                            data, outcome_var, var, equal_var=False
                        )
                        
                        results.append({
                            '變數': var,
                            '類型': '類別(2)',
                            '檢定方法': 't檢定',
                            '統計量': f"t = {result['t_statistic']:.3f}",
                            'p值': result['p_value'],
                            '效果量': result['cohens_d'],
                            '效果解釋': result['effect_interpretation'],
                            '顯著': '✓' if result['significant'] else '',
                            '樣本數': result['n1'] + result['n2']
                        })
                        
                    else:
                        # ANOVA
                        result = self.one_way_anova(data, outcome_var, var)
                        
                        results.append({
                            '變數': var,
                            '類型': f"類別({n_categories})",
                            '檢定方法': 'ANOVA',
                            '統計量': f"F = {result['f_statistic']:.3f}",
                            'p值': result['p_value'],
                            '效果量': result['eta_squared'],
                            '效果解釋': result['effect_interpretation'],
                            '顯著': '✓' if result['significant'] else '',
                            '樣本數': len(data)
                        })
                        
            except Exception as e:
                print(f"分析 {var} 時發生錯誤: {e}")
        
        return pd.DataFrame(results)
    
    def create_boxplot(self,
                      data: pd.DataFrame,
                      outcome_var: str,
                      group_var: str,
                      title: str = None) -> go.Figure:
        """
        建立箱型圖
        
        Parameters:
        -----------
        data : DataFrame
            資料
        outcome_var : str
            連續型變數（Y軸）
        group_var : str
            分組變數（X軸）
        title : str
            圖表標題
            
        Returns:
        --------
        Figure : Plotly圖表
        """
        fig = go.Figure()
        
        for group in data[group_var].unique():
            group_data = data[data[group_var] == group][outcome_var]
            
            fig.add_trace(go.Box(
                y=group_data,
                name=str(group),
                boxmean='sd'  # 顯示平均值和標準差
            ))
        
        if title is None:
            title = f"{outcome_var} 依 {group_var} 分組"
        
        fig.update_layout(
            title=title,
            xaxis_title=group_var,
            yaxis_title=outcome_var,
            font=dict(size=12),
            height=500,
            width=700,
            showlegend=True
        )
        
        return fig
    
    def create_violin_plot(self,
                          data: pd.DataFrame,
                          outcome_var: str,
                          group_var: str,
                          title: str = None) -> go.Figure:
        """
        建立小提琴圖
        
        Parameters:
        -----------
        data : DataFrame
            資料
        outcome_var : str
            連續型變數
        group_var : str
            分組變數
        title : str
            圖表標題
            
        Returns:
        --------
        Figure : Plotly圖表
        """
        fig = go.Figure()
        
        for group in data[group_var].unique():
            group_data = data[data[group_var] == group][outcome_var]
            
            fig.add_trace(go.Violin(
                y=group_data,
                name=str(group),
                box_visible=True,
                meanline_visible=True
            ))
        
        if title is None:
            title = f"{outcome_var} 分佈（依 {group_var} 分組）"
        
        fig.update_layout(
            title=title,
            xaxis_title=group_var,
            yaxis_title=outcome_var,
            font=dict(size=12),
            height=500,
            width=700
        )
        
        return fig
    
    def create_summary_forest_plot(self,
                                  results_df: pd.DataFrame,
                                  title: str = "單變量分析效果量森林圖") -> go.Figure:
        """
        建立效果量森林圖
        
        Parameters:
        -----------
        results_df : DataFrame
            batch_univariate_tests的結果
        title : str
            圖表標題
            
        Returns:
        --------
        Figure : Plotly圖表
        """
        # 排序（按效果量）
        plot_data = results_df.sort_values('效果量', ascending=True).copy()
        
        # 顏色（依顯著性）
        colors = ['red' if x == '✓' else 'gray' for x in plot_data['顯著']]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=plot_data['效果量'],
            y=plot_data['變數'],
            mode='markers',
            marker=dict(
                size=10,
                color=colors,
                line=dict(width=1, color='black')
            ),
            text=plot_data['效果解釋'],
            hovertemplate='<b>%{y}</b><br>效果量: %{x:.3f}<br>%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="效果量",
            yaxis_title="變數",
            font=dict(size=12),
            height=max(400, len(plot_data) * 30),
            width=800,
            showlegend=False
        )
        
        # 添加垂直參考線
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        
        return fig


def run_univariate_analysis(data: pd.DataFrame,
                           outcome_var: str,
                           test_vars: List[str],
                           var_types: Dict[str, str],
                           alpha: float = 0.05,
                           save_dir: Optional[str] = None) -> Dict:
    """
    執行完整的單變量分析流程
    
    Parameters:
    -----------
    data : DataFrame
        資料
    outcome_var : str
        結果變數
    test_vars : list
        要測試的變數
    var_types : dict
        變數類型
    alpha : float
        顯著水準
    save_dir : str, optional
        儲存目錄
        
    Returns:
    --------
    dict : 分析結果
    """
    analyzer = UnivariateAnalysis(alpha=alpha)
    
    # 批次檢定
    print("\n執行批次單變量檢定...")
    results_df = analyzer.batch_univariate_tests(
        data, outcome_var, test_vars, var_types
    )
    
    print(f"\n完成 {len(results_df)} 個檢定")
    print(f"顯著變數數: {(results_df['顯著'] == '✓').sum()}")
    
    # 視覺化
    print("\n生成視覺化...")
    fig_forest = analyzer.create_summary_forest_plot(results_df)
    
    # 儲存
    if save_dir:
        from pathlib import Path
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 儲存表格
        results_df.to_csv(
            save_path / 'univariate_results.csv',
            index=False,
            encoding='utf-8-sig'
        )
        
        # 儲存圖表
        fig_forest.write_html(str(save_path / 'effect_size_forest_plot.html'))
        
        print(f"✅ 結果已儲存至: {save_dir}")
    
    return {
        'results_table': results_df,
        'figures': {'forest_plot': fig_forest}
    }