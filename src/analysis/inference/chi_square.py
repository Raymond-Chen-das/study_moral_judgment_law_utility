"""
卡方檢定模組
===========

功能：
1. 文化圈差異的卡方檢定
2. 列聯表分析
3. 事後比較（多重比較校正）
4. 視覺化（Plotly互動式圖表）
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Dict, Optional
import plotly.graph_objects as go
import plotly.express as px
from itertools import combinations

from .effect_size import cramers_v, interpret_cramers_v


class ChiSquareTest:
    """卡方檢定類別"""
    
    def __init__(self, alpha: float = 0.05):
        """
        Parameters:
        -----------
        alpha : float
            顯著水準（預設0.05）
        """
        self.alpha = alpha
        self.results = {}
    
    def test_culture_difference(self, 
                               data: pd.DataFrame,
                               outcome_var: str = 'chose_lawful',
                               group_var: str = 'Cluster') -> Dict:
        """
        測試文化圈差異
        
        Parameters:
        -----------
        data : DataFrame
            場景層級資料
        outcome_var : str
            結果變數（二元：0/1）
        group_var : str
            分組變數（文化圈）
            
        Returns:
        --------
        dict : 檢定結果
        """
        # 建立列聯表
        contingency = pd.crosstab(
            data[group_var],
            data[outcome_var],
            margins=True
        )
        
        # 卡方檢定
        chi2, p_value, dof, expected_freq = stats.chi2_contingency(
            contingency.iloc[:-1, :-1]  # 排除margins
        )
        
        # 計算效果量
        n = len(data)
        n_groups = contingency.shape[0] - 1  # 排除margins
        n_outcomes = contingency.shape[1] - 1
        min_dim = min(n_groups - 1, n_outcomes - 1)
        
        v = cramers_v(chi2, n, min_dim)
        v_interpretation = interpret_cramers_v(v)
        
        # 計算各組比例
        proportions = {}
        for cluster in contingency.index[:-1]:  # 排除'All'
            total = contingency.loc[cluster, 'All']
            chose_lawful = contingency.loc[cluster, 1] if 1 in contingency.columns else 0
            proportions[cluster] = chose_lawful / total if total > 0 else 0
        
        # 儲存結果
        self.results = {
            'test': 'Chi-Square Test',
            'outcome_var': outcome_var,
            'group_var': group_var,
            'chi2': chi2,
            'p_value': p_value,
            'dof': dof,
            'cramers_v': v,
            'effect_interpretation': v_interpretation,
            'significant': p_value < self.alpha,
            'contingency_table': contingency,
            'expected_freq': expected_freq,
            'proportions': proportions,
            'n_total': n
        }
        
        return self.results
    
    def post_hoc_pairwise(self, 
                         data: pd.DataFrame,
                         outcome_var: str = 'chose_lawful',
                         group_var: str = 'Cluster',
                         correction: str = 'bonferroni') -> pd.DataFrame:
        """
        事後成對比較（多重比較校正）
        
        Parameters:
        -----------
        data : DataFrame
            資料
        outcome_var : str
            結果變數
        group_var : str
            分組變數
        correction : str
            校正方法：'bonferroni', 'fdr'
            
        Returns:
        --------
        DataFrame : 成對比較結果
        """
        groups = data[group_var].unique()
        comparisons = []
        
        for group1, group2 in combinations(groups, 2):
            # 篩選兩組資料
            subset = data[data[group_var].isin([group1, group2])]
            
            # 建立2x2列聯表
            contingency = pd.crosstab(
                subset[group_var],
                subset[outcome_var]
            )
            
            # 卡方檢定
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            
            # 效果量
            n = len(subset)
            min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
            v = cramers_v(chi2, n, min_dim)
            
            # 計算比例
            prop1 = (subset[subset[group_var] == group1][outcome_var].mean() 
                    if len(subset[subset[group_var] == group1]) > 0 else 0)
            prop2 = (subset[subset[group_var] == group2][outcome_var].mean()
                    if len(subset[subset[group_var] == group2]) > 0 else 0)
            
            comparisons.append({
                'Group1': group1,
                'Group2': group2,
                'Prop1': prop1,
                'Prop2': prop2,
                'Diff': prop1 - prop2,
                'chi2': chi2,
                'p_value': p,
                'cramers_v': v
            })
        
        results_df = pd.DataFrame(comparisons)
        
        # 多重比較校正
        if correction == 'bonferroni':
            n_tests = len(comparisons)
            results_df['p_adjusted'] = results_df['p_value'] * n_tests
            results_df['p_adjusted'] = results_df['p_adjusted'].clip(upper=1.0)
        elif correction == 'fdr':
            from statsmodels.stats.multitest import fdrcorrection
            _, p_adj = fdrcorrection(results_df['p_value'].values)
            results_df['p_adjusted'] = p_adj
        
        results_df['significant'] = results_df['p_adjusted'] < self.alpha
        
        return results_df
    
    def create_contingency_heatmap(self,
                                  contingency_table: pd.DataFrame,
                                  title: str = "列聯表熱圖") -> go.Figure:
        """
        建立列聯表熱圖（Plotly）
        
        Parameters:
        -----------
        contingency_table : DataFrame
            列聯表
        title : str
            圖表標題
            
        Returns:
        --------
        Figure : Plotly圖表
        """
        # 移除margins（如果存在）
        ct = contingency_table.copy()
        if 'All' in ct.index:
            ct = ct.drop('All')
        if 'All' in ct.columns:
            ct = ct.drop('All', axis=1)
        
        # 建立熱圖
        fig = go.Figure(data=go.Heatmap(
            z=ct.values,
            x=[f"選擇={col}" for col in ct.columns],
            y=ct.index,
            text=ct.values,
            texttemplate='%{text}',
            textfont={"size": 14},
            colorscale='Blues',
            colorbar=dict(title="次數")
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="道德選擇",
            yaxis_title="文化圈",
            font=dict(size=12),
            height=400,
            width=600
        )
        
        return fig
    
    def create_proportion_bar_chart(self,
                                   proportions: Dict[str, float],
                                   title: str = "各文化圈守法選擇比例") -> go.Figure:
        """
        建立比例長條圖
        
        Parameters:
        -----------
        proportions : dict
            各組比例 {組名: 比例}
        title : str
            圖表標題
            
        Returns:
        --------
        Figure : Plotly圖表
        """
        df = pd.DataFrame({
            'Cluster': list(proportions.keys()),
            'Proportion': list(proportions.values())
        })
        
        # 文化圈名稱對應
        cluster_names = {
            0: 'Western',
            1: 'Eastern', 
            2: 'Southern'
        }
        df['Cluster_Name'] = df['Cluster'].map(cluster_names)
        
        fig = go.Figure(data=[
            go.Bar(
                x=df['Cluster_Name'],
                y=df['Proportion'],
                text=[f"{p:.1%}" for p in df['Proportion']],
                textposition='auto',
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="文化圈",
            yaxis_title="守法選擇比例",
            yaxis=dict(tickformat='.0%', range=[0, 1]),
            font=dict(size=12),
            height=500,
            width=700,
            showlegend=False
        )
        
        # 添加參考線（全球平均）
        global_mean = sum(proportions.values()) / len(proportions)
        fig.add_hline(
            y=global_mean,
            line_dash="dash",
            line_color="red",
            annotation_text=f"全球平均: {global_mean:.1%}",
            annotation_position="right"
        )
        
        return fig
    
    def create_pairwise_comparison_plot(self,
                                       pairwise_df: pd.DataFrame,
                                       title: str = "成對比較結果") -> go.Figure:
        """
        建立成對比較視覺化
        
        Parameters:
        -----------
        pairwise_df : DataFrame
            成對比較結果
        title : str
            圖表標題
            
        Returns:
        --------
        Figure : Plotly圖表
        """
        # 建立比較標籤
        pairwise_df['Comparison'] = (pairwise_df['Group1'].astype(str) + 
                                     ' vs ' + 
                                     pairwise_df['Group2'].astype(str))
        
        # 顏色（顯著性）
        colors = ['red' if sig else 'gray' for sig in pairwise_df['significant']]
        
        fig = go.Figure()
        
        # 差異條形圖
        fig.add_trace(go.Bar(
            y=pairwise_df['Comparison'],
            x=pairwise_df['Diff'],
            orientation='h',
            text=[f"{d:+.3f}" for d in pairwise_df['Diff']],
            textposition='auto',
            marker_color=colors,
            name='比例差異'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="比例差異 (Group1 - Group2)",
            yaxis_title="比較組合",
            font=dict(size=12),
            height=400,
            width=800,
            showlegend=False
        )
        
        # 添加垂直線於0
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        
        return fig
    
    def generate_report(self) -> str:
        """
        生成文字報告
        
        Returns:
        --------
        str : 報告文字
        """
        if not self.results:
            return "尚未執行檢定"
        
        report = []
        report.append("=" * 60)
        report.append("卡方檢定結果")
        report.append("=" * 60)
        report.append(f"\n【檢定類型】 {self.results['test']}")
        report.append(f"【分組變數】 {self.results['group_var']}")
        report.append(f"【結果變數】 {self.results['outcome_var']}")
        report.append(f"【樣本數】   {self.results['n_total']:,}")
        report.append(f"\n【統計量】")
        report.append(f"  χ² = {self.results['chi2']:.3f}")
        report.append(f"  df = {self.results['dof']}")
        report.append(f"  p = {self.results['p_value']:.4f}")
        report.append(f"\n【效果量】")
        report.append(f"  Cramér's V = {self.results['cramers_v']:.3f}")
        report.append(f"  解釋: {self.results['effect_interpretation']}")
        report.append(f"\n【結論】")
        
        if self.results['significant']:
            report.append(f"  ✅ 文化圈間存在顯著差異 (p < {self.alpha})")
        else:
            report.append(f"  ❌ 文化圈間無顯著差異 (p ≥ {self.alpha})")
        
        report.append(f"\n【各組比例】")
        for cluster, prop in self.results['proportions'].items():
            cluster_names = {0: 'Western', 1: 'Eastern', 2: 'Southern'}
            name = cluster_names.get(cluster, str(cluster))
            report.append(f"  {name}: {prop:.1%}")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def run_chi_square_analysis(data: pd.DataFrame,
                           outcome_var: str = 'chose_lawful',
                           group_var: str = 'Cluster',
                           alpha: float = 0.05,
                           save_dir: Optional[str] = None) -> Dict:
    """
    執行完整的卡方分析流程
    
    Parameters:
    -----------
    data : DataFrame
        資料
    outcome_var : str
        結果變數
    group_var : str
        分組變數
    alpha : float
        顯著水準
    save_dir : str, optional
        儲存目錄
        
    Returns:
    --------
    dict : 分析結果
    """
    # 初始化
    chi_test = ChiSquareTest(alpha=alpha)
    
    # 主要檢定
    print("\n執行卡方檢定...")
    main_results = chi_test.test_culture_difference(data, outcome_var, group_var)
    
    # 事後比較
    print("執行事後成對比較...")
    pairwise_results = chi_test.post_hoc_pairwise(data, outcome_var, group_var)
    
    # 視覺化
    print("生成視覺化...")
    fig_heatmap = chi_test.create_contingency_heatmap(
        main_results['contingency_table'],
        title=f"列聯表：{group_var} × {outcome_var}"
    )
    
    fig_bar = chi_test.create_proportion_bar_chart(
        main_results['proportions'],
        title="各文化圈守法選擇比例"
    )
    
    fig_pairwise = chi_test.create_pairwise_comparison_plot(
        pairwise_results,
        title="文化圈成對比較"
    )
    
    # 儲存
    if save_dir:
        from pathlib import Path
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 儲存圖表
        fig_heatmap.write_html(str(save_path / 'contingency_heatmap.html'))
        fig_bar.write_html(str(save_path / 'proportion_bar_chart.html'))
        fig_pairwise.write_html(str(save_path / 'pairwise_comparison.html'))
        
        # 儲存表格
        main_results['contingency_table'].to_csv(
            save_path / 'contingency_table.csv',
            encoding='utf-8-sig'
        )
        pairwise_results.to_csv(
            save_path / 'pairwise_results.csv',
            index=False,
            encoding='utf-8-sig'
        )
        
        print(f"✅ 結果已儲存至: {save_dir}")
    
    # 輸出報告
    print(chi_test.generate_report())
    
    return {
        'main_results': main_results,
        'pairwise_results': pairwise_results,
        'figures': {
            'heatmap': fig_heatmap,
            'bar_chart': fig_bar,
            'pairwise': fig_pairwise
        }
    }