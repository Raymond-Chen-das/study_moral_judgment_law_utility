"""
交互作用分析模組
==============

功能：
1. 邏輯迴歸交互作用項檢驗
2. Likelihood Ratio Test (LRT)
3. Simple Slope Analysis
4. 交互作用視覺化
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class InteractionAnalysis:
    """交互作用分析類別"""
    
    def __init__(self, alpha: float = 0.05):
        """
        Parameters:
        -----------
        alpha : float
            顯著水準
        """
        self.alpha = alpha
        self.base_model = None
        self.interaction_model = None
    
    def test_interaction(self,
                        data: pd.DataFrame,
                        outcome_var: str,
                        main_effects: List[str],
                        interaction_terms: List[Tuple[str, str]]) -> Dict:
        """
        測試交互作用效應
        
        Parameters:
        -----------
        data : DataFrame
            資料
        outcome_var : str
            結果變數
        main_effects : list
            主效應變數
        interaction_terms : list of tuples
            交互作用項，例如 [('Cluster', 'political'), ('age', 'religious')]
            
        Returns:
        --------
        dict : 交互作用檢定結果
        """
        # 準備資料
        model_data = data[[outcome_var] + main_effects].dropna()
        
        # 建立基礎模型（無交互作用）
        X_base = model_data[main_effects].copy()
        
        # 處理類別變數
        for col in X_base.columns:
            if X_base[col].dtype == 'object' or X_base[col].dtype.name == 'category':
                dummies = pd.get_dummies(X_base[col], prefix=col, drop_first=True)
                X_base = pd.concat([X_base.drop(col, axis=1), dummies], axis=1)
        
        X_base = sm.add_constant(X_base)
        y = model_data[outcome_var]
        
        self.base_model = sm.GLM(y, X_base, family=sm.families.Binomial()).fit()
        
        # 建立交互作用模型
        X_interaction = X_base.copy()
        
        interaction_names = []
        for var1, var2 in interaction_terms:
            # 檢查變數是否存在
            v1_cols = [c for c in X_base.columns if c.startswith(var1)]
            v2_cols = [c for c in X_base.columns if c.startswith(var2)]
            
            if not v1_cols or not v2_cols:
                # 使用原始變數
                if var1 in model_data.columns and var2 in model_data.columns:
                    interaction_col = f"{var1}_x_{var2}"
                    X_interaction[interaction_col] = (
                        model_data[var1].values * model_data[var2].values
                    )
                    interaction_names.append(interaction_col)
            else:
                # 類別變數的交互作用
                for v1_col in v1_cols:
                    for v2_col in v2_cols:
                        interaction_col = f"{v1_col}_x_{v2_col}"
                        X_interaction[interaction_col] = (
                            X_interaction[v1_col] * X_interaction[v2_col]
                        )
                        interaction_names.append(interaction_col)
        
        self.interaction_model = sm.GLM(
            y, X_interaction, family=sm.families.Binomial()
        ).fit()
        
        # Likelihood Ratio Test
        lrt_statistic = 2 * (self.interaction_model.llf - self.base_model.llf)
        lrt_dof = self.interaction_model.df_model - self.base_model.df_model
        lrt_p = 1 - stats.chi2.cdf(lrt_statistic, lrt_dof)
        
        # 提取交互作用項係數
        interaction_coefs = []
        for name in interaction_names:
            if name in self.interaction_model.params.index:
                coef = self.interaction_model.params[name]
                se = self.interaction_model.bse[name]
                z = self.interaction_model.tvalues[name]
                p = self.interaction_model.pvalues[name]
                
                interaction_coefs.append({
                    '交互作用項': name,
                    '係數': coef,
                    '標準誤': se,
                    'z值': z,
                    'p值': p,
                    'Odds_Ratio': np.exp(coef),
                    '顯著': p < self.alpha
                })
        
        interaction_coefs_df = pd.DataFrame(interaction_coefs)
        
        return {
            'base_model': self.base_model,
            'interaction_model': self.interaction_model,
            'lrt_statistic': lrt_statistic,
            'lrt_dof': lrt_dof,
            'lrt_p_value': lrt_p,
            'lrt_significant': lrt_p < self.alpha,
            'interaction_coefficients': interaction_coefs_df,
            'aic_base': self.base_model.aic,
            'aic_interaction': self.interaction_model.aic,
            'aic_improvement': self.base_model.aic - self.interaction_model.aic
        }
    
    def simple_slopes(self,
                     data: pd.DataFrame,
                     outcome_var: str,
                     focal_var: str,
                     moderator_var: str,
                     moderator_values: Optional[List] = None) -> pd.DataFrame:
        """
        Simple Slope Analysis（簡單斜率分析）
        
        Parameters:
        -----------
        data : DataFrame
            資料
        outcome_var : str
            結果變數
        focal_var : str
            焦點變數
        moderator_var : str
            調節變數
        moderator_values : list, optional
            調節變數的特定值（預設：mean ± 1 SD）
            
        Returns:
        --------
        DataFrame : 簡單斜率結果
        """
        if moderator_values is None:
            # 使用 mean ± 1 SD
            mean = data[moderator_var].mean()
            std = data[moderator_var].std()
            moderator_values = [mean - std, mean, mean + std]
            value_labels = ['低(-1SD)', '中(Mean)', '高(+1SD)']
        else:
            value_labels = [str(v) for v in moderator_values]
        
        slopes = []
        
        for value, label in zip(moderator_values, value_labels):
            # 建立條件資料
            conditional_data = data.copy()
            conditional_data[f'{moderator_var}_centered'] = (
                conditional_data[moderator_var] - value
            )
            
            # 擬合條件模型
            formula = f"{outcome_var} ~ {focal_var} * {moderator_var}_centered"
            
            try:
                model = smf.glm(
                    formula=formula,
                    data=conditional_data,
                    family=sm.families.Binomial()
                ).fit()
                
                # 提取focal_var的係數
                if focal_var in model.params.index:
                    slope = model.params[focal_var]
                    se = model.bse[focal_var]
                    p = model.pvalues[focal_var]
                    
                    slopes.append({
                        f'{moderator_var}_水準': label,
                        f'{moderator_var}_值': value,
                        f'{focal_var}_斜率': slope,
                        '標準誤': se,
                        'p值': p,
                        '顯著': p < self.alpha
                    })
            except Exception as e:
                print(f"計算 {label} 的簡單斜率時發生錯誤: {e}")
        
        return pd.DataFrame(slopes)
    
    def create_interaction_plot(self,
                               data: pd.DataFrame,
                               outcome_var: str,
                               focal_var: str,
                               moderator_var: str,
                               focal_range: Optional[np.ndarray] = None,
                               moderator_levels: Optional[List] = None,
                               title: str = None) -> go.Figure:
        """
        建立交互作用圖
        
        Parameters:
        -----------
        data : DataFrame
            資料
        outcome_var : str
            結果變數
        focal_var : str
            焦點變數（X軸）
        moderator_var : str
            調節變數（不同線條）
        focal_range : array, optional
            焦點變數的範圍
        moderator_levels : list, optional
            調節變數的水準
        title : str, optional
            圖表標題
            
        Returns:
        --------
        Figure : Plotly圖表
        """
        if focal_range is None:
            focal_min = data[focal_var].min()
            focal_max = data[focal_var].max()
            focal_range = np.linspace(focal_min, focal_max, 100)
        
        if moderator_levels is None:
            # 使用 mean ± 1 SD
            mean = data[moderator_var].mean()
            std = data[moderator_var].std()
            moderator_levels = [
                (mean - std, '低(-1SD)'),
                (mean, '中(Mean)'),
                (mean + std, '高(+1SD)')
            ]
        
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, (mod_value, mod_label) in enumerate(moderator_levels):
            # 建立預測資料
            pred_data = pd.DataFrame({
                focal_var: focal_range,
                moderator_var: mod_value
            })
            
            # 添加交互作用項
            pred_data[f'{focal_var}_x_{moderator_var}'] = (
                pred_data[focal_var] * pred_data[moderator_var]
            )
            
            # 預測
            if self.interaction_model is not None:
                # 使用已擬合的模型
                X_pred = sm.add_constant(pred_data)
                
                # 確保欄位順序一致
                missing_cols = set(self.interaction_model.params.index) - set(X_pred.columns)
                for col in missing_cols:
                    X_pred[col] = 0
                
                X_pred = X_pred[self.interaction_model.params.index]
                
                pred_probs = self.interaction_model.predict(X_pred)
            else:
                # 簡化預測（線性）
                pred_probs = focal_range * 0.5  # 佔位符
            
            # 繪製線條
            fig.add_trace(go.Scatter(
                x=focal_range,
                y=pred_probs,
                mode='lines',
                name=f"{moderator_var} = {mod_label}",
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        if title is None:
            title = f"{focal_var} × {moderator_var} 交互作用圖"
        
        fig.update_layout(
            title=title,
            xaxis_title=focal_var,
            yaxis_title=f"P({outcome_var} = 1)",
            yaxis=dict(range=[0, 1]),
            font=dict(size=12),
            height=500,
            width=700,
            legend=dict(
                title=dict(text=moderator_var),
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        
        return fig
    
    def create_comparison_plot(self,
                              interaction_results: Dict,
                              title: str = "模型比較") -> go.Figure:
        """
        建立模型比較圖
        
        Parameters:
        -----------
        interaction_results : dict
            test_interaction的結果
        title : str
            圖表標題
            
        Returns:
        --------
        Figure : Plotly圖表
        """
        # 提取AIC
        models = ['基礎模型', '交互作用模型']
        aics = [
            interaction_results['aic_base'],
            interaction_results['aic_interaction']
        ]
        
        # 提取Log-Likelihood
        llfs = [
            interaction_results['base_model'].llf,
            interaction_results['interaction_model'].llf
        ]
        
        # 建立子圖
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('AIC (越小越好)', 'Log-Likelihood (越大越好)')
        )
        
        # AIC條形圖
        fig.add_trace(
            go.Bar(
                x=models,
                y=aics,
                text=[f"{aic:.1f}" for aic in aics],
                textposition='auto',
                marker_color=['lightblue', 'lightcoral'],
                name='AIC'
            ),
            row=1, col=1
        )
        
        # Log-Likelihood條形圖
        fig.add_trace(
            go.Bar(
                x=models,
                y=llfs,
                text=[f"{llf:.1f}" for llf in llfs],
                textposition='auto',
                marker_color=['lightblue', 'lightcoral'],
                name='Log-Likelihood'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text=title,
            showlegend=False,
            height=400,
            width=800,
            font=dict(size=12)
        )
        
        return fig
    
    def generate_report(self, interaction_results: Dict) -> str:
        """
        生成交互作用分析報告
        
        Parameters:
        -----------
        interaction_results : dict
            test_interaction的結果
            
        Returns:
        --------
        str : 報告文字
        """
        report = []
        report.append("=" * 60)
        report.append("交互作用分析報告")
        report.append("=" * 60)
        
        # Likelihood Ratio Test
        report.append("\n【Likelihood Ratio Test】")
        report.append(f"  LRT χ² = {interaction_results['lrt_statistic']:.3f}")
        report.append(f"  df = {interaction_results['lrt_dof']}")
        report.append(f"  p = {interaction_results['lrt_p_value']:.4f}")
        
        if interaction_results['lrt_significant']:
            report.append("  ✅ 交互作用模型顯著優於基礎模型 (p < 0.05)")
        else:
            report.append("  ❌ 交互作用模型未顯著改善 (p ≥ 0.05)")
        
        # AIC比較
        report.append("\n【模型比較】")
        report.append(f"  基礎模型 AIC: {interaction_results['aic_base']:.2f}")
        report.append(f"  交互作用模型 AIC: {interaction_results['aic_interaction']:.2f}")
        report.append(f"  AIC改善: {interaction_results['aic_improvement']:.2f}")
        
        if interaction_results['aic_improvement'] > 0:
            report.append("  ✅ 交互作用模型AIC較低（更佳）")
        else:
            report.append("  ⚠️  交互作用模型AIC較高（較差）")
        
        # 交互作用項
        report.append("\n【交互作用項係數】")
        coef_df = interaction_results['interaction_coefficients']
        
        for _, row in coef_df.iterrows():
            sig = "***" if row['p值'] < 0.001 else ("**" if row['p值'] < 0.01 else ("*" if row['p值'] < 0.05 else ""))
            report.append(f"\n  {row['交互作用項']} {sig}")
            report.append(f"    係數: {row['係數']:.4f}")
            report.append(f"    OR: {row['Odds_Ratio']:.3f}")
            report.append(f"    p = {row['p值']:.4f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def run_interaction_analysis(data: pd.DataFrame,
                            outcome_var: str,
                            main_effects: List[str],
                            interaction_terms: List[Tuple[str, str]],
                            focal_var: str,
                            moderator_var: str,
                            alpha: float = 0.05,
                            save_dir: Optional[str] = None) -> Dict:
    """
    執行完整的交互作用分析流程
    
    Parameters:
    -----------
    data : DataFrame
        資料
    outcome_var : str
        結果變數
    main_effects : list
        主效應變數
    interaction_terms : list of tuples
        交互作用項
    focal_var : str
        焦點變數（用於繪圖）
    moderator_var : str
        調節變數（用於繪圖）
    alpha : float
        顯著水準
    save_dir : str, optional
        儲存目錄
        
    Returns:
    --------
    dict : 分析結果
    """
    analyzer = InteractionAnalysis(alpha=alpha)
    
    # 測試交互作用
    print("\n測試交互作用...")
    results = analyzer.test_interaction(
        data, outcome_var, main_effects, interaction_terms
    )
    
    # Simple Slopes（如果交互作用顯著）
    simple_slopes_df = None
    if results['lrt_significant']:
        print("執行Simple Slopes分析...")
        simple_slopes_df = analyzer.simple_slopes(
            data, outcome_var, focal_var, moderator_var
        )
    
    # 視覺化
    print("生成視覺化...")
    fig_interaction = analyzer.create_interaction_plot(
        data, outcome_var, focal_var, moderator_var,
        title=f"{focal_var} × {moderator_var} 交互作用"
    )
    
    fig_comparison = analyzer.create_comparison_plot(
        results, title="基礎模型 vs 交互作用模型"
    )
    
    # 儲存
    if save_dir:
        from pathlib import Path
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 儲存表格
        results['interaction_coefficients'].to_csv(
            save_path / 'interaction_coefficients.csv',
            index=False,
            encoding='utf-8-sig'
        )
        
        if simple_slopes_df is not None:
            simple_slopes_df.to_csv(
                save_path / 'simple_slopes.csv',
                index=False,
                encoding='utf-8-sig'
            )
        
        # 儲存圖表
        fig_interaction.write_html(str(save_path / 'interaction_plot.html'))
        fig_comparison.write_html(str(save_path / 'model_comparison.html'))
        
        print(f"✅ 結果已儲存至: {save_dir}")
    
    # 輸出報告
    print(analyzer.generate_report(results))
    
    return {
        'interaction_results': results,
        'simple_slopes': simple_slopes_df,
        'figures': {
            'interaction_plot': fig_interaction,
            'comparison_plot': fig_comparison
        }
    }