"""
邏輯迴歸模組
===========

功能：
1. 多變量邏輯迴歸
2. 模型診斷（VIF、Hosmer-Lemeshow）
3. 係數與Odds Ratio視覺化
4. 預測評估
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class LogisticRegressionAnalysis:
    """邏輯迴歸分析類別"""
    
    def __init__(self, alpha: float = 0.05):
        """
        Parameters:
        -----------
        alpha : float
            顯著水準
        """
        self.alpha = alpha
        self.model = None
        self.results = None
    
    def fit(self,
            data: pd.DataFrame,
            outcome_var: str,
            predictor_vars: List[str],
            add_constant: bool = True) -> sm.GLM:
        """
        擬合邏輯迴歸模型
        
        Parameters:
        -----------
        data : DataFrame
            資料
        outcome_var : str
            二元結果變數
        predictor_vars : list
            預測變數列表
        add_constant : bool
            是否添加截距
            
        Returns:
        --------
        GLMResultsWrapper : 模型結果
        """
        # 準備資料
        y = data[outcome_var]
        X = data[predictor_vars].copy()
        
        # 處理類別變數（如果有）
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                # One-hot編碼（去除第一個類別避免共線性）
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
        
        # 添加常數項
        if add_constant:
            X = sm.add_constant(X)
        
        # 擬合模型
        self.model = sm.GLM(y, X, family=sm.families.Binomial())
        self.results = self.model.fit()
        
        return self.results
    
    def get_coefficients_table(self) -> pd.DataFrame:
        """
        獲取係數表（含Odds Ratio）
        
        Returns:
        --------
        DataFrame : 係數統計表
        """
        if self.results is None:
            raise ValueError("請先執行 fit() 方法")
        
        # 提取係數統計
        coef_df = pd.DataFrame({
            '變數': self.results.params.index,
            '係數': self.results.params.values,
            '標準誤': self.results.bse.values,
            'z值': self.results.tvalues.values,
            'p值': self.results.pvalues.values,
            'Odds_Ratio': np.exp(self.results.params.values)
        })
        
        # 信賴區間
        conf_int = self.results.conf_int(alpha=self.alpha)
        coef_df['CI_下界'] = conf_int[0].values
        coef_df['CI_上界'] = conf_int[1].values
        coef_df['OR_CI_下界'] = np.exp(conf_int[0].values)
        coef_df['OR_CI_上界'] = np.exp(conf_int[1].values)
        
        # 顯著性標記
        coef_df['顯著'] = coef_df['p值'] < self.alpha
        coef_df['星號'] = coef_df['p值'].apply(self._get_significance_stars)
        
        return coef_df
    
    def _get_significance_stars(self, p_value: float) -> str:
        """獲取顯著性星號"""
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return ''
    
    def calculate_vif(self, 
                     data: pd.DataFrame,
                     predictor_vars: List[str]) -> pd.DataFrame:
        """
        計算變異數膨脹因子（VIF）檢測共線性
        
        Parameters:
        -----------
        data : DataFrame
            資料
        predictor_vars : list
            預測變數
            
        Returns:
        --------
        DataFrame : VIF表
        
        判斷標準：
        - VIF < 5: 無共線性問題
        - 5 ≤ VIF < 10: 中等共線性
        - VIF ≥ 10: 嚴重共線性
        """
        X = data[predictor_vars].copy()
        
        # 處理類別變數
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
        
        # 計算VIF
        vif_data = pd.DataFrame()
        vif_data['變數'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X.values, i) 
                          for i in range(len(X.columns))]
        
        # 判斷
        vif_data['共線性'] = vif_data['VIF'].apply(
            lambda x: '嚴重' if x >= 10 else ('中等' if x >= 5 else '無')
        )
        
        return vif_data.sort_values('VIF', ascending=False)
    
    def hosmer_lemeshow_test(self, n_groups: int = 10) -> Dict:
        """
        Hosmer-Lemeshow擬合度檢定
        
        Parameters:
        -----------
        n_groups : int
            分組數（預設10）
            
        Returns:
        --------
        dict : 檢定結果
        
        解釋：
        - H0: 模型擬合良好
        - p > 0.05: 無法拒絕H0，模型擬合良好
        - p < 0.05: 拒絕H0，模型擬合不佳
        """
        if self.results is None:
            raise ValueError("請先執行 fit() 方法")
        
        # 預測機率
        pred_probs = self.results.fittedvalues
        actual = self.results.model.endog
        
        # 分組
        groups = pd.qcut(pred_probs, n_groups, duplicates='drop')
        
        # 計算觀察值與期望值
        obs_freq = pd.crosstab(groups, actual)
        exp_freq = pd.DataFrame({
            0: obs_freq.sum(axis=1) - pred_probs.groupby(groups).sum(),
            1: pred_probs.groupby(groups).sum()
        })
        
        # 卡方統計量
        chi2_stat = ((obs_freq - exp_freq) ** 2 / exp_freq).sum().sum()
        dof = n_groups - 2
        p_value = 1 - stats.chi2.cdf(chi2_stat, dof)
        
        return {
            'chi2': chi2_stat,
            'dof': dof,
            'p_value': p_value,
            'good_fit': p_value > self.alpha
        }
    
    def get_model_summary(self) -> Dict:
        """
        獲取模型摘要統計
        
        Returns:
        --------
        dict : 模型統計
        """
        if self.results is None:
            raise ValueError("請先執行 fit() 方法")
        
        # Pseudo R²
        null_model = sm.GLM(
            self.results.model.endog,
            np.ones(len(self.results.model.endog)),
            family=sm.families.Binomial()
        ).fit()
        
        mcfadden_r2 = 1 - (self.results.llf / null_model.llf)
        
        return {
            'log_likelihood': self.results.llf,
            'aic': self.results.aic,
            'bic': self.results.bic,
            'pseudo_r2_mcfadden': mcfadden_r2,
            'n_obs': int(self.results.nobs),
            'df_model': int(self.results.df_model),
            'df_resid': int(self.results.df_resid)
        }
    
    def create_coefficient_plot(self,
                               coef_df: pd.DataFrame,
                               title: str = "邏輯迴歸係數圖") -> go.Figure:
        """
        建立係數森林圖
        
        Parameters:
        -----------
        coef_df : DataFrame
            係數表
        title : str
            圖表標題
            
        Returns:
        --------
        Figure : Plotly圖表
        """
        # 排除截距
        plot_data = coef_df[coef_df['變數'] != 'const'].copy()
        
        # 排序
        plot_data = plot_data.sort_values('係數')
        
        # 顏色
        colors = ['red' if sig else 'gray' for sig in plot_data['顯著']]
        
        fig = go.Figure()
        
        # 係數點
        fig.add_trace(go.Scatter(
            x=plot_data['係數'],
            y=plot_data['變數'],
            mode='markers',
            marker=dict(size=10, color=colors),
            error_x=dict(
                type='data',
                symmetric=False,
                array=plot_data['CI_上界'] - plot_data['係數'],
                arrayminus=plot_data['係數'] - plot_data['CI_下界']
            ),
            name='係數',
            hovertemplate='<b>%{y}</b><br>係數: %{x:.3f}<br>p = %{customdata:.4f}<extra></extra>',
            customdata=plot_data['p值']
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="係數 (95% CI)",
            yaxis_title="變數",
            font=dict(size=12),
            height=max(400, len(plot_data) * 40),
            width=800,
            showlegend=False
        )
        
        # 參考線
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        
        return fig
    
    def create_odds_ratio_plot(self,
                              coef_df: pd.DataFrame,
                              title: str = "Odds Ratio圖") -> go.Figure:
        """
        建立Odds Ratio森林圖
        
        Parameters:
        -----------
        coef_df : DataFrame
            係數表
        title : str
            圖表標題
            
        Returns:
        --------
        Figure : Plotly圖表
        """
        # 排除截距
        plot_data = coef_df[coef_df['變數'] != 'const'].copy()
        
        # 排序
        plot_data = plot_data.sort_values('Odds_Ratio')
        
        # 顏色
        colors = ['red' if sig else 'gray' for sig in plot_data['顯著']]
        
        fig = go.Figure()
        
        # OR點
        fig.add_trace(go.Scatter(
            x=plot_data['Odds_Ratio'],
            y=plot_data['變數'],
            mode='markers',
            marker=dict(size=10, color=colors),
            error_x=dict(
                type='data',
                symmetric=False,
                array=plot_data['OR_CI_上界'] - plot_data['Odds_Ratio'],
                arrayminus=plot_data['Odds_Ratio'] - plot_data['OR_CI_下界']
            ),
            name='OR',
            hovertemplate='<b>%{y}</b><br>OR: %{x:.3f}<br>p = %{customdata:.4f}<extra></extra>',
            customdata=plot_data['p值']
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Odds Ratio (95% CI)",
            yaxis_title="變數",
            font=dict(size=12),
            height=max(400, len(plot_data) * 40),
            width=800,
            showlegend=False,
            xaxis_type="log"  # 對數尺度
        )
        
        # 參考線
        fig.add_vline(x=1, line_dash="dash", line_color="red",
                     annotation_text="OR = 1 (無效應)")
        
        return fig
    
    def generate_report(self, 
                       coef_df: pd.DataFrame,
                       vif_df: pd.DataFrame,
                       hl_test: Dict,
                       model_summary: Dict) -> str:
        """
        生成文字報告
        
        Returns:
        --------
        str : 報告文字
        """
        report = []
        report.append("=" * 60)
        report.append("邏輯迴歸分析報告")
        report.append("=" * 60)
        
        # 模型摘要
        report.append("\n【模型摘要】")
        report.append(f"  樣本數: {model_summary['n_obs']:,}")
        report.append(f"  Log-Likelihood: {model_summary['log_likelihood']:.2f}")
        report.append(f"  AIC: {model_summary['aic']:.2f}")
        report.append(f"  BIC: {model_summary['bic']:.2f}")
        report.append(f"  Pseudo R² (McFadden): {model_summary['pseudo_r2_mcfadden']:.4f}")
        
        # 共線性診斷
        report.append("\n【共線性診斷】")
        severe_vif = vif_df[vif_df['共線性'] == '嚴重']
        if len(severe_vif) > 0:
            report.append("  ⚠️  檢測到嚴重共線性:")
            for _, row in severe_vif.iterrows():
                report.append(f"    - {row['變數']}: VIF = {row['VIF']:.2f}")
        else:
            report.append("  ✅ 無嚴重共線性問題")
        
        # Hosmer-Lemeshow檢定
        report.append("\n【模型擬合度】")
        p_str = "< .001" if hl_test['p_value'] < 0.001 else f"= {hl_test['p_value']:.3f}"
        report.append(f"  Hosmer-Lemeshow χ²({hl_test['dof']}) = {hl_test['chi2']:.3f}, p {p_str}")
        if hl_test['good_fit']:
            report.append("  ✅ 模型擬合良好 (p > 0.05)")
        else:
            report.append("  ⚠️  模型擬合可能不佳 (p < 0.05)")
        
        # 顯著係數
        report.append("\n【顯著預測變數】")
        sig_coefs = coef_df[(coef_df['顯著']) & (coef_df['變數'] != 'const')]
        
        if len(sig_coefs) == 0:
            report.append("  無顯著預測變數")
        else:
            for _, row in sig_coefs.iterrows():
                direction = "增加" if row['係數'] > 0 else "減少"
                report.append(f"  • {row['變數']}")
                report.append(f"    係數: {row['係數']:.4f} ({row['星號']})")
                report.append(f"    OR: {row['Odds_Ratio']:.3f} [95% CI: {row['OR_CI_下界']:.3f}, {row['OR_CI_上界']:.3f}]")
                report.append(f"    效應: {direction}守法選擇機率")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def run_logistic_regression(data: pd.DataFrame,
                           outcome_var: str,
                           predictor_vars: List[str],
                           alpha: float = 0.05,
                           save_dir: Optional[str] = None) -> Dict:
    """
    執行完整的邏輯迴歸分析流程
    
    Parameters:
    -----------
    data : DataFrame
        資料
    outcome_var : str
        結果變數
    predictor_vars : list
        預測變數
    alpha : float
        顯著水準
    save_dir : str, optional
        儲存目錄
        
    Returns:
    --------
    dict : 分析結果
    """
    analyzer = LogisticRegressionAnalysis(alpha=alpha)
    
    # 擬合模型
    print("\n擬合邏輯迴歸模型...")
    results = analyzer.fit(data, outcome_var, predictor_vars)
    
    # 係數表
    print("提取係數表...")
    coef_df = analyzer.get_coefficients_table()
    
    # VIF診斷
    print("計算VIF...")
    vif_df = analyzer.calculate_vif(data, predictor_vars)
    
    # Hosmer-Lemeshow檢定
    print("執行Hosmer-Lemeshow檢定...")
    hl_test = analyzer.hosmer_lemeshow_test()
    
    # 模型摘要
    model_summary = analyzer.get_model_summary()
    
    # 視覺化
    print("生成視覺化...")
    fig_coef = analyzer.create_coefficient_plot(coef_df)
    fig_or = analyzer.create_odds_ratio_plot(coef_df)
    
    # 儲存
    if save_dir:
        from pathlib import Path
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 儲存表格
        coef_df.to_csv(
            save_path / 'logistic_regression_coefficients.csv',
            index=False,
            encoding='utf-8-sig'
        )
        vif_df.to_csv(
            save_path / 'vif_diagnostics.csv',
            index=False,
            encoding='utf-8-sig'
        )
        
        # 儲存圖表
        fig_coef.write_html(str(save_path / 'coefficient_plot.html'))
        fig_or.write_html(str(save_path / 'odds_ratio_plot.html'))
        
        print(f"✅ 結果已儲存至: {save_dir}")
    
    # 輸出報告
    print(analyzer.generate_report(coef_df, vif_df, hl_test, model_summary))
    
    return {
        'model_results': results,
        'coefficients': coef_df,
        'vif': vif_df,
        'hosmer_lemeshow': hl_test,
        'model_summary': model_summary,
        'figures': {
            'coefficient_plot': fig_coef,
            'odds_ratio_plot': fig_or
        }
    }