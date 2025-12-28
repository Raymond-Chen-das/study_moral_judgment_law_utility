"""
第5章視覺化模組
================
包含：
- 5.1 HLM隨機效應相關圖
- 5.2 XGBoost ROC曲線、SHAP圖、混淆矩陣
- 5.3 全球道德光譜散點圖
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 5.1 HLM 隨機效應視覺化
# ============================================================

def plot_random_effect_correlations(
    correlation_df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "HLM 隨機效應與 AMCE 維度相關性"
) -> go.Figure:
    """
    繪製隨機效應與 AMCE 維度的相關係數條形圖
    
    Parameters
    ----------
    correlation_df : pd.DataFrame
        來自 RandomEffectExplorer.compute_correlations() 的結果
    output_path : str, optional
        輸出路徑
    title : str
        圖表標題
        
    Returns
    -------
    go.Figure
        Plotly 圖表物件
    """
    # 排序（按絕對值）
    df = correlation_df.sort_values('pearson_r', key=abs, ascending=True).copy()
    
    # 設定顏色（正相關藍色，負相關紅色）
    colors = ['#2E86AB' if r > 0 else '#E94F37' for r in df['pearson_r']]
    
    # 顯著性標記
    sig_markers = []
    for _, row in df.iterrows():
        if row['significant_001']:
            sig_markers.append('***')
        elif row['significant_01']:
            sig_markers.append('**')
        elif row['significant_05']:
            sig_markers.append('*')
        else:
            sig_markers.append('')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df['chinese_name'],
        x=df['pearson_r'],
        orientation='h',
        marker_color=colors,
        text=[f"{r:.3f}{s}" for r, s in zip(df['pearson_r'], sig_markers)],
        textposition='outside',
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Pearson r: %{x:.3f}<br>"
            "p-value: %{customdata:.4f}<br>"
            "<extra></extra>"
        ),
        customdata=df['pearson_p']
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Pearson 相關係數",
        yaxis_title="AMCE 維度",
        height=500,
        width=800,
        showlegend=False,
        xaxis=dict(zeroline=True, zerolinecolor='gray', zerolinewidth=1),
        margin=dict(l=120, r=80)
    )
    
    # 添加註解
    fig.add_annotation(
        text="註：* p<.05, ** p<.01, *** p<.001",
        xref="paper", yref="paper",
        x=1, y=-0.12,
        showarrow=False,
        font=dict(size=10, color="gray")
    )
    
    if output_path:
        fig.write_html(output_path)
        print(f"圖表已儲存: {output_path}")
    
    return fig


def plot_random_effect_scatter(
    scatter_data: pd.DataFrame,
    amce_col: str = 'law_preference',
    amce_label: str = '守法偏好 AMCE',
    output_path: Optional[str] = None,
    highlight_countries: Optional[List[str]] = None
) -> go.Figure:
    """
    繪製隨機效應 vs AMCE 維度散點圖
    
    Parameters
    ----------
    scatter_data : pd.DataFrame
        包含 UserCountry3, random_intercept, amce_col 的資料
    amce_col : str
        AMCE 維度欄位名
    amce_label : str
        AMCE 維度標籤
    output_path : str, optional
        輸出路徑
    highlight_countries : List[str], optional
        要標註的國家代碼
        
    Returns
    -------
    go.Figure
        Plotly 圖表物件
    """
    if highlight_countries is None:
        highlight_countries = ['TWN', 'JPN', 'KOR', 'CHN', 'USA', 'DEU']
    
    df = scatter_data.copy()
    df['is_highlight'] = df['UserCountry3'].isin(highlight_countries)
    
    # 計算相關係數
    from scipy import stats
    r, p = stats.pearsonr(df['random_intercept'], df[amce_col])
    
    fig = go.Figure()
    
    # 一般國家
    fig.add_trace(go.Scatter(
        x=df[~df['is_highlight']][amce_col],
        y=df[~df['is_highlight']]['random_intercept'],
        mode='markers',
        marker=dict(size=8, color='#CCCCCC', opacity=0.6),
        name='其他國家',
        hovertemplate="<b>%{text}</b><br>AMCE: %{x:.3f}<br>隨機效應: %{y:.3f}<extra></extra>",
        text=df[~df['is_highlight']]['UserCountry3']
    ))
    
    # 標註國家
    fig.add_trace(go.Scatter(
        x=df[df['is_highlight']][amce_col],
        y=df[df['is_highlight']]['random_intercept'],
        mode='markers+text',
        marker=dict(size=12, color='#E94F37', symbol='diamond'),
        text=df[df['is_highlight']]['UserCountry3'],
        textposition='top center',
        name='標註國家',
        hovertemplate="<b>%{text}</b><br>AMCE: %{x:.3f}<br>隨機效應: %{y:.3f}<extra></extra>"
    ))
    
    # 添加趨勢線
    z = np.polyfit(df[amce_col], df['random_intercept'], 1)
    x_line = np.linspace(df[amce_col].min(), df[amce_col].max(), 100)
    y_line = z[0] * x_line + z[1]
    
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode='lines',
        line=dict(color='#2E86AB', dash='dash', width=2),
        name=f'趨勢線 (r={r:.3f})'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"HLM 隨機效應 vs {amce_label}<br><sup>Pearson r = {r:.3f}, p {'< .001' if p < 0.001 else f'= {p:.3f}'}</sup>",
            font=dict(size=14)
        ),
        xaxis_title=amce_label,
        yaxis_title="HLM 隨機效應（國家對全球平均的偏離）",
        height=500,
        width=700,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    if output_path:
        fig.write_html(output_path)
        print(f"圖表已儲存: {output_path}")
    
    return fig


# ============================================================
# 5.2 XGBoost 視覺化
# ============================================================

def plot_roc_curve(
    metrics: Dict[str, Any],
    output_path: Optional[str] = None,
    title: str = "ROC 曲線"
) -> go.Figure:
    """
    繪製 ROC 曲線
    
    Parameters
    ----------
    metrics : Dict[str, Any]
        來自 XGBoostClassifier.evaluate() 的結果
    output_path : str, optional
        輸出路徑
    title : str
        圖表標題
        
    Returns
    -------
    go.Figure
        Plotly 圖表物件
    """
    roc_data = metrics['roc_curve']
    auc = metrics['roc_auc']
    
    fig = go.Figure()
    
    # ROC 曲線
    fig.add_trace(go.Scatter(
        x=roc_data['fpr'],
        y=roc_data['tpr'],
        mode='lines',
        name=f'XGBoost (AUC = {auc:.4f})',
        line=dict(color='#2E86AB', width=2),
        fill='tozeroy',
        fillcolor='rgba(46, 134, 171, 0.2)'
    ))
    
    # 隨機猜測基準線
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='隨機猜測 (AUC = 0.5)',
        line=dict(color='gray', dash='dash', width=1)
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Sensitivity)",
        height=500,
        width=550,
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
        xaxis=dict(range=[0, 1], constrain='domain'),
        yaxis=dict(range=[0, 1], scaleanchor="x", scaleratio=1)
    )
    
    if output_path:
        fig.write_html(output_path)
        print(f"圖表已儲存: {output_path}")
    
    return fig


def plot_confusion_matrix(
    metrics: Dict[str, Any],
    output_path: Optional[str] = None,
    title: str = "混淆矩陣"
) -> go.Figure:
    """
    繪製混淆矩陣熱圖
    
    Parameters
    ----------
    metrics : Dict[str, Any]
        來自 XGBoostClassifier.evaluate() 的結果
    output_path : str, optional
        輸出路徑
    title : str
        圖表標題
        
    Returns
    -------
    go.Figure
        Plotly 圖表物件
    """
    cm = np.array(metrics['confusion_matrix'])
    
    # 計算百分比
    cm_pct = cm / cm.sum() * 100
    
    # 標籤
    labels = ['效益主義 (0)', '守法 (1)']
    
    # 創建文字標籤
    text_matrix = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append(f"{cm[i,j]:,}<br>({cm_pct[i,j]:.1f}%)")
        text_matrix.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=14),
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title="數量")
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="預測類別",
        yaxis_title="實際類別",
        height=450,
        width=500,
        yaxis=dict(autorange='reversed')
    )
    
    if output_path:
        fig.write_html(output_path)
        print(f"圖表已儲存: {output_path}")
    
    return fig


def plot_shap_importance(
    importance_df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "SHAP 特徵重要性",
    top_n: int = 10
) -> go.Figure:
    """
    繪製 SHAP 特徵重要性條形圖
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        來自 SHAPAnalyzer.get_feature_importance() 的結果
    output_path : str, optional
        輸出路徑
    title : str
        圖表標題
    top_n : int
        顯示前 N 個特徵
        
    Returns
    -------
    go.Figure
        Plotly 圖表物件
    """
    df = importance_df.head(top_n).sort_values('shap_importance', ascending=True)
    
    # 顏色根據方向
    colors = ['#2E86AB' if d == '↑守法' else '#E94F37' for d in df['direction']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df['feature'],
        x=df['shap_importance'],
        orientation='h',
        marker_color=colors,
        text=[f"{v:.4f}" for v in df['shap_importance']],
        textposition='outside',
        hovertemplate=(
            "<b>%{y}</b><br>"
            "SHAP 重要性: %{x:.4f}<br>"
            "影響方向: %{customdata}<br>"
            "<extra></extra>"
        ),
        customdata=df['direction']
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="平均 |SHAP 值|",
        yaxis_title="特徵",
        height=400 + top_n * 25,
        width=700,
        showlegend=False,
        margin=dict(l=150, r=80)
    )
    
    # 添加圖例說明
    fig.add_annotation(
        text="<span style='color:#2E86AB'>■</span> 正向影響守法 | <span style='color:#E94F37'>■</span> 正向影響效益主義",
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(size=11)
    )
    
    if output_path:
        fig.write_html(output_path)
        print(f"圖表已儲存: {output_path}")
    
    return fig


def plot_shap_vs_chapter4(
    comparison_df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "SHAP 重要性 vs 第4章效果量比較"
) -> go.Figure:
    """
    繪製 SHAP 與第4章效果量的比較圖
    
    Parameters
    ----------
    comparison_df : pd.DataFrame
        來自 SHAPAnalyzer.compare_with_chapter4() 的結果
    output_path : str, optional
        輸出路徑
    title : str
        圖表標題
        
    Returns
    -------
    go.Figure
        Plotly 圖表物件
    """
    df = comparison_df.dropna(subset=['chapter4_effect']).copy()
    
    # 只保留數值型效果量
    df = df[df['chapter4_effect'].apply(lambda x: isinstance(x, (int, float)))]
    
    if len(df) == 0:
        print("警告：沒有可比較的數值型效果量")
        return None
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("SHAP 重要性排序", "第4章 Odds Ratio"),
        horizontal_spacing=0.15
    )
    
    # SHAP 排序
    df_sorted = df.sort_values('shap_rank')
    
    fig.add_trace(
        go.Bar(
            y=df_sorted['feature'],
            x=df_sorted['shap_importance'],
            orientation='h',
            marker_color='#2E86AB',
            name='SHAP'
        ),
        row=1, col=1
    )
    
    # OR 值（以 1 為基準）
    fig.add_trace(
        go.Bar(
            y=df_sorted['feature'],
            x=df_sorted['chapter4_effect'] - 1,  # 偏離 1 的程度
            orientation='h',
            marker_color='#E94F37',
            name='OR - 1'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        height=400,
        width=900,
        showlegend=False
    )
    
    fig.update_xaxes(title_text="SHAP 值", row=1, col=1)
    fig.update_xaxes(title_text="OR - 1（偏離基準）", row=1, col=2)
    
    if output_path:
        fig.write_html(output_path)
        print(f"圖表已儲存: {output_path}")
    
    return fig


# ============================================================
# 5.3 全球道德光譜視覺化
# ============================================================

def plot_moral_spectrum(
    spectrum_data: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "全球道德光譜：效益主義 vs 守法偏好"
) -> go.Figure:
    """
    繪製全球道德光譜散點圖
    
    Parameters
    ----------
    spectrum_data : pd.DataFrame
        來自 MoralSpectrumAnalyzer.get_country_positions() 的結果
    output_path : str, optional
        輸出路徑
    title : str
        圖表標題
        
    Returns
    -------
    go.Figure
        Plotly 圖表物件
    """
    df = spectrum_data.copy()
    
    # 文化圈顏色
    cluster_colors = {
        'Western': '#2E86AB',
        'Eastern': '#E94F37',
        'Southern': '#A23B72'
    }
    
    fig = go.Figure()
    
    # 計算全球平均（用於畫象限線）
    util_mean = df['utilitarian'].mean()
    law_mean = df['law_preference'].mean()
    
    # 添加象限線
    fig.add_hline(y=law_mean, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=util_mean, line_dash="dash", line_color="gray", opacity=0.5)
    
    # 按文化圈分組繪製
    for cluster_name, color in cluster_colors.items():
        cluster_data = df[df['Cluster_Name'] == cluster_name]
        
        # 非標註國家
        non_highlight = cluster_data[~cluster_data['is_highlight']]
        fig.add_trace(go.Scatter(
            x=non_highlight['utilitarian'],
            y=non_highlight['law_preference'],
            mode='markers',
            marker=dict(size=8, color=color, opacity=0.6),
            name=cluster_name,
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "效益主義: %{x:.3f}<br>"
                "守法偏好: %{y:.3f}<br>"
                "<extra></extra>"
            ),
            customdata=non_highlight[['Country', 'UserCountry3']].values
        ))
    
    # 標註重點國家（所有文化圈）
    highlight_data = df[df['is_highlight']]
    for cluster_name, color in cluster_colors.items():
        cluster_highlight = highlight_data[highlight_data['Cluster_Name'] == cluster_name]
        if len(cluster_highlight) > 0:
            fig.add_trace(go.Scatter(
                x=cluster_highlight['utilitarian'],
                y=cluster_highlight['law_preference'],
                mode='markers+text',
                marker=dict(size=14, color=color, symbol='diamond', 
                           line=dict(color='white', width=1)),
                text=cluster_highlight['label'],
                textposition='top center',
                textfont=dict(size=10, color='black'),
                name=f'{cluster_name} (標註)',
                showlegend=False,
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                    "效益主義: %{x:.3f}<br>"
                    "守法偏好: %{y:.3f}<br>"
                    "<extra></extra>"
                ),
                customdata=cluster_highlight[['Country', 'UserCountry3']].values
            ))
    
    # 添加象限標籤
    x_range = df['utilitarian'].max() - df['utilitarian'].min()
    y_range = df['law_preference'].max() - df['law_preference'].min()
    
    quadrant_labels = [
        (util_mean + x_range * 0.2, law_mean + y_range * 0.15, "Q1: 高效益+高守法"),
        (util_mean - x_range * 0.2, law_mean + y_range * 0.15, "Q2: 低效益+高守法"),
        (util_mean - x_range * 0.2, law_mean - y_range * 0.15, "Q3: 低效益+低守法"),
        (util_mean + x_range * 0.2, law_mean - y_range * 0.15, "Q4: 高效益+低守法"),
    ]
    
    for x, y, text in quadrant_labels:
        fig.add_annotation(
            x=x, y=y,
            text=text,
            showarrow=False,
            font=dict(size=9, color='gray'),
            opacity=0.7
        )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="效益主義 AMCE（數量偏好）",
        yaxis_title="守法偏好 AMCE",
        height=600,
        width=800,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        hovermode='closest'
    )
    
    if output_path:
        fig.write_html(output_path)
        print(f"圖表已儲存: {output_path}")
    
    return fig


def plot_cluster_centroids(
    spectrum_data: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "文化圈重心比較"
) -> go.Figure:
    """
    繪製文化圈重心比較圖
    
    Parameters
    ----------
    spectrum_data : pd.DataFrame
        來自 MoralSpectrumAnalyzer.get_country_positions() 的結果
    output_path : str, optional
        輸出路徑
    title : str
        圖表標題
        
    Returns
    -------
    go.Figure
        Plotly 圖表物件
    """
    # 計算重心
    centroids = spectrum_data.groupby('Cluster_Name').agg({
        'utilitarian': ['mean', 'std'],
        'law_preference': ['mean', 'std'],
        'UserCountry3': 'count'
    }).reset_index()
    centroids.columns = ['cluster', 'util_mean', 'util_std', 'law_mean', 'law_std', 'n']
    
    cluster_colors = {
        'Western': '#2E86AB',
        'Eastern': '#E94F37',
        'Southern': '#A23B72'
    }
    
    fig = go.Figure()
    
    for _, row in centroids.iterrows():
        color = cluster_colors.get(row['cluster'], 'gray')
        
        # 繪製誤差橢圓（近似）
        fig.add_trace(go.Scatter(
            x=[row['util_mean']],
            y=[row['law_mean']],
            mode='markers+text',
            marker=dict(size=20, color=color, symbol='circle'),
            text=[row['cluster']],
            textposition='top center',
            name=f"{row['cluster']} (n={row['n']})",
            error_x=dict(type='data', array=[row['util_std']], visible=True, color=color),
            error_y=dict(type='data', array=[row['law_std']], visible=True, color=color),
            hovertemplate=(
                f"<b>{row['cluster']}</b><br>"
                f"效益主義: {row['util_mean']:.3f} ± {row['util_std']:.3f}<br>"
                f"守法偏好: {row['law_mean']:.3f} ± {row['law_std']:.3f}<br>"
                f"國家數: {row['n']}<br>"
                "<extra></extra>"
            )
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="效益主義 AMCE（平均值）",
        yaxis_title="守法偏好 AMCE（平均值）",
        height=500,
        width=600,
        showlegend=True
    )
    
    if output_path:
        fig.write_html(output_path)
        print(f"圖表已儲存: {output_path}")
    
    return fig


# ============================================================
# 輔助函數
# ============================================================

def create_performance_table_html(
    metrics: Dict[str, Any],
    output_path: Optional[str] = None
) -> str:
    """
    創建性能指標 HTML 表格
    
    Parameters
    ----------
    metrics : Dict[str, Any]
        評估指標字典
    output_path : str, optional
        輸出路徑
        
    Returns
    -------
    str
        HTML 表格字串
    """
    html = """
    <table style="border-collapse: collapse; width: 100%; max-width: 500px;">
        <thead>
            <tr style="background-color: #2E86AB; color: white;">
                <th style="padding: 10px; text-align: left;">指標</th>
                <th style="padding: 10px; text-align: right;">數值</th>
            </tr>
        </thead>
        <tbody>
    """
    
    metrics_display = [
        ('Accuracy', metrics['accuracy']),
        ('Precision', metrics['precision']),
        ('Recall', metrics['recall']),
        ('F1 Score', metrics['f1']),
        ('ROC-AUC', metrics['roc_auc']),
    ]
    
    for i, (name, value) in enumerate(metrics_display):
        bg_color = '#f8f9fa' if i % 2 == 0 else 'white'
        html += f"""
            <tr style="background-color: {bg_color};">
                <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{name}</td>
                <td style="padding: 8px; border-bottom: 1px solid #dee2e6; text-align: right;">{value:.4f}</td>
            </tr>
        """
    
    html += """
        </tbody>
    </table>
    """
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"表格已儲存: {output_path}")
    
    return html


if __name__ == "__main__":
    print("第5章視覺化模組")
    print("請導入此模組並使用各繪圖函數")