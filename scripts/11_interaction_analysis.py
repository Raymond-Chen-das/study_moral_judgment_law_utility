"""
11_interaction_analysis.py
==============================================
第四章:統計推論 - 交互作用分析

功能：
1. Cluster × Review_political 交互作用
2. Review_age × Review_religious 交互作用
3. Likelihood Ratio Test
4. Simple Slopes分析
5. VIF共線性檢查

執行方式：
    python scripts/11_interaction_analysis.py
"""

import sys
from pathlib import Path

# 將專案根目錄加入路徑
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.inference.interaction_analysis import (
    InteractionAnalysis,
    run_interaction_analysis
)
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from statsmodels.stats.outliers_influence import variance_inflation_factor


def setup_logging(log_dir: str = 'outputs/logs') -> logging.Logger:
    """設定日誌"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    log_file = log_path / 'interaction_analysis.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_and_prepare_data(data_path: str = 'data/processed/featured_data.csv') -> pd.DataFrame:
    """
    載入並準備分析資料
    
    Returns:
    --------
    DataFrame : 過濾後的資料
    """
    print("\n" + "="*60)
    print("載入資料...")
    print("="*60)
    
    df = pd.read_csv(data_path)
    print(f"✅ 原始資料: {len(df):,} 行")
    
    # 過濾 Cluster == -1
    df_filtered = df[df['Cluster'] != -1].copy()
    removed = len(df) - len(df_filtered)
    print(f"✅ 過濾後資料: {len(df_filtered):,} 行")
    print(f"   (移除 {removed:,} 行 Cluster==-1)")
    
    # 移除缺失值
    required_cols = ['chose_lawful', 'Cluster', 'Review_age', 
                    'Review_political', 'Review_religious']
    
    df_clean = df_filtered[required_cols].dropna()
    na_removed = len(df_filtered) - len(df_clean)
    
    print(f"✅ 移除缺失值後: {len(df_clean):,} 行")
    if na_removed > 0:
        print(f"   (移除 {na_removed:,} 行缺失值, {na_removed/len(df_filtered)*100:.2f}%)")
    
    return df_clean


def prepare_interaction_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    準備交互作用分析資料
    
    Parameters:
    -----------
    df : DataFrame
        資料
        
    Returns:
    --------
    DataFrame : 準備好的資料
    """
    print("\n" + "="*60)
    print("準備交互作用變數...")
    print("="*60)
    
    df_model = df.copy()
    
    # 建立Cluster dummy變數
    df_model['Cluster_Eastern'] = (df['Cluster'] == 1).astype(int)
    df_model['Cluster_Southern'] = (df['Cluster'] == 2).astype(int)
    
    # 中心化連續變數（便於解釋交互作用）
    df_model['political_centered'] = df_model['Review_political'] - df_model['Review_political'].mean()
    df_model['age_centered'] = df_model['Review_age'] - df_model['Review_age'].mean()
    df_model['religious_centered'] = df_model['Review_religious'] - df_model['Review_religious'].mean()
    
    print("✅ Cluster dummy變數已建立")
    print("✅ 連續變數已中心化")
    
    # 檢查中心化結果
    print(f"\n中心化後的變數均值（應接近0）:")
    print(f"  political_centered: {df_model['political_centered'].mean():.6f}")
    print(f"  age_centered: {df_model['age_centered'].mean():.6f}")
    print(f"  religious_centered: {df_model['religious_centered'].mean():.6f}")
    
    return df_model

def calculate_vif(df: pd.DataFrame,
                  output_dir: str = 'outputs/tables/chapter4') -> pd.DataFrame:
    """
    計算VIF檢查共線性問題
    
    Parameters:
    -----------
    df : DataFrame
        包含所有變數的資料
    output_dir : str
        輸出目錄
        
    Returns:
    --------
    DataFrame : VIF結果
    """
    print("\n" + "="*60)
    print("VIF共線性檢查")
    print("="*60)
    
    # 建立交互作用項
    df_vif = df.copy()
    df_vif['Cluster_Eastern_x_political_centered'] = (
        df_vif['Cluster_Eastern'] * df_vif['political_centered']
    )
    df_vif['Cluster_Southern_x_political_centered'] = (
        df_vif['Cluster_Southern'] * df_vif['political_centered']
    )
    df_vif['age_centered_x_religious_centered'] = (
        df_vif['age_centered'] * df_vif['religious_centered']
    )
    
    # VIF檢查的變數清單
    vif_vars = [
        'Cluster_Eastern',
        'Cluster_Southern',
        'political_centered',
        'age_centered',
        'religious_centered',
        'Cluster_Eastern_x_political_centered',
        'Cluster_Southern_x_political_centered',
        'age_centered_x_religious_centered'
    ]
    
    # 計算VIF
    print("計算VIF...")
    vif_results = []
    
    for i, var in enumerate(vif_vars):
        try:
            vif_value = variance_inflation_factor(df_vif[vif_vars].values, i)
            vif_results.append({
                '變數': var,
                'VIF': vif_value,
                '判定': '✓ 良好' if vif_value < 5 else ('⚠️ 中等' if vif_value < 10 else '❌ 嚴重')
            })
        except Exception as e:
            print(f"⚠️ 計算 {var} 的VIF時發生錯誤: {e}")
            vif_results.append({
                '變數': var,
                'VIF': np.nan,
                '判定': '❌ 無法計算'
            })
    
    vif_df = pd.DataFrame(vif_results)
    
    # 顯示結果
    print("\nVIF結果:")
    print(vif_df.to_string(index=False))
    
    # 判定整體共線性狀況
    max_vif = vif_df['VIF'].max()
    print(f"\n最大VIF: {max_vif:.2f}")
    
    if max_vif < 5:
        print("✅ 所有VIF < 5，共線性問題不嚴重")
    elif max_vif < 10:
        print("⚠️ 部分VIF介於5-10，共線性問題中等")
    else:
        print("❌ 部分VIF > 10，共線性問題嚴重，建議調整模型")
    
    # 儲存結果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    vif_df.to_csv(
        output_path / 'interaction_vif.csv',
        index=False,
        encoding='utf-8-sig'
    )
    print(f"\n✅ VIF結果: {output_path / 'interaction_vif.csv'}")
    
    return vif_df


def analyze_cluster_political_interaction(df: pd.DataFrame,
                                         output_dir: str = 'outputs/tables/chapter4',
                                         figure_dir: str = 'outputs/figures/chapter4_inference') -> dict:
    """
    分析 Cluster × Review_political 交互作用
    
    Parameters:
    -----------
    df : DataFrame
        資料
    output_dir : str
        表格輸出目錄
    figure_dir : str
        圖表輸出目錄
        
    Returns:
    --------
    dict : 分析結果
    """
    print("\n" + "="*60)
    print("交互作用分析: Cluster × Review_political")
    print("="*60)
    print("理論意義: 測試文化圈是否調節政治傾向對道德選擇的影響")
    
    # 主效應變數
    main_effects = [
        'Cluster_Eastern',
        'Cluster_Southern',
        'political_centered',
        'age_centered',
        'religious_centered'
    ]
    
    # 交互作用項
    interaction_terms = [
        ('Cluster_Eastern', 'political_centered'),
        ('Cluster_Southern', 'political_centered')
    ]
    
    # 執行分析
    results = run_interaction_analysis(
        data=df,
        outcome_var='chose_lawful',
        main_effects=main_effects,
        interaction_terms=interaction_terms,
        focal_var='political_centered',
        moderator_var='Cluster',  # 注意：這裡用原始Cluster做圖
        alpha=0.05,
        save_dir=None
    )
    
    # 儲存結果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig_path = Path(figure_dir)
    fig_path.mkdir(parents=True, exist_ok=True)
    
    # 交互作用係數
    interaction_coef = results['interaction_results']['interaction_coefficients']
    interaction_coef.to_csv(
        output_path / 'interaction_cluster_political_coefficients.csv',
        index=False,
        encoding='utf-8-sig'
    )
    print(f"\n✅ 交互作用係數: {output_path / 'interaction_cluster_political_coefficients.csv'}")
    
    # LRT結果
    lrt_results = results['interaction_results']
    lrt_df = pd.DataFrame([{
        '比較': 'Base vs Interaction Model',
        'LRT_χ²': lrt_results['lrt_statistic'],
        'df': lrt_results['lrt_dof'],
        'p值': lrt_results['lrt_p_value'],
        'AIC_base': lrt_results['aic_base'],
        'AIC_interaction': lrt_results['aic_interaction'],
        'AIC_improvement': lrt_results['aic_improvement'],
        '顯著': '✓' if lrt_results['lrt_significant'] else ''
    }])
    
    lrt_df.to_csv(
        output_path / 'interaction_cluster_political_lrt.csv',
        index=False,
        encoding='utf-8-sig'
    )
    print(f"✅ LRT結果: {output_path / 'interaction_cluster_political_lrt.csv'}")
    
    # Simple Slopes（如果交互作用顯著）
    if results['simple_slopes'] is not None:
        results['simple_slopes'].to_csv(
            output_path / 'interaction_cluster_political_simple_slopes.csv',
            index=False,
            encoding='utf-8-sig'
        )
        print(f"✅ Simple Slopes: {output_path / 'interaction_cluster_political_simple_slopes.csv'}")
    
    # 圖表
    results['figures']['interaction_plot'].write_html(
        str(fig_path / 'interaction_cluster_political_plot.html')
    )
    print(f"✅ 交互作用圖: {fig_path / 'interaction_cluster_political_plot.html'}")
    
    results['figures']['comparison_plot'].write_html(
        str(fig_path / 'interaction_cluster_political_comparison.html')
    )
    print(f"✅ 模型比較圖: {fig_path / 'interaction_cluster_political_comparison.html'}")
    
    return results


def analyze_age_religious_interaction(df: pd.DataFrame,
                                     output_dir: str = 'outputs/tables/chapter4',
                                     figure_dir: str = 'outputs/figures/chapter4_inference') -> dict:
    """
    分析 Review_age × Review_religious 交互作用
    
    Parameters:
    -----------
    df : DataFrame
        資料
    output_dir : str
        表格輸出目錄
    figure_dir : str
        圖表輸出目錄
        
    Returns:
    --------
    dict : 分析結果
    """
    print("\n" + "="*60)
    print("交互作用分析: Review_age × Review_religious")
    print("="*60)
    print("理論意義: 測試年齡是否調節宗教信仰對道德選擇的影響")
    
    # 主效應變數
    main_effects = [
        'Cluster_Eastern',
        'Cluster_Southern',
        'political_centered',
        'age_centered',
        'religious_centered'
    ]
    
    # 交互作用項
    interaction_terms = [
        ('age_centered', 'religious_centered')
    ]
    
    # 執行分析
    results = run_interaction_analysis(
        data=df,
        outcome_var='chose_lawful',
        main_effects=main_effects,
        interaction_terms=interaction_terms,
        focal_var='religious_centered',
        moderator_var='age_centered',
        alpha=0.05,
        save_dir=None
    )
    
    # 儲存結果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig_path = Path(figure_dir)
    fig_path.mkdir(parents=True, exist_ok=True)
    
    # 交互作用係數
    interaction_coef = results['interaction_results']['interaction_coefficients']
    interaction_coef.to_csv(
        output_path / 'interaction_age_religious_coefficients.csv',
        index=False,
        encoding='utf-8-sig'
    )
    print(f"\n✅ 交互作用係數: {output_path / 'interaction_age_religious_coefficients.csv'}")
    
    # LRT結果
    lrt_results = results['interaction_results']
    lrt_df = pd.DataFrame([{
        '比較': 'Base vs Interaction Model',
        'LRT_χ²': lrt_results['lrt_statistic'],
        'df': lrt_results['lrt_dof'],
        'p值': lrt_results['lrt_p_value'],
        'AIC_base': lrt_results['aic_base'],
        'AIC_interaction': lrt_results['aic_interaction'],
        'AIC_improvement': lrt_results['aic_improvement'],
        '顯著': '✓' if lrt_results['lrt_significant'] else ''
    }])
    
    lrt_df.to_csv(
        output_path / 'interaction_age_religious_lrt.csv',
        index=False,
        encoding='utf-8-sig'
    )
    print(f"✅ LRT結果: {output_path / 'interaction_age_religious_lrt.csv'}")
    
    # Simple Slopes（如果交互作用顯著）
    if results['simple_slopes'] is not None:
        results['simple_slopes'].to_csv(
            output_path / 'interaction_age_religious_simple_slopes.csv',
            index=False,
            encoding='utf-8-sig'
        )
        print(f"✅ Simple Slopes: {output_path / 'interaction_age_religious_simple_slopes.csv'}")
    
    # 圖表
    results['figures']['interaction_plot'].write_html(
        str(fig_path / 'interaction_age_religious_plot.html')
    )
    print(f"✅ 交互作用圖: {fig_path / 'interaction_age_religious_plot.html'}")
    
    results['figures']['comparison_plot'].write_html(
        str(fig_path / 'interaction_age_religious_comparison.html')
    )
    print(f"✅ 模型比較圖: {fig_path / 'interaction_age_religious_comparison.html'}")
    
    return results


def generate_summary_report(cluster_political_results: dict,
                            age_religious_results: dict,
                            vif_results: pd.DataFrame,
                            output_dir: str = 'report/drafts') -> None:
    """
    生成摘要報告 (Markdown格式)
    
    Parameters:
    -----------
    cluster_political_results : dict
        Cluster×Political分析結果
    age_religious_results : dict
        Age×Religious分析結果
    vif_results : DataFrame
        VIF檢查結果
    output_dir : str
        報告輸出目錄
    """
    print("\n" + "="*60)
    print("生成摘要報告")
    print("="*60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_file = output_path / 'chapter4_section3_interaction_analysis.md'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 第4章 統計推論\n\n")
        f.write("## 4.3 交互作用分析\n\n")
        
        # 分析目的
        f.write("### 分析目的\n\n")
        f.write("檢驗調節效應（moderation effect），探討：\n\n")
        f.write("1. 文化圈是否調節政治傾向對道德選擇的影響\n")
        f.write("2. 年齡是否調節宗教信仰對道德選擇的影響\n\n")
    
        f.write("### 共線性檢查\n\n")
        f.write("**VIF (Variance Inflation Factor) 結果**:\n\n")
        
        f.write("| 變數 | VIF | 判定 |\n")
        f.write("|------|-----|------|\n")
        
        for _, row in vif_results.iterrows():
            f.write(f"| {row['變數']} | {row['VIF']:.2f} | {row['判定']} |\n")
        
        max_vif = vif_results['VIF'].max()
        f.write(f"\n最大VIF = {max_vif:.2f}")
        
        if max_vif < 5:
            f.write("，所有變數的VIF均 < 5，共線性問題不嚴重。\n\n")
        elif max_vif < 10:
            f.write("，部分變數的VIF介於5-10，共線性問題中等。\n\n")
        else:
            f.write("，部分變數的VIF > 10，存在嚴重共線性問題。\n\n")
        
        f.write("---\n\n")
        
        # Cluster × Political
        f.write("### 交互作用1: Cluster × Review_political\n\n")
        
        cp_results = cluster_political_results['interaction_results']
        
        f.write("**理論意義**: 不同文化圈中，政治傾向對守法選擇的影響是否不同？\n\n")
        
        f.write("**Likelihood Ratio Test**:\n\n")
        f.write(f"- LRT χ²({cp_results['lrt_dof']}) = {cp_results['lrt_statistic']:.3f}\n")
        f.write(f"- p = {cp_results['lrt_p_value']:.4f}\n")
        
        if cp_results['lrt_significant']:
            f.write("- ✅ 交互作用模型顯著優於基礎模型 (p < 0.05)\n\n")
        else:
            f.write("- ❌ 交互作用模型未顯著改善 (p ≥ 0.05)\n\n")
        
        f.write("**模型比較**:\n\n")
        f.write(f"- 基礎模型 AIC: {cp_results['aic_base']:.2f}\n")
        f.write(f"- 交互作用模型 AIC: {cp_results['aic_interaction']:.2f}\n")
        f.write(f"- AIC改善: {cp_results['aic_improvement']:.2f}\n\n")
        
        # 交互作用項係數
        f.write("**交互作用項係數**:\n\n")
        cp_coef = cp_results['interaction_coefficients']
        
        f.write("| 交互作用項 | 係數 | SE | p值 | OR | 顯著 |\n")
        f.write("|-----------|------|-----|-----|-----|------|\n")
        
        for _, row in cp_coef.iterrows():
            sig = '✓' if row['顯著'] else ''
            f.write(f"| {row['交互作用項']} | {row['係數']:.4f} | {row['標準誤']:.4f} | "
                   f"{row['p值']:.4f} | {row['Odds_Ratio']:.3f} | {sig} |\n")
        
        f.write("\n")
        
        # Simple Slopes
        if cluster_political_results['simple_slopes'] is not None:
            f.write("**Simple Slopes分析**:\n\n")
            ss_df = cluster_political_results['simple_slopes']
            
            f.write("| Cluster水準 | Political斜率 | SE | p值 | 顯著 |\n")
            f.write("|------------|--------------|-----|-----|------|\n")
            
            for _, row in ss_df.iterrows():
                sig = '✓' if row['顯著'] else ''
                f.write(f"| {row.iloc[0]} | {row.iloc[2]:.4f} | {row.iloc[3]:.4f} | "
                       f"{row.iloc[4]:.4f} | {sig} |\n")
            
            f.write("\n")
        
        f.write("**結論**: ")
        if cp_results['lrt_significant']:
            f.write("文化圈調節了政治傾向對守法選擇的影響。不同文化圈中，政治傾向的效應強度或方向存在差異。\n\n")
        else:
            f.write("文化圈未調節政治傾向的效應。政治傾向對守法選擇的影響在各文化圈中保持一致。\n\n")
        
        f.write("---\n\n")
        
        # Age × Religious
        f.write("### 交互作用2: Review_age × Review_religious\n\n")
        
        ar_results = age_religious_results['interaction_results']
        
        f.write("**理論意義**: 不同年齡層中，宗教信仰對守法選擇的影響是否不同？\n\n")
        
        f.write("**Likelihood Ratio Test**:\n\n")
        f.write(f"- LRT χ²({ar_results['lrt_dof']}) = {ar_results['lrt_statistic']:.3f}\n")
        f.write(f"- p = {ar_results['lrt_p_value']:.4f}\n")
        
        if ar_results['lrt_significant']:
            f.write("- ✅ 交互作用模型顯著優於基礎模型 (p < 0.05)\n\n")
        else:
            f.write("- ❌ 交互作用模型未顯著改善 (p ≥ 0.05)\n\n")
        
        f.write("**模型比較**:\n\n")
        f.write(f"- 基礎模型 AIC: {ar_results['aic_base']:.2f}\n")
        f.write(f"- 交互作用模型 AIC: {ar_results['aic_interaction']:.2f}\n")
        f.write(f"- AIC改善: {ar_results['aic_improvement']:.2f}\n\n")
        
        # 交互作用項係數
        f.write("**交互作用項係數**:\n\n")
        ar_coef = ar_results['interaction_coefficients']
        
        f.write("| 交互作用項 | 係數 | SE | p值 | OR | 顯著 |\n")
        f.write("|-----------|------|-----|-----|-----|------|\n")
        
        for _, row in ar_coef.iterrows():
            sig = '✓' if row['顯著'] else ''
            f.write(f"| {row['交互作用項']} | {row['係數']:.4f} | {row['標準誤']:.4f} | "
                   f"{row['p值']:.4f} | {row['Odds_Ratio']:.3f} | {sig} |\n")
        
        f.write("\n")
        
        # Simple Slopes
        if age_religious_results['simple_slopes'] is not None:
            f.write("**Simple Slopes分析**:\n\n")
            ss_df = age_religious_results['simple_slopes']
            
            f.write("| Age水準 | Religious斜率 | SE | p值 | 顯著 |\n")
            f.write("|---------|--------------|-----|-----|------|\n")
            
            for _, row in ss_df.iterrows():
                sig = '✓' if row['顯著'] else ''
                f.write(f"| {row.iloc[0]} | {row.iloc[2]:.4f} | {row.iloc[3]:.4f} | "
                       f"{row.iloc[4]:.4f} | {sig} |\n")
            
            f.write("\n")
        
        f.write("**結論**: ")
        if ar_results['lrt_significant']:
            f.write("年齡調節了宗教信仰對守法選擇的影響。不同年齡層中，宗教信仰的效應強度存在差異。\n\n")
        else:
            f.write("年齡未調節宗教信仰的效應。宗教信仰對守法選擇的影響在各年齡層中保持一致。\n\n")
        
        f.write("---\n\n")
        
        # 整體討論
        f.write("### 整體討論\n\n")
        
        sig_count = sum([
            cp_results['lrt_significant'],
            ar_results['lrt_significant']
        ])
        
        f.write(f"1. **顯著交互作用**: {sig_count}/2 個交互作用達顯著水準\n\n")
        
        f.write("2. **理論意涵**:\n")
        
        if cp_results['lrt_significant']:
            f.write("   - 文化調節效應存在，支持文化心理學理論\n")
            f.write("   - 不同文化圈的道德判斷機制可能存在質性差異\n")
        
        if ar_results['lrt_significant']:
            f.write("   - 發展心理學視角：宗教對道德的影響隨年齡變化\n")
            f.write("   - 可能反映世代差異或生命歷程效應\n")
        
        if sig_count == 0:
            f.write("   - 未發現顯著調節效應\n")
            f.write("   - 主效應模型（邏輯迴歸）已足夠解釋資料\n")
        
        f.write("\n3. **方法學啟示**: ")
        f.write("交互作用分析有助於理解道德判斷的情境依賴性(context-dependency)，避免過度簡化的線性理解。\n")
    
    print(f"✅ 摘要報告: {report_file}")


def main():
    """主執行函數"""
    print("\n" + "=" * 70)
    print("🔀 MIT Moral Machine - 交互作用分析 (Chapter 4.3)")
    print("=" * 70)
    
    # 設定日誌
    logger = setup_logging()
    logger.info("開始執行交互作用分析...")
    
    try:
        # Step 1: 載入資料
        df = load_and_prepare_data()
        
        # Step 2: 準備交互作用變數
        df_model = prepare_interaction_data(df)
        
        # Step 2.5: VIF共線性檢查
        vif_results = calculate_vif(df_model)
        
        # Step 3: Cluster × Political
        cp_results = analyze_cluster_political_interaction(df_model)
        
        # Step 4: Age × Religious
        ar_results = analyze_age_religious_interaction(df_model)
        
        # Step 5: 生成報告
        generate_summary_report(cp_results, ar_results, vif_results) 
        
        # 完成
        print("\n" + "=" * 70)
        print("✅ 交互作用分析完成！")
        print("=" * 70)
        print("\n📊 已產生以下輸出:")
        print("  【表格】")
        print("  - outputs/tables/chapter4/interaction_vif.csv")
        print("  - outputs/tables/chapter4/interaction_cluster_political_*.csv")
        print("  - outputs/tables/chapter4/interaction_age_religious_*.csv")
        print("\n  【圖表】")
        print("  - outputs/figures/chapter4_inference/interaction_cluster_political_*.html")
        print("  - outputs/figures/chapter4_inference/interaction_age_religious_*.html")
        print("\n  【報告】")
        print("  - report/drafts/chapter4_section3_interaction_analysis.md")
        print("  - outputs/logs/interaction_analysis.log")
        print("\n💡 下一步: python scripts/12_hierarchical_linear_model.py")
        print("=" * 70 + "\n")
        
        logger.info("交互作用分析完成")
        
    except Exception as e:
        logger.error(f"執行失敗: {e}", exc_info=True)
        print(f"\n❌ 錯誤: {e}")
        raise


if __name__ == '__main__':
    main()