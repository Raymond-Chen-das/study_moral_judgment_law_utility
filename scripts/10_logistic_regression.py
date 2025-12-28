"""
10_logistic_regression.py
=========================
第四章：統計推論 - 邏輯迴歸分析

功能：
1. 多變量邏輯迴歸
2. VIF共線性診斷
3. 模型擬合度檢驗
4. 係數與Odds Ratio視覺化

執行方式：
    python scripts/10_logistic_regression.py
"""

import sys
from pathlib import Path

# 將專案根目錄加入路徑
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.inference.logistic_regression import (
    LogisticRegressionAnalysis, 
    run_logistic_regression
)
import pandas as pd
import numpy as np
import logging
from datetime import datetime


def setup_logging(log_dir: str = 'outputs/logs') -> logging.Logger:
    """設定日誌"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    log_file = log_path / 'logistic_regression.log'
    
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
    
    # 檢查資料分佈
    print(f"\n結果變數分佈:")
    chose_lawful_dist = df_clean['chose_lawful'].value_counts()
    for value, count in chose_lawful_dist.items():
        print(f"  chose_lawful={value}: {count:,} ({count/len(df_clean)*100:.1f}%)")
    
    print(f"\n文化圈分佈:")
    cluster_counts = df_clean['Cluster'].value_counts().sort_index()
    cluster_names = {0: 'Western', 1: 'Eastern', 2: 'Southern'}
    for cluster, count in cluster_counts.items():
        name = cluster_names.get(cluster, str(cluster))
        print(f"  {name}: {count:,} ({count/len(df_clean)*100:.1f}%)")
    
    return df_clean


def prepare_predictors(df: pd.DataFrame) -> tuple:
    """
    準備預測變數
    
    Parameters:
    -----------
    df : DataFrame
        資料
        
    Returns:
    --------
    tuple : (準備好的資料, 預測變數列表)
    """
    print("\n" + "="*60)
    print("準備預測變數...")
    print("="*60)
    
    # 建立Cluster的dummy變數 (Western為參考組)
    cluster_dummies = pd.get_dummies(df['Cluster'], prefix='Cluster', drop_first=False)
    
    # 選擇Eastern和Southern作為預測變數 (Western=參考組)
    # Western: Cluster==0, Eastern: Cluster==1, Southern: Cluster==2
    df_model = df.copy()
    df_model['Cluster_Eastern'] = (df['Cluster'] == 1).astype(int)
    df_model['Cluster_Southern'] = (df['Cluster'] == 2).astype(int)
    
    print("✅ Cluster變數編碼:")
    print("  參考組: Western (Cluster==0)")
    print("  Cluster_Eastern: 1 if Eastern, 0 otherwise")
    print("  Cluster_Southern: 1 if Southern, 0 otherwise")
    
    # 預測變數列表
    predictor_vars = [
        'Cluster_Eastern',
        'Cluster_Southern', 
        'Review_age',
        'Review_political',
        'Review_religious'
    ]
    
    print(f"\n✅ 預測變數 ({len(predictor_vars)} 個):")
    for var in predictor_vars:
        print(f"  - {var}")
    
    # 描述統計
    print(f"\n預測變數描述統計:")
    for var in predictor_vars:
        mean_val = df_model[var].mean()
        std_val = df_model[var].std()
        print(f"  {var}: Mean={mean_val:.3f}, SD={std_val:.3f}")
    
    return df_model, predictor_vars


def run_main_logistic_regression(df: pd.DataFrame,
                                 predictor_vars: list,
                                 output_dir: str = 'outputs/tables/chapter4',
                                 figure_dir: str = 'outputs/figures/chapter4_inference') -> dict:
    """
    執行主要邏輯迴歸分析
    
    Parameters:
    -----------
    df : DataFrame
        資料
    predictor_vars : list
        預測變數列表
    output_dir : str
        表格輸出目錄
    figure_dir : str
        圖表輸出目錄
        
    Returns:
    --------
    dict : 分析結果
    """
    print("\n" + "="*60)
    print("邏輯迴歸分析")
    print("="*60)
    
    # 執行分析
    results = run_logistic_regression(
        data=df,
        outcome_var='chose_lawful',
        predictor_vars=predictor_vars,
        alpha=0.05,
        save_dir=None  # 稍後手動儲存
    )
    
    # 儲存結果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig_path = Path(figure_dir)
    fig_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 係數表
    coef_df = results['coefficients']
    coef_df.to_csv(
        output_path / 'logistic_regression_coefficients.csv',
        index=False,
        encoding='utf-8-sig'
    )
    print(f"\n✅ 係數表: {output_path / 'logistic_regression_coefficients.csv'}")
    
    # 2. VIF診斷
    vif_df = results['vif']
    vif_df.to_csv(
        output_path / 'logistic_regression_vif.csv',
        index=False,
        encoding='utf-8-sig'
    )
    print(f"✅ VIF診斷: {output_path / 'logistic_regression_vif.csv'}")
    
    # 3. 模型摘要
    model_summary = results['model_summary']
    summary_df = pd.DataFrame([model_summary])
    summary_df.to_csv(
        output_path / 'logistic_regression_model_summary.csv',
        index=False,
        encoding='utf-8-sig'
    )
    print(f"✅ 模型摘要: {output_path / 'logistic_regression_model_summary.csv'}")
    
    # 4. Hosmer-Lemeshow檢定
    hl_test = results['hosmer_lemeshow']
    hl_df = pd.DataFrame([hl_test])
    hl_df.to_csv(
        output_path / 'logistic_regression_hosmer_lemeshow.csv',
        index=False,
        encoding='utf-8-sig'
    )
    print(f"✅ H-L檢定: {output_path / 'logistic_regression_hosmer_lemeshow.csv'}")
    
    # 5. 圖表
    results['figures']['coefficient_plot'].write_html(
        str(fig_path / 'logistic_coefficients_plot.html')
    )
    print(f"✅ 係數圖: {fig_path / 'logistic_coefficients_plot.html'}")
    
    results['figures']['odds_ratio_plot'].write_html(
        str(fig_path / 'logistic_odds_ratio_plot.html')
    )
    print(f"✅ OR圖: {fig_path / 'logistic_odds_ratio_plot.html'}")
    
    return results


def interpret_results(results: dict) -> dict:
    """
    解釋邏輯迴歸結果
    
    Parameters:
    -----------
    results : dict
        分析結果
        
    Returns:
    --------
    dict : 解釋文字
    """
    coef_df = results['coefficients']
    model_summary = results['model_summary']
    hl_test = results['hosmer_lemeshow']
    vif_df = results['vif']
    
    interpretation = {}
    
    # 1. 模型整體評估
    interpretation['model_fit'] = {
        'pseudo_r2': model_summary['pseudo_r2_mcfadden'],
        'aic': model_summary['aic'],
        'bic': model_summary['bic'],
        'hosmer_lemeshow_p': hl_test['p_value'],
        'good_fit': hl_test['good_fit']
    }
    
    # 2. 共線性評估
    severe_vif = vif_df[vif_df['共線性'] == '嚴重']
    interpretation['multicollinearity'] = {
        'severe_issues': len(severe_vif) > 0,
        'max_vif': vif_df['VIF'].max(),
        'severe_vars': severe_vif['變數'].tolist() if len(severe_vif) > 0 else []
    }
    
    # 3. 顯著預測變數
    sig_predictors = coef_df[(coef_df['顯著']) & (coef_df['變數'] != 'const')]
    
    interpretation['significant_predictors'] = []
    
    for _, row in sig_predictors.iterrows():
        direction = "增加" if row['係數'] > 0 else "減少"
        
        pred_info = {
            'variable': row['變數'],
            'coefficient': row['係數'],
            'odds_ratio': row['Odds_Ratio'],
            'p_value': row['p值'],
            'ci_lower': row['OR_CI_下界'],
            'ci_upper': row['OR_CI_上界'],
            'direction': direction,
            'interpretation': f"{row['變數']}: OR={row['Odds_Ratio']:.3f}, {direction}守法選擇機率"
        }
        
        interpretation['significant_predictors'].append(pred_info)
    
    # 4. 文化圈效應
    cluster_effects = []
    for var in ['Cluster_Eastern', 'Cluster_Southern']:
        if var in coef_df['變數'].values:
            row = coef_df[coef_df['變數'] == var].iloc[0]
            cluster_effects.append({
                'cluster': var.replace('Cluster_', ''),
                'vs_reference': 'Western',
                'odds_ratio': row['Odds_Ratio'],
                'significant': row['顯著'],
                'p_value': row['p值']
            })
    
    interpretation['cluster_effects'] = cluster_effects
    
    return interpretation


def generate_summary_report(results: dict,
                            interpretation: dict,
                            output_dir: str = 'report/drafts') -> None:
    """
    生成摘要報告 (Markdown格式)
    
    Parameters:
    -----------
    results : dict
        分析結果
    interpretation : dict
        結果解釋
    output_dir : str
        報告輸出目錄
    """
    print("\n" + "="*60)
    print("生成摘要報告")
    print("="*60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_file = output_path / 'chapter4_section2_logistic_regression.md'
    
    coef_df = results['coefficients']
    model_summary = results['model_summary']
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 第4章 統計推論\n\n")
        f.write("## 4.2 多變量邏輯迴歸分析\n\n")
        
        # 模型設定
        f.write("### 模型設定\n\n")
        f.write("**依變數**: `chose_lawful` (0=未選守法方, 1=選擇守法方)\n\n")
        f.write("**自變數**:\n")
        f.write("- **文化圈**: `Cluster_Eastern`, `Cluster_Southern` (參考組: Western)\n")
        f.write("- **年齡**: `Review_age`\n")
        f.write("- **政治傾向**: `Review_political`\n")
        f.write("- **宗教信仰**: `Review_religious`\n\n")
        
        # 模型摘要
        f.write("### 模型摘要\n\n")
        f.write(f"- **樣本數**: {model_summary['n_obs']:,}\n")
        f.write(f"- **Pseudo R² (McFadden)**: {model_summary['pseudo_r2_mcfadden']:.4f}\n")
        f.write(f"- **AIC**: {model_summary['aic']:.2f}\n")
        f.write(f"- **BIC**: {model_summary['bic']:.2f}\n\n")
        
        # 擬合度檢定
        hl_test = results['hosmer_lemeshow']
        f.write("**Hosmer-Lemeshow擬合度檢定**:\n")
        p_str = "< .001" if hl_test['p_value'] < 0.001 else f"= {hl_test['p_value']:.3f}"
        f.write(f"- χ²({hl_test['dof']}) = {hl_test['chi2']:.3f}, p {p_str}\n")
        if hl_test['good_fit']:
            f.write("- ✅ 模型擬合良好 (p > 0.05)\n\n")
        else:
            f.write("- ⚠️  模型擬合可能不佳 (p < 0.05)\n\n")
        
        # 共線性診斷
        f.write("**共線性診斷 (VIF)**:\n")
        vif_df = results['vif']
        max_vif = vif_df['VIF'].max()
        
        if interpretation['multicollinearity']['severe_issues']:
            f.write(f"- ⚠️  檢測到嚴重共線性 (VIF > 10)\n")
            for var in interpretation['multicollinearity']['severe_vars']:
                vif_val = vif_df[vif_df['變數'] == var]['VIF'].values[0]
                f.write(f"  - {var}: VIF = {vif_val:.2f}\n")
        else:
            f.write(f"- ✅ 無嚴重共線性問題 (最大VIF = {max_vif:.2f})\n")
        
        f.write("\n---\n\n")
        
        # 係數表
        f.write("### 迴歸係數\n\n")
        f.write("| 變數 | 係數 | SE | z值 | p值 | OR | 95% CI | 顯著 |\n")
        f.write("|------|------|-----|-----|-----|-----|--------|------|\n")
        
        for _, row in coef_df.iterrows():
            if row['變數'] == 'const':
                var_name = '截距'
            else:
                var_name = row['變數']
            
            ci = f"[{row['OR_CI_下界']:.3f}, {row['OR_CI_上界']:.3f}]"
            sig = row['星號']
            
            p_display = "< .001" if row['p值'] < 0.001 else f"{row['p值']:.3f}"
            f.write(f"| {var_name} | {row['係數']:.4f} | {row['標準誤']:.4f} | "
                f"{row['z值']:.3f} | {p_display} | {row['Odds_Ratio']:.3f} | "
                f"{ci} | {sig} |\n")
        
        f.write("\n註: *** p<0.001, ** p<0.01, * p<0.05\n\n")
        
        f.write("---\n\n")
        
        # 結果詮釋
        f.write("### 結果詮釋\n\n")
        
        # 顯著預測變數
        sig_preds = interpretation['significant_predictors']
        
        if len(sig_preds) > 0:
            f.write("**顯著預測變數**:\n\n")
            
            for pred in sig_preds:
                f.write(f"**{pred['variable']}**:\n")
                f.write(f"- Odds Ratio: {pred['odds_ratio']:.3f} "
                       f"(95% CI: [{pred['ci_lower']:.3f}, {pred['ci_upper']:.3f}])\n")
                p_str = "< .001" if pred['p_value'] < 0.001 else f"= {pred['p_value']:.3f}"
                f.write(f"- p {p_str}\n")
                
                # 實質解釋
                if pred['odds_ratio'] > 1:
                    change_pct = (pred['odds_ratio'] - 1) * 100
                    f.write(f"- 解釋: {pred['variable']}每增加1單位，守法選擇的勝算增加{change_pct:.1f}%\n")
                else:
                    change_pct = (1 - pred['odds_ratio']) * 100
                    f.write(f"- 解釋: {pred['variable']}每增加1單位，守法選擇的勝算減少{change_pct:.1f}%\n")
                
                f.write("\n")
        else:
            f.write("**無顯著預測變數** (α = 0.05)\n\n")
        
        # 文化圈效應
        f.write("**文化圈效應** (相對於Western):\n\n")
        
        for effect in interpretation['cluster_effects']:
            cluster = effect['cluster']
            or_val = effect['odds_ratio']
            sig = '✓' if effect['significant'] else ''
            
            p_str = "< .001" if effect['p_value'] < 0.001 else f"= {effect['p_value']:.3f}"
            f.write(f"- **{cluster} vs Western**: OR = {or_val:.3f}, p {p_str} {sig}\n")
        
        f.write("\n---\n\n")
        
        # 關鍵發現
        f.write("### 關鍵發現\n\n")
        
        f.write(f"1. **模型解釋力**: Pseudo R² = {model_summary['pseudo_r2_mcfadden']:.4f}")
        if model_summary['pseudo_r2_mcfadden'] < 0.02:
            f.write(" (極低，模型解釋力有限)\n")
        elif model_summary['pseudo_r2_mcfadden'] < 0.15:
            f.write(" (低，符合道德判斷的複雜性)\n")
        else:
            f.write(" (中等)\n")
        
        f.write(f"\n2. **顯著變數數**: {len(sig_preds)}/{len(coef_df)-1} 個預測變數達顯著\n")
        
        # 檢查文化圈是否顯著
        cluster_sig = any(e['significant'] for e in interpretation['cluster_effects'])
        
        if cluster_sig:
            f.write("\n3. **文化圈效應**: 在控制其他變數後，文化圈對守法選擇仍有獨立影響\n")
        else:
            f.write("\n3. **文化圈效應**: 在控制其他變數後，文化圈效應不顯著\n")
        
        f.write("\n4. **實務啟示**: 道德判斷受多重因素影響，單一變數的預測力有限\n")
        
    print(f"✅ 摘要報告: {report_file}")


def main():
    """主執行函數"""
    print("\n" + "=" * 70)
    print("📊 MIT Moral Machine - 邏輯迴歸分析 (Chapter 4.2)")
    print("=" * 70)
    
    # 設定日誌
    logger = setup_logging()
    logger.info("開始執行邏輯迴歸分析...")
    
    try:
        # Step 1: 載入資料
        df = load_and_prepare_data()
        
        # Step 2: 準備預測變數
        df_model, predictor_vars = prepare_predictors(df)
        
        # Step 3: 執行邏輯迴歸
        results = run_main_logistic_regression(df_model, predictor_vars)
        
        # Step 4: 解釋結果
        interpretation = interpret_results(results)
        
        # Step 5: 生成報告
        generate_summary_report(results, interpretation)
        
        # 完成
        print("\n" + "=" * 70)
        print("✅ 邏輯迴歸分析完成！")
        print("=" * 70)
        print("\n📊 已產生以下輸出:")
        print("  【表格】")
        print("  - outputs/tables/chapter4/logistic_regression_coefficients.csv")
        print("  - outputs/tables/chapter4/logistic_regression_vif.csv")
        print("  - outputs/tables/chapter4/logistic_regression_model_summary.csv")
        print("  - outputs/tables/chapter4/logistic_regression_hosmer_lemeshow.csv")
        print("\n  【圖表】")
        print("  - outputs/figures/chapter4_inference/logistic_coefficients_plot.html")
        print("  - outputs/figures/chapter4_inference/logistic_odds_ratio_plot.html")
        print("\n  【報告】")
        print("  - report/drafts/chapter4_section2_logistic_regression.md")
        print("  - outputs/logs/logistic_regression.log")
        print("\n💡 下一步: python scripts/11_interaction_analysis.py")
        print("=" * 70 + "\n")
        
        logger.info("邏輯迴歸分析完成")
        
    except Exception as e:
        logger.error(f"執行失敗: {e}", exc_info=True)
        print(f"\n❌ 錯誤: {e}")
        raise


if __name__ == '__main__':
    main()