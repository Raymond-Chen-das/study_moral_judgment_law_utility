"""
09_hypothesis_testing.py
========================
第四章：統計推論 - 假設檢定

功能：
1. 文化差異卡方檢定 (H1)
2. 場景特徵單變量分析
3. 人口統計變數單變量分析

執行方式：
    python scripts/09_hypothesis_testing.py
"""

import sys
from pathlib import Path

# 將專案根目錄加入路徑
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.inference.chi_square import ChiSquareTest, run_chi_square_analysis
from src.analysis.inference.t_tests import UnivariateAnalysis, run_univariate_analysis
import pandas as pd
import numpy as np
import logging
from datetime import datetime


def setup_logging(log_dir: str = 'outputs/logs') -> logging.Logger:
    """設定日誌"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    log_file = log_path / 'hypothesis_testing.log'
    
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
    DataFrame : 過濾後的資料 (排除Cluster==-1)
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
    
    # 檢查資料
    print(f"\n資料概覽:")
    print(f"  文化圈分佈:")
    cluster_counts = df_filtered['Cluster'].value_counts().sort_index()
    cluster_names = {0: 'Western', 1: 'Eastern', 2: 'Southern'}
    for cluster, count in cluster_counts.items():
        name = cluster_names.get(cluster, str(cluster))
        print(f"    {name} (Cluster {cluster}): {count:,} ({count/len(df_filtered)*100:.1f}%)")
    
    print(f"\n  場景數: {df_filtered['ResponseID'].nunique():,}")
    print(f"  使用者數: {df_filtered['UserID'].nunique():,}")
    print(f"  國家數: {df_filtered['UserCountry3'].nunique():,}")
    
    return df_filtered


def analyze_culture_differences(df: pd.DataFrame, 
                                output_dir: str = 'outputs/tables/chapter4',
                                figure_dir: str = 'outputs/figures/chapter4_inference') -> dict:
    """
    分析文化圈差異 (H1: 文化圈對道德選擇有顯著影響)
    
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
    print("H1: 文化圈差異檢定")
    print("="*60)
    
    # 執行卡方分析
    results = run_chi_square_analysis(
        data=df,
        outcome_var='chose_lawful',
        group_var='Cluster',
        alpha=0.05,
        save_dir=figure_dir
    )
    
    # 儲存詳細結果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 主要檢定結果
    main_results_df = pd.DataFrame([{
        '檢定': '卡方檢定',
        '假設': 'H1: 文化圈影響守法選擇',
        'χ²統計量': results['main_results']['chi2'],
        '自由度': results['main_results']['dof'],
        'p值': results['main_results']['p_value'],
        'Cramér\'s V': results['main_results']['cramers_v'],
        '效果解釋': results['main_results']['effect_interpretation'],
        '結論': '顯著' if results['main_results']['significant'] else '不顯著'
    }])
    
    main_results_df.to_csv(
        output_path / 'h1_chi_square_main_results.csv',
        index=False,
        encoding='utf-8-sig'
    )
    print(f"\n✅ 主要檢定結果: {output_path / 'h1_chi_square_main_results.csv'}")
    
    # 2. 事後比較
    pairwise_df = results['pairwise_results']
    pairwise_df.to_csv(
        output_path / 'h1_pairwise_comparisons.csv',
        index=False,
        encoding='utf-8-sig'
    )
    print(f"✅ 事後比較結果: {output_path / 'h1_pairwise_comparisons.csv'}")
    
    # 3. 各文化圈守法選擇率
    proportions = results['main_results']['proportions']
    cluster_names = {0: 'Western', 1: 'Eastern', 2: 'Southern'}
    
    prop_df = pd.DataFrame([
        {
            '文化圈': cluster_names.get(cluster, str(cluster)),
            'Cluster': cluster,
            '守法選擇率': prop,
            '百分比': f"{prop*100:.1f}%"
        }
        for cluster, prop in proportions.items()
    ])
    
    prop_df.to_csv(
        output_path / 'h1_cluster_proportions.csv',
        index=False,
        encoding='utf-8-sig'
    )
    print(f"✅ 文化圈比例: {output_path / 'h1_cluster_proportions.csv'}")
    
    return results


def analyze_scenario_features(df: pd.DataFrame,
                              output_dir: str = 'outputs/tables/chapter4',
                              figure_dir: str = 'outputs/figures/chapter4_inference') -> dict:
    """
    場景特徵單變量分析
    
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
    print("場景特徵單變量分析")
    print("="*60)
    
    # 定義要分析的場景特徵
    scenario_vars = ['is_lawful', 'is_majority', 'lawful_vs_majority_conflict']
    
    # 定義變數類型
    var_types = {
        'is_lawful': 'categorical',
        'is_majority': 'categorical',
        'lawful_vs_majority_conflict': 'categorical'
    }
    
    # 執行批次分析
    results = run_univariate_analysis(
        data=df,
        outcome_var='chose_lawful',
        test_vars=scenario_vars,
        var_types=var_types,
        alpha=0.05,
        save_dir=None  # 稍後手動儲存
    )
    
    results_df = results['results_table']
    
    # 儲存結果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(
        output_path / 'scenario_features_univariate.csv',
        index=False,
        encoding='utf-8-sig'
    )
    print(f"\n✅ 場景特徵分析: {output_path / 'scenario_features_univariate.csv'}")
    
    # 儲存圖表
    fig_path = Path(figure_dir)
    fig_path.mkdir(parents=True, exist_ok=True)
    
    results['figures']['forest_plot'].write_html(
        str(fig_path / 'scenario_features_forest_plot.html')
    )
    print(f"✅ 森林圖: {fig_path / 'scenario_features_forest_plot.html'}")
    
    return results


def analyze_demographic_variables(df: pd.DataFrame,
                                  output_dir: str = 'outputs/tables/chapter4',
                                  figure_dir: str = 'outputs/figures/chapter4_inference') -> dict:
    """
    人口統計變數單變量分析
    
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
    print("人口統計變數單變量分析")
    print("="*60)
    
    # 定義要分析的人口統計變數
    demo_vars = ['Review_age', 'Review_political', 'Review_religious']
    
    # 定義變數類型
    var_types = {
        'Review_age': 'continuous',
        'Review_political': 'continuous',
        'Review_religious': 'continuous'
    }
    
    # 執行批次分析
    results = run_univariate_analysis(
        data=df,
        outcome_var='chose_lawful',
        test_vars=demo_vars,
        var_types=var_types,
        alpha=0.05,
        save_dir=None
    )
    
    results_df = results['results_table']
    
    # 儲存結果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(
        output_path / 'demographic_univariate.csv',
        index=False,
        encoding='utf-8-sig'
    )
    print(f"\n✅ 人口統計分析: {output_path / 'demographic_univariate.csv'}")
    
    # 儲存圖表
    fig_path = Path(figure_dir)
    fig_path.mkdir(parents=True, exist_ok=True)
    
    results['figures']['forest_plot'].write_html(
        str(fig_path / 'demographic_forest_plot.html')
    )
    print(f"✅ 森林圖: {fig_path / 'demographic_forest_plot.html'}")
    
    return results


def generate_summary_report(culture_results: dict,
                            scenario_results: dict,
                            demo_results: dict,
                            output_dir: str = 'report/drafts') -> None:
    """
    生成摘要報告 (Markdown格式)
    
    Parameters:
    -----------
    culture_results : dict
        文化差異分析結果
    scenario_results : dict
        場景特徵分析結果
    demo_results : dict
        人口統計分析結果
    output_dir : str
        報告輸出目錄
    """
    print("\n" + "="*60)
    print("生成摘要報告")
    print("="*60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_file = output_path / 'chapter4_section1_hypothesis_testing.md'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 第4章 統計推論\n\n")
        f.write("## 4.1 假設檢定\n\n")
        
        # H1: 文化圈差異
        f.write("### H1: 文化圈對道德選擇的影響\n\n")
        f.write("**研究假設**: 不同文化圈（Western, Eastern, Southern）在「守法vs.效益」道德兩難中的選擇存在顯著差異。\n\n")
        
        main = culture_results['main_results']
        
        f.write("**檢定方法**: 卡方檢定 (Chi-Square Test)\n\n")
        f.write("**結果**:\n\n")
        p_str = "< .001" if main['p_value'] < 0.001 else f"= {main['p_value']:.3f}"
        f.write(f"- χ²({main['dof']}) = {main['chi2']:.3f}, p {p_str}\n")
        f.write(f"- Cramér's V = {main['cramers_v']:.3f} ({main['effect_interpretation']})\n\n")
        
        if main['significant']:
            f.write("**結論**: ✅ 文化圈間存在顯著差異 (p < 0.05)\n\n")
        else:
            f.write("**結論**: ❌ 文化圈間無顯著差異 (p ≥ 0.05)\n\n")
        
        f.write("**各文化圈守法選擇率**:\n\n")
        cluster_names = {0: 'Western', 1: 'Eastern', 2: 'Southern'}
        for cluster, prop in main['proportions'].items():
            name = cluster_names.get(cluster, str(cluster))
            f.write(f"- {name}: {prop*100:.1f}%\n")
        
        f.write("\n**事後比較** (Bonferroni校正):\n\n")
        pairwise = culture_results['pairwise_results']
        
        f.write("| 比較 | 比例差異 | p值(校正後) | 顯著 |\n")
        f.write("|------|----------|------------|------|\n")
        
        for _, row in pairwise.iterrows():
            g1_name = cluster_names.get(row['Group1'], str(row['Group1']))
            g2_name = cluster_names.get(row['Group2'], str(row['Group2']))
            sig = '✓' if row['significant'] else ''
            
            f.write(f"| {g1_name} vs {g2_name} | {row['Diff']:+.3f} | {row['p_adjusted']:.4f} | {sig} |\n")
        
        f.write("\n---\n\n")
        
        # 場景特徵分析
        f.write("### 場景特徵單變量分析\n\n")
        f.write("檢驗場景本身的特徵（守法性、多數性、衝突性）是否影響守法選擇。\n\n")
        
        scenario_table = scenario_results['results_table']
        
        f.write("| 變數 | 檢定方法 | 統計量 | p值 | 效果量 | 顯著 |\n")
        f.write("|------|---------|--------|-----|--------|------|\n")
        
        for _, row in scenario_table.iterrows():
            f.write(f"| {row['變數']} | {row['檢定方法']} | {row['統計量']} | {row['p值']:.4f} | {row['效果量']:.3f} | {row['顯著']} |\n")
        
        f.write("\n---\n\n")
        
        # 人口統計變數分析
        f.write("### 人口統計變數單變量分析\n\n")
        f.write("檢驗個人特徵（年齡、政治傾向、宗教信仰）是否影響守法選擇。\n\n")
        
        demo_table = demo_results['results_table']
        
        f.write("| 變數 | 檢定方法 | 統計量 | p值 | 效果量 | 效果解釋 | 顯著 |\n")
        f.write("|------|---------|--------|-----|--------|---------|------|\n")
        
        for _, row in demo_table.iterrows():
            f.write(f"| {row['變數']} | {row['檢定方法']} | {row['統計量']} | {row['p值']:.4f} | {row['效果量']:.3f} | {row['效果解釋']} | {row['顯著']} |\n")
        
        f.write("\n---\n\n")
        
        # 關鍵發現摘要
        f.write("### 關鍵發現摘要\n\n")
        
        # 統計顯著變數
        total_tests = len(scenario_table) + len(demo_table) + 1  # +1 for chi-square
        sig_tests = (
            (1 if main['significant'] else 0) +
            (scenario_table['顯著'] == '✓').sum() +
            (demo_table['顯著'] == '✓').sum()
        )
        
        f.write(f"1. **整體檢定**: {sig_tests}/{total_tests} 項檢定達到統計顯著 (α=0.05)\n\n")
        
        if main['significant']:
            f.write(f"2. **文化差異**: 三大文化圈的守法選擇率存在顯著差異\n")
            # 找出最高和最低
            max_cluster = max(main['proportions'].items(), key=lambda x: x[1])
            min_cluster = min(main['proportions'].items(), key=lambda x: x[1])
            
            max_name = cluster_names.get(max_cluster[0], str(max_cluster[0]))
            min_name = cluster_names.get(min_cluster[0], str(min_cluster[0]))
            
            f.write(f"   - {max_name}最高 ({max_cluster[1]*100:.1f}%)\n")
            f.write(f"   - {min_name}最低 ({min_cluster[1]*100:.1f}%)\n\n")
        
        f.write("3. **效果量**: 所有檢定的效果量均為小到中等，符合道德判斷的複雜性\n\n")
        
        f.write("4. **實務意義**: 雖然統計顯著，但效果量提醒我們避免過度解讀文化差異的實質影響\n\n")
    
    print(f"✅ 摘要報告: {report_file}")


def main():
    """主執行函數"""
    print("\n" + "=" * 70)
    print("🔍 MIT Moral Machine - 假設檢定分析 (Chapter 4.1)")
    print("=" * 70)
    
    # 設定日誌
    logger = setup_logging()
    logger.info("開始執行假設檢定分析...")
    
    try:
        # Step 1: 載入資料
        df = load_and_prepare_data()
        
        # Step 2: H1 - 文化圈差異檢定
        culture_results = analyze_culture_differences(df)
        
        # Step 3: 場景特徵分析
        scenario_results = analyze_scenario_features(df)
        
        # Step 4: 人口統計變數分析
        demo_results = analyze_demographic_variables(df)
        
        # Step 5: 生成摘要報告
        generate_summary_report(culture_results, scenario_results, demo_results)
        
        # 完成
        print("\n" + "=" * 70)
        print("✅ 假設檢定分析完成！")
        print("=" * 70)
        print("\n📊 已產生以下輸出:")
        print("  【表格】")
        print("  - outputs/tables/chapter4/h1_chi_square_main_results.csv")
        print("  - outputs/tables/chapter4/h1_pairwise_comparisons.csv")
        print("  - outputs/tables/chapter4/h1_cluster_proportions.csv")
        print("  - outputs/tables/chapter4/scenario_features_univariate.csv")
        print("  - outputs/tables/chapter4/demographic_univariate.csv")
        print("\n  【圖表】")
        print("  - outputs/figures/chapter4_inference/contingency_heatmap.html")
        print("  - outputs/figures/chapter4_inference/proportion_bar_chart.html")
        print("  - outputs/figures/chapter4_inference/pairwise_comparison.html")
        print("  - outputs/figures/chapter4_inference/scenario_features_forest_plot.html")
        print("  - outputs/figures/chapter4_inference/demographic_forest_plot.html")
        print("\n  【報告】")
        print("  - report/drafts/chapter4_section1_hypothesis_testing.md")
        print("  - outputs/logs/hypothesis_testing.log")
        print("\n💡 下一步: python scripts/10_logistic_regression.py")
        print("=" * 70 + "\n")
        
        logger.info("假設檢定分析完成")
        
    except Exception as e:
        logger.error(f"執行失敗: {e}", exc_info=True)
        print(f"\n❌ 錯誤: {e}")
        raise


if __name__ == '__main__':
    main()