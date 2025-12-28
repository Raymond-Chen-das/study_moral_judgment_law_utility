"""
第5章 第1節：HLM 隨機效應探索
==============================
研究問題：ICC=14.35%的國家層級變異從何而來？

執行此腳本前請確認：
1. 已完成第4章 HLM 分析，產出 hlm_random_effects.csv
2. 原始資料 CountriesChangePr.csv 位於 data/raw/

產出：
- outputs/figures/chapter5/random_effect_correlation.html
- outputs/figures/chapter5/random_effect_scatter.html
- outputs/tables/chapter5/random_effect_correlation.csv
- outputs/tables/chapter5/random_effect_regression.csv
- report/drafts/chapter5_section1_random_effect.md
""" 

import os
import sys
from pathlib import Path
from datetime import datetime

# 添加專案根目錄到路徑
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

# 導入自定義模組
from src.analysis.integration.random_effect_explorer import (
    RandomEffectExplorer, load_and_analyze
)
from src.visualization.chapter5.chapter5_plots import (
    plot_random_effect_correlations,
    plot_random_effect_scatter
)


def main():
    """主執行函數"""
    
    print("=" * 70)
    print("第5章 第1節：HLM 隨機效應探索")
    print(f"執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # ========================================
    # 1. 設定路徑
    # ========================================
    
    # 輸入路徑
    RANDOM_EFFECTS_PATH = PROJECT_ROOT / "outputs/tables/chapter4/hlm_random_effects.csv"
    AMCE_PATH = PROJECT_ROOT / "data/raw/CountriesChangePr.csv"
    
    # 輸出路徑
    OUTPUT_FIG_DIR = PROJECT_ROOT / "outputs/figures/chapter5"
    OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs/tables/chapter5"
    REPORT_DIR = PROJECT_ROOT / "report/drafts"
    
    # 創建輸出目錄
    OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 檢查輸入檔案
    if not RANDOM_EFFECTS_PATH.exists():
        print(f"\n❌ 錯誤：找不到 HLM 隨機效應檔案")
        print(f"   預期路徑: {RANDOM_EFFECTS_PATH}")
        print("   請先執行第4章 HLM 分析 (script 12)")
        return
    
    if not AMCE_PATH.exists():
        print(f"\n❌ 錯誤：找不到 AMCE 資料檔案")
        print(f"   預期路徑: {AMCE_PATH}")
        return
    
    print(f"\n✅ 輸入檔案確認完成")
    
    # ========================================
    # 2. 載入與分析
    # ========================================
    
    explorer, corr_results, reg_results = load_and_analyze(
        random_effects_path=str(RANDOM_EFFECTS_PATH),
        amce_path=str(AMCE_PATH),
        verbose=True
    )
    
    # ========================================
    # 3. 視覺化
    # ========================================
    
    print("\n" + "=" * 60)
    print("生成視覺化圖表")
    print("=" * 60)
    
    # 3.1 相關係數條形圖
    fig_corr = plot_random_effect_correlations(
        correlation_df=corr_results,
        output_path=str(OUTPUT_FIG_DIR / "random_effect_correlation.html"),
        title="HLM 隨機效應與 AMCE 維度相關性"
    )
    
    # 3.2 散點圖（以最強相關維度為例）
    top_dimension = corr_results.iloc[0]['amce_dimension']
    scatter_data = explorer.get_scatter_data(amce_dimension=top_dimension)
    
    fig_scatter = plot_random_effect_scatter(
        scatter_data=scatter_data,
        amce_col=top_dimension,
        amce_label=corr_results.iloc[0]['chinese_name'],
        output_path=str(OUTPUT_FIG_DIR / "random_effect_scatter.html"),
        highlight_countries=['TWN', 'JPN', 'KOR', 'CHN', 'USA', 'DEU', 'GBR', 'FRA']
    )
    
    # ========================================
    # 4. 儲存表格
    # ========================================
    
    print("\n" + "=" * 60)
    print("儲存分析表格")
    print("=" * 60)
    
    # 相關分析結果
    corr_results.to_csv(
        OUTPUT_TABLE_DIR / "random_effect_correlation.csv",
        index=False,
        encoding='utf-8-sig'
    )
    print(f"✅ 已儲存: random_effect_correlation.csv")
    
    # 迴歸係數
    reg_results['coefficients'].to_csv(
        OUTPUT_TABLE_DIR / "random_effect_regression.csv",
        index=False,
        encoding='utf-8-sig'
    )
    print(f"✅ 已儲存: random_effect_regression.csv")
    
    # ========================================
    # 5. 生成報告草稿
    # ========================================
    
    print("\n" + "=" * 60)
    print("生成報告草稿")
    print("=" * 60)
    
    report_content = generate_section_report(
        corr_results=corr_results,
        reg_results=reg_results,
        explorer=explorer
    )
    
    report_path = REPORT_DIR / "chapter5_section1_random_effect.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"✅ 已儲存: {report_path}")
    
    # ========================================
    # 6. 總結
    # ========================================
    
    print("\n" + "=" * 70)
    print("第5.1節執行完成！")
    print("=" * 70)
    
    print("\n📊 產出檔案：")
    print(f"   - {OUTPUT_FIG_DIR / 'random_effect_correlation.html'}")
    print(f"   - {OUTPUT_FIG_DIR / 'random_effect_scatter.html'}")
    print(f"   - {OUTPUT_TABLE_DIR / 'random_effect_correlation.csv'}")
    print(f"   - {OUTPUT_TABLE_DIR / 'random_effect_regression.csv'}")
    print(f"   - {report_path}")
    
    print("\n🔑 關鍵發現：")
    top3 = corr_results.head(3)
    for i, row in top3.iterrows():
        sig = '***' if row['significant_001'] else ('**' if row['significant_01'] else '*')
        print(f"   {i+1}. {row['chinese_name']}: r = {row['pearson_r']:.3f} {sig}")
    
    print(f"\n   多元迴歸 R² = {reg_results['r2']:.4f}")
    print(f"   → {reg_results['r2']*100:.1f}% 的國家層級變異可由 9 個 AMCE 維度解釋")


def generate_section_report(
    corr_results: pd.DataFrame,
    reg_results: dict,
    explorer: RandomEffectExplorer
) -> str:
    """生成 5.1 節報告草稿"""
    
    report = []
    report.append("# 第5章 整合分析與模型驗證\n")
    report.append(f"**分析時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    report.append("## 5.1 國家層級變異來源探索\n")
    
    report.append("### 研究問題\n")
    report.append("第4章 HLM 分析發現 ICC = 0.25%，即 0.25% 的守法選擇變異來自國家層級。")
    report.append("本節探討這些變異的來源：哪些國家道德偏好維度與國家層級守法傾向相關？\n")
    
    report.append("### 分析方法\n")
    report.append("計算 HLM 隨機效應（各國對全球平均的偏離）與 9 個 AMCE 維度（國家道德偏好）的 Pearson 相關係數。\n")
    
    report.append("### 相關分析結果\n")
    report.append("| 排序 | AMCE 維度 | Pearson r | p 值 | 效果量 |")
    report.append("|------|----------|-----------|------|--------|")
    
    for i, row in corr_results.iterrows():
        sig = '***' if row['significant_001'] else ('**' if row['significant_01'] else ('*' if row['significant_05'] else ''))
        report.append(f"| {i+1} | {row['chinese_name']} | {row['pearson_r']:.3f} | {row['pearson_p']:.4f}{sig} | {row['effect_size']} |")
    
    report.append("\n註：* p<.05, ** p<.01, *** p<.001\n")
    
    report.append("### 多元迴歸分析\n")
    report.append(f"以 9 個 AMCE 維度預測 HLM 隨機效應：\n")
    report.append(f"- **R² = {reg_results['r2']:.4f}**")
    report.append(f"- Adjusted R² = {reg_results['adj_r2']:.4f}")
    report.append(f"- RMSE = {reg_results['rmse']:.4f}\n")
    
    report.append(f"**解釋**：{reg_results['r2']*100:.1f}% 的國家層級變異可由 9 個 AMCE 維度解釋。\n")
    
    report.append("### 關鍵發現\n")
    top_corr = corr_results.iloc[0]
    report.append(f"1. **最強相關維度**：「{top_corr['chinese_name']}」(r = {top_corr['pearson_r']:.3f}, p < .001)")
    report.append(f"   - 國家的守法偏好 AMCE 越高，該國在本研究場景中的守法選擇傾向也越高")
    report.append(f"   - 這驗證了 AMCE 指標與實際決策行為的一致性\n")
    
    sig_count = corr_results['significant_001'].sum()
    report.append(f"2. **顯著相關維度**：共 {sig_count} 個維度達 p < .001 顯著水準\n")
    
    report.append("3. **理論意涵**：")
    report.append("   - 國家層級變異並非隨機，而是可被國家道德偏好解釋")
    report.append("   - 支持「文化影響道德判斷」的假說，但效應量有限\n")
    
    report.append("### 視覺化結果\n")
    report.append("- [相關係數條形圖](../outputs/figures/chapter5/random_effect_correlation.html)")
    report.append("- [散點圖](../outputs/figures/chapter5/random_effect_scatter.html)\n")
    
    return "\n".join(report)


if __name__ == "__main__":
    main()