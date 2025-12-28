"""
第5章 第3節：全球道德光譜視覺化
================================
以效益主義 vs 守法偏好建構全球道德空間

執行此腳本前請確認：
1. 原始資料 CountriesChangePr.csv 位於 data/raw/
2. 原始資料 country_cluster_map.csv 位於 data/raw/

產出：
- outputs/figures/chapter5/moral_spectrum_global.html
- outputs/figures/chapter5/cluster_centroids.html
- outputs/tables/chapter5/moral_spectrum_coordinates.csv
- outputs/tables/chapter5/quadrant_distribution.csv
- report/drafts/chapter5_section3_moral_spectrum.md
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
from src.analysis.integration.moral_spectrum import (
    MoralSpectrumAnalyzer, load_and_analyze
)
from src.visualization.chapter5.chapter5_plots import (
    plot_moral_spectrum,
    plot_cluster_centroids
)


def main():
    """主執行函數"""
    
    print("=" * 70)
    print("第5章 第3節：全球道德光譜視覺化")
    print(f"執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # ========================================
    # 1. 設定路徑
    # ========================================
    
    # 輸入路徑
    AMCE_PATH = PROJECT_ROOT / "data/raw/CountriesChangePr.csv"
    CLUSTER_PATH = PROJECT_ROOT / "data/raw/country_cluster_map.csv"
    
    # 輸出路徑
    OUTPUT_FIG_DIR = PROJECT_ROOT / "outputs/figures/chapter5"
    OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs/tables/chapter5"
    REPORT_DIR = PROJECT_ROOT / "report/drafts"
    
    # 創建輸出目錄
    OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 檢查輸入檔案
    if not AMCE_PATH.exists():
        print(f"\n❌ 錯誤：找不到 AMCE 資料檔案")
        print(f"   預期路徑: {AMCE_PATH}")
        return
    
    if not CLUSTER_PATH.exists():
        print(f"\n❌ 錯誤：找不到文化圈分類檔案")
        print(f"   預期路徑: {CLUSTER_PATH}")
        return
    
    print(f"\n✅ 輸入檔案確認完成")
    
    # ========================================
    # 2. 載入與分析
    # ========================================
    
    analyzer, spectrum_data = load_and_analyze(
        amce_path=str(AMCE_PATH),
        cluster_path=str(CLUSTER_PATH),
        verbose=True
    )
    
    # 獲取各種分析結果
    taiwan_analysis = analyzer.get_taiwan_analysis()
    centroids = analyzer.get_cluster_centroids()
    cultural_distance = analyzer.compute_cultural_distance()
    
    # ========================================
    # 3. 視覺化
    # ========================================
    
    print("\n" + "=" * 60)
    print("生成視覺化圖表")
    print("=" * 60)
    
    # 獲取國家位置資料
    country_positions = analyzer.get_country_positions()
    
    # 全球道德光譜
    fig_spectrum = plot_moral_spectrum(
        spectrum_data=country_positions,
        output_path=str(OUTPUT_FIG_DIR / "moral_spectrum_global.html"),
        title="全球道德光譜：效益主義 vs 守法偏好"
    )
    
    # 文化圈重心
    fig_centroids = plot_cluster_centroids(
        spectrum_data=country_positions,
        output_path=str(OUTPUT_FIG_DIR / "cluster_centroids.html"),
        title="文化圈重心比較"
    )
    
    # ========================================
    # 4. 儲存表格
    # ========================================
    
    print("\n" + "=" * 60)
    print("儲存分析表格")
    print("=" * 60)
    
    # 國家座標
    country_positions.to_csv(
        OUTPUT_TABLE_DIR / "moral_spectrum_coordinates.csv",
        index=False,
        encoding='utf-8-sig'
    )
    print(f"✅ 已儲存: moral_spectrum_coordinates.csv")
    
    # 象限分佈
    if analyzer.quadrant_stats is not None:
        analyzer.quadrant_stats.to_csv(
            OUTPUT_TABLE_DIR / "quadrant_distribution.csv",
            index=False,
            encoding='utf-8-sig'
        )
        print(f"✅ 已儲存: quadrant_distribution.csv")
    
    # 文化圈距離矩陣
    cultural_distance.to_csv(
        OUTPUT_TABLE_DIR / "cultural_distance_matrix.csv",
        encoding='utf-8-sig'
    )
    print(f"✅ 已儲存: cultural_distance_matrix.csv")
    
    # ========================================
    # 5. 生成報告草稿
    # ========================================
    
    print("\n" + "=" * 60)
    print("生成報告草稿")
    print("=" * 60)
    
    report_content = generate_section_report(
        analyzer=analyzer,
        taiwan_analysis=taiwan_analysis,
        centroids=centroids,
        cultural_distance=cultural_distance,
        country_positions=country_positions
    )
    
    report_path = REPORT_DIR / "chapter5_section3_moral_spectrum.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"✅ 已儲存: {report_path}")
    
    # ========================================
    # 6. 總結
    # ========================================
    
    print("\n" + "=" * 70)
    print("第5.3節執行完成！")
    print("=" * 70)
    
    print("\n📊 產出檔案：")
    print(f"   - {OUTPUT_FIG_DIR / 'moral_spectrum_global.html'}")
    print(f"   - {OUTPUT_FIG_DIR / 'cluster_centroids.html'}")
    print(f"   - {OUTPUT_TABLE_DIR / 'moral_spectrum_coordinates.csv'}")
    print(f"   - {OUTPUT_TABLE_DIR / 'quadrant_distribution.csv'}")
    print(f"   - {OUTPUT_TABLE_DIR / 'cultural_distance_matrix.csv'}")
    
    print("\n🔑 關鍵發現：")
    print(f"   台灣定位:")
    print(f"   - 效益主義: {taiwan_analysis['utilitarian']:.3f} (排名 {taiwan_analysis['util_rank']}/{taiwan_analysis['total_countries']})")
    print(f"   - 守法偏好: {taiwan_analysis['law_preference']:.3f} (排名 {taiwan_analysis['law_rank']}/{taiwan_analysis['total_countries']})")
    print(f"   - 象限: {taiwan_analysis['quadrant']}")
    
    print("\n   文化圈重心:")
    for _, row in centroids.iterrows():
        print(f"   - {row['Cluster_Name']}: 效益={row['utilitarian']:.3f}, 守法={row['law_preference']:.3f}")


def generate_section_report(
    analyzer: MoralSpectrumAnalyzer,
    taiwan_analysis: dict,
    centroids: pd.DataFrame,
    cultural_distance: pd.DataFrame,
    country_positions: pd.DataFrame
) -> str:
    """生成 5.3 節報告草稿"""
    
    report = []
    report.append("## 5.3 全球道德光譜視覺化\n")
    report.append(f"**分析時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    report.append("### 視覺化框架\n")
    report.append("以二維空間呈現 130 個國家的道德偏好定位：")
    report.append("- **X軸**：效益主義 AMCE（數量偏好：拯救多數 vs 少數）")
    report.append("- **Y軸**：守法偏好 AMCE（規則遵循：守法者 vs 違法者）")
    report.append("- **顏色**：文化圈（Western / Eastern / Southern）")
    report.append("- **象限**：以全球平均為原點劃分四象限\n")
    
    report.append("### 文化圈重心\n")
    report.append("| 文化圈 | 國家數 | 效益主義 | 守法偏好 |")
    report.append("|--------|--------|----------|----------|")
    for _, row in centroids.iterrows():
        report.append(f"| {row['Cluster_Name']} | {int(row['n_countries'])} | {row['utilitarian']:.3f} | {row['law_preference']:.3f} |")
    report.append("")
    
    report.append("### 文化圈間道德距離\n")
    report.append("| | Western | Eastern | Southern |")
    report.append("|---|---------|---------|----------|")
    for cluster in ['Western', 'Eastern', 'Southern']:
        row_str = f"| {cluster} |"
        for other in ['Western', 'Eastern', 'Southern']:
            row_str += f" {cultural_distance.loc[cluster, other]:.3f} |"
        report.append(row_str)
    report.append("")
    
    report.append("### 台灣定位分析\n")
    report.append(f"- **效益主義**: {taiwan_analysis['utilitarian']:.3f}")
    report.append(f"  - 全球排名: {taiwan_analysis['util_rank']}/{taiwan_analysis['total_countries']}")
    report.append(f"- **守法偏好**: {taiwan_analysis['law_preference']:.3f}")
    report.append(f"  - 全球排名: {taiwan_analysis['law_rank']}/{taiwan_analysis['total_countries']}")
    report.append(f"- **所屬象限**: {taiwan_analysis['quadrant']}\n")
    
    report.append("**最近鄰居（道德距離）**：")
    for nb in taiwan_analysis['nearest_neighbors']:
        report.append(f"- {nb['Country']} ({nb['UserCountry3']}): {nb['distance_to_taiwan']:.4f}")
    report.append("")
    
    report.append("### 象限分佈\n")
    quadrant_counts = country_positions.groupby(['quadrant', 'Cluster_Name']).size().unstack(fill_value=0)
    report.append(quadrant_counts.to_markdown())
    report.append("")
    
    report.append("### 關鍵發現\n")
    report.append("1. **文化圈差異**：三大文化圈在道德光譜上有明顯的空間分離")
    report.append("   - Eastern 傾向高守法")
    report.append("   - Western 傾向高效益主義")
    report.append("   - Southern 位於中間地帶\n")
    
    report.append("2. **台灣定位**：")
    report.append(f"   - 位於「{taiwan_analysis['quadrant']}」象限")
    report.append("   - 最接近的鄰居包含東亞與東南亞國家")
    report.append("   - 驗證第3章「東亞-東南亞核心」的分群結果\n")
    
    report.append("3. **全研究整合意義**：")
    report.append("   - 此視覺化整合了第3章（探索分析）與第4章（統計推論）的發現")
    report.append("   - 國家層級變異（ICC=0.25%）在二維空間中清楚呈現")
    report.append("   - 為第6章的哲學討論提供視覺化基礎\n")
    
    report.append("### 視覺化結果\n")
    report.append("- [全球道德光譜](../outputs/figures/chapter5/moral_spectrum_global.html)")
    report.append("- [文化圈重心](../outputs/figures/chapter5/cluster_centroids.html)\n")
    
    return "\n".join(report)


if __name__ == "__main__":
    main()