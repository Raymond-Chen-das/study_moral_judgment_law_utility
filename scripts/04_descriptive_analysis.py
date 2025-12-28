"""
04_descriptive_analysis.py
==========================
第四步：描述性分析（第3章）

功能：
1. 全球道德地圖（3.1）
2. 台灣與東亞定位（3.2）
3. 階層式分群（3.3）
4. 增加3.3補充：替代分群方法比較
5. 潛在類別分析（3.4）
6. 增加3.4補充：敏感度分析

執行方式：
    python scripts/04_descriptive_analysis.py
"""

import sys
from pathlib import Path

# 將專案根目錄加入路徑
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import logging
from datetime import datetime


def setup_file_logger(log_dir: str = 'outputs/logs') -> logging.Logger:
    """設定檔案日誌記錄器"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    log_file = log_path / 'descriptive_analysis.log'
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    return logger


def generate_markdown_report(results: dict, output_dir: str = 'report/drafts'):
    """生成Markdown格式的報告草稿"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_file = output_path / 'chapter3_exploration.md'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 第3章 探索性分析與類型學建構\n\n")
        f.write(f"**分析時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 3.1 全球道德地圖
        f.write("## 3.1 全球道德地圖\n\n")
        f.write("### 研究問題\n\n")
        f.write("- 全球在「守法vs.效益」衝突的分佈模式為何？\n")
        f.write("- 是否呈現地理聚集或文化聚集？\n\n")
        
        f.write("### 視覺化結果\n\n")
        if 'global' in results:
            f.write(f"- [世界地圖]({results['global']['world_map']})\n")
            f.write(f"- [文化圈比較]({results['global']['cluster_comparison']})\n")
            f.write(f"- [描述統計表]({results['global']['descriptive_stats']})\n\n")
        
        f.write("### 關鍵發現\n\n")
        f.write("（待填入：根據圖表結果撰寫）\n\n")
        
        # 3.2 台灣與東亞定位
        f.write("## 3.2 台灣與東亞的道德定位\n\n")
        f.write("### 研究問題\n\n")
        f.write("- 台灣在9個道德維度的表現為何？\n")
        f.write("- 與日本、韓國、中國大陸的異同？\n\n")
        
        f.write("### 視覺化結果\n\n")
        if 'east_asia' in results:
            f.write(f"- [東亞四國雷達圖]({results['east_asia']['radar_chart']})\n")
            f.write(f"- [距離熱圖]({results['east_asia']['distance_heatmap']})\n")
            f.write(f"- [比較表]({results['east_asia']['comparison_table']})\n\n")
        
        f.write("### 關鍵發現\n\n")
        f.write("（待填入：根據圖表結果撰寫）\n\n")
        
        # 3.3 階層式分群
        f.write("## 3.3 階層式分群：道德距離的拓撲結構\n\n")
        f.write("### 研究問題\n\n")
        f.write("- 基於道德判斷，國家如何自然分群？\n")
        f.write("- 是否存在超越地理的「道德親緣關係」？\n\n")
        
        f.write("### 視覺化結果\n\n")
        if 'hierarchical' in results:
            f.write(f"- [130國樹狀圖]({results['hierarchical']['dendrogram']})\n")
            f.write(f"- [道德距離熱圖]({results['hierarchical']['distance_heatmap']})\n\n")
            
            f.write("### 評估指標\n\n")
            f.write(f"- **Cophenetic Correlation**: {results['hierarchical']['cophenetic_correlation']:.4f}\n")
            f.write(f"- **Adjusted Rand Index**: {results['hierarchical']['ari']:.4f}\n\n")
        
        # 3.3補充
        if 'alternative_clustering' in results:
            f.write("### 3.3補充：替代分群方法比較\n\n")
            f.write(f"- [K-means評估]({results['alternative_clustering']['plots']['kmeans_evaluation']})\n")
            f.write(f"- [t-SNE視覺化]({results['alternative_clustering']['plots']['tsne_visualization']})\n")
            f.write(f"- [方法比較]({results['alternative_clustering']['plots']['methods_comparison']})\n\n")
        
        f.write("### 關鍵發現\n\n")
        f.write("（待填入：根據圖表結果撰寫）\n\n")
        
        # 3.4 潛在類別分析
        f.write("## 3.4 潛在類別分析：道德人格類型學\n\n")
        f.write("### 研究問題\n\n")
        f.write("- 是否存在不同的「道德決策模式」？\n")
        f.write("- 這些模式是否對應倫理理論？\n\n")
        
        f.write("### 視覺化結果\n\n")
        if 'lca' in results:
            f.write(f"- [BIC曲線]({results['lca']['bic_curve']})\n")
            f.write(f"- [類別雷達圖]({results['lca']['class_radar']})\n")
            f.write(f"- [文化分佈]({results['lca']['culture_distribution']})\n\n")
            
            f.write(f"### 最佳類別數: {results['lca']['optimal_k']}\n\n")
        
        # 3.4補充
        if 'lca_sensitivity' in results:
            f.write("### 3.4補充：敏感度分析\n\n")
            f.write(f"- [極端比例比較]({results['lca_sensitivity']['extreme_plot']})\n")
            f.write(f"- [詮釋報告](outputs/tables/chapter3/lca_sensitivity_interpretation.md)\n\n")
        
        f.write("### 關鍵發現\n\n")
        f.write("（待填入：根據圖表結果撰寫）\n\n")
        
        f.write("---\n\n")
        f.write("**註**: 本報告為自動生成的草稿，關鍵發現需根據圖表結果手動填寫。\n")
    
    print(f"✅ Markdown報告: {report_file}")


def main():
    """主執行函數"""
    print("\n" + "=" * 60)
    print("📊 MIT Moral Machine - 探索性分析 (Step 04)")
    print("=" * 60)
    
    # 設定日誌
    logger = setup_file_logger()
    logger.info("開始執行探索性分析腳本...")
    
    try:
        # 載入資料
        print("\n【載入資料】")
        featured_file = Path('data/processed/featured_data.csv')
        
        if not featured_file.exists():
            print(f"❌ 找不到檔案: {featured_file}")
            print("請先執行 03_feature_engineering.py")
            return
        
        df = pd.read_csv(featured_file)
        print(f"✅ 載入資料: {len(df):,} 行")
        
        # 篩選資料：排除 Cluster == -1
        if 'has_country_features' not in df.columns:
            print("⚠️  找不到 has_country_features 欄位，使用 Cluster != -1 篩選")
            df_ch3 = df[df['Cluster'] != -1].copy()
        else:
            df_ch3 = df[df['has_country_features']].copy()
        
        print(f"✅ 第3章分析資料: {len(df_ch3):,} 行")
        print(f"   已排除 {len(df) - len(df_ch3):,} 行 (Cluster == -1)")
        
        results = {}
        
        # ============================================
        # 3.1 全球道德地圖
        # ============================================
        print("\n" + "=" * 60)
        print("【3.1】全球道德地圖")
        print("=" * 60)
        
        from src.analysis.descriptive.global_patterns import GlobalPatternAnalyzer
        
        global_analyzer = GlobalPatternAnalyzer()
        results['global'] = global_analyzer.run_analysis(df_ch3)
        
        # ============================================
        # 3.2 台灣與東亞定位
        # ============================================
        print("\n" + "=" * 60)
        print("【3.2】台灣與東亞定位")
        print("=" * 60)
        
        from src.analysis.descriptive.east_asia_focus import EastAsiaAnalyzer
        
        east_asia_analyzer = EastAsiaAnalyzer()
        results['east_asia'] = east_asia_analyzer.run_analysis()
        
        # ============================================
        # 3.3 階層式分群
        # ============================================
        print("\n" + "=" * 60)
        print("【3.3】階層式分群")
        print("=" * 60)
        
        from src.analysis.clustering.hierarchical import HierarchicalClusterAnalyzer
        
        hierarchical_analyzer = HierarchicalClusterAnalyzer()
        results['hierarchical'] = hierarchical_analyzer.run_analysis()
        
        # ============================================
        # 3.4 潛在類別分析
        # ============================================
        print("\n" + "=" * 60)
        print("【3.4】潛在類別分析")
        print("=" * 60)
        
        from src.analysis.clustering.latent_class import LatentClassAnalyzer
        
        lca_analyzer = LatentClassAnalyzer()
        results['lca'] = lca_analyzer.run_analysis(df_ch3)
        
        # ============================================
        # 【新增】3.3補充：替代分群方法比較
        # ============================================
        RUN_ALTERNATIVE_CLUSTERING = True  # 設為False可跳過
        
        if RUN_ALTERNATIVE_CLUSTERING:
            print("\n" + "=" * 60)
            print("【3.3補充】替代分群方法比較")
            print("=" * 60)
            
            try:
                from src.analysis.clustering.alternative_methods import AlternativeClusteringAnalyzer
                
                alt_analyzer = AlternativeClusteringAnalyzer()
                results['alternative_clustering'] = alt_analyzer.run_full_analysis()
            except Exception as e:
                logger.warning(f"替代分群方法執行失敗: {e}")
                print(f"⚠️  替代分群方法跳過: {e}")
        
        # ============================================
        # 【新增】3.4補充：敏感度分析
        # ============================================
        RUN_SENSITIVITY_ANALYSIS = True  # 設為False可跳過
        
        if RUN_SENSITIVITY_ANALYSIS:
            print("\n" + "=" * 60)
            print("【3.4補充】LCA敏感度分析")
            print("=" * 60)
            
            try:
                results['lca_sensitivity'] = lca_analyzer.run_sensitivity_analysis()
            except Exception as e:
                logger.warning(f"敏感度分析執行失敗: {e}")
                print(f"⚠️  敏感度分析跳過: {e}")
        
        # ============================================
        # 生成報告
        # ============================================
        print("\n" + "=" * 60)
        print("【生成報告】")
        print("=" * 60)
        
        generate_markdown_report(results)
        
        # 完成
        print("\n" + "=" * 60)
        print("✅ 探索性分析完成！")
        print("=" * 60)
        print("\n📊 已產生以下輸出:")
        
        print("\n【3.1 全球道德地圖】")
        for key, value in results['global'].items():
            print(f"  - {key}: {value}")
        
        print("\n【3.2 台灣與東亞定位】")
        for key, value in results['east_asia'].items():
            print(f"  - {key}: {value}")
        
        print("\n【3.3 階層式分群】")
        print(f"  - dendrogram: {results['hierarchical']['dendrogram']}")
        print(f"  - distance_heatmap: {results['hierarchical']['distance_heatmap']}")
        print(f"  - Cophenetic Correlation: {results['hierarchical']['cophenetic_correlation']:.4f}")
        print(f"  - ARI: {results['hierarchical']['ari']:.4f}")
        
        # 【新增】顯示補充分析結果
        if 'alternative_clustering' in results:
            print("\n【3.3補充 替代分群方法】")
            print(f"  - K-means最佳k: {results['alternative_clustering']['kmeans']['best_k']}")
            print(f"  - 方法比較表: outputs/tables/chapter3/clustering_methods_comparison.csv")
            print(f"  - t-SNE視覺化: {results['alternative_clustering']['plots']['tsne_visualization']}")
        
        print("\n【3.4 潛在類別分析】")
        print(f"  - bic_curve: {results['lca']['bic_curve']}")
        print(f"  - class_radar: {results['lca']['class_radar']}")
        print(f"  - culture_distribution: {results['lca']['culture_distribution']}")
        print(f"  - 最佳類別數: {results['lca']['optimal_k']}")
        
        # 【新增】顯示敏感度分析結果
        if 'lca_sensitivity' in results:
            print("\n【3.4補充 敏感度分析】")
            print(f"  - 極端比例圖: {results['lca_sensitivity']['extreme_plot']}")
            print(f"  - 統計摘要: outputs/tables/chapter3/lca_sensitivity_summary.csv")
            print(f"  - 詮釋報告: outputs/tables/chapter3/lca_sensitivity_interpretation.md")
        
        print("\n📄 報告草稿:")
        print("  - report/drafts/chapter3_exploration.md")
        
        print("\n💡 下一步: 根據圖表結果填寫報告中的「關鍵發現」")
        print("=" * 60 + "\n")
        
        logger.info("探索性分析腳本執行完成")
        
    except Exception as e:
        logger.error(f"執行失敗: {e}", exc_info=True)
        print(f"\n❌ 錯誤: {e}")
        raise


if __name__ == '__main__':
    main()