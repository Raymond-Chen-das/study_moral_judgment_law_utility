"""
02_data_cleaning.py
===================
第二步：資料清理

功能：
1. 處理缺失值
2. 處理異常值
3. 篩選「守法vs.效益」衝突情境
4. 檢查場景完整性
5. 合併文化圈分類
6. 產生清理報告

執行方式：
    python scripts/02_data_cleaning.py
"""

import sys
from pathlib import Path

# 將專案根目錄加入路徑
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
import pandas as pd
import logging
from datetime import datetime
import json

def setup_file_logger(log_dir: str = 'outputs/logs') -> logging.Logger:
    """
    設定檔案日誌記錄器
    
    Parameters:
    -----------
    log_dir : str
        日誌目錄路徑
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    log_file = log_path / 'data_cleaning.log'
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

def save_cleaned_data(df: pd.DataFrame, output_dir: str = 'data/processed'):
    """
    儲存清理後的資料
    
    Parameters:
    -----------
    df : pd.DataFrame
        清理後的資料框
    output_dir : str
        輸出目錄
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / 'cleaned_survey.csv'
    
    print(f"\n儲存清理後的資料...")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    file_size_mb = output_file.stat().st_size / 1024**2
    print(f"✅ 已儲存: {output_file}")
    print(f"   檔案大小: {file_size_mb:.2f} MB")

def generate_cleaning_report(report: dict, output_dir: str = 'outputs/tables/chapter2'):
    """
    生成清理報告CSV
    
    Parameters:
    -----------
    report : dict
        清理報告字典
    output_dir : str
        輸出目錄
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n生成清理報告...")
    
    # 1. 各步驟統計表
    steps_data = []
    original = report['original_rows']
    
    steps_data.append({
        '步驟': '0. 原始資料',
        '行數': f"{original:,}",
        '刪除/過濾': '-',
        '保留比例': '100.00%'
    })
    
    for i, step in enumerate(report['steps'], 1):
        step_names = {
            'remove_missing_key_vars': f'{i}. 刪除關鍵變數缺失',
            'remove_outliers': f'{i}. 刪除異常值',
            'filter_law_vs_utility': f'{i}. 篩選守法vs.效益情境',
            'check_completeness': f'{i}. 檢查場景完整性',
            'merge_cluster': f'{i}. 合併文化圈分類'
        }
        
        step_name = step_names.get(step['step'], step['step'])
        remaining = step['remaining']
        removed = step.get('removed', 0)
        retention = (remaining / original * 100) if original > 0 else 0
        
        steps_data.append({
            '步驟': step_name,
            '行數': f"{remaining:,}",
            '刪除/過濾': f"{removed:,}" if removed > 0 else '-',
            '保留比例': f"{retention:.2f}%"
        })
    
    steps_df = pd.DataFrame(steps_data)
    steps_file = output_path / 'cleaning_steps_summary.csv'
    steps_df.to_csv(steps_file, index=False, encoding='utf-8-sig')
    print(f"✅ 清理步驟摘要: {steps_file}")
    
    # 2. 儲存完整報告為JSON
    json_file = output_path / 'cleaning_report.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"✅ 完整報告(JSON): {json_file}")

def generate_markdown_report(report: dict, 
                            cleaned_df: pd.DataFrame,
                            output_dir: str = 'report/drafts'):
    """
    生成Markdown格式的報告草稿
    
    Parameters:
    -----------
    report : dict
        清理報告
    cleaned_df : pd.DataFrame
        清理後的資料
    output_dir : str
        輸出目錄
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_file = output_path / 'chapter2_section2_data_cleaning.md'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 第2章 資料處理\n\n")
        f.write("## 2.2 資料清理\n\n")
        
        # 清理流程
        f.write("### 清理流程\n\n")
        f.write("本研究聚焦於「守法vs.效益」的道德兩難情境，")
        f.write("因此需要篩選符合此條件的場景，並處理資料品質問題。\n\n")
        
        # 清理步驟表格
        f.write("### 清理步驟摘要\n\n")
        f.write("| 步驟 | 行數 | 刪除/過濾 | 保留比例 |\n")
        f.write("|------|------|-----------|----------|\n")
        
        original = report['original_rows']
        f.write(f"| 原始資料 | {original:,} | - | 100.00% |\n")
        
        step_names = {
            'remove_missing_key_vars': '刪除關鍵變數缺失',
            'remove_outliers': '刪除異常值',
            'filter_law_vs_utility': '篩選守法vs.效益情境',
            'check_completeness': '檢查場景完整性',
            'merge_cluster': '合併文化圈分類'
        }
        
        for step in report['steps']:
            step_name = step_names.get(step['step'], step['step'])
            remaining = step['remaining']
            removed = step.get('removed', 0)
            retention = (remaining / original * 100) if original > 0 else 0
            
            removed_str = f"{removed:,}" if removed > 0 else "-"
            f.write(f"| {step_name} | {remaining:,} | {removed_str} | {retention:.2f}% |\n")
        
        # 各步驟詳細說明
        f.write("\n### 各步驟說明\n\n")
        
        f.write("#### Step 1: 刪除關鍵變數缺失\n\n")
        f.write("關鍵變數包括：`Saved`、`ScenarioType`、`UserCountry3`、`ResponseID`。")
        f.write("這些變數對於分析至關重要，缺失則無法進行後續分析。\n\n")
        
        step1 = next((s for s in report['steps'] if s['step'] == 'remove_missing_key_vars'), None)
        if step1:
            f.write(f"- **刪除**: {step1['removed']:,} 行 ({step1['removed_pct']})\n")
            f.write(f"- **剩餘**: {step1['remaining']:,} 行\n\n")
        
        f.write("#### Step 2: 刪除異常值\n\n")
        f.write("檢查以下變數的異常值：\n")
        f.write("- `Review_age`: 年齡範圍 [18, 75]\n")
        f.write("- `Review_political`: 政治立場 [0, 1]\n")
        f.write("- `Review_religious`: 宗教程度 [0, 1]\n\n")
        f.write("**處理策略**: 直接刪除超出合理範圍的資料，")
        f.write("避免填補可能引入的偏誤。\n\n")
        
        step2 = next((s for s in report['steps'] if s['step'] == 'remove_outliers'), None)
        if step2:
            f.write(f"- **刪除**: {step2['removed']:,} 行 ({step2['removed_pct']})\n")
            f.write(f"- **剩餘**: {step2['remaining']:,} 行\n\n")
        
        f.write("#### Step 3: 篩選「守法vs.效益」衝突情境\n\n")
        f.write("本研究聚焦於道德兩難情境，需同時滿足以下條件：\n\n")
        f.write("1. **ScenarioType = 'Utilitarian'**: 場景涉及人數差異\n")
        f.write("2. **CrossingSignal ∈ {1, 2}**: 有法律考量（綠燈合法或紅燈違法）\n")
        f.write("3. **DiffNumberOFCharacters > 0**: 兩側人數確實有差異\n\n")
        f.write("**篩選邏輯**: 只有當「守法」和「救多數」產生衝突時，")
        f.write("才構成真正的道德兩難。例如：\n\n")
        f.write("- ✅ **有衝突**: 3人闖紅燈 vs. 1人等綠燈\n")
        f.write("- ❌ **無衝突**: 5人等綠燈 vs. 3人闖紅燈（守法和救多數一致）\n\n")
        
        step3 = next((s for s in report['steps'] if s['step'] == 'filter_law_vs_utility'), None)
        if step3:
            f.write(f"- **保留**: {step3['remaining']:,} 行 ({step3['remaining_pct']})\n")
            f.write(f"- **過濾**: {step3['removed']:,} 行\n\n")
        
        f.write("#### Step 4: 檢查場景完整性\n\n")
        f.write("每個場景（`ResponseID`）應有2行資料，代表兩個可能的結果。")
        f.write("刪除只有1行的不完整場景。\n\n")
        
        step4 = next((s for s in report['steps'] if s['step'] == 'check_completeness'), None)
        if step4:
            f.write(f"- **刪除**: {step4['removed']:,} 行不完整場景 ({step4['removed_pct']})\n")
            f.write(f"- **剩餘**: {step4['remaining']:,} 行\n")
            if 'complete_scenarios' in step4:
                f.write(f"- **完整場景數**: {step4['complete_scenarios']:,}\n\n")
        
        f.write("#### Step 5: 合併文化圈分類\n\n")
        f.write("將 `country_cluster_map.csv` 的文化圈資訊合併到主資料，")
        f.write("便於後續跨文化比較分析。\n\n")
        
        # 文化圈分佈
        if 'Cluster' in cleaned_df.columns:
            cluster_dist = cleaned_df['Cluster'].value_counts().sort_index()
            f.write("**文化圈分佈**:\n\n")
            cluster_names = {-1: 'Unclassified (未分類小國)', 0: 'Western', 1: 'Eastern', 2: 'Southern'}
            for cluster, count in cluster_dist.items():
                if pd.notna(cluster):
                    cluster_name = cluster_names.get(int(cluster), f'Cluster {int(cluster)}')
                    pct = count / len(cleaned_df) * 100
                    f.write(f"- {cluster_name}: {count:,} 行 ({pct:.1f}%)\n")
        
        # 最終結果
        f.write("\n### 最終結果\n\n")
        final = report['steps'][-1]['remaining']
        retention = (final / original * 100) if original > 0 else 0
        
        f.write(f"- **原始資料**: {original:,} 行\n")
        f.write(f"- **清理後**: {final:,} 行\n")
        f.write(f"- **保留比例**: {retention:.2f}%\n\n")
        
        f.write("清理後的資料聚焦於「守法vs.效益」衝突情境，")
        f.write("可進行後續的描述性分析、統計推論與預測建模。\n\n")
    
    print(f"✅ Markdown報告: {report_file}")

def analyze_cleaned_data(df: pd.DataFrame, output_dir: str = 'outputs/tables/chapter2'):
    """
    分析清理後的資料特性
    
    Parameters:
    -----------
    df : pd.DataFrame
        清理後的資料
    output_dir : str
        輸出目錄
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n分析清理後的資料...")
    
    # 1. 國家分佈
    if 'UserCountry3' in df.columns:
        country_dist = df['UserCountry3'].value_counts().head(20).reset_index()
        country_dist.columns = ['國家代碼', '決策數量']
        country_dist['比例'] = (country_dist['決策數量'] / len(df) * 100).round(2).astype(str) + '%'
        
        country_file = output_path / 'cleaned_top20_countries.csv'
        country_dist.to_csv(country_file, index=False, encoding='utf-8-sig')
        print(f"✅ 前20名國家分佈: {country_file}")
    
    # 2. 場景數量統計
    if 'ResponseID' in df.columns:
        n_scenarios = df['ResponseID'].nunique()
        n_users = df['UserID'].nunique() if 'UserID' in df.columns else 'N/A'
        
        scenario_stats = pd.DataFrame([{
            '指標': '完整場景數',
            '數值': f"{n_scenarios:,}"
        }, {
            '指標': '使用者數',
            '數值': f"{n_users:,}" if isinstance(n_users, int) else n_users
        }, {
            '指標': '平均每場景資料行數',
            '數值': f"{len(df) / n_scenarios:.2f}" if n_scenarios > 0 else 'N/A'
        }])
        
        stats_file = output_path / 'cleaned_data_stats.csv'
        scenario_stats.to_csv(stats_file, index=False, encoding='utf-8-sig')
        print(f"✅ 基本統計: {stats_file}")

def main():
    """主執行函數"""
    print("\n" + "=" * 60)
    print("📋 MIT Moral Machine - 資料清理 (Step 02)")
    print("=" * 60)
    
    # 設定檔案日誌
    logger = setup_file_logger()
    logger.info("開始執行資料清理腳本...")
    
    try:
        # Step 1: 載入資料
        print("\n【Step 1】載入資料...")
        loader = DataLoader(data_dir='data/raw')
        
        # 檢查檔案
        files_status = loader.check_files_exist()
        if not all(files_status.values()):
            print("\n❌ 錯誤: 部分資料檔案缺失")
            return
        
        # 載入必要的資料
        print("\n載入問卷資料...")
        survey_df = loader.load_survey_data(nrows=None)
        
        print("\n載入文化圈分類...")
        cluster_map_df = loader.load_cluster_map()
        
        # Step 2: 清理資料
        print("\n【Step 2】清理資料...")
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean_data(survey_df, cluster_map_df)
        
        # Step 3: 儲存清理後的資料
        print("\n【Step 3】儲存清理後的資料...")
        save_cleaned_data(cleaned_df)
        
        # Step 4: 生成報告
        print("\n【Step 4】生成清理報告...")
        report = cleaner.get_cleaning_report()
        generate_cleaning_report(report)
        generate_markdown_report(report, cleaned_df)
        analyze_cleaned_data(cleaned_df)
        
        # 完成
        print("\n" + "=" * 60)
        print("✅ 資料清理完成！")
        print("=" * 60)
        print("\n📊 已產生以下輸出:")
        print("  - data/processed/cleaned_survey.csv")
        print("  - outputs/logs/data_cleaning.log")
        print("  - outputs/tables/chapter2/cleaning_steps_summary.csv")
        print("  - outputs/tables/chapter2/cleaning_report.json")
        print("  - outputs/tables/chapter2/cleaned_top20_countries.csv")
        print("  - outputs/tables/chapter2/cleaned_data_stats.csv")
        print("  - report/drafts/chapter2_section2_data_cleaning.md")
        print("\n💡 下一步: python scripts/03_feature_engineering.py")
        print("=" * 60 + "\n")
        
        logger.info("資料清理腳本執行完成")
        
    except Exception as e:
        logger.error(f"執行失敗: {e}", exc_info=True)
        print(f"\n❌ 錯誤: {e}")
        raise

if __name__ == '__main__':
    main()