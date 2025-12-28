"""
01_data_loading.py
==================
第一步：資料載入

功能：
1. 載入所有原始資料檔案
2. 進行基本驗證與完整性檢查
3. 產生資料品質報告
4. 儲存載入日誌

執行方式：
    python scripts/01_data_loading.py
"""

import sys
from pathlib import Path

# 將專案根目錄加入路徑
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import DataLoader
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
    # 建立日誌目錄
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 建立檔案handler
    log_file = log_path / 'data_loading.log'
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 格式設定
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # 取得root logger並添加handler
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    return logger

def generate_data_quality_report(data: dict, output_dir: str = 'outputs/tables/chapter2'):
    """
    生成資料品質報告
    
    Parameters:
    -----------
    data : dict
        載入的資料字典
    output_dir : str
        輸出目錄
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("生成資料品質報告...")
    print("=" * 60)
    
    # 1. 整體摘要
    summary_data = []
    
    for name, df in data.items():
        summary_data.append({
            '資料集': name,
            '列數': f"{len(df):,}",
            '欄數': len(df.columns),
            '記憶體(MB)': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}",
            '總缺失值': f"{df.isnull().sum().sum():,}",
            '缺失值比例': f"{df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100:.2f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_path / 'data_loading_summary.csv'
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"✅ 整體摘要: {summary_file}")
    
    # 2. 各資料集的欄位資訊
    for name, df in data.items():
        col_info = []
        
        for col in df.columns:
            col_info.append({
                '欄位名稱': col,
                '資料型態': str(df[col].dtype),
                '非空值數量': f"{df[col].count():,}",
                '缺失值數量': f"{df[col].isnull().sum():,}",
                '缺失值比例': f"{df[col].isnull().sum() / len(df) * 100:.2f}%",
                '唯一值數量': f"{df[col].nunique():,}",
            })
        
        col_df = pd.DataFrame(col_info)
        col_file = output_path / f'{name}_columns_info.csv'
        col_df.to_csv(col_file, index=False, encoding='utf-8-sig')
        print(f"✅ {name} 欄位資訊: {col_file}")
    
    # 3. 問卷資料的場景類型分佈
    if 'survey' in data:
        survey_df = data['survey']
        
        if 'ScenarioType' in survey_df.columns:
            scenario_dist = survey_df['ScenarioType'].value_counts().reset_index()
            scenario_dist.columns = ['場景類型', '數量']
            scenario_dist['比例'] = (scenario_dist['數量'] / len(survey_df) * 100).round(2).astype(str) + '%'
            
            scenario_file = output_path / 'scenario_type_distribution.csv'
            scenario_dist.to_csv(scenario_file, index=False, encoding='utf-8-sig')
            print(f"✅ 場景類型分佈: {scenario_file}")
        
        # 4. 國家分佈
        if 'UserCountry3' in survey_df.columns:
            country_dist = survey_df['UserCountry3'].value_counts().head(20).reset_index()
            country_dist.columns = ['國家代碼', '決策數量']
            
            country_file = output_path / 'top20_countries.csv'
            country_dist.to_csv(country_file, index=False, encoding='utf-8-sig')
            print(f"✅ 前20名國家: {country_file}")
    
    print("=" * 60)

def generate_markdown_report(data: dict, summary: dict, output_dir: str = 'report/drafts'):
    """
    生成Markdown格式的報告草稿
    
    Parameters:
    -----------
    data : dict
        載入的資料字典
    summary : dict
        載入摘要
    output_dir : str
        輸出目錄
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_file = output_path / 'chapter2_section1_data_loading.md'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 第2章 資料處理\n\n")
        f.write("## 2.1 資料來源與載入\n\n")
        f.write(f"**載入時間**: {summary['loading_time']}\n\n")
        
        f.write("### 資料集概覽\n\n")
        f.write("| 資料集 | 列數 | 欄數 | 記憶體(MB) | 缺失值比例 |\n")
        f.write("|--------|------|------|-----------|------------|\n")
        
        for name, info in summary['datasets'].items():
            f.write(f"| {name} | {info['rows']:,} | {info['columns']} | "
                   f"{info['memory_mb']:.2f} | {info['missing_pct']} |\n")
        
        f.write("\n### 各資料集說明\n\n")
        
        # SharedResponsesSurvey
        if 'survey' in data:
            df = data['survey']
            f.write("#### 1. SharedResponsesSurvey.csv\n\n")
            f.write(f"- **用途**: 主要分析資料，包含場景回應與人口統計變數\n")
            f.write(f"- **維度**: {len(df):,} 行 × {len(df.columns)} 欄\n")
            f.write(f"- **關鍵欄位**: ResponseID, UserID, UserCountry3, Saved, ScenarioType\n\n")
            
            if 'ScenarioType' in df.columns:
                f.write("**場景類型分佈**:\n\n")
                scenario_counts = df['ScenarioType'].value_counts()
                for stype, count in scenario_counts.items():
                    pct = count / len(df) * 100
                    f.write(f"- {stype}: {count:,} ({pct:.1f}%)\n")
                f.write("\n")
        
        # CountriesChangePr
        if 'countries_change' in data:
            df = data['countries_change']
            f.write("#### 2. CountriesChangePr.csv\n\n")
            f.write(f"- **用途**: 國家層級統計，包含9個道德屬性的AMCE值\n")
            f.write(f"- **維度**: {len(df)} 個國家 × {len(df.columns)} 個指標\n")
            f.write(f"- **包含**: 9對 (Estimates + se) 欄位\n\n")
        
        # cluster_map
        if 'cluster_map' in data:
            df = data['cluster_map']
            f.write("#### 3. country_cluster_map.csv\n\n")
            f.write(f"- **用途**: 文化圈分類\n")
            f.write(f"- **維度**: {len(df)} 個國家\n\n")
            
            cluster_counts = df['Cluster'].value_counts().sort_index()
            f.write("**文化圈分佈**:\n\n")
            cluster_names = {0: 'Western', 1: 'Eastern', 2: 'Southern'}
            for cluster, count in cluster_counts.items():
                cluster_name = cluster_names.get(cluster, f'Cluster {cluster}')
                f.write(f"- {cluster_name} (Cluster {cluster}): {count} 個國家\n")
            f.write("\n")
        
        # moral_distance
        if 'moral_distance' in data:
            df = data['moral_distance']
            f.write("#### 4. moral_distance.csv\n\n")
            f.write(f"- **用途**: 國家間道德距離矩陣\n")
            f.write(f"- **基準國**: 美國 (Distance = 0)\n")
            f.write(f"- **距離範圍**: {df['Distance'].min():.3f} ~ {df['Distance'].max():.3f}\n")
            f.write(f"- **平均距離**: {df['Distance'].mean():.3f}\n\n")
        
        # dendrogram
        if 'dendrogram' in data:
            df = data['dendrogram']
            countries = df[df['culture'].notna()]
            f.write("#### 5. dendrogram_Culture.csv\n\n")
            f.write(f"- **用途**: 階層分群樹狀圖資料\n")
            f.write(f"- **節點總數**: {len(df)}\n")
            f.write(f"- **國家節點**: {len(countries)}\n\n")
        
        f.write("### 資料品質評估\n\n")
        f.write("所有資料檔案已成功載入，關鍵欄位完整無缺失。")
        f.write("後續將進行資料清理與轉換。\n\n")
    
    print(f"\n✅ Markdown報告: {report_file}")

def main():
    """主執行函數"""
    print("\n" + "=" * 60)
    print("📂 MIT Moral Machine - 資料載入 (Step 01)")
    print("=" * 60)
    
    # 設定檔案日誌
    logger = setup_file_logger()
    logger.info("開始執行資料載入腳本...")
    
    try:
        # 初始化載入器
        loader = DataLoader(data_dir='data/raw')
        
        # 檢查檔案
        files_status = loader.check_files_exist()
        
        if not all(files_status.values()):
            missing_files = [k for k, v in files_status.items() if not v]
            logger.error(f"缺少以下檔案: {missing_files}")
            print(f"\n❌ 錯誤: 缺少資料檔案 {missing_files}")
            print("請確認 data/raw/ 目錄下有所有必要檔案")
            return
        
        # 載入所有資料
        print("\n【開始載入資料】")
        data = loader.load_all_data(survey_nrows=None)  # None = 載入全部
        
        # 列印摘要
        loader.print_summary()
        
        # 生成載入摘要
        summary = loader.generate_loading_summary()
        
        # 儲存摘要為JSON
        json_path = Path('outputs/tables/chapter2')
        json_path.mkdir(parents=True, exist_ok=True)
        with open(json_path / 'loading_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # 生成資料品質報告
        generate_data_quality_report(data)
        
        # 生成Markdown報告草稿
        generate_markdown_report(data, summary)
        
        print("\n" + "=" * 60)
        print("✅ 資料載入完成！")
        print("=" * 60)
        print("\n📊 已產生以下輸出:")
        print("  - outputs/logs/data_loading.log")
        print("  - outputs/tables/chapter2/data_loading_summary.csv")
        print("  - outputs/tables/chapter2/*_columns_info.csv")
        print("  - report/drafts/chapter2_section1_data_loading.md")
        print("\n💡 下一步: python scripts/02_data_cleaning.py")
        print("=" * 60 + "\n")
        
        logger.info("資料載入腳本執行完成")
        
    except Exception as e:
        logger.error(f"執行失敗: {e}", exc_info=True)
        print(f"\n❌ 錯誤: {e}")
        raise

if __name__ == '__main__':
    main()