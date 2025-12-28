"""
03_feature_engineering.py
==========================
第三步：特徵工程

功能：
1. 建立場景層級特徵（守法、多數、衝突等）
2. ⚠️ 關鍵過濾：只保留「守法 vs. 效益」衝突場景
3. 建立使用者道德側寫
4. 分割訓練/測試集
5. 產生特徵說明文件

執行方式：
    python scripts/03_feature_engineering.py
"""

import sys
from pathlib import Path

# 將專案根目錄加入路徑
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.feature_engineer import FeatureEngineer
import pandas as pd
import logging
from datetime import datetime
import json

def setup_file_logger(log_dir: str = 'outputs/logs') -> logging.Logger:
    """設定檔案日誌記錄器"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    log_file = log_path / 'feature_engineering.log'
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

def save_featured_data(df: pd.DataFrame, output_dir: str = 'data/processed'):
    """儲存增加特徵的資料"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / 'featured_data.csv'
    
    print(f"\n儲存特徵化資料...")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    file_size_mb = output_file.stat().st_size / 1024**2
    print(f"✅ 已儲存: {output_file}")
    print(f"   檔案大小: {file_size_mb:.2f} MB")
    print(f"   欄位數: {len(df.columns)}")

def save_user_profiles(profiles_df: pd.DataFrame, output_dir: str = 'data/processed'):
    """儲存使用者道德側寫"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / 'user_moral_profiles.csv'
    
    print(f"\n儲存使用者道德側寫...")
    profiles_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"✅ 已儲存: {output_file}")
    print(f"   使用者數: {len(profiles_df):,}")

def save_train_test_split(train_df: pd.DataFrame, 
                          test_df: pd.DataFrame,
                          output_dir: str = 'data/processed'):
    """儲存訓練/測試集"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n儲存訓練/測試集...")
    
    # 儲存訓練集
    train_file = output_path / 'train_data.csv'
    train_df.to_csv(train_file, index=False, encoding='utf-8-sig')
    print(f"✅ 訓練集: {train_file}")
    print(f"   {len(train_df):,} 行")
    
    # 儲存測試集
    test_file = output_path / 'test_data.csv'
    test_df.to_csv(test_file, index=False, encoding='utf-8-sig')
    print(f"✅ 測試集: {test_file}")
    print(f"   {len(test_df):,} 行")
    
    # 儲存分割索引
    split_index = {
        'train_users': train_df['UserID'].unique().tolist(),
        'test_users': test_df['UserID'].unique().tolist(),
        'split_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_type': 'conflict_only'
    }
    
    index_file = output_path / 'train_test_split.json'
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(split_index, f, ensure_ascii=False, indent=2)

def save_feature_descriptions(descriptions: dict, output_dir: str = 'outputs/tables/chapter2'):
    """儲存特徵說明文件"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    desc_df = pd.DataFrame([
        {'特徵名稱': name, '說明': desc}
        for name, desc in descriptions.items()
    ])
    
    csv_file = output_path / 'feature_descriptions.csv'
    desc_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 特徵說明文件: {csv_file}")

def generate_feature_statistics(df: pd.DataFrame, output_dir: str = 'outputs/tables/chapter2'):
    """生成特徵統計報告"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    scenario_stats = []
    feature_cols = ['is_lawful', 'is_majority', 'chose_lawful', 
                   'chose_majority', 'lawful_vs_majority_conflict']
    
    for col in feature_cols:
        if col in df.columns:
            scenario_stats.append({
                '特徵': col,
                '平均值': f"{df[col].mean():.3f}",
                '標準差': f"{df[col].std():.3f}",
                '最小值': int(df[col].min()),
                '最大值': int(df[col].max()),
                '總和': f"{df[col].sum():,}"
            })
    
    stats_df = pd.DataFrame(scenario_stats)
    stats_file = output_path / 'scenario_feature_stats.csv'
    stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
    print(f"✅ 場景特徵統計: {stats_file}")

def generate_markdown_report(df: pd.DataFrame,
                            profiles_df: pd.DataFrame,
                            descriptions: dict,
                            original_count: int,
                            output_dir: str = 'report/drafts'):
    """生成Markdown格式的報告草稿"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_file = output_path / 'chapter2_section3_feature_engineering.md'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 第2章 資料處理\n\n")
        f.write("## 2.3 特徵工程與場景篩選\n\n")
        
        f.write("### 核心目標\n\n")
        f.write("本步驟將資料**嚴格限縮於「道德兩難」情境**，並建立相關特徵。\n")
        f.write("為了確保後續分析與模型預測的有效性，我們移除了所有「非衝突」場景（即守法與救多數一致的送分題）。\n\n")
        
        f.write("### 資料過濾結果\n\n")
        filtered_count = len(df)
        removed_count = original_count - filtered_count
        
        f.write(f"- **原始資料量**: {original_count:,} 行\n")
        f.write(f"- **過濾後資料量**: {filtered_count:,} 行 (僅包含衝突場景)\n")
        f.write(f"- **移除資料量**: {removed_count:,} 行 ({removed_count/original_count*100:.1f}%)\n")
        f.write("- **篩選標準**: `lawful_vs_majority_conflict == 1`\n\n")
        
        f.write("### 場景層級特徵統計\n\n")
        f.write("由於資料已篩選為衝突場景，`lawful_vs_majority_conflict` 的平均值為 1.0。\n\n")
        f.write("| 特徵名稱 | 說明 | 平均值 |\n")
        f.write("|---------|------|--------|\n")
        
        scenario_features = ['is_lawful', 'is_majority', 'chose_lawful', 'chose_majority']
        
        for feat in scenario_features:
            if feat in df.columns:
                desc = descriptions.get(feat, '')
                mean_val = df[feat].mean()
                f.write(f"| {feat} | {desc} | {mean_val:.3f} |\n")
        
        f.write("\n#### 關鍵發現\n\n")
        if 'chose_lawful' in df.columns and 'Saved' in df.columns:
            chose_lawful_rate = df[df['Saved'] == 1]['is_lawful'].mean()
            f.write(f"- **真實守法選擇率**: {chose_lawful_rate*100:.1f}%\n")
            f.write("  （註：此數值反映在必須犧牲多數人時，選擇守法的比例）\n")
        
        # 國家層級特徵
        f.write("\n### 國家層級特徵\n\n")
        f.write("已整合 `CountriesChangePr.csv` 的 AMCE 值，用於後續階層模型分析。\n")
        
        # 使用者側寫
        f.write("\n### 使用者道德側寫\n\n")
        f.write(f"- **側寫使用者數**: {len(profiles_df):,} 位\n")
        f.write(f"- **平均完成衝突場景數**: {profiles_df['n_scenarios'].mean():.1f} 個\n\n")
        
        if 'utilitarian_score' in profiles_df.columns:
            strong_util = (profiles_df['utilitarian_score'] > 0.7).sum()
            f.write("**效益主義傾向分佈**:\n")
            f.write(f"- 強效益主義 (>0.7): {strong_util:,} 位 ({strong_util/len(profiles_df)*100:.1f}%)\n")

        f.write("\n### 結論\n\n")
        f.write("本資料集現已準備好進行分析。所有後續章節（探索性分析、推論統計、預測模型）\n")
        f.write("都將基於此「純衝突」資料集進行，確保研究聚焦於真實的道德權衡。\n")
    
    print(f"✅ Markdown報告: {report_file}")

def main():
    """主執行函數"""
    print("\n" + "=" * 60)
    print("🔧 MIT Moral Machine - 特徵工程 (Step 03)")
    print("=" * 60)
    
    logger = setup_file_logger()
    logger.info("開始執行特徵工程腳本...")
    
    try:
        # Step 1: 載入資料
        print("\n【Step 1】載入資料...")
        cleaned_file = Path('data/processed/cleaned_survey.csv')
        if not cleaned_file.exists():
            print(f"❌ 錯誤: 找不到 {cleaned_file}，請先執行 02_data_cleaning.py")
            return
        
        df = pd.read_csv(cleaned_file)
        print(f"✅ 載入資料: {len(df):,} 行")
        
        countries_file = Path('data/raw/CountriesChangePr.csv')
        countries_df = pd.read_csv(countries_file) if countries_file.exists() else None
        
        # Step 2: 建立特徵
        print("\n【Step 2】建立場景特徵...")
        engineer = FeatureEngineer()
        df_featured = engineer.engineer_features(df)
        
        # Step 3: 合併國家特徵
        if countries_df is not None:
            print("\n【Step 3】合併國家特徵...")
            df_featured = engineer.merge_country_features(df_featured, countries_df)
            df_featured = engineer.add_feature_availability_flag(df_featured)
        
        # ==========================================
        # 🟢 Step 3.5: 關鍵過濾
        # ==========================================
        print("\n【Step 3.5】⚠️  強制篩選：只保留「守法vs.效益」衝突場景...")
        n_before = len(df_featured)
        
        if 'lawful_vs_majority_conflict' in df_featured.columns:
            # 過濾資料
            df_featured = df_featured[df_featured['lawful_vs_majority_conflict'] == 1].copy()
            n_after = len(df_featured)
            
            print(f"   過濾前: {n_before:,} 行")
            print(f"   過濾後: {n_after:,} 行 (已移除 {n_before - n_after:,} 行非衝突資料)")
            print("   ✅ Dataset 現在僅包含真正的道德兩難情境")
        else:
            print("   ❌ 錯誤：找不到衝突標記欄位，無法篩選！")
            return
        
        # Step 4: 分割訓練/測試集
        print("\n【Step 4】分割訓練/測試集...")
        train_df, test_df = engineer.split_train_test(df_featured)
        
        # Step 5: 建立使用者側寫
        print("\n【Step 5】建立使用者道德側寫...")
        train_profiles = engineer.create_user_profiles(train_df)
        train_profiles['split'] = 'train'
        
        test_profiles = engineer.create_user_profiles(test_df)
        test_profiles['split'] = 'test'
        
        all_profiles = pd.concat([train_profiles, test_profiles], ignore_index=True)
        
        # Step 6: 儲存
        print("\n【Step 6】儲存結果...")
        save_featured_data(df_featured)
        save_user_profiles(all_profiles)
        save_train_test_split(train_df, test_df)
        
        # Step 7: 報告
        print("\n【Step 7】產生報告...")
        descriptions = engineer.get_feature_descriptions()
        save_feature_descriptions(descriptions)
        generate_feature_statistics(df_featured)
        generate_markdown_report(df_featured, all_profiles, descriptions, n_before)
        
        print("\n" + "=" * 60)
        print("✅ 特徵工程完成！(Conflict-Only Dataset Created)")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"執行失敗: {e}", exc_info=True)
        print(f"\n❌ 錯誤: {e}")
        raise

if __name__ == '__main__':
    main()