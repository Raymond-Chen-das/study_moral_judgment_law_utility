"""
cleaning_diagnostic.py
=========================
資料清理診斷腳本

功能：
1. 診斷 Step 2 異常值刪除的詳細情況
2. 分析無法對應到文化圈的國家
3. 產生詳細診斷報告

執行方式：
    python diagnostic/cleaning_diagnostic.py
"""

import sys
from pathlib import Path

# 將專案根目錄加入路徑
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime

def analyze_step2_outliers(survey_df: pd.DataFrame):
    """
    分析 Step 2 異常值的詳細情況
    
    Parameters:
    -----------
    survey_df : pd.DataFrame
        原始問卷資料（經過Step 1處理）
    """
    print("\n" + "=" * 70)
    print("【診斷】Step 2: 異常值分析")
    print("=" * 70)
    
    outlier_details = []
    total_outliers = 0
    
    # 1. 年齡異常值
    if 'Review_age' in survey_df.columns:
        print("\n1️⃣ 年齡異常值分析")
        print("-" * 70)
        
        # 轉換為數值
        age_numeric = pd.to_numeric(survey_df['Review_age'], errors='coerce')
        
        # 統計
        age_missing = age_numeric.isna().sum()
        age_below_18 = (age_numeric < 18).sum()
        age_above_75 = (age_numeric > 75).sum()
        age_total_outliers = age_below_18 + age_above_75
        
        print(f"   缺失值（無法轉換為數值）: {age_missing:,} 行")
        print(f"   年齡 < 18: {age_below_18:,} 行")
        print(f"   年齡 > 75: {age_above_75:,} 行")
        print(f"   合計異常: {age_total_outliers:,} 行")
        
        # 異常值分佈
        if age_below_18 > 0 or age_above_75 > 0:
            age_outliers = age_numeric[(age_numeric < 18) | (age_numeric > 75)]
            print(f"\n   異常值範圍: {age_outliers.min():.1f} ~ {age_outliers.max():.1f}")
            
            # 顯示前10個最極端的值
            extreme_ages = age_outliers.value_counts().head(10)
            print("\n   最常出現的異常年齡:")
            for age, count in extreme_ages.items():
                print(f"      {age:.0f} 歲: {count:,} 次")
        
        outlier_details.append({
            '類型': '年齡異常',
            '缺失值': age_missing,
            '< 18歲': age_below_18,
            '> 75歲': age_above_75,
            '合計': age_total_outliers
        })
        total_outliers += age_total_outliers
    
    # 2. 政治立場異常值
    if 'Review_political' in survey_df.columns:
        print("\n2️⃣ 政治立場異常值分析")
        print("-" * 70)
        
        political = survey_df['Review_political']
        political_outliers = (political.notna()) & ((political < 0) | (political > 1))
        political_outlier_count = political_outliers.sum()
        
        print(f"   超出範圍 [0, 1]: {political_outlier_count:,} 行")
        
        if political_outlier_count > 0:
            outlier_values = political[political_outliers]
            print(f"   異常值範圍: {outlier_values.min():.3f} ~ {outlier_values.max():.3f}")
            
            # 顯示異常值分佈
            extreme_values = outlier_values.value_counts().head(10)
            print("\n   最常出現的異常值:")
            for val, count in extreme_values.items():
                print(f"      {val:.3f}: {count:,} 次")
        
        outlier_details.append({
            '類型': '政治立場異常',
            '超出範圍': political_outlier_count,
            '合計': political_outlier_count
        })
        total_outliers += political_outlier_count
    
    # 3. 宗教程度異常值
    if 'Review_religious' in survey_df.columns:
        print("\n3️⃣ 宗教程度異常值分析")
        print("-" * 70)
        
        religious = survey_df['Review_religious']
        religious_outliers = (religious.notna()) & ((religious < 0) | (religious > 1))
        religious_outlier_count = religious_outliers.sum()
        
        print(f"   超出範圍 [0, 1]: {religious_outlier_count:,} 行")
        
        if religious_outlier_count > 0:
            outlier_values = religious[religious_outliers]
            print(f"   異常值範圍: {outlier_values.min():.3f} ~ {outlier_values.max():.3f}")
            
            # 顯示異常值分佈
            extreme_values = outlier_values.value_counts().head(10)
            print("\n   最常出現的異常值:")
            for val, count in extreme_values.items():
                print(f"      {val:.3f}: {count:,} 次")
        
        outlier_details.append({
            '類型': '宗教程度異常',
            '超出範圍': religious_outlier_count,
            '合計': religious_outlier_count
        })
        total_outliers += religious_outlier_count
    
    # 總結
    print("\n" + "=" * 70)
    print("【總結】")
    print(f"預期刪除的行數: {total_outliers:,}")
    print("=" * 70)
    
    return pd.DataFrame(outlier_details)

def analyze_missing_cluster(cleaned_df: pd.DataFrame, cluster_map_df: pd.DataFrame):
    """
    分析無法對應到文化圈的國家
    
    Parameters:
    -----------
    cleaned_df : pd.DataFrame
        清理後的資料
    cluster_map_df : pd.DataFrame
        文化圈分類資料
    """
    print("\n" + "=" * 70)
    print("【診斷】無法對應文化圈的國家")
    print("=" * 70)
    
    # 找出無法對應的資料
    missing_cluster = cleaned_df[cleaned_df['Cluster'].isna()]
    
    if len(missing_cluster) == 0:
        print("\n✅ 所有資料都成功對應到文化圈")
        return None
    
    print(f"\n⚠️  共有 {len(missing_cluster):,} 行無法對應")
    
    # 統計各國家的數量
    missing_countries = missing_cluster['UserCountry3'].value_counts()
    
    print(f"\n涉及 {len(missing_countries)} 個國家:")
    print("-" * 70)
    print(f"{'國家代碼':<15} {'決策數量':>15} {'比例':>15}")
    print("-" * 70)
    
    for country, count in missing_countries.items():
        pct = count / len(missing_cluster) * 100
        print(f"{country:<15} {count:>15,} {pct:>14.2f}%")
    
    # 檢查這些國家是否存在於 cluster_map 中
    print("\n" + "-" * 70)
    print("檢查 country_cluster_map.csv 中是否有這些國家:")
    print("-" * 70)
    
    available_countries = set(cluster_map_df['ISO3'].unique())
    
    for country in missing_countries.index:
        if country in available_countries:
            # 找出對應的資料
            country_info = cluster_map_df[cluster_map_df['ISO3'] == country]
            cluster = country_info['Cluster'].values[0] if len(country_info) > 0 else 'N/A'
            print(f"   {country}: ✅ 存在於 cluster_map (Cluster={cluster})")
        else:
            print(f"   {country}: ❌ 不存在於 cluster_map")
    
    # 可能的原因分析
    print("\n" + "-" * 70)
    print("【可能原因】")
    print("-" * 70)
    print("1. 國家代碼不一致（例如：大小寫、空白）")
    print("2. cluster_map.csv 缺少某些國家")
    print("3. 資料合併時的問題")
    
    return missing_countries

def check_data_consistency(survey_df: pd.DataFrame, cleaned_df: pd.DataFrame):
    """
    檢查資料一致性
    
    Parameters:
    -----------
    survey_df : pd.DataFrame
        原始資料（經過Step 1）
    cleaned_df : pd.DataFrame
        清理後的資料
    """
    print("\n" + "=" * 70)
    print("【診斷】資料一致性檢查")
    print("=" * 70)
    
    # 1. 檢查 ResponseID 是否都有2行
    response_counts = cleaned_df['ResponseID'].value_counts()
    
    incomplete = response_counts[response_counts != 2]
    
    if len(incomplete) == 0:
        print("\n✅ 所有場景都完整（每個ResponseID都有2行）")
    else:
        print(f"\n⚠️  發現 {len(incomplete)} 個不完整場景")
        print(f"   涉及 {incomplete.sum()} 行資料")
    
    # 2. 檢查 Saved 欄位的分佈
    if 'Saved' in cleaned_df.columns:
        saved_counts = cleaned_df['Saved'].value_counts()
        print(f"\n【Saved 欄位分佈】")
        print(f"   Saved = 0 (未選擇): {saved_counts.get(0, 0):,} 行")
        print(f"   Saved = 1 (選擇): {saved_counts.get(1, 0):,} 行")
        
        # 理論上應該是1:1
        ratio = saved_counts.get(1, 0) / saved_counts.get(0, 1) if saved_counts.get(0, 0) > 0 else 0
        print(f"   比例: {ratio:.3f} (理論上應接近 1.0)")
    
    # 3. 檢查文化圈分佈
    if 'Cluster' in cleaned_df.columns:
        cluster_dist = cleaned_df['Cluster'].value_counts(dropna=False).sort_index()
        
        print(f"\n【文化圈分佈】")
        cluster_names = {0: 'Western', 1: 'Eastern', 2: 'Southern'}
        for cluster, count in cluster_dist.items():
            if pd.isna(cluster):
                print(f"   缺失值: {count:,} 行")
            else:
                cluster_name = cluster_names.get(int(cluster), f'Cluster {int(cluster)}')
                pct = count / len(cleaned_df) * 100
                print(f"   {cluster_name}: {count:,} 行 ({pct:.1f}%)")

def save_diagnostic_report(outlier_df: pd.DataFrame, 
                          missing_countries: pd.Series,
                          output_dir: str = 'outputs/diagnostic'):
    """
    儲存診斷報告
    
    Parameters:
    -----------
    outlier_df : pd.DataFrame
        異常值統計
    missing_countries : pd.Series
        無法對應的國家
    output_dir : str
        輸出目錄
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("儲存診斷報告...")
    print("=" * 70)
    
    # 1. 儲存異常值統計
    if outlier_df is not None:
        outlier_file = output_path / 'step2_outliers_detail.csv'
        outlier_df.to_csv(outlier_file, index=False, encoding='utf-8-sig')
        print(f"✅ 異常值詳情: {outlier_file}")
    
    # 2. 儲存無法對應的國家
    if missing_countries is not None:
        missing_df = missing_countries.reset_index()
        missing_df.columns = ['國家代碼', '決策數量']
        
        missing_file = output_path / 'missing_cluster_countries.csv'
        missing_df.to_csv(missing_file, index=False, encoding='utf-8-sig')
        print(f"✅ 無法對應國家: {missing_file}")
    
    # 3. 產生文字報告
    report_file = output_path / 'cleaning_diagnostic_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("資料清理診斷報告\n")
        f.write("=" * 70 + "\n")
        f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("【Step 2 異常值統計】\n")
        f.write("-" * 70 + "\n")
        if outlier_df is not None:
            f.write(outlier_df.to_string(index=False))
            f.write("\n\n")
        
        f.write("【無法對應文化圈的國家】\n")
        f.write("-" * 70 + "\n")
        if missing_countries is not None:
            for country, count in missing_countries.items():
                f.write(f"{country}: {count:,} 行\n")
        else:
            f.write("所有國家都成功對應\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"✅ 文字報告: {report_file}")
    print("=" * 70)

def main():
    """主執行函數"""
    print("\n" + "=" * 70)
    print("🔍 資料清理診斷 (Diagnostic)")
    print("=" * 70)
    
    try:
        # 載入原始資料（經過Step 1處理）
        print("\n【載入資料】")
        print("-" * 70)
        
        # 因為我們沒有儲存Step 1的中繼資料，所以重新載入並簡單處理
        print("載入原始問卷資料...")
        survey_df = pd.read_csv('data/raw/SharedResponsesSurvey.csv', low_memory=False)
        
        # 簡單處理關鍵變數缺失（模擬Step 1）
        key_vars = ['Saved', 'ScenarioType', 'UserCountry3', 'ResponseID']
        survey_df = survey_df.dropna(subset=key_vars)
        print(f"✅ 載入 {len(survey_df):,} 行（經Step 1處理）")
        
        # 載入清理後的資料
        print("\n載入清理後的資料...")
        cleaned_df = pd.read_csv('data/processed/cleaned_survey.csv')
        print(f"✅ 載入 {len(cleaned_df):,} 行（最終資料）")
        
        # 載入文化圈分類
        print("\n載入文化圈分類...")
        cluster_map_df = pd.read_csv('data/raw/country_cluster_map.csv')
        print(f"✅ 載入 {len(cluster_map_df)} 個國家的分類")
        
        # 診斷1: Step 2 異常值
        outlier_df = analyze_step2_outliers(survey_df)
        
        # 診斷2: 無法對應文化圈的國家
        missing_countries = analyze_missing_cluster(cleaned_df, cluster_map_df)
        
        # 診斷3: 資料一致性
        check_data_consistency(survey_df, cleaned_df)
        
        # 儲存診斷報告
        save_diagnostic_report(outlier_df, missing_countries)
        
        print("\n" + "=" * 70)
        print("✅ 診斷完成！")
        print("=" * 70)
        print("\n📊 已產生以下輸出:")
        print("  - outputs/diagnostic/step2_outliers_detail.csv")
        print("  - outputs/diagnostic/missing_cluster_countries.csv")
        print("  - outputs/diagnostic/cleaning_diagnostic_report.txt")
        print("=" * 70 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ 錯誤: 找不到檔案 - {e}")
        print("請確認以下檔案存在:")
        print("  - data/raw/SharedResponsesSurvey.csv")
        print("  - data/processed/cleaned_survey.csv")
        print("  - data/raw/country_cluster_map.csv")
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        raise

if __name__ == '__main__':
    main()