"""
特徵工程診斷腳本
==================
檢查特徵工程執行結果，特別是：
1. Cluster == -1 的分佈與特徵狀態
2. 國家層級特徵的缺失情況
3. 訓練/測試集的分割品質
4. 使用者側寫的完整性

執行方式：
    python diagnostic/feature_engineering_diagnostic.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def check_cluster_distribution(df: pd.DataFrame) -> dict:
    """檢查 Cluster 分佈"""
    print("\n" + "=" * 60)
    print("【1】Cluster 分佈檢查")
    print("=" * 60)
    
    if 'Cluster' not in df.columns:
        print("❌ 找不到 Cluster 欄位")
        return {}
    
    cluster_mapping = {
        -1: 'Unclassified (未分類)',
        0: 'Western (西方)',
        1: 'Eastern (東方)',
        2: 'Southern (南方)'
    }
    
    print("\n📊 Cluster 分佈:")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    total_rows = len(df)
    
    results = {}
    for cluster_id, count in cluster_counts.items():
        cluster_name = cluster_mapping.get(cluster_id, f'Unknown ({cluster_id})')
        pct = count / total_rows * 100
        print(f"  {cluster_name:30s}: {count:8,} 行 ({pct:5.2f}%)")
        results[cluster_id] = {'count': count, 'percentage': pct}
    
    # 檢查是否有 NaN
    nan_count = df['Cluster'].isna().sum()
    if nan_count > 0:
        pct = nan_count / total_rows * 100
        print(f"  {'NaN (真正的缺失值)':30s}: {nan_count:8,} 行 ({pct:5.2f}%)")
        print("  ⚠️  警告: Cluster 不應該有 NaN 值!")
    
    return results


def check_country_features_missing(df: pd.DataFrame) -> pd.DataFrame:
    """檢查國家層級特徵的缺失情況"""
    print("\n" + "=" * 60)
    print("【2】國家層級特徵缺失檢查")
    print("=" * 60)
    
    # 找出所有國家層級特徵
    country_cols = [col for col in df.columns if col.startswith('country_')]
    
    if not country_cols:
        print("❌ 找不到國家層級特徵 (country_* 欄位)")
        return pd.DataFrame()
    
    print(f"\n📊 找到 {len(country_cols)} 個國家層級特徵")
    
    # 建立缺失統計表
    missing_stats = []
    for col in country_cols:
        total = len(df)
        missing = df[col].isna().sum()
        present = total - missing
        missing_pct = missing / total * 100
        
        missing_stats.append({
            '特徵名稱': col,
            '有效值數量': present,
            '缺失值數量': missing,
            '缺失比例(%)': f"{missing_pct:.2f}%"
        })
    
    missing_df = pd.DataFrame(missing_stats)
    print("\n" + missing_df.to_string(index=False))
    
    # 檢查缺失值是否與 Cluster == -1 一致
    print("\n" + "-" * 60)
    print("【缺失值與 Cluster == -1 的關係】")
    print("-" * 60)
    
    if 'Cluster' in df.columns:
        unclassified_rows = (df['Cluster'] == -1).sum()
        print(f"\nCluster == -1 的行數: {unclassified_rows:,}")
        
        # 檢查第一個國家特徵的缺失數量
        first_country_col = country_cols[0]
        country_missing = df[first_country_col].isna().sum()
        print(f"{first_country_col} 的缺失數量: {country_missing:,}")
        
        if unclassified_rows == country_missing:
            print("✅ 缺失數量與 Cluster == -1 數量一致")
            print("   → 這些國家無法在 CountriesChangePr.csv 中找到對應的 AMCE 值")
        else:
            print("⚠️  缺失數量與 Cluster == -1 數量不一致")
            print("   → 可能存在其他原因導致的缺失")
        
        # 交叉檢查
        if country_missing > 0:
            # 檢查這些缺失值的 Cluster 分佈
            missing_mask = df[first_country_col].isna()
            missing_cluster_dist = df[missing_mask]['Cluster'].value_counts().sort_index()
            
            print("\n缺失值的 Cluster 分佈:")
            for cluster_id, count in missing_cluster_dist.items():
                cluster_name = {-1: 'Unclassified', 0: 'Western', 1: 'Eastern', 2: 'Southern'}.get(cluster_id, f'Unknown')
                print(f"  Cluster {cluster_id} ({cluster_name}): {count:,} 行")
    
    return missing_df


def check_unclassified_countries(df: pd.DataFrame):
    """詳細檢查 Cluster == -1 的國家"""
    print("\n" + "=" * 60)
    print("【3】Unclassified 國家詳細檢查")
    print("=" * 60)
    
    if 'Cluster' not in df.columns:
        print("❌ 找不到 Cluster 欄位")
        return
    
    if 'UserCountry3' not in df.columns:
        print("❌ 找不到 UserCountry3 欄位")
        return
    
    # 篩選 Cluster == -1 的資料
    unclassified_df = df[df['Cluster'] == -1]
    
    if len(unclassified_df) == 0:
        print("✅ 沒有 Cluster == -1 的資料")
        return
    
    print(f"\n📊 總共 {len(unclassified_df):,} 行資料 (Cluster == -1)")
    
    # 統計涉及的國家
    country_counts = unclassified_df['UserCountry3'].value_counts()
    print(f"\n涉及 {len(country_counts)} 個國家:")
    print("\n排名前 20 的國家:")
    for i, (country, count) in enumerate(country_counts.head(20).items(), 1):
        pct = count / len(unclassified_df) * 100
        print(f"  {i:2d}. {country:5s}: {count:5,} 行 ({pct:5.2f}%)")
    
    if len(country_counts) > 20:
        other_count = country_counts.iloc[20:].sum()
        other_pct = other_count / len(unclassified_df) * 100
        print(f"  ... 其他 {len(country_counts)-20} 個國家: {other_count:5,} 行 ({other_pct:5.2f}%)")
    
    # 檢查使用者分佈
    if 'UserID' in unclassified_df.columns:
        unique_users = unclassified_df['UserID'].nunique()
        print(f"\n涉及 {unique_users:,} 位使用者")


def check_scenario_features(df: pd.DataFrame):
    """檢查場景層級特徵"""
    print("\n" + "=" * 60)
    print("【4】場景層級特徵檢查")
    print("=" * 60)
    
    scenario_features = ['is_lawful', 'is_majority', 'chose_lawful', 
                        'chose_majority', 'lawful_vs_majority_conflict']
    
    missing_features = [f for f in scenario_features if f not in df.columns]
    if missing_features:
        print(f"❌ 缺少以下特徵: {', '.join(missing_features)}")
        return
    
    print("\n📊 場景特徵統計:")
    stats = []
    for feat in scenario_features:
        if feat in df.columns:
            mean_val = df[feat].mean()
            std_val = df[feat].std()
            min_val = df[feat].min()
            max_val = df[feat].max()
            nan_count = df[feat].isna().sum()
            
            stats.append({
                '特徵': feat,
                '平均值': f"{mean_val:.3f}",
                '標準差': f"{std_val:.3f}",
                '範圍': f"[{min_val:.0f}, {max_val:.0f}]",
                '缺失數': nan_count
            })
    
    stats_df = pd.DataFrame(stats)
    print("\n" + stats_df.to_string(index=False))
    
    # 檢查邏輯一致性
    print("\n" + "-" * 60)
    print("【邏輯一致性檢查】")
    print("-" * 60)
    
    # 檢查 is_lawful 和 is_majority 的分佈
    if 'is_lawful' in df.columns and 'is_majority' in df.columns:
        lawful_rate = df['is_lawful'].mean()
        majority_rate = df['is_majority'].mean()
        
        print(f"\nis_lawful 為 1 的比例: {lawful_rate:.1%}")
        print(f"is_majority 為 1 的比例: {majority_rate:.1%}")
        
        if abs(lawful_rate - 0.5) < 0.05 and abs(majority_rate - 0.5) < 0.05:
            print("✅ 分佈接近 50/50 (符合預期)")
        else:
            print("⚠️  分佈偏離 50/50 (可能需要檢查)")


def check_train_test_split():
    """檢查訓練/測試集分割"""
    print("\n" + "=" * 60)
    print("【5】訓練/測試集分割檢查")
    print("=" * 60)
    
    train_file = Path('data/processed/train_data.csv')
    test_file = Path('data/processed/test_data.csv')
    split_file = Path('data/processed/train_test_split.json')
    
    if not train_file.exists() or not test_file.exists():
        print("❌ 找不到訓練/測試集檔案")
        return
    
    # 讀取分割資訊
    if split_file.exists():
        import json
        with open(split_file, 'r', encoding='utf-8') as f:
            split_info = json.load(f)
        
        train_users = set(split_info['train_users'])
        test_users = set(split_info['test_users'])
        
        print(f"\n📊 分割資訊 (來自 {split_file.name}):")
        print(f"  訓練集使用者: {len(train_users):,} 位")
        print(f"  測試集使用者: {len(test_users):,} 位")
        print(f"  測試集比例: {split_info['test_size']:.1%}")
        print(f"  分割時間: {split_info['split_date']}")
        
        # 檢查是否有重疊
        overlap = train_users & test_users
        if len(overlap) > 0:
            print(f"\n❌ 發現 {len(overlap)} 位使用者同時出現在訓練集和測試集!")
            print("   這會造成資料洩漏!")
        else:
            print("\n✅ 訓練集和測試集完全分離，無資料洩漏")
    
    # 讀取檔案大小
    print("\n📊 檔案資訊:")
    train_size_mb = train_file.stat().st_size / 1024**2
    test_size_mb = test_file.stat().st_size / 1024**2
    print(f"  訓練集: {train_size_mb:.2f} MB")
    print(f"  測試集: {test_size_mb:.2f} MB")
    print(f"  總大小: {train_size_mb + test_size_mb:.2f} MB")


def check_user_profiles():
    """檢查使用者道德側寫"""
    print("\n" + "=" * 60)
    print("【6】使用者道德側寫檢查")
    print("=" * 60)
    
    profile_file = Path('data/processed/user_moral_profiles.csv')
    
    if not profile_file.exists():
        print("❌ 找不到使用者側寫檔案")
        return
    
    profiles_df = pd.read_csv(profile_file)
    print(f"\n📊 總共 {len(profiles_df):,} 位使用者")
    
    # 檢查分割標記
    if 'split' in profiles_df.columns:
        train_count = (profiles_df['split'] == 'train').sum()
        test_count = (profiles_df['split'] == 'test').sum()
        
        print(f"\n分割情況:")
        print(f"  訓練集: {train_count:,} 位 ({train_count/len(profiles_df)*100:.1f}%)")
        print(f"  測試集: {test_count:,} 位 ({test_count/len(profiles_df)*100:.1f}%)")
        
        if abs(train_count / len(profiles_df) - 0.8) < 0.02:
            print("  ✅ 分割比例接近 80/20")
        else:
            print("  ⚠️  分割比例偏離 80/20")
    
    # 檢查側寫特徵
    profile_features = ['utilitarian_score', 'deontology_score', 
                       'consistency_score', 'n_scenarios']
    
    print("\n📊 側寫特徵統計:")
    for feat in profile_features:
        if feat in profiles_df.columns:
            mean_val = profiles_df[feat].mean()
            std_val = profiles_df[feat].std()
            min_val = profiles_df[feat].min()
            max_val = profiles_df[feat].max()
            
            print(f"\n{feat}:")
            print(f"  平均: {mean_val:.3f}")
            print(f"  標準差: {std_val:.3f}")
            print(f"  範圍: [{min_val:.3f}, {max_val:.3f}]")
    
    # 檢查道德傾向分佈
    if 'utilitarian_score' in profiles_df.columns:
        print("\n📊 道德傾向分佈:")
        strong_util = (profiles_df['utilitarian_score'] > 0.7).sum()
        moderate = ((profiles_df['utilitarian_score'] >= 0.3) & 
                   (profiles_df['utilitarian_score'] <= 0.7)).sum()
        strong_deont = (profiles_df['utilitarian_score'] < 0.3).sum()
        
        print(f"  強效益主義 (>0.7): {strong_util:,} 位 ({strong_util/len(profiles_df)*100:.1f}%)")
        print(f"  中間派 (0.3-0.7): {moderate:,} 位 ({moderate/len(profiles_df)*100:.1f}%)")
        print(f"  強義務論 (<0.3): {strong_deont:,} 位 ({strong_deont/len(profiles_df)*100:.1f}%)")


def generate_diagnostic_summary(df: pd.DataFrame):
    """生成診斷摘要"""
    print("\n" + "=" * 60)
    print("【診斷摘要】")
    print("=" * 60)
    
    issues = []
    warnings = []
    success = []
    
    # 檢查 Cluster
    if 'Cluster' in df.columns:
        if df['Cluster'].isna().sum() > 0:
            issues.append("Cluster 欄位有 NaN 值")
        else:
            success.append("Cluster 欄位無缺失值")
        
        unclassified_count = (df['Cluster'] == -1).sum()
        if unclassified_count > 0:
            warnings.append(f"{unclassified_count:,} 行屬於 Unclassified (這是正常的)")
    
    # 檢查國家特徵
    country_cols = [col for col in df.columns if col.startswith('country_')]
    if country_cols:
        first_country_col = country_cols[0]
        missing_count = df[first_country_col].isna().sum()
        if missing_count > 0:
            warnings.append(f"國家特徵有 {missing_count:,} 個缺失值 (對應 Cluster == -1)")
        else:
            success.append("國家特徵無缺失值")
    
    # 輸出摘要
    print("\n✅ 成功項目:")
    for item in success:
        print(f"  • {item}")
    
    if warnings:
        print("\n⚠️  注意事項:")
        for item in warnings:
            print(f"  • {item}")
    
    if issues:
        print("\n❌ 發現問題:")
        for item in issues:
            print(f"  • {item}")
    else:
        print("\n🎉 未發現嚴重問題")


def main():
    """主執行函數"""
    print("\n" + "=" * 80)
    print("🔍 MIT Moral Machine - 特徵工程診斷報告")
    print("=" * 80)
    
    # 載入資料
    featured_file = Path('data/processed/featured_data.csv')
    
    if not featured_file.exists():
        print(f"\n❌ 找不到檔案: {featured_file}")
        print("請先執行 03_feature_engineering.py")
        return
    
    print(f"\n📂 載入資料: {featured_file}")
    df = pd.read_csv(featured_file)
    print(f"   總行數: {len(df):,}")
    print(f"   總欄位: {len(df.columns)}")
    
    # 執行各項檢查
    check_cluster_distribution(df)
    missing_df = check_country_features_missing(df)
    check_unclassified_countries(df)
    check_scenario_features(df)
    check_train_test_split()
    check_user_profiles()
    generate_diagnostic_summary(df)
    
    # 儲存診斷結果
    output_dir = Path('outputs/diagnostic')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not missing_df.empty:
        output_file = output_dir / 'country_features_missing_report.csv'
        missing_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 診斷報告已儲存: {output_file}")
    
    print("\n" + "=" * 80)
    print("✅ 診斷完成")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()