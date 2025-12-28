"""
特徵工程模組 (Feature Engineering Module)
建立衍生變數、使用者道德側寫、訓練/測試集分割

主要功能：
1. 建立場景層級特徵（守法標記、多數標記等）
2. 建立使用者道德側寫（效益主義傾向、守法傾向等）
3. 分割訓練/測試集（80/20，使用者層級分割）
4. 產生特徵說明文件
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import logging
from sklearn.model_selection import train_test_split

class FeatureEngineer:
    """特徵工程器"""
    
    def __init__(self):
        """初始化特徵工程器"""
        self.logger = self._setup_logger()
        self.feature_log = {
            'scenario_features': [],
            'user_features': []
        }
        
    def _setup_logger(self) -> logging.Logger:
        """設定日誌記錄器"""
        logger = logging.getLogger('FeatureEngineer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        執行完整的特徵工程流程
        
        Parameters:
        -----------
        df : pd.DataFrame
            清理後的資料
            
        Returns:
        --------
        pd.DataFrame
            增加特徵後的資料
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("開始特徵工程...")
        self.logger.info("=" * 60)
        
        # Step 1: 建立場景層級特徵
        df = self._create_scenario_features(df)
        
        # Step 2: 驗證資料一致性
        self._validate_scenario_structure(df)
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("特徵工程完成！")
        self.logger.info("=" * 60)
        
        return df
    
    def _create_scenario_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 1: 建立場景層級特徵
        
        新增欄位：
        - is_lawful: 該側是否守法（1=守法, 0=違法）
        - is_majority: 該側是否為多數（1=多數, 0=少數）
        - chose_lawful: 使用者是否選擇守法側（1=是, 0=否）
        - chose_majority: 使用者是否選擇多數側（1=是, 0=否）
        - lawful_vs_majority_conflict: 守法和多數是否衝突（1=衝突, 0=一致）
        """
        self.logger.info("\n" + "-" * 60)
        self.logger.info("【Step 1】建立場景層級特徵")
        self.logger.info("-" * 60)
        
        df = df.copy()
        
        # 1. 守法標記 (is_lawful)
        # CrossingSignal: 1=綠燈(守法), 2=紅燈(違法)
        df['is_lawful'] = (df['CrossingSignal'] == 1).astype(int)
        self.logger.info("✅ 建立 is_lawful (該側是否守法)")
        
        # 2. 多數標記 (is_majority)
        # 需要比較同一場景兩側的人數
        # 先建立一個場景內的排序
        df = df.sort_values(['ResponseID', 'NumberOfCharacters'], ascending=[True, False])
        
        # 為每個場景標記多數側
        df['scenario_rank'] = df.groupby('ResponseID').cumcount()
        df['is_majority'] = (df['scenario_rank'] == 0).astype(int)
        df = df.drop(columns=['scenario_rank'])
        
        self.logger.info("✅ 建立 is_majority (該側是否為多數)")
        
        # 3. 使用者選擇 (chose_lawful, chose_majority)
        # Saved=1 表示該側被選擇拯救
        df['chose_lawful'] = (df['Saved'] == 1) & (df['is_lawful'] == 1)
        df['chose_majority'] = (df['Saved'] == 1) & (df['is_majority'] == 1)
        
        # 轉換為整數
        df['chose_lawful'] = df['chose_lawful'].astype(int)
        df['chose_majority'] = df['chose_majority'].astype(int)
        
        self.logger.info("✅ 建立 chose_lawful (使用者是否選擇守法側)")
        self.logger.info("✅ 建立 chose_majority (使用者是否選擇多數側)")
        
        # 4. 衝突指標 (lawful_vs_majority_conflict)
        # 在同一場景中，守法側和多數側是否不同
        scenario_conflict = df.groupby('ResponseID').apply(
            lambda x: (x['is_lawful'].sum() == 1) and (x['is_majority'].sum() == 1) and 
                     (x['is_lawful'] & x['is_majority']).sum() == 0
        ).reset_index(name='has_conflict')
        
        df = df.merge(scenario_conflict, on='ResponseID', how='left')
        df['lawful_vs_majority_conflict'] = df['has_conflict'].astype(int)
        df = df.drop(columns=['has_conflict'])
        
        self.logger.info("✅ 建立 lawful_vs_majority_conflict (守法vs.多數衝突)")
        
        # 統計新特徵
        self.logger.info("\n特徵統計:")
        self.logger.info(f"  守法側: {df['is_lawful'].sum():,} / {len(df):,} ({df['is_lawful'].mean()*100:.1f}%)")
        self.logger.info(f"  多數側: {df['is_majority'].sum():,} / {len(df):,} ({df['is_majority'].mean()*100:.1f}%)")
        self.logger.info(f"  選擇守法: {df['chose_lawful'].sum():,} / {len(df):,} ({df['chose_lawful'].mean()*100:.1f}%)")
        self.logger.info(f"  選擇多數: {df['chose_majority'].sum():,} / {len(df):,} ({df['chose_majority'].mean()*100:.1f}%)")
        
        conflict_scenarios = df['ResponseID'].nunique()
        conflict_count = df[df['lawful_vs_majority_conflict'] == 1]['ResponseID'].nunique()
        self.logger.info(f"  有衝突場景: {conflict_count:,} / {conflict_scenarios:,} ({conflict_count/conflict_scenarios*100:.1f}%)")
        
        # 記錄到特徵日誌
        self.feature_log['scenario_features'] = [
            'is_lawful', 'is_majority', 'chose_lawful', 
            'chose_majority', 'lawful_vs_majority_conflict'
        ]
        
        return df
    
    def _validate_scenario_structure(self, df: pd.DataFrame):
        """
        驗證場景結構的正確性
        
        檢查：
        1. 每個場景是否都有且僅有一個多數側和一個少數側
        2. 每個場景的 Saved 總和是否為1（只有一側被選擇）
        3. 衝突場景的定義是否正確
        """
        self.logger.info("\n" + "-" * 60)
        self.logger.info("【驗證】場景結構檢查")
        self.logger.info("-" * 60)
        
        # 檢查1: 多數標記
        majority_check = df.groupby('ResponseID')['is_majority'].sum()
        if not (majority_check == 1).all():
            incorrect_count = (majority_check != 1).sum()
            self.logger.warning(f"⚠️  {incorrect_count} 個場景的多數標記異常")
        else:
            self.logger.info("✅ 所有場景都有且僅有一個多數側")
        
        # 檢查2: Saved 總和
        saved_check = df.groupby('ResponseID')['Saved'].sum()
        if not (saved_check == 1).all():
            incorrect_count = (saved_check != 1).sum()
            self.logger.warning(f"⚠️  {incorrect_count} 個場景的 Saved 總和不為1")
        else:
            self.logger.info("✅ 所有場景都只有一側被選擇")
        
        # 檢查3: 衝突定義
        # 有衝突的場景，應該守法和多數分屬不同側
        conflict_scenarios = df[df['lawful_vs_majority_conflict'] == 1]
        if len(conflict_scenarios) > 0:
            sample_scenario = conflict_scenarios.groupby('ResponseID').first().iloc[0]
            self.logger.info(f"✅ 衝突場景定義正確（共 {conflict_scenarios['ResponseID'].nunique():,} 個）")
    
    def create_user_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        建立使用者道德側寫
        
        為每位使用者計算：
        - utilitarian_score: 效益主義分數（選擇多數的比例）
        - deontology_score: 義務論分數（選擇守法的比例）
        - consistency_score: 決策一致性（避免矛盾決策）
        - n_scenarios: 完成的場景數
        - avg_response_time: 平均回應時間（如果有的話）
        
        Parameters:
        -----------
        df : pd.DataFrame
            包含場景特徵的資料
            
        Returns:
        --------
        pd.DataFrame
            使用者側寫資料框
        """
        self.logger.info("\n" + "-" * 60)
        self.logger.info("【Step 2】建立使用者道德側寫")
        self.logger.info("-" * 60)
        
        # 只保留有衝突的場景（才能真正反映道德傾向）
        conflict_df = df[df['lawful_vs_majority_conflict'] == 1].copy()
        
        self.logger.info(f"使用 {conflict_df['ResponseID'].nunique():,} 個衝突場景")
        self.logger.info(f"涉及 {conflict_df['UserID'].nunique():,} 位使用者")
        
        # 按使用者分組計算
        user_profiles = []
        
        for user_id, user_df in conflict_df.groupby('UserID'):
            # 只計算被選擇的那一側（Saved=1）
            chosen_sides = user_df[user_df['Saved'] == 1]
            
            if len(chosen_sides) == 0:
                continue
            
            profile = {
                'UserID': user_id,
                # 效益主義分數：選擇多數的比例
                'utilitarian_score': chosen_sides['is_majority'].mean(),
                # 義務論分數：選擇守法的比例
                'deontology_score': chosen_sides['is_lawful'].mean(),
                # 完成場景數
                'n_scenarios': len(chosen_sides),
            }
            
            # 決策一致性：看是否有「既選過多數又選過少數」的情況
            # 一致性越高，表示決策模式越穩定
            majority_choices = chosen_sides['is_majority'].values
            if len(majority_choices) > 1:
                # 使用標準差的倒數作為一致性指標（標準差越小越一致）
                std = majority_choices.std()
                profile['consistency_score'] = 1 - std if std < 1 else 0
            else:
                profile['consistency_score'] = 1.0  # 只有一個場景，視為完全一致
            
            user_profiles.append(profile)
        
        profiles_df = pd.DataFrame(user_profiles)
        
        # 統計
        self.logger.info(f"\n建立了 {len(profiles_df):,} 位使用者的道德側寫")
        self.logger.info("\n側寫統計:")
        self.logger.info(f"  效益主義分數: 平均 {profiles_df['utilitarian_score'].mean():.3f}, "
                        f"標準差 {profiles_df['utilitarian_score'].std():.3f}")
        self.logger.info(f"  義務論分數: 平均 {profiles_df['deontology_score'].mean():.3f}, "
                        f"標準差 {profiles_df['deontology_score'].std():.3f}")
        self.logger.info(f"  決策一致性: 平均 {profiles_df['consistency_score'].mean():.3f}, "
                        f"標準差 {profiles_df['consistency_score'].std():.3f}")
        self.logger.info(f"  場景數量: 平均 {profiles_df['n_scenarios'].mean():.1f}, "
                        f"中位數 {profiles_df['n_scenarios'].median():.0f}")
        
        # 記錄到特徵日誌
        self.feature_log['user_features'] = list(profiles_df.columns)
        
        return profiles_df
    
    def merge_country_features(self, 
                              df: pd.DataFrame,
                              countries_change_df: pd.DataFrame) -> pd.DataFrame:
        """
        合併國家層級特徵
        
        從 CountriesChangePr.csv 合併國家的 AMCE 值，
        用於階層線性模型分析
        
        Parameters:
        -----------
        df : pd.DataFrame
            場景資料
        countries_change_df : pd.DataFrame
            國家變化概率資料
            
        Returns:
        --------
        pd.DataFrame
            合併國家特徵後的資料
        """
        self.logger.info("\n" + "-" * 60)
        self.logger.info("【合併國家層級特徵】")
        self.logger.info("-" * 60)
        
        # 選擇要合併的欄位（AMCE estimates）
        # 核心欄位：守法和效益主義
        core_cols = [
            'Law [Illegal -> Legal]: Estimates',
            'No. Characters [Less -> More]: Estimates'
        ]
        
        # 其他道德維度（可選）
        other_cols = [
            '[Omission -> Commission]: Estimates',
            '[Passengers -> Pedestrians]: Estimates',
            'Gender [Male -> Female]: Estimates',
            'Fitness [Large -> Fit]: Estimates',
            'Social Status [Low -> High]: Estimates',
            'Age [Elderly -> Young]: Estimates',
            'Species [Pets -> Humans]: Estimates'
        ]
        
        # 檢查哪些欄位存在
        available_cols = [col for col in core_cols + other_cols 
                         if col in countries_change_df.columns]
        
        # 準備合併資料（國家代碼在 index 或第一欄）
        if countries_change_df.index.name or countries_change_df.columns[0] == '':
            # 國家代碼在 index
            merge_df = countries_change_df[available_cols].copy()
            merge_df['country_code'] = merge_df.index
        else:
            # 國家代碼在第一欄（可能叫 '' 或其他名稱）
            first_col = countries_change_df.columns[0]
            merge_df = countries_change_df[[first_col] + available_cols].copy()
            merge_df = merge_df.rename(columns={first_col: 'country_code'})
        
        # 重新命名欄位為簡短名稱
        rename_map = {
            'Law [Illegal -> Legal]: Estimates': 'country_law_preference',
            'No. Characters [Less -> More]: Estimates': 'country_utilitarian',
            '[Omission -> Commission]: Estimates': 'country_intervention',
            '[Passengers -> Pedestrians]: Estimates': 'country_pedestrian_pref',
            'Gender [Male -> Female]: Estimates': 'country_gender_pref',
            'Fitness [Large -> Fit]: Estimates': 'country_fitness_pref',
            'Social Status [Low -> High]: Estimates': 'country_status_pref',
            'Age [Elderly -> Young]: Estimates': 'country_age_pref',
            'Species [Pets -> Humans]: Estimates': 'country_species_pref'
        }
        
        merge_df = merge_df.rename(columns={k: v for k, v in rename_map.items() if k in merge_df.columns})
        
        # 合併到主資料
        initial_rows = len(df)
        df_merged = df.merge(
            merge_df,
            left_on='UserCountry3',
            right_on='country_code',
            how='left'
        )
        
        # 刪除重複的 country_code 欄位
        if 'country_code' in df_merged.columns:
            df_merged = df_merged.drop(columns=['country_code'])
        
        # 檢查合併結果
        merged_cols = [v for v in rename_map.values() if v in df_merged.columns]
        missing_country_features = df_merged[merged_cols[0]].isnull().sum() if merged_cols else 0
        
        self.logger.info(f"合併了 {len(merged_cols)} 個國家層級特徵:")
        for col in merged_cols:
            self.logger.info(f"  - {col}")
        
        if missing_country_features > 0:
            missing_pct = missing_country_features / len(df_merged) * 100
            self.logger.warning(f"\n⚠️  {missing_country_features:,} 行無法對應國家特徵 ({missing_pct:.2f}%)")
        else:
            self.logger.info("\n✅ 所有資料都成功對應到國家特徵")
        
        self.logger.info(f"\n合併完成: {initial_rows:,} 行 → {len(df_merged):,} 行")
        
        # 記錄到特徵日誌
        self.feature_log['country_features'] = merged_cols
        
        return df_merged
    
    def add_feature_availability_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        增加特徵可用性標記
        
        Parameters:
        -----------
        df : pd.DataFrame
            特徵化後的資料
            
        Returns:
        --------
        pd.DataFrame
            增加 has_country_features 欄位的資料
        """
        self.logger.info("\n" + "-" * 60)
        self.logger.info("【增加特徵可用性標記】")
        self.logger.info("-" * 60)
        
        # 檢查是否有國家特徵
        if 'country_law_preference' not in df.columns:
            self.logger.warning("找不到國家特徵，跳過標記")
            return df
        
        # 建立標記（任一國家特徵非空即為 True）
        df['has_country_features'] = df['country_law_preference'].notna()
        
        # 統計
        has_features = df['has_country_features'].sum()
        no_features = (~df['has_country_features']).sum()
        
        self.logger.info(f"有國家特徵: {has_features:,} 行 ({has_features/len(df)*100:.2f}%)")
        self.logger.info(f"無國家特徵: {no_features:,} 行 ({no_features/len(df)*100:.2f}%)")
        
        # 驗證：無特徵的資料應該都是 Cluster == -1
        if 'Cluster' in df.columns:
            no_features_cluster = df[~df['has_country_features']]['Cluster'].value_counts()
            self.logger.info("\n無國家特徵資料的 Cluster 分佈:")
            for cluster, count in no_features_cluster.items():
                self.logger.info(f"  Cluster {cluster}: {count:,} 行")
            
            if len(no_features_cluster) == 1 and -1 in no_features_cluster:
                self.logger.info("✅ 確認：所有無國家特徵的資料都是 Cluster == -1")
            else:
                self.logger.warning("⚠️  發現無國家特徵但 Cluster != -1 的資料!")
        
        return df
    
    def split_train_test(self, 
                        df: pd.DataFrame,
                        test_size: float = 0.2,
                        random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        分割訓練集和測試集（使用者層級分割，避免資料洩漏）
        
        Parameters:
        -----------
        df : pd.DataFrame
            完整資料
        test_size : float
            測試集比例（預設0.2）
        random_state : int
            隨機種子
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            (訓練集, 測試集)
        """
        self.logger.info("\n" + "-" * 60)
        self.logger.info("【Step 3】分割訓練/測試集")
        self.logger.info("-" * 60)
        
        # 取得所有使用者ID
        unique_users = df['UserID'].unique()
        
        # 使用者層級分割
        train_users, test_users = train_test_split(
            unique_users,
            test_size=test_size,
            random_state=random_state
        )
        
        # 根據使用者分割資料
        train_df = df[df['UserID'].isin(train_users)].copy()
        test_df = df[df['UserID'].isin(test_users)].copy()
        
        # 統計
        self.logger.info(f"\n分割結果:")
        self.logger.info(f"  訓練集: {len(train_df):,} 行, {len(train_users):,} 位使用者 "
                        f"({len(train_df)/len(df)*100:.1f}%)")
        self.logger.info(f"  測試集: {len(test_df):,} 行, {len(test_users):,} 位使用者 "
                        f"({len(test_df)/len(df)*100:.1f}%)")
        
        # 檢查是否有使用者跨集
        overlap = set(train_users) & set(test_users)
        if len(overlap) > 0:
            self.logger.warning(f"⚠️  發現 {len(overlap)} 位使用者同時出現在訓練集和測試集！")
        else:
            self.logger.info("✅ 訓練集和測試集完全分離，無資料洩漏")
        
        # 檢查文化圈分佈是否平衡
        if 'Cluster' in df.columns:
            self.logger.info("\n文化圈分佈:")
            for cluster in sorted(df['Cluster'].unique()):
                train_pct = (train_df['Cluster'] == cluster).mean() * 100
                test_pct = (test_df['Cluster'] == cluster).mean() * 100
                
                cluster_names = {-1: 'Unclassified', 0: 'Western', 1: 'Eastern', 2: 'Southern'}
                cluster_name = cluster_names.get(int(cluster), f'Cluster {int(cluster)}')
                
                self.logger.info(f"  {cluster_name}: 訓練 {train_pct:.1f}%, 測試 {test_pct:.1f}%")
        
        return train_df, test_df
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        取得特徵說明
        
        Returns:
        --------
        Dict[str, str]
            特徵名稱與說明的字典
        """
        descriptions = {
            # 場景層級特徵
            'is_lawful': '該側是否守法（1=守法綠燈, 0=違法紅燈）',
            'is_majority': '該側是否為多數（1=人數較多, 0=人數較少）',
            'chose_lawful': '使用者是否選擇守法側（1=是, 0=否）',
            'chose_majority': '使用者是否選擇多數側（1=是, 0=否）',
            'lawful_vs_majority_conflict': '守法和多數是否衝突（1=衝突, 0=一致）',
            
            # 使用者側寫特徵
            'utilitarian_score': '效益主義分數：選擇多數的比例 [0, 1]',
            'deontology_score': '義務論分數：選擇守法的比例 [0, 1]',
            'consistency_score': '決策一致性：決策模式的穩定度 [0, 1]',
            'n_scenarios': '使用者完成的衝突場景數量',
            
            # 國家層級特徵（AMCE值）
            'country_law_preference': '國家層級守法偏好（AMCE）',
            'country_utilitarian': '國家層級效益主義傾向（AMCE）',
            'country_intervention': '國家層級介入偏好（AMCE）',
            'country_pedestrian_pref': '國家層級行人優先偏好（AMCE）',
            'country_gender_pref': '國家層級性別偏好（AMCE）',
            'country_fitness_pref': '國家層級體型偏好（AMCE）',
            'country_status_pref': '國家層級社會地位偏好（AMCE）',
            'country_age_pref': '國家層級年齡偏好（AMCE）',
            'country_species_pref': '國家層級物種偏好（AMCE）',
        }
        
        return descriptions


def engineer_all_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    便捷函數：執行完整的特徵工程
    
    Parameters:
    -----------
    df : pd.DataFrame
        清理後的資料
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, Dict]
        (增加特徵的資料, 使用者側寫, 特徵說明)
    """
    engineer = FeatureEngineer()
    
    # 建立場景特徵
    df_featured = engineer.engineer_features(df)
    
    # 建立使用者側寫
    user_profiles = engineer.create_user_profiles(df_featured)
    
    # 取得特徵說明
    feature_descriptions = engineer.get_feature_descriptions()
    
    return df_featured, user_profiles, feature_descriptions


# 測試模組
if __name__ == '__main__':
    print("特徵工程模組載入成功")