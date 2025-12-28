"""
資料清理模組 (Data Cleaner Module)
負責清理原始資料並篩選「守法vs.效益」衝突情境

主要功能：
1. 缺失值處理
2. 異常值檢測與刪除
3. 情境篩選（守法vs.效益衝突）
4. 場景完整性檢查
5. 資料整合（合併文化圈分類）
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import logging
from datetime import datetime

class DataCleaner:
    """MIT Moral Machine 資料清理器"""
    
    def __init__(self):
        """初始化資料清理器"""
        self.logger = self._setup_logger()
        self.cleaning_log = {
            'original_rows': 0,
            'steps': []
        }
        
    def _setup_logger(self) -> logging.Logger:
        """設定日誌記錄器"""
        logger = logging.getLogger('DataCleaner')
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
    
    def clean_data(self, 
                   survey_df: pd.DataFrame,
                   cluster_map_df: pd.DataFrame) -> pd.DataFrame:
        """
        執行完整的資料清理流程
        
        Parameters:
        -----------
        survey_df : pd.DataFrame
            原始問卷資料
        cluster_map_df : pd.DataFrame
            文化圈分類資料
            
        Returns:
        --------
        pd.DataFrame
            清理後的資料
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("開始資料清理流程...")
        self.logger.info("=" * 60)
        
        # 記錄原始資料量
        self.cleaning_log['original_rows'] = len(survey_df)
        self.logger.info(f"原始資料: {len(survey_df):,} 行")
        
        # Step 1: 處理關鍵變數缺失
        df = self._remove_missing_key_variables(survey_df)
        
        # Step 2: 處理異常值
        df = self._remove_outliers(df)
        
        # Step 3: 篩選「守法vs.效益」衝突情境
        df = self._filter_law_vs_utility_scenarios(df)
        
        # Step 4: 檢查場景完整性
        df = self._check_scenario_completeness(df)
        
        # Step 5: 合併文化圈分類
        df = self._merge_cluster_info(df, cluster_map_df)
        
        # 最終報告
        self._print_cleaning_summary()
        
        return df
    
    def _remove_missing_key_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 1: 刪除關鍵變數缺失的行
        
        關鍵變數：Saved, ScenarioType, UserCountry3, ResponseID
        """
        self.logger.info("\n" + "-" * 60)
        self.logger.info("【Step 1】處理關鍵變數缺失")
        self.logger.info("-" * 60)
        
        initial_rows = len(df)
        
        # 定義關鍵變數
        key_variables = ['Saved', 'ScenarioType', 'UserCountry3', 'ResponseID']
        
        # 檢查各變數缺失情況
        for var in key_variables:
            if var in df.columns:
                missing_count = df[var].isnull().sum()
                if missing_count > 0:
                    self.logger.info(f"  {var} 缺失: {missing_count:,} 行")
        
        # 刪除任一關鍵變數缺失的行
        df_cleaned = df.dropna(subset=key_variables)
        
        removed_rows = initial_rows - len(df_cleaned)
        removed_pct = (removed_rows / initial_rows * 100) if initial_rows > 0 else 0
        
        self.logger.info(f"\n✅ 刪除 {removed_rows:,} 行 ({removed_pct:.2f}%)")
        self.logger.info(f"   剩餘 {len(df_cleaned):,} 行")
        
        # 記錄到清理日誌
        self.cleaning_log['steps'].append({
            'step': 'remove_missing_key_vars',
            'removed': int(removed_rows),
            'remaining': int(len(df_cleaned)),
            'removed_pct': f"{removed_pct:.2f}%"
        })
        
        return df_cleaned
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: 刪除異常值
        
        檢查項目：
        - Review_age: 18-75
        - Review_political: 0-1
        - Review_religious: 0-1
        """
        self.logger.info("\n" + "-" * 60)
        self.logger.info("【Step 2】處理異常值")
        self.logger.info("-" * 60)
        
        initial_rows = len(df)
        outlier_mask = pd.Series([False] * len(df), index=df.index)
        
        # 1. 年齡異常值
        if 'Review_age' in df.columns:
            # 先轉換為數值型態（可能有字串）
            df['Review_age'] = pd.to_numeric(df['Review_age'], errors='coerce')
            
            age_outliers = (
                df['Review_age'].notna() & 
                ((df['Review_age'] < 18) | (df['Review_age'] > 75))
            )
            age_outlier_count = age_outliers.sum()
            
            if age_outlier_count > 0:
                self.logger.info(f"  年齡超出範圍 [18, 75]: {age_outlier_count:,} 行")
                outlier_mask |= age_outliers
        
        # 2. 政治立場異常值
        if 'Review_political' in df.columns:
            political_outliers = (
                df['Review_political'].notna() & 
                ((df['Review_political'] < 0) | (df['Review_political'] > 1))
            )
            political_outlier_count = political_outliers.sum()
            
            if political_outlier_count > 0:
                self.logger.info(f"  政治立場超出範圍 [0, 1]: {political_outlier_count:,} 行")
                outlier_mask |= political_outliers
        
        # 3. 宗教程度異常值
        if 'Review_religious' in df.columns:
            religious_outliers = (
                df['Review_religious'].notna() & 
                ((df['Review_religious'] < 0) | (df['Review_religious'] > 1))
            )
            religious_outlier_count = religious_outliers.sum()
            
            if religious_outlier_count > 0:
                self.logger.info(f"  宗教程度超出範圍 [0, 1]: {religious_outlier_count:,} 行")
                outlier_mask |= religious_outliers
        
        # 刪除異常值
        df_cleaned = df[~outlier_mask].copy()
        
        removed_rows = initial_rows - len(df_cleaned)
        removed_pct = (removed_rows / initial_rows * 100) if initial_rows > 0 else 0
        
        self.logger.info(f"\n✅ 刪除 {removed_rows:,} 行異常值 ({removed_pct:.2f}%)")
        self.logger.info(f"   剩餘 {len(df_cleaned):,} 行")
        
        # 記錄到清理日誌
        self.cleaning_log['steps'].append({
            'step': 'remove_outliers',
            'removed': int(removed_rows),
            'remaining': int(len(df_cleaned)),
            'removed_pct': f"{removed_pct:.2f}%"
        })
        
        return df_cleaned
    
    def _filter_law_vs_utility_scenarios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 3: 篩選「守法vs.效益」衝突情境
        
        篩選條件：
        1. ScenarioType == "Utilitarian" (有人數差異的場景)
        2. CrossingSignal in [1, 2] (有法律考量)
        3. DiffNumberOFCharacters > 0 (兩側人數確實有差異)
        """
        self.logger.info("\n" + "-" * 60)
        self.logger.info("【Step 3】篩選「守法vs.效益」衝突情境")
        self.logger.info("-" * 60)
        
        initial_rows = len(df)
        
        # 條件1: Utilitarian場景
        condition1 = df['ScenarioType'] == 'Utilitarian'
        self.logger.info(f"  條件1 - Utilitarian場景: {condition1.sum():,} 行")
        
        # 條件2: 有法律考量（綠燈或紅燈）
        condition2 = df['CrossingSignal'].isin([1, 2])
        self.logger.info(f"  條件2 - 有法律考量: {condition2.sum():,} 行")
        
        # 條件3: 兩側人數有差異
        condition3 = df['DiffNumberOFCharacters'] > 0
        self.logger.info(f"  條件3 - 人數有差異: {condition3.sum():,} 行")
        
        # 合併所有條件
        final_mask = condition1 & condition2 & condition3
        df_filtered = df[final_mask].copy()
        
        remaining_rows = len(df_filtered)
        remaining_pct = (remaining_rows / initial_rows * 100) if initial_rows > 0 else 0
        removed_rows = initial_rows - remaining_rows
        
        self.logger.info(f"\n✅ 保留 {remaining_rows:,} 行 ({remaining_pct:.2f}%)")
        self.logger.info(f"   過濾 {removed_rows:,} 行")
        
        # 記錄到清理日誌
        self.cleaning_log['steps'].append({
            'step': 'filter_law_vs_utility',
            'removed': int(removed_rows),
            'remaining': int(remaining_rows),
            'remaining_pct': f"{remaining_pct:.2f}%"
        })
        
        return df_filtered
    
    def _check_scenario_completeness(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4: 檢查場景完整性
        
        每個場景（ResponseID）應該有2行（兩個可能的結果）
        刪除只有1行的不完整場景
        """
        self.logger.info("\n" + "-" * 60)
        self.logger.info("【Step 4】檢查場景完整性")
        self.logger.info("-" * 60)
        
        initial_rows = len(df)
        
        # 計算每個ResponseID的出現次數
        response_counts = df['ResponseID'].value_counts()
        
        # 找出完整的場景（有2行的）
        complete_responses = response_counts[response_counts == 2].index
        
        # 找出不完整的場景
        incomplete_responses = response_counts[response_counts != 2]
        
        if len(incomplete_responses) > 0:
            self.logger.info(f"  發現 {len(incomplete_responses)} 個不完整場景")
            self.logger.info(f"  （涉及 {incomplete_responses.sum():,} 行資料）")
        
        # 只保留完整的場景
        df_complete = df[df['ResponseID'].isin(complete_responses)].copy()
        
        removed_rows = initial_rows - len(df_complete)
        removed_pct = (removed_rows / initial_rows * 100) if initial_rows > 0 else 0
        
        self.logger.info(f"\n✅ 刪除 {removed_rows:,} 行不完整場景 ({removed_pct:.2f}%)")
        self.logger.info(f"   剩餘 {len(df_complete):,} 行")
        self.logger.info(f"   完整場景數: {len(complete_responses):,}")
        
        # 記錄到清理日誌
        self.cleaning_log['steps'].append({
            'step': 'check_completeness',
            'removed': int(removed_rows),
            'remaining': int(len(df_complete)),
            'removed_pct': f"{removed_pct:.2f}%",
            'complete_scenarios': int(len(complete_responses))
        })
        
        return df_complete
    
    def _merge_cluster_info(self, 
                           df: pd.DataFrame,
                           cluster_map_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 5: 合併文化圈分類資訊
        
        將 country_cluster_map.csv 的 Cluster 資訊合併到主資料
        """
        self.logger.info("\n" + "-" * 60)
        self.logger.info("【Step 5】合併文化圈分類")
        self.logger.info("-" * 60)
        
        initial_rows = len(df)
        
        # 合併文化圈資訊
        df_merged = df.merge(
            cluster_map_df[['ISO3', 'Cluster']],
            left_on='UserCountry3',
            right_on='ISO3',
            how='left'
        )
        
        # 刪除重複的ISO3欄位
        if 'ISO3' in df_merged.columns:
            df_merged = df_merged.drop(columns=['ISO3'])
        
        # 檢查有多少國家沒有對應到文化圈
        missing_cluster = df_merged['Cluster'].isnull().sum()
        
        if missing_cluster > 0:
            missing_pct = (missing_cluster / len(df_merged) * 100)
            self.logger.warning(f"  ⚠️  {missing_cluster:,} 行無法對應到文化圈 ({missing_pct:.2f}%)")
            
            # 列出沒有對應的國家
            missing_countries = df_merged[df_merged['Cluster'].isnull()]['UserCountry3'].unique()
            if len(missing_countries) <= 10:
                self.logger.warning(f"  無法對應的國家: {list(missing_countries)}")
        else:
            self.logger.info("  ✓ 所有國家都成功對應到文化圈")
        
        # 將無法對應的國家標註為 -1（未分類）
        df_merged['Cluster'] = df_merged['Cluster'].fillna(-1).astype(int)
        
        # 檢查文化圈分佈
        if 'Cluster' in df_merged.columns:
            cluster_dist = df_merged['Cluster'].value_counts().sort_index()
            self.logger.info("\n  文化圈分佈:")
            cluster_names = {-1: 'Unclassified', 0: 'Western', 1: 'Eastern', 2: 'Southern'}
            for cluster, count in cluster_dist.items():
                cluster_name = cluster_names.get(int(cluster), f'Cluster {int(cluster)}')
                pct = count / len(df_merged) * 100
                self.logger.info(f"    {cluster_name}: {count:,} 行 ({pct:.1f}%)")
        
        self.logger.info(f"\n✅ 合併完成，維持 {len(df_merged):,} 行")
        
        # 記錄到清理日誌
        unclassified_count = int((df_merged['Cluster'] == -1).sum())
        self.cleaning_log['steps'].append({
            'step': 'merge_cluster',
            'unclassified_count': unclassified_count,
            'remaining': int(len(df_merged))
        })
        
        return df_merged
    
    def _print_cleaning_summary(self):
        """列印完整的清理摘要"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("資料清理完成摘要")
        self.logger.info("=" * 60)
        
        original = self.cleaning_log['original_rows']
        
        self.logger.info(f"\n【原始資料】{original:,} 行")
        
        for step in self.cleaning_log['steps']:
            step_name = step['step']
            removed = step.get('removed', 0)
            remaining = step['remaining']
            
            self.logger.info(f"\n【{step_name}】")
            if removed > 0:
                self.logger.info(f"  刪除: {removed:,} 行 ({step.get('removed_pct', 'N/A')})")
            self.logger.info(f"  剩餘: {remaining:,} 行")
            
            # 特殊資訊
            if 'complete_scenarios' in step:
                self.logger.info(f"  完整場景數: {step['complete_scenarios']:,}")
            if 'unclassified_count' in step:
                if step['unclassified_count'] > 0:
                    self.logger.info(f"  未分類國家: {step['unclassified_count']:,} 行 (標註為 Cluster=-1)")
        
        final_rows = self.cleaning_log['steps'][-1]['remaining']
        retention_rate = (final_rows / original * 100) if original > 0 else 0
        
        self.logger.info("\n" + "-" * 60)
        self.logger.info(f"【最終結果】")
        self.logger.info(f"  原始資料: {original:,} 行")
        self.logger.info(f"  清理後: {final_rows:,} 行")
        self.logger.info(f"  保留比例: {retention_rate:.2f}%")
        self.logger.info("=" * 60 + "\n")
    
    def get_cleaning_report(self) -> Dict:
        """
        取得清理報告
        
        Returns:
        --------
        Dict
            清理報告字典
        """
        return self.cleaning_log


def clean_survey_data(survey_df: pd.DataFrame,
                     cluster_map_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    便捷函數：清理問卷資料
    
    Parameters:
    -----------
    survey_df : pd.DataFrame
        原始問卷資料
    cluster_map_df : pd.DataFrame
        文化圈分類資料
        
    Returns:
    --------
    Tuple[pd.DataFrame, Dict]
        (清理後的資料, 清理報告)
    """
    cleaner = DataCleaner()
    cleaned_df = cleaner.clean_data(survey_df, cluster_map_df)
    report = cleaner.get_cleaning_report()
    
    return cleaned_df, report


# 測試模組
if __name__ == '__main__':
    print("資料清理模組載入成功")