"""
資料轉換模組：選項層級 → 場景層級
=======================================
用於第5章機器學習分析的資料前處理
本模組將選項層級資料（每場景2行）轉換為場景層級資料（每場景1行），
並提供特徵與目標變數的分離功能。

版本更新 (2024-11-04):
- 新增 conflict_only 參數，只保留衝突場景（守法 ≠ 多數）
- 修正驗證訊息，區分「預期的不一致」和「真正的錯誤」
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import warnings


class SceneLevelTransformer:
    """
    將選項層級資料（每場景2行）轉換為場景層級資料（每場景1行）
    
    原始資料結構：
    - 每個 ResponseID 對應2行，分別描述兩個選項
    - chose_lawful 在選項層級是按 (Saved==1) & (is_lawful==1) 計算
    - 因此同一 ResponseID 的兩行 chose_lawful 值會不同（這是預期行為）
    
    轉換後結構：
    - 每個 ResponseID 對應1行
    - chose_lawful 重新計算為「守法方是否被選中」
    """
    
    # 場景層級變數（不隨選項變化）
    SCENE_LEVEL_VARS = [
        'ResponseID',
        'ExtendedSessionID', 
        'UserID',
        'UserCountry3',
        'ScenarioOrder',
        'ScenarioType',
        'ScenarioTypeStrict',
        'PedPed',
        'DiffNumberOFCharacters',
        'DefaultChoice',
        'NonDefaultChoice',
        'DefaultChoiceIsOmission',
        'Template',
        # 使用者背景
        'Review_age',
        'Review_education',
        'Review_gender',
        'Review_income',
        'Review_political',
        'Review_religious',
        # 目標變數（將被重新計算）
        'chose_lawful',
        'chose_majority',
        'lawful_vs_majority_conflict',
        # 文化圈與國家特徵
        'Cluster',
        'has_country_features',
        'country_law_preference',
        'country_utilitarian',
        'country_intervention',
        'country_pedestrian_pref',
        'country_gender_pref',
        'country_fitness_pref',
        'country_status_pref',
        'country_age_pref',
        'country_species_pref',
    ]
    
    # 需要從選項層級聚合的變數
    OPTION_LEVEL_VARS = [
        'Intervention',  # 需要取守法方的 Intervention 值
        'NumberOfCharacters',  # 可能需要兩側的人數
    ]
    
    def __init__(self, verbose: bool = True):
        """
        初始化轉換器
        
        Parameters
        ----------
        verbose : bool
            是否顯示詳細資訊
        """
        self.verbose = verbose
        self.validation_results = {}
        
    def validate_option_level_data(self, data: pd.DataFrame) -> bool:
        """
        驗證選項層級資料的完整性
        
        Parameters
        ----------
        data : pd.DataFrame
            選項層級資料
            
        Returns
        -------
        bool
            驗證是否通過（僅檢查嚴重問題）
        """
        if self.verbose:
            print("=" * 60)
            print("資料驗證：選項層級資料完整性檢查")
            print("=" * 60)
        
        # 檢查1：每個 ResponseID 應該有2行
        rows_per_response = data.groupby('ResponseID').size()
        valid_responses = rows_per_response[rows_per_response == 2]
        invalid_responses = rows_per_response[rows_per_response != 2]
        
        if self.verbose:
            print(f"\n檢查1：每場景應有2行")
            print(f"  - 有效場景數（2行）: {len(valid_responses):,}")
            print(f"  - 無效場景數（非2行）: {len(invalid_responses):,}")
            if len(invalid_responses) > 0:
                print(f"  - ⚠️ 無效場景行數分佈: {invalid_responses.value_counts().to_dict()}")
        
        self.validation_results['valid_responses'] = len(valid_responses)
        self.validation_results['invalid_responses'] = len(invalid_responses)
        
        # 檢查2：chose_lawful 一致性（在選項層級預期會不一致）
        chose_lawful_check = data.groupby('ResponseID')['chose_lawful'].nunique()
        inconsistent_chose_lawful = chose_lawful_check[chose_lawful_check != 1]
        
        if self.verbose:
            print(f"\n檢查2：chose_lawful 一致性（選項層級）")
            print(f"  - 一致的場景數: {(chose_lawful_check == 1).sum():,}")
            print(f"  - 不一致的場景數: {len(inconsistent_chose_lawful):,}")
            if len(inconsistent_chose_lawful) > 0:
                print(f"  - ℹ️ 這是預期行為：chose_lawful 在特徵工程時按選項層級計算")
                print(f"       後續將重新計算正確的場景層級目標變數")
        
        self.validation_results['consistent_chose_lawful'] = (chose_lawful_check == 1).sum()
        self.validation_results['inconsistent_chose_lawful'] = len(inconsistent_chose_lawful)
        
        # 檢查3：is_lawful 在同一 ResponseID 應該有0和1各一個
        if 'is_lawful' in data.columns:
            is_lawful_check = data.groupby('ResponseID')['is_lawful'].apply(
                lambda x: set(x) == {0, 1}
            )
            valid_is_lawful = is_lawful_check.sum()
            
            if self.verbose:
                print(f"\n檢查3：is_lawful 配對完整性")
                print(f"  - 完整配對（0,1各一）: {valid_is_lawful:,}")
                print(f"  - 不完整配對: {(~is_lawful_check).sum():,}")
            
            self.validation_results['valid_is_lawful_pairs'] = valid_is_lawful
        
        # 總結驗證結果（只有場景行數不對才是嚴重問題）
        critical_issues = len(invalid_responses) > 0
        
        if self.verbose:
            if critical_issues:
                print(f"\n驗證結果: ❌ 有嚴重問題（部分場景行數不對）")
            else:
                print(f"\n驗證結果: ✅ 通過")
            print("=" * 60)
        
        return not critical_issues
    
    def transform(
        self, 
        data: pd.DataFrame,
        exclude_unclassified: bool = True,
        conflict_only: bool = True,
        add_intervention_feature: bool = True
    ) -> pd.DataFrame:
        """
        將選項層級資料（每場景2行）轉換為場景層級資料（每場景1行）
        
        Parameters
        ----------
        data : pd.DataFrame
            選項層級資料
        exclude_unclassified : bool
            是否排除 Cluster == -1 的未分類國家
        conflict_only : bool
            是否只保留衝突場景（守法 ≠ 多數）
            True = 只保留「守法少數 vs. 違法多數」的道德兩難場景
            False = 保留所有場景
        add_intervention_feature : bool
            是否添加 Intervention 特徵（守法方是否需要介入）
            
        Returns
        -------
        pd.DataFrame
            場景層級資料
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("資料轉換：選項層級 → 場景層級")
            print("=" * 60)
            print(f"輸入資料: {len(data):,} 行（選項層級）")
            print(f"\n轉換設定:")
            print(f"  - exclude_unclassified: {exclude_unclassified}")
            print(f"  - conflict_only: {conflict_only}")
            print(f"  - add_intervention_feature: {add_intervention_feature}")
        
        # 先驗證資料
        self.validate_option_level_data(data)
        
        # 複製資料避免修改原始資料
        df = data.copy()
        
        # 排除未分類國家
        if exclude_unclassified and 'Cluster' in df.columns:
            before_filter = len(df)
            df = df[df['Cluster'] != -1]
            if self.verbose:
                print(f"\n排除 Cluster == -1: {before_filter:,} → {len(df):,} 行")
        
        # 只保留衝突場景（守法 ≠ 多數）
        if conflict_only and 'lawful_vs_majority_conflict' in df.columns:
            before_conflict = len(df)
            # 衝突場景：守法方是少數，違法方是多數
            df = df[df['lawful_vs_majority_conflict'] == 1]
            if self.verbose:
                print(f"只保留衝突場景: {before_conflict:,} → {len(df):,} 行")
                print(f"  （篩選條件: lawful_vs_majority_conflict == 1）")
                print(f"  （意義: 守法少數 vs. 違法多數 的道德兩難）")
        
        # 只保留有效場景（每個 ResponseID 有2行）
        rows_per_response = df.groupby('ResponseID').size()
        valid_response_ids = rows_per_response[rows_per_response == 2].index
        df = df[df['ResponseID'].isin(valid_response_ids)]
        
        if self.verbose:
            print(f"保留有效場景（2行）: {len(df):,} 行")
        
        # ========================================
        # 🔧 關鍵：重新計算場景層級的 chose_lawful 和 chose_majority
        # ========================================
        if self.verbose:
            print("\n重新計算場景層級的目標變數...")
        
        # chose_lawful: 找到守法方(is_lawful=1)的 Saved 值
        # 意義：使用者是否選擇拯救守法方（即使他們是少數）
        scene_chose_lawful = df[df['is_lawful'] == 1].groupby('ResponseID')['Saved'].first()
        scene_chose_lawful = scene_chose_lawful.reset_index()
        scene_chose_lawful.columns = ['ResponseID', 'chose_lawful_scene']
        
        # chose_majority: 找到多數方(is_majority=1)的 Saved 值
        # 意義：使用者是否選擇拯救多數
        scene_chose_majority = df[df['is_majority'] == 1].groupby('ResponseID')['Saved'].first()
        scene_chose_majority = scene_chose_majority.reset_index()
        scene_chose_majority.columns = ['ResponseID', 'chose_majority_scene']
        
        # ========================================
        
        # 確定可用的場景層級變數（排除需要重新計算的，以及 ResponseID）
        vars_to_exclude = ['chose_lawful', 'chose_majority', 'ResponseID']
        available_scene_vars = [
            col for col in self.SCENE_LEVEL_VARS 
            if col in df.columns and col not in vars_to_exclude
        ]
        
        if self.verbose:
            print(f"\n可用場景層級變數: {len(available_scene_vars)} 個")
        
        # 按 ResponseID 分組，取第一行（場景層級變數在兩行相同）
        # 使用 reset_index() 讓 ResponseID 成為欄位
        scene_data = df.groupby('ResponseID')[available_scene_vars].first().reset_index()
        
        # 合併正確的 chose_lawful 和 chose_majority
        scene_data = scene_data.merge(scene_chose_lawful, on='ResponseID', how='left')
        scene_data = scene_data.merge(scene_chose_majority, on='ResponseID', how='left')
        
        # 重命名為標準欄位名並轉換為整數
        scene_data['chose_lawful'] = scene_data['chose_lawful_scene'].astype(int)
        scene_data['chose_majority'] = scene_data['chose_majority_scene'].astype(int)
        scene_data = scene_data.drop(columns=['chose_lawful_scene', 'chose_majority_scene'])
        
        # 計算統計
        lawful_rate = scene_data['chose_lawful'].mean()
        majority_rate = scene_data['chose_majority'].mean()
        
        if self.verbose:
            print(f"\n修正後目標變數統計:")
            print(f"  - chose_lawful=0 (選效益/多數): {(scene_data['chose_lawful']==0).sum():,}")
            print(f"  - chose_lawful=1 (選守法/少數): {(scene_data['chose_lawful']==1).sum():,}")
            print(f"  - 守法選擇率: {lawful_rate*100:.1f}%")
            print(f"  - 多數選擇率: {majority_rate*100:.1f}%")
            
            # 驗證：在衝突場景中，chose_lawful + chose_majority 應該 = 1
            if conflict_only:
                both_check = scene_data['chose_lawful'] + scene_data['chose_majority']
                if (both_check == 1).all():
                    print(f"  ✅ 驗證通過: chose_lawful + chose_majority = 1（互斥）")
                else:
                    print(f"  ⚠️ 驗證異常: 有 {(both_check != 1).sum()} 筆不符合互斥條件")
        
        # 添加 Intervention 特徵：守法方是否需要介入才能拯救
        if add_intervention_feature and 'is_lawful' in df.columns and 'Intervention' in df.columns:
            lawful_intervention = df[df['is_lawful'] == 1].groupby('ResponseID')['Intervention'].first()
            lawful_intervention = lawful_intervention.reset_index()
            lawful_intervention.columns = ['ResponseID', 'lawful_requires_intervention']
            
            scene_data = scene_data.merge(lawful_intervention, on='ResponseID', how='left')
            
            if self.verbose:
                intervention_rate = scene_data['lawful_requires_intervention'].mean()
                print(f"\n添加特徵: lawful_requires_intervention")
                print(f"  - 需要介入才能救守法方的比例: {intervention_rate*100:.1f}%")
        
        if self.verbose:
            print(f"\n輸出資料: {len(scene_data):,} 行（場景層級）")
            print(f"欄位數: {len(scene_data.columns)}")
            print("=" * 60)
        
        return scene_data
    
    def get_feature_target_split(
        self, 
        scene_data: pd.DataFrame,
        target_col: str = 'chose_lawful',
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        分離特徵與目標變數
        
        Parameters
        ----------
        scene_data : pd.DataFrame
            場景層級資料
        target_col : str
            目標變數欄位名
        feature_cols : List[str], optional
            特徵欄位列表，若為 None 則使用預設特徵
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            (特徵 DataFrame, 目標 Series)
        """
        if feature_cols is None:
            # 預設特徵組合（根據知識庫設計）
            feature_cols = [
                # 場景結構
                'DiffNumberOFCharacters',
                'PedPed',
                # 使用者特徵
                'Review_age',
                'Review_political',
                'Review_religious',
                # 文化圈（將進行 One-Hot 編碼）
                'Cluster',
                # 國家層級特徵
                'country_law_preference',
                'country_utilitarian',
            ]
            
            # 添加 Intervention 特徵（如果存在）
            if 'lawful_requires_intervention' in scene_data.columns:
                feature_cols.append('lawful_requires_intervention')
        
        # 過濾可用特徵
        available_features = [col for col in feature_cols if col in scene_data.columns]
        
        if self.verbose:
            print(f"\n特徵選擇:")
            print(f"  - 請求特徵: {len(feature_cols)} 個")
            print(f"  - 可用特徵: {len(available_features)} 個")
            missing = set(feature_cols) - set(available_features)
            if missing:
                print(f"  - 缺失特徵: {missing}")
        
        X = scene_data[available_features].copy()
        y = scene_data[target_col].copy()
        
        return X, y


def prepare_features_for_xgboost(
    X: pd.DataFrame,
    cluster_onehot: bool = True
) -> pd.DataFrame:
    """
    為 XGBoost 準備特徵（One-Hot 編碼等）
    
    Parameters
    ----------
    X : pd.DataFrame
        原始特徵 DataFrame
    cluster_onehot : bool
        是否對 Cluster 進行 One-Hot 編碼
        
    Returns
    -------
    pd.DataFrame
        處理後的特徵 DataFrame
    """
    X_processed = X.copy()
    
    # Cluster One-Hot 編碼
    if cluster_onehot and 'Cluster' in X_processed.columns:
        # 創建虛擬變數，以 Western (0) 為參照組
        cluster_dummies = pd.get_dummies(
            X_processed['Cluster'], 
            prefix='Cluster',
            drop_first=False  # 保留所有類別，便於 SHAP 解釋
        )
        # 重命名為更有意義的名稱
        cluster_dummies.columns = [
            'Cluster_Western' if c == 'Cluster_0' else
            'Cluster_Eastern' if c == 'Cluster_1' else
            'Cluster_Southern' if c == 'Cluster_2' else c
            for c in cluster_dummies.columns
        ]
        
        # 移除原始 Cluster 欄位，添加虛擬變數
        X_processed = X_processed.drop(columns=['Cluster'])
        X_processed = pd.concat([X_processed, cluster_dummies], axis=1)
    
    return X_processed


# 便利函數
def load_and_transform_data(
    train_path: str,
    test_path: str,
    conflict_only: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    載入並轉換訓練集與測試集
    
    Parameters
    ----------
    train_path : str
        訓練集路徑
    test_path : str
        測試集路徑
    conflict_only : bool
        是否只保留衝突場景（預設 True）
    verbose : bool
        是否顯示詳細資訊
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (訓練集場景層級, 測試集場景層級)
    """
    transformer = SceneLevelTransformer(verbose=verbose)
    
    # 載入資料
    if verbose:
        print("載入訓練集...")
    train_data = pd.read_csv(train_path)
    
    if verbose:
        print("載入測試集...")
    test_data = pd.read_csv(test_path)
    
    # 轉換
    if verbose:
        print("\n" + "=" * 60)
        print("轉換訓練集")
    train_scene = transformer.transform(train_data, conflict_only=conflict_only)
    
    if verbose:
        print("\n" + "=" * 60)
        print("轉換測試集")
    test_scene = transformer.transform(test_data, conflict_only=conflict_only)
    
    return train_scene, test_scene


if __name__ == "__main__":
    # 測試程式碼
    print("資料轉換模組測試")
    print("請使用 load_and_transform_data() 函數載入並轉換資料")