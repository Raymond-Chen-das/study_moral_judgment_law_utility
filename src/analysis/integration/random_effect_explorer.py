"""
HLM 隨機效應探索模組
======================
用於第5.1節：國家層級變異來源分析

核心問題：ICC=14.35%的國家層級變異從何而來？
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class RandomEffectExplorer:
    """
    HLM 隨機效應與 AMCE 維度的相關性分析
    
    探索國家層級道德判斷差異的來源
    """
    
    # AMCE 維度名稱對應
    AMCE_COLUMNS = {
        '[Omission -> Commission]: Estimates': 'intervention',
        '[Passengers -> Pedestrians]: Estimates': 'pedestrian_pref',
        'Law [Illegal -> Legal]: Estimates': 'law_preference',
        'Gender [Male -> Female]: Estimates': 'gender_pref',
        'Fitness [Large -> Fit]: Estimates': 'fitness_pref',
        'Social Status [Low -> High]: Estimates': 'status_pref',
        'Age [Elderly -> Young]: Estimates': 'age_pref',
        'No. Characters [Less -> More]: Estimates': 'utilitarian',
        'Species [Pets -> Humans]: Estimates': 'species_pref'
    }
    
    # 維度的中文名稱
    AMCE_CHINESE = {
        'intervention': '介入偏好',
        'pedestrian_pref': '行人優先',
        'law_preference': '守法偏好',
        'gender_pref': '性別偏好',
        'fitness_pref': '體型偏好',
        'status_pref': '地位偏好',
        'age_pref': '年齡偏好',
        'utilitarian': '效益主義',
        'species_pref': '物種偏好'
    }
    
    def __init__(self, verbose: bool = True):
        """
        初始化探索器
        
        Parameters
        ----------
        verbose : bool
            是否顯示詳細資訊
        """
        self.verbose = verbose
        self.random_effects = None
        self.amce_data = None
        self.merged_data = None
        self.correlation_results = None
        
    def load_data(
        self,
        random_effects_path: str,
        amce_path: str
    ) -> pd.DataFrame:
        """
        載入隨機效應與 AMCE 資料
        
        Parameters
        ----------
        random_effects_path : str
            HLM 隨機效應檔案路徑
        amce_path : str
            CountriesChangePr.csv 路徑
            
        Returns
        -------
        pd.DataFrame
            合併後的資料
        """
        if self.verbose:
            print("=" * 60)
            print("5.1 國家層級變異來源探索")
            print("=" * 60)
        
        # 載入隨機效應
        self.random_effects = pd.read_csv(random_effects_path)
        if self.verbose:
            print(f"\n載入隨機效應: {len(self.random_effects)} 個國家")
            print(f"  隨機效應範圍: [{self.random_effects['random_intercept'].min():.3f}, "
                  f"{self.random_effects['random_intercept'].max():.3f}]")
        
        # 載入 AMCE
        amce_raw = pd.read_csv(amce_path, index_col=0)
        if self.verbose:
            print(f"\n載入 AMCE: {len(amce_raw)} 個國家 × {len(amce_raw.columns)} 個指標")
        
        # 重新命名 AMCE 欄位
        amce_renamed = {}
        for old_col, new_col in self.AMCE_COLUMNS.items():
            if old_col in amce_raw.columns:
                amce_renamed[new_col] = amce_raw[old_col]
        
        self.amce_data = pd.DataFrame(amce_renamed)
        self.amce_data['UserCountry3'] = amce_raw.index
        self.amce_data = self.amce_data.reset_index(drop=True)
        
        # 合併資料
        self.merged_data = self.random_effects.merge(
            self.amce_data,
            on='UserCountry3',
            how='inner'
        )
        
        if self.verbose:
            print(f"\n合併後: {len(self.merged_data)} 個國家")
            if len(self.merged_data) < len(self.random_effects):
                missing = set(self.random_effects['UserCountry3']) - set(self.merged_data['UserCountry3'])
                print(f"  缺失國家: {missing}")
        
        return self.merged_data
    
    def compute_correlations(self) -> pd.DataFrame:
        """
        計算隨機效應與各 AMCE 維度的相關係數
        
        Returns
        -------
        pd.DataFrame
            相關分析結果
        """
        if self.merged_data is None:
            raise ValueError("請先呼叫 load_data()")
        
        if self.verbose:
            print("\n" + "-" * 60)
            print("相關分析：隨機效應 vs. AMCE 維度")
            print("-" * 60)
        
        results = []
        amce_cols = [col for col in self.AMCE_COLUMNS.values() if col in self.merged_data.columns]
        
        for col in amce_cols:
            # Pearson 相關
            r, p = stats.pearsonr(
                self.merged_data['random_intercept'],
                self.merged_data[col]
            )
            
            # Spearman 相關（穩健性檢驗）
            rho, p_spearman = stats.spearmanr(
                self.merged_data['random_intercept'],
                self.merged_data[col]
            )
            
            # 效果量解釋
            if abs(r) >= 0.5:
                effect_size = '大'
            elif abs(r) >= 0.3:
                effect_size = '中'
            elif abs(r) >= 0.1:
                effect_size = '小'
            else:
                effect_size = '極微'
            
            results.append({
                'amce_dimension': col,
                'chinese_name': self.AMCE_CHINESE.get(col, col),
                'pearson_r': r,
                'pearson_p': p,
                'spearman_rho': rho,
                'spearman_p': p_spearman,
                'effect_size': effect_size,
                'significant_05': p < 0.05,
                'significant_01': p < 0.01,
                'significant_001': p < 0.001
            })
        
        self.correlation_results = pd.DataFrame(results)
        self.correlation_results = self.correlation_results.sort_values(
            'pearson_r', ascending=False, key=abs
        ).reset_index(drop=True)
        
        if self.verbose:
            print("\n相關係數排序（按絕對值）：")
            print("-" * 80)
            print(f"{'維度':<15} {'中文':<10} {'Pearson r':>10} {'p值':>12} {'效果量':>8}")
            print("-" * 80)
            for _, row in self.correlation_results.iterrows():
                sig = '***' if row['significant_001'] else ('**' if row['significant_01'] else ('*' if row['significant_05'] else ''))
                print(f"{row['amce_dimension']:<15} {row['chinese_name']:<10} "
                      f"{row['pearson_r']:>10.3f} {row['pearson_p']:>10.4f}{sig:<2} {row['effect_size']:>8}")
            print("-" * 80)
            print("註：* p<.05, ** p<.01, *** p<.001")
        
        return self.correlation_results
    
    def get_top_correlates(self, n: int = 3) -> List[str]:
        """
        獲取與隨機效應最相關的前 N 個 AMCE 維度
        
        Parameters
        ----------
        n : int
            返回數量
            
        Returns
        -------
        List[str]
            維度名稱列表
        """
        if self.correlation_results is None:
            self.compute_correlations()
        
        return self.correlation_results.head(n)['amce_dimension'].tolist()
    
    def compute_regression(self) -> Dict[str, Any]:
        """
        多元迴歸：以 AMCE 維度預測隨機效應
        
        Returns
        -------
        Dict[str, Any]
            迴歸結果
        """
        if self.merged_data is None:
            raise ValueError("請先呼叫 load_data()")
        
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_squared_error
        
        if self.verbose:
            print("\n" + "-" * 60)
            print("多元迴歸：AMCE 維度 → 隨機效應")
            print("-" * 60)
        
        # 準備資料
        amce_cols = [col for col in self.AMCE_COLUMNS.values() if col in self.merged_data.columns]
        X = self.merged_data[amce_cols]
        y = self.merged_data['random_intercept']
        
        # 迴歸
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # 計算 R²
        r2 = r2_score(y, y_pred)
        adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - len(amce_cols) - 1)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # 係數
        coefficients = pd.DataFrame({
            'feature': amce_cols,
            'chinese_name': [self.AMCE_CHINESE.get(col, col) for col in amce_cols],
            'coefficient': model.coef_,
            'abs_coefficient': np.abs(model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        results = {
            'r2': r2,
            'adj_r2': adj_r2,
            'rmse': rmse,
            'coefficients': coefficients,
            'intercept': model.intercept_
        }
        
        if self.verbose:
            print(f"\nR² = {r2:.4f}")
            print(f"Adjusted R² = {adj_r2:.4f}")
            print(f"RMSE = {rmse:.4f}")
            print(f"\n解釋：{r2*100:.1f}% 的國家層級變異可由 9 個 AMCE 維度解釋")
            
            print("\n迴歸係數（按絕對值排序）：")
            for _, row in coefficients.head(5).iterrows():
                print(f"  {row['chinese_name']}: {row['coefficient']:.4f}")
        
        return results
    
    def get_scatter_data(
        self, 
        amce_dimension: str = 'law_preference'
    ) -> pd.DataFrame:
        """
        獲取散點圖資料
        
        Parameters
        ----------
        amce_dimension : str
            AMCE 維度名稱
            
        Returns
        -------
        pd.DataFrame
            散點圖資料（含國家標籤）
        """
        if self.merged_data is None:
            raise ValueError("請先呼叫 load_data()")
        
        return self.merged_data[['UserCountry3', 'random_intercept', amce_dimension]].copy()
    
    def generate_interpretation(self) -> str:
        """
        生成分析解釋文字
        
        Returns
        -------
        str
            解釋文字
        """
        if self.correlation_results is None:
            self.compute_correlations()
        
        # 找出最強相關
        top_corr = self.correlation_results.iloc[0]
        
        interpretation = []
        interpretation.append("### 5.1 國家層級變異來源分析\n")
        interpretation.append(f"**研究問題**：HLM 發現 ICC=14.35%，即 14.35% 的守法選擇變異來自國家層級。")
        interpretation.append("這些變異從何而來？\n")
        
        interpretation.append("**分析方法**：計算 HLM 隨機效應（各國對全球平均的偏離）")
        interpretation.append("與 9 個 AMCE 維度（國家道德偏好）的相關係數。\n")
        
        interpretation.append("**主要發現**：")
        interpretation.append(f"- 與隨機效應最強相關的維度是「{top_corr['chinese_name']}」")
        interpretation.append(f"  (r = {top_corr['pearson_r']:.3f}, p < .001)")
        
        significant_dims = self.correlation_results[self.correlation_results['significant_001']]
        interpretation.append(f"- 共有 {len(significant_dims)} 個維度達 p < .001 顯著水準")
        
        return "\n".join(interpretation)


def load_and_analyze(
    random_effects_path: str,
    amce_path: str,
    verbose: bool = True
) -> Tuple[RandomEffectExplorer, pd.DataFrame, Dict[str, Any]]:
    """
    一鍵載入與分析
    
    Parameters
    ----------
    random_effects_path : str
        HLM 隨機效應檔案路徑
    amce_path : str
        CountriesChangePr.csv 路徑
    verbose : bool
        是否顯示詳細資訊
        
    Returns
    -------
    Tuple
        (探索器實例, 相關分析結果, 迴歸結果)
    """
    explorer = RandomEffectExplorer(verbose=verbose)
    explorer.load_data(random_effects_path, amce_path)
    corr_results = explorer.compute_correlations()
    reg_results = explorer.compute_regression()
    
    return explorer, corr_results, reg_results


if __name__ == "__main__":
    print("HLM 隨機效應探索模組")
    print("請使用 load_and_analyze() 函數進行分析")