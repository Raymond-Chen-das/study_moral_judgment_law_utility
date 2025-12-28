"""
SHAP 可解釋性分析模組
======================
用於第5章機器學習模型的可解釋性分析
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import shap
import warnings
warnings.filterwarnings('ignore')


class SHAPAnalyzer:
    """
    SHAP 可解釋性分析器
    
    提供特徵重要性排序與依賴性分析
    """
    
    def __init__(
        self,
        model,
        feature_names: List[str],
        verbose: bool = True
    ):
        """
        初始化 SHAP 分析器
        
        Parameters
        ----------
        model : XGBoost model
            已訓練的 XGBoost 模型
        feature_names : List[str]
            特徵名稱列表
        verbose : bool
            是否顯示詳細資訊
        """
        self.model = model
        self.feature_names = feature_names
        self.verbose = verbose
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        
    def compute_shap_values(
        self, 
        X: pd.DataFrame,
        sample_size: Optional[int] = None
    ) -> np.ndarray:
        """
        計算 SHAP 值
        
        Parameters
        ----------
        X : pd.DataFrame
            特徵資料
        sample_size : int, optional
            抽樣大小（大資料集時使用）
            
        Returns
        -------
        np.ndarray
            SHAP 值矩陣
        """
        if self.verbose:
            print("=" * 60)
            print("SHAP 分析")
            print("=" * 60)
        
        # 抽樣（如果需要）
        if sample_size is not None and len(X) > sample_size:
            if self.verbose:
                print(f"\n資料抽樣: {len(X):,} → {sample_size:,}")
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X
            if self.verbose:
                print(f"\n分析樣本數: {len(X_sample):,}")
        
        # 創建 SHAP 解釋器
        if self.verbose:
            print("計算 SHAP 值中...")
        
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(X_sample)
        self.expected_value = self.explainer.expected_value
        
        # 保存用於視覺化的資料
        self.X_sample = X_sample
        
        if self.verbose:
            print(f"完成！SHAP 值矩陣大小: {self.shap_values.shape}")
        
        return self.shap_values
    
    def get_feature_importance(
        self, 
        method: str = 'mean_abs'
    ) -> pd.DataFrame:
        """
        計算基於 SHAP 的特徵重要性
        
        Parameters
        ----------
        method : str
            計算方法：'mean_abs'（平均絕對值）或 'mean'（平均值）
            
        Returns
        -------
        pd.DataFrame
            特徵重要性表格
        """
        if self.shap_values is None:
            raise ValueError("請先呼叫 compute_shap_values()")
        
        if method == 'mean_abs':
            importance = np.abs(self.shap_values).mean(axis=0)
        elif method == 'mean':
            importance = self.shap_values.mean(axis=0)
        else:
            raise ValueError(f"不支援的方法: {method}")
        
        # 創建 DataFrame
        df = pd.DataFrame({
            'feature': self.feature_names,
            'shap_importance': importance
        })
        
        # 標準化
        df['importance_normalized'] = df['shap_importance'] / df['shap_importance'].sum()
        
        # 排序
        df = df.sort_values('shap_importance', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        
        # 添加方向（正向/負向對 chose_lawful 的平均影響）
        mean_shap = self.shap_values.mean(axis=0)
        df['mean_shap'] = [mean_shap[self.feature_names.index(f)] for f in df['feature']]
        df['direction'] = df['mean_shap'].apply(lambda x: '↑守法' if x > 0 else '↓效益')
        
        return df
    
    def get_feature_summary(self) -> pd.DataFrame:
        """
        生成特徵摘要報告
        
        Returns
        -------
        pd.DataFrame
            特徵摘要表格
        """
        importance_df = self.get_feature_importance()
        
        # 添加統計資訊
        summaries = []
        for i, row in importance_df.iterrows():
            feat = row['feature']
            feat_idx = self.feature_names.index(feat)
            shap_vals = self.shap_values[:, feat_idx]
            
            summaries.append({
                'rank': row['rank'],
                'feature': feat,
                'shap_importance': row['shap_importance'],
                'importance_pct': f"{row['importance_normalized']*100:.1f}%",
                'direction': row['direction'],
                'shap_mean': shap_vals.mean(),
                'shap_std': shap_vals.std(),
                'shap_min': shap_vals.min(),
                'shap_max': shap_vals.max()
            })
        
        return pd.DataFrame(summaries)
    
    def get_dependence_data(
        self, 
        feature: str,
        interaction_feature: Optional[str] = None
    ) -> pd.DataFrame:
        """
        獲取特徵依賴性資料（用於視覺化）
        
        Parameters
        ----------
        feature : str
            主要特徵名稱
        interaction_feature : str, optional
            交互作用特徵名稱
            
        Returns
        -------
        pd.DataFrame
            依賴性資料
        """
        if self.shap_values is None or self.X_sample is None:
            raise ValueError("請先呼叫 compute_shap_values()")
        
        feat_idx = self.feature_names.index(feature)
        
        data = {
            'feature_value': self.X_sample[feature].values,
            'shap_value': self.shap_values[:, feat_idx]
        }
        
        if interaction_feature is not None:
            data['interaction_value'] = self.X_sample[interaction_feature].values
        
        return pd.DataFrame(data)
    
    def compare_with_chapter4(
        self,
        chapter4_effects: Dict[str, float]
    ) -> pd.DataFrame:
        """
        比較 SHAP 重要性與第4章效果量
        
        Parameters
        ----------
        chapter4_effects : Dict[str, float]
            第4章的效果量（如 Odds Ratio）
            
        Returns
        -------
        pd.DataFrame
            比較表格
        """
        importance_df = self.get_feature_importance()
        
        # 映射特徵名稱（處理可能的命名差異）
        feature_mapping = {
            'Cluster_Eastern': 'Cluster (Eastern vs Western)',
            'Cluster_Southern': 'Cluster (Southern vs Western)',
            'Review_age': 'Review_age',
            'Review_political': 'Review_political',
            'Review_religious': 'Review_religious',
            'country_law_preference': 'country_law_preference',
            'country_utilitarian': 'country_utilitarian',
            'DiffNumberOFCharacters': 'DiffNumberOFCharacters',
            'PedPed': 'PedPed',
        }
        
        comparisons = []
        for _, row in importance_df.iterrows():
            feat = row['feature']
            mapped_feat = feature_mapping.get(feat, feat)
            
            comparison = {
                'feature': feat,
                'shap_rank': row['rank'],
                'shap_importance': row['shap_importance'],
                'chapter4_effect': chapter4_effects.get(mapped_feat, np.nan),
                'direction': row['direction']
            }
            comparisons.append(comparison)
        
        return pd.DataFrame(comparisons)
    
    def generate_interpretation(self) -> str:
        """
        生成 SHAP 分析的文字解釋
        
        Returns
        -------
        str
            解釋文字
        """
        importance_df = self.get_feature_importance()
        top_features = importance_df.head(5)
        
        interpretation = []
        interpretation.append("### SHAP 特徵重要性解釋\n")
        interpretation.append(f"基於 {len(self.X_sample):,} 筆樣本的 SHAP 分析，")
        interpretation.append("各特徵對「選擇守法」預測的貢獻如下：\n")
        
        for i, row in top_features.iterrows():
            feat_name = row['feature']
            importance_pct = row['importance_normalized'] * 100
            direction = row['direction']
            
            interpretation.append(
                f"**{row['rank']}. {feat_name}** (貢獻度: {importance_pct:.1f}%)\n"
                f"   - 平均影響方向: {direction}\n"
            )
        
        return "\n".join(interpretation)


def prepare_chapter4_comparison_data() -> Dict[str, float]:
    """
    準備第4章效果量資料（用於與 SHAP 比較）
    
    Returns
    -------
    Dict[str, float]
        特徵名稱 → Odds Ratio
    """
    # 來自第4章邏輯迴歸結果
    return {
        'Cluster (Eastern vs Western)': 1.115,
        'Cluster (Southern vs Western)': 1.007,
        'Review_age': 0.997,
        'Review_political': 0.963,
        'Review_religious': 0.931,
        # 場景特徵在第4章未納入邏輯迴歸
        'DiffNumberOFCharacters': None,
        'PedPed': None,
        # 國家層級特徵（HLM結果）
        'country_utilitarian': 'not significant (p=.530)',
    }


if __name__ == "__main__":
    print("SHAP 分析器模組")
    print("請使用 SHAPAnalyzer 類別進行可解釋性分析")