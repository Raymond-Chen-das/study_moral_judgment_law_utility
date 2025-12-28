"""
XGBoost 分類器模組
==================
用於第5章機器學習分析的 XGBoost 分類器實現
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import pickle
from datetime import datetime

# XGBoost
import xgboost as xgb

# Sklearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

import warnings
warnings.filterwarnings('ignore')


class MoralChoiceXGBoostClassifier:
    """
    道德選擇預測的 XGBoost 分類器
    
    針對「守法 vs. 效益」選擇的二元分類問題設計
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: bool = True
    ):
        """
        初始化分類器
        
        Parameters
        ----------
        n_estimators : int
            樹的數量
        max_depth : int
            樹的最大深度
        learning_rate : float
            學習率
        subsample : float
            每棵樹的樣本抽樣比例
        colsample_bytree : float
            每棵樹的特徵抽樣比例
        random_state : int
            隨機種子
        n_jobs : int
            並行處理數（-1表示使用所有CPU）
        verbose : bool
            是否顯示詳細資訊
        """
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
        }
        
        self.model = None
        self.feature_names = None
        self.training_info = {}
        self.verbose = verbose
        
    def fit(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: Optional[int] = 10
    ) -> 'MoralChoiceXGBoostClassifier':
        """
        訓練模型
        
        Parameters
        ----------
        X_train : pd.DataFrame
            訓練特徵
        y_train : pd.Series
            訓練標籤
        X_val : pd.DataFrame, optional
            驗證特徵（用於早停）
        y_val : pd.Series, optional
            驗證標籤
        early_stopping_rounds : int, optional
            早停輪數
            
        Returns
        -------
        self
        """
        if self.verbose:
            print("=" * 60)
            print("XGBoost 模型訓練")
            print("=" * 60)
            print(f"\n訓練集大小: {len(X_train):,} 筆")
            print(f"特徵數量: {X_train.shape[1]}")
            print(f"目標變數分佈:")
            print(f"  - chose_lawful=0: {(y_train == 0).sum():,} ({(y_train == 0).mean():.1%})")
            print(f"  - chose_lawful=1: {(y_train == 1).sum():,} ({(y_train == 1).mean():.1%})")
        
        # 儲存特徵名稱
        self.feature_names = list(X_train.columns)
        
        # 處理缺失值標記（XGBoost 可以處理 NaN）
        X_train_clean = X_train.copy()
        
        # 創建模型
        self.model = xgb.XGBClassifier(**self.params)
        
        # 訓練
        start_time = datetime.now()
        
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train_clean, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train_clean, y_train, verbose=False)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # 儲存訓練資訊
        self.training_info = {
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'feature_names': self.feature_names,
            'training_time_seconds': training_time,
            'params': self.params,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.verbose:
            print(f"\n訓練完成！耗時: {training_time:.2f} 秒")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        預測類別
        
        Parameters
        ----------
        X : pd.DataFrame
            特徵資料
            
        Returns
        -------
        np.ndarray
            預測類別（0 或 1）
        """
        if self.model is None:
            raise ValueError("模型尚未訓練！請先呼叫 fit() 方法")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        預測機率
        
        Parameters
        ----------
        X : pd.DataFrame
            特徵資料
            
        Returns
        -------
        np.ndarray
            預測機率 (n_samples, 2)
        """
        if self.model is None:
            raise ValueError("模型尚未訓練！請先呼叫 fit() 方法")
        
        return self.model.predict_proba(X)
    
    def evaluate(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        評估模型性能
        
        Parameters
        ----------
        X_test : pd.DataFrame
            測試特徵
        y_test : pd.Series
            測試標籤
        return_predictions : bool
            是否返回預測值
            
        Returns
        -------
        Dict[str, Any]
            評估指標字典
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("模型評估")
            print("=" * 60)
        
        # 預測
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)[:, 1]
        
        # 計算指標
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'n_test_samples': len(y_test),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # 計算 ROC 曲線數據
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        metrics['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
        
        if self.verbose:
            print(f"\n測試集大小: {len(y_test):,} 筆")
            print(f"\n性能指標:")
            print(f"  - Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  - Precision: {metrics['precision']:.4f}")
            print(f"  - Recall:    {metrics['recall']:.4f}")
            print(f"  - F1 Score:  {metrics['f1']:.4f}")
            print(f"  - ROC-AUC:   {metrics['roc_auc']:.4f}")
            
            print(f"\n混淆矩陣:")
            cm = np.array(metrics['confusion_matrix'])
            print(f"              預測=0    預測=1")
            print(f"  實際=0    {cm[0,0]:>8,}  {cm[0,1]:>8,}")
            print(f"  實際=1    {cm[1,0]:>8,}  {cm[1,1]:>8,}")
        
        if return_predictions:
            metrics['y_pred'] = y_pred
            metrics['y_prob'] = y_prob
        
        return metrics
    
    def cross_validate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        交叉驗證
        
        Parameters
        ----------
        X : pd.DataFrame
            特徵資料
        y : pd.Series
            標籤資料
        cv : int
            折數
            
        Returns
        -------
        Dict[str, Any]
            交叉驗證結果
        """
        if self.verbose:
            print(f"\n{cv}-折交叉驗證...")
        
        # 創建新模型進行交叉驗證
        cv_model = xgb.XGBClassifier(**self.params)
        
        # 分層抽樣
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.params['random_state'])
        
        # 計算各指標
        cv_results = {}
        
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            scores = cross_val_score(
                cv_model, X, y, 
                cv=skf, 
                scoring=metric if metric != 'precision' else 'precision_macro'
            )
            cv_results[f'{metric}_mean'] = scores.mean()
            cv_results[f'{metric}_std'] = scores.std()
        
        if self.verbose:
            print(f"\n交叉驗證結果 ({cv}-fold):")
            print(f"  - Accuracy:  {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
            print(f"  - ROC-AUC:   {cv_results['roc_auc_mean']:.4f} ± {cv_results['roc_auc_std']:.4f}")
        
        return cv_results
    
    def get_feature_importance(
        self, 
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """
        獲取特徵重要性
        
        Parameters
        ----------
        importance_type : str
            重要性類型: 'gain', 'weight', 'cover'
            
        Returns
        -------
        pd.DataFrame
            特徵重要性表格
        """
        if self.model is None:
            raise ValueError("模型尚未訓練！")
        
        # 獲取重要性
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        
        # 轉換為 DataFrame
        df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ])
        
        # 補齊缺失的特徵（重要性為0）
        all_features = set(self.feature_names)
        existing_features = set(df['feature'].values)
        missing_features = all_features - existing_features
        
        if missing_features:
            missing_df = pd.DataFrame([
                {'feature': f, 'importance': 0.0}
                for f in missing_features
            ])
            df = pd.concat([df, missing_df], ignore_index=True)
        
        # 標準化並排序
        df['importance_normalized'] = df['importance'] / df['importance'].sum()
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        return df
    
    def save_model(self, filepath: str) -> None:
        """
        儲存模型
        
        Parameters
        ----------
        filepath : str
            儲存路徑
        """
        save_dict = {
            'model': self.model,
            'feature_names': self.feature_names,
            'training_info': self.training_info,
            'params': self.params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        if self.verbose:
            print(f"\n模型已儲存至: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'MoralChoiceXGBoostClassifier':
        """
        載入模型
        
        Parameters
        ----------
        filepath : str
            模型檔案路徑
            
        Returns
        -------
        MoralChoiceXGBoostClassifier
            載入的模型實例
        """
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        instance = cls(**save_dict['params'])
        instance.model = save_dict['model']
        instance.feature_names = save_dict['feature_names']
        instance.training_info = save_dict['training_info']
        
        return instance


def create_performance_summary(metrics: Dict[str, Any]) -> pd.DataFrame:
    """
    創建性能摘要表格
    
    Parameters
    ----------
    metrics : Dict[str, Any]
        評估指標字典
        
    Returns
    -------
    pd.DataFrame
        性能摘要表格
    """
    summary = pd.DataFrame([
        {'指標': 'Accuracy', '數值': metrics['accuracy'], '說明': '整體預測準確率'},
        {'指標': 'Precision', '數值': metrics['precision'], '說明': '預測為守法者中的真正守法比例'},
        {'指標': 'Recall', '數值': metrics['recall'], '說明': '真正守法者中被正確預測的比例'},
        {'指標': 'F1 Score', '數值': metrics['f1'], '說明': 'Precision與Recall的調和平均'},
        {'指標': 'ROC-AUC', '數值': metrics['roc_auc'], '說明': '分類能力的綜合指標'},
    ])
    
    summary['數值'] = summary['數值'].apply(lambda x: f"{x:.4f}")
    
    return summary


if __name__ == "__main__":
    print("XGBoost 分類器模組")
    print("請使用 MoralChoiceXGBoostClassifier 類別進行模型訓練與評估")