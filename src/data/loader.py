"""
資料載入模組 (Data Loader Module)
負責載入所有原始資料檔案並進行基本驗證

主要功能：
1. 載入MIT Moral Machine各類資料集
2. 基本資料驗證與完整性檢查
3. 產生資料載入報告
4. 統一的錯誤處理機制
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging
from datetime import datetime

class DataLoader:
    """MIT Moral Machine資料載入器"""
    
    def __init__(self, data_dir: str = 'data/raw'):
        """
        初始化資料載入器
        
        Parameters:
        -----------
        data_dir : str
            原始資料目錄路徑
        """
        self.data_dir = Path(data_dir)
        self.logger = self._setup_logger()
        
        # 預期的資料檔案
        self.expected_files = {
            'survey': 'SharedResponsesSurvey.csv',
            'countries_change': 'CountriesChangePr.csv',
            'cluster_map': 'country_cluster_map.csv',
            'moral_distance': 'moral_distance.csv',
            'dendrogram': 'dendrogram_Culture.csv'
        }
        
        # 儲存載入的資料
        self.data = {}
        
    def _setup_logger(self) -> logging.Logger:
        """設定日誌記錄器"""
        logger = logging.getLogger('DataLoader')
        logger.setLevel(logging.INFO)
        
        # 避免重複添加handler
        if not logger.handlers:
            # 終端機輸出
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 格式設定
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def check_files_exist(self) -> Dict[str, bool]:
        """
        檢查所有預期檔案是否存在
        
        Returns:
        --------
        Dict[str, bool]
            檔案存在狀態字典
        """
        self.logger.info("=" * 60)
        self.logger.info("檢查資料檔案...")
        self.logger.info("=" * 60)
        
        files_status = {}
        
        for key, filename in self.expected_files.items():
            filepath = self.data_dir / filename
            exists = filepath.exists()
            files_status[key] = exists
            
            status = "✅ 存在" if exists else "❌ 缺失"
            self.logger.info(f"{filename:40s} {status}")
            
            if not exists:
                self.logger.warning(f"找不到檔案: {filepath}")
        
        return files_status
    
    def load_survey_data(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        載入主要問卷資料 (SharedResponsesSurvey.csv)
        
        Parameters:
        -----------
        nrows : int, optional
            載入的行數限制（用於測試），None表示載入全部
            
        Returns:
        --------
        pd.DataFrame
            問卷資料框
        """
        filename = self.expected_files['survey']
        filepath = self.data_dir / filename
        
        self.logger.info(f"\n載入中: {filename}")
        
        try:
            # 使用較優化的參數載入
            df = pd.read_csv(
                filepath,
                nrows=nrows,
                low_memory=False  # 避免dtype警告
            )
            
            self.logger.info(f"✅ 成功載入 {len(df):,} 行 × {len(df.columns)} 欄")
            
            # 基本驗證
            self._validate_survey_data(df)
            
            return df
            
        except FileNotFoundError:
            self.logger.error(f"❌ 找不到檔案: {filepath}")
            raise
        except Exception as e:
            self.logger.error(f"❌ 載入失敗: {e}")
            raise
    
    def load_countries_change_pr(self) -> pd.DataFrame:
        """
        載入國家層級AMCE資料 (CountriesChangePr.csv)
        
        Returns:
        --------
        pd.DataFrame
            國家AMCE值資料框（130國×18欄）
        """
        filename = self.expected_files['countries_change']
        filepath = self.data_dir / filename
        
        self.logger.info(f"\n載入中: {filename}")
        
        try:
            df = pd.read_csv(filepath, index_col=0)  # 第一欄是國家代碼
            
            self.logger.info(f"✅ 成功載入 {len(df)} 個國家 × {len(df.columns)} 個指標")
            
            # 基本驗證
            self._validate_countries_data(df)
            
            return df
            
        except FileNotFoundError:
            self.logger.error(f"❌ 找不到檔案: {filepath}")
            raise
        except Exception as e:
            self.logger.error(f"❌ 載入失敗: {e}")
            raise
    
    def load_cluster_map(self) -> pd.DataFrame:
        """
        載入文化圈分類資料 (country_cluster_map.csv)
        
        Returns:
        --------
        pd.DataFrame
            文化圈分類資料框（130國×3欄）
        """
        filename = self.expected_files['cluster_map']
        filepath = self.data_dir / filename
        
        self.logger.info(f"\n載入中: {filename}")
        
        try:
            df = pd.read_csv(filepath)
            
            self.logger.info(f"✅ 成功載入 {len(df)} 個國家的文化圈分類")
            
            # 檢查Cluster的分佈
            cluster_counts = df['Cluster'].value_counts().sort_index()
            for cluster, count in cluster_counts.items():
                self.logger.info(f"   Cluster {cluster}: {count} 個國家")
            
            return df
            
        except FileNotFoundError:
            self.logger.error(f"❌ 找不到檔案: {filepath}")
            raise
        except Exception as e:
            self.logger.error(f"❌ 載入失敗: {e}")
            raise
    
    def load_moral_distance(self) -> pd.DataFrame:
        """
        載入道德距離矩陣 (moral_distance.csv)
        
        Returns:
        --------
        pd.DataFrame
            道德距離資料框（130國×2欄）
        """
        filename = self.expected_files['moral_distance']
        filepath = self.data_dir / filename
        
        self.logger.info(f"\n載入中: {filename}")
        
        try:
            df = pd.read_csv(filepath)
            
            self.logger.info(f"✅ 成功載入 {len(df)} 個國家的道德距離")
            self.logger.info(f"   距離範圍: {df['Distance'].min():.3f} ~ {df['Distance'].max():.3f}")
            self.logger.info(f"   平均距離: {df['Distance'].mean():.3f}")
            
            return df
            
        except FileNotFoundError:
            self.logger.error(f"❌ 找不到檔案: {filepath}")
            raise
        except Exception as e:
            self.logger.error(f"❌ 載入失敗: {e}")
            raise
    
    def load_dendrogram(self) -> pd.DataFrame:
        """
        載入樹狀圖資料 (dendrogram_Culture.csv)
        
        Returns:
        --------
        pd.DataFrame
            樹狀圖資料框
        """
        filename = self.expected_files['dendrogram']
        filepath = self.data_dir / filename
        
        self.logger.info(f"\n載入中: {filename}")
        
        try:
            df = pd.read_csv(filepath)
            
            self.logger.info(f"✅ 成功載入 {len(df)} 個節點")
            
            # 計算有多少個實際國家（非空culture欄位）
            countries = df[df['culture'].notna()]
            self.logger.info(f"   包含 {len(countries)} 個國家節點")
            
            return df
            
        except FileNotFoundError:
            self.logger.error(f"❌ 找不到檔案: {filepath}")
            raise
        except Exception as e:
            self.logger.error(f"❌ 載入失敗: {e}")
            raise
    
    def load_all_data(self, survey_nrows: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        載入所有資料檔案
        
        Parameters:
        -----------
        survey_nrows : int, optional
            問卷資料載入行數限制
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            包含所有資料集的字典
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("開始載入所有資料...")
        self.logger.info("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # 載入各個資料集
            self.data['survey'] = self.load_survey_data(nrows=survey_nrows)
            self.data['countries_change'] = self.load_countries_change_pr()
            self.data['cluster_map'] = self.load_cluster_map()
            self.data['moral_distance'] = self.load_moral_distance()
            self.data['dendrogram'] = self.load_dendrogram()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.info("\n" + "=" * 60)
            self.logger.info(f"✅ 所有資料載入完成！耗時 {duration:.2f} 秒")
            self.logger.info("=" * 60)
            
            return self.data
            
        except Exception as e:
            self.logger.error(f"\n❌ 資料載入失敗: {e}")
            raise
    
    def _validate_survey_data(self, df: pd.DataFrame):
        """驗證問卷資料的完整性"""
        # 檢查關鍵欄位是否存在
        required_cols = [
            'ResponseID', 'UserID', 'UserCountry3', 'Saved',
            'ScenarioType', 'NumberOfCharacters'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            self.logger.warning(f"⚠️  缺少關鍵欄位: {missing_cols}")
        else:
            self.logger.info("   ✓ 所有關鍵欄位都存在")
        
        # 檢查Saved欄位的值範圍
        if 'Saved' in df.columns:
            unique_saved = df['Saved'].unique()
            if not set(unique_saved).issubset({0, 1, np.nan}):
                self.logger.warning(f"⚠️  Saved欄位包含異常值: {unique_saved}")
    
    def _validate_countries_data(self, df: pd.DataFrame):
        """驗證國家資料的完整性"""
        # 檢查是否有9對欄位（Estimates + se）
        estimate_cols = [col for col in df.columns if 'Estimates' in col]
        se_cols = [col for col in df.columns if ': se' in col]
        
        self.logger.info(f"   ✓ {len(estimate_cols)} 個AMCE估計值欄位")
        self.logger.info(f"   ✓ {len(se_cols)} 個標準誤欄位")
        
        if len(estimate_cols) != len(se_cols):
            self.logger.warning("⚠️  Estimates和se欄位數量不匹配")
    
    def generate_loading_summary(self) -> Dict[str, any]:
        """
        生成資料載入摘要
        
        Returns:
        --------
        Dict
            載入摘要字典
        """
        if not self.data:
            self.logger.warning("尚未載入任何資料")
            return {}
        
        summary = {
            'loading_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'datasets': {}
        }
        
        for name, df in self.data.items():
            # 轉換為 Python 原生類型以支援 JSON 序列化
            missing_count = int(df.isnull().sum().sum())
            total_cells = len(df) * len(df.columns)
            missing_pct = (missing_count / total_cells * 100) if total_cells > 0 else 0
            
            summary['datasets'][name] = {
                'rows': int(len(df)),
                'columns': int(len(df.columns)),
                'memory_mb': float(df.memory_usage(deep=True).sum() / 1024**2),
                'missing_values': missing_count,
                'missing_pct': f"{missing_pct:.2f}%"
            }
        
        return summary
    
    def print_summary(self):
        """列印資料載入摘要"""
        summary = self.generate_loading_summary()
        
        if not summary:
            return
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("資料載入摘要")
        self.logger.info("=" * 60)
        
        total_rows = 0
        total_memory = 0
        
        for name, info in summary['datasets'].items():
            self.logger.info(f"\n【{name}】")
            self.logger.info(f"  維度: {info['rows']:,} 行 × {info['columns']} 欄")
            self.logger.info(f"  記憶體: {info['memory_mb']:.2f} MB")
            self.logger.info(f"  缺失值: {info['missing_values']:,} ({info['missing_pct']})")
            
            total_rows += info['rows']
            total_memory += info['memory_mb']
        
        self.logger.info("\n" + "-" * 60)
        self.logger.info(f"總計: {total_rows:,} 行資料，佔用 {total_memory:.2f} MB記憶體")
        self.logger.info("=" * 60)


def load_raw_data(data_dir: str = 'data/raw', 
                  survey_nrows: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """
    便捷函數：載入所有原始資料
    
    Parameters:
    -----------
    data_dir : str
        原始資料目錄
    survey_nrows : int, optional
        問卷資料載入行數限制
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        包含所有資料集的字典
    """
    loader = DataLoader(data_dir=data_dir)
    
    # 檢查檔案
    files_status = loader.check_files_exist()
    
    if not all(files_status.values()):
        raise FileNotFoundError("部分資料檔案缺失，請檢查data/raw/目錄")
    
    # 載入資料
    data = loader.load_all_data(survey_nrows=survey_nrows)
    
    # 列印摘要
    loader.print_summary()
    
    return data


# 如果直接執行此模組，進行測試
if __name__ == '__main__':
    print("測試資料載入模組...")
    
    # 測試載入（只載入前1000行以節省時間）
    data = load_raw_data(survey_nrows=1000)
    
    print("\n✅ 資料載入測試完成！")