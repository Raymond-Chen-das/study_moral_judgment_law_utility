"""
全球道德光譜模組
=================
用於第5.3節：效益主義 vs. 守法偏好的全球視覺化
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class MoralSpectrumAnalyzer:
    """
    全球道德光譜分析器
    
    以效益主義（X軸）vs. 守法偏好（Y軸）建構二維道德空間，
    定位130個國家並標註文化圈
    """
    
    # 文化圈名稱對應
    CLUSTER_NAMES = {
        0: 'Western',
        1: 'Eastern',
        2: 'Southern'
    }
    
    # 重點標註國家
    HIGHLIGHT_COUNTRIES = {
        'TWN': '台灣',
        'JPN': '日本',
        'KOR': '韓國',
        'CHN': '中國',
        'USA': '美國',
        'DEU': '德國',
        'GBR': '英國',
        'FRA': '法國',
        'BRA': '巴西',
        'IND': '印度'
    }
    
    # AMCE 欄位對應
    AMCE_COLUMNS = {
        'utilitarian': 'No. Characters [Less -> More]: Estimates',
        'law_preference': 'Law [Illegal -> Legal]: Estimates'
    }
    
    def __init__(self, verbose: bool = True):
        """
        初始化分析器
        
        Parameters
        ----------
        verbose : bool
            是否顯示詳細資訊
        """
        self.verbose = verbose
        self.spectrum_data = None
        self.quadrant_stats = None
        
    def load_data(
        self,
        amce_path: str,
        cluster_path: str
    ) -> pd.DataFrame:
        """
        載入 AMCE 與文化圈資料
        
        Parameters
        ----------
        amce_path : str
            CountriesChangePr.csv 路徑
        cluster_path : str
            country_cluster_map.csv 路徑
            
        Returns
        -------
        pd.DataFrame
            合併後的光譜資料
        """
        if self.verbose:
            print("=" * 60)
            print("5.3 全球道德光譜")
            print("=" * 60)
        
        # 載入 AMCE
        amce_raw = pd.read_csv(amce_path, index_col=0)
        if self.verbose:
            print(f"\n載入 AMCE: {len(amce_raw)} 個國家")
        
        # 載入文化圈
        cluster_data = pd.read_csv(cluster_path)
        if self.verbose:
            print(f"載入文化圈分類: {len(cluster_data)} 個國家")
        
        # 提取關鍵維度
        spectrum = pd.DataFrame({
            'UserCountry3': amce_raw.index,
            'utilitarian': amce_raw[self.AMCE_COLUMNS['utilitarian']].values,
            'law_preference': amce_raw[self.AMCE_COLUMNS['law_preference']].values
        })
        
        # 合併文化圈
        self.spectrum_data = spectrum.merge(
            cluster_data[['ISO3', 'Country', 'Cluster']],
            left_on='UserCountry3',
            right_on='ISO3',
            how='left'
        )
        
        # 添加文化圈名稱
        self.spectrum_data['Cluster_Name'] = self.spectrum_data['Cluster'].map(self.CLUSTER_NAMES)
        
        # 標記重點國家
        self.spectrum_data['is_highlight'] = self.spectrum_data['UserCountry3'].isin(
            self.HIGHLIGHT_COUNTRIES.keys()
        )
        self.spectrum_data['label'] = self.spectrum_data['UserCountry3'].map(
            self.HIGHLIGHT_COUNTRIES
        )
        
        if self.verbose:
            print(f"\n光譜資料: {len(self.spectrum_data)} 個國家")
            print(f"  - 效益主義 (X軸) 範圍: [{self.spectrum_data['utilitarian'].min():.3f}, "
                  f"{self.spectrum_data['utilitarian'].max():.3f}]")
            print(f"  - 守法偏好 (Y軸) 範圍: [{self.spectrum_data['law_preference'].min():.3f}, "
                  f"{self.spectrum_data['law_preference'].max():.3f}]")
        
        return self.spectrum_data
    
    def compute_quadrants(self) -> pd.DataFrame:
        """
        計算象限分佈
        
        以全球平均為原點，將國家分為四個象限
        
        Returns
        -------
        pd.DataFrame
            象限統計
        """
        if self.spectrum_data is None:
            raise ValueError("請先呼叫 load_data()")
        
        # 計算全球平均
        util_mean = self.spectrum_data['utilitarian'].mean()
        law_mean = self.spectrum_data['law_preference'].mean()
        
        # 分類象限
        def get_quadrant(row):
            if row['utilitarian'] >= util_mean and row['law_preference'] >= law_mean:
                return 'Q1: 高效益+高守法'
            elif row['utilitarian'] < util_mean and row['law_preference'] >= law_mean:
                return 'Q2: 低效益+高守法'
            elif row['utilitarian'] < util_mean and row['law_preference'] < law_mean:
                return 'Q3: 低效益+低守法'
            else:
                return 'Q4: 高效益+低守法'
        
        self.spectrum_data['quadrant'] = self.spectrum_data.apply(get_quadrant, axis=1)
        
        # 統計各象限
        quadrant_stats = []
        for quadrant in ['Q1: 高效益+高守法', 'Q2: 低效益+高守法', 
                         'Q3: 低效益+低守法', 'Q4: 高效益+低守法']:
            q_data = self.spectrum_data[self.spectrum_data['quadrant'] == quadrant]
            
            for cluster in [0, 1, 2]:
                cluster_name = self.CLUSTER_NAMES[cluster]
                count = len(q_data[q_data['Cluster'] == cluster])
                quadrant_stats.append({
                    'quadrant': quadrant,
                    'cluster': cluster_name,
                    'count': count
                })
        
        self.quadrant_stats = pd.DataFrame(quadrant_stats)
        
        if self.verbose:
            print("\n象限分佈統計：")
            print("-" * 60)
            pivot = self.quadrant_stats.pivot_table(
                index='quadrant', columns='cluster', values='count', fill_value=0
            )
            print(pivot)
        
        return self.quadrant_stats
    
    def get_country_positions(self) -> pd.DataFrame:
        """
        獲取國家座標資料（用於視覺化）
        
        Returns
        -------
        pd.DataFrame
            國家座標資料
        """
        if self.spectrum_data is None:
            raise ValueError("請先呼叫 load_data()")
        
        return self.spectrum_data[[
            'UserCountry3', 'Country', 'utilitarian', 'law_preference',
            'Cluster', 'Cluster_Name', 'is_highlight', 'label', 'quadrant'
        ]].copy()
    
    def get_taiwan_analysis(self) -> Dict[str, Any]:
        """
        台灣定位分析
        
        Returns
        -------
        Dict[str, Any]
            台灣分析結果
        """
        if self.spectrum_data is None:
            raise ValueError("請先呼叫 load_data()")
        
        taiwan = self.spectrum_data[self.spectrum_data['UserCountry3'] == 'TWN'].iloc[0]
        
        # 計算排名
        util_rank = (self.spectrum_data['utilitarian'] > taiwan['utilitarian']).sum() + 1
        law_rank = (self.spectrum_data['law_preference'] > taiwan['law_preference']).sum() + 1
        
        # 找最近鄰居
        self.spectrum_data['distance_to_taiwan'] = np.sqrt(
            (self.spectrum_data['utilitarian'] - taiwan['utilitarian'])**2 +
            (self.spectrum_data['law_preference'] - taiwan['law_preference'])**2
        )
        nearest = self.spectrum_data[
            self.spectrum_data['UserCountry3'] != 'TWN'
        ].nsmallest(5, 'distance_to_taiwan')
        
        analysis = {
            'utilitarian': taiwan['utilitarian'],
            'law_preference': taiwan['law_preference'],
            'quadrant': taiwan['quadrant'],
            'util_rank': util_rank,
            'law_rank': law_rank,
            'total_countries': len(self.spectrum_data),
            'nearest_neighbors': nearest[['UserCountry3', 'Country', 'distance_to_taiwan']].to_dict('records')
        }
        
        if self.verbose:
            print("\n台灣定位分析：")
            print("-" * 60)
            print(f"  效益主義: {analysis['utilitarian']:.3f} (排名: {analysis['util_rank']}/{analysis['total_countries']})")
            print(f"  守法偏好: {analysis['law_preference']:.3f} (排名: {analysis['law_rank']}/{analysis['total_countries']})")
            print(f"  象限: {analysis['quadrant']}")
            print(f"\n  最近鄰居（道德距離）:")
            for nb in analysis['nearest_neighbors']:
                print(f"    - {nb['Country']} ({nb['UserCountry3']}): {nb['distance_to_taiwan']:.4f}")
        
        return analysis
    
    def get_cluster_centroids(self) -> pd.DataFrame:
        """
        計算各文化圈的重心
        
        Returns
        -------
        pd.DataFrame
            文化圈重心座標
        """
        if self.spectrum_data is None:
            raise ValueError("請先呼叫 load_data()")
        
        centroids = self.spectrum_data.groupby('Cluster_Name').agg({
            'utilitarian': 'mean',
            'law_preference': 'mean',
            'UserCountry3': 'count'
        }).rename(columns={'UserCountry3': 'n_countries'}).reset_index()
        
        if self.verbose:
            print("\n文化圈重心：")
            print("-" * 60)
            for _, row in centroids.iterrows():
                print(f"  {row['Cluster_Name']} (n={row['n_countries']}):")
                print(f"    效益主義: {row['utilitarian']:.3f}")
                print(f"    守法偏好: {row['law_preference']:.3f}")
        
        return centroids
    
    def compute_cultural_distance(self) -> pd.DataFrame:
        """
        計算文化圈間的道德距離
        
        Returns
        -------
        pd.DataFrame
            文化圈距離矩陣
        """
        centroids = self.get_cluster_centroids()
        
        distances = []
        cluster_names = centroids['Cluster_Name'].tolist()
        
        for i, c1 in enumerate(cluster_names):
            row = {'cluster': c1}
            c1_data = centroids[centroids['Cluster_Name'] == c1].iloc[0]
            
            for c2 in cluster_names:
                c2_data = centroids[centroids['Cluster_Name'] == c2].iloc[0]
                dist = np.sqrt(
                    (c1_data['utilitarian'] - c2_data['utilitarian'])**2 +
                    (c1_data['law_preference'] - c2_data['law_preference'])**2
                )
                row[c2] = dist
            
            distances.append(row)
        
        return pd.DataFrame(distances).set_index('cluster')
    
    def generate_interpretation(self) -> str:
        """
        生成分析解釋文字
        
        Returns
        -------
        str
            解釋文字
        """
        if self.spectrum_data is None:
            raise ValueError("請先呼叫 load_data()")
        
        taiwan_analysis = self.get_taiwan_analysis()
        centroids = self.get_cluster_centroids()
        
        interpretation = []
        interpretation.append("### 5.3 全球道德光譜\n")
        interpretation.append("**視覺化框架**：以效益主義（X軸）vs. 守法偏好（Y軸）建構二維道德空間\n")
        
        interpretation.append("**文化圈重心**：")
        for _, row in centroids.iterrows():
            interpretation.append(f"- {row['Cluster_Name']}: 效益主義={row['utilitarian']:.3f}, 守法={row['law_preference']:.3f}")
        
        interpretation.append(f"\n**台灣定位**：")
        interpretation.append(f"- 效益主義排名: {taiwan_analysis['util_rank']}/{taiwan_analysis['total_countries']}")
        interpretation.append(f"- 守法偏好排名: {taiwan_analysis['law_rank']}/{taiwan_analysis['total_countries']}")
        interpretation.append(f"- 象限: {taiwan_analysis['quadrant']}")
        
        return "\n".join(interpretation)


def load_and_analyze(
    amce_path: str,
    cluster_path: str,
    verbose: bool = True
) -> Tuple[MoralSpectrumAnalyzer, pd.DataFrame]:
    """
    一鍵載入與分析
    
    Parameters
    ----------
    amce_path : str
        CountriesChangePr.csv 路徑
    cluster_path : str
        country_cluster_map.csv 路徑
    verbose : bool
        是否顯示詳細資訊
        
    Returns
    -------
    Tuple
        (分析器實例, 光譜資料)
    """
    analyzer = MoralSpectrumAnalyzer(verbose=verbose)
    spectrum_data = analyzer.load_data(amce_path, cluster_path)
    analyzer.compute_quadrants()
    analyzer.get_taiwan_analysis()
    analyzer.get_cluster_centroids()
    
    return analyzer, spectrum_data


if __name__ == "__main__":
    print("全球道德光譜模組")
    print("請使用 load_and_analyze() 函數進行分析")