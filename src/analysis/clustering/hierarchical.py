"""
階層分群分析模組
================
負責第3章 3.3節的分析：道德距離的拓撲結構

功能：
1. 使用9個道德維度進行階層分群
2. 生成樹狀圖（Dendrogram）
3. 計算Cophenetic correlation
4. 與原始分類比較（ARI）
5. 道德距離矩陣熱圖
"""

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import plotly.graph_objects as go
import plotly.figure_factory as ff
from pathlib import Path
from typing import Dict, Tuple
import logging


class HierarchicalClusterAnalyzer:
    """階層分群分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """設定日誌記錄器"""
        logger = logging.getLogger('HierarchicalClusterAnalyzer')
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
    
    def prepare_clustering_data(self, 
                               countries_filepath: str = 'data/raw/CountriesChangePr.csv',
                               cluster_map_filepath: str = 'data/raw/country_cluster_map.csv') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        準備分群資料
        
        Parameters:
        -----------
        countries_filepath : str
            CountriesChangePr.csv 路徑
        cluster_map_filepath : str
            country_cluster_map.csv 路徑
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            (特徵矩陣, 原始分類)
        """
        self.logger.info("準備分群資料...")
        
        # 載入資料
        countries_df = pd.read_csv(countries_filepath, index_col=0)
        cluster_map = pd.read_csv(cluster_map_filepath)
        
        # 選擇9個道德維度的AMCE值
        amce_cols = [col for col in countries_df.columns if 'Estimates' in col and 'se' not in col.lower()]
        features_df = countries_df[amce_cols].copy()
        
        # 標準化（Z-score）
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df)
        features_scaled_df = pd.DataFrame(
            features_scaled,
            index=features_df.index,
            columns=features_df.columns
        )
        
        # 合併原始分類
        cluster_map_indexed = cluster_map.set_index('ISO3')
        
        self.logger.info(f"準備完成：{len(features_scaled_df)} 個國家，{len(amce_cols)} 個維度")
        
        return features_scaled_df, cluster_map_indexed
    
    def perform_hierarchical_clustering(self, features_df: pd.DataFrame) -> Tuple:
        """
        執行階層分群
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            標準化後的特徵矩陣
            
        Returns:
        --------
        Tuple
            (linkage矩陣, 距離矩陣, cophenetic correlation)
        """
        self.logger.info("執行階層分群...")
        
        # 計算距離矩陣（歐式距離）
        distance_matrix = pdist(features_df.values, metric='euclidean')
        distance_squareform = squareform(distance_matrix)
        
        # 執行階層分群（Ward's method）
        linkage_matrix = linkage(distance_matrix, method='ward')
        
        # 計算Cophenetic correlation
        coph_corr, coph_dists = cophenet(linkage_matrix, distance_matrix)
        
        self.logger.info(f"Cophenetic correlation: {coph_corr:.4f}")
        
        if coph_corr > 0.75:
            self.logger.info("✅ Cophenetic correlation > 0.75 (良好)")
        else:
            self.logger.warning("⚠️  Cophenetic correlation < 0.75")
        
        return linkage_matrix, distance_squareform, coph_corr
    
    def create_dendrogram(self,
                         linkage_matrix: np.ndarray,
                         labels: list,
                         output_path: str = 'outputs/figures/chapter3_exploration/dendrogram.html') -> str:
        """
        建立樹狀圖
        
        Parameters:
        -----------
        linkage_matrix : np.ndarray
            Linkage矩陣
        labels : list
            國家標籤
        output_path : str
            輸出檔案路徑
            
        Returns:
        --------
        str
            輸出檔案路徑
        """
        self.logger.info("建立樹狀圖...")
        
        # 使用plotly的dendrogram
        fig = ff.create_dendrogram(
            linkage_matrix,
            labels=labels,
            orientation='left',
            linkagefun=lambda x: linkage_matrix
        )
        
        # 更新布局
        fig.update_layout(
            title='130國階層分群樹狀圖（Ward\'s Method）',
            font=dict(family="Arial, sans-serif", size=10),
            title_font_size=20,
            title_x=0.5,
            height=2000,  # 因為有130個國家，需要較高的圖表
            width=1200,
            xaxis_title='距離',
            yaxis_title='',
            showlegend=False
        )
        
        # 儲存
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        self.logger.info(f"樹狀圖已儲存: {output_path}")
        
        return output_path
    
    def create_distance_heatmap(self,
                               distance_matrix: np.ndarray,
                               labels: list,
                               output_path: str = 'outputs/figures/chapter3_exploration/moral_distance_heatmap.html') -> str:
        """
        建立道德距離矩陣熱圖
        
        Parameters:
        -----------
        distance_matrix : np.ndarray
            距離矩陣
        labels : list
            國家標籤
        output_path : str
            輸出檔案路徑
            
        Returns:
        --------
        str
            輸出檔案路徑
        """
        self.logger.info("建立距離矩陣熱圖...")
        
        fig = go.Figure(data=go.Heatmap(
            z=distance_matrix,
            x=labels,
            y=labels,
            colorscale='Blues',
            colorbar=dict(title='歐式距離')
        ))
        
        fig.update_layout(
            title='130國道德距離矩陣',
            font=dict(family="Arial, sans-serif", size=8),
            title_font_size=20,
            title_x=0.5,
            height=1000,
            width=1000,
            xaxis_title='',
            yaxis_title=''
        )
        
        # 儲存
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        self.logger.info(f"距離熱圖已儲存: {output_path}")
        
        return output_path
    
    def compare_with_original_clusters(self,
                                      linkage_matrix: np.ndarray,
                                      original_clusters: pd.Series,
                                      n_clusters: int = 3) -> Tuple[float, pd.DataFrame]:
        """
        與原始分類比較
        
        Parameters:
        -----------
        linkage_matrix : np.ndarray
            Linkage矩陣
        original_clusters : pd.Series
            原始分類
        n_clusters : int
            切割的群數
            
        Returns:
        --------
        Tuple[float, pd.DataFrame]
            (ARI分數, 比較表)
        """
        self.logger.info(f"與原始分類比較（切割為 {n_clusters} 群）...")
        
        # 從樹狀圖切割出群組
        new_clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # 對齊索引
        new_clusters_series = pd.Series(new_clusters, index=original_clusters.index)
        
        # 計算ARI
        ari = adjusted_rand_score(original_clusters.values, new_clusters_series.values)
        
        self.logger.info(f"Adjusted Rand Index (ARI): {ari:.4f}")
        
        if ari > 0.60:
            self.logger.info("✅ ARI > 0.60 (高度一致)")
        else:
            self.logger.warning("⚠️  ARI < 0.60 (一致性較低)")
        
        # 建立比較表
        comparison_df = pd.DataFrame({
            '原始分類': original_clusters,
            '新分類': new_clusters_series
        })
        
        # 交叉表
        crosstab = pd.crosstab(
            comparison_df['原始分類'],
            comparison_df['新分類'],
            margins=True
        )
        
        return ari, crosstab
    
    def find_taiwan_neighbors(self,
                             distance_matrix: np.ndarray,
                             labels: list,
                             n_neighbors: int = 10) -> pd.DataFrame:
        """
        找出與台灣最接近的國家
        
        Parameters:
        -----------
        distance_matrix : np.ndarray
            距離矩陣
        labels : list
            國家標籤
        n_neighbors : int
            要找的鄰居數量
            
        Returns:
        --------
        pd.DataFrame
            最近鄰居表
        """
        self.logger.info("尋找台灣的最近鄰居...")
        
        if 'TWN' not in labels:
            self.logger.warning("資料中找不到台灣（TWN）")
            return pd.DataFrame()
        
        # 找到台灣的索引
        twn_idx = labels.index('TWN')
        
        # 取得台灣與其他國家的距離
        distances_from_twn = distance_matrix[twn_idx, :]
        
        # 排序（排除台灣自己）
        sorted_indices = np.argsort(distances_from_twn)[1:n_neighbors+1]
        
        neighbors_df = pd.DataFrame({
            '排名': range(1, n_neighbors+1),
            '國家': [labels[i] for i in sorted_indices],
            '距離': distances_from_twn[sorted_indices]
        })
        
        self.logger.info(f"台灣最接近的 {n_neighbors} 個國家:")
        for _, row in neighbors_df.iterrows():
            self.logger.info(f"  {row['排名']}. {row['國家']}: {row['距離']:.3f}")
        
        return neighbors_df
    
    def run_analysis(self) -> Dict[str, any]:
        """
        執行完整的階層分群分析
        
        Returns:
        --------
        Dict[str, any]
            分析結果字典
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("開始階層分群分析...")
        self.logger.info("=" * 60)
        
        # 1. 準備資料
        features_df, cluster_map = self.prepare_clustering_data()
        
        # 2. 執行分群
        linkage_matrix, distance_matrix, coph_corr = self.perform_hierarchical_clustering(features_df)
        
        # 3. 建立樹狀圖
        labels = features_df.index.tolist()
        dendrogram_path = self.create_dendrogram(linkage_matrix, labels)
        
        # 4. 建立距離熱圖
        heatmap_path = self.create_distance_heatmap(distance_matrix, labels)
        
        # 5. 與原始分類比較
        original_clusters = cluster_map.loc[features_df.index, 'Cluster']
        ari, crosstab = self.compare_with_original_clusters(linkage_matrix, original_clusters)
        
        # 6. 找出台灣鄰居
        taiwan_neighbors = self.find_taiwan_neighbors(distance_matrix, labels)
        
        # 7. 儲存結果
        output_dir = Path('outputs/tables/chapter3')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        crosstab.to_csv(output_dir / 'cluster_crosstab.csv', encoding='utf-8-sig')
        taiwan_neighbors.to_csv(output_dir / 'taiwan_neighbors.csv', index=False, encoding='utf-8-sig')
        
        # 儲存評估指標
        metrics_df = pd.DataFrame({
            '指標': ['Cophenetic Correlation', 'Adjusted Rand Index'],
            '數值': [coph_corr, ari],
            '評估': [
                '良好' if coph_corr > 0.75 else '普通',
                '高度一致' if ari > 0.60 else '一致性較低'
            ]
        })
        metrics_df.to_csv(output_dir / 'clustering_metrics.csv', index=False, encoding='utf-8-sig')
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("階層分群分析完成！")
        self.logger.info("=" * 60)
        
        return {
            'dendrogram': dendrogram_path,
            'distance_heatmap': heatmap_path,
            'cophenetic_correlation': coph_corr,
            'ari': ari,
            'taiwan_neighbors': taiwan_neighbors
        }


if __name__ == '__main__':
    print("階層分群分析模組載入成功")