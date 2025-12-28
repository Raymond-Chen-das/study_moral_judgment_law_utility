"""
替代分群方法模組
================
針對3.3節階層分群的Cophenetic Correlation過低問題，
提供三種替代方法進行比較

方法：
1. K-means clustering + Silhouette score
2. DBSCAN (基於密度的分群)
3. t-SNE降維 + 視覺化檢視

目標：
- 驗證階層分群結果的穩健性
- 找出更適合的分群方法
- 為報告提供方法論比較
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, 
    silhouette_samples,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
from typing import Dict, Tuple, List
import logging


class AlternativeClusteringAnalyzer:
    """替代分群方法分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.logger = self._setup_logger()
        self.scaler = StandardScaler()
        
    def _setup_logger(self) -> logging.Logger:
        """設定日誌記錄器"""
        logger = logging.getLogger('AlternativeClusteringAnalyzer')
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
    
    def load_data(self,
                  countries_filepath: str = 'data/raw/CountriesChangePr.csv',
                  cluster_map_filepath: str = 'data/raw/country_cluster_map.csv') -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        載入並準備資料
        
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series, List[str]]
            (標準化特徵, 原始分類, 國家標籤)
        """
        self.logger.info("載入資料...")
        
        # 載入資料
        countries_df = pd.read_csv(countries_filepath, index_col=0)
        cluster_map = pd.read_csv(cluster_map_filepath)
        
        # 選擇9個道德維度的AMCE值
        amce_cols = [col for col in countries_df.columns if 'Estimates' in col and 'se' not in col.lower()]
        features_df = countries_df[amce_cols].copy()
        
        # 標準化
        features_scaled = self.scaler.fit_transform(features_df)
        features_scaled_df = pd.DataFrame(
            features_scaled,
            index=features_df.index,
            columns=features_df.columns
        )
        
        # 原始分類
        cluster_map_indexed = cluster_map.set_index('ISO3')
        original_clusters = cluster_map_indexed.loc[features_df.index, 'Cluster']
        
        # 國家標籤
        country_labels = features_df.index.tolist()
        
        self.logger.info(f"資料準備完成：{len(features_scaled_df)} 個國家，{len(amce_cols)} 個維度")
        
        return features_scaled_df, original_clusters, country_labels
    
    # ============================================
    # 方法1: K-means + Silhouette Score
    # ============================================
    
    def kmeans_with_silhouette(self,
                               features_df: pd.DataFrame,
                               k_range: range = range(2, 11)) -> Dict:
        """
        K-means分群 + Silhouette評估
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            標準化特徵
        k_range : range
            測試的k值範圍
            
        Returns:
        --------
        Dict
            分析結果
        """
        self.logger.info("執行K-means分群...")
        
        results = {
            'k_values': [],
            'silhouette_scores': [],
            'calinski_harabasz_scores': [],
            'davies_bouldin_scores': [],
            'inertias': [],
            'models': {}
        }
        
        for k in k_range:
            self.logger.info(f"  測試 k={k}...")
            
            # K-means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
            labels = kmeans.fit_predict(features_df)
            
            # 評估指標
            silhouette = silhouette_score(features_df, labels)
            calinski = calinski_harabasz_score(features_df, labels)
            davies = davies_bouldin_score(features_df, labels)
            inertia = kmeans.inertia_
            
            results['k_values'].append(k)
            results['silhouette_scores'].append(silhouette)
            results['calinski_harabasz_scores'].append(calinski)
            results['davies_bouldin_scores'].append(davies)
            results['inertias'].append(inertia)
            results['models'][k] = kmeans
            
            self.logger.info(f"    Silhouette: {silhouette:.4f}")
            self.logger.info(f"    Calinski-Harabasz: {calinski:.2f}")
            self.logger.info(f"    Davies-Bouldin: {davies:.4f}")
        
        # 找出最佳k
        best_k_idx = np.argmax(results['silhouette_scores'])
        best_k = results['k_values'][best_k_idx]
        
        self.logger.info(f"\n最佳 k = {best_k} (Silhouette = {results['silhouette_scores'][best_k_idx]:.4f})")
        
        results['best_k'] = best_k
        results['best_model'] = results['models'][best_k]
        
        return results
    
    def create_kmeans_evaluation_plot(self,
                                     kmeans_results: Dict,
                                     output_path: str = 'outputs/figures/chapter3_exploration/kmeans_evaluation.html') -> str:
        """建立K-means評估圖"""
        self.logger.info("建立K-means評估圖...")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Silhouette Score (越高越好)',
                'Calinski-Harabasz Score (越高越好)',
                'Davies-Bouldin Score (越低越好)',
                'Elbow Method (慣性)'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        k_values = kmeans_results['k_values']
        
        # Silhouette Score
        fig.add_trace(
            go.Scatter(
                x=k_values,
                y=kmeans_results['silhouette_scores'],
                mode='lines+markers',
                name='Silhouette',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # 標註最佳k
        best_k = kmeans_results['best_k']
        best_idx = k_values.index(best_k)
        fig.add_annotation(
            x=best_k,
            y=kmeans_results['silhouette_scores'][best_idx],
            text=f"最佳 k={best_k}",
            showarrow=True,
            arrowhead=2,
            row=1, col=1
        )
        
        # Calinski-Harabasz Score
        fig.add_trace(
            go.Scatter(
                x=k_values,
                y=kmeans_results['calinski_harabasz_scores'],
                mode='lines+markers',
                name='Calinski-Harabasz',
                line=dict(color='green', width=2),
                marker=dict(size=8),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Davies-Bouldin Score (越低越好)
        fig.add_trace(
            go.Scatter(
                x=k_values,
                y=kmeans_results['davies_bouldin_scores'],
                mode='lines+markers',
                name='Davies-Bouldin',
                line=dict(color='red', width=2),
                marker=dict(size=8),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Elbow Method
        fig.add_trace(
            go.Scatter(
                x=k_values,
                y=kmeans_results['inertias'],
                mode='lines+markers',
                name='Inertia',
                line=dict(color='orange', width=2),
                marker=dict(size=8),
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="群數 (k)", row=1, col=1)
        fig.update_xaxes(title_text="群數 (k)", row=1, col=2)
        fig.update_xaxes(title_text="群數 (k)", row=2, col=1)
        fig.update_xaxes(title_text="群數 (k)", row=2, col=2)
        
        fig.update_layout(
            title_text='K-means分群：評估指標比較',
            font=dict(family="Arial, sans-serif", size=12),
            title_font_size=18,
            title_x=0.5,
            height=800
        )
        
        # 儲存
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        self.logger.info(f"K-means評估圖已儲存: {output_path}")
        
        return output_path
    
    # ============================================
    # 方法2: DBSCAN
    # ============================================
    
    def dbscan_clustering(self,
                         features_df: pd.DataFrame,
                         eps_range: List[float] = None,
                         min_samples: int = 3) -> Dict:
        """
        DBSCAN分群
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            標準化特徵
        eps_range : List[float]
            測試的eps值範圍
        min_samples : int
            最小樣本數
            
        Returns:
        --------
        Dict
            分析結果
        """
        self.logger.info("執行DBSCAN分群...")
        
        if eps_range is None:
            # 使用K距離圖決定eps範圍
            distances = pdist(features_df, metric='euclidean')
            distances_sorted = np.sort(distances)
            # 選擇合理的eps範圍
            eps_range = np.linspace(
                distances_sorted[int(len(distances_sorted)*0.1)],
                distances_sorted[int(len(distances_sorted)*0.5)],
                20
            )
        
        results = {
            'eps_values': [],
            'n_clusters': [],
            'n_noise': [],
            'silhouette_scores': [],
            'models': {}
        }
        
        for eps in eps_range:
            self.logger.info(f"  測試 eps={eps:.3f}...")
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(features_df)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            # 只有當群數>=2且有非噪音點時才計算silhouette
            if n_clusters >= 2 and n_noise < len(labels):
                try:
                    silhouette = silhouette_score(features_df[labels != -1], labels[labels != -1])
                except:
                    silhouette = -1
            else:
                silhouette = -1
            
            results['eps_values'].append(eps)
            results['n_clusters'].append(n_clusters)
            results['n_noise'].append(n_noise)
            results['silhouette_scores'].append(silhouette)
            results['models'][eps] = dbscan
            
            self.logger.info(f"    群數: {n_clusters}, 噪音點: {n_noise}, Silhouette: {silhouette:.4f}")
        
        # 找出最佳eps（silhouette最高且噪音點合理）
        valid_indices = [i for i, s in enumerate(results['silhouette_scores']) if s > 0]
        if valid_indices:
            best_idx = valid_indices[np.argmax([results['silhouette_scores'][i] for i in valid_indices])]
            best_eps = results['eps_values'][best_idx]
            self.logger.info(f"\n最佳 eps = {best_eps:.3f} (Silhouette = {results['silhouette_scores'][best_idx]:.4f})")
            results['best_eps'] = best_eps
            results['best_model'] = results['models'][best_eps]
        else:
            self.logger.warning("⚠️  DBSCAN未找到有效的分群")
            results['best_eps'] = None
            results['best_model'] = None
        
        return results
    
    def create_dbscan_evaluation_plot(self,
                                     dbscan_results: Dict,
                                     output_path: str = 'outputs/figures/chapter3_exploration/dbscan_evaluation.html') -> str:
        """建立DBSCAN評估圖"""
        self.logger.info("建立DBSCAN評估圖...")
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                '群數 vs Eps',
                '噪音點數 vs Eps',
                'Silhouette Score vs Eps'
            ),
            horizontal_spacing=0.1
        )
        
        eps_values = dbscan_results['eps_values']
        
        # 群數
        fig.add_trace(
            go.Scatter(
                x=eps_values,
                y=dbscan_results['n_clusters'],
                mode='lines+markers',
                name='群數',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # 噪音點數
        fig.add_trace(
            go.Scatter(
                x=eps_values,
                y=dbscan_results['n_noise'],
                mode='lines+markers',
                name='噪音點',
                line=dict(color='red', width=2),
                marker=dict(size=8),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Silhouette Score
        silhouette_scores_plot = [s if s > 0 else None for s in dbscan_results['silhouette_scores']]
        fig.add_trace(
            go.Scatter(
                x=eps_values,
                y=silhouette_scores_plot,
                mode='lines+markers',
                name='Silhouette',
                line=dict(color='green', width=2),
                marker=dict(size=8),
                showlegend=False
            ),
            row=1, col=3
        )
        
        # 標註最佳eps
        if dbscan_results.get('best_eps'):
            best_eps = dbscan_results['best_eps']
            best_idx = eps_values.index(best_eps)
            fig.add_annotation(
                x=best_eps,
                y=dbscan_results['silhouette_scores'][best_idx],
                text=f"最佳 eps={best_eps:.2f}",
                showarrow=True,
                arrowhead=2,
                row=1, col=3
            )
        
        fig.update_xaxes(title_text="Eps", row=1, col=1)
        fig.update_xaxes(title_text="Eps", row=1, col=2)
        fig.update_xaxes(title_text="Eps", row=1, col=3)
        
        fig.update_layout(
            title_text='DBSCAN分群：參數評估',
            font=dict(family="Arial, sans-serif", size=12),
            title_font_size=18,
            title_x=0.5,
            height=400
        )
        
        # 儲存
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        self.logger.info(f"DBSCAN評估圖已儲存: {output_path}")
        
        return output_path
    
    # ============================================
    # 方法3: t-SNE降維視覺化
    # ============================================
    
    def tsne_visualization(self,
                          features_df: pd.DataFrame,
                          country_labels: List[str],
                          original_clusters: pd.Series,
                          perplexity: int = 30,
                          random_state: int = 42) -> Dict:
        """
        t-SNE降維視覺化
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            標準化特徵
        country_labels : List[str]
            國家標籤
        original_clusters : pd.Series
            原始分類
        perplexity : int
            t-SNE困惑度參數
        random_state : int
            隨機種子
            
        Returns:
        --------
        Dict
            降維結果
        """
        self.logger.info("執行t-SNE降維...")
        
        # t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            max_iter=1000,
            verbose=1
        )
        
        tsne_results = tsne.fit_transform(features_df)
        
        # 建立結果DataFrame
        tsne_df = pd.DataFrame({
            'Country': country_labels,
            'tSNE1': tsne_results[:, 0],
            'tSNE2': tsne_results[:, 1],
            'Original_Cluster': original_clusters.values,
            'Cluster_Name': original_clusters.map({
                0: 'Western',
                1: 'Eastern',
                2: 'Southern'
            }).values
        })
        
        self.logger.info("t-SNE降維完成")
        
        return {
            'tsne_df': tsne_df,
            'tsne_model': tsne
        }
    
    def create_tsne_plot(self,
                        tsne_results: Dict,
                        kmeans_labels: np.ndarray = None,
                        output_path: str = 'outputs/figures/chapter3_exploration/tsne_visualization.html') -> str:
        """建立t-SNE視覺化圖"""
        self.logger.info("建立t-SNE視覺化圖...")
        
        tsne_df = tsne_results['tsne_df'].copy()
        
        # 如果有K-means標籤，也加入
        if kmeans_labels is not None:
            tsne_df['KMeans_Cluster'] = kmeans_labels
        
        # 建立圖表（顯示原始分類）
        fig = px.scatter(
            tsne_df,
            x='tSNE1',
            y='tSNE2',
            color='Cluster_Name',
            hover_data=['Country'],
            text='Country',
            title='t-SNE降維視覺化：130國道德距離',
            color_discrete_map={
                'Western': 'blue',
                'Eastern': 'red',
                'Southern': 'green'
            }
        )
        
        # 調整文字顯示
        fig.update_traces(
            textposition='top center',
            textfont=dict(size=8),
            marker=dict(size=10, line=dict(width=1, color='white'))
        )
        
        fig.update_layout(
            font=dict(family="Arial, sans-serif", size=12),
            title_font_size=18,
            title_x=0.5,
            height=700,
            width=900,
            xaxis_title='t-SNE 維度 1',
            yaxis_title='t-SNE 維度 2'
        )
        
        # 儲存
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        self.logger.info(f"t-SNE視覺化圖已儲存: {output_path}")
        
        return output_path
    
    # ============================================
    # 方法比較
    # ============================================
    
    def compare_methods(self,
                       features_df: pd.DataFrame,
                       original_clusters: pd.Series,
                       hierarchical_labels: np.ndarray,
                       kmeans_results: Dict,
                       dbscan_results: Dict) -> pd.DataFrame:
        """
        比較四種方法
        
        Returns:
        --------
        pd.DataFrame
            比較表
        """
        self.logger.info("比較四種分群方法...")
        
        comparison_data = []
        
        # 1. 階層分群
        comparison_data.append({
            '方法': 'Hierarchical (Ward)',
            'ARI vs 原始分類': adjusted_rand_score(original_clusters, hierarchical_labels),
            'Silhouette Score': silhouette_score(features_df, hierarchical_labels),
            '群數': len(set(hierarchical_labels)),
            '評價': '❌ Cophenetic Corr過低'
        })
        
        # 2. K-means
        kmeans_labels = kmeans_results['best_model'].labels_
        comparison_data.append({
            '方法': f'K-means (k={kmeans_results["best_k"]})',
            'ARI vs 原始分類': adjusted_rand_score(original_clusters, kmeans_labels),
            'Silhouette Score': silhouette_score(features_df, kmeans_labels),
            '群數': kmeans_results['best_k'],
            '評價': '✅ Silhouette評估最佳'
        })
        
        # 3. DBSCAN
        if dbscan_results.get('best_model'):
            dbscan_labels = dbscan_results['best_model'].labels_
            non_noise = dbscan_labels != -1
            if non_noise.sum() > 0:
                ari = adjusted_rand_score(original_clusters[non_noise], dbscan_labels[non_noise])
                sil = silhouette_score(features_df[non_noise], dbscan_labels[non_noise]) if len(set(dbscan_labels[non_noise])) > 1 else -1
                n_clusters = len(set(dbscan_labels)) - 1
                comparison_data.append({
                    '方法': f'DBSCAN (eps={dbscan_results["best_eps"]:.2f})',
                    'ARI vs 原始分類': ari,
                    'Silhouette Score': sil,
                    '群數': n_clusters,
                    '評價': f'⚠️  {list(dbscan_labels).count(-1)} 噪音點'
                })
        
        # 4. 原始分類（基準）
        comparison_data.append({
            '方法': '原始分類 (Baseline)',
            'ARI vs 原始分類': 1.0,
            'Silhouette Score': silhouette_score(features_df, original_clusters),
            '群數': 3,
            '評價': '📌 基準'
        })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        self.logger.info("\n方法比較結果:")
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def create_comparison_plot(self,
                              comparison_df: pd.DataFrame,
                              output_path: str = 'outputs/figures/chapter3_exploration/clustering_methods_comparison.html') -> str:
        """建立方法比較圖"""
        self.logger.info("建立方法比較圖...")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('ARI vs 原始分類', 'Silhouette Score'),
            horizontal_spacing=0.15
        )
        
        methods = comparison_df['方法'].tolist()
        
        # ARI
        fig.add_trace(
            go.Bar(
                x=methods,
                y=comparison_df['ARI vs 原始分類'],
                name='ARI',
                marker_color='lightblue',
                text=comparison_df['ARI vs 原始分類'].round(3),
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # Silhouette
        fig.add_trace(
            go.Bar(
                x=methods,
                y=comparison_df['Silhouette Score'],
                name='Silhouette',
                marker_color='lightgreen',
                text=comparison_df['Silhouette Score'].round(3),
                textposition='outside',
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(tickangle=-45, row=1, col=1)
        fig.update_xaxes(tickangle=-45, row=1, col=2)
        
        fig.update_yaxes(title_text="ARI", row=1, col=1)
        fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
        
        fig.update_layout(
            title_text='四種分群方法比較',
            font=dict(family="Arial, sans-serif", size=12),
            title_font_size=18,
            title_x=0.5,
            height=500
        )
        
        # 儲存
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        self.logger.info(f"方法比較圖已儲存: {output_path}")
        
        return output_path
    
    # ============================================
    # 主執行函數
    # ============================================
    
    def run_full_analysis(self) -> Dict:
        """執行完整的替代方法分析"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("開始替代分群方法分析...")
        self.logger.info("=" * 60)
        
        # 1. 載入資料
        features_df, original_clusters, country_labels = self.load_data()
        
        # 2. K-means
        self.logger.info("\n" + "=" * 60)
        self.logger.info("【方法1】K-means + Silhouette")
        self.logger.info("=" * 60)
        kmeans_results = self.kmeans_with_silhouette(features_df)
        kmeans_plot = self.create_kmeans_evaluation_plot(kmeans_results)
        
        # 3. DBSCAN
        self.logger.info("\n" + "=" * 60)
        self.logger.info("【方法2】DBSCAN")
        self.logger.info("=" * 60)
        dbscan_results = self.dbscan_clustering(features_df)
        dbscan_plot = self.create_dbscan_evaluation_plot(dbscan_results)
        
        # 4. t-SNE
        self.logger.info("\n" + "=" * 60)
        self.logger.info("【方法3】t-SNE降維視覺化")
        self.logger.info("=" * 60)
        tsne_results = self.tsne_visualization(features_df, country_labels, original_clusters)
        tsne_plot = self.create_tsne_plot(
            tsne_results,
            kmeans_labels=kmeans_results['best_model'].labels_
        )
        
        # 5. 取得階層分群結果（從3.3節）
        self.logger.info("\n" + "=" * 60)
        self.logger.info("【比較】四種方法")
        self.logger.info("=" * 60)
        
        # 重新執行階層分群以取得標籤
        distance_matrix = pdist(features_df.values, metric='euclidean')
        linkage_matrix = linkage(distance_matrix, method='ward')
        hierarchical_labels = fcluster(linkage_matrix, 3, criterion='maxclust')
        
        # 比較方法
        comparison_df = self.compare_methods(
            features_df,
            original_clusters,
            hierarchical_labels,
            kmeans_results,
            dbscan_results
        )
        
        comparison_plot = self.create_comparison_plot(comparison_df)
        
        # 6. 儲存結果
        output_dir = Path('outputs/tables/chapter3')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        comparison_df.to_csv(
            output_dir / 'clustering_methods_comparison.csv',
            index=False,
            encoding='utf-8-sig'
        )
        
        # 儲存t-SNE結果
        tsne_results['tsne_df'].to_csv(
            output_dir / 'tsne_coordinates.csv',
            index=False,
            encoding='utf-8-sig'
        )
        
        # 儲存K-means結果
        kmeans_clusters_df = pd.DataFrame({
            'Country': country_labels,
            'KMeans_Cluster': kmeans_results['best_model'].labels_,
            'Original_Cluster': original_clusters.values
        })
        kmeans_clusters_df.to_csv(
            output_dir / 'kmeans_clusters.csv',
            index=False,
            encoding='utf-8-sig'
        )
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("✅ 替代分群方法分析完成！")
        self.logger.info("=" * 60)
        
        return {
            'kmeans': kmeans_results,
            'dbscan': dbscan_results,
            'tsne': tsne_results,
            'comparison_df': comparison_df,
            'plots': {
                'kmeans_evaluation': kmeans_plot,
                'dbscan_evaluation': dbscan_plot,
                'tsne_visualization': tsne_plot,
                'methods_comparison': comparison_plot
            }
        }


if __name__ == '__main__':
    analyzer = AlternativeClusteringAnalyzer()
    results = analyzer.run_full_analysis()