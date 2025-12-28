"""
全球模式分析模組
================
負責第3章 3.1節的分析：全球道德地圖

功能：
1. 計算各國守法選擇率
2. 生成世界地圖熱圖（Plotly Choropleth）
3. 三大文化圈的箱型圖比較
4. 描述統計表格
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, Tuple
import logging


class GlobalPatternAnalyzer:
    """全球道德模式分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """設定日誌記錄器"""
        logger = logging.getLogger('GlobalPatternAnalyzer')
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
    
    def calculate_country_lawful_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算各國守法選擇率
        
        Parameters:
        -----------
        df : pd.DataFrame
            包含 chose_lawful 和 UserCountry3 的資料
            
        Returns:
        --------
        pd.DataFrame
            各國守法率統計
        """
        self.logger.info("計算各國守法選擇率...")
        
        country_stats = df.groupby('UserCountry3').agg({
            'chose_lawful': ['mean', 'std', 'count'],
            'Cluster': 'first'  # 取得文化圈分類
        }).reset_index()
        
        # 扁平化欄位名稱
        country_stats.columns = ['country', 'lawful_rate', 'lawful_std', 'n_responses', 'cluster']
        
        # 計算信賴區間
        country_stats['se'] = country_stats['lawful_std'] / np.sqrt(country_stats['n_responses'])
        country_stats['ci_lower'] = country_stats['lawful_rate'] - 1.96 * country_stats['se']
        country_stats['ci_upper'] = country_stats['lawful_rate'] + 1.96 * country_stats['se']
        
        # 確保CI在[0,1]範圍內
        country_stats['ci_lower'] = country_stats['ci_lower'].clip(0, 1)
        country_stats['ci_upper'] = country_stats['ci_upper'].clip(0, 1)
        
        self.logger.info(f"計算完成，共 {len(country_stats)} 個國家")
        
        return country_stats
    
    def create_world_map(self, 
                        country_stats: pd.DataFrame,
                        output_path: str = 'outputs/figures/chapter3_exploration/global_lawful_map.html') -> str:
        """
        建立全球守法率世界地圖
        
        Parameters:
        -----------
        country_stats : pd.DataFrame
            各國統計資料
        output_path : str
            輸出檔案路徑
            
        Returns:
        --------
        str
            輸出檔案路徑
        """
        self.logger.info("建立世界地圖...")
        
        # 建立文化圈標籤
        cluster_names = {
            0: 'Western',
            1: 'Eastern', 
            2: 'Southern'
        }
        country_stats['cluster_name'] = country_stats['cluster'].map(cluster_names)
        
        # 建立hover text
        country_stats['hover_text'] = (
            country_stats['country'] + '<br>' +
            '守法率: ' + (country_stats['lawful_rate'] * 100).round(1).astype(str) + '%<br>' +
            '文化圈: ' + country_stats['cluster_name'] + '<br>' +
            '樣本數: ' + country_stats['n_responses'].astype(str)
        )
        
        # 建立地圖
        fig = px.choropleth(
            country_stats,
            locations='country',
            locationmode='ISO-3',
            color='lawful_rate',
            hover_name='country',
            hover_data={
                'lawful_rate': ':.2%',
                'cluster_name': True,
                'n_responses': ':,',
                'country': False
            },
            color_continuous_scale='RdYlGn',  # 紅黃綠色階：紅=效益主義，綠=守法
            range_color=[0.2, 0.8],
            labels={'lawful_rate': '守法選擇率'},
            title='全球「守法vs.效益」道德地圖'
        )
        
        # 更新布局
        fig.update_layout(
            font=dict(family="Arial, sans-serif", size=14),
            title_font_size=20,
            title_x=0.5,
            height=600,
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth'
            )
        )
        
        # 儲存
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        self.logger.info(f"世界地圖已儲存: {output_path}")
        
        return output_path
    
    def create_cluster_comparison(self,
                                 df: pd.DataFrame,
                                 output_path: str = 'outputs/figures/chapter3_exploration/cluster_comparison.html') -> str:
        """
        建立三大文化圈的箱型圖比較
        
        Parameters:
        -----------
        df : pd.DataFrame
            原始資料（場景層級）
        output_path : str
            輸出檔案路徑
            
        Returns:
        --------
        str
            輸出檔案路徑
        """
        self.logger.info("建立文化圈比較圖...")
        
        # 計算每位使用者的守法率
        user_lawful = df.groupby(['UserID', 'Cluster']).agg({
            'chose_lawful': 'mean'
        }).reset_index()
        
        # 建立文化圈標籤
        cluster_names = {
            0: 'Western (西方)',
            1: 'Eastern (東方)', 
            2: 'Southern (南方)'
        }
        user_lawful['cluster_name'] = user_lawful['Cluster'].map(cluster_names)
        
        # 建立箱型圖
        fixed_order = ['Western (西方)', 'Eastern (東方)', 'Southern (南方)']
        fig = px.box(
            user_lawful,
            x='cluster_name',
            y='chose_lawful',
            color='cluster_name',
            points='outliers',  # 只顯示離群值
            labels={
                'cluster_name': '文化圈',
                'chose_lawful': '守法選擇率'
            },
            title='三大文化圈的守法選擇率分佈'
        )
        
        # 更新布局
        fig.update_layout(
            font=dict(family="Arial, sans-serif", size=14),
            title_font_size=20,
            title_x=0.5,
            height=500,
            showlegend=False,
            yaxis_tickformat='.0%',
            xaxis_title='',
            yaxis_title='守法選擇率'
        )
        
        # 加上平均線
        cluster_means = user_lawful.groupby('cluster_name')['chose_lawful'].mean()
        
        for i, name in enumerate(fixed_order):
            mean_val = user_lawful[user_lawful['cluster_name']==name]['chose_lawful'].mean()
            fig.add_annotation(x=i, y=mean_val, text=f'平均: {mean_val:.1%}', 
                            showarrow=False, yshift=10, font=dict(color='darkred', size=13))
        
        # 儲存
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        self.logger.info(f"文化圈比較圖已儲存: {output_path}")
        
        return output_path
    
    def generate_descriptive_stats(self,
                                  country_stats: pd.DataFrame,
                                  df: pd.DataFrame,
                                  output_path: str = 'outputs/tables/chapter3/global_descriptive_stats.csv') -> pd.DataFrame:
        """
        生成描述統計表格
        
        Parameters:
        -----------
        country_stats : pd.DataFrame
            各國統計
        df : pd.DataFrame
            原始資料
        output_path : str
            輸出檔案路徑
            
        Returns:
        --------
        pd.DataFrame
            描述統計表
        """
        self.logger.info("生成描述統計表...")
        
        stats_list = []
        
        # 全球統計
        stats_list.append({
            '層級': '全球',
            '分類': 'Overall',
            '平均守法率': f"{df['chose_lawful'].mean():.3f}",
            '標準差': f"{df['chose_lawful'].std():.3f}",
            '最小值': f"{df['chose_lawful'].min():.3f}",
            '最大值': f"{df['chose_lawful'].max():.3f}",
            '樣本數': f"{len(df):,}"
        })
        
        # 各文化圈統計
        cluster_names = {0: 'Western', 1: 'Eastern', 2: 'Southern'}
        
        for cluster_id, cluster_name in cluster_names.items():
            cluster_data = df[df['Cluster'] == cluster_id]
            if len(cluster_data) > 0:
                stats_list.append({
                    '層級': '文化圈',
                    '分類': cluster_name,
                    '平均守法率': f"{cluster_data['chose_lawful'].mean():.3f}",
                    '標準差': f"{cluster_data['chose_lawful'].std():.3f}",
                    '最小值': f"{cluster_data['chose_lawful'].min():.3f}",
                    '最大值': f"{cluster_data['chose_lawful'].max():.3f}",
                    '樣本數': f"{len(cluster_data):,}"
                })
        
        # 各國統計（前10名和後10名）
        top_10 = country_stats.nlargest(10, 'lawful_rate')
        bottom_10 = country_stats.nsmallest(10, 'lawful_rate')
        
        stats_df = pd.DataFrame(stats_list)
        
        # 儲存
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        # 另外儲存top/bottom國家
        top_bottom_path = output_path.replace('global_descriptive_stats', 'top_bottom_countries')
        
        top_10_export = top_10[['country', 'lawful_rate', 'cluster', 'n_responses']].copy()
        top_10_export['rank'] = 'Top 10'
        bottom_10_export = bottom_10[['country', 'lawful_rate', 'cluster', 'n_responses']].copy()
        bottom_10_export['rank'] = 'Bottom 10'
        
        top_bottom = pd.concat([top_10_export, bottom_10_export])
        top_bottom.to_csv(top_bottom_path, index=False, encoding='utf-8-sig')
        
        self.logger.info(f"描述統計表已儲存: {output_path}")
        self.logger.info(f"Top/Bottom國家已儲存: {top_bottom_path}")
        
        return stats_df
    
    def run_analysis(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        執行完整的全球模式分析
        
        Parameters:
        -----------
        df : pd.DataFrame
            清理後的資料（已排除 Cluster == -1）
            
        Returns:
        --------
        Dict[str, str]
            輸出檔案路徑字典
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("開始全球模式分析...")
        self.logger.info("=" * 60)
        
        # 1. 計算各國統計
        country_stats = self.calculate_country_lawful_rate(df)
        
        # 2. 建立世界地圖
        map_path = self.create_world_map(country_stats)
        
        # 3. 建立文化圈比較圖
        comparison_path = self.create_cluster_comparison(df)
        
        # 4. 生成描述統計
        stats_path = 'outputs/tables/chapter3/global_descriptive_stats.csv'
        self.generate_descriptive_stats(country_stats, df, stats_path)
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("全球模式分析完成！")
        self.logger.info("=" * 60)
        
        return {
            'world_map': map_path,
            'cluster_comparison': comparison_path,
            'descriptive_stats': stats_path
        }


if __name__ == '__main__':
    print("全球模式分析模組載入成功")