"""
東亞聚焦分析模組
================
負責第3章 3.2節的分析：台灣與東亞的道德定位

功能：
1. 雷達圖：台、日、韓、中四國在9個道德維度的比較
2. 分組點圖：更清晰的信賴區間視覺化
3. 計算國家間的歐式距離
4. 生成比較表格
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Tuple
import logging


class EastAsiaAnalyzer:
    """東亞聚焦分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.logger = self._setup_logger()
        self.east_asia_countries = {
            'TWN': '台灣',
            'JPN': '日本',
            'KOR': '韓國',
            'CHN': '中國'
        }
        
    def _setup_logger(self) -> logging.Logger:
        """設定日誌記錄器"""
        logger = logging.getLogger('EastAsiaAnalyzer')
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
    
    def load_country_amce(self, filepath: str = 'data/raw/CountriesChangePr.csv') -> pd.DataFrame:
        """
        載入國家AMCE值
        
        Parameters:
        -----------
        filepath : str
            CountriesChangePr.csv 路徑
            
        Returns:
        --------
        pd.DataFrame
            國家AMCE資料
        """
        self.logger.info(f"載入國家AMCE資料: {filepath}")
        df = pd.read_csv(filepath, index_col=0)
        return df
    
    def extract_east_asia_data(self, countries_df: pd.DataFrame) -> pd.DataFrame:
        """
        提取東亞四國資料
        
        Parameters:
        -----------
        countries_df : pd.DataFrame
            完整國家資料
            
        Returns:
        --------
        pd.DataFrame
            東亞四國資料
        """
        self.logger.info("提取東亞四國資料...")
        
        # 篩選東亞四國
        east_asia_df = countries_df.loc[list(self.east_asia_countries.keys())].copy()
        
        # 選擇9個道德維度的AMCE值（Estimates欄位）
        amce_cols = [col for col in east_asia_df.columns if 'Estimates' in col]
        
        # 重新命名為簡短名稱
        rename_map = {
            'Law [Illegal -> Legal]: Estimates': '守法偏好',
            'No. Characters [Less -> More]: Estimates': '效益主義',
            '[Omission -> Commission]: Estimates': '介入偏好',
            '[Passengers -> Pedestrians]: Estimates': '行人優先',
            'Gender [Male -> Female]: Estimates': '性別偏好',
            'Fitness [Large -> Fit]: Estimates': '體型偏好',
            'Social Status [Low -> High]: Estimates': '地位偏好',
            'Age [Elderly -> Young]: Estimates': '年齡偏好',
            'Species [Pets -> Humans]: Estimates': '物種偏好'
        }
        
        east_asia_df = east_asia_df[amce_cols].rename(columns=rename_map)
        
        # 加上中文國名
        east_asia_df['國家'] = east_asia_df.index.map(self.east_asia_countries)
        
        self.logger.info(f"提取完成，共 {len(east_asia_df)} 個國家，{len(amce_cols)} 個維度")
        
        return east_asia_df
    
    def create_radar_chart(self,
                          east_asia_df: pd.DataFrame,
                          output_path: str = 'outputs/figures/chapter3_exploration/east_asia_radar.html') -> str:
        """
        建立東亞四國雷達圖
        
        Parameters:
        -----------
        east_asia_df : pd.DataFrame
            東亞四國資料
        output_path : str
            輸出檔案路徑
            
        Returns:
        --------
        str
            輸出檔案路徑
        """
        self.logger.info("建立東亞四國雷達圖...")
        
        # 準備資料
        dimensions = [col for col in east_asia_df.columns if col != '國家']
        
        # 建立圖表
        fig = go.Figure()
        
        # 顏色配置
        colors = {
            '台灣': 'rgb(31, 119, 180)',    # 藍色
            '日本': 'rgb(255, 127, 14)',    # 橙色
            '韓國': 'rgb(44, 160, 44)',     # 綠色
            '中國': 'rgb(214, 39, 40)'      # 紅色
        }
        
        # 為每個國家添加trace
        for idx, row in east_asia_df.iterrows():
            country_name = row['國家']
            values = [row[dim] for dim in dimensions]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=dimensions,
                fill='toself',
                name=country_name,
                line=dict(color=colors[country_name], width=2),
                opacity=0.6
            ))
        
        # 更新布局
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-0.2, 0.8]  # 根據AMCE值範圍調整
                )
            ),
            showlegend=True,
            title='東亞四國道德維度比較（雷達圖）',
            font=dict(family="Arial, sans-serif", size=14),
            title_font_size=20,
            title_x=0.5,
            height=700
        )
        
        # 儲存
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        self.logger.info(f"雷達圖已儲存: {output_path}")
        
        return output_path
    
    def create_grouped_dot_plot(self,
                                east_asia_df: pd.DataFrame,
                                countries_df: pd.DataFrame,
                                output_path: str = 'outputs/figures/chapter3_exploration/east_asia_dot_plot.html') -> str:
        """
        建立分組點圖（含信賴區間）
        
        Parameters:
        -----------
        east_asia_df : pd.DataFrame
            東亞四國資料（已重命名的維度）
        countries_df : pd.DataFrame
            原始完整資料（包含 se 欄位）
        output_path : str
            輸出檔案路徑
        
        Returns:
        --------
        str
            輸出檔案路徑
        """
        self.logger.info("建立分組點圖（含信賴區間）...")
        
        # 準備資料
        dimensions = [col for col in east_asia_df.columns if col != '國家']
        
        # 建立 AMCE 欄位到 se 欄位的映射
        amce_to_se_map = {
            '守法偏好': 'Law [Illegal -> Legal]: se',
            '效益主義': 'No. Characters [Less -> More]: se',
            '介入偏好': '[Omission -> Commission]: se',
            '行人優先': '[Passengers -> Pedestrians]: se',
            '性別偏好': 'Gender [Male -> Female]: se',
            '體型偏好': 'Fitness [Large -> Fit]: se',
            '地位偏好': 'Social Status [Low -> High]: se',
            '年齡偏好': 'Age [Elderly -> Young]: se',
            '物種偏好': 'Species [Pets -> Humans]: se'
        }
        
        # 顏色與符號配置
        country_styles = {
            '台灣': {'color': '#1f77b4', 'symbol': 'circle'},
            '日本': {'color': '#ff7f0e', 'symbol': 'square'},
            '韓國': {'color': '#2ca02c', 'symbol': 'diamond'},
            '中國': {'color': '#d62728', 'symbol': 'triangle-up'}
        }
        
        # 建立圖表
        fig = go.Figure()
        
        # 為每個國家添加軌跡
        for country_code, country_row in east_asia_df.iterrows():
            country_name = country_row['國家']
            style = country_styles[country_name]
            
            # 收集該國所有維度的資料
            y_values = []
            x_values = []
            error_x = []
            hover_texts = []
            
            for dim_idx, dimension in enumerate(dimensions):
                amce_value = country_row[dimension]
                
                # 取得標準誤
                se_col = amce_to_se_map[dimension]
                se_value = countries_df.loc[country_code, se_col]
                
                ci_lower = amce_value - 1.96 * se_value
                ci_upper = amce_value + 1.96 * se_value
                
                # 用維度索引作為y軸位置，每個國家稍微偏移
                offset_map = {'台灣': -0.15, '日本': -0.05, '韓國': 0.05, '中國': 0.15}
                y_position = dim_idx + offset_map[country_name]
                
                y_values.append(y_position)
                x_values.append(amce_value)
                
                # 計算誤差線長度
                error_x.append({
                    'lower': amce_value - ci_lower,
                    'upper': ci_upper - amce_value
                })
                
                hover_texts.append(
                    f'<b>{country_name}</b><br>' +
                    f'維度: {dimension}<br>' +
                    f'AMCE: {amce_value:.3f}<br>' +
                    f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]'
                )
            
            # 添加散點+誤差線
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='markers',
                name=country_name,
                marker=dict(
                    color=style['color'],
                    size=10,
                    symbol=style['symbol'],
                    line=dict(width=1, color='white')
                ),
                error_x=dict(
                    type='data',
                    symmetric=False,
                    array=[e['upper'] for e in error_x],
                    arrayminus=[e['lower'] for e in error_x],
                    thickness=1.5,
                    width=3,
                    color=style['color']
                ),
                hovertext=hover_texts,
                hoverinfo='text',
                showlegend=True
            ))
        
        # 添加零線
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        # 更新佈局
        fig.update_layout(
            title={
                'text': '東亞四國道德維度比較（含95%信賴區間）',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial'}
            },
            xaxis_title='AMCE 值',
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(dimensions))),
                ticktext=dimensions,
                autorange='reversed'  # 反轉y軸，維度從上到下
            ),
            height=700,
            width=1000,
            legend=dict(
                orientation='v',
                yanchor='top',
                y=1,
                xanchor='left',
                x=1.02,
                font=dict(size=13)
            ),
            font=dict(family="Arial, sans-serif", size=13),
            plot_bgcolor='white',
            hovermode='closest',
            margin=dict(l=120, r=150, t=80, b=80)
        )
        
        # 添加網格線
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')
        
        # 添加註解
        fig.add_annotation(
            text='註：橫線表示95%信賴區間。不同國家在同一維度略微上下錯開以利閱讀',
            xref='paper', yref='paper',
            x=0.5, y=-0.12,
            showarrow=False,
            font=dict(size=11, color='gray'),
            xanchor='center'
        )
        
        # 儲存
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        self.logger.info(f"分組點圖已儲存: {output_path}")
        
        return output_path
    
    def calculate_euclidean_distances(self, east_asia_df: pd.DataFrame) -> pd.DataFrame:
        """
        計算國家間的歐式距離
        
        Parameters:
        -----------
        east_asia_df : pd.DataFrame
            東亞四國資料
            
        Returns:
        --------
        pd.DataFrame
            距離矩陣
        """
        self.logger.info("計算國家間歐式距離...")
        
        # 取得維度欄位
        dimensions = [col for col in east_asia_df.columns if col != '國家']
        
        # 提取數值
        countries = east_asia_df['國家'].values
        values = east_asia_df[dimensions].values
        
        # 計算距離矩陣
        n = len(countries)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                distance_matrix[i, j] = np.linalg.norm(values[i] - values[j])
        
        # 轉換為DataFrame
        distance_df = pd.DataFrame(
            distance_matrix,
            index=countries,
            columns=countries
        )
        
        self.logger.info("距離計算完成")
        
        return distance_df
    
    def create_comparison_table(self,
                               east_asia_df: pd.DataFrame,
                               output_path: str = 'outputs/tables/chapter3/east_asia_comparison.csv') -> pd.DataFrame:
        """
        建立比較表格
        
        Parameters:
        -----------
        east_asia_df : pd.DataFrame
            東亞四國資料
        output_path : str
            輸出檔案路徑
            
        Returns:
        --------
        pd.DataFrame
            比較表
        """
        self.logger.info("建立比較表格...")
        
        # 轉置資料（維度為列，國家為欄）
        dimensions = [col for col in east_asia_df.columns if col != '國家']
        comparison_df = east_asia_df.set_index('國家')[dimensions].T
        
        # 加上排序
        for country in comparison_df.columns:
            comparison_df[f'{country}_排名'] = comparison_df[country].rank(ascending=False).astype(int)
        
        # 儲存
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(output_path, encoding='utf-8-sig')
        
        self.logger.info(f"比較表已儲存: {output_path}")
        
        return comparison_df
    
    def create_distance_heatmap(self,
                               distance_df: pd.DataFrame,
                               output_path: str = 'outputs/figures/chapter3_exploration/east_asia_distance.html') -> str:
        """
        建立距離熱圖
        
        Parameters:
        -----------
        distance_df : pd.DataFrame
            距離矩陣
        output_path : str
            輸出檔案路徑
            
        Returns:
        --------
        str
            輸出檔案路徑
        """
        self.logger.info("建立距離熱圖...")
        
        fig = go.Figure(data=go.Heatmap(
            z=distance_df.values,
            x=distance_df.columns,
            y=distance_df.index,
            colorscale='Blues',
            text=np.round(distance_df.values, 3),
            texttemplate='%{text}',
            textfont={"size": 14},
            colorbar=dict(title='歐式距離')
        ))
        
        fig.update_layout(
            title='東亞四國道德距離矩陣',
            font=dict(family="Arial, sans-serif", size=14),
            title_font_size=20,
            title_x=0.5,
            height=500,
            xaxis_title='',
            yaxis_title=''
        )
        
        # 儲存
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        self.logger.info(f"距離熱圖已儲存: {output_path}")
        
        return output_path
    
    def run_analysis(self, countries_filepath: str = 'data/raw/CountriesChangePr.csv') -> Dict[str, str]:
        """
        執行完整的東亞分析
        
        Parameters:
        -----------
        countries_filepath : str
            CountriesChangePr.csv 路徑
            
        Returns:
        --------
        Dict[str, str]
            輸出檔案路徑字典
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("開始東亞聚焦分析...")
        self.logger.info("=" * 60)
        
        # 1. 載入並提取東亞資料
        countries_df = self.load_country_amce(countries_filepath)
        east_asia_df = self.extract_east_asia_data(countries_df)
        
        # 2. 建立雷達圖（保留）
        radar_path = self.create_radar_chart(east_asia_df)
        
        # 3. 🆕 建立分組點圖（主要視覺化）
        dot_plot_path = self.create_grouped_dot_plot(east_asia_df, countries_df)
        
        # 4. 計算距離
        distance_df = self.calculate_euclidean_distances(east_asia_df)
        
        # 5. 建立距離熱圖
        heatmap_path = self.create_distance_heatmap(distance_df)
        
        # 6. 建立比較表
        comparison_path = self.create_comparison_table(east_asia_df)
        
        # 7. 儲存距離矩陣
        distance_path = 'outputs/tables/chapter3/east_asia_distances.csv'
        Path(distance_path).parent.mkdir(parents=True, exist_ok=True)
        distance_df.to_csv(distance_path, encoding='utf-8-sig')
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("東亞聚焦分析完成！")
        self.logger.info("=" * 60)
        
        return {
            'radar_chart': radar_path,
            'dot_plot': dot_plot_path,  # 🆕 主要視覺化
            'distance_heatmap': heatmap_path,
            'comparison_table': comparison_path,
            'distance_matrix': distance_path
        }


if __name__ == '__main__':
    print("東亞聚焦分析模組載入成功")