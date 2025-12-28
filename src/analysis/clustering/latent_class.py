"""
潛在類別分析模組
================
負責第3章 3.4節的分析：道德人格類型學

功能：
1. 建立使用者道德側寫（基於訓練集）
2. 使用Gaussian Mixture Model進行潛在類別分析
3. BIC曲線選擇最佳類別數
4. 類別特徵雷達圖
5. 類別在文化圈的分佈
"""

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, Tuple
import logging


class LatentClassAnalyzer:
    """潛在類別分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """設定日誌記錄器"""
        logger = logging.getLogger('LatentClassAnalyzer')
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
    
    def load_user_profiles(self, filepath: str = 'data/processed/user_moral_profiles.csv') -> pd.DataFrame:
        """
        載入使用者道德側寫
        
        Parameters:
        -----------
        filepath : str
            使用者側寫檔案路徑
            
        Returns:
        --------
        pd.DataFrame
            使用者側寫資料（僅訓練集）
        """
        self.logger.info(f"載入使用者道德側寫: {filepath}")
        profiles_df = pd.read_csv(filepath)
        
        # 僅使用訓練集（避免資料洩漏）
        if 'split' in profiles_df.columns:
            train_profiles = profiles_df[profiles_df['split'] == 'train'].copy()
            self.logger.info(f"僅使用訓練集: {len(train_profiles):,} 位使用者")
        else:
            train_profiles = profiles_df.copy()
            self.logger.info(f"使用全部資料: {len(train_profiles):,} 位使用者")
        
        return train_profiles
    
    def prepare_features(self, profiles_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        準備特徵矩陣
        
        Parameters:
        -----------
        profiles_df : pd.DataFrame
            使用者側寫
            
        Returns:
        --------
        Tuple[pd.DataFrame, np.ndarray]
            (原始特徵, 標準化特徵)
        """
        self.logger.info("準備特徵矩陣...")
        
        # 選擇特徵（效益主義和義務論分數）
        feature_cols = ['utilitarian_score', 'deontology_score']
        
        features_df = profiles_df[feature_cols].copy()
        
        # 標準化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df)
        
        self.logger.info(f"特徵準備完成: {features_df.shape}")
        
        return features_df, features_scaled
    
    def find_optimal_clusters(self,
                            features_scaled: np.ndarray,
                            k_range: range = range(2, 7)) -> Tuple[int, Dict]:
        """
        使用BIC找出最佳類別數
        
        Parameters:
        -----------
        features_scaled : np.ndarray
            標準化特徵
        k_range : range
            測試的類別數範圍
            
        Returns:
        --------
        Tuple[int, Dict]
            (最佳k, BIC分數字典)
        """
        self.logger.info(f"測試類別數: {list(k_range)}")
        
        bic_scores = {}
        
        for k in k_range:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type='full',
                random_state=42,
                n_init=10
            )
            gmm.fit(features_scaled)
            bic_scores[k] = gmm.bic(features_scaled)
            
            self.logger.info(f"  k={k}: BIC={bic_scores[k]:.2f}")
        
        # 找出BIC最小的k
        optimal_k = min(bic_scores, key=bic_scores.get)
        
        self.logger.info(f"✅ 最佳類別數: {optimal_k} (BIC={bic_scores[optimal_k]:.2f})")
        
        return optimal_k, bic_scores
    
    def fit_gmm(self,
               features_scaled: np.ndarray,
               n_clusters: int) -> GaussianMixture:
        """
        訓練GMM模型
        
        Parameters:
        -----------
        features_scaled : np.ndarray
            標準化特徵
        n_clusters : int
            類別數
            
        Returns:
        --------
        GaussianMixture
            訓練好的模型
        """
        self.logger.info(f"訓練GMM模型 (k={n_clusters})...")
        
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type='full',
            random_state=42,
            n_init=10
        )
        gmm.fit(features_scaled)
        
        self.logger.info("模型訓練完成")
        
        return gmm
    
    def create_bic_curve(self,
                        bic_scores: Dict,
                        optimal_k: int,
                        output_path: str = 'outputs/figures/chapter3_exploration/lca_bic_curve.html') -> str:
        """
        建立BIC曲線圖
        
        Parameters:
        -----------
        bic_scores : Dict
            BIC分數字典
        optimal_k : int
            最佳k值
        output_path : str
            輸出檔案路徑
            
        Returns:
        --------
        str
            輸出檔案路徑
        """
        self.logger.info("建立BIC曲線圖...")
        
        k_values = list(bic_scores.keys())
        bic_values = list(bic_scores.values())
        
        fig = go.Figure()
        
        # BIC曲線
        fig.add_trace(go.Scatter(
            x=k_values,
            y=bic_values,
            mode='lines+markers',
            name='BIC',
            line=dict(color='blue', width=2),
            marker=dict(size=10)
        ))
        
        # 標記最佳k
        fig.add_trace(go.Scatter(
            x=[optimal_k],
            y=[bic_scores[optimal_k]],
            mode='markers',
            name=f'最佳 k={optimal_k}',
            marker=dict(color='red', size=15, symbol='star')
        ))
        
        fig.update_layout(
            title='BIC曲線：潛在類別數選擇',
            xaxis_title='類別數 (k)',
            yaxis_title='BIC分數（越小越好）',
            font=dict(family="Arial, sans-serif", size=14),
            title_font_size=20,
            title_x=0.5,
            height=500,
            showlegend=True
        )
        
        # 儲存
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        self.logger.info(f"BIC曲線已儲存: {output_path}")
        
        return output_path
    
    def analyze_class_profiles(self,
                               profiles_df: pd.DataFrame,
                               features_df: pd.DataFrame,
                               labels: np.ndarray) -> pd.DataFrame:
        """
        分析各類別的特徵
        
        Parameters:
        -----------
        profiles_df : pd.DataFrame
            使用者側寫
        features_df : pd.DataFrame
            原始特徵
        labels : np.ndarray
            類別標籤
            
        Returns:
        --------
        pd.DataFrame
            類別特徵表
        """
        self.logger.info("分析類別特徵...")
        
        # 加上類別標籤
        profiles_with_class = profiles_df.copy()
        profiles_with_class['class'] = labels
        
        # 計算各類別的平均值
        class_profiles = profiles_with_class.groupby('class').agg({
            'utilitarian_score': 'mean',
            'deontology_score': 'mean',
            'consistency_score': 'mean',
            'n_scenarios': 'mean'
        }).reset_index()
        
        # 加上樣本數
        class_profiles['n_users'] = profiles_with_class.groupby('class').size().values
        class_profiles['percentage'] = (class_profiles['n_users'] / len(profiles_with_class) * 100).round(1)
        
        # 類別命名（基於特徵）
        class_names = []
        for _, row in class_profiles.iterrows():
            if row['utilitarian_score'] > 0.7:
                name = '強效益主義者'
            elif row['deontology_score'] > 0.7:
                name = '強義務論者'
            elif abs(row['utilitarian_score'] - 0.5) < 0.2:
                name = '中間派'
            else:
                name = f"類別{int(row['class'])}"
            class_names.append(name)
        
        class_profiles['class_name'] = class_names
        
        self.logger.info("\n類別特徵:")
        for _, row in class_profiles.iterrows():
            self.logger.info(f"  {row['class_name']}: {row['n_users']:,} 位 ({row['percentage']:.1f}%)")
            self.logger.info(f"    效益主義: {row['utilitarian_score']:.3f}, 義務論: {row['deontology_score']:.3f}")
        
        return class_profiles
    
    def create_class_radar(self,
                          class_profiles: pd.DataFrame,
                          output_path: str = 'outputs/figures/chapter3_exploration/lca_class_profiles.html') -> str:
        """
        建立類別特徵雷達圖
        
        Parameters:
        -----------
        class_profiles : pd.DataFrame
            類別特徵
        output_path : str
            輸出檔案路徑
            
        Returns:
        --------
        str
            輸出檔案路徑
        """
        self.logger.info("建立類別特徵雷達圖...")
        
        fig = go.Figure()
        
        dimensions = ['utilitarian_score', 'deontology_score', 'consistency_score']
        dimension_labels = ['效益主義', '義務論', '一致性']
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for idx, row in class_profiles.iterrows():
            values = [row[dim] for dim in dimensions]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=dimension_labels,
                fill='toself',
                name=f"{row['class_name']} ({row['percentage']:.1f}%)",
                line=dict(color=colors[idx % len(colors)], width=2),
                opacity=0.6
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title='潛在類別特徵雷達圖',
            font=dict(family="Arial, sans-serif", size=14),
            title_font_size=20,
            title_x=0.5,
            height=600
        )
        
        # 儲存
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        self.logger.info(f"類別雷達圖已儲存: {output_path}")
        
        return output_path
    
    def analyze_class_by_culture(self,
                                 profiles_df: pd.DataFrame,
                                 labels: np.ndarray,
                                 df: pd.DataFrame,
                                 output_path: str = 'outputs/figures/chapter3_exploration/lca_culture_distribution.html') -> str:
        """
        分析類別在文化圈的分佈
        
        Parameters:
        -----------
        profiles_df : pd.DataFrame
            使用者側寫
        labels : np.ndarray
            類別標籤
        df : pd.DataFrame
            原始資料（包含Cluster資訊）
        output_path : str
            輸出檔案路徑
            
        Returns:
        --------
        str
            輸出檔案路徑
        """
        self.logger.info("分析類別與文化圈的關係...")
        
        # 取得每位使用者的文化圈
        user_cluster = df.groupby('UserID')['Cluster'].first()
        
        # 合併
        profiles_with_class = profiles_df.copy()
        profiles_with_class['class'] = labels
        profiles_with_class['cluster'] = profiles_with_class['UserID'].map(user_cluster)
        
        # 移除沒有cluster資訊的
        profiles_with_class = profiles_with_class.dropna(subset=['cluster'])
        
        # 交叉表
        crosstab = pd.crosstab(
            profiles_with_class['class'],
            profiles_with_class['cluster'],
            normalize='index'
        ) * 100
        
        # 建立堆疊長條圖
        cluster_names = {0: 'Western', 1: 'Eastern', 2: 'Southern'}
        
        fig = go.Figure()
        
        for cluster_id in sorted(crosstab.columns):
            fig.add_trace(go.Bar(
                name=cluster_names[cluster_id],
                x=crosstab.index,
                y=crosstab[cluster_id],
                text=crosstab[cluster_id].round(1).astype(str) + '%',
                textposition='inside'
            ))
        
        fig.update_layout(
            barmode='stack',
            title='各類別在文化圈的分佈',
            xaxis_title='類別',
            yaxis_title='百分比 (%)',
            font=dict(family="Arial, sans-serif", size=14),
            title_font_size=20,
            title_x=0.5,
            height=500,
            showlegend=True,
            yaxis_ticksuffix='%'
        )
        
        # 儲存
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        self.logger.info(f"文化分佈圖已儲存: {output_path}")
        
        return output_path
    
    def run_analysis(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        執行完整的潛在類別分析
        
        Parameters:
        -----------
        df : pd.DataFrame
            原始資料
            
        Returns:
        --------
        Dict[str, any]
            分析結果字典
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("開始潛在類別分析...")
        self.logger.info("=" * 60)
        
        # 1. 載入使用者側寫
        profiles_df = self.load_user_profiles()
        
        # 2. 準備特徵
        features_df, features_scaled = self.prepare_features(profiles_df)
        
        # 3. 找出最佳類別數
        optimal_k, bic_scores = self.find_optimal_clusters(features_scaled)
        
        # 4. 建立BIC曲線
        bic_path = self.create_bic_curve(bic_scores, optimal_k)
        
        # 5. 訓練最佳模型
        gmm = self.fit_gmm(features_scaled, optimal_k)
        labels = gmm.predict(features_scaled)
        
        # 6. 分析類別特徵
        class_profiles = self.analyze_class_profiles(profiles_df, features_df, labels)
        
        # 7. 建立類別雷達圖
        radar_path = self.create_class_radar(class_profiles)
        
        # 8. 分析類別與文化的關係
        culture_path = self.analyze_class_by_culture(profiles_df, labels, df)
        
        # 9. 儲存結果
        output_dir = Path('outputs/tables/chapter3')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        class_profiles.to_csv(output_dir / 'lca_class_profiles.csv', index=False, encoding='utf-8-sig')
        
        # 儲存BIC分數
        bic_df = pd.DataFrame({
            'k': list(bic_scores.keys()),
            'BIC': list(bic_scores.values())
        })
        bic_df.to_csv(output_dir / 'lca_bic_scores.csv', index=False, encoding='utf-8-sig')
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("潛在類別分析完成！")
        self.logger.info("=" * 60)
        
        return {
            'bic_curve': bic_path,
            'class_radar': radar_path,
            'culture_distribution': culture_path,
            'optimal_k': optimal_k,
            'class_profiles': class_profiles
        }
    def run_sensitivity_analysis(self, profiles_df: pd.DataFrame = None) -> Dict:
        """
        執行敏感度分析：檢驗「極端分數」是否為測量假象
        
        【新增方法】此方法為3.4節的補充分析，不影響原有的run_analysis()
        
        Parameters:
        -----------
        profiles_df : pd.DataFrame, optional
            使用者側寫資料。如果為None，則自動載入
            
        Returns:
        --------
        Dict
            敏感度分析結果
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("開始LCA敏感度分析...")
        self.logger.info("=" * 60)
        
        # 載入資料（如果沒有提供）
        if profiles_df is None:
            profiles_df = self.load_user_profiles()
        
        # 依場景數量分層
        self.logger.info("依場景數量分層...")
        
        strata = {
            '1次': profiles_df[profiles_df['n_scenarios'] == 1].copy(),
            '2次': profiles_df[profiles_df['n_scenarios'] == 2].copy(),
            '3次以上': profiles_df[profiles_df['n_scenarios'] >= 3].copy()
        }
        
        stratified_results = {}
        
        for stratum_name, stratum_df in strata.items():
            if len(stratum_df) == 0:
                self.logger.warning(f"{stratum_name} 無資料")
                continue
            
            # 計算極端比例
            extreme_mask = stratum_df['utilitarian_score'].isin([0, 1])
            extreme_ratio = extreme_mask.mean()
            
            stratified_results[stratum_name] = {
                'n_users': len(stratum_df),
                'extreme_ratio': extreme_ratio,
                'consistency_mean': stratum_df['consistency_score'].mean(),
                'consistency_std': stratum_df['consistency_score'].std(),
                'util_mean': stratum_df['utilitarian_score'].mean(),
                'util_std': stratum_df['utilitarian_score'].std()
            }
            
            self.logger.info(f"\n【{stratum_name}】")
            self.logger.info(f"  使用者數: {len(stratum_df):,}")
            self.logger.info(f"  極端比例: {extreme_ratio:.1%}")
        
        # 建立極端比例比較圖
        extreme_plot = self._create_extreme_ratio_plot(stratified_results)
        
        # 統計檢定
        test_result = self._perform_chi_square_test(stratified_results, profiles_df)
        
        # 生成詮釋
        interpretation = self._generate_sensitivity_interpretation(stratified_results)
        
        # 儲存結果
        output_dir = Path('outputs/tables/chapter3')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary_df = pd.DataFrame([
            {
                '觀測次數': name,
                '使用者數': results['n_users'],
                '極端比例': results['extreme_ratio'],
                '一致性均值': results['consistency_mean'],
                '效益主義均值': results['util_mean']
            }
            for name, results in stratified_results.items()
        ])
        
        summary_df.to_csv(
            output_dir / 'lca_sensitivity_summary.csv',
            index=False,
            encoding='utf-8-sig'
        )
        
        test_result.to_csv(
            output_dir / 'lca_sensitivity_chi_square.csv',
            index=False,
            encoding='utf-8-sig'
        )
        
        with open(output_dir / 'lca_sensitivity_interpretation.md', 'w', encoding='utf-8') as f:
            f.write(interpretation)
        
        self.logger.info("\n" + interpretation)
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("敏感度分析完成！")
        self.logger.info("=" * 60)
        
        return {
            'extreme_plot': extreme_plot,
            'test_result': test_result,
            'interpretation': interpretation,
            'stratified_results': stratified_results
        }

    def _create_extreme_ratio_plot(self, stratified_results: Dict) -> str:
        """建立極端比例比較圖"""
        strata_names = list(stratified_results.keys())
        extreme_ratios = [stratified_results[s]['extreme_ratio'] for s in strata_names]
        n_users = [stratified_results[s]['n_users'] for s in strata_names]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=strata_names,
            y=extreme_ratios,
            text=[f'{r:.1%}<br>n={n:,}' for r, n in zip(extreme_ratios, n_users)],
            textposition='outside',
            marker=dict(
                color=extreme_ratios,
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title='極端比例')
            )
        ))
        
        fig.update_layout(
            title='敏感度分析：不同觀測次數的極端比例',
            xaxis_title='觀測次數',
            yaxis_title='極端分數比例（0或1）',
            yaxis_tickformat='.0%',
            yaxis_range=[0, 1.1],
            font=dict(family="Arial, sans-serif", size=14),
            title_font_size=20,
            title_x=0.5,
            height=500
        )
        
        fig.add_hline(y=0.9, line_dash="dash", line_color="red", 
                    annotation_text="90% 閾值", annotation_position="right")
        
        output_path = 'outputs/figures/chapter3_exploration/lca_sensitivity_extreme_ratio.html'
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        
        self.logger.info(f"極端比例圖已儲存: {output_path}")
        
        return output_path

    def _perform_chi_square_test(self, stratified_results: Dict, profiles_df: pd.DataFrame) -> pd.DataFrame:
        """執行卡方檢定"""
        from scipy.stats import chi2_contingency
        
        contingency_data = []
        for name, results in stratified_results.items():
            if '1次' in name:
                data = profiles_df[profiles_df['n_scenarios'] == 1]
            elif '2次' in name:
                data = profiles_df[profiles_df['n_scenarios'] == 2]
            else:  # 3次以上
                data = profiles_df[profiles_df['n_scenarios'] >= 3]
            
            if len(data) > 0:
                n_extreme = (data['utilitarian_score'].isin([0, 1])).sum()
                n_non_extreme = len(data) - n_extreme
                contingency_data.append([n_extreme, n_non_extreme])
        
        if len(contingency_data) >= 2:
            contingency_table = np.array(contingency_data)
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            self.logger.info(f"\n【卡方檢定】χ² = {chi2:.2f}, p = {p_value:.4f}")
            
            return pd.DataFrame({
                '檢定類型': ['卡方檢定'],
                'χ²值': [chi2],
                'p值': [p_value],
                '自由度': [dof],
                '顯著性': ['***' if p_value < 0.001 else ('*' if p_value < 0.05 else 'ns')]
            })
        else:
            return pd.DataFrame()

    def _generate_sensitivity_interpretation(self, stratified_results: Dict) -> str:
        """生成詮釋"""
        extreme_ratios = {k: v['extreme_ratio'] for k, v in stratified_results.items()}
        
        interpretation = []
        interpretation.append("## 敏感度分析：詮釋\n")
        
        three_plus = extreme_ratios.get('3次以上', 0)
        
        if three_plus > 0.90:
            interpretation.append("### ✅ 支持「真實決策模式」假設\n")
            interpretation.append(f"- 即使完成3次以上場景，極端比例仍高達 {three_plus:.1%}\n")
            interpretation.append("- 表示使用者確實有**穩定的道德立場**\n")
            interpretation.append("- 不太可能是測量假象\n")
        elif three_plus < 0.70:
            interpretation.append("### ⚠️ 支持「測量假象」假設\n")
            interpretation.append(f"- 完成3次以上場景的極端比例降至 {three_plus:.1%}\n")
            interpretation.append("- 表示**觀測次數不足**導致極端分數\n")
            interpretation.append("- 需要更多場景才能準確測量道德傾向\n")
        else:
            interpretation.append("### 🤔 混合模式\n")
            interpretation.append(f"- 完成3次以上場景的極端比例為 {three_plus:.1%}\n")
            interpretation.append("- 部分使用者確實有穩定立場\n")
            interpretation.append("- 但測量限制仍然存在\n")
        
        return "\n".join(interpretation)

if __name__ == '__main__':
    print("潛在類別分析模組載入成功")