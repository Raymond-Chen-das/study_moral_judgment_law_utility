"""
資料字典（Codebook）自動生成器
自動掃描CSV檔案並生成繁體中文說明文件
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import json
from datetime import datetime

class CodebookGenerator:
    """生成資料字典的工具類別"""

    def __init__(self, data_dir: str = 'data/raw'):
        """
        初始化Codebook生成器

        Parameters:
        -----------
        data_dir : str
            原始資料目錄路徑
        """
        self.data_dir = Path(data_dir)
        self.codebook_data = {}

        # 欄位中文說明對照表（MIT Moral Machine專用）
        self.field_descriptions = {
            # 通用欄位
            'Unnamed: 0': '資料列索引（由pandas自動生成）',
            '': '資料列索引或國家代碼',
            
            # 識別欄位
            'ResponseID': '場景唯一識別碼',
            'ExtendedSessionID': '完整Session識別碼（含UserID）',
            'UserID': '使用者唯一識別碼（瀏覽器指紋）',
            'UserCountry3': '使用者所在國家（ISO 3166-1 alpha-3代碼）',

            # 場景設計欄位
            'ScenarioOrder': '場景在Session中的順序（1-13）',
            'ScenarioType': '場景類型（Utilitarian/Gender/Age等）',
            'ScenarioTypeStrict': '嚴格場景類型分類',
            'AttributeLevel': '該結果的屬性層級',

            # 決策相關欄位
            'Saved': '該結果是否被選擇拯救（1=是, 0=否）',
            'Intervention': '車輛是否介入（0=維持原路線, 1=轉向）',
            'PedPed': '是否為行人vs.行人（1=是, 0=行人vs.乘客）',
            'Barrier': '該側是否有障礙物（1=乘客, 0=行人）',
            'CrossingSignal': '過馬路的合法性（0=無, 1=綠燈合法, 2=紅燈違法）',
            'NumberOfCharacters': '該側的角色數量（1-5）',
            'DiffNumberOFCharacters': '兩側人數差異（0-4）',

            # 預設選擇相關
            'DefaultChoice': '預設選擇（Males/Young/Fit/High/Hoomans/More）',
            'NonDefaultChoice': '非預設選擇（Females/Old/Fat/Low/Pets/Less）',
            'DefaultChoiceIsOmission': '預設選擇是否為不介入（1=是, 0=否）',

            # 人口統計欄位
            'Review_age': '使用者年齡（18-75）',
            'Review_gender': '使用者性別（male/female/other）',
            'Review_education': '最高教育程度',
            'Review_income': '年收入（美金）',
            'Review_political': '政治立場（0=保守, 1=進步）',
            'Review_religious': '宗教虔誠度（0=不虔誠, 1=虔誠）',

            # 技術欄位
            'Template': '使用裝置類型（desktop/mobile/tablet）',
            'DescriptionShown': '是否點擊查看場景描述（1=是, 0=否）',
            'LeftHand': '該結果顯示位置（1=左側, 0=右側）',

            # CountriesChangePr.csv 欄位 - Estimates (平均邊際因果效應估計值)
            '[Omission -> Commission]: Estimates': '不作為→作為(介入偏好) - AMCE估計值',
            '[Passengers -> Pedestrians]: Estimates': '乘客→行人 - AMCE估計值',
            'Law [Illegal -> Legal]: Estimates': '違法→合法(守法偏好) - AMCE估計值',
            'Gender [Male -> Female]: Estimates': '男性→女性(性別偏好) - AMCE估計值',
            'Fitness [Large -> Fit]: Estimates': '大型體型→健壯體型(體型偏好) - AMCE估計值',
            'Social Status [Low -> High]: Estimates': '低社會地位→高社會地位(階級偏好) - AMCE估計值',
            'Age [Elderly -> Young]: Estimates': '年長→年輕(年齡偏好) - AMCE估計值',
            'No. Characters [Less -> More]: Estimates': '較少人數→較多人數(效益主義指標) - AMCE估計值',
            'Species [Pets -> Humans]: Estimates': '寵物→人類(物種偏好) - AMCE估計值',
            
            # CountriesChangePr.csv 欄位 - se (標準誤)
            '[Omission -> Commission]: se': '不作為→作為(介入偏好) - 標準誤',
            '[Passengers -> Pedestrians]: se': '乘客→行人 - 標準誤',
            'Law [Illegal -> Legal]: se': '違法→合法(守法偏好) - 標準誤',
            'Gender [Male -> Female]: se': '男性→女性(性別偏好) - 標準誤',
            'Fitness [Large -> Fit]: se': '大型體型→健壯體型(體型偏好) - 標準誤',
            'Social Status [Low -> High]: se': '低社會地位→高社會地位(階級偏好) - 標準誤',
            'Age [Elderly -> Young]: se': '年長→年輕(年齡偏好) - 標準誤',
            'No. Characters [Less -> More]: se': '較少人數→較多人數(效益主義指標) - 標準誤',
            'Species [Pets -> Humans]: se': '寵物→人類(物種偏好) - 標準誤',

            # dendrogram_Culture.csv 欄位
            'id': '節點識別碼 - 樹狀圖的層級結構編號',
            'culture': '文化圈名稱 - 國家所屬的文化集群',
            'continent': '洲別 - 地理位置分類',
            
            # moral_distance.csv 欄位
            'Country': '國家代碼 - ISO 3166-1 alpha-3標準',
            'Distance': '道德距離值 - 該國與美國(基準國)的道德偏好差異程度',

            # countries.csv 欄位
            'ISO3': '國家代碼(ISO 3166-1 alpha-3)',
            'Country': '國家名稱(英文)',  
            'Cluster': '文化集群編號(0=Western, 1=Eastern, 2=Southern)',
        }

    def analyze_dataframe(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """
        分析單一DataFrame並生成統計摘要

        Parameters:
        -----------
        df : pd.DataFrame
            要分析的資料框
        filename : str
            檔案名稱

        Returns:
        --------
        Dict[str, Any]
            包含統計摘要的字典
        """
        summary = {
            '檔案名稱': filename,
            '分析時間': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '資料維度': {
                '列數': len(df),
                '欄數': len(df.columns),
            },
            '欄位資訊': []
        }

        for col in df.columns:
            col_info = self._analyze_column(df[col], col)
            summary['欄位資訊'].append(col_info)

        return summary

    def _analyze_column(self, series: pd.Series, col_name: str) -> Dict[str, Any]:
        """
        分析單一欄位

        Parameters:
        -----------
        series : pd.Series
            要分析的序列
        col_name : str
            欄位名稱

        Returns:
        --------
        Dict[str, Any]
            欄位統計資訊
        """
        info = {
            '欄位名稱': col_name,
            '中文說明': self.field_descriptions.get(col_name, '（待補充說明）'),
            '資料型態': str(series.dtype),
            '缺失值數量': int(series.isna().sum()),
            '缺失值比例': f"{series.isna().mean() * 100:.2f}%",
            '唯一值數量': series.nunique(),
        }

        # 根據資料型態提供不同的統計
        if pd.api.types.is_numeric_dtype(series):
            info.update(self._numeric_stats(series))
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            info.update(self._categorical_stats(series))

        return info

    def _numeric_stats(self, series: pd.Series) -> Dict[str, Any]:
        """數值型欄位的統計"""
        valid_data = series.dropna()

        if len(valid_data) == 0:
            return {'統計摘要': '無有效資料'}

        stats = {
            '統計摘要': {
                '平均值': f"{valid_data.mean():.4f}",
                '標準差': f"{valid_data.std():.4f}",
                '最小值': f"{valid_data.min():.4f}",
                '25%分位數': f"{valid_data.quantile(0.25):.4f}",
                '中位數': f"{valid_data.median():.4f}",
                '75%分位數': f"{valid_data.quantile(0.75):.4f}",
                '最大值': f"{valid_data.max():.4f}",
            }
        }

        # 檢測異常值（使用IQR方法）
        Q1 = valid_data.quantile(0.25)
        Q3 = valid_data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = valid_data[(valid_data < Q1 - 1.5 * IQR) | (valid_data > Q3 + 1.5 * IQR)]

        stats['異常值資訊'] = {
            '異常值數量': len(outliers),
            '異常值比例': f"{len(outliers) / len(valid_data) * 100:.2f}%"
        }

        return stats

    def _categorical_stats(self, series: pd.Series) -> Dict[str, Any]:
        """類別型欄位的統計"""
        valid_data = series.dropna()

        if len(valid_data) == 0:
            return {'統計摘要': '無有效資料'}

        # 取前10個最常出現的值
        top_values = valid_data.value_counts().head(10)

        stats = {
            '統計摘要': {
                '最常出現值': str(top_values.index[0]) if len(top_values) > 0 else 'N/A',
                '最常出現次數': int(top_values.iloc[0]) if len(top_values) > 0 else 0,
            },
            '前10個值的分佈': {
                str(k): {
                    '次數': int(v),
                    '比例': f"{v / len(valid_data) * 100:.2f}%"
                }
                for k, v in top_values.items()
            }
        }

        return stats

    def generate_codebook(self, output_format: str = 'markdown') -> str:
        """
        生成所有CSV檔案的Codebook

        Parameters:
        -----------
        output_format : str
            輸出格式（'markdown', 'excel', 'json'）

        Returns:
        --------
        str
            輸出檔案路徑
        """
        # 掃描所有CSV檔案
        csv_files = list(self.data_dir.glob('*.csv'))

        print(f"找到 {len(csv_files)} 個CSV檔案")

        for csv_file in csv_files:
            print(f"分析中: {csv_file.name}")

            # 讀取CSV（只讀取前10000行以節省記憶體）
            try:
                df = pd.read_csv(csv_file, nrows=10000)
                summary = self.analyze_dataframe(df, csv_file.name)
                self.codebook_data[csv_file.name] = summary
            except Exception as e:
                print(f"  錯誤: {e}")
                continue

        # 根據格式輸出
        if output_format == 'markdown':
            return self._generate_markdown()
        elif output_format == 'excel':
            return self._generate_excel()
        elif output_format == 'json':
            return self._generate_json()
        else:
            raise ValueError(f"不支援的輸出格式: {output_format}")

    def _generate_markdown(self) -> str:
        """生成Markdown格式的Codebook"""
        output_path = self.data_dir.parent / 'metadata' / 'data_dictionary.md'
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# MIT Moral Machine 資料字典\n\n")
            f.write(f"生成時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            for filename, summary in self.codebook_data.items():
                f.write(f"## {filename}\n\n")
                f.write(f"**資料維度**：{summary['資料維度']['列數']} 列 × {summary['資料維度']['欄數']} 欄\n\n")

                f.write("### 欄位清單\n\n")
                f.write("| 欄位名稱 | 中文說明 | 資料型態 | 缺失值 | 唯一值數 |\n")
                f.write("|---------|---------|---------|--------|----------|\n")

                for col_info in summary['欄位資訊']:
                    f.write(f"| {col_info['欄位名稱']} | {col_info['中文說明']} | "
                           f"{col_info['資料型態']} | {col_info['缺失值比例']} | "
                           f"{col_info['唯一值數量']} |\n")

                f.write("\n### 欄位詳細資訊\n\n")

                for col_info in summary['欄位資訊']:
                    f.write(f"#### {col_info['欄位名稱']}\n\n")
                    f.write(f"**中文說明**：{col_info['中文說明']}\n\n")
                    f.write(f"**資料型態**：{col_info['資料型態']}\n\n")
                    f.write(f"**缺失值**：{col_info['缺失值數量']} ({col_info['缺失值比例']})\n\n")

                    if '統計摘要' in col_info:
                        if isinstance(col_info['統計摘要'], dict):
                            f.write("**統計摘要**：\n\n")
                            for key, value in col_info['統計摘要'].items():
                                if isinstance(value, dict):
                                    f.write(f"- {key}：\n")
                                    for k, v in value.items():
                                        f.write(f"  - {k}：{v}\n")
                                else:
                                    f.write(f"- {key}：{value}\n")
                            f.write("\n")

                    if '前10個值的分佈' in col_info:
                        f.write("**前10個值的分佈**：\n\n")
                        f.write("| 值 | 次數 | 比例 |\n")
                        f.write("|----|------|------|\n")
                        for value, stats in col_info['前10個值的分佈'].items():
                            f.write(f"| {value} | {stats['次數']} | {stats['比例']} |\n")
                        f.write("\n")

                    f.write("---\n\n")

                f.write("\n\n")

        print(f"✅ Markdown Codebook 已生成：{output_path}")
        return str(output_path)

    def _generate_excel(self) -> str:
        """生成Excel格式的Codebook"""
        output_path = self.data_dir.parent / 'metadata' / 'data_dictionary.xlsx'
        output_path.parent.mkdir(exist_ok=True)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 建立總覽工作表
            overview_data = []
            for filename, summary in self.codebook_data.items():
                overview_data.append({
                    '檔案名稱': filename,
                    '列數': summary['資料維度']['列數'],
                    '欄數': summary['資料維度']['欄數'],
                    '總缺失值': sum(col['缺失值數量'] for col in summary['欄位資訊']),
                })

            overview_df = pd.DataFrame(overview_data)
            overview_df.to_excel(writer, sheet_name='總覽', index=False)

            # 為每個檔案建立工作表
            for filename, summary in self.codebook_data.items():
                # 簡化檔名作為工作表名稱（Excel限制31字元）
                sheet_name = filename.replace('.csv', '')[:31]

                # 建立欄位資訊表
                fields_data = []
                for col_info in summary['欄位資訊']:
                    row = {
                        '欄位名稱': col_info['欄位名稱'],
                        '中文說明': col_info['中文說明'],
                        '資料型態': col_info['資料型態'],
                        '缺失值數量': col_info['缺失值數量'],
                        '缺失值比例': col_info['缺失值比例'],
                        '唯一值數量': col_info['唯一值數量'],
                    }

                    # 加入統計摘要
                    if '統計摘要' in col_info and isinstance(col_info['統計摘要'], dict):
                        for key, value in col_info['統計摘要'].items():
                            if not isinstance(value, dict):
                                row[key] = value

                    fields_data.append(row)

                fields_df = pd.DataFrame(fields_data)
                fields_df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"✅ Excel Codebook 已生成：{output_path}")
        return str(output_path)

    def _generate_json(self) -> str:
        """生成JSON格式的Codebook"""
        output_path = self.data_dir.parent / 'metadata' / 'data_dictionary.json'
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.codebook_data, f, ensure_ascii=False, indent=2)

        print(f"✅ JSON Codebook 已生成：{output_path}")
        return str(output_path)


# 使用範例
if __name__ == '__main__':
    # 建立生成器
    generator = CodebookGenerator(data_dir='data/raw')

    # 生成所有格式的Codebook
    print("=" * 60)
    print("開始生成 Codebook...")
    print("=" * 60)

    markdown_path = generator.generate_codebook(output_format='markdown')
    excel_path = generator.generate_codebook(output_format='excel')
    json_path = generator.generate_codebook(output_format='json')

    print("\n" + "=" * 60)
    print("✅ 所有 Codebook 生成完成！")
    print("=" * 60)
    print(f"\n📄 Markdown: {markdown_path}")
    print(f"📊 Excel: {excel_path}")
    print(f"📋 JSON: {json_path}")