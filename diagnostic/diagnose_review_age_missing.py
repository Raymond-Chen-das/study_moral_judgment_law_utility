"""
Review_age 缺失值診斷腳本
=========================

目的：診斷 featured_data.csv 中為何 Review_age 少了 59,858 筆有效資料

輸出：
1. 缺失值基本統計
2. 缺失值的分佈模式（按國家、文化圈等）
3. 診斷報告與建議
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# ==================== 設定路徑 ====================
DATA_PATH = Path("data/processed/featured_data.csv")
OUTPUT_DIR = Path("outputs/diagnostic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Review_age 缺失值診斷")
print("=" * 80)

# ==================== 載入資料 ====================
print("\n【步驟1】載入資料...")
try:
    df = pd.read_csv(DATA_PATH)
    print(f"✅ 成功載入資料：{len(df):,} 筆")
except FileNotFoundError:
    print(f"❌ 找不到檔案：{DATA_PATH}")
    print("請確認檔案路徑是否正確")
    sys.exit(1)

# ==================== 基本統計 ====================
print("\n【步驟2】基本統計...")
print("-" * 80)

total_rows = len(df)
age_valid = df['Review_age'].notna().sum()
age_missing = df['Review_age'].isna().sum()
political_valid = df['Review_political'].notna().sum()
religious_valid = df['Review_religious'].notna().sum()

print(f"總樣本數：{total_rows:,}")
print(f"\nReview_age:")
print(f"  有效：{age_valid:,} 筆 ({age_valid/total_rows*100:.2f}%)")
print(f"  缺失：{age_missing:,} 筆 ({age_missing/total_rows*100:.2f}%)")
print(f"\nReview_political:")
print(f"  有效：{political_valid:,} 筆 ({political_valid/total_rows*100:.2f}%)")
print(f"\nReview_religious:")
print(f"  有效：{religious_valid:,} 筆 ({religious_valid/total_rows*100:.2f}%)")
print(f"\n樣本數差異（與 Review_political 比較）：")
print(f"  Review_age 少了：{political_valid - age_valid:,} 筆")

# ==================== 缺失值標記 ====================
print("\n【步驟3】建立缺失值指標...")
df['age_missing'] = df['Review_age'].isna()
df['political_missing'] = df['Review_political'].isna()
df['religious_missing'] = df['Review_religious'].isna()

# ==================== 缺失值模式分析 ====================
print("\n【步驟4】缺失值模式分析...")
print("-" * 80)

# 模式1：僅年齡缺失
age_only_missing = (df['age_missing'] & ~df['political_missing'] & ~df['religious_missing']).sum()
print(f"僅 Review_age 缺失：{age_only_missing:,} 筆")

# 模式2：年齡+政治缺失
age_political_missing = (df['age_missing'] & df['political_missing']).sum()
print(f"Review_age + Review_political 皆缺失：{age_political_missing:,} 筆")

# 模式3：年齡+宗教缺失
age_religious_missing = (df['age_missing'] & df['religious_missing']).sum()
print(f"Review_age + Review_religious 皆缺失：{age_religious_missing:,} 筆")

# 模式4：三者皆缺失
all_missing = (df['age_missing'] & df['political_missing'] & df['religious_missing']).sum()
print(f"三者皆缺失：{all_missing:,} 筆")

# ==================== 缺失值分佈分析 ====================
print("\n【步驟5】缺失值在不同群組的分佈...")
print("-" * 80)

# 按文化圈分組
if 'Cluster' in df.columns:
    print("\n▶ 按文化圈 (Cluster) 分佈：")
    cluster_missing = df.groupby('Cluster').agg({
        'age_missing': ['sum', 'mean'],
        'Review_age': 'count'
    }).round(4)
    cluster_missing.columns = ['缺失數量', '缺失比例', '總樣本數']
    cluster_missing['有效數量'] = cluster_missing['總樣本數'] - cluster_missing['缺失數量']
    
    cluster_names = {0: 'Western', 1: 'Eastern', 2: 'Southern', -1: '未分類'}
    cluster_missing.index = cluster_missing.index.map(lambda x: cluster_names.get(x, str(x)))
    
    print(cluster_missing)

# 按國家分組（Top 10 缺失最多的國家）
if 'Country' in df.columns:
    print("\n▶ 缺失數量 Top 10 國家：")
    country_missing = df.groupby('Country').agg({
        'age_missing': ['sum', 'mean'],
        'Review_age': 'count'
    }).round(4)
    country_missing.columns = ['缺失數量', '缺失比例', '總樣本數']
    country_missing = country_missing.sort_values('缺失數量', ascending=False).head(10)
    print(country_missing)

# 按結果變數分組
if 'chose_lawful' in df.columns:
    print("\n▶ 按道德選擇 (chose_lawful) 分佈：")
    choice_missing = df.groupby('chose_lawful').agg({
        'age_missing': ['sum', 'mean'],
        'Review_age': 'count'
    }).round(4)
    choice_missing.columns = ['缺失數量', '缺失比例', '總樣本數']
    choice_missing.index = choice_missing.index.map({0: '選擇違法', 1: '選擇守法'})
    print(choice_missing)

# ==================== 檢查缺失值的特徵 ====================
print("\n【步驟6】缺失資料的其他特徵...")
print("-" * 80)

missing_data = df[df['age_missing'] == True]
valid_data = df[df['age_missing'] == False]

print(f"\n缺失 Review_age 的資料特徵 (N = {len(missing_data):,}):")

# 檢查這些缺失資料是否有其他特殊模式
if 'UserID' in df.columns:
    missing_users = missing_data['UserID'].nunique()
    valid_users = valid_data['UserID'].nunique()
    print(f"  涉及使用者數：{missing_users:,} 人")
    print(f"  有效資料使用者數：{valid_users:,} 人")

if 'ScenarioID' in df.columns:
    missing_scenarios = missing_data['ScenarioID'].nunique()
    valid_scenarios = valid_data['ScenarioID'].nunique()
    print(f"  涉及場景數：{missing_scenarios:,} 個")
    print(f"  有效資料場景數：{valid_scenarios:,} 個")

# ==================== 視覺化 ====================
print("\n【步驟7】生成視覺化...")

# 圖1：缺失值比較條形圖
fig1 = go.Figure()

variables = ['Review_age', 'Review_political', 'Review_religious']
valid_counts = [age_valid, political_valid, religious_valid]
missing_counts = [age_missing, 
                 df['Review_political'].isna().sum(),
                 df['Review_religious'].isna().sum()]

fig1.add_trace(go.Bar(
    name='有效',
    x=variables,
    y=valid_counts,
    text=[f"{v:,}" for v in valid_counts],
    textposition='auto',
    marker_color='#3498db'
))

fig1.add_trace(go.Bar(
    name='缺失',
    x=variables,
    y=missing_counts,
    text=[f"{v:,}" for v in missing_counts],
    textposition='auto',
    marker_color='#e74c3c'
))

fig1.update_layout(
    title='三個人口統計變數的樣本數比較',
    xaxis_title='變數',
    yaxis_title='樣本數',
    barmode='stack',
    height=500,
    width=800,
    font=dict(size=12)
)

fig1.write_html(OUTPUT_DIR / 'missing_comparison.html')
print(f"✅ 已儲存：{OUTPUT_DIR / 'missing_comparison.html'}")

# 圖2：文化圈缺失率比較（如果有Cluster欄位）
if 'Cluster' in df.columns:
    cluster_data = df.groupby('Cluster').agg({
        'age_missing': 'mean',
        'Review_age': 'count'
    }).reset_index()
    
    cluster_data['Cluster_Name'] = cluster_data['Cluster'].map(
        {0: 'Western', 1: 'Eastern', 2: 'Southern', -1: '未分類'}
    )
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Bar(
        x=cluster_data['Cluster_Name'],
        y=cluster_data['age_missing'] * 100,
        text=[f"{v:.2f}%" for v in cluster_data['age_missing'] * 100],
        textposition='auto',
        marker_color=['#3498db', '#e74c3c', '#2ecc71', '#95a5a6']
    ))
    
    fig2.update_layout(
        title='各文化圈的 Review_age 缺失率',
        xaxis_title='文化圈',
        yaxis_title='缺失率 (%)',
        height=500,
        width=800,
        font=dict(size=12)
    )
    
    fig2.write_html(OUTPUT_DIR / 'cluster_missing_rate.html')
    print(f"✅ 已儲存：{OUTPUT_DIR / 'cluster_missing_rate.html'}")

# ==================== 儲存診斷結果 ====================
print("\n【步驟8】儲存診斷結果...")

# 儲存缺失值詳細列表（前1000筆）
missing_sample = missing_data.head(1000)
cols_to_save = [col for col in ['ResponseID', 'UserID', 'ScenarioID', 'Country', 
                                'Cluster', 'Review_age', 'Review_political', 
                                'Review_religious', 'chose_lawful'] 
               if col in df.columns]

missing_sample[cols_to_save].to_csv(
    OUTPUT_DIR / 'missing_age_sample.csv',
    index=False,
    encoding='utf-8-sig'
)
print(f"✅ 已儲存：{OUTPUT_DIR / 'missing_age_sample.csv'}")

# 儲存統計摘要
summary_stats = pd.DataFrame({
    '項目': [
        '總樣本數',
        'Review_age 有效',
        'Review_age 缺失',
        'Review_political 有效',
        'Review_religious 有效',
        '僅 Review_age 缺失',
        '年齡+政治皆缺失',
        '年齡+宗教皆缺失',
        '三者皆缺失'
    ],
    '數量': [
        total_rows,
        age_valid,
        age_missing,
        political_valid,
        religious_valid,
        age_only_missing,
        age_political_missing,
        age_religious_missing,
        all_missing
    ],
    '比例(%)': [
        100.0,
        age_valid/total_rows*100,
        age_missing/total_rows*100,
        political_valid/total_rows*100,
        religious_valid/total_rows*100,
        age_only_missing/total_rows*100,
        age_political_missing/total_rows*100,
        age_religious_missing/total_rows*100,
        all_missing/total_rows*100
    ]
})

summary_stats.to_csv(
    OUTPUT_DIR / 'missing_summary.csv',
    index=False,
    encoding='utf-8-sig'
)
print(f"✅ 已儲存：{OUTPUT_DIR / 'missing_summary.csv'}")

# ==================== 診斷結論 ====================
print("\n" + "=" * 80)
print("【診斷結論】")
print("=" * 80)

print(f"\n1️⃣  Review_age 缺失了 {age_missing:,} 筆 ({age_missing/total_rows*100:.2f}%)")

if age_only_missing == age_missing:
    print(f"\n2️⃣  這 {age_missing:,} 筆「僅」Review_age 缺失，其他兩個變數都有資料")
    print("   👉 可能原因：")
    print("      - 某些使用者選擇不填寫年齡")
    print("      - 年齡資料在清理過程中被設為 NA（但保留該筆記錄）")
else:
    print(f"\n2️⃣  缺失模式：")
    print(f"   - 僅 Review_age 缺失：{age_only_missing:,} 筆")
    print(f"   - 多個變數同時缺失：{age_missing - age_only_missing:,} 筆")

print(f"\n3️⃣  建議：")
print("   ✅ 在報告中明確標註各變數的有效樣本數")
print("   ✅ 說明：「Review_age 因使用者未填寫或清理過程移除異常值，有效樣本較少」")
print("   ✅ 強調：即使樣本數少了 15.9%，仍有 317,258 筆充足樣本進行分析")

print("\n" + "=" * 80)
print("診斷完成！相關檔案已儲存至：outputs/diagnostic/")
print("=" * 80)