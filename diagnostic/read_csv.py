"""
診斷程式碼：檢查 is_lawful 與 is_majority 的數據問題
"""

import pandas as pd
import numpy as np
from scipy import stats

# ============================================================
# 1. 載入資料
# ============================================================
print("=" * 60)
print("1. 載入資料")
print("=" * 60)

df = pd.read_csv('data/processed/featured_data.csv')
print(f"原始資料筆數: {len(df):,}")

# 過濾 Cluster != -1
df_filtered = df[df['Cluster'] != -1].copy()
print(f"過濾後資料筆數: {len(df_filtered):,}")

# ============================================================
# 2. 檢查關鍵變數是否存在
# ============================================================
print("\n" + "=" * 60)
print("2. 檢查關鍵變數")
print("=" * 60)

key_vars = ['chose_lawful', 'is_lawful', 'is_majority', 'lawful_vs_majority_conflict']
for var in key_vars:
    if var in df_filtered.columns:
        print(f"✅ {var} 存在")
    else:
        print(f"❌ {var} 不存在！")

# ============================================================
# 3. 變數分佈檢查
# ============================================================
print("\n" + "=" * 60)
print("3. 變數分佈檢查")
print("=" * 60)

for var in key_vars:
    if var in df_filtered.columns:
        print(f"\n【{var}】")
        print(df_filtered[var].value_counts().sort_index())
        print(f"平均值: {df_filtered[var].mean():.4f}")

# ============================================================
# 4. 檢查 is_lawful 與 is_majority 的關係
# ============================================================
print("\n" + "=" * 60)
print("4. is_lawful 與 is_majority 的關係")
print("=" * 60)

# 交叉表
cross_tab = pd.crosstab(df_filtered['is_lawful'], df_filtered['is_majority'])
print("\n交叉表 (is_lawful × is_majority):")
print(cross_tab)

# 相關係數
corr = df_filtered['is_lawful'].corr(df_filtered['is_majority'])
print(f"\n皮爾森相關係數: r = {corr:.4f}")

# 完全相同比例
same_ratio = (df_filtered['is_lawful'] == df_filtered['is_majority']).mean()
print(f"兩變數相同的比例: {same_ratio:.2%}")

# ============================================================
# 5. 分組統計：chose_lawful 依 is_lawful 分組
# ============================================================
print("\n" + "=" * 60)
print("5. chose_lawful 依 is_lawful 分組")
print("=" * 60)

group_by_lawful = df_filtered.groupby('is_lawful')['chose_lawful'].agg(['count', 'mean', 'std'])
print(group_by_lawful)

# ============================================================
# 6. 分組統計：chose_lawful 依 is_majority 分組
# ============================================================
print("\n" + "=" * 60)
print("6. chose_lawful 依 is_majority 分組")
print("=" * 60)

group_by_majority = df_filtered.groupby('is_majority')['chose_lawful'].agg(['count', 'mean', 'std'])
print(group_by_majority)

# ============================================================
# 7. 手動執行 t 檢定驗證
# ============================================================
print("\n" + "=" * 60)
print("7. 手動 t 檢定驗證")
print("=" * 60)

# is_lawful 的 t 檢定
group0_lawful = df_filtered[df_filtered['is_lawful'] == 0]['chose_lawful']
group1_lawful = df_filtered[df_filtered['is_lawful'] == 1]['chose_lawful']

t_lawful, p_lawful = stats.ttest_ind(group0_lawful, group1_lawful, equal_var=False)
print(f"\n【is_lawful t檢定】")
print(f"  Group 0 (違法側): n={len(group0_lawful):,}, mean={group0_lawful.mean():.4f}")
print(f"  Group 1 (守法側): n={len(group1_lawful):,}, mean={group1_lawful.mean():.4f}")
print(f"  t = {t_lawful:.3f}, p = {p_lawful:.2e}")

# is_majority 的 t 檢定
group0_majority = df_filtered[df_filtered['is_majority'] == 0]['chose_lawful']
group1_majority = df_filtered[df_filtered['is_majority'] == 1]['chose_lawful']

t_majority, p_majority = stats.ttest_ind(group0_majority, group1_majority, equal_var=False)
print(f"\n【is_majority t檢定】")
print(f"  Group 0 (少數側): n={len(group0_majority):,}, mean={group0_majority.mean():.4f}")
print(f"  Group 1 (多數側): n={len(group1_majority):,}, mean={group1_majority.mean():.4f}")
print(f"  t = {t_majority:.3f}, p = {p_majority:.2e}")

# ============================================================
# 8. Cohen's d 計算
# ============================================================
print("\n" + "=" * 60)
print("8. Cohen's d 效果量")
print("=" * 60)

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (g1.mean() - g2.mean()) / pooled_std

d_lawful = cohens_d(group0_lawful, group1_lawful)
d_majority = cohens_d(group0_majority, group1_majority)

print(f"is_lawful 的 Cohen's d: {d_lawful:.4f}")
print(f"is_majority 的 Cohen's d: {d_majority:.4f}")

# ============================================================
# 9. 全局守法選擇率
# ============================================================
print("\n" + "=" * 60)
print("9. 全局守法選擇率")
print("=" * 60)

global_rate = df_filtered['chose_lawful'].mean()
print(f"全局 chose_lawful 平均值: {global_rate:.4f} ({global_rate*100:.2f}%)")

# ============================================================
# 10. 檢查資料結構問題
# ============================================================
print("\n" + "=" * 60)
print("10. 資料結構檢查")
print("=" * 60)

# 檢查是否每個 ResponseID 都有兩筆資料
response_counts = df_filtered.groupby('ResponseID').size()
print(f"每個 ResponseID 的資料筆數分佈:")
print(response_counts.value_counts().sort_index())

# 檢查同一 ResponseID 中 is_lawful 的分佈
print(f"\n同一場景中 is_lawful 的總和分佈:")
lawful_sum_per_response = df_filtered.groupby('ResponseID')['is_lawful'].sum()
print(lawful_sum_per_response.value_counts().sort_index())

print("\n" + "=" * 60)
print("診斷完成！")
print("=" * 60)