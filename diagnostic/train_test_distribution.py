import pandas as pd

# 檔案路徑
train_path = "data/processed/train_data.csv"
test_path = "data/processed/test_data.csv"

# 載入資料
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 定義一個函式用來顯示基本分布
def summary_info(df, name="dataset"):
    print(f"\n===== {name} 基本資訊 =====")
    print(f"資料筆數（rows）: {df.shape[0]}")
    print(f"欄位數（columns）: {df.shape[1]}")

    print("\n--- 欄位缺失值分布 ---")
    print(df.isnull().sum())

    print("\n--- 資料型態 ---")
    print(df.dtypes)

    print("\n--- 每個欄位唯一值數量（可用於檢查類別變數） ---")
    print(df.nunique())
    print("\n")

# 顯示 train / test 分布
summary_info(train_df, "Train Data")
summary_info(test_df, "Test Data")
