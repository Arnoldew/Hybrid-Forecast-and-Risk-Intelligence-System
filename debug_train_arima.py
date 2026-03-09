"""
Debug script untuk melihat apa yang terjadi di train_arima
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Load dataset
dataset_folder = "dataset"
files = [
    "1 Jan 2022 - 31 Dec 2022.xlsx",
    "1 Jan 2023 - 31 Dec 2023.xlsx",
    "1 Jan 2024 - 31 Dec 2024.xlsx",
    "1 Jan 2025 - 31 Dec 2025.xlsx",
    "1 Januari 2026 - 31 Januari 2026 (1).xlsx"
]

dfs = []
for file in files:
    filepath = os.path.join(dataset_folder, file)
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.lower().str.strip()
    
    # Handle different column names
    if "tanggal" in df.columns:
        df = df.rename(columns={"tanggal": "date"})
    if "date" not in df.columns:
        date_cols = [col for col in df.columns if "date" in col.lower()]
        if date_cols:
            df = df.rename(columns={date_cols[0]: "date"})
    
    if "harga" in df.columns:
        df = df.rename(columns={"harga": "price"})
    if "price" not in df.columns:
        price_cols = [col for col in df.columns if "price" in col.lower() or "harga" in col.lower()]
        if price_cols:
            df = df.rename(columns={price_cols[0]: "price"})
    
    df = df[["date", "price"]].copy()
    df["price"] = df["price"].astype(str).str.strip()
    df["price"] = df["price"].str.replace(".", "", regex=False)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date", "price"])
    df = df[df["price"] > 0]
    
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
combined_df = combined_df.sort_values("date").reset_index(drop=True)
combined_df = combined_df.drop_duplicates(subset=["date"], keep="first")
combined_df.set_index("date", inplace=True)
combined_df = combined_df.sort_index()

print("="*80)
print("TESTING TRAIN_ARIMA LOGIC")
print("="*80)

# Test short term
train_end = pd.Timestamp("2025-12-31")
train_data = combined_df[combined_df.index <= train_end].copy()

print(f"\nOriginal train_data shape: {train_data.shape}")
print(f"Original train_data index type: {type(train_data.index)}")
print(f"Original train_data columns: {train_data.columns.tolist()}")
print(f"First 5 rows:\n{train_data.head()}")
print(f"Last 5 rows:\n{train_data.tail()}")

# Simulate what train_arima does
train = train_data.copy()
print(f"\n--- Step 1: asfreq('D') ---")
print(f"Before asfreq: shape={train.shape}")
train = train.asfreq('D')
print(f"After asfreq: shape={train.shape}")
print(f"NaN count: {train['price'].isna().sum()}")

print(f"\n--- Step 2: replace 0 with NaN ---")
train['price'] = train['price'].replace(0, np.nan)
print(f"NaN count after replace: {train['price'].isna().sum()}")

print(f"\n--- Step 3: interpolate ---")
train['price'] = train['price'].interpolate(method='linear')
print(f"NaN count after interpolate: {train['price'].isna().sum()}")

print(f"\n--- Step 4: dropna ---")
train = train.dropna()
print(f"Shape after dropna: {train.shape}")
print(f"Remaining data:\n{train.head()}\n{train.tail()}")

if len(train) > 0:
    print(f"\n--- Step 5: log transformation ---")
    train['log_price'] = np.log(train['price'])
    print(f"Log price shape: {train['log_price'].shape}")
    print(f"Log price values:\n{train['log_price'].head()}")
else:
    print("\n!!! ERROR: No data left after dropna() !!!")
