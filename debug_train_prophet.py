"""
Debug script untuk melihat apa yang terjadi di train_prophet
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
print("TESTING TRAIN_PROPHET LOGIC")
print("="*80)

# Test short term
train_end = pd.Timestamp("2025-12-31")
train_data = combined_df[combined_df.index <= train_end].copy()

print(f"\nOriginal train_data shape: {train_data.shape}")
print(f"Original train_data index: {train_data.index}")
print(f"Original train_data columns: {train_data.columns.tolist()}")
print(f"First 5 rows:\n{train_data.head()}")

# Simulate what evaluate_short_term does
print(f"\n--- Step 1: reset_index() ---")
prophet_train = train_data.reset_index()
print(f"After reset_index shape: {prophet_train.shape}")
print(f"After reset_index columns: {prophet_train.columns.tolist()}")
print(f"First 5 rows:\n{prophet_train.head()}")

print(f"\n--- Step 2: rename columns ---")
prophet_train.columns = ['ds', 'y']
print(f"After rename columns: {prophet_train.columns.tolist()}")
print(f"Shape: {prophet_train.shape}")
print(f"First 5 rows:\n{prophet_train.head()}")

# Try to train prophet
print(f"\n--- Step 3: train prophet ---")
try:
    from models.prophet_model import train_prophet
    prophet_model = train_prophet(prophet_train)
    print("✓ Prophet training successful!")
except Exception as e:
    print(f"✗ Prophet training failed: {e}")
    import traceback
    traceback.print_exc()
