"""
Debug script untuk melihat struktur data yang dikombinasikan
"""
import pandas as pd
import numpy as np
import os

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
    print(f"\n{'='*60}")
    print(f"File: {file}")
    print(f"{'='*60}")
    
    df = pd.read_excel(filepath)
    print(f"Original columns: {df.columns.tolist()}")
    print(f"Original shape: {df.shape}")
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()
    print(f"After lowercase: {df.columns.tolist()}")
    
    df = df.rename(columns={
        "tanggal": "date",
        "harga": "price"
    })
    print(f"After rename tanggal/harga: {df.columns.tolist()}")
    
    # Clean price data
    df["price"] = df["price"].astype(str)
    df["price"] = df["price"].str.replace(".", "", regex=False)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    
    # Convert date
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    
    # Remove NaN and invalid prices
    df = df.dropna(subset=["date", "price"])
    df = df[df["price"] > 0]
    
    print(f"After cleaning - shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First 3 rows:\n{df.head(3)}")
    
    dfs.append(df)

# Combine all dataframes
print(f"\n{'='*60}")
print("COMBINING DATAFRAMES")
print(f"{'='*60}")

combined_df = pd.concat(dfs, ignore_index=True)
print(f"After concat - shape: {combined_df.shape}")
print(f"Columns: {combined_df.columns.tolist()}")
print(f"First 5 rows:\n{combined_df.head()}")

combined_df = combined_df.sort_values("date").reset_index(drop=True)
print(f"\nAfter sort - shape: {combined_df.shape}")

# Remove duplicates
combined_df = combined_df.drop_duplicates(subset=["date"], keep="first")
print(f"After drop_duplicates - shape: {combined_df.shape}")

# Keep only date and price columns
print(f"\nBefore selecting columns - shape: {combined_df.shape}, columns: {combined_df.columns.tolist()}")
combined_df = combined_df[["date", "price"]].copy()
print(f"After selecting columns - shape: {combined_df.shape}, columns: {combined_df.columns.tolist()}")

# Set date as index
combined_df.set_index("date", inplace=True)
combined_df = combined_df.sort_index()

print(f"\nFinal combined dataset:")
print(f"Shape: {combined_df.shape}")
print(f"Columns: {combined_df.columns.tolist()}")
print(f"Index type: {type(combined_df.index)}")
print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
print(f"\nFirst 5 rows:\n{combined_df.head()}")
print(f"\nLast 5 rows:\n{combined_df.tail()}")
print(f"\nData types:\n{combined_df.dtypes}")
