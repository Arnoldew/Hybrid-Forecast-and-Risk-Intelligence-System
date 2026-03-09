"""
Debug script untuk melihat struktur data
"""
import pandas as pd
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
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First 3 rows:\n{df.head(3)}")
    print(f"Last 3 rows:\n{df.tail(3)}")
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()
    df = df.rename(columns={
        "tanggal": "date",
        "harga": "price"
    })
    
    # Clean price data
    df["price"] = df["price"].astype(str)
    df["price"] = df["price"].str.replace(".", "", regex=False)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    
    # Convert date
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    
    # Remove NaN and invalid prices
    df = df.dropna(subset=["date", "price"])
    df = df[df["price"] > 0]
    
    print(f"After cleaning - Shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    dfs.append(df)

# Combine all dataframes
combined_df = pd.concat(dfs, ignore_index=True)
combined_df = combined_df.sort_values("date").reset_index(drop=True)

# Remove duplicates
combined_df = combined_df.drop_duplicates(subset=["date"], keep="first")

# Set date as index
combined_df.set_index("date", inplace=True)
combined_df = combined_df.sort_index()

print(f"\n{'='*60}")
print("COMBINED DATASET")
print(f"{'='*60}")
print(f"Shape: {combined_df.shape}")
print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
print(f"Total days: {len(combined_df)}")
print(f"\nFirst 5 rows:\n{combined_df.head()}")
print(f"\nLast 5 rows:\n{combined_df.tail()}")

# Check for gaps
print(f"\n{'='*60}")
print("CHECKING FOR GAPS")
print(f"{'='*60}")
date_range = pd.date_range(combined_df.index.min(), combined_df.index.max(), freq='D')
missing_dates = date_range.difference(combined_df.index)
print(f"Total expected days: {len(date_range)}")
print(f"Missing days: {len(missing_dates)}")
if len(missing_dates) > 0:
    print(f"First 10 missing dates: {missing_dates[:10].tolist()}")
