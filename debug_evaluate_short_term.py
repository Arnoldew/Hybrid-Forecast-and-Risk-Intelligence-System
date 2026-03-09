"""
Debug script untuk melihat apa yang terjadi di dalam evaluate_short_term
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load dataset dengan cara yang sama seperti evaluate_models.py
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
    df = df.rename(columns={
        "tanggal": "date",
        "harga": "price"
    })
    df["price"] = df["price"].astype(str)
    df["price"] = df["price"].str.replace(".", "", regex=False)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.dropna(subset=["date", "price"])
    df = df[df["price"] > 0]
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
combined_df = combined_df.sort_values("date").reset_index(drop=True)
combined_df = combined_df.drop_duplicates(subset=["date"], keep="first")
combined_df = combined_df[["date", "price"]].copy()
combined_df.set_index("date", inplace=True)
combined_df = combined_df.sort_index()

print("="*80)
print("TESTING EVALUATE_SHORT_TERM LOGIC")
print("="*80)

train_end = pd.Timestamp("2025-12-31")
test_start = pd.Timestamp("2026-01-01")
test_end = pd.Timestamp("2026-01-31")

train_data = combined_df[combined_df.index <= train_end].copy()
test_data = combined_df[(combined_df.index >= test_start) & (combined_df.index <= test_end)].copy()

print(f"\nTrain data:")
print(f"  Shape: {train_data.shape}")
print(f"  Columns: {train_data.columns.tolist()}")
print(f"  Index: {train_data.index}")
print(f"  First 3 rows:\n{train_data.head(3)}")

print(f"\nTest data:")
print(f"  Shape: {test_data.shape}")
print(f"  Columns: {test_data.columns.tolist()}")
print(f"  First 3 rows:\n{test_data.head(3)}")

# Test ARIMA
print(f"\n{'='*80}")
print("TESTING ARIMA")
print(f"{'='*80}")

try:
    from models.arima_model import train_arima, forecast_arima
    
    print(f"\nStep 1: Calling train_arima(train_data)")
    print(f"  train_data type: {type(train_data)}")
    print(f"  train_data shape: {train_data.shape}")
    print(f"  train_data columns: {train_data.columns.tolist()}")
    
    arima_model = train_arima(train_data)
    print(f"✓ ARIMA training successful")
    
    print(f"\nStep 2: Calling forecast_arima(arima_model, 30)")
    arima_forecast = forecast_arima(arima_model, 30)
    print(f"✓ ARIMA forecast successful")
    print(f"  Forecast type: {type(arima_forecast)}")
    print(f"  Forecast shape: {arima_forecast.shape}")
    
except Exception as e:
    print(f"✗ ARIMA failed: {e}")
    import traceback
    traceback.print_exc()

# Test PROPHET
print(f"\n{'='*80}")
print("TESTING PROPHET")
print(f"{'='*80}")

try:
    from models.prophet_model import train_prophet, forecast_prophet
    
    print(f"\nStep 1: Preparing data for Prophet")
    prophet_train = train_data.reset_index()
    print(f"  After reset_index:")
    print(f"    Shape: {prophet_train.shape}")
    print(f"    Columns: {prophet_train.columns.tolist()}")
    print(f"    First 3 rows:\n{prophet_train.head(3)}")
    
    print(f"\nStep 2: Renaming columns")
    prophet_train.columns = ['ds', 'y']
    print(f"  After rename:")
    print(f"    Columns: {prophet_train.columns.tolist()}")
    print(f"    First 3 rows:\n{prophet_train.head(3)}")
    
    print(f"\nStep 3: Calling train_prophet(prophet_train)")
    prophet_model = train_prophet(prophet_train)
    print(f"✓ Prophet training successful")
    
    print(f"\nStep 4: Calling forecast_prophet(prophet_model, 30)")
    prophet_forecast_df = forecast_prophet(prophet_model, 30)
    print(f"✓ Prophet forecast successful")
    print(f"  Forecast type: {type(prophet_forecast_df)}")
    print(f"  Forecast shape: {prophet_forecast_df.shape}")
    print(f"  Forecast columns: {prophet_forecast_df.columns.tolist()}")
    
    prophet_forecast = prophet_forecast_df['yhat'].values
    print(f"  yhat values shape: {prophet_forecast.shape}")
    
except Exception as e:
    print(f"✗ Prophet failed: {e}")
    import traceback
    traceback.print_exc()
