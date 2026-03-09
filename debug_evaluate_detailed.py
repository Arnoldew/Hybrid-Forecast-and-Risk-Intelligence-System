"""
Debug script untuk melihat error di evaluate_models.py
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from models.arima_model import train_arima, forecast_arima
from models.prophet_model import train_prophet, forecast_prophet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

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
print("SHORT TERM EVALUATION - DETAILED DEBUG")
print("="*80)

train_end = pd.Timestamp("2025-12-31")
test_start = pd.Timestamp("2026-01-01")
test_end = pd.Timestamp("2026-01-31")

train_data = combined_df[combined_df.index <= train_end].copy()
test_data = combined_df[(combined_df.index >= test_start) & (combined_df.index <= test_end)].copy()

print(f"\nTrain data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

actual_prices = test_data["price"].values
print(f"Actual prices shape: {actual_prices.shape}")
print(f"Actual prices: {actual_prices}")

# ---- ARIMA ----
print("\n" + "="*80)
print("ARIMA DEBUG")
print("="*80)
try:
    print("Step 1: Training ARIMA...")
    arima_model = train_arima(train_data)
    print("✓ ARIMA training successful")
    
    print("Step 2: Forecasting 30 days...")
    arima_forecast = forecast_arima(arima_model, 30)
    print(f"✓ ARIMA forecast shape: {arima_forecast.shape}")
    print(f"  Forecast values: {arima_forecast}")
    
    print("Step 3: Aligning forecast with test data...")
    arima_forecast_aligned = arima_forecast[:len(actual_prices)]
    print(f"✓ Aligned forecast shape: {arima_forecast_aligned.shape}")
    print(f"  Aligned forecast: {arima_forecast_aligned}")
    
    print("Step 4: Calculating RMSE...")
    arima_rmse = np.sqrt(mean_squared_error(actual_prices, arima_forecast_aligned))
    print(f"✓ ARIMA RMSE: {arima_rmse:.2f}")
    
except Exception as e:
    print(f"✗ ARIMA failed: {e}")
    import traceback
    traceback.print_exc()

# ---- PROPHET ----
print("\n" + "="*80)
print("PROPHET DEBUG")
print("="*80)
try:
    print("Step 1: Preparing data for Prophet...")
    prophet_train = train_data.reset_index()
    print(f"  After reset_index - shape: {prophet_train.shape}, columns: {prophet_train.columns.tolist()}")
    
    prophet_train.columns = ['ds', 'y']
    print(f"  After rename - shape: {prophet_train.shape}, columns: {prophet_train.columns.tolist()}")
    print(f"  First 5 rows:\n{prophet_train.head()}")
    
    print("Step 2: Training Prophet...")
    prophet_model = train_prophet(prophet_train)
    print("✓ Prophet training successful")
    
    print("Step 3: Forecasting 30 days...")
    prophet_forecast_df = forecast_prophet(prophet_model, 30)
    print(f"✓ Prophet forecast shape: {prophet_forecast_df.shape}")
    print(f"  Forecast columns: {prophet_forecast_df.columns.tolist()}")
    print(f"  First 5 rows:\n{prophet_forecast_df.head()}")
    
    prophet_forecast = prophet_forecast_df['yhat'].values
    print(f"  yhat values shape: {prophet_forecast.shape}")
    print(f"  yhat values: {prophet_forecast}")
    
    print("Step 4: Aligning forecast with test data...")
    prophet_forecast_aligned = prophet_forecast[:len(actual_prices)]
    print(f"✓ Aligned forecast shape: {prophet_forecast_aligned.shape}")
    print(f"  Aligned forecast: {prophet_forecast_aligned}")
    
    print("Step 5: Calculating RMSE...")
    prophet_rmse = np.sqrt(mean_squared_error(actual_prices, prophet_forecast_aligned))
    print(f"✓ Prophet RMSE: {prophet_rmse:.2f}")
    
except Exception as e:
    print(f"✗ Prophet failed: {e}")
    import traceback
    traceback.print_exc()
