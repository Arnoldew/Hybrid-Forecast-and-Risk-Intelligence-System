"""
Model Evaluation Script
Menggabungkan dataset 2022-2026 dan mengevaluasi ARIMA vs Prophet untuk setiap horizon
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

# ============================================================================
# STEP 1: LOAD AND COMBINE ALL DATASET FILES
# ============================================================================

def load_and_combine_datasets():
    """Load all Excel files from dataset folder and combine them"""
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
        print(f"Loading: {file}")
        
        df = pd.read_excel(filepath)
        
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
        
        # Keep only date and price columns BEFORE appending
        df = df[["date", "price"]].copy()
        
        dfs.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sort_values("date").reset_index(drop=True)
    
    # Remove duplicates
    combined_df = combined_df.drop_duplicates(subset=["date"], keep="first")
    
    # Set date as index
    combined_df.set_index("date", inplace=True)
    combined_df = combined_df.sort_index()
    
    print(f"\nCombined dataset shape: {combined_df.shape}")
    print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    print(f"Total days: {len(combined_df)}")
    
    return combined_df

# ============================================================================
# STEP 2: EVALUATION METRICS
# ============================================================================

def calculate_rmse(actual, predicted):
    """Calculate RMSE"""
    return np.sqrt(mean_squared_error(actual, predicted))

def calculate_mape(actual, predicted):
    """Calculate MAPE"""
    # Filter out zero values to avoid division by zero
    mask = actual != 0
    actual_filtered = actual[mask]
    predicted_filtered = predicted[mask]
    
    if len(actual_filtered) == 0:
        return np.nan
    
    return mean_absolute_percentage_error(actual_filtered, predicted_filtered) * 100

# ============================================================================
# STEP 3: BACKTESTING FOR EACH HORIZON
# ============================================================================

def evaluate_short_term(df):
    """
    Short Term Evaluation
    Train: 1 Jan 2022 - 31 Des 2025
    Test: 1 Jan 2026 - 31 Jan 2026
    Forecast: 30 days
    """
    print("\n" + "="*80)
    print("SHORT TERM EVALUATION (30 days)")
    print("="*80)
    
    train_end = pd.Timestamp("2025-12-31")
    test_start = pd.Timestamp("2026-01-01")
    test_end = pd.Timestamp("2026-01-31")
    
    train_data = df[df.index <= train_end].copy()
    test_data = df[(df.index >= test_start) & (df.index <= test_end)].copy()
    
    print(f"Train data: {train_data.index.min()} to {train_data.index.max()} ({len(train_data)} days)")
    print(f"Test data: {test_data.index.min()} to {test_data.index.max()} ({len(test_data)} days)")
    
    actual_prices = test_data["price"].values
    
    results = {}
    
    # ---- ARIMA ----
    print("\n--- ARIMA Model ---")
    try:
        arima_model = train_arima(train_data)
        arima_forecast = forecast_arima(arima_model, 30)
        
        # Align forecast with test data
        arima_forecast_aligned = arima_forecast[:len(actual_prices)]
        
        arima_rmse = calculate_rmse(actual_prices, arima_forecast_aligned)
        arima_mape = calculate_mape(actual_prices, arima_forecast_aligned)
        
        print(f"RMSE: {arima_rmse:.2f}")
        print(f"MAPE: {arima_mape:.2f}%")
        
        results['arima'] = {
            'rmse': arima_rmse,
            'mape': arima_mape,
            'forecast': arima_forecast_aligned
        }
    except Exception as e:
        print(f"ARIMA failed: {e}")
        results['arima'] = None
    
    # ---- PROPHET ----
    print("\n--- PROPHET Model ---")
    try:
        prophet_train = train_data.reset_index()
        prophet_train.columns = ['ds', 'y']
        
        prophet_model = train_prophet(prophet_train)
        prophet_forecast_df = forecast_prophet(prophet_model, 30)
        prophet_forecast = prophet_forecast_df['yhat'].values
        
        # Align forecast with test data
        prophet_forecast_aligned = prophet_forecast[:len(actual_prices)]
        
        prophet_rmse = calculate_rmse(actual_prices, prophet_forecast_aligned)
        prophet_mape = calculate_mape(actual_prices, prophet_forecast_aligned)
        
        print(f"RMSE: {prophet_rmse:.2f}")
        print(f"MAPE: {prophet_mape:.2f}%")
        
        results['prophet'] = {
            'rmse': prophet_rmse,
            'mape': prophet_mape,
            'forecast': prophet_forecast_aligned
        }
    except Exception as e:
        print(f"PROPHET failed: {e}")
        results['prophet'] = None
    
    # ---- COMPARISON ----
    print("\n--- COMPARISON ---")
    if results['arima'] and results['prophet']:
        arima_rmse = results['arima']['rmse']
        prophet_rmse = results['prophet']['rmse']
        arima_mape = results['arima']['mape']
        prophet_mape = results['prophet']['mape']
        
        print(f"ARIMA  - RMSE: {arima_rmse:.2f}, MAPE: {arima_mape:.2f}%")
        print(f"PROPHET- RMSE: {prophet_rmse:.2f}, MAPE: {prophet_mape:.2f}%")
        
        if arima_rmse < prophet_rmse:
            print(f"✓ ARIMA is better (RMSE: {arima_rmse:.2f} < {prophet_rmse:.2f})")
            best_model = 'arima'
        else:
            print(f"✓ PROPHET is better (RMSE: {prophet_rmse:.2f} < {arima_rmse:.2f})")
            best_model = 'prophet'
    else:
        best_model = None
    
    return best_model, results

def evaluate_mid_term(df):
    """
    Mid Term Evaluation
    Train: 1 Jan 2022 - 30 Jul 2025
    Test: 1 Aug 2025 - 31 Jan 2026
    Forecast: 180 days
    """
    print("\n" + "="*80)
    print("MID TERM EVALUATION (180 days)")
    print("="*80)
    
    train_end = pd.Timestamp("2025-07-30")
    test_start = pd.Timestamp("2025-08-01")
    test_end = pd.Timestamp("2026-01-31")
    
    train_data = df[df.index <= train_end].copy()
    test_data = df[(df.index >= test_start) & (df.index <= test_end)].copy()
    
    print(f"Train data: {train_data.index.min()} to {train_data.index.max()} ({len(train_data)} days)")
    print(f"Test data: {test_data.index.min()} to {test_data.index.max()} ({len(test_data)} days)")
    
    actual_prices = test_data["price"].values
    
    results = {}
    
    # ---- ARIMA ----
    print("\n--- ARIMA Model ---")
    try:
        arima_model = train_arima(train_data)
        arima_forecast = forecast_arima(arima_model, 180)
        
        # Align forecast with test data
        arima_forecast_aligned = arima_forecast[:len(actual_prices)]
        
        arima_rmse = calculate_rmse(actual_prices, arima_forecast_aligned)
        arima_mape = calculate_mape(actual_prices, arima_forecast_aligned)
        
        print(f"RMSE: {arima_rmse:.2f}")
        print(f"MAPE: {arima_mape:.2f}%")
        
        results['arima'] = {
            'rmse': arima_rmse,
            'mape': arima_mape,
            'forecast': arima_forecast_aligned
        }
    except Exception as e:
        print(f"ARIMA failed: {e}")
        results['arima'] = None
    
    # ---- PROPHET ----
    print("\n--- PROPHET Model ---")
    try:
        prophet_train = train_data.reset_index()
        prophet_train.columns = ['ds', 'y']
        
        prophet_model = train_prophet(prophet_train)
        prophet_forecast_df = forecast_prophet(prophet_model, 180)
        prophet_forecast = prophet_forecast_df['yhat'].values
        
        # Align forecast with test data
        prophet_forecast_aligned = prophet_forecast[:len(actual_prices)]
        
        prophet_rmse = calculate_rmse(actual_prices, prophet_forecast_aligned)
        prophet_mape = calculate_mape(actual_prices, prophet_forecast_aligned)
        
        print(f"RMSE: {prophet_rmse:.2f}")
        print(f"MAPE: {prophet_mape:.2f}%")
        
        results['prophet'] = {
            'rmse': prophet_rmse,
            'mape': prophet_mape,
            'forecast': prophet_forecast_aligned
        }
    except Exception as e:
        print(f"PROPHET failed: {e}")
        results['prophet'] = None
    
    # ---- COMPARISON ----
    print("\n--- COMPARISON ---")
    if results['arima'] and results['prophet']:
        arima_rmse = results['arima']['rmse']
        prophet_rmse = results['prophet']['rmse']
        arima_mape = results['arima']['mape']
        prophet_mape = results['prophet']['mape']
        
        print(f"ARIMA  - RMSE: {arima_rmse:.2f}, MAPE: {arima_mape:.2f}%")
        print(f"PROPHET- RMSE: {prophet_rmse:.2f}, MAPE: {prophet_mape:.2f}%")
        
        if arima_rmse < prophet_rmse:
            print(f"✓ ARIMA is better (RMSE: {arima_rmse:.2f} < {prophet_rmse:.2f})")
            best_model = 'arima'
        else:
            print(f"✓ PROPHET is better (RMSE: {prophet_rmse:.2f} < {arima_rmse:.2f})")
            best_model = 'prophet'
    else:
        best_model = None
    
    return best_model, results

def evaluate_long_term(df):
    """
    Long Term Evaluation
    Train: 1 Jan 2022 - 31 Jan 2025
    Test: 1 Feb 2025 - 31 Jan 2026
    Forecast: 365 days
    """
    print("\n" + "="*80)
    print("LONG TERM EVALUATION (365 days)")
    print("="*80)
    
    train_end = pd.Timestamp("2025-01-31")
    test_start = pd.Timestamp("2025-02-01")
    test_end = pd.Timestamp("2026-01-31")
    
    train_data = df[df.index <= train_end].copy()
    test_data = df[(df.index >= test_start) & (df.index <= test_end)].copy()
    
    print(f"Train data: {train_data.index.min()} to {train_data.index.max()} ({len(train_data)} days)")
    print(f"Test data: {test_data.index.min()} to {test_data.index.max()} ({len(test_data)} days)")
    
    actual_prices = test_data["price"].values
    
    results = {}
    
    # ---- ARIMA ----
    print("\n--- ARIMA Model ---")
    try:
        arima_model = train_arima(train_data)
        arima_forecast = forecast_arima(arima_model, 365)
        
        # Align forecast with test data
        arima_forecast_aligned = arima_forecast[:len(actual_prices)]
        
        arima_rmse = calculate_rmse(actual_prices, arima_forecast_aligned)
        arima_mape = calculate_mape(actual_prices, arima_forecast_aligned)
        
        print(f"RMSE: {arima_rmse:.2f}")
        print(f"MAPE: {arima_mape:.2f}%")
        
        results['arima'] = {
            'rmse': arima_rmse,
            'mape': arima_mape,
            'forecast': arima_forecast_aligned
        }
    except Exception as e:
        print(f"ARIMA failed: {e}")
        results['arima'] = None
    
    # ---- PROPHET ----
    print("\n--- PROPHET Model ---")
    try:
        prophet_train = train_data.reset_index()
        prophet_train.columns = ['ds', 'y']
        
        prophet_model = train_prophet(prophet_train)
        prophet_forecast_df = forecast_prophet(prophet_model, 365)
        prophet_forecast = prophet_forecast_df['yhat'].values
        
        # Align forecast with test data
        prophet_forecast_aligned = prophet_forecast[:len(actual_prices)]
        
        prophet_rmse = calculate_rmse(actual_prices, prophet_forecast_aligned)
        prophet_mape = calculate_mape(actual_prices, prophet_forecast_aligned)
        
        print(f"RMSE: {prophet_rmse:.2f}")
        print(f"MAPE: {prophet_mape:.2f}%")
        
        results['prophet'] = {
            'rmse': prophet_rmse,
            'mape': prophet_mape,
            'forecast': prophet_forecast_aligned
        }
    except Exception as e:
        print(f"PROPHET failed: {e}")
        results['prophet'] = None
    
    # ---- COMPARISON ----
    print("\n--- COMPARISON ---")
    if results['arima'] and results['prophet']:
        arima_rmse = results['arima']['rmse']
        prophet_rmse = results['prophet']['rmse']
        arima_mape = results['arima']['mape']
        prophet_mape = results['prophet']['mape']
        
        print(f"ARIMA  - RMSE: {arima_rmse:.2f}, MAPE: {arima_mape:.2f}%")
        print(f"PROPHET- RMSE: {prophet_rmse:.2f}, MAPE: {prophet_mape:.2f}%")
        
        if arima_rmse < prophet_rmse:
            print(f"✓ ARIMA is better (RMSE: {arima_rmse:.2f} < {prophet_rmse:.2f})")
            best_model = 'arima'
        else:
            print(f"✓ PROPHET is better (RMSE: {prophet_rmse:.2f} < {arima_rmse:.2f})")
            best_model = 'prophet'
    else:
        best_model = None
    
    return best_model, results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("MODEL EVALUATION: ARIMA vs PROPHET")
    print("="*80)
    
    # Load and combine datasets
    df = load_and_combine_datasets()
    
    # Evaluate each horizon
    short_best, short_results = evaluate_short_term(df)
    mid_best, mid_results = evaluate_mid_term(df)
    long_best, long_results = evaluate_long_term(df)
    
    # Summary
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    print(f"Short Term (30 days): {short_best.upper() if short_best else 'UNKNOWN'}")
    print(f"Mid Term (180 days): {mid_best.upper() if mid_best else 'UNKNOWN'}")
    print(f"Long Term (365 days): {long_best.upper() if long_best else 'UNKNOWN'}")
    print("="*80)
