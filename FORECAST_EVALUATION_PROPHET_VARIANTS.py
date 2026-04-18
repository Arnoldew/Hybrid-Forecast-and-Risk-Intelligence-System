"""
FORECAST EVALUATION - PROPHET VARIANTS (WITH vs WITHOUT LOG)
================================================

Evaluasi Prophet model dengan 2 variant:
1. use_log=False (raw price, additive seasonality)
2. use_log=True (log-transformed price, multiplicative seasonality)

Untuk setiap horizon, tentukan variant mana yang lebih cocok.

Hasil output akan digunakan untuk:
- Determine best Prophet variant per horizon
- Update config untuk FORECAST_EVALUATION.py
"""

import pandas as pd
import sqlite3
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from config import DATABASE_PATH, EVAL_WINDOWS
from services.forecast_service import prepare_dataframe
from models.prophet_model import train_prophet, forecast_prophet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


# ============================================================================
# UTILITIES
# ============================================================================

def calculate_rmse(actual, predicted):
    """Calculate RMSE"""
    return np.sqrt(mean_squared_error(actual, predicted))


def calculate_mape(actual, predicted):
    """Calculate MAPE"""
    mask = actual != 0
    actual_filtered = actual[mask]
    predicted_filtered = predicted[mask]
    
    if len(actual_filtered) == 0:
        return np.nan
    
    return mean_absolute_percentage_error(actual_filtered, predicted_filtered) * 100


def load_data():
    """Load price data from database"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        df = pd.read_sql("SELECT date, price FROM price_data ORDER BY date", conn)
        conn.close()
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df.sort_index()
        
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


# ============================================================================
# EVALUATION PER HORIZON
# ============================================================================

def evaluate_prophet_variant_short_term(df):
    """
    SHORT TERM EVALUATION (30 days)
    Train: 2019-01-01 sd 2025-11-30
    Test: 2025-12-01 sd 2025-12-31
    """
    print("\n" + "="*80)
    print("SHORT TERM EVALUATION (30 days)")
    print("="*80)
    
    cfg = EVAL_WINDOWS["short"]
    
    print(f"\nTraining data: 2019-01-01 sd {cfg['train_end']} (11 months)")
    print(f"Test period:  {cfg['forecast_start']} sd {cfg['forecast_end']} (31 days)")
    
    # Prepare training data
    train_df = df.loc[:cfg['train_end']].copy()
    
    # Actual test data
    actual_start = pd.to_datetime(cfg['forecast_start'])
    actual_end = pd.to_datetime(cfg['forecast_end'])
    actual_data = df.loc[actual_start:actual_end, 'price'].values
    steps = len(actual_data)
    
    print(f"Steps to forecast: {steps}")
    
    results = {}
    
    # ---- Variant 1: WITHOUT log ----
    print(f"\n[Variant 1] Prophet WITHOUT log transform...")
    try:
        model_no_log, log_flag = train_prophet(train_df, use_log=False)
        forecast_no_log = forecast_prophet(model_no_log, steps, use_log=False)
        
        forecast_vals_no_log = forecast_no_log['yhat'].values
        forecast_vals_no_log = np.maximum(forecast_vals_no_log, 0)  # Clip lower=0
        
        rmse_no_log = calculate_rmse(actual_data, forecast_vals_no_log)
        mape_no_log = calculate_mape(actual_data, forecast_vals_no_log)
        
        print(f"  ✓ RMSE: {rmse_no_log:,.2f}")
        print(f"  ✓ MAPE: {mape_no_log:.2f}%")
        
        results['no_log'] = {
            'model': model_no_log,
            'forecast': forecast_no_log,
            'rmse': rmse_no_log,
            'mape': mape_no_log,
            'use_log': False
        }
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results['no_log'] = None
    
    # ---- Variant 2: WITH log ----
    print(f"\n[Variant 2] Prophet WITH log transform...")
    try:
        model_log, log_flag = train_prophet(train_df, use_log=True)
        forecast_log = forecast_prophet(model_log, steps, use_log=True)
        
        forecast_vals_log = forecast_log['yhat'].values
        forecast_vals_log = np.maximum(forecast_vals_log, 0)  # Clip lower=0
        
        rmse_log = calculate_rmse(actual_data, forecast_vals_log)
        mape_log = calculate_mape(actual_data, forecast_vals_log)
        
        print(f"  ✓ RMSE: {rmse_log:,.2f}")
        print(f"  ✓ MAPE: {mape_log:.2f}%")
        
        results['log'] = {
            'model': model_log,
            'forecast': forecast_log,
            'rmse': rmse_log,
            'mape': mape_log,
            'use_log': True
        }
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results['log'] = None
    
    # ---- COMPARISON ----
    print(f"\n--- WINNER ---")
    if results['no_log'] and results['log']:
        mape_no_log = results['no_log']['mape']
        mape_log = results['log']['mape']
        
        print(f"MAPE (without log): {mape_no_log:.2f}%")
        print(f"MAPE (with log):    {mape_log:.2f}%")
        
        if mape_no_log < mape_log:
            print(f"\n🏆 WINNER: WITHOUT log (MAPE {mape_no_log:.2f}% < {mape_log:.2f}%)")
            results['winner'] = 'no_log'
        else:
            print(f"\n🏆 WINNER: WITH log (MAPE {mape_log:.2f}% < {mape_no_log:.2f}%)")
            results['winner'] = 'log'
    else:
        results['winner'] = None
    
    return results


def evaluate_prophet_variant_mid_term(df):
    """
    MID TERM EVALUATION (180 days)
    Train: 2019-01-01 sd 2025-06-30
    Test: 2025-07-01 sd 2025-12-31
    """
    print("\n" + "="*80)
    print("MID TERM EVALUATION (180 days)")
    print("="*80)
    
    cfg = EVAL_WINDOWS["mid"]
    
    print(f"\nTraining data: 2019-01-01 sd {cfg['train_end']} (6+ months)")
    print(f"Test period:  {cfg['forecast_start']} sd {cfg['forecast_end']} (184 days)")
    
    # Prepare training data
    train_df = df.loc[:cfg['train_end']].copy()
    
    # Actual test data
    actual_start = pd.to_datetime(cfg['forecast_start'])
    actual_end = pd.to_datetime(cfg['forecast_end'])
    actual_data = df.loc[actual_start:actual_end, 'price'].values
    steps = len(actual_data)
    
    print(f"Steps to forecast: {steps}")
    
    results = {}
    
    # ---- Variant 1: WITHOUT log ----
    print(f"\n[Variant 1] Prophet WITHOUT log transform...")
    try:
        model_no_log, log_flag = train_prophet(train_df, use_log=False)
        forecast_no_log = forecast_prophet(model_no_log, steps, use_log=False)
        
        forecast_vals_no_log = forecast_no_log['yhat'].values
        forecast_vals_no_log = np.maximum(forecast_vals_no_log, 0)
        
        rmse_no_log = calculate_rmse(actual_data, forecast_vals_no_log)
        mape_no_log = calculate_mape(actual_data, forecast_vals_no_log)
        
        print(f"  ✓ RMSE: {rmse_no_log:,.2f}")
        print(f"  ✓ MAPE: {mape_no_log:.2f}%")
        
        results['no_log'] = {
            'model': model_no_log,
            'forecast': forecast_no_log,
            'rmse': rmse_no_log,
            'mape': mape_no_log,
            'use_log': False
        }
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results['no_log'] = None
    
    # ---- Variant 2: WITH log ----
    print(f"\n[Variant 2] Prophet WITH log transform...")
    try:
        model_log, log_flag = train_prophet(train_df, use_log=True)
        forecast_log = forecast_prophet(model_log, steps, use_log=True)
        
        forecast_vals_log = forecast_log['yhat'].values
        forecast_vals_log = np.maximum(forecast_vals_log, 0)
        
        rmse_log = calculate_rmse(actual_data, forecast_vals_log)
        mape_log = calculate_mape(actual_data, forecast_vals_log)
        
        print(f"  ✓ RMSE: {rmse_log:,.2f}")
        print(f"  ✓ MAPE: {mape_log:.2f}%")
        
        results['log'] = {
            'model': model_log,
            'forecast': forecast_log,
            'rmse': rmse_log,
            'mape': mape_log,
            'use_log': True
        }
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results['log'] = None
    
    # ---- COMPARISON ----
    print(f"\n--- WINNER ---")
    if results['no_log'] and results['log']:
        mape_no_log = results['no_log']['mape']
        mape_log = results['log']['mape']
        
        print(f"MAPE (without log): {mape_no_log:.2f}%")
        print(f"MAPE (with log):    {mape_log:.2f}%")
        
        if mape_no_log < mape_log:
            print(f"\n🏆 WINNER: WITHOUT log (MAPE {mape_no_log:.2f}% < {mape_log:.2f}%)")
            results['winner'] = 'no_log'
        else:
            print(f"\n🏆 WINNER: WITH log (MAPE {mape_log:.2f}% < {mape_no_log:.2f}%)")
            results['winner'] = 'log'
    else:
        results['winner'] = None
    
    return results


def evaluate_prophet_variant_long_term(df):
    """
    LONG TERM EVALUATION (365 days)
    Train: 2019-01-01 sd 2024-12-31
    Test: 2025-01-01 sd 2025-12-31
    """
    print("\n" + "="*80)
    print("LONG TERM EVALUATION (365 days)")
    print("="*80)
    
    cfg = EVAL_WINDOWS["long"]
    
    print(f"\nTraining data: 2019-01-01 sd {cfg['train_end']} (6 years)")
    print(f"Test period:  {cfg['forecast_start']} sd {cfg['forecast_end']} (365 days)")
    
    # Prepare training data
    train_df = df.loc[:cfg['train_end']].copy()
    
    # Actual test data
    actual_start = pd.to_datetime(cfg['forecast_start'])
    actual_end = pd.to_datetime(cfg['forecast_end'])
    actual_data = df.loc[actual_start:actual_end, 'price'].values
    steps = len(actual_data)
    
    print(f"Steps to forecast: {steps}")
    
    results = {}
    
    # ---- Variant 1: WITHOUT log ----
    print(f"\n[Variant 1] Prophet WITHOUT log transform...")
    try:
        model_no_log, log_flag = train_prophet(train_df, use_log=False)
        forecast_no_log = forecast_prophet(model_no_log, steps, use_log=False)
        
        forecast_vals_no_log = forecast_no_log['yhat'].values
        forecast_vals_no_log = np.maximum(forecast_vals_no_log, 0)
        
        rmse_no_log = calculate_rmse(actual_data, forecast_vals_no_log)
        mape_no_log = calculate_mape(actual_data, forecast_vals_no_log)
        
        print(f"  ✓ RMSE: {rmse_no_log:,.2f}")
        print(f"  ✓ MAPE: {mape_no_log:.2f}%")
        
        results['no_log'] = {
            'model': model_no_log,
            'forecast': forecast_no_log,
            'rmse': rmse_no_log,
            'mape': mape_no_log,
            'use_log': False
        }
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results['no_log'] = None
    
    # ---- Variant 2: WITH log ----
    print(f"\n[Variant 2] Prophet WITH log transform...")
    try:
        model_log, log_flag = train_prophet(train_df, use_log=True)
        forecast_log = forecast_prophet(model_log, steps, use_log=True)
        
        forecast_vals_log = forecast_log['yhat'].values
        forecast_vals_log = np.maximum(forecast_vals_log, 0)
        
        rmse_log = calculate_rmse(actual_data, forecast_vals_log)
        mape_log = calculate_mape(actual_data, forecast_vals_log)
        
        print(f"  ✓ RMSE: {rmse_log:,.2f}")
        print(f"  ✓ MAPE: {mape_log:.2f}%")
        
        results['log'] = {
            'model': model_log,
            'forecast': forecast_log,
            'rmse': rmse_log,
            'mape': mape_log,
            'use_log': True
        }
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results['log'] = None
    
    # ---- COMPARISON ----
    print(f"\n--- WINNER ---")
    if results['no_log'] and results['log']:
        mape_no_log = results['no_log']['mape']
        mape_log = results['log']['mape']
        
        print(f"MAPE (without log): {mape_no_log:.2f}%")
        print(f"MAPE (with log):    {mape_log:.2f}%")
        
        if mape_no_log < mape_log:
            print(f"\n🏆 WINNER: WITHOUT log (MAPE {mape_no_log:.2f}% < {mape_log:.2f}%)")
            results['winner'] = 'no_log'
        else:
            print(f"\n🏆 WINNER: WITH log (MAPE {mape_log:.2f}% < {mape_no_log:.2f}%)")
            results['winner'] = 'log'
    else:
        results['winner'] = None
    
    return results


# ============================================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================================

def print_final_summary(short_results, mid_results, long_results):
    """Print final summary and recommendations"""
    
    print("\n\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "FINAL SUMMARY: PROPHET VARIANT SELECTION".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    print("\n" + "-"*80)
    print("SHORT TERM (30 days)")
    print("-"*80)
    if short_results['no_log'] and short_results['log']:
        print(f"WITHOUT log:  RMSE={short_results['no_log']['rmse']:>10,.0f}  |  MAPE={short_results['no_log']['mape']:>6.2f}%")
        print(f"WITH log:     RMSE={short_results['log']['rmse']:>10,.0f}  |  MAPE={short_results['log']['mape']:>6.2f}%")
        print(f"\n✓ RECOMMENDED: use_log={short_results['winner']=='log'}")
    
    print("\n" + "-"*80)
    print("MID TERM (180 days)")
    print("-"*80)
    if mid_results['no_log'] and mid_results['log']:
        print(f"WITHOUT log:  RMSE={mid_results['no_log']['rmse']:>10,.0f}  |  MAPE={mid_results['no_log']['mape']:>6.2f}%")
        print(f"WITH log:     RMSE={mid_results['log']['rmse']:>10,.0f}  |  MAPE={mid_results['log']['mape']:>6.2f}%")
        print(f"\n✓ RECOMMENDED: use_log={mid_results['winner']=='log'}")
    
    print("\n" + "-"*80)
    print("LONG TERM (365 days)")
    print("-"*80)
    if long_results['no_log'] and long_results['log']:
        print(f"WITHOUT log:  RMSE={long_results['no_log']['rmse']:>10,.0f}  |  MAPE={long_results['no_log']['mape']:>6.2f}%")
        print(f"WITH log:     RMSE={long_results['log']['rmse']:>10,.0f}  |  MAPE={long_results['log']['mape']:>6.2f}%")
        print(f"\n✓ RECOMMENDED: use_log={long_results['winner']=='log'}")
    
    print("\n" + "="*80)
    print("NEXT STEP: Use results above to determine ARIMA vs PROPHET winner")
    print("="*80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "PROPHET VARIANT EVALUATION (WITH vs WITHOUT LOG)".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    print("\n")
    
    # Load data
    print("Loading data from database...")
    df = load_data()
    
    if df is None or df.empty:
        print("❌ Failed to load data. Exiting.")
        exit(1)
    
    print(f"✓ Data loaded: {df.index.min()} to {df.index.max()}")
    print(f"✓ Total records: {len(df)}\n")
    
    # Evaluate each horizon
    short_results = evaluate_prophet_variant_short_term(df)
    mid_results = evaluate_prophet_variant_mid_term(df)
    long_results = evaluate_prophet_variant_long_term(df)
    
    # Print summary
    print_final_summary(short_results, mid_results, long_results)