"""
COMPREHENSIVE FORECAST EVALUATION
===================================

Compare ARIMA (dengan log transform) vs PROPHET (tanpa log transform)
untuk ketiga horizon: short, mid, long.

Hasil akhir: Tentukan model terbaik per horizon.
"""

import pandas as pd
import sqlite3
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from config import DATABASE_PATH, EVAL_WINDOWS
from services.forecast_service import prepare_dataframe
from models.arima_model import train_arima, forecast_arima
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
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_short_term(df):
    """
    SHORT TERM EVALUATION (30 days)
    Train: 2019-01-01 sd 2025-11-30
    Test: 2025-12-01 sd 2025-12-31
    """
    print("\n" + "="*80)
    print("SHORT TERM EVALUATION (30 days)")
    print("="*80)
    
    cfg = EVAL_WINDOWS["short"]
    
    print(f"Training data: 2019-01-01 sd {cfg['train_end']} (11 months)")
    print(f"Test period:  {cfg['forecast_start']} sd {cfg['forecast_end']} (31 days)")
    
    # Prepare training data
    train_df = df.loc[:cfg['train_end']].copy()
    
    # Actual test data
    actual_start = pd.to_datetime(cfg['forecast_start'])
    actual_end = pd.to_datetime(cfg['forecast_end'])
    actual_data = df.loc[actual_start:actual_end, 'price'].values
    steps = len(actual_data)
    
    print(f"Steps to forecast: {steps}\n")
    
    results = {}
    
    # ---- ARIMA (with log transform) ----
    print("[1] ARIMA Model (with log transform)...")
    try:
        model_arima = train_arima(train_df)
        forecast_arima_vals = forecast_arima(model_arima, steps)
        forecast_arima_vals = np.maximum(forecast_arima_vals, 0)  # Clip lower=0
        
        rmse_arima = calculate_rmse(actual_data, forecast_arima_vals)
        mape_arima = calculate_mape(actual_data, forecast_arima_vals)
        
        print(f"    ✓ RMSE: {rmse_arima:>10,.2f}")
        print(f"    ✓ MAPE: {mape_arima:>10.2f}%")
        
        results['arima'] = {
            'model': 'ARIMA',
            'rmse': rmse_arima,
            'mape': mape_arima,
            'forecast': forecast_arima_vals
        }
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        results['arima'] = None
    
    # ---- PROPHET (without log transform) ----
    print("\n[2] PROPHET Model (without log transform)...")
    try:
        prophet_train = train_df.reset_index()
        prophet_train.columns = ['ds', 'y']
        
        model_prophet, _ = train_prophet(prophet_train, use_log=False)
        forecast_prophet_df = forecast_prophet(model_prophet, steps, use_log=False)
        forecast_prophet_vals = forecast_prophet_df['yhat'].values
        forecast_prophet_vals = np.maximum(forecast_prophet_vals, 0)
        
        rmse_prophet = calculate_rmse(actual_data, forecast_prophet_vals)
        mape_prophet = calculate_mape(actual_data, forecast_prophet_vals)
        
        print(f"    ✓ RMSE: {rmse_prophet:>10,.2f}")
        print(f"    ✓ MAPE: {mape_prophet:>10.2f}%")
        
        results['prophet'] = {
            'model': 'PROPHET',
            'rmse': rmse_prophet,
            'mape': mape_prophet,
            'forecast': forecast_prophet_vals
        }
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        results['prophet'] = None
    
    # ---- COMPARISON ----
    print("\n" + "-"*80)
    print("COMPARISON & WINNER")
    print("-"*80)
    
    if results['arima'] and results['prophet']:
        mape_arima = results['arima']['mape']
        mape_prophet = results['prophet']['mape']
        rmse_arima = results['arima']['rmse']
        rmse_prophet = results['prophet']['rmse']
        
        print(f"ARIMA   - RMSE: {rmse_arima:>10,.2f}  |  MAPE: {mape_arima:>6.2f}%")
        print(f"PROPHET - RMSE: {rmse_prophet:>10,.2f}  |  MAPE: {mape_prophet:>6.2f}%")
        
        if mape_arima < mape_prophet:
            print(f"\n🏆 WINNER: ARIMA (MAPE {mape_arima:.2f}% < {mape_prophet:.2f}%)")
            results['winner'] = 'ARIMA'
        else:
            print(f"\n🏆 WINNER: PROPHET (MAPE {mape_prophet:.2f}% < {mape_arima:.2f}%)")
            results['winner'] = 'PROPHET'
    else:
        results['winner'] = None
    
    return results


def evaluate_mid_term(df):
    """
    MID TERM EVALUATION (180 days)
    Train: 2019-01-01 sd 2025-06-30
    Test: 2025-07-01 sd 2025-12-31
    """
    print("\n" + "="*80)
    print("MID TERM EVALUATION (180 days)")
    print("="*80)
    
    cfg = EVAL_WINDOWS["mid"]
    
    print(f"Training data: 2019-01-01 sd {cfg['train_end']} (6+ months)")
    print(f"Test period:  {cfg['forecast_start']} sd {cfg['forecast_end']} (184 days)")
    
    # Prepare training data
    train_df = df.loc[:cfg['train_end']].copy()
    
    # Actual test data
    actual_start = pd.to_datetime(cfg['forecast_start'])
    actual_end = pd.to_datetime(cfg['forecast_end'])
    actual_data = df.loc[actual_start:actual_end, 'price'].values
    steps = len(actual_data)
    
    print(f"Steps to forecast: {steps}\n")
    
    results = {}
    
    # ---- ARIMA (with log transform) ----
    print("[1] ARIMA Model (with log transform)...")
    try:
        model_arima = train_arima(train_df)
        forecast_arima_vals = forecast_arima(model_arima, steps)
        forecast_arima_vals = np.maximum(forecast_arima_vals, 0)
        
        rmse_arima = calculate_rmse(actual_data, forecast_arima_vals)
        mape_arima = calculate_mape(actual_data, forecast_arima_vals)
        
        print(f"    ✓ RMSE: {rmse_arima:>10,.2f}")
        print(f"    ✓ MAPE: {mape_arima:>10.2f}%")
        
        results['arima'] = {
            'model': 'ARIMA',
            'rmse': rmse_arima,
            'mape': mape_arima,
            'forecast': forecast_arima_vals
        }
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        results['arima'] = None
    
    # ---- PROPHET (without log transform) ----
    print("\n[2] PROPHET Model (without log transform)...")
    try:
        prophet_train = train_df.reset_index()
        prophet_train.columns = ['ds', 'y']
        
        model_prophet, _ = train_prophet(prophet_train, use_log=False)
        forecast_prophet_df = forecast_prophet(model_prophet, steps, use_log=False)
        forecast_prophet_vals = forecast_prophet_df['yhat'].values
        forecast_prophet_vals = np.maximum(forecast_prophet_vals, 0)
        
        rmse_prophet = calculate_rmse(actual_data, forecast_prophet_vals)
        mape_prophet = calculate_mape(actual_data, forecast_prophet_vals)
        
        print(f"    ✓ RMSE: {rmse_prophet:>10,.2f}")
        print(f"    ✓ MAPE: {mape_prophet:>10.2f}%")
        
        results['prophet'] = {
            'model': 'PROPHET',
            'rmse': rmse_prophet,
            'mape': mape_prophet,
            'forecast': forecast_prophet_vals
        }
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        results['prophet'] = None
    
    # ---- COMPARISON ----
    print("\n" + "-"*80)
    print("COMPARISON & WINNER")
    print("-"*80)
    
    if results['arima'] and results['prophet']:
        mape_arima = results['arima']['mape']
        mape_prophet = results['prophet']['mape']
        rmse_arima = results['arima']['rmse']
        rmse_prophet = results['prophet']['rmse']
        
        print(f"ARIMA   - RMSE: {rmse_arima:>10,.2f}  |  MAPE: {mape_arima:>6.2f}%")
        print(f"PROPHET - RMSE: {rmse_prophet:>10,.2f}  |  MAPE: {mape_prophet:>6.2f}%")
        
        if mape_arima < mape_prophet:
            print(f"\n🏆 WINNER: ARIMA (MAPE {mape_arima:.2f}% < {mape_prophet:.2f}%)")
            results['winner'] = 'ARIMA'
        else:
            print(f"\n🏆 WINNER: PROPHET (MAPE {mape_prophet:.2f}% < {mape_arima:.2f}%)")
            results['winner'] = 'PROPHET'
    else:
        results['winner'] = None
    
    return results


def evaluate_long_term(df):
    """
    LONG TERM EVALUATION (365 days)
    Train: 2019-01-01 sd 2024-12-31
    Test: 2025-01-01 sd 2025-12-31
    """
    print("\n" + "="*80)
    print("LONG TERM EVALUATION (365 days)")
    print("="*80)
    
    cfg = EVAL_WINDOWS["long"]
    
    print(f"Training data: 2019-01-01 sd {cfg['train_end']} (6 years)")
    print(f"Test period:  {cfg['forecast_start']} sd {cfg['forecast_end']} (365 days)")
    
    # Prepare training data
    train_df = df.loc[:cfg['train_end']].copy()
    
    # Actual test data
    actual_start = pd.to_datetime(cfg['forecast_start'])
    actual_end = pd.to_datetime(cfg['forecast_end'])
    actual_data = df.loc[actual_start:actual_end, 'price'].values
    steps = len(actual_data)
    
    print(f"Steps to forecast: {steps}\n")
    
    results = {}
    
    # ---- ARIMA (with log transform) ----
    print("[1] ARIMA Model (with log transform)...")
    try:
        model_arima = train_arima(train_df)
        forecast_arima_vals = forecast_arima(model_arima, steps)
        forecast_arima_vals = np.maximum(forecast_arima_vals, 0)
        
        rmse_arima = calculate_rmse(actual_data, forecast_arima_vals)
        mape_arima = calculate_mape(actual_data, forecast_arima_vals)
        
        print(f"    ✓ RMSE: {rmse_arima:>10,.2f}")
        print(f"    ✓ MAPE: {mape_arima:>10.2f}%")
        
        results['arima'] = {
            'model': 'ARIMA',
            'rmse': rmse_arima,
            'mape': mape_arima,
            'forecast': forecast_arima_vals
        }
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        results['arima'] = None
    
    # ---- PROPHET (without log transform) ----
    print("\n[2] PROPHET Model (without log transform)...")
    try:
        prophet_train = train_df.reset_index()
        prophet_train.columns = ['ds', 'y']
        
        model_prophet, _ = train_prophet(prophet_train, use_log=False)
        forecast_prophet_df = forecast_prophet(model_prophet, steps, use_log=False)
        forecast_prophet_vals = forecast_prophet_df['yhat'].values
        forecast_prophet_vals = np.maximum(forecast_prophet_vals, 0)
        
        rmse_prophet = calculate_rmse(actual_data, forecast_prophet_vals)
        mape_prophet = calculate_mape(actual_data, forecast_prophet_vals)
        
        print(f"    ✓ RMSE: {rmse_prophet:>10,.2f}")
        print(f"    ✓ MAPE: {mape_prophet:>10.2f}%")
        
        results['prophet'] = {
            'model': 'PROPHET',
            'rmse': rmse_prophet,
            'mape': mape_prophet,
            'forecast': forecast_prophet_vals
        }
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        results['prophet'] = None
    
    # ---- COMPARISON ----
    print("\n" + "-"*80)
    print("COMPARISON & WINNER")
    print("-"*80)
    
    if results['arima'] and results['prophet']:
        mape_arima = results['arima']['mape']
        mape_prophet = results['prophet']['mape']
        rmse_arima = results['arima']['rmse']
        rmse_prophet = results['prophet']['rmse']
        
        print(f"ARIMA   - RMSE: {rmse_arima:>10,.2f}  |  MAPE: {mape_arima:>6.2f}%")
        print(f"PROPHET - RMSE: {rmse_prophet:>10,.2f}  |  MAPE: {mape_prophet:>6.2f}%")
        
        if mape_arima < mape_prophet:
            print(f"\n🏆 WINNER: ARIMA (MAPE {mape_arima:.2f}% < {mape_prophet:.2f}%)")
            results['winner'] = 'ARIMA'
        else:
            print(f"\n🏆 WINNER: PROPHET (MAPE {mape_prophet:.2f}% < {mape_arima:.2f}%)")
            results['winner'] = 'PROPHET'
    else:
        results['winner'] = None
    
    return results


def print_final_summary(short_results, mid_results, long_results):
    """Print final comprehensive summary"""
    
    print("\n\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "FINAL SUMMARY: HYBRID FORECAST MODEL SELECTION".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    # Short Term
    print("\n" + "─"*80)
    print("SHORT TERM (30 days)")
    print("─"*80)
    if short_results['arima'] and short_results['prophet']:
        print(f"ARIMA   - RMSE: {short_results['arima']['rmse']:>10,.2f}  |  MAPE: {short_results['arima']['mape']:>6.2f}%")
        print(f"PROPHET - RMSE: {short_results['prophet']['rmse']:>10,.2f}  |  MAPE: {short_results['prophet']['mape']:>6.2f}%")
        print(f"\n✓ WINNER: {short_results['winner']}")
        print(f"  └─ Recommended: Use {short_results['winner'].lower()} for short-term forecast")
    
    # Mid Term
    print("\n" + "─"*80)
    print("MID TERM (180 days)")
    print("─"*80)
    if mid_results['arima'] and mid_results['prophet']:
        print(f"ARIMA   - RMSE: {mid_results['arima']['rmse']:>10,.2f}  |  MAPE: {mid_results['arima']['mape']:>6.2f}%")
        print(f"PROPHET - RMSE: {mid_results['prophet']['rmse']:>10,.2f}  |  MAPE: {mid_results['prophet']['mape']:>6.2f}%")
        print(f"\n✓ WINNER: {mid_results['winner']}")
        print(f"  └─ Recommended: Use {mid_results['winner'].lower()} for mid-term forecast")
    
    # Long Term
    print("\n" + "─"*80)
    print("LONG TERM (365 days)")
    print("─"*80)
    if long_results['arima'] and long_results['prophet']:
        print(f"ARIMA   - RMSE: {long_results['arima']['rmse']:>10,.2f}  |  MAPE: {long_results['arima']['mape']:>6.2f}%")
        print(f"PROPHET - RMSE: {long_results['prophet']['rmse']:>10,.2f}  |  MAPE: {long_results['prophet']['mape']:>6.2f}%")
        print(f"\n✓ WINNER: {long_results['winner']}")
        print(f"  └─ Recommended: Use {long_results['winner'].lower()} for long-term forecast")
    
    # Configuration Recommendation
    print("\n" + "="*80)
    print("RECOMMENDED CONFIG.PY UPDATE")
    print("="*80)
    print("""
FORECAST_MODELS = {
    'short': {
        'days': SHORT_TERM_DAYS,
        'model': '""" + short_results['winner'].lower() + """',  # ← Winner dari evaluation
        'train_end': '2026-01-31',
        'test_start': '2026-02-01',
        'test_end': '2026-08-31'
    },
    'mid': {
        'days': MID_TERM_DAYS,
        'model': '""" + mid_results['winner'].lower() + """',  # ← Winner dari evaluation
        'train_end': '2025-07-31',
        'test_start': '2025-08-01',
        'test_end': '2026-01-31'
    },
    'long': {
        'days': LONG_TERM_DAYS,
        'model': '""" + long_results['winner'].lower() + """',  # ← Winner dari evaluation
        'train_end': '2026-01-31',
        'test_start': '2026-02-01',
        'test_end': '2027-01-31'
    }
}
    """)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "COMPREHENSIVE FORECAST EVALUATION: ARIMA vs PROPHET".center(78) + "║")
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
    short_results = evaluate_short_term(df)
    mid_results = evaluate_mid_term(df)
    long_results = evaluate_long_term(df)
    
    # Print final summary
    print_final_summary(short_results, mid_results, long_results)