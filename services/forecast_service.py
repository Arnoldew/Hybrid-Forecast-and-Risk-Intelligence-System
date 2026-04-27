"""
Forecast Service - Hybrid Ensemble Edition
Menggabungkan ARIMA dan Prophet secara paralel dengan pembobotan dinamis
untuk menghasilkan chart yang fluktuatif namun tetap akurat.
"""

import pandas as pd
import numpy as np
from config import FORECAST_MODELS, EVAL_WINDOWS, HYBRID_WEIGHTS
from models.arima_model import train_arima, forecast_arima
from models.prophet_model import train_prophet, forecast_prophet
from services.model_cache_service import (
    is_cache_valid,
    load_cached_model,
    save_model_to_cache
)

def prepare_dataframe(df):
    """Mempersiapkan dataframe untuk peramalan"""
    df = df.copy()
    df = df.sort_index()
    df = df.asfreq("D")
    # Interpolasi linear untuk menangani harga 0 atau missing values
    df["price"] = df["price"].replace(0, np.nan)
    df["price"] = df["price"].interpolate(method="linear", limit=30).ffill().bfill()
    return df

def get_hybrid_forecast(train_df, steps, horizon_type, cache_prefix=""):
    """
    Logika Inti Hybrid: Menjalankan ARIMA & Prophet, lalu menggabungkannya.
    """
    # 1. Ambil Bobot dari Config
    weights = HYBRID_WEIGHTS.get(horizon_type, {'arima': 0.5, 'prophet': 0.5})

     # FIX: Tambahkan anchor_date ke cache key
    # Sehingga setelah upload data baru, key berubah dan model diretrain
    anchor_date = train_df.index.max().strftime("%Y%m%d")
    cache_key_arima   = f"{cache_prefix}arima_{horizon_type}_{anchor_date}"
    cache_key_prophet = f"{cache_prefix}prophet_{horizon_type}_{anchor_date}"
    
    # 2. Proses ARIMA
    model_arima = None
    if is_cache_valid(cache_key_arima):
        model_arima = load_cached_model(cache_key_arima)
    
    if model_arima is None:
        model_arima = train_arima(train_df)
        save_model_to_cache(cache_key_arima, model_arima)
    
    # Forecast ARIMA (hasilnya sudah dalam skala asli/inverse-log)
    preds_arima = forecast_arima(model_arima, steps)

    # 3. Proses Prophet
    model_prophet = None
    if is_cache_valid(cache_key_prophet):
        model_prophet = load_cached_model(cache_key_prophet)
    
    if model_prophet is None:
        # Input Prophet perlu format ds dan y
        train_reset = train_df.reset_index()
        train_reset.columns = ["ds", "y"]
        # Kita gunakan use_log=False agar scaling konsisten saat penggabungan
        model_prophet, _ = train_prophet(train_reset, use_log=False)
        save_model_to_cache(cache_key_prophet, model_prophet)
    
    # Forecast Prophet
    forecast_df_prophet = forecast_prophet(model_prophet, steps, use_log=False)
    preds_prophet = forecast_df_prophet["yhat"].values

    # 4. Penggabungan (Weighted Ensemble)
    # Rumus: (Hasil ARIMA * Bobot A) + (Hasil Prophet * Bobot P)
    hybrid_values = (preds_arima * weights['arima']) + (preds_prophet * weights['prophet'])
    
    return hybrid_values

def forecast_horizon(df, horizon_name):
    """Fungsi generic untuk peramalan operasional di dashboard"""
    cfg = FORECAST_MODELS[horizon_name]
    df = prepare_dataframe(df)
    
    # Tentukan anchor (titik akhir data latih)
    train_df = df.loc[:cfg["train_end"]].copy()
    steps = cfg["days"]

    # FIX: Trim flat tail agar ARIMA tidak menghasilkan forecast konstan
    flat_days = 0
    price_clean = train_df["price"].dropna()
    for window in range(2, min(60, len(price_clean))):
        if price_clean.tail(window).std() > 1.0:
            flat_days = window - 1
            break

    if 0 < flat_days < len(train_df) * 0.10:
        train_df = train_df.iloc[:-flat_days].copy()
        print(f"[{horizon_name}] Trimmed {flat_days} flat tail rows for ARIMA training")

    # Jalankan Hybrid
    forecast_values = get_hybrid_forecast(train_df, steps, horizon_name, cache_prefix="op_")
    raw_values = np.asarray(forecast_values, dtype=float)
    
    # Buat Index Tanggal untuk Chart
    forecast_start = pd.to_datetime(cfg["train_end"]) + pd.Timedelta(days=1)
    forecast_index = pd.date_range(start=forecast_start, periods=steps, freq="D")
    
    # Gunakan array positional untuk menghindari index alignment
    wrapped_series = pd.Series(raw_values, index=forecast_index)

    return wrapped_series

# Wrapper functions untuk dipanggil oleh app.py
def forecast_short_term(df):
    return forecast_horizon(df, "short")

def forecast_mid_term(df):
    return forecast_horizon(df, "mid")

def forecast_long_term(df):
    return forecast_horizon(df, "long")

def generate_evaluation_forecasts(df):
    """Fungsi khusus untuk halaman evaluasi (backtesting)"""
    df = prepare_dataframe(df)
    forecasts = {}

    for horizon, cfg in EVAL_WINDOWS.items():
        train_df = df.loc[:cfg["train_end"]].copy()
        start = pd.to_datetime(cfg["forecast_start"])
        end = pd.to_datetime(cfg["forecast_end"])
        steps = (end - start).days + 1
        
        # Jalankan Hybrid
        forecast_values = get_hybrid_forecast(train_df, steps, horizon, cache_prefix="eval_")
        
        forecast_index = pd.date_range(start=start, periods=steps, freq="D")
        forecasts[horizon] = pd.Series(forecast_values, index=forecast_index)

    return forecasts