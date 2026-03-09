"""
Forecast Service
Menyediakan forecast untuk short, mid, dan long term dengan model optimal
"""

import pandas as pd
import numpy as np
from config import FORECAST_MODELS
from models.arima_model import train_arima, forecast_arima
from models.prophet_model import train_prophet, forecast_prophet
from services.model_cache_service import (
    is_cache_valid,
    load_cached_model,
    save_model_to_cache
)


def prepare_dataframe(df):
    """Prepare dataframe for forecasting"""
    df = df.copy()
    df = df.sort_index()
    df = df.asfreq("D")
    df["price"] = df["price"].interpolate(method="linear", limit=7)
    return df


def forecast_horizon(df, horizon_name):
    """
    Generic function untuk forecast semua horizon dengan skema sesuai konteks:

    - Garis hitam = data aktual (tahun target tetap actual, bukan hasil forecast)
    - Garis forecast (short/mid/long) dimulai pada "start window" masing-masing:
        short:  1 bulan terakhir dari horizon window (≈ 30 hari terakhir)
        mid:    6 bulan terakhir dari horizon window (≈ 180 hari terakhir)
        long:   12 bulan terakhir dari horizon window (≈ 365 hari terakhir)

    Operational shifting:
    - Saat ada data aktual baru (mis. Jan 2026), window horizon maju otomatis.
      Forecast selalu untuk periode (anchor+1 ... anchor+days).
      Data sebelum start forecast = data train.
    """
    if horizon_name not in FORECAST_MODELS:
        raise ValueError(f"Unknown horizon: {horizon_name}")

    cfg = FORECAST_MODELS[horizon_name]
    model_type = cfg["model"]
    days = int(cfg["days"])

    df = prepare_dataframe(df)
    if df.empty:
        return pd.Series(dtype=float)

    anchor_date = df.index.max().normalize()
    forecast_start = anchor_date + pd.Timedelta(days=1)
    forecast_end = forecast_start + pd.Timedelta(days=days - 1)

    # Train = semua data sebelum forecast_start
    train = df.loc[df.index < forecast_start].copy()
    if train.dropna().shape[0] < 10:
        return pd.Series(dtype=float)

    # Cache key harus mempertimbangkan anchor_date supaya "maju sebulan" tidak pakai model lama
    cache_key = f"{horizon_name}_{anchor_date.strftime('%Y%m%d')}"

    if is_cache_valid(cache_key):
        model = load_cached_model(cache_key)
        if model is not None:
            if model_type == "arima":
                forecast = forecast_arima(model, days)
            else:
                forecast_df = forecast_prophet(model, days)
                forecast = forecast_df["yhat"].values

            full_index = pd.date_range(start=forecast_start, periods=len(forecast), freq="D")
            full_series = pd.Series(forecast, index=full_index).clip(lower=0)

            # Untuk visualisasi: tampilkan hanya bagian yang sesuai start window horizon
            display_start = forecast_end - pd.Timedelta(days=days - 1)
            if days >= 30:
                if horizon_name == "short":
                    display_start = forecast_end - pd.Timedelta(days=30 - 1)
                elif horizon_name == "mid":
                    display_start = forecast_end - pd.Timedelta(days=180 - 1)
                elif horizon_name == "long":
                    display_start = forecast_start

            return full_series.loc[full_series.index >= display_start]

    print(f"Training {model_type.upper()} model for {horizon_name} term (anchor={anchor_date.date()})...")

    if model_type == "arima":
        model = train_arima(train)
        forecast = forecast_arima(model, days)
    else:
        train_reset = train.reset_index()
        train_reset.columns = ["ds", "y"]
        model = train_prophet(train_reset)
        forecast_df = forecast_prophet(model, days)
        forecast = forecast_df["yhat"].values

    save_model_to_cache(cache_key, model)

    full_index = pd.date_range(start=forecast_start, periods=len(forecast), freq="D")
    full_series = pd.Series(forecast, index=full_index).clip(lower=0)

    # Untuk visualisasi: start window sesuai horizon (short=30, mid=180, long=365)
    if horizon_name == "short":
        display_start = forecast_end - pd.Timedelta(days=30 - 1)
    elif horizon_name == "mid":
        display_start = forecast_end - pd.Timedelta(days=180 - 1)
    else:
        display_start = forecast_start

    return full_series.loc[full_series.index >= display_start]


def forecast_short_term(df):
    """Forecast short term (30 days) using PROPHET)"""
    # Sesuai konteks chart: short-term = forecast 1 bulan ke depan dari anchor (data terakhir)
    return forecast_horizon(df, "short")


def forecast_mid_term(df):
    """Forecast mid term (180 days) using PROPHET"""
    # Mid-term = 6 bulan ke depan dari anchor
    return forecast_horizon(df, "mid")


def forecast_long_term(df):
    """Forecast long term (365 days) using ARIMA"""
    # Long-term = 12 bulan ke depan dari anchor
    return forecast_horizon(df, "long")
