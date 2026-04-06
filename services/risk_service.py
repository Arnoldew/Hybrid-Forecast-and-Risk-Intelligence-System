import sqlite3
import pandas as pd
from config import DATABASE_PATH, RISK_WINDOW_DAYS, TREND_WINDOW_DAYS, VOL_RECENT_DAYS, VOL_BASELINE_DAYS
from models.risk_engine import (
    calculate_volatility_ratio,
    calculate_trend_slope,
    get_seasonal_baseline,
    forecast_deviation,
    risk_scoring,
    generate_risk_message
)


def calculate_risk(df, forecast_series, horizon):
    """
    Hitung risk score dan detail dari forecast dan data historis.

    Returns:
        tuple: (score, level, direction, fdi_value, vol_ratio, trend_slope, message)
    """
    try:
        df = df.copy()
        df = df.asfreq("D")
        df["price"] = df["price"].interpolate(method="linear")

        price = df["price"]

        # --- 1. FDI ---
        moving_avg  = price.rolling(RISK_WINDOW_DAYS).mean().iloc[-1]
        rolling_std = price.rolling(RISK_WINDOW_DAYS).std().iloc[-1]
        last_forecast = float(forecast_series.iloc[-1])
        fdi = forecast_deviation(last_forecast, moving_avg, rolling_std)

        # --- 2. Volatility ratio ---
        recent_std = price.rolling(VOL_RECENT_DAYS).std().iloc[-1]
        baseline_std = price.rolling(VOL_BASELINE_DAYS).std().iloc[-1]
        vol_ratio = calculate_volatility_ratio(
            price,
            recent_days   = VOL_RECENT_DAYS,
            baseline_days = VOL_BASELINE_DAYS
        )

        # --- 3. Trend slope ---
        trend_slope, trend_direction = calculate_trend_slope(price, window=TREND_WINDOW_DAYS)

        # --- 4. Seasonal deviation ---
        anchor_month = df.index.max().month
        seasonal_mean = get_seasonal_baseline(price, anchor_month)
        seasonal_deviation = None
        if seasonal_mean and seasonal_mean > 0:
            seasonal_deviation = ((last_forecast - seasonal_mean) / seasonal_mean) * 100

        # --- 5. Scoring ---
        score, level, breakdown = risk_scoring(
            fdi                = fdi,
            vol_ratio          = vol_ratio,
            trend_direction    = trend_direction,
            seasonal_deviation = seasonal_deviation
        )

        # --- 6. Risk message ---
        message = generate_risk_message(fdi, vol_ratio, trend_direction, level, horizon)

        print(f"[{horizon}] FDI: {fdi:.3f} (forecast={last_forecast:.0f}, "
              f"avg={moving_avg:.0f}, std={rolling_std:.2f}) | "
              f"Vol ratio: {vol_ratio:.2f} (recent_std={recent_std if 'recent_std' in locals() else 'N/A'}, baseline_std={baseline_std if 'baseline_std' in locals() else 'N/A'}) | "
              f"Trend: {trend_direction} (slope={trend_slope:.4f}) | "
              f"Score: {score} | Level: {level}")

        return score, level, trend_direction, fdi, vol_ratio, trend_slope, message

    except Exception as e:
        print(f"Error calculating risk [{horizon}]: {e}")
        return 0, "Normal", "stable", 0.0, 1.0, 0.0, "Data tidak cukup untuk kalkulasi risiko."

