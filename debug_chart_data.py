"""
Debug script untuk melihat data yang dikirim ke chart
"""
import pandas as pd
import sqlite3
from config import DATABASE_PATH, ROLLING_WINDOW_DAYS
from services.forecast_service import (
    forecast_short_term,
    forecast_mid_term,
    forecast_long_term
)

def load_data():
    conn = sqlite3.connect(DATABASE_PATH)
    df = pd.read_sql("SELECT date, price FROM price_data", conn)
    conn.close()
    
    if df.empty:
        return pd.DataFrame()
    
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = df.sort_index()

    return df

df = load_data()

print("="*80)
print("FULL DATASET")
print("="*80)
print(f"Shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Last 10 rows:\n{df.tail(10)}")

# Generate forecasts
print("\n" + "="*80)
print("GENERATING FORECASTS")
print("="*80)

short_forecast = forecast_short_term(df)
mid_forecast = forecast_mid_term(df)
long_forecast = forecast_long_term(df)

print(f"\nShort Forecast:")
print(f"  Shape: {short_forecast.shape}")
print(f"  Date range: {short_forecast.index.min()} to {short_forecast.index.max()}")
print(f"  Values: {short_forecast.values}")

print(f"\nMid Forecast:")
print(f"  Shape: {mid_forecast.shape}")
print(f"  Date range: {mid_forecast.index.min()} to {mid_forecast.index.max()}")
print(f"  First 5 values: {mid_forecast.values[:5]}")

print(f"\nLong Forecast:")
print(f"  Shape: {long_forecast.shape}")
print(f"  Date range: {long_forecast.index.min()} to {long_forecast.index.max()}")
print(f"  First 5 values: {long_forecast.values[:5]}")

# Chart data preparation
print("\n" + "="*80)
print("CHART DATA PREPARATION")
print("="*80)

df_last_year = df.loc[df.index >= (df.index.max() - pd.Timedelta(days=ROLLING_WINDOW_DAYS))]
print(f"\nHistorical data (last 365 days):")
print(f"  Shape: {df_last_year.shape}")
print(f"  Date range: {df_last_year.index.min()} to {df_last_year.index.max()}")

actual_2026_start = pd.Timestamp("2026-01-01")
actual_2026_end = pd.Timestamp("2026-01-31")
df_actual_2026 = df[(df.index >= actual_2026_start) & (df.index <= actual_2026_end)].copy()
print(f"\nActual 2026 data:")
print(f"  Shape: {df_actual_2026.shape}")
print(f"  Date range: {df_actual_2026.index.min()} to {df_actual_2026.index.max()}")
print(f"  Values:\n{df_actual_2026}")

df_chart_data = pd.concat([df_last_year, df_actual_2026])
print(f"\nCombined chart data:")
print(f"  Shape: {df_chart_data.shape}")
print(f"  Date range: {df_chart_data.index.min()} to {df_chart_data.index.max()}")

chart_dates = df_chart_data.index.strftime("%Y-%m-%d").tolist()
print(f"\nChart dates (first 10): {chart_dates[:10]}")
print(f"Chart dates (last 10): {chart_dates[-10:]}")

short_dates = short_forecast.index.strftime("%Y-%m-%d").tolist()
print(f"\nShort forecast dates: {short_dates}")

mid_dates = mid_forecast.index.strftime("%Y-%m-%d").tolist()
print(f"\nMid forecast dates (first 10): {mid_dates[:10]}")
print(f"Mid forecast dates (last 10): {mid_dates[-10:]}")

long_dates = long_forecast.index.strftime("%Y-%m-%d").tolist()
print(f"\nLong forecast dates (first 10): {long_dates[:10]}")
print(f"Long forecast dates (last 10): {long_dates[-10:]}")

# Check if dates match
print("\n" + "="*80)
print("DATE MATCHING CHECK")
print("="*80)

short_match = sum(1 for d in short_dates if d in chart_dates)
print(f"Short forecast dates in chart: {short_match}/{len(short_dates)}")

mid_match = sum(1 for d in mid_dates if d in chart_dates)
print(f"Mid forecast dates in chart: {mid_match}/{len(mid_dates)}")

long_match = sum(1 for d in long_dates if d in chart_dates)
print(f"Long forecast dates in chart: {long_match}/{len(long_dates)}")
