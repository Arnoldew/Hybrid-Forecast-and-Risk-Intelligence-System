import pandas as pd
import numpy as np

from models.arima_model import train_arima, forecast_arima, evaluate_forecast
from models.prophet_model import train_prophet, forecast_prophet
from models.risk_engine import calculate_volatility, forecast_deviation, risk_scoring
import os

# =========================
# LOAD DATA
# =========================
folder_path = "dataset"

all_file = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if f.endswith(".xlsx")
]

df_list = []

for file in all_file:
    df_temp = pd.read_excel(file)
    df_list.append(df_temp)

#Gabungkan semua file
df = pd.concat(df_list,ignore_index=True)

#Pastikan nama kolom sesuai
df.columns = df.columns.str.lower()

#Jika kolom beda nama
df = df.rename(columns={
    'harga' : 'price',
    'price' : 'price',
    'tanggal' : 'date',
    'date' : 'date',
})

#Konversi tipe tanggal
df['date'] = pd.to_datetime(df['date'], dayfirst=True)

#Urutkan berdasarkan tanggal
df = df.sort_values('date')

#Set index
df.set_index('date', inplace=True)

# Replace 0 dengan NaN
df['price'] = df['price'].replace(0, np.nan)
# Interpolasi seluruh dataset
df['price'] = df['price'].interpolate(method='linear')
# Drop jika masih ada NaN di awal/akhir
df = df.dropna()

# Pastikan index daily continuous
df = df.asfreq('D')

# Interpolasi lagi setelah asfreq
df['price'] = df['price'].interpolate(method='linear')

#Hanya ambil kolom price
df = df[['price']]

print("Data Loaded Successfully")
print(df.head())

print("Total data:", len(df))
print("Date range:", df.index.min(), "to", df.index.max())
print("Any zero left?", (df['price'] == 0).sum())
print("Any NaN?", df['price'].isna().sum())

#Function Evaluation
def run_evaluation(train_end_date, test_start_date, test_end_date, label):
    print(f"\n========== {label} ==========")

    train = df[df.index < train_end_date].copy()
    test = df[(df.index >= test_start_date) & (df.index <= test_end_date)].copy()

    steps = len(test)

    print("Train until:", train.index.max())
    print("Test from:", test.index.min(), "to", test.index.max())
    print("Test length:", steps)

    # ================= ARIMA =================
    print("\n--- ARIMA ---")
    arima_model = train_arima(train)
    arima_forecast = forecast_arima(arima_model, steps)

    rmse_a, mape_a = evaluate_forecast(test['price'], arima_forecast)
    print("ARIMA RMSE:", rmse_a)
    print("ARIMA MAPE:", mape_a)

    # ================= PROPHET =================
    print("\n--- PROPHET ---")
    prophet_train = train.reset_index()
    prophet_train.columns = ['ds', 'y']
    prophet_model = train_prophet(prophet_train)
    forecast = forecast_prophet(prophet_model, steps)

    forecast = forecast.set_index('ds')

    common_index = test.index.intersection(forecast.index)
    actual = test.loc[common_index, 'price']
    predicted = forecast.loc[common_index, 'yhat']

    rmse_p, mape_p = evaluate_forecast(actual, predicted)
    print("Prophet RMSE:", rmse_p)
    print("Prophet MAPE:", mape_p)

    return {
        "ARIMA_MAPE": mape_a,
        "PROPHET_MAPE": mape_p
    }

# =========================
# SPLIT 3 HORIZON
# =========================

results = {}

# SHORT TERM (Desember 2025)
results["Short-Term"] = run_evaluation(
    train_end_date="2025-12-01",
    test_start_date="2025-12-01",
    test_end_date="2025-12-31",
    label="SHORT TERM (30 Hari)"
)

# MID TERM (Juni–Des 2025)
results["Mid-Term"] = run_evaluation(
    train_end_date="2025-06-01",
    test_start_date="2025-06-01",
    test_end_date="2025-12-31",
    label="MID TERM (7 Bulan)"
)

# LONG TERM (Jan–Des 2025)
results["Long-Term"] = run_evaluation(
    train_end_date="2025-01-01",
    test_start_date="2025-01-01",
    test_end_date="2025-12-31",
    label="LONG TERM (12 Bulan)"
)

print("\n================ FINAL SUMMARY ================")
for horizon, value in results.items():
    print(horizon, "=>", value)
