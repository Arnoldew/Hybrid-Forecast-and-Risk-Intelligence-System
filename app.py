from flask import Flask, render_template
from flask import request, redirect
import os
import pandas as pd
import json
import sqlite3
from utils.metrics import calculate_mape
from services.evaluation_service import calculate_mape
from services.forecast_service import generate_evaluation_forecasts
from config import DATABASE_PATH, ROLLING_WINDOW_DAYS
from datetime import datetime
from services.risk_service import calculate_risk, save_risk
from services.database_service import init_db, save_forecast
from services.forecast_service import (
    forecast_short_term,
    forecast_mid_term,
    forecast_long_term
)

app = Flask(__name__)
init_db()

def load_data():
    """Load price data from database
    
    Returns:
        DataFrame with date index and price column, or empty DataFrame if no data
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        df = pd.read_sql("SELECT date, price FROM price_data", conn)
        conn.close()
        
        if df.empty:
            return pd.DataFrame()
        
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df = df.sort_index()

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()


@app.route("/")
def dashboard():
    try:
        df = load_data()

        if df.empty:
            return "Dataset Kosong"

        # Generate forecasts
        try:
            short_forecast = forecast_short_term(df)
            mid_forecast = forecast_mid_term(df)
            long_forecast = forecast_long_term(df)
        except Exception as e:
            print(f"Error generating forecasts: {e}")
            return f"Error generating forecasts: {str(e)}"

        eval_forecasts = generate_evaluation_forecasts(df)
        short_fc = eval_forecasts['short']
        mid_fc = eval_forecasts['mid']
        long_fc = eval_forecasts['long']

        actual_short = df.loc["2025-12-01":"2025-12-31"]["price"]
        actual_mid = df.loc["2025-07-01":"2025-12-31"]["price"]
        actual_long = df.loc["2025-01-01":"2025-12-31"]["price"]
    
        short_mape = calculate_mape(actual_short, short_fc["yhat"])
        mid_mape = calculate_mape(actual_mid, mid_fc["yhat"])
        long_mape = calculate_mape(actual_long, long_fc["yhat"])

        eval_short = eval_forecasts["short"]
        eval_mid = eval_forecasts["mid"]
        eval_long = eval_forecasts["long"]

        # Hitung MAPE untuk setiap horizon
        # (ambil data aktual yang indeksnya ada di forecast, lalu bandingkan)
        actual_short = df["price"].reindex(eval_short.index).dropna()
        actual_mid   = df["price"].reindex(eval_mid.index).dropna()
        actual_long  = df["price"].reindex(eval_long.index).dropna()

        mape_short = calculate_mape(
            actual_short.values,
            eval_short.loc[actual_short.index].values
        )
        mape_mid = calculate_mape(
            actual_mid.values,
            eval_mid.loc[actual_mid.index].values
        )
        mape_long = calculate_mape(
            actual_long.values,
            eval_long.loc[actual_long.index].values
        )

        save_forecast("short", short_forecast)
        save_forecast("mid", mid_forecast)
        save_forecast("long", long_forecast)
        
        # Risk Calculation 
        try:
            short_score, short_level = calculate_risk(df, short_forecast, "short")
            save_risk(str(df.index.max()), "short", short_score, short_level)

            mid_score, mid_level = calculate_risk(df, mid_forecast, "mid")
            save_risk(str(df.index.max()), "mid", mid_score, mid_level)

            long_score, long_level = calculate_risk(df, long_forecast, "long")
            save_risk(str(df.index.max()), "long", long_score, long_level)
        except Exception as e:
            print(f"Error calculating risk: {e}")
            short_level = "Normal"
            mid_level = "Normal"
            long_level = "Normal"

        # ============================================================================
        # DATA PREPARATION FOR CHART (sesuai konteks user)
        # ============================================================================
        # Garis hitam:
        # - Data aktual tahun 2025 full
        # - Jika ada data terbaru (mis. Jan 2026), ikut ditampilkan sebagai actual juga.
        #
        # Garis forecast:
        # - Short  : 1 bulan terakhir dari window forecast (≈ 30 hari terakhir)
        # - Mid    : 6 bulan terakhir dari window forecast (≈ 180 hari terakhir)
        # - Long   : 12 bulan dari window forecast (≈ 365 hari)
        #
        # Catatan: window forecast sendiri selalu "maju" mengikuti data aktual terakhir (anchor).

        # =====================================================================
        # DATA UNTUK CHART (TRAIN + ACTUAL)
        # =====================================================================
        df_chart_data = df.loc[
            (df.index >= pd.Timestamp("2022-01-01")) &
            (df.index <= pd.Timestamp("2025-12-31"))
        ].copy()

        chart_dates = df_chart_data.index.strftime("%Y-%m-%d").tolist()
        chart_prices = df_chart_data["price"].tolist()

        # =====================================================================
        # FORECAST DATA UNTUK CHART (EVALUATION FORECAST)
        # =====================================================================
        short_dates = eval_short.index.strftime("%Y-%m-%d").tolist()
        short_values = eval_short.tolist()

        mid_dates = eval_mid.index.strftime("%Y-%m-%d").tolist()
        mid_values = eval_mid.tolist()

        long_dates = eval_long.index.strftime("%Y-%m-%d").tolist()
        long_eval_values = eval_long.tolist()

        # Combine chart dates dengan forecast dates untuk timeline lengkap
        all_dates = sorted(list(set(chart_dates + short_dates + mid_dates + long_dates)))

        # Calculate averages
        short_avg = round(sum(short_values)/len(short_values), 2) if short_values else 0
        mid_avg = round(sum(mid_values)/len(mid_values), 2) if mid_values else 0
        long_avg = round(sum(long_values)/len(long_values), 2) if long_values else 0

        last_update = datetime.now().strftime("%d %B %Y %H:%M")

        # Volatility calculation (dari seluruh data)
        volatility = df['price'].rolling(30).std()
        vol_dates = volatility.index.strftime('%Y-%m-%d').tolist()
        vol_values = volatility.fillna(0).tolist()

        # Forecast start dates untuk setiap horizon
        short_forecast_start = eval_short.index.min().strftime("%Y-%m-%d")
        mid_forecast_start = eval_mid.index.min().strftime("%Y-%m-%d")
        long_forecast_start = eval_long.index.min().strftime("%Y-%m-%d")

        if isinstance(long_forecast, pd.DataFrame):
            long_upper = long_forecast['yhat_upper'].tolist()
            long_lower = long_forecast['yhat_lower'].tolist()
            long_prod_values = long_forecast['yhat'].tolist()
        else:
            long_upper = None
            long_lower = None
            long_prod_values = long_forecast.tolist()

        return render_template(
            "dashboard.html",
            all_dates=all_dates,
            chart_dates=chart_dates,
            chart_prices=chart_prices,
            short_dates=short_dates,
            short_values=short_values,
            mid_dates=mid_dates,
            mid_values=mid_values,
            long_dates=long_dates,
            long_eval_values=long_eval_values,
            long_prod_values=long_prod_values,
            short_level=short_level,
            mid_level=mid_level,
            long_level=long_level,
            short_forecast_start=short_forecast_start,
            mid_forecast_start=mid_forecast_start,
            long_forecast_start=long_forecast_start,
            last_update=last_update,
            short_avg=short_avg,
            mid_avg=mid_avg,
            long_avg=long_avg,
            long_upper=long_upper,
            long_lower=long_lower,
            vol_dates=vol_dates,
            vol_values=vol_values,
            mape_short=round(mape_short, 2),
            mape_mid=round(mape_mid, 2),
            mape_long=round(mape_long, 2)
        )
        
    except Exception as e:
        print(f"Error in dashboard: {e}")
        return f"Error loading dashboard: {str(e)}"

@app.route("/download")
def download_forecast():
    from flask import send_file
    import pandas as pd
    
    df = load_data()

    if df.empty:
        return "Database Kosong"

    long_forecast = forecast_long_term(df)

    if isinstance(long_forecast, pd.DataFrame):
        forecast_values = long_forecast['yhat'].values
        forecast_index = long_forecast.index
    else: 
        forecast_values = long_forecast.values
        forecast_index = long_forecast.index

    df_download = pd.DataFrame({
        "date": forecast_index.strftime("%Y-%m-%d"),
        "long_term_forecast": forecast_values
    })

    file_path = "Forecast_export.csv"
    df_download.to_csv(file_path, index=False)

    
    return send_file(file_path, as_attachment=True)


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        try:
            file = request.files["file"]

            if file.filename == "":
                return "No file selected"
            
            # Validate file extension
            allowed_extensions = {'.xlsx', '.xls'}
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in allowed_extensions:
                return "Invalid file format. Please upload Excel files (.xlsx or .xls)"
            
            df = pd.read_excel(file)

            #Standarrisasi kolom
            df.columns= df.columns.str.lower().str.strip()

            df = df.rename(columns={
                "tanggal": "date",
                "harga": "price"
            })

            df["price"] = df["price"].str.replace(".", "", regex=False)
            df["price"] = df["price"].str.replace(",", "", regex=False)
            df["price"] = pd.to_numeric(df["price"], errors="coerce")

            if"date" not in df.columns or "price" not in df.columns:
                return "Format file tidak sesuai. Kolom yang diperlukan: 'date' dan 'price'"
            
            df = df.dropna(subset=["date", "price"])
            df = df[df["price"] > 0]
            
            
            df["date"] = pd.to_datetime(df["date"], dayfirst=True)
            df = df.sort_values("date")

            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()

            print("Rows to Insert:", len(df))

            inserted_count = 0
            for _, row in df.iterrows():
                try:
                    cursor.execute(
                        "INSERT OR REPLACE INTO price_data (date, price) VALUES (?, ?)",
                        (row["date"].strftime("%Y-%m-%d"), float(row["price"]))
                    )
                    inserted_count += 1
                except Exception as e:
                    print(f"Error inserting row: {e}")

            conn.commit()
            conn.close()

            print(f"Successfully inserted {inserted_count} rows")
            return redirect("/")

        except Exception as e:
            print(f"Error during upload: {e}")
            return f"Error uploading file: {str(e)}"

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)
