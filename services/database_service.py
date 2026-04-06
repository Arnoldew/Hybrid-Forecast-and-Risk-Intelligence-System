import sqlite3
import pandas as pd 
from datetime import datetime
from config import DATABASE_PATH


def init_db():
    """Initialize database with required tables"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Table harga
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE,
            price REAL
        )
        """)

        # Table forecast
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS forecast_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            horizon TEXT,
            forecast_date TEXT,
            predicted_price REAL,
            UNIQUE (horizon, forecast_date)
        )
        """)

        # Table risk Status
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS risk_status (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            date         TEXT NOT NULL,
            horizon      TEXT NOT NULL,
            risk_score   INTEGER,
            risk_level   TEXT,
            direction    TEXT,
            fdi_value    REAL,
            vol_ratio    REAL,
            trend_slope  REAL,
            risk_message TEXT,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        conn.commit()
        conn.close()
        print("✓ Database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {e}")


def save_risk(date, horizon, score, level, direction, fdi_value, vol_ratio, trend_slope, message):
    """
    Simpan hasil kalkulasi risiko ke database (append-only — tidak menghapus history).
    """
    try:
        if isinstance(date, str):
            try:
                date = pd.to_datetime(date).strftime("%Y-%m-%d")
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not parse date '{date}': {e}")
                date = datetime.now().strftime("%Y-%m-%d")
        
        # Validasi nilai numerik sebelum INSERT
        fdi_value   = float(fdi_value)   if fdi_value   is not None else 0.0
        vol_ratio   = float(vol_ratio)   if vol_ratio   is not None else 1.0
        trend_slope = float(trend_slope) if trend_slope is not None else 0.0
        score       = int(score)         if score       is not None else 0

        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO risk_status 
            (date, horizon, risk_score, risk_level, direction, fdi_value, vol_ratio, trend_slope, risk_message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (date, horizon, int(score), level, direction,
              round(float(fdi_value), 4), round(float(vol_ratio), 4),
              round(float(trend_slope), 4), message))

        conn.commit()
        conn.close()

    except Exception as e:
        print(f"Error saving risk [{horizon}]: {e}")


def get_risk_history(horizon, limit=30):
    """
    Ambil history risk status untuk horizon tertentu.
    Berguna untuk menampilkan tren risiko dari waktu ke waktu.
    
    Returns:
        list of dict
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT date, risk_score, risk_level, direction, fdi_value, vol_ratio, risk_message, created_at
            FROM risk_status
            WHERE horizon = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (horizon, limit))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                'date'        : r[0],
                'score'       : r[1],
                'level'       : r[2],
                'direction'   : r[3],
                'fdi_value'   : r[4],
                'vol_ratio'   : r[5],
                'message'     : r[6],
                'created_at'  : r[7]
            }
            for r in rows
        ]

    except Exception as e:
        print(f"Error getting risk history: {e}")
        return []
    

def save_forecast(horizon, forecast_series):
    
    """Save forecast results to database
    
    Args:
        horizon: 'short', 'mid', or 'long'
        forecast_series: Series with forecast dates as index and prices as values
    """
    
    # Normalize: jika DataFrame ambil kolom yhat
    if isinstance(forecast_series, pd.DataFrame):
        forecast_series = forecast_series['yhat']

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM forecast_results WHERE horizon = ?
        """, (horizon,))

        for date, value in forecast_series.items():
            cursor.execute("""
            INSERT INTO forecast_results (horizon, forecast_date, predicted_price)
            VALUES (?, ?, ?)
            """, (horizon, str(date), float(value)))

        conn.commit()
        conn.close()
        print(f"✓ Saved {horizon} forecast to database")
    except Exception as e:
        print(f"Error saving forecast: {e}")
