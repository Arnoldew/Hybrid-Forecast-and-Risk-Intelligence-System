import sqlite3
import pandas as pd 
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
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            horizon TEXT,
            risk_score INTEGER,
            risk_level TEXT,
            UNIQUE(date, horizon)
        )
        """)

        conn.commit()
        conn.close()
        print("✓ Database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {e}")


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
