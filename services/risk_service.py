import sqlite3
import pandas as pd
from config import DATABASE_PATH, ROLLING_WINDOW_DAYS
from models.risk_engine import (
    calculate_volatility,
    forecast_deviation,
    risk_scoring
)


def calculate_risk(df, forecast_series, horizon):
    """Calculate risk score and level based on forecast and historical data
    
    Args:
        df: DataFrame with price data
        forecast_series: Series with forecasted values
        horizon: 'short', 'mid', or 'long'
        
    Returns:
        tuple: (score, level)
    """
    try:
        df = df.copy()
        df = df.asfreq("D")
        df["price"] = df["price"].interpolate(method="linear")

        # Use config for rolling window instead of hardcoded value
        window = ROLLING_WINDOW_DAYS if ROLLING_WINDOW_DAYS else 30
        moving_avg = df["price"].rolling(window).mean().iloc[-1]
        rolling_std = df["price"].rolling(window).std().iloc[-1]

        last_forecast_value = forecast_series.iloc[-1]

        fdi = forecast_deviation(last_forecast_value, moving_avg, rolling_std)

        volatility_series = calculate_volatility(df["price"])
        volatility_flag = (
            volatility_series.iloc[-1] > 
            volatility_series.rolling(90).mean().iloc[-1] * 1.2
        )

        ci_breach = abs(fdi) > 3  

        score, level = risk_scoring(fdi, volatility_flag, ci_breach)

        print("FDI:", fdi)
        print("Volatility flag:", volatility_flag)
        print("CI breach", ci_breach)

        return score, level
    
    except Exception as e:
        print(f"Error calculating risk: {e}")
        # Return default values on error
        return 0, "Normal"


def save_risk(date, horizon, score, level):
    """Save risk calculation result to database
    
    Args:
        date: Date string (YYYY-MM-DD format)
        horizon: 'short', 'mid', or 'long'
        score: Risk score (integer)
        level: Risk level string ('Normal', 'Waspada', 'Bahaya')
    """
    try:
        # Ensure date is in correct format
        if isinstance(date, str):
            # Try to parse and reformat
            try:
                date_obj = pd.to_datetime(date)
                date = date_obj.strftime("%Y-%m-%d")
            except:
                pass  # Keep original string if parsing fails
        
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM risk_status WHERE horizon = ?
        """, (horizon,))

        cursor.execute("""
        INSERT INTO risk_status (date, horizon, risk_score, risk_level)
        VALUES (?, ?, ?, ?)
        """, (date, horizon, score, level))

        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"Error saving risk: {e}")

