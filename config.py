import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DATABASE_PATH = os.path.join(BASE_DIR, "database.db")

# Forecast horizons configuration
SHORT_TERM_DAYS = 30
MID_TERM_DAYS = 180
LONG_TERM_DAYS = 365

# Model selection based on evaluation results
# Short Term: PROPHET (RMSE: 61,874.25)
# Mid Term: PROPHET (RMSE: 30,697.56)
# Long Term: ARIMA (RMSE: 27,088.97)
FORECAST_MODELS = {
    'short': {
        'days': SHORT_TERM_DAYS,
        'model': 'prophet',
        'train_end': '2025-12-31',
        'test_start': '2026-01-01',
        'test_end': '2026-01-31'
    },
    'mid': {
        'days': MID_TERM_DAYS,
        'model': 'prophet',
        'train_end': '2025-07-31',
        'test_start': '2025-08-01',
        'test_end': '2026-01-31'
    },
    'long': {
        'days': LONG_TERM_DAYS,
        'model': 'arima',
        'train_end': '2025-01-31',
        'test_start': '2025-02-01',
        'test_end': '2026-01-31'
    }
}

# =========================================
# Evaluation Forecast Window (2025)
# =========================================
# Untuk backtesting dan validasi model akurasi di tahun 2025
# Gunakan MODEL dan TRAIN_END yang SAMA dengan FORECAST_MODELS

EVAL_WINDOWS = {
    "short": {
        "train_end": "2025-11-30",
        "forecast_start": "2025-12-01",
        "forecast_end": "2025-12-31",
        "model": "prophet"
    },

    "mid": {
        "train_end": "2025-06-30",
        "forecast_start": "2025-07-01",
        "forecast_end": "2025-12-31",
        "model": "prophet"
    },

    "long": {
        "train_end": "2024-12-31",
        "forecast_start": "2025-01-01",
        "forecast_end": "2025-12-31",
        "model": "arima"
    }

}

# Model caching configuration
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "model_cache")
MODEL_CACHE_TTL = 3600  # Cache for 1 hour (in seconds)

# Data configuration
ROLLING_WINDOW_DAYS = 365  # Display last 365 days in chart

# Risk engine configuration
RISK_WINDOW_DAYS    = 30   # window FDI + moving average
TREND_WINDOW_DAYS   = 7    # window slope detection (tren jangka pendek)
VOL_RECENT_DAYS     = 7    # window volatility "saat ini"
VOL_BASELINE_DAYS   = 30   # window volatility "normal" sebagai pembanding   
