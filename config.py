# config.py

import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "database.db")

# ============================================================================
# 1. MODEL CACHE CONFIGURATION (Penting untuk model_cache_service)
# ============================================================================
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "cache")
MODEL_CACHE_TTL = 86400  # Waktu simpan cache dalam detik (24 jam)

# Pastikan folder cache ada, jika tidak, buat otomatis
if not os.path.exists(MODEL_CACHE_DIR):
    os.makedirs(MODEL_CACHE_DIR)

# ============================================================================
# 2. FORECAST HORIZONS CONFIGURATION (Days)
# ============================================================================
SHORT_TERM_DAYS = 30
MID_TERM_DAYS = 180
LONG_TERM_DAYS = 365

# ============================================================================
# 3. HYBRID ENSEMBLE CONFIGURATION
# ============================================================================
# Bobot Dinamis: Jangka pendek fokus ke akurasi (ARIMA), 
# Jangka panjang fokus ke pola musiman (Prophet).
HYBRID_WEIGHTS = {
    'short': {
        'arima': 0.8,   # 80% ARIMA
        'prophet': 0.2  # 20% Prophet
    },
    'mid': {
        'arima': 0.5,
        'prophet': 0.5
    },
    'long': {
        'arima': 0.3,   # 30% ARIMA
        'prophet': 0.7  # 70% Prophet
    }
}

# Konfigurasi Window Operasional
FORECAST_MODELS = {
    'short': {
        'days': SHORT_TERM_DAYS,
        'train_end': '2026-01-31',
        'test_start': '2026-02-01',
        'test_end': '2026-08-31'
    },
    'mid': {
        'days': MID_TERM_DAYS,
        'train_end': '2025-07-31',
        'test_start': '2025-08-01',
        'test_end': '2026-01-31'
    },
    'long': {
        'days': LONG_TERM_DAYS,
        'train_end': '2026-01-31',
        'test_start': '2026-02-01',
        'test_end': '2027-01-31'
    }
}

# ============================================================================
# 4. RISK SERVICE CONFIGURATION (Early Warning System)
# ============================================================================
RISK_WINDOW_DAYS = 30     # Jendela evaluasi risiko
TREND_WINDOW_DAYS = 7     # Jendela tren kenaikan
VOL_RECENT_DAYS = 90      # Volatilitas jangka pendek
VOL_BASELINE_DAYS = 365    # Volatilitas jangka panjang
ROLLING_WINDOW_DAYS = 30  # Moving average window

# ============================================================================
# 5. EVALUATION WINDOWS (Backtesting)
# ============================================================================
EVAL_WINDOWS = {
    "short": {
        "train_end": "2025-11-30",
        "forecast_start": "2025-12-01",
        "forecast_end": "2025-12-31"
    },
    "mid": {
        "train_end": "2025-06-30",
        "forecast_start": "2025-07-01",
        "forecast_end": "2025-12-31"
    },
    "long": {
        "train_end": "2024-12-31",
        "forecast_start": "2025-01-01",
        "forecast_end": "2025-12-31"
    }
}