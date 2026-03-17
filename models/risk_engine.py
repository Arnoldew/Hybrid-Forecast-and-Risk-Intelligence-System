import numpy as np
from scipy import stats



def calculate_volatility_ratio(price_series, recent_days=7, baseline_days=30):
    """
    Hitung rasio volatilitas: current vs baseline.
    Rasio > 1.2 berarti volatilitas saat ini 20% lebih tinggi dari normal.
    
    Returns:
        float: rasio volatilitas (1.0 = normal, >1.2 = meningkat)
    """
    recent_std   = price_series.rolling(recent_days).std().iloc[-1]
    baseline_std = price_series.rolling(baseline_days).std().iloc[-1]

    if baseline_std is None or baseline_std == 0 or np.isnan(baseline_std):
        return 1.0

    return float(recent_std / baseline_std)


def calculate_trend_slope(price_series, window=7):
    """
    Hitung slope tren harga menggunakan linear regression.
    Slope positif = tren naik, negatif = tren turun.
    
    Returns:
        float: slope (Rp/hari), positif=naik, negatif=turun
        str: 'up' | 'down' | 'stable'
    """
    recent = price_series.dropna().tail(window)

    if len(recent) < 3:
        return 0.0, 'stable'

    x = np.arange(len(recent))
    y = recent.values
    slope, _, _, _, _ = stats.linregress(x, y)

    mean_price = y.mean()
    if mean_price == 0:
        return 0.0, 'stable'

    # Normalisasi slope sebagai % perubahan per hari terhadap rata-rata harga
    slope_pct = (slope / mean_price) * 100

    if slope_pct > 0.5:
        direction = 'up'
    elif slope_pct < -0.5:
        direction = 'down'
    else:
        direction = 'stable'

    return float(slope), direction


def get_seasonal_baseline(price_series, month):
    """
    Hitung rata-rata harga historis untuk bulan tertentu.
    Digunakan sebagai referensi seasonal agar tidak false alarm
    saat kenaikan harga adalah pola musiman normal.
    
    Returns:
        float: rata-rata harga untuk bulan tersebut, atau None jika data tidak cukup
    """
    seasonal_data = price_series[price_series.index.month == month]

    if len(seasonal_data) < 10:
        return None

    return float(seasonal_data.mean())

def forecast_deviation(forecast_value, moving_avg, rolling_std):
    """
    Hitung Forecast Deviation Index (FDI) — z-score bidireksional.
    Positif = forecast di atas rata-rata (risiko kenaikan harga).
    Negatif = forecast di bawah rata-rata (risiko penurunan harga).
    
    Returns:
        float: FDI value
    """
    if rolling_std is None or rolling_std == 0 or abs(rolling_std) < 1e-10:
        return 0.0

    return float((forecast_value - moving_avg) / rolling_std)

def risk_scoring(fdi, vol_ratio, trend_direction, seasonal_deviation=None):
    """
    Hitung risk score 0-100 dari beberapa faktor tertimbang.
    
    Komponen dan bobot:
      - FDI (forecast deviation)    : 40%
      - Volatility ratio            : 30%
      - Trend direction             : 20%
      - Seasonal deviation          : 10%

    Args:
        fdi               : float, z-score forecast vs moving average (positif/negatif)
        vol_ratio         : float, rasio volatilitas recent/baseline
        trend_direction   : str, 'up' | 'down' | 'stable'
        seasonal_deviation: float atau None, % deviasi dari seasonal baseline

    Returns:
        score : int (0-100)
        level : str 'Normal' | 'Waspada' | 'Bahaya'
        breakdown : dict komponen skor
    """
    score = 0.0

    # --- Komponen 1: FDI (40 poin maks) ---
    # abs(fdi) karena kenaikan DAN penurunan sama-sama berisiko
    fdi_abs = min(abs(fdi), 5.0)  # cap di 5 sigma
    fdi_score = (fdi_abs / 5.0) * 40

    # --- Komponen 2: Volatility ratio (30 poin maks) ---
    # vol_ratio 1.0 = normal (0 poin), 3.0+ = sangat tinggi (30 poin)
    vol_excess = max(vol_ratio - 1.0, 0.0)
    vol_score = min((vol_excess / 2.0) * 30, 30)

    # --- Komponen 3: Trend direction (20 poin maks) ---
    # Tren naik tajam saat harga sudah tinggi = berbahaya
    # Tren turun tajam = juga berbahaya (kerugian petani)
    if trend_direction in ('up', 'down'):
        trend_score = 20
    else:
        trend_score = 0

    # Kurangi skor tren jika arahnya berlawanan dengan FDI
    # (tren turun tapi forecast naik = sinyal mixed, kurang yakin)
    if trend_direction == 'down' and fdi > 0:
        trend_score *= 0.5
    elif trend_direction == 'up' and fdi < 0:
        trend_score *= 0.5

    # --- Komponen 4: Seasonal deviation (10 poin maks) ---
    seasonal_score = 0
    if seasonal_deviation is not None:
        # seasonal_deviation dalam % — > 20% di luar normal musiman
        seasonal_score = min((abs(seasonal_deviation) / 30.0) * 10, 10)

    score = fdi_score + vol_score + trend_score + seasonal_score
    score = int(min(round(score), 100))

    # Mapping skor ke level
    if score < 30:
        level = 'Normal'
    elif score < 60:
        level = 'Waspada'
    else:
        level = 'Bahaya'

    breakdown = {
        'fdi_score'      : round(fdi_score, 1),
        'vol_score'      : round(vol_score, 1),
        'trend_score'    : round(trend_score, 1),
        'seasonal_score' : round(seasonal_score, 1),
        'total'          : score
    }

    return score, level, breakdown

def generate_risk_message(fdi, vol_ratio, trend_direction, level, horizon):
    """
    Buat kalimat interpretasi otomatis dari kondisi risiko.
    
    Returns:
        str: kalimat deskriptif untuk ditampilkan di dashboard
    """
    horizon_label = {
        'short': '30 hari ke depan',
        'mid'  : '6 bulan ke depan',
        'long' : '12 bulan ke depan'
    }.get(horizon, horizon)

    parts = []

    # Deskripsi FDI
    fdi_abs = abs(fdi)
    if fdi_abs < 0.5:
        parts.append(f"Forecast {horizon_label} berada dalam rentang normal")
    elif fdi > 0:
        pct = round((fdi_abs / 1.0) * 15)  # estimasi kasar % deviasi
        parts.append(f"Forecast {horizon_label} berpotensi {pct}% di atas rata-rata 30 hari")
    else:
        pct = round((fdi_abs / 1.0) * 15)
        parts.append(f"Forecast {horizon_label} berpotensi {pct}% di bawah rata-rata 30 hari")

    # Deskripsi volatilitas
    if vol_ratio >= 2.0:
        parts.append(f"volatilitas saat ini {round(vol_ratio, 1)}x lebih tinggi dari normal")
    elif vol_ratio >= 1.3:
        parts.append(f"volatilitas mulai meningkat ({round(vol_ratio, 1)}x dari normal)")

    # Deskripsi tren
    if trend_direction == 'up':
        parts.append("tren harga cenderung naik dalam 7 hari terakhir")
    elif trend_direction == 'down':
        parts.append("tren harga cenderung turun dalam 7 hari terakhir")

    if not parts:
        return "Kondisi harga stabil dan dalam batas normal."

    message = parts[0]
    if len(parts) > 1:
        message += " — " + ", ".join(parts[1:])
    message += "."

    return message