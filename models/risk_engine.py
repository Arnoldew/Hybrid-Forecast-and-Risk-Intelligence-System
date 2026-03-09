import numpy as np


def calculate_volatility(price_series, window=30):
    return price_series.rolling(window).std()


def forecast_deviation(forecast_value, moving_avg, rolling_std):
    """Calculate Forecast Deviation Index (FDI)
    
    Args:
        forecast_value: The forecasted price
        moving_avg: 30-day moving average
        rolling_std: 30-day rolling standard deviation
        
    Returns:
        FDI value (z-score)
    """
    # Handle edge case where rolling_std is 0 or very small
    if rolling_std is None or rolling_std == 0 or abs(rolling_std) < 1e-10:
        return 0 
    return (forecast_value - moving_avg) / rolling_std


def risk_scoring(fdi, volatility_flag, ci_breach):
    score = 0

    if fdi >= 1:
        score += 1

    if volatility_flag:
        score += 1

    if ci_breach:
        score += 2

    if score == 0:
        return score, "Normal"
    elif score == 1:
        return score, "Waspada"
    else:
        return score, "Bahaya"