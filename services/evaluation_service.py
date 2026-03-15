import numpy as np 

def calculate_mape(actual, forecast):

    """Calculate MAPE (Mean Absolute Percentage Error)"""
    actual = np.array(actual)
    forecast = np.array(forecast)   

    mask = actual != 0

    mape = np.mean(
        np.abs((actual[mask] - forecast[mask]) / actual[mask])
    ) * 100

    return round(mape,2)