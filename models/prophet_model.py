import pandas as pd
from prophet import Prophet
import numpy as np
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def train_prophet(train_df, use_log=False):
    """
    Train Prophet model dengan opsi log transform.
    
    Args:
        train_df: DataFrame dengan kolom 'price' dan date index
                  ATAU DataFrame dengan kolom ['ds', 'y']
        use_log: bool, apply log transform sebelum training
                 Default=False untuk backward compatibility
    
    Returns:
        tuple: (model, use_log) untuk tracking
    
    Notes:
        Log transform di Prophet OPTIONAL (berbeda dengan ARIMA)
        - use_log=False: Raw price, seasonality_mode='additive'
        - use_log=True: Log price, seasonality_mode='multiplicative'
    """
    # Handle both input formats
    if 'ds' not in train_df.columns:
        # Input adalah DataFrame dengan date index dan 'price' column
        df = train_df.reset_index()
        df.columns = ['ds', 'y']
    else:
        # Input sudah dalam format ['ds', 'y']
        df = train_df[['ds', 'y']].copy()
    
    df = df.dropna()
    
    # Apply log transform jika diperlukan
    if use_log:
        df['y'] = np.log(df['y'] + 1)  # +1 untuk handle y=0
    
    # Create Prophet model (REMOVED invalid parameter)
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative' if use_log else 'additive',
        interval_width=0.95
    )
    
    # Fit model (suppress verbose output)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(df)
    
    return model, use_log


def forecast_prophet(model, periods, use_log=False):
    """
    Forecast future periods menggunakan Prophet.
    
    Args:
        model: Trained Prophet model
        periods: Jumlah periode forecast (dalam hari)
        use_log: bool, True jika model dilatih dengan log transform
    
    Returns:
        DataFrame dengan kolom [ds, yhat, yhat_lower, yhat_upper]
        Nilai sudah di-inverse jika use_log=True (dalam unit Rp)
    """
    future = model.make_future_dataframe(periods=periods)
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        forecast = model.predict(future)
    
    result = forecast.tail(periods)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    
    # Inverse transform jika training dengan log
    if use_log:
        result['yhat'] = np.exp(result['yhat']) - 1
        result['yhat_lower'] = np.exp(result['yhat_lower']) - 1
        result['yhat_upper'] = np.exp(result['yhat_upper']) - 1
    
    return result


def evaluate_forecast(actual, predicted):
    """
    Calculate RMSE and MAPE
    
    Args:
        actual: numpy array of actual values
        predicted: numpy array of predicted values
    
    Returns:
        tuple: (rmse, mape)
    """
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    # Calculate MAPE
    mask = actual != 0
    if len(actual[mask]) == 0:
        mape = np.nan
    else:
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    
    return rmse, mape