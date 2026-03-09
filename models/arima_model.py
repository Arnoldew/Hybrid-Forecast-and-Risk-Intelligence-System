import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error


def train_arima(train_df):
    """
    Train ARIMA model using log transformation
    """
    train = train_df.copy()
    train = train.asfreq('D')
    train['price'] = train['price'].replace(0, np.nan)
    train['price'] = train['price'].interpolate(method='linear')
    train = train.dropna()
    train['log_price'] = np.log(train['price'])

    # Constrain auto_arima search space to avoid high memory usage in Kalman filter/smoother
    auto_model = auto_arima(
        train['log_price'],
        seasonal=True,  # keep weekly seasonality if data is daily
        m=7,
        start_p=0,
        start_q=0,
        max_p=3,
        max_q=3,
        max_d=2,
        start_P=0,
        start_Q=0,
        max_P=1,
        max_Q=1,
        max_D=1,
        max_order=6,  # limit total order
        stepwise=True,  # faster, less memory
        n_fits=25,
        information_criterion='aicc',
        seasonal_test='ocsb',
        stationary=False,
        trace=False,
        error_action='ignore',
        suppress_warnings=True
    )

    # Use simple filter method to reduce memory; disable unnecessary outputs during fit
    model = SARIMAX(
        train['log_price'],
        order=auto_model.order,
        seasonal_order=auto_model.seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    model_fit = model.fit(disp=False, low_memory=True)

    return model_fit


def forecast_arima(model_fit, steps):
    """
    Forecast future values using trained ARIMA model
    """
    forecast_log = model_fit.forecast(steps=steps)
    forecast_price = np.exp(forecast_log)

    return forecast_price


def evaluate_forecast(actual, predicted):
    """
    Calculate RMSE and MAPE
    """
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    return rmse, mape