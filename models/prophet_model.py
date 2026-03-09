import pandas as pd
from prophet import Prophet
import numpy as np
from sklearn.metrics import mean_squared_error


def train_prophet(train_df):
    """
    Train Prophet model
    """
    df = train_df.copy()
    df = df.reset_index()
    df = df.rename(columns={'date': 'ds', 'price': 'y'})

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        # changepoint_prior_scale=0.2,
        # seasonality_prior_scale=15
    )

    model.fit(df)

    return model


def forecast_prophet(model, periods):
    """
    Forecast future periods - returns only the future forecast, not historical
    """
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    # Return only the future forecast (last 'periods' rows)
    return forecast.tail(periods)


def evaluate_forecast(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    return rmse, mape
