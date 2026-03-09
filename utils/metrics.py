import numpy as np
from sklearn.metrics import mean_squared_error


def calculate_mape(actual, predicted):

    actual = np.array(actual)
    predicted = np.array(predicted)

    # Hapus nilai actual == 0
    mask = actual != 0

    actual = actual[mask]
    predicted = predicted[mask]

    return np.mean(np.abs((actual - predicted) / actual)) * 100


def calculate_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))