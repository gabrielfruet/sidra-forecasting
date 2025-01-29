import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np


def rolling_window_cv(df: pd.DataFrame, order: tuple[int,int,int] | list[tuple[int,int,int]], test_train_proportion: float = 0.7) -> pd.DataFrame:
    """
    Performs rolling window cross-validation for an ARIMA model on a DataFrame of time series data.

    Args:
        df: DataFrame containing time series data.
        order: The order of the ARIMA model (p,d,q).
        test_train_proportion: The proportion of data to use for training.

    Returns:
        pd.DataFrame: DataFrame containing the RMSE, AR order, I order, MA order, and Test Train Proportion for each region.
    """

    if isinstance(order, tuple) and isinstance(order[0], int):
        orders: list[tuple[int,int,int]] = [order]
    elif isinstance(order, list):
        orders = order
    else:
        raise ValueError('order must be a tuple or list of tuples')

    regions = df.index
    results = pd.DataFrame({
        'RMSE': [],
        'AR order': [],
        'I order': [],
        'MA order': [],
        'Test Train Proportion': [],
        'Region': []
    })

    for arima_order in orders:
        ar_order, i_order, ma_order = arima_order
        for region in regions:
            train_size = int(len(df.columns) * test_train_proportion)
            df.columns = [int(col) for col in df.columns]
            ts = df.loc[region].values
            rolling_predictions = []
            test_actuals = ts[train_size:]

            for i in range(train_size, len(ts)):

                train = ts[:i]

                model = ARIMA(train, order=arima_order)
                fitted_model = model.fit()

                forecast = fitted_model.forecast(steps=1)
                rolling_predictions.append(forecast[0])

            rmse = np.sqrt(mean_squared_error(test_actuals, rolling_predictions))
            results.loc[len(results)] = (rmse, ar_order, i_order, ma_order, test_train_proportion, region)

    return results

