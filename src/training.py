import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResultsWrapper

def train_for_each_region(df: pd.DataFrame, order: tuple[int,int,int]) -> dict[str, ARIMAResultsWrapper]:
    regions = df.index

    results = {}

    for region in regions:
        result = ARIMA(order).fit(df.loc[region])
        results[region] = result

    return results
