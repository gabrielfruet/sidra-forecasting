import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResultsWrapper

def train_for_each_region(df: pd.DataFrame, order: dict[str,tuple[int,int,int]]) -> dict[str, ARIMAResultsWrapper]:
    regions = df.index

    results = {}

    for region in regions:
        arima_order = order[region]
        print(f"Training ARIMA model for {region} with order {arima_order}")
        result = ARIMA(df.loc[region].values, order=arima_order).fit()
        results[region] = result

    return results
