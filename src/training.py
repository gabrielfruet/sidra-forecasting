import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResultsWrapper

def train_for_each_region(df: pd.DataFrame, order: tuple[int,int,int]) -> dict[str, ARIMAResultsWrapper]:
    regions = df.index

    results = {}

    for region in regions:
        print(f"Training ARIMA model for {region}")
        print(df.loc[region].values)
        result = ARIMA(df.loc[region].values, order=order).fit()
        print(result.forecast())
        results[region] = result

    return results
