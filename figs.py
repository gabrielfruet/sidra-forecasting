from src import preprocess
from src import fetching
from src import validation
from src import plots

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import pyplot as plt
from pathlib import Path

FIGURES = Path('./figures')

if __name__ == '__main__':
    # Fetching data
    df_bus_raw = fetching.fetch_business_data()
    df_pop_raw = fetching.fetch_population_data()

    assert not isinstance(df_bus_raw, tuple), "fetch_business_data should return only the DataFrame"

    # Preprocess data
    df_bus, regions = preprocess.preprocess_business_data(df_bus_raw)
    df_pop = preprocess.preprocess_population_data(df_pop_raw, regions)

    df_ratio = df_pop/df_bus

    plots.plot_acf_pacf(df_ratio, path=FIGURES/"acf_pacf.png")
    plots.plot_time_series(df_bus, df_pop, FIGURES)

    result = validation.rolling_window_cv(df_ratio, [(1,0,0),(1,1,0),(1,1,1),(0,1,1),(0,0,1),(0,1,0)])
