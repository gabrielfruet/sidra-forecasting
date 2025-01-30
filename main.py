from src import preprocess
from src import fetching
from src import validation
from src import plots
from src import training

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import pyplot as plt
from pathlib import Path

import numpy as np

FIGURES = Path('./figures')

def plot_fourier(signal):
    fourier = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal))

    idx = np.argsort(freq)
    fourier = fourier[idx]
    freq = freq[idx]

    plt.plot(freq, np.abs(fourier**2))
    plt.show()

def get_best_nrmse(validation_metrics_df):
    df = validation_metrics_df
    df = df.loc[df.groupby("Region")["NRMSE"].idxmin()]
    df.index = df.Region
    df = df[["AR order", "I order", "MA order", "NRMSE"]] 
    return df.values
    

if __name__ == '__main__':
    # Fetching data
    df_bus_raw = fetching.fetch_business_data()
    df_pop_raw = fetching.fetch_population_data()
    df_bus_raw_true = fetching.fetch_business_data(period=(2021,2022))

    assert not isinstance(df_bus_raw, tuple), "fetch_business_data should return only the DataFrame"
    assert not isinstance(df_bus_raw_true, tuple), "fetch_business_data should return only the DataFrame"

    # Preprocess data
    df_bus, regions = preprocess.preprocess_business_data(df_bus_raw)
    df_bus_true, regions = preprocess.preprocess_business_data(df_bus_raw_true)
    df_pop = preprocess.preprocess_population_data(df_pop_raw, regions)
    df_pop_true = preprocess.preprocess_population_data(df_pop_raw, regions, period=(2021,2022))

    df_ratio = df_pop/df_bus
    df_ratio_true = df_pop_true/df_bus_true

    plots.plot_acf_pacf(df_ratio, path=FIGURES/"acf_pacf.png")
    plots.plot_time_series(df_bus, df_pop, FIGURES)

    # validation_metrics2 = validation.rolling_window_cv(
    #     df_ratio,
    #     [(1,0,0),(1,1,0),(1,1,1),(0,1,1),(0,0,1),(0,1,0)],
    #     test_train_proportion=0.7
    # )
    validation_metrics = validation.rolling_window_cv(df_ratio, [(1,0,0), (0,1,0)])

    results = training.train_for_each_region(df_ratio, {
        'Centro-Oeste': (1,0,0),
        'Nordeste':     (1,0,0),
        'Sudeste':      (1,0,0),
        'Norte':        (1,0,0),
        'Sul':          (0,1,1),
    })

    forecast = plots.plot_regions_forecast(df_ratio, df_ratio_true, results, FIGURES)
