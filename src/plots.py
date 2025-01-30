from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd

NUM_OF_REGIONS = 5

def plot_acf_pacf(df, path=None, lags=None):
    fig, ax = plt.subplots(2, NUM_OF_REGIONS, figsize=(20, 10))
    for i, region in enumerate(df.index):
        plot_acf(df.loc[region], lags=lags, ax=ax[0,i])
        plot_pacf(df.loc[region], lags=lags, ax=ax[1,i])
        ax[0,i].set_title(f"{region} ACF", fontsize=15)
        ax[1,i].set_title(f"{region} PACF", fontsize=15)
        ax[0,i].tick_params(axis='both', which='major', labelsize=12)
        ax[1,i].tick_params(axis='both', which='major', labelsize=12)

    if path is None:
        fig.show()
    else:
        fig.savefig(path)

def plot_time_series(df_bus, df_pop, base_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    region_colors = {}
    for each in df_bus.iterrows():
        region, values = each
        p = ax.plot(values.index, values.values, label=region)
        region_colors[region] = p[0].get_color()

    ax.set_title('Quantidade de empresas ativas por região')
    ax.set_xlabel('Ano')
    ax.set_ylabel('Quantidade')
    fig.legend()
    fig.savefig(base_path/'time_series_business.png')

    fig, ax = plt.subplots(figsize=(8, 5))


    for each in df_pop.iterrows():
        region, values = each
        ax.plot(values.index, values.values, label=region, c=region_colors[region])


    ax.set_title('Quantidade de pessoas na faixa de interesse por região')
    ax.set_xlabel('Ano')
    ax.set_ylabel('Quantidade')
    fig.legend()
    fig.savefig(base_path/'time_series_population.png')

    df_ratio = df_pop/df_bus

    fig, ax = plt.subplots(figsize=(8, 5))

    for each in df_ratio.iterrows():
        region, values = each
        ax.plot(values.index, values.values, label=region, c=region_colors[region])

    ax.set_title('Quantidade de pessoas na faixa de interesse por empresa ativa')
    ax.set_xlabel('Ano')
    ax.set_ylabel('Quantidade')
    fig.legend()
    fig.savefig(base_path/'time_series_ratio.png')

def plot_regions_forecast(df, df_true, arima_results, base_path):
    regions = df.index
    fig, ax = plt.subplots(1, figsize=(8, 5))
    df_dict = {}
    for i, region in enumerate(regions):
        forecast = plot_forecast(df, df_true, arima_results[region], region, ax)
        df_dict[region] = {2021: forecast[0], 2022: forecast[1]}

    ax.set_xlabel('Ano')
    ax.set_ylabel('Consumidores por empresa ativa')

    fig.legend()
    fig.savefig(base_path/f'forecast.png', dpi=300)
    return pd.DataFrame.from_dict(df_dict, orient='index')

def plot_forecast(df, df_true, arima_result, region, ax):
    start = 2020
    end = 2022

    forecast = arima_result.predict(start=start-2007,end=end-2007-1)
    p = ax.plot(np.arange(start,end+1), np.r_[df.loc[region].values[-1],
                forecast],
                ls='--')
    color = p[0].get_color()
    ax.plot(np.arange(start,end+1), np.r_[df.loc[region].values[-1],
                df_true.loc[region].values],
                ls='dashdot', c = color)
    ax.plot(df.columns, df.loc[region].values, c=color, label=region,)
    return forecast

