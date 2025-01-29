from src import preprocess
from src import fetching
from src import validation

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import pyplot as plt
from pathlib import Path

FIGURES = Path('./figures')

NUM_OF_REGIONS = 5

def plot_acf_pacf(df, path=None, lags=None):
    fig, ax = plt.subplots(2, NUM_OF_REGIONS, figsize=(20, 10))
    for i, region in enumerate(df.index):
        plot_acf(df.loc[region], lags=lags, ax=ax[0,i])
        plot_pacf(df.loc[region], lags=lags, ax=ax[1,i])
        ax[0,i].set_title(f"{region} ACF")
        ax[1,i].set_title(f"{region} PACF")

    if path is None:
        fig.show()
    else:
        fig.savefig(path)

def plot_time_series(df_bus, df_pop, path):
    fig, ax = plt.subplots(figsize=(8, 5))

    for each in df_bus.iterrows():
      region, values = each
      ax.plot(values.index, values.values, label=region)

    ax.set_title('Quantidade de empresas ativas por região')
    ax.set_xlabel('Ano')
    ax.set_ylabel('Quantidade')
    fig.legend()
    fig.savefig(path/'time_series_business.png')

    fig, ax = plt.subplots(figsize=(8, 5))

    for each in df_pop.iterrows():
      region, values = each
      ax.plot(values.index, values.values, label=region)

    ax.set_title('Quantidade de pessoas na faixa de interesse por região')
    ax.set_xlabel('Ano')
    ax.set_ylabel('Quantidade')
    fig.legend()
    fig.savefig(path/'time_series_population.png')



if __name__ == '__main__':
    # Fetching data
    df_bus_raw = fetching.fetch_business_data()
    df_pop_raw = fetching.fetch_population_data()

    assert not isinstance(df_bus_raw, tuple), "fetch_business_data should return only the DataFrame"

    # Preprocess data
    df_bus, regions = preprocess.preprocess_business_data(df_bus_raw)
    df_pop = preprocess.preprocess_population_data(df_pop_raw, regions)

    df_ratio = df_pop/df_bus

    plot_acf_pacf(df_ratio, path=FIGURES/"acf_pacf.png")
    plot_time_series(df_bus, df_pop, FIGURES)

     result = validation.rolling_window_cv(df_ratio, [(1,0,0),(1,1,0),(1,1,1),(0,1,1),(0,0,1),(0,1,0)])
