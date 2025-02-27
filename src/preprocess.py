from typing import Iterable
import pandas as pd

CODE_ACTIVE_BUSINESS = '410'

def preprocess_business_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess the business data to be used in the analysis.
    The final format should be as follow:
        
    Região   | 2007 | ... | 2020
    ----------------------------
    Nordeste | 123  | ... | 8746
    ...
    Sudeste | 347  | ... | 9387

    args:
        df: DataFrame with the raw business data from SIDRA
    return:
        df: DataFrame with the preprocessed business data

    """

    global CODE_ACTIVE_BUSINESS

    df = df.query('`D3C` == @CODE_ACTIVE_BUSINESS') # Filter only open businesses
    df = pd.DataFrame(df[['V', 'D2N', 'D1N']]) # Select only value, region and year
    renaming = {
        'V': 'Quantidade',
        'D2N': 'Ano',
        'D1N': 'Região'
    }
    df = df.rename(columns=renaming)
    regions = df['Região'].unique()
    df_dict = {region: dict() for region in regions}

    for each in df.iterrows():
        _, values = each
        df_dict[values['Região']][values['Ano']] = int(values['Quantidade'])

    df = pd.DataFrame.from_dict(df_dict, orient='index')
    df.index.name = 'Região'
    df.columns = [int(col) for col in df.columns if str(col).isdigit()]

    return df, pd.Series(regions)

def preprocess_population_data(df: pd.DataFrame,
                               regions: Iterable[str],
                               period: tuple[int,int]=(2007,2020)):
    """
    Preprocess the population data to be used in the analysis.
    The final format should be as follow:
        
    Região   | 2007 | ... | 2020
    ----------------------------
    Nordeste | 123  | ... | 8746
    ...
    Sudeste | 347  | ... | 9387

    args:
        df: DataFrame with the raw population data from IBGE
    return:
        df: DataFrame with the preprocessed business data

    """
    df = df.query("`IDADE`>38 and `IDADE`<58 and `SEXO` == 'Ambos' and `LOCAL` in @regions")
    df = df.drop(['SEXO', 'SIGLA', 'CÓD.'], axis=1)

    _df = df.groupby('LOCAL').sum()
    assert isinstance(_df, pd.DataFrame), "Groupby sum should return a DataFrame"
    df = _df
    df = df.drop(['IDADE'], axis=1)
    start,end = period
    df = pd.DataFrame(df[[col for col in df.columns if str(col).isdigit() and start <= int(col) <= end]])
    df.index.name = 'Região'
    df.columns = [int(col) for col in df.columns if str(col).isdigit()]

    return df
