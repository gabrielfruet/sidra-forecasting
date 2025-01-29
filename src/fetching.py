import requests
import wget
import pandas as pd
from pathlib import Path
import json

POPULATION_DATA_URL = "https://ftp.ibge.gov.br/Projecao_da_Populacao/Projecao_da_Populacao_2024/projecoes_2024_tab1_idade_simples.xlsx"
POPULATION_FNAME = Path("projecoes_2024_tab1_idade_simples.xlsx")

def fetch_business_data(region='n2', return_description=False) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    http_response = requests.get(f"https://apisidra.ibge.gov.br/values/t/1757/{region}/all/p/2007-2020")
    response = json.loads(http_response.content)
    description = response.pop(0)
    df = pd.DataFrame(response)
    if return_description:
        return df, description
    else:
        return df

def fetch_population_data(regions):
    if not POPULATION_FNAME.exists():
        wget.download(POPULATION_DATA_URL,POPULATION_FNAME)
    df = pd.read_excel(POPULATION_FNAME, skiprows=5)
    pass
