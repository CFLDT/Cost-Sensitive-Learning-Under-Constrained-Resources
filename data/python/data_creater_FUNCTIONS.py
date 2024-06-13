import pandas as pd
from pathlib import Path
import numpy as np
import warnings

warnings.filterwarnings(action='ignore', message='Mean of empty slice')
base_path = Path(__file__).parent

def company_codes_merger_cik(df):
    path = (base_path / "../original_data/COMPUSTAT_company_codes.csv").resolve()
    df_company_indicators = pd.read_csv(path)

    df_company_indicators["datadate"] = pd.to_datetime(df_company_indicators["datadate"],
                                                       infer_datetime_format=True).dt.year
    df_company_indicators = df_company_indicators.drop('fyear', axis=1)
    df_company_indicators = df_company_indicators.rename(columns={"datadate": "Year", 'cik': 'CIK'})
    df_company_indicators['CIK'] = df_company_indicators['CIK'].astype('Int64')
    df_company_indicators['Year'] = df_company_indicators['Year'].astype('Int64')
    df_company_indicators = df_company_indicators[df_company_indicators['Year'] > 1989]
    df_company_indicators = df_company_indicators[df_company_indicators['Year'] < 2022]
    df_company_indicators = df_company_indicators[df_company_indicators['loc'] == 'USA']
    df_company_indicators = df_company_indicators.drop(['gvkey', 'consol', 'popsrc',
                                                        'datafmt', 'indfmt', 'conm', 'curcd', 'costat', 'tic', 'cusip',
                                                        'fyr', 'loc', 'exchg', 'fic'], axis=1)
    df_company_indicators = df_company_indicators.dropna()
    df_company_indicators = df_company_indicators.drop_duplicates(['CIK', 'Year'])
    df_company_indicators = df_company_indicators.sort_values(by=['CIK', 'Year']).reset_index(drop=True)

    df = pd.merge(df_company_indicators, df, how='left', on=['Year', 'CIK'])

    return df