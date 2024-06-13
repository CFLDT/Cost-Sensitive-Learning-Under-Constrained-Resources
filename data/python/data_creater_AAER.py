import numpy as np
import pandas as pd
from pathlib import Path
from data.python.data_creater_FUNCTIONS import company_codes_merger_cik

base_path = Path(__file__).parent

# AAER Database from CFRM. Updated until December 31st 2021

path = (base_path / "../original_data/CFRM_AAER_All.xlsx").resolve()
xls = pd.ExcelFile(path)

df_AAER_firm_year = pd.read_excel(xls, 'ann')

# P_AAER: The p_aaer number is the AAER number that matches the SEC issued number.
# rename --> AAER_ID

df_AAER_firm_year = df_AAER_firm_year.rename(columns={"P_AAER": "AAER_ID", "YEAR": "Year"})
df_AAER_firm_year['Year'] = df_AAER_firm_year['Year'].astype('Int64')
df_AAER_firm_year['CIK'] = df_AAER_firm_year['CIK'].astype('Int64')
df_AAER_firm_year = df_AAER_firm_year[['AAER_ID', 'Year', 'CIK']]
df_AAER_firm_year = df_AAER_firm_year.dropna()
df_AAER_firm_year = df_AAER_firm_year.sort_values(by=['CIK', 'AAER_ID']).reset_index(drop=True)

#If multiple AAERS affect the same firm in the same year, we keep the one with lowest AAER_ID (the oldest)
df_AAER_firm_year = df_AAER_firm_year.drop_duplicates(subset=['CIK', 'AAER_ID', 'Year'], keep='first')

df_AAER_firm_year_first = df_AAER_firm_year.copy()
df_AAER_firm_year_first = df_AAER_firm_year_first.rename(columns={"AAER_ID": "AAER_ID_first"})
df_AAER_firm_year_first = df_AAER_firm_year_first.drop_duplicates(subset=['AAER_ID_first'], keep='first')

# Merge with compustat data

df = company_codes_merger_cik(df_AAER_firm_year)
df_first = company_codes_merger_cik(df_AAER_firm_year_first)
df = pd.merge(df, df_first, how='left', on=['Year', 'CIK'])

df['AAER'] = np.where(df['AAER_ID'].isnull(), 0, 1)
df['AAER_first'] = np.where(df['AAER_ID_first'].isnull(), 0, 1)

print('The number of original AAERs: ' + str(df_AAER_firm_year.shape[0]))
print('The number of AAERs merged using CIK: ' + str(df[df['AAER'] == 1].shape[0]))
print('The number of first AAERs merged using CIK: ' + str(df[df['AAER_first'] == 1].shape[0]))
print(
    'The number of distinct AAER ID: original AAER: ' + str(df_AAER_firm_year['AAER_ID'].drop_duplicates().shape[0]))
print('The number of distinct AAER firms: merged using CIK: ' + str(
    df['AAER_ID'][df['AAER'] == 1].drop_duplicates().shape[0]))


path = (base_path / "../csv/AAER_Data.csv").resolve()
df.to_csv(path, index=True)
