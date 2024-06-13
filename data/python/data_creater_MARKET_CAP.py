import numpy as np
import pandas as pd
from cpi import inflate
from pathlib import Path
from data.python.data_creater_FUNCTIONS import company_codes_merger_cik

base_path = Path(__file__).parent

# Market capitalization

path = (base_path / "../original_data/COMPUSTAT_Dechow_1.csv").resolve()
df_mark_cap = pd.read_csv(path, )

df_mark_cap["datadate"] = pd.to_datetime(df_mark_cap["datadate"], infer_datetime_format=True).dt.year
df_mark_cap = df_mark_cap.drop('fyear', axis=1)

df_mark_cap = df_mark_cap.rename(columns={"datadate": "Year", "csho": "CSHO", "prcc_c": "PRCC_C",
                                          'cik': 'CIK'})
df_mark_cap = df_mark_cap[['Year', 'CSHO', 'PRCC_C', 'CIK']]

df_mark_cap['CIK'] = df_mark_cap['CIK'].astype('Int64')
df_mark_cap['Year'] = df_mark_cap['Year'].astype('Int64')
df_mark_cap = df_mark_cap.loc[df_mark_cap['CIK'].notna(), :]
df_mark_cap = df_mark_cap.loc[df_mark_cap['Year'].notna(), :]

# keep the duplicate row with fewest nan values
df_mark_cap = df_mark_cap.assign(counts=df_mark_cap.count(axis=1)).sort_values(['counts']). \
    drop_duplicates(['Year', 'CIK'], keep='last').drop('counts', axis=1)
df_mark_cap = df_mark_cap.sort_values(by=['CIK', 'Year']).reset_index(drop=True)

df_mark_cap = company_codes_merger_cik(df_mark_cap)

df_mark_cap['Market_cap'] = df_mark_cap['CSHO'] * df_mark_cap['PRCC_C']


def inflates(row):
    return inflate(row['Market_cap'], int(row['Year'])) / inflate(1, 2016)



df_mark_cap['Market_cap_2016'] = df_mark_cap.apply(inflates, axis=1)
df_mark_cap['Market_cap_2016'] = df_mark_cap['Market_cap_2016'].where(df_mark_cap['Market_cap_2016'].notnull(),
                                                                      np.nan).astype(np.float64)

df_mark_cap['Log_market_cap'] = np.log(df_mark_cap['Market_cap'] + 1)
df_mark_cap['Log_market_cap_2016'] = np.log(df_mark_cap['Market_cap_2016'] + 1)

df_mark_cap = df_mark_cap[['Year', 'CIK', 'Market_cap', 'Market_cap_2016', 'Log_market_cap', 'Log_market_cap_2016']]

groupby_obj = df_mark_cap[['Year', 'Market_cap_2016']].groupby(['Year']).sum().reset_index()
groupby_obj = groupby_obj.rename(columns={"Market_cap_2016": "Market_cap_all_2016"})

df_mark_cap = pd.merge(df_mark_cap, groupby_obj, how='left', on=['Year'])
df_mark_cap['Market_cap_all_loss_2016'] = df_mark_cap['Market_cap_all_2016'] * 0.00119
df_mark_cap['Market_cap_1_per_loss_2016'] = df_mark_cap['Market_cap_2016'] * 0.01
df_mark_cap['Market_cap_3_per_loss_2016'] = df_mark_cap['Market_cap_2016'] * 0.03
df_mark_cap['Market_cap_5_per_loss_2016'] = df_mark_cap['Market_cap_2016'] * 0.05
df_mark_cap['Market_cap_15_per_loss_2016'] = df_mark_cap['Market_cap_2016'] * 0.15

df = company_codes_merger_cik(df_mark_cap)

path = (base_path / "../csv/Market_Cap.csv").resolve()
df.to_csv(path, index=True)
