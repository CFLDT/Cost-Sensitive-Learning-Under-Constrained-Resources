import numpy as np
import pandas as pd
from pathlib import Path
from data.python.data_creater_FUNCTIONS import company_codes_merger_cik

# M-score

# code similar to
# https://medium.com/analytics-vidhya/earning-manipulation-detection-hey-why-not-try-my-m-score-generator-f8a7895743e0#:~:text=Professor%20Beneish%20used%20Compustat%20database,single%20company's%20possible%20earnings%20manipulation.


base_path = Path(__file__).parent

path = (base_path / "../original_data/COMPUSTAT_Beneish.csv").resolve()
df_ben = pd.read_csv(path)

df_ben["datadate"] = pd.to_datetime(df_ben["datadate"], infer_datetime_format=True).dt.year
df_ben = df_ben.drop('fyear', axis=1)

df_ben = df_ben.rename(columns={"datadate": "Year", 'cik': 'CIK'})
df_ben['CIK'] = df_ben['CIK'].astype('Int64')
df_ben['Year'] = df_ben['Year'].astype('Int64')
df_ben = df_ben.loc[df_ben['CIK'].notna(), :]
df_ben = df_ben.loc[df_ben['Year'].notna(), :]
df_ben = df_ben.drop(['gvkey', 'consol', 'datafmt', 'popsrc', 'indfmt', 'curcd', 'costat'], axis=1)

# keep the duplicate row with fewest nan values
df_ben = df_ben.assign(counts=df_ben.count(axis=1)).sort_values(['counts']).drop_duplicates(['Year', 'CIK'],
                                                                                            keep='last').drop('counts',
                                                                                                              axis=1)
df_ben = df_ben.sort_values(by=['CIK', 'Year']).reset_index(drop=True)

df_ben['pre_sale'] = df_ben.groupby('CIK')['sale'].shift()
df_ben['pre_cogs'] = df_ben.groupby('CIK')['cogs'].shift()
df_ben['pre_rect'] = df_ben.groupby('CIK')['rect'].shift()
df_ben['pre_act'] = df_ben.groupby('CIK')['act'].shift()
df_ben['pre_ppegt'] = df_ben.groupby('CIK')['ppegt'].shift()
df_ben['pre_dp'] = df_ben.groupby('CIK')['dp'].shift()
df_ben['pre_at'] = df_ben.groupby('CIK')['at'].shift()
df_ben['pre_xsga'] = df_ben.groupby('CIK')['xsga'].shift()
df_ben['pre_ni'] = df_ben.groupby('CIK')['ni'].shift()
df_ben['pre_oancf'] = df_ben.groupby('CIK')['oancf'].shift()
df_ben['pre_lct'] = df_ben.groupby('CIK')['lct'].shift()
df_ben['pre_dltt'] = df_ben.groupby('CIK')['dltt'].shift()

# For later calculation
df_ben['asset_qual'] = (df_ben['at'] - df_ben['act'] - df_ben['ppegt']) / df_ben['at']
df_ben['pre_asset_qual'] = (df_ben['pre_at'] - df_ben['pre_act'] - df_ben['pre_ppegt']) / df_ben['pre_at']
# Eight ratios
df_ben['DSRI'] = (df_ben['rect'] / df_ben['sale']) / (df_ben['pre_rect'] / df_ben['pre_sale'])
df_ben['GMI'] = ((df_ben['pre_sale'] - df_ben['pre_cogs']) / df_ben['pre_sale']) / (
            (df_ben['sale'] - df_ben['cogs']) / df_ben['sale'])
df_ben['AQI'] = df_ben['asset_qual'] / df_ben['pre_asset_qual']
df_ben['SGI'] = df_ben['sale'] / df_ben['pre_sale']
df_ben['DEPI'] = (df_ben['pre_dp'] / (df_ben['pre_dp'] + df_ben['pre_ppegt'])) / (
            df_ben['dp'] / (df_ben['dp'] + df_ben['ppegt']))
df_ben['SGAI'] = (df_ben['xsga'] / df_ben['sale']) / (df_ben['pre_xsga'] / df_ben['pre_sale'])
df_ben['TATA'] = (df_ben['ni'] - df_ben['oancf']) / df_ben['at']
df_ben['LVGI'] = ((df_ben['lct'] + df_ben['dltt']) / df_ben['at']) / (
            (df_ben['pre_lct'] + df_ben['pre_dltt']) / df_ben['pre_at'])


df_ben = df_ben[['DSRI','GMI', 'AQI','SGI','DEPI','SGAI','TATA','LVGI','Year', 'CIK']]
df_ben.replace([np.inf, -np.inf], np.nan, inplace=True)

df = company_codes_merger_cik(df_ben)

df['M_score'] = -4.84 + .920 * df['DSRI'].fillna(df['DSRI'].median()) + .528 * df['GMI'].fillna(
    df['GMI'].median()) \
                    + .404 * df['AQI'].fillna(df['AQI'].median()) + \
                    .892 * df['SGI'].fillna(df['SGI'].median()) + .115 * df['DEPI'].fillna(
    df['DEPI'].median()) \
                    - .172 * df['SGAI'].fillna(df['SGAI'].median()) + 4.679 * \
                    df['TATA'].fillna(df['TATA'].median()) - .327 * df['LVGI'].fillna(
    df['LVGI'].median())

df = df[['M_score', 'Year', 'CIK']]

path = (base_path / "../csv/Beneish.csv").resolve()
df.to_csv(path, index=True)
