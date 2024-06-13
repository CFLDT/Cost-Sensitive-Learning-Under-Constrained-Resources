import numpy as np
import pandas as pd
from pathlib import Path
from data.python.data_creater_FUNCTIONS import company_codes_merger_cik

base_path = Path(__file__).parent

path = (base_path / "../original_data/COMPUSTAT_Controls.csv").resolve()
df_controls = pd.read_csv(path)

df_controls["datadate"] = pd.to_datetime(df_controls["datadate"], infer_datetime_format=True).dt.year
df_controls = df_controls.drop('fyear', axis=1)

df_controls = df_controls.rename(columns={"datadate": "Year", 'cik': 'CIK', 'sic': 'SIC', 'at': 'Total_assets'})
df_controls['CIK'] = df_controls['CIK'].astype('Int64')
df_controls['Year'] = df_controls['Year'].astype('Int64')
df_controls = df_controls.loc[df_controls['CIK'].notna(), :]
df_controls = df_controls.loc[df_controls['Year'].notna(), :]

df_controls = df_controls[['Year', 'CIK', 'gsector', 'Total_assets']]
df_controls['Total_assets'] = np.log(df_controls['Total_assets'] +1)

# keep the duplicate row with fewest nan values
df_controls = df_controls.assign(counts=df_controls.count(axis=1)).sort_values(['counts']).drop_duplicates(
    ['Year', 'CIK'], keep='last').drop(
    'counts', axis=1)
df_controls = df_controls.sort_values(by=['CIK', 'Year']).reset_index(drop=True)


# Global Industry Classification Standard
# firm industry
df_controls.loc[10 == df_controls['gsector'], 'Industry'] = 'Energy'

df_controls.loc[15 == df_controls['gsector'], 'Industry'] = 'Materials'

df_controls.loc[20 == df_controls['gsector'], 'Industry'] = 'Industrials'

df_controls.loc[25 == df_controls['gsector'], 'Industry'] = 'Consumer Discretionary'

df_controls.loc[30 == df_controls['gsector'], 'Industry'] = 'Consumer Staples'

df_controls.loc[35 == df_controls['gsector'], 'Industry'] = 'Health Care'

df_controls.loc[40 == df_controls['gsector'], 'Industry'] = 'Financials'

df_controls.loc[45 == df_controls['gsector'], 'Industry'] = 'Information Technology'

df_controls.loc[50 == df_controls['gsector'], 'Industry'] = 'Communication Services'

df_controls.loc[55 == df_controls['gsector'], 'Industry'] = 'Utilities'

df_controls.loc[60 == df_controls['gsector'], 'Industry'] = 'Real Estate'

df_controls = df_controls.drop(['gsector'], axis=1)
df = company_codes_merger_cik(df_controls)

path = (base_path / "../csv/Controls.csv").resolve()
df.to_csv(path, index=True)
