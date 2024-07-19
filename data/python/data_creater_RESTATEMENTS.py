import numpy as np
import pandas as pd
from pathlib import Path
from data.python.data_creater_FUNCTIONS import company_codes_merger_cik, company_codes_merger_cik_2

base_path = Path(__file__).parent

# Restatements
# Restatement file data from 1995
# Actual restatement data from 1983
# Material restatement file data from 2004

path = (base_path / "../original_data/AUDIT_ANALYTICS_Restatements.csv").resolve()
df_restatements = pd.read_csv(path, encoding='latin-1', index_col=False, low_memory=False)

df_restatements = df_restatements.rename(columns={'CIK Code': 'CIK'})

# keep the duplicate row with fewest nan values

df_restatements = df_restatements.assign(counts=df_restatements.count(axis=1)).sort_values(['counts']). \
    drop_duplicates(['Restatement Key'], keep='last').drop('counts', axis=1)
df_restatements = df_restatements.sort_values(by=['Restatement Key']).reset_index(drop=True)

# Disclosure date: Date on which the filing was submitted to the SEC.

df_restatements["Restatement_begin_year_year"] = pd.to_datetime(df_restatements["Restated Period Begin"],
                                                                infer_datetime_format=True).dt.year
df_restatements["Restatement_end_year_year"] = pd.to_datetime(df_restatements["Restated Period Ended"],
                                                              infer_datetime_format=True).dt.year
df_restatements["Year_8K_402"] = pd.to_datetime(df_restatements["Date of 8-K Item 4.02"],
                                                infer_datetime_format=True).dt.year

df_restatements = df_restatements.dropna(subset=['Restatement_begin_year_year', 'Restatement_end_year_year'], axis=0)

# We create additional rows to capture all years that contain restatement information
date_range = lambda x: range(int(x['Restatement_begin_year_year']),
                             int(x['Restatement_end_year_year'] + 1))
df_restatements = df_restatements.assign(Year=df_restatements.apply(date_range, axis=1)).explode('Year',
                                                                                                 ignore_index=True)

df_restatements['CIK'] = df_restatements['CIK'].astype('Int64')
df_restatements['Year'] = df_restatements['Year'].astype('Int64')
df_restatements = df_restatements.loc[df_restatements['CIK'].notna(), :]
df_restatements = df_restatements.loc[df_restatements['Year'].notna(), :]

df_restatements = df_restatements.rename(columns={"Restated Period Begin": "Begin_date",
                                                  "Restated Period Ended": "End_date"})

df_restatements = df_restatements[['Year', 'CIK', 'Year_8K_402', 'Restatement Key', 'Begin_date', 'End_date']]
df_restatements = df_restatements.dropna(subset=['Restatement Key', 'Year', 'CIK'])


# Merge with compustat data & ensure that the information already affects the year of interest
df = company_codes_merger_cik_2(df_restatements)

# The restatement of an annual report can reflect the severity of a restatement because annual reports must be audited by
# an independent accounting firm.
# An error or misstatement in an annual report is not only missed by management but also by the firm conducting the audit.
# Hence, we only use restatements that at least end during the date of the annual close of fiscal period.
df['Res_per'] = np.where(
    ((pd.to_datetime(df["Begin_date"], infer_datetime_format=True) <= pd.to_datetime(df["datadate"],
                                                                                      infer_datetime_format=True))
      & (pd.to_datetime(df["datadate"], infer_datetime_format=True) <= pd.to_datetime(df["End_date"],
                                                                                      infer_datetime_format=True)))
    ,
    1, 0)

df['Restatement Key'] = np.where(df['Res_per'] == 0, np.nan, df["Restatement Key"])

df['Res_m_per'] = np.where(
    ((pd.to_datetime(df["Begin_date"], infer_datetime_format=True) <= pd.to_datetime(df["datadate"],
                                                                                      infer_datetime_format=True))
      & (pd.to_datetime(df["datadate"], infer_datetime_format=True) <= pd.to_datetime(df["End_date"],
                                                                                      infer_datetime_format=True))
      & (df['Year_8K_402'].notnull() == True))
    , 1, 0)

df['Restatement Key'] = np.where(df['Res_m_per'] == 0, np.nan, df["Restatement Key"])
df['Year_8K_402'] = np.where(df['Res_per'] == 0, np.nan, df["Year_8K_402"])


df = df[['Year', 'CIK', 'Res_per', 'Res_m_per', 'Restatement Key', 'Year_8K_402']]

# keep the duplicate row with fewest nan values ('will focus on nan Year_8K_402 observations with key)
df = df.assign(counts=df.count(axis=1)).sort_values(['counts']). \
    drop_duplicates(['Year', 'CIK'], keep='last').drop('counts', axis=1)
df = df.sort_values(by=['CIK', 'Year']).reset_index(drop=True)

df = df[['Year', 'CIK', 'Res_per', 'Res_m_per', 'Restatement Key']]

df = company_codes_merger_cik(df)

print('The number of Restatements merged using CIK: ' + str(df[df['Res_per'] == 1].shape[0]))
print('The number of severe Restatements merged using CIK: ' + str(df[df['Res_m_per'] == 1].shape[0]))

counter_df = df.copy()
counter_df['counter'] = 1
grouped = counter_df[['Year', 'counter', 'Res_per', 'Res_m_per']].groupby('Year').sum()
grouped = grouped.rename(columns={"Res_per": "Number of Restatements", 'Res_m_per': 'Number of Severe Restatements',
                                  "counter": "Number of Firms"})
grouped['Percentage Restatements'] = grouped['Number of Restatements'] / grouped['Number of Firms'] * 100
grouped['Percentage Severe Restatements'] = grouped['Number of Severe Restatements'] / grouped['Number of Firms'] * 100

grouped = grouped.reset_index(level=0)
grouped[["Number of Restatements", "Number of Firms",
         "Number of Severe Restatements"]] = \
    grouped[["Number of Restatements", "Number of Firms",
             "Number of Severe Restatements"]].applymap('{:,.0f}'.format)
grouped[['Percentage Restatements']] = \
    grouped[['Percentage Restatements']].applymap('{:,.2f}'.format)
grouped[['Percentage Severe Restatements']] = \
    grouped[['Percentage Severe Restatements']].applymap('{:,.2f}'.format)
print(grouped.to_latex(index=False))

# Furthermore, handling of missing values
df[df.columns.difference(['Year', 'CIK'])] = \
    df[df.columns.difference(['Year', 'CIK'])].fillna(0)

path = (base_path / "../csv/Restatements.csv").resolve()
df.to_csv(path, index=True)
