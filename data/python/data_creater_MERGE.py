import numpy as np
import pandas as pd
from pathlib import Path

base_path = Path(__file__).parent
from functools import reduce

# MERGE ALL

path = (base_path / "../csv/AAER_Data.csv").resolve()
df_AAER = pd.read_csv(path, index_col=0)

path = (base_path / "../csv/Beneish.csv").resolve()
df_beneish = pd.read_csv(path, index_col=0)

path = (base_path / "../csv/Controls.csv").resolve()
df_controls = pd.read_csv(path, index_col=0)

path = (base_path / "../csv/Dechow.csv").resolve()
df_dechow = pd.read_csv(path, index_col=0)

path = (base_path / "../csv/Restatements.csv").resolve()
df_restatements = pd.read_csv(path, index_col=0)

path = (base_path / "../csv/Market_Cap.csv").resolve()
df_market_cap = pd.read_csv(path, index_col=0)

# compile the list of dataframes you want to merge
data_frames = [df_AAER, df_beneish, df_controls, df_dechow, df_restatements, df_market_cap]

df_merged = reduce(lambda left, right: pd.merge(left, right, on=['CIK', 'Year'],
                                                how='left'), data_frames)

print('The number of distinct AAER firms: merged using CIK: ' + str(
    df_merged['CIK'][df_merged['AAER'] == 1].drop_duplicates().shape[0]))
print('The number of AAERs: merged using CIK: ' + str(np.count_nonzero(df_merged['AAER'] == 1)))

min_year = df_merged['Year'].min()
max_year = df_merged['Year'].max()

df_merged = df_merged.reset_index(drop=True)

# We only work with data containing cost information
nan_gain_indices = df_merged[((df_merged['Market_cap_all_loss_2016'].isnull()) | (
    df_merged['Market_cap_5_per_loss_2016'].isnull()))].index.tolist()
df_merged = df_merged.drop(nan_gain_indices)
df_merged = df_merged.reset_index(drop=True)

counter_df = df_merged.copy()
counter_df['counter'] = 1
grouped = counter_df[['Year', 'counter', 'AAER', 'Res_per', 'Res_m_per']].groupby('Year').sum()
grouped = grouped.rename(columns={"AAER": "Number of AAERs", "counter": "Number of Firms", "Res_per": "Number of Restatements",
                                    "Res_m_per": "Number of Severe Restatements"})
grouped['Percentage AAERs'] = grouped['Number of AAERs'] / grouped['Number of Firms'] * 100
grouped['Percentage Restatements'] = grouped['Number of Restatements'] / grouped['Number of Firms'] * 100
grouped['Percentage Severe Restatements'] = grouped['Number of Severe Restatements'] / grouped['Number of Firms'] * 100

grouped = grouped.reset_index(level=0)
grouped[['Number of AAERs', 'Number of Firms', 'Number of Restatements', 'Number of Severe Restatements']] = \
    grouped[['Number of AAERs', 'Number of Firms', 'Number of Restatements', 'Number of Severe Restatements']].applymap('{:,.0f}'.format)
grouped[['Percentage AAERs']] = \
    grouped[['Percentage AAERs']].applymap('{:,.2f}'.format)
grouped[['Percentage Restatements']] = \
    grouped[['Percentage Restatements']].applymap('{:,.2f}'.format)
grouped[['Percentage Severe Restatements']] = \
    grouped[['Percentage Severe Restatements']].applymap('{:,.2f}'.format)
print(grouped.to_latex(index=False))

for i in range(min_year, max_year+1):
    df_merged = df_merged.drop(
        df_merged.query('Year == '+str(i)).query('AAER == 0').sample(frac=0.8, random_state=2290).index)
    df_merged = df_merged.reset_index(drop=True)


counter_df = df_merged.copy()
counter_df['counter'] = 1
grouped = counter_df[['Year', 'counter', 'AAER', 'Res_per', 'Res_m_per']].groupby('Year').sum()
grouped = grouped.rename(columns={"AAER": "Number of AAERs", "counter": "Number of Firms", "Res_per": "Number of Restatements",
                                    "Res_m_per": "Number of Severe Restatements"})
grouped['Percentage AAERs'] = grouped['Number of AAERs'] / grouped['Number of Firms'] * 100
grouped['Percentage Restatements'] = grouped['Number of Restatements'] / grouped['Number of Firms'] * 100
grouped['Percentage Severe Restatements'] = grouped['Number of Severe Restatements'] / grouped['Number of Firms'] * 100

grouped = grouped.reset_index(level=0)
grouped[['Number of AAERs', 'Number of Firms', 'Number of Restatements', 'Number of Severe Restatements']] = \
    grouped[['Number of AAERs', 'Number of Firms', 'Number of Restatements', 'Number of Severe Restatements']].applymap('{:,.0f}'.format)
grouped[['Percentage AAERs']] = \
    grouped[['Percentage AAERs']].applymap('{:,.2f}'.format)
grouped[['Percentage Restatements']] = \
    grouped[['Percentage Restatements']].applymap('{:,.2f}'.format)
grouped[['Percentage Severe Restatements']] = \
    grouped[['Percentage Severe Restatements']].applymap('{:,.2f}'.format)
print(grouped.to_latex(index=False))


path = (base_path / "../csv/All_data.csv").resolve()
df_merged.to_csv(path, index=True)
