import numpy as np
import pandas as pd
from pathlib import Path
from data.python.data_creater_FUNCTIONS import company_codes_merger_cik

# Compustat from 01-1990 until 08-2022 to recreate the variables in
# https://www.thecaq.org/wp-content/uploads/2018/03/Dechow-et-al-2011-Contemporary_Accounting_Research.pdf
# (see also https://www.jstor.org/stable/26550651)

# act Current Assets - Total
# ap Accounts Payable - Trade
# at Assets - Total
# ceq Common/Ordinary Equity - Total
# che Cash and Short-Term Investments
# cogs Cost of Goods Sold
# csho Common Shares Outstanding
# dlc Debt in Current Liabilities
# dltis Long-Term Debt Issuance
# dltt Long-Term Debt Total
# dp Depreciation and Amortization
# ib Income Before Extraordinary Items
# invt Inventories - Total
# ivao Investment and Advances Other
# ivst Short-Term Investments - Total
# lct Current Liabilities - Total
# lt Liabilities - Total
# ni Net Income (Loss)
# ppegt Property, Plant and Equipment - Total (Gross)
# pstk Preferred/Preference Stock (Capital) - Total
# re Retained Earnings
# rect Receivables Total
# sale Sales/Turnover (Net)
# sstk Sale of Common and Preferred Stock
# txp Income Taxes Payable
# txt Income Taxes - Total
# xint Interest and Related Expense - Total
# prcc_c Price Close - Annual - Calender
# txdi Income Taxes-Deferred
# mrc Minimal rental commitments
# fincf Financing Activities - Net Cash Flow
# capx Capital Expenditures
# oancf Operating Activities - Net Cash Flow
# ob Order Backlog
# ppror Pension Plans/Anticipated Long-Term Rate of Return on Plan Assets
# emp Number Of Employees

base_path = Path(__file__).parent

path = (base_path / "../original_data/COMPUSTAT_Dechow_1.csv").resolve()
df_dech = pd.read_csv(path)

df_dech["datadate"] = pd.to_datetime(df_dech["datadate"], infer_datetime_format=True).dt.year
df_dech = df_dech.drop('fyear', axis=1)

df_dech = df_dech.rename(columns={"datadate": "Year", 'cik': 'CIK'})
df_dech['CIK'] = df_dech['CIK'].astype('Int64')
df_dech['Year'] = df_dech['Year'].astype('Int64')
df_dech = df_dech.loc[df_dech['CIK'].notna(), :]
df_dech = df_dech.loc[df_dech['Year'].notna(), :]
df_dech = df_dech.drop(['gvkey', 'consol', 'datafmt', 'popsrc', 'indfmt', 'conm', 'curcd', 'fyr', 'tic', 'costat'], axis=1)


# keep the duplicate row with fewest nan values
df_dech = df_dech.assign(counts=df_dech.count(axis=1)).sort_values(['counts']).drop_duplicates(['Year', 'CIK'],
                                                                                               keep='last').drop('counts', axis=1)
df_dech = df_dech.sort_values(by=['CIK', 'Year']).reset_index(drop=True)


path = (base_path / "../original_data/COMPUSTAT_Dechow_2.csv").resolve()
df_pens = pd.read_csv(path)

df_pens["datadate"] = pd.to_datetime(df_pens["datadate"], infer_datetime_format=True).dt.year

df_pens = df_pens.rename(columns={"datadate": "Year", 'cik': 'CIK'})
df_pens['CIK'] = df_pens['CIK'].astype('Int64')
df_pens['Year'] = df_pens['Year'].astype('Int64')
df_pens = df_pens.loc[df_pens['CIK'].notna(), :]
df_pens = df_pens.loc[df_pens['Year'].notna(), :]

df_pens = df_pens[['Year', 'CIK', 'ppror']]

# keep the duplicate row with fewest nan values
df_pens = df_pens.assign(counts=df_pens.count(axis=1)).sort_values(['counts']).drop_duplicates(['Year', 'CIK'],
                         keep='last').drop('counts', axis=1)

df_pens = df_pens.sort_values(by=['CIK', 'Year']).reset_index(drop=True)


# Working capital accruals

df_dech['wc_acc_numer'] = df_dech['act'] - df_dech['che'] - (df_dech['lct'] - df_dech['dlc'] - df_dech['txp'])
df_dech['wc_acc_denom'] = df_dech['at']

groupby_obj = df_dech.groupby(['CIK'])
wc_acc_numer_diff = groupby_obj['wc_acc_numer'].diff().reset_index(drop=True)
wc_acc_denom_mean = groupby_obj['wc_acc_denom'].rolling(2).mean().reset_index(drop=True)

df_dech['WC_accruals'] = wc_acc_numer_diff / wc_acc_denom_mean
df_dech = df_dech.drop(['wc_acc_numer', 'wc_acc_denom'], axis=1)

# RSST_accruals

df_dech['wc_rsst_acc_numer'] = (df_dech['act'] - df_dech['che']) - (df_dech['lct'] - df_dech['dlc'] - df_dech['txp'])
df_dech['nco_rsst_acc_numer'] = (df_dech['at'] - df_dech['act'] - df_dech['ivao']) - (df_dech['lt'] - df_dech['lct'] - df_dech['dltt'])
df_dech['fin_rsst_acc_numer'] = (df_dech['ivst'] + df_dech['ivao']) - (df_dech['dltt'] + df_dech['dlc'] + df_dech['pstk'])
df_dech['rsst_acc_denom'] = df_dech['at']

groupby_obj = df_dech.groupby(['CIK'])

wc_rsst_acc_numer_diff = groupby_obj['wc_rsst_acc_numer'].diff().reset_index(drop=True)
nco_rsst_acc_numer_diff = groupby_obj['nco_rsst_acc_numer'].diff().reset_index(drop=True)
fin_rsst_acc_numer_diff = groupby_obj['fin_rsst_acc_numer'].diff().reset_index(drop=True)
rsst_acc_denom_mean = groupby_obj['rsst_acc_denom'].rolling(2).mean().reset_index(drop=True)
df_dech['RSST_accruals'] = wc_rsst_acc_numer_diff + nco_rsst_acc_numer_diff + fin_rsst_acc_numer_diff \
                           / rsst_acc_denom_mean
df_dech = df_dech.drop(['wc_rsst_acc_numer', 'nco_rsst_acc_numer', 'fin_rsst_acc_numer', 'rsst_acc_denom'], axis=1)

# Change in receivables

groupby_obj = df_dech.groupby(['CIK'])
ch_rec_numer_diff = groupby_obj['rect'].diff().reset_index(drop=True)
ch_rec_denom_mean = groupby_obj['at'].rolling(2).mean().reset_index(drop=True)

df_dech['Change_receivables'] = ch_rec_numer_diff / ch_rec_denom_mean

# Change in inventory

groupby_obj = df_dech.groupby(['CIK'])
ch_inv_numer_diff = groupby_obj['invt'].diff().reset_index(drop=True)
ch_inv_denom_mean = groupby_obj['at'].rolling(2).mean().reset_index(drop=True)

df_dech['Change_inventory'] = ch_inv_numer_diff / ch_inv_denom_mean

# Percentage soft assets

df_dech['Perc_soft_assets'] = (df_dech['at'] - df_dech['ppegt'] - df_dech['che']) / df_dech['at']

# Percentage change in cash sales

groupby_obj = df_dech.groupby(['CIK'])
ch_rec_diff = groupby_obj['rect'].diff().reset_index(drop=True)

df_dech['Change_cash_sales_int'] = df_dech['sale'] - ch_rec_diff
df_dech['Change_cash_sales'] = groupby_obj['Change_cash_sales_int'].pct_change()

df_dech = df_dech.drop(['Change_cash_sales_int'], axis=1)

# Percentage change in cash margin

groupby_obj = df_dech.groupby(['CIK'])

ch_cm_inv_numer_diff = groupby_obj['invt'].diff().reset_index(drop=True)
ch_cm_ap_numer_diff = groupby_obj['ap'].diff().reset_index(drop=True)
ch_cm_rect_denom_diff = groupby_obj['rect'].diff().reset_index(drop=True)

df_dech['Change_cash_margin_int'] = 1 - ((df_dech['cogs'] - ch_cm_inv_numer_diff + ch_cm_ap_numer_diff) /
                                         (df_dech['sale'] - ch_cm_rect_denom_diff))

df_dech['Change_cash_margin'] = groupby_obj['Change_cash_margin_int'].pct_change()

df_dech = df_dech.drop(['Change_cash_margin_int'], axis=1)

# Change in return on assets

groupby_obj = df_dech.groupby(['CIK'])

df_dech['Ret_ass'] = df_dech['ib'] / groupby_obj['at'].rolling(2).mean().reset_index(drop=True)

df_dech['Change_return_assets'] = groupby_obj['Ret_ass'].diff().reset_index(drop=True)

df_dech = df_dech.drop(['Ret_ass'], axis=1)

# Change in free cash flow

df_dech['ib_rsst'] = df_dech['ib'] - df_dech['RSST_accruals']

groupby_obj = df_dech.groupby(['CIK'])

fr_ca_fl_numer_diff = groupby_obj['ib_rsst'].diff().reset_index(drop=True)
fr_ca_fl_denom_mean = groupby_obj['at'].rolling(2).mean().reset_index(drop=True)

df_dech['Change_free_cash_flow'] = fr_ca_fl_numer_diff / fr_ca_fl_denom_mean

df_dech = df_dech.drop(['ib_rsst'], axis=1)


# Deferred tax expense

groupby_obj = df_dech.groupby(['CIK'])

df_dech['deferred_tax_expense'] = df_dech['txdi'] / (df_dech['at'] - groupby_obj['at'].diff().reset_index(drop=True))

# Change in employees

groupby_obj = df_dech.groupby(['CIK'])

perc_ch_empl = groupby_obj['emp'].pct_change()

perc_ch_assets = groupby_obj['at'].pct_change()

df_dech['Ch_emp'] = perc_ch_empl - perc_ch_assets
df_dech.replace([np.inf, -np.inf], np.nan, inplace=True)

# Abnormal change in order backlog

groupby_obj = df_dech.groupby(['CIK'])

perc_ch_sales = groupby_obj['sale'].pct_change()

perc_ch_order_backlog = groupby_obj['ob'].pct_change()

df_dech['Abnormal_change_in_order_backlog'] = perc_ch_sales - perc_ch_order_backlog

# Operating leases existence

bool = ((np.array(df_dech['mrc1']) > 0) | (np.array(df_dech['mrc2']) > 0) |
        (np.array(df_dech['mrc3']) > 0) | (np.array(df_dech['mrc4']) > 0) | (np.array(df_dech['mrc5']) > 0))
df_dech['Operational_lease_existence'] = np.where(bool, 1, 0)

# Change in operating lease activity

current_op_lease = df_dech['mrc1'] / 1.1 + df_dech['mrc2'] / (1.1 ** 2) + df_dech['mrc3'] / (1.1 ** 3) \
                   + df_dech['mrc4'] / (1.1 ** 4) + df_dech['mrc5'] / (1.1 ** 5)
prev_op_lease = (df_dech['mrc1'] - groupby_obj['mrc1'].diff().reset_index(drop=True)) / 1.1 + \
                (df_dech['mrc2'] - groupby_obj['mrc2'].diff().reset_index(drop=True)) / (1.1 ** 2) + \
                (df_dech['mrc3'] - groupby_obj['mrc3'].diff().reset_index(drop=True)) / (1.1 ** 3) + \
                (df_dech['mrc4'] - groupby_obj['mrc4'].diff().reset_index(drop=True)) / (1.1 ** 4) + \
                (df_dech['mrc5'] - groupby_obj['mrc5'].diff().reset_index(drop=True)) / (1.1 ** 5)

df_dech['Change_operating_lease_activity'] = current_op_lease - prev_op_lease

# Expected return on pension plan assets

df_pens['Pension'] = df_pens['ppror']

# Change in expected return on pension plan assets

groupby_obj = df_pens.groupby(['CIK'])
df_pens['Ch_pension'] = groupby_obj['ppror'].diff().reset_index(drop=True)

# Demand for financing (ex ante)

groupby_obj = df_dech.groupby(['CIK'])
previous_capx_1 = df_dech['capx'] - groupby_obj['capx'].diff().reset_index(drop=True)
previous_capx_2 = df_dech['capx'] - groupby_obj['capx'].diff(periods=2).reset_index(drop=True)
previous_capx_3 = df_dech['capx'] - groupby_obj['capx'].diff(periods=3).reset_index(drop=True)

bool = ((df_dech['oancf'] - (((previous_capx_3 + previous_capx_2 + previous_capx_1) / 3)) / (df_dech['act'])) < -0.5)
df_dech['Exfin'] = np.where(bool, 1, 0)


# Actual issuance

bool = ((df_dech['sstk'] > 0) | (df_dech['dltis'] > 0))
df_dech['Issue'] = np.where(bool, 1, 0)

# Level of finance raised

df_dech['Cff'] = df_dech['fincf'] / groupby_obj['at'].rolling(2).mean().reset_index(drop=True)

# leverage

df_dech['Leverage'] = df_dech['dltt'] / df_dech['at']

# Book to market

df_dech['Market_cap'] = df_dech['csho'] * df_dech['prcc_c']
df_dech['Bm'] = df_dech['ceq'] / df_dech['Market_cap']

# Earnings to price

df_dech['Ep'] = df_dech['ib'] / df_dech['Market_cap']

# df_dech = df_dech.drop(["act", "ap", 'ceq', 'che', 'cogs', 'csho',
#               'dlc', 'dltis', 'dltt', 'dp', 'ib', 'invt',
#               'ivao', 'ivst', 'lct', 'lt', 'ni', 'ppegt', 'pstk',
#               're', 'rect', 'sale', 'sstk', 'txp', 'xint', 'txt',
#               'prcc_c', 'txdb', 'txdi', 'mrc1', 'mrc2', 'mrc3', 'mrc4', 'mrc5', 'fincf', 'capx',
#               'oancf', 'ob', 'emp', 'sic','at','Market_cap'], axis=1)
# df_pens = df_pens.drop('ppror', axis=1)

df_dech = df_dech.rename(columns={'WC_accruals': 'Wc_acc', "RSST_accruals": "Rsst_acc", "Change_receivables": "Ch_rec",
                        "Change_inventory": "Ch_inv", 'Perc_soft_assets': 'Soft_assets',
                        'Change_cash_sales': 'Ch_cs', 'Change_cash_margin': 'Ch_cm', 'Change_return_assets': 'Ch_roa',
                        'Change_free_cash_flow': 'Ch_fcf', 'deferred_tax_expense': 'Tax',
                        'Mod_jones': 'Da', 'Abnormal_change_in_order_backlog': 'Ch_backlog',
                        'Operational_lease_existence': 'Leasedum',
                        'Change_operating_lease_activity': 'Oplease'
                                  })
df_dech.replace([np.inf, -np.inf], np.nan, inplace=True)

df = company_codes_merger_cik(df_dech)
df = pd.merge(df, df_pens, how='left', on=['Year', 'CIK'])

df['F_score'] = -7.893 + 0.790*df['Rsst_acc'].fillna(df['Rsst_acc'].median()) + 2.518*df['Ch_rec'].fillna(df['Ch_rec'].median()) + \
                     1.191*df['Ch_inv'].fillna(df['Ch_inv'].median()) +  1.979*df['Soft_assets'].fillna(df['Soft_assets'].median()) + \
                     0.171*df['Ch_cs'].fillna(df['Ch_cs'].median()) - 0.932*df['Ch_roa'].fillna(df['Ch_roa'].median()) \
                     + 1.029*df['Issue'].fillna(df['Issue'].median())


path = (base_path / "../csv/Dechow.csv").resolve()
df.to_csv(path, index=True)
