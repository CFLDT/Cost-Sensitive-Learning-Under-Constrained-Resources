import numpy as np
import pandas as pd
from pathlib import Path
from ALIGNED_learning.testing import performance_check

base_path = Path(__file__).parent

# German Credit

path = (base_path / "data/real/cleaned/UCI_German_Credit.csv").resolve()
df_german_credit = pd.read_csv(path, sep=",", index_col=0)

X_german_credit = df_german_credit.drop('y', axis=1)
y_german_credit = df_german_credit.loc[:, 'y']
ratio_german_credit = np.count_nonzero(y_german_credit) / len(y_german_credit)  # 0.041
y_c_german_credit = np.multiply(5, y_german_credit) + np.multiply(-1, 1 - y_german_credit)

# Churn

path = (base_path / "data/real/cleaned/Telco_Customer_Churn.csv").resolve()
df_telco_customer_churn = pd.read_csv(path)
df_telco_customer_churn = df_telco_customer_churn.drop_duplicates()


df_telco_customer_churn.loc[:, 'Churn'].replace({'Yes': 1, 'No': 0}, inplace=True)
X_telco_customer_churn = df_telco_customer_churn.drop(['Churn', 'customerID', 'TotalCharges', 'MonthlyCharges'], axis=1)
y_telco_customer_churn = df_telco_customer_churn['Churn']

# y_c_telco_customer_churn = np.multiply(df_telco_customer_churn['MonthlyCharges']-5, y_telco_customer_churn) + \
#                        np.multiply(- 5, 1 - y_telco_customer_churn)

y_churn_effect_telco_customer_churn = np.multiply(y_telco_customer_churn, df_telco_customer_churn['SeniorCitizen'])
y_no_churn_effect_telco_customer_churn = np.multiply(1-y_telco_customer_churn, 1-df_telco_customer_churn['SeniorCitizen'])

y_c_telco_customer_churn = np.multiply(df_telco_customer_churn['MonthlyCharges']-5, y_churn_effect_telco_customer_churn) + \
                           np.multiply(-5, y_churn_effect_telco_customer_churn) + \
                           np.multiply(-5, y_no_churn_effect_telco_customer_churn) + \
                           np.multiply(-5-df_telco_customer_churn['MonthlyCharges'], y_no_churn_effect_telco_customer_churn)


ratio_telco_customer_churn = np.count_nonzero(y_telco_customer_churn) / len(y_telco_customer_churn)

# Fraud

path = (base_path / "data/real/cleaned/data_FraudDetection_JAR2020.csv").resolve()
df_accounting_fraud = pd.read_csv(path)
df_accounting_fraud['market_cap'] = df_accounting_fraud['csho'] * df_accounting_fraud['prcc_f']
df_accounting_fraud = df_accounting_fraud.sort_values(by='fyear', ascending=True)
df_accounting_fraud = df_accounting_fraud[df_accounting_fraud['fyear'] >= 1995]
df_accounting_fraud = df_accounting_fraud[df_accounting_fraud['fyear'] <= 2008]
df_accounting_fraud = df_accounting_fraud.drop(
    df_accounting_fraud.query('misstate == 0').sample(frac=.95, random_state=2290).index)  #0.9 original
df_accounting_fraud = df_accounting_fraud.reset_index(drop=True)

y_accounting_fraud = df_accounting_fraud['misstate']
X_accounting_fraud = df_accounting_fraud.drop(['fyear', 'gvkey', 'p_aaer', 'misstate', 'market_cap'], axis=1)
ratio_accounting_fraud = np.count_nonzero(y_accounting_fraud) / len(y_accounting_fraud)

# market cap

df_mark_cap = df_accounting_fraud[['fyear', 'market_cap']]

df_mark_cap['Market_cap_1_per_loss'] = df_mark_cap['market_cap'] * 0.01
df_mark_cap['Market_cap_3_per_loss'] = df_mark_cap['market_cap'] * 0.03
df_mark_cap['Market_cap_5_per_loss'] = df_mark_cap['market_cap'] * 0.05
df_mark_cap['Market_cap_15_per_loss'] = df_mark_cap['market_cap'] * 0.15

groupby_obj = df_mark_cap[['fyear', 'market_cap']].groupby(['fyear']).mean().reset_index()
groupby_obj = groupby_obj.rename(columns={"market_cap": "Market_cap_all_average"})

df_mark_cap = pd.merge(df_mark_cap, groupby_obj, how='left', on=['fyear'])

df_mark_cap['Market_cap_scaled_average'] = df_mark_cap['market_cap'] / df_mark_cap['Market_cap_all_average']
df_mark_cap['Market_cap_5_per_loss_scaled_average'] = df_mark_cap['Market_cap_5_per_loss'] / df_mark_cap[
    'Market_cap_all_average']
df_mark_cap['Market_cap_15_per_loss_scaled_average'] = df_mark_cap['Market_cap_15_per_loss'] / df_mark_cap[
    'Market_cap_all_average']

# y_c_accounting_fraud = np.multiply(df_mark_cap['Market_cap_scaled_average'], y_accounting_fraud) + \
#                        np.multiply(- df_mark_cap['Market_cap_15_per_loss_scaled_average'], 1 - y_accounting_fraud)

# y_c_accounting_fraud = np.multiply(np.log(df_mark_cap['Market_cap_scaled_average']+1), y_accounting_fraud) + \
#                        np.multiply(- np.log(0.15*df_mark_cap['Market_cap_scaled_average']+1), 1 - y_accounting_fraud)

y_c_accounting_fraud = np.multiply(np.log(df_mark_cap['Market_cap_scaled_average']+1), y_accounting_fraud) + \
                       np.multiply(- 0.15* np.log(df_mark_cap['Market_cap_scaled_average']+1), 1 - y_accounting_fraud)



train_index_list = [df_accounting_fraud.index[((df_accounting_fraud['fyear'] >= 1995) & (df_accounting_fraud['fyear'] <= 1998))].tolist(),
                    df_accounting_fraud.index[((df_accounting_fraud['fyear'] >= 1996) & (df_accounting_fraud['fyear'] <= 1999))].tolist(),
                    df_accounting_fraud.index[((df_accounting_fraud['fyear'] >= 1997) & (df_accounting_fraud['fyear'] <= 2000))].tolist(),
                    df_accounting_fraud.index[((df_accounting_fraud['fyear'] >= 1998) & (df_accounting_fraud['fyear'] <= 2001))].tolist()
                    ]

validation_index_list = [df_accounting_fraud.index[((df_accounting_fraud['fyear'] >= 2001) & (df_accounting_fraud['fyear'] <= 2002))].tolist(),
                    df_accounting_fraud.index[((df_accounting_fraud['fyear'] >= 2002) & (df_accounting_fraud['fyear'] <= 2003))].tolist(),
                    df_accounting_fraud.index[((df_accounting_fraud['fyear'] >= 2003) & (df_accounting_fraud['fyear'] <= 2004))].tolist(),
                    df_accounting_fraud.index[((df_accounting_fraud['fyear'] >= 2004) & (df_accounting_fraud['fyear'] <= 2005))].tolist()
                    ]

test_index_list = [df_accounting_fraud.index[((df_accounting_fraud['fyear'] >= 2003) & (df_accounting_fraud['fyear'] <= 2005))].tolist(),
                    df_accounting_fraud.index[((df_accounting_fraud['fyear'] >= 2004) & (df_accounting_fraud['fyear'] <= 2006))].tolist(),
                    df_accounting_fraud.index[((df_accounting_fraud['fyear'] >= 2005) & (df_accounting_fraud['fyear'] <= 2007))].tolist(),
                    df_accounting_fraud.index[((df_accounting_fraud['fyear'] >= 2006) & (df_accounting_fraud['fyear'] <= 2008))].tolist()
                    ]

# hyperparameters

par_dict = {'General_val_test': {'n_ratio': 1,
                                 'n_p_prec': 100,
                                 'p_rbp': 0.9,
                                 'n_p_ep': 100,
                                 'p_ep_val': 1/3,
                                 'n_n_found': 100},
            'Logit': {'lambd': [0, 0.1, 1],
                      'sigma': [1],
                      'subsample_undersample': [[None, None]],
                      'indic_approx': ['lambdaloss']  # 'lambdaloss', 'logit'
                      },
            'Lgbm': {"num_leaves": [5],
                     "n_estimators": [25, 100],  # [50, 100],
                     "lambd": [0, 10],  # [0, 10],
                     "alpha": [0],
                     "learning_rate": [0.01, 0.001],
                     "colsample_bytree": [0.75],
                     "sample_subsample_undersample": [[0.5, None],[0.75, None]],
                     "subsample_freq": [1],
                     "min_child_samples": [0],
                     "min_child_weight": [1e-3],
                     # 1e-3 do not change to zero. this causes issues regarding validation 'binary' and 'lambdarank'
                     "sigma": [1],
                     "indic_approx": ['lambdaloss'] # 'lambdaloss', 'logit'
                      },
            # basic, lambdarank, arp, roc_auc, ap, dcg, ep, rbp, ep, precision
            'ENSImb': {"max_depth": [1, 5],
                       "n_estimators": [25, 100],
                       "learning_rate": [0.01, 0.001],
                       "undersample": [0.5, 1],
                       "method": ['RUSBoost']}}


######
##EP##
######

methods = ['Logit', 'Lgbm']

par_dict["Logit"]["metric"] = ['ep'] # basic, arp, roc_auc, ap, dcg, ep, rbp, ep, precision
par_dict["Logit"]["n_ratio"] = [1]
par_dict["Logit"]["p_ep"] = [1/3]

par_dict["Lgbm"]["metric"] = ['ep'] # basic, arp, roc_auc, ap, dcg, ep, rbp, ep, precision
par_dict["Lgbm"]["n_ratio"] = [1]
par_dict["Lgbm"]["p_ep"] = [1/3]


# name = 'ep_german_credit'
# performance_check(methods=methods, par_dict_init=par_dict, X=X_german_credit, y=y_german_credit, y_c=y_c_german_credit,
#                   name=name, fold=4, repeats=1, perf_ind="ep", cost_train=True, cost_validate=True,
#                   time_series_split = False, train_index_list=None, validation_index_list=None, test_index_list=None, cross_val=True)
#
# name = 'ep_financial_misconduct'
# performance_check(methods=methods, par_dict_init=par_dict, X=X_accounting_fraud, y=y_accounting_fraud,
#                   y_c=y_c_accounting_fraud,
#                   name=name, fold=None, repeats=None, perf_ind="ep", cost_train=True, cost_validate=True,
#                   time_series_split=True, train_index_list=train_index_list, validation_index_list=validation_index_list,
#                   test_index_list=test_index_list, cross_val=True)

# name = 'ep_telco'
# performance_check(methods=methods, par_dict_init=par_dict, X=X_telco_customer_churn, y=y_telco_customer_churn,
#                   y_c=y_c_telco_customer_churn,
#                   name=name, fold=4, repeats=1, perf_ind="ep", cost_train=True, cost_validate=True,
#                   time_series_split=False, train_index_list=None, validation_index_list=None, test_index_list=None, cross_val=True)

#################
##cross_entropy##
#################

methods = ['Logit', 'Lgbm', 'ENSImb']

par_dict["Logit"]["metric"] = ['basic'] # basic, arp, roc_auc, ap, dcg, ep, rbp, ep, precision
par_dict["Lgbm"]["metric"] = ['basic'] # basic, arp, roc_auc, ap, dcg, ep, rbp, ep, precision

# name = 'basic_german_credit'
# performance_check(methods=methods, par_dict_init=par_dict, X=X_german_credit, y=y_german_credit, y_c=y_c_german_credit,
#                   name=name, fold=4, repeats=1, perf_ind="ap", cost_train=False, cost_validate=False,
#                   time_series_split = False, train_index_list=None, validation_index_list=None, test_index_list=None, cross_val=True)
#
name = 'basic_financial_misconduct'
performance_check(methods=methods, par_dict_init=par_dict, X=X_accounting_fraud, y=y_accounting_fraud,
                  y_c=y_c_accounting_fraud,
                  name=name,  fold=None, repeats=None, perf_ind="ap", cost_train=False, cost_validate=False,
                  time_series_split=True, train_index_list=train_index_list, validation_index_list=validation_index_list,
                  test_index_list=test_index_list, cross_val=True)

# name = 'basic_telco'
# performance_check(methods=methods, par_dict_init=par_dict, X=X_telco_customer_churn, y=y_telco_customer_churn, y_c=y_c_telco_customer_churn,
#                   name=name, fold=4, repeats=1, perf_ind="ap", cost_train=False, cost_validate=False,
#                   time_series_split = False, train_index_list=None, validation_index_list=None, test_index_list=None, cross_val=True)
#

################
##squared_loss##
################

methods = ['Lgbm']

par_dict["Logit"]["metric"] = ['basic'] # basic, arp, roc_auc, ap, dcg, ep, rbp, ep, precision
par_dict["Lgbm"]["metric"] = ['basic'] # basic, arp, roc_auc, ap, dcg, ep, rbp, ep, precision

# name = 'basic_ql_german_credit'
# performance_check(methods=methods, par_dict_init=par_dict, X=X_german_credit, y=y_german_credit, y_c=y_c_german_credit,
#                   name=name, fold=4, repeats=1, perf_ind="ql", cost_train=True, cost_validate=True,
#                   time_series_split = False, train_index_list=None, validation_index_list=None, test_index_list=None, cross_val=True)
#
name = 'basic_ql_financial_misconduct'
performance_check(methods=methods, par_dict_init=par_dict, X=X_accounting_fraud, y=y_accounting_fraud,
                  y_c=y_c_accounting_fraud,
                  name=name,  fold=None, repeats=None, perf_ind="ql", cost_train=True, cost_validate=True,
                  time_series_split=True, train_index_list=train_index_list, validation_index_list=validation_index_list,
                  test_index_list=test_index_list, cross_val=True)

# name = 'basic_ql_telco'
# performance_check(methods=methods, par_dict_init=par_dict, X=X_telco_customer_churn, y=y_telco_customer_churn, y_c=y_c_telco_customer_churn,
#                   name=name, fold=4, repeats=1, perf_ind="ql", cost_train=True, cost_validate=True,
#                   time_series_split = False, train_index_list=None, validation_index_list=None, test_index_list=None, cross_val=True)


