import numpy as np
import pandas as pd
from ALIGNED_learning.testing import performance_check
from pathlib import Path
import random
import warnings

warnings.filterwarnings("ignore")

random.seed(2290)
np.random.seed(2290)

base_path = Path(__file__).parent

experiments = 'experiment_1'


path = (base_path / "data/csv/All_data_1.csv").resolve()
df_aaer_1 = pd.read_csv(path, index_col=0)

path = (base_path / "data/csv/All_data_2.csv").resolve()
df_aaer_2 = pd.read_csv(path, index_col=0)

df_aaer = pd.concat([df_aaer_1, df_aaer_2], ignore_index=True)


def setting_creater(df, feature_names, data_majority_undersample, train_period_list,
                    test_period_list, stakeholder, train_frac_list):
    train_index_list = []
    validation_index_list = []
    test_index_list = []
    name_list = []

    df_experiment_info = pd.DataFrame()
    df_experiment_info['number_train'] = ""
    df_experiment_info['number_val'] = ""
    df_experiment_info['number_test'] = ""
    df_experiment_info['train_ones'] = ""
    df_experiment_info['val_ones'] = ""
    df_experiment_info['test_ones'] = ""

    """
    features and nans
    """

    df = df[['AAER', 'CIK', 'Year', 'AAER_ID', 'Market_cap_all_loss_2016', 'Market_cap_5_per_loss_2016', 'F_score',
             'M_score'] + feature_names]

    # replace infs with nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    """
    undersampling
    """

    if data_majority_undersample is not None:
        df = df.drop(
            df.query('AAER == 0').sample(frac=data_majority_undersample, random_state=2290).index)
        df = df.reset_index(drop=True)

    """
    train/val/test split
    """

    X = df[feature_names]

    df_y = df.copy()
    df_y_c = df.copy()
    df_m_score = df.copy()
    df_f_score = df.copy()

    id_fraud_indenticator = df_y['AAER_ID'].copy()
    labeled_fraud_indicator = df_y['AAER'].copy()

    if stakeholder == 'Regulator':
        y_c_copy_cost = df_y_c['AAER'].copy()
        y_c_copy_cost = np.multiply(df_y_c['Market_cap_all_loss_2016'], y_c_copy_cost) + \
                        np.multiply(- df_y_c['Market_cap_5_per_loss_2016'], 1 - y_c_copy_cost)
        df_y_c['AAER'] = y_c_copy_cost

    y = df_y['AAER']
    y_c = df_y_c['AAER']
    m_score = df_m_score['M_score']
    f_score = df_f_score['F_score']

    for train_periods, test_periods, train_frac in zip(train_period_list, test_period_list, train_frac_list):

        train_indexs = []
        validation_indexs = []
        test_indexs = []
        names = []

        for train_period in train_periods:

            financial_misconduct_bool = ((df['Year'] >= train_period[0]) & (df['Year'] <= train_period[1]) & (
                        labeled_fraud_indicator == 1))
            non_label_bool = ((df['Year'] >= train_period[0]) & (df['Year'] <= train_period[1]) & (
                        labeled_fraud_indicator == 0))

            financial_misconduct = df[financial_misconduct_bool]
            non_label = df[non_label_bool]

            df_per_fin_mis_train_x = pd.Series(financial_misconduct.groupby('CIK').groups.keys()).sample(
                frac=train_frac, random_state=2290)
            df_per_fin_mis_train = financial_misconduct[
                financial_misconduct['CIK'].isin(df_per_fin_mis_train_x.tolist())]

            df_per_fin_mis_val = financial_misconduct.drop(df_per_fin_mis_train.index)
            df_per_fin_mis_val_x = pd.Series(df_per_fin_mis_val.groupby('CIK').groups.keys())

            df_per_label_bool_train_x = pd.Series(non_label.groupby('CIK').groups.keys()).sample(frac=train_frac,
                                                                                                 random_state=2290)

            # Use all CIK in the non label train sample. Additionally include all observations of CIK in the train that are labelled.
            # Omit all observations of CIK in the validation that are labelled
            df_per_label_bool_train = non_label[((((non_label['CIK'].isin(df_per_label_bool_train_x.tolist())) |
                                                   (non_label['CIK'].isin(df_per_fin_mis_train_x.tolist())))) &
                                                 (~non_label['CIK'].isin(df_per_fin_mis_val_x.tolist())))]

            df_per_fin_mis_val = financial_misconduct.drop(df_per_fin_mis_train.index)
            df_per_label_bool_val = non_label.drop(df_per_label_bool_train.index)

            train_bool_fin_mis = financial_misconduct_bool.index.isin(df_per_fin_mis_train.index)
            train_bool_label_bool = non_label_bool.index.isin(df_per_label_bool_train.index)
            train_bool = train_bool_fin_mis ^ train_bool_label_bool

            val_bool_fin_mis = financial_misconduct_bool.index.isin(df_per_fin_mis_val.index)
            val_bool_label_bool = non_label_bool.index.isin(df_per_label_bool_val.index)
            val_bool = val_bool_fin_mis ^ val_bool_label_bool

            train_index = df.index[train_bool].tolist()
            validation_index = df.index[val_bool].tolist()

            train_id = id_fraud_indenticator[train_bool]
            validation_id = id_fraud_indenticator[val_bool]

            train_id_used = train_id[labeled_fraud_indicator[train_bool] == 1]
            validation_id_used = validation_id[labeled_fraud_indicator[val_bool] == 1]

            mask = np.array(((~validation_id.isin(train_id_used)) | (np.isnan(validation_id))))
            validation_index = np.squeeze(np.array(validation_index))[mask].tolist()

            train_indexs.append(train_index)
            validation_indexs.append(validation_index)

            for test_period in test_periods:
                test_bool = ((df['Year'] >= test_period[0]) & (df['Year'] <= test_period[1]))

                test_index = df.index[test_bool].tolist()
                test_id = id_fraud_indenticator[test_bool]

                mask = np.array(
                    ((~(test_id.isin(train_id_used) | (test_id.isin(validation_id_used)))) | (np.isnan(test_id))))
                test_index = np.squeeze(np.array(test_index))[mask].tolist()

                train_set = y[train_index]
                val_set = y[validation_index]
                test_set = y[test_index]

                number_train = len(train_set)
                number_val = len(val_set)
                number_test = len(test_set)

                train_ones = np.count_nonzero(train_set)
                val_ones = np.count_nonzero(val_set)
                test_ones = np.count_nonzero(test_set)

                name = str(train_period[0]) + '_' + str(train_period[1]) + '_' + str(test_period[0]) + '_' + str(
                    test_period[1])

                df_experiment_info.loc[name, 'number_train'] = number_train
                df_experiment_info.loc[name, 'number_val'] = number_val
                df_experiment_info.loc[name, 'number_test'] = number_test
                df_experiment_info.loc[name, 'train_ones'] = train_ones
                df_experiment_info.loc[name, 'val_ones'] = val_ones
                df_experiment_info.loc[name, 'test_ones'] = test_ones

                test_indexs.append(test_index)
                names.append(name)

        train_index_list.append(train_indexs)
        validation_index_list.append(validation_indexs)
        test_index_list.append(test_indexs)
        name_list.append(names)

    return X, y, y_c, m_score, f_score, train_index_list, validation_index_list, \
           test_index_list, name_list, df_experiment_info


# pas self.max zaken aan zoals subsample...
def get_par_dict(n, p_prec, p_rbp, p_ep, n_c_ep, optimisation_metric):
    par_dict = {'General': {'n': n,
                            'p_prec': p_prec,
                            'p_rbp': p_rbp,
                            'p_ep': p_ep,
                            'n_c_ep': n_c_ep},
                'Logit': {'lambd': [1],
                          'sigma': [1],
                          'indic_approx': ['lambdaloss'],  # 'lambdaloss', 'logit'
                          'metric': optimisation_metric  # basic, arp, roc_auc, ap, dcg, ep, rbp, ep, precision
                          },
                'Lgbm': {"num_leaves": [5],
                         "n_estimators": [50],
                         "lambd": [1],
                         "learning_rate": [0.01],
                         "subsample": [1],
                         "min_child_samples": [0],
                         "min_child_weight": [1e-3], # 1e-3 do not change to zero. this causes issues regarding validation 'binary' and 'lambdarank'
                         "sigma": [1],  # 1 for validation 'binary' and 'lambdarank'
                         "indic_approx": ['lambdaloss'],  # 'lambdaloss', 'logit'   #lambdaloss for validation 'binary' and 'lambdarank'
                         "metric": optimisation_metric},
                # basic, lambdarank, arp, roc_auc, ap, dcg, ep, rbp, ep, precision
                'ENSImb': {"max_depth": [1],
                           "n_estimators": [50],
                           "learning_rate": [0.01],
                           "sampling_strategy": [1],
                           "method": ['RUSBoost']}}

    return par_dict


# Dechow
# feature_names = ['Wc_acc', 'Rsst_acc', 'Ch_rec', 'Ch_inv', 'Soft_assets', 'Ch_cs', 'Ch_cm', 'Ch_roa',
#                  'Ch_fcf', 'Tax', 'Ch_emp', 'Ch_backlog', 'Leasedum', 'Oplease', 'Pension', 'Ch_pension',
#                  'Exfin', 'Issue', 'Cff', 'Leverage', 'Bm', 'Ep']

# Bao
feature_names = ['csho', 'act', 'sstk', 'ppegt',
                 'ap', 'che', 'prcc_c', 're',
                 'invt', 'ceq', 'dlc', 'dp',
                 'rect', 'cogs', 'at', 'dltis',
                 'ib', 'dltt', 'xint', 'txt',
                 'lct', 'sale', 'txp', 'ivao',
                 'lt', 'ivst', 'ni', 'pstk']

methods = ['Logit', 'Lgbm', 'ENSImb', 'M_score', 'F_score']  # Logit, Lgbm, ENSImb

# We impose 2 year gap to mitigate serial fraud issue in test set
data_majority_undersample = 0.9
train_period_list = [[[1995, 2001]], [[1995, 2002]], [[1995, 2003]], [[1995, 2004]], [[1995, 2005]]]
test_period_list = [[[2004, 2004]], [[2005, 2005]], [[2006, 2006]], [[2007, 2007]], [[2008, 2008]]]
train_frac_list = [0.75, 1, 1, 1, 1]

feature_importance = False
stakeholder = 'Regulator'
cost_train = False
cost_validate = False

X, y, y_c, m_score, f_score, train_index_list, validation_index_list, test_index_list, \
name_list, df_experiment_info = \
    setting_creater(df_aaer,
                    feature_names=feature_names,
                    data_majority_undersample=data_majority_undersample,
                    train_period_list=train_period_list,
                    test_period_list=test_period_list,
                    stakeholder=stakeholder, train_frac_list=train_frac_list)

if 'experiment_1' in experiments:

    cross_val_perf_ind = 'ep'
    optimisation_metric = 'ep'

    n = 1
    p_prec = 0.1
    p_rbp = 0.9

    p_ep = 0.1
    n_c_ep = 12

    for i_1 in range(len(name_list)):
        for i_2 in range(len(name_list[i_1])):
            name_list[i_1][i_2] = 'AAER_experiment_1_' + name_list[i_1][i_2]

    par_dict = get_par_dict(n=n, p_prec=p_prec, p_rbp=p_rbp, p_ep=p_ep, n_c_ep=n_c_ep,
                            optimisation_metric=[optimisation_metric])
    name = 'AAER_experiment_1_info' + '.csv'
    df_experiment_info.to_csv((base_path / "tables/tables experiment info" / name).resolve())

    performance_check(methods=methods,
                      par_dict_init=par_dict,
                      X=X, y=y, y_c=y_c, m_score=m_score, f_score=f_score,
                      name_list=name_list, train_list=train_index_list,
                      validate_list=validation_index_list, test_list=test_index_list,
                      feature_importance=feature_importance, cross_val_perf_ind=cross_val_perf_ind,
                      n=n, p_prec=p_prec, p_rbp=p_rbp, p_ep=p_ep, n_c_ep=n_c_ep, cost_train=cost_train,
                      cost_validate=cost_validate, keep_first=True)

if 'experiment_2' in experiments:

    cross_val_perf_ind = 'arp'
    optimisation_metric = 'basic'

    n = 1
    p_prec = 0.1
    p_rbp = 0.9

    p_ep = 0.1
    n_c_ep = 12

    for i_1 in range(len(name_list)):
        for i_2 in range(len(name_list[i_1])):
            name_list[i_1][i_2] = 'AAER_experiment_2_' + name_list[i_1][i_2]

    par_dict = get_par_dict(n=n, p_prec=p_prec, p_rbp=p_rbp, p_ep=p_ep, n_c_ep=n_c_ep,
                            optimisation_metric=[optimisation_metric])
    name = 'AAER_experiment_2_info' + '.csv'
    df_experiment_info.to_csv((base_path / "tables/tables experiment info" / name).resolve())

    performance_check(methods=methods,
                      par_dict_init=par_dict,
                      X=X, y=y, y_c=y_c, m_score=m_score, f_score=f_score,
                      name_list=name_list, train_list=train_index_list,
                      validate_list=validation_index_list, test_list=test_index_list,
                      feature_importance=feature_importance, cross_val_perf_ind=cross_val_perf_ind,
                      n=n, p_prec=p_prec, p_rbp=p_rbp, p_ep=p_ep, n_c_ep=n_c_ep, cost_train=cost_train,
                      cost_validate=cost_validate, keep_first=True)

if 'experiment_3' in experiments:

    cross_val_perf_ind = 'precision'
    optimisation_metric = 'precision'

    n = 1
    p_prec = 0.1
    p_rbp = 0.9

    p_ep = 0.1
    n_c_ep = 12

    for i_1 in range(len(name_list)):
        for i_2 in range(len(name_list[i_1])):
            name_list[i_1][i_2] = 'AAER_experiment_3_' + name_list[i_1][i_2]

    par_dict = get_par_dict(n=n, p_prec=p_prec, p_rbp=p_rbp, p_ep=p_ep, n_c_ep=n_c_ep,
                            optimisation_metric=[optimisation_metric])
    name = 'AAER_experiment_3_info' + '.csv'
    df_experiment_info.to_csv((base_path / "tables/tables experiment info" / name).resolve())

    performance_check(methods=methods,
                      par_dict_init=par_dict,
                      X=X, y=y, y_c=y_c, m_score=m_score, f_score=f_score,
                      name_list=name_list, train_list=train_index_list,
                      validate_list=validation_index_list, test_list=test_index_list,
                      feature_importance=feature_importance, cross_val_perf_ind=cross_val_perf_ind,
                      n=n, p_prec=p_prec, p_rbp=p_rbp, p_ep=p_ep, n_c_ep=n_c_ep, cost_train=cost_train,
                      cost_validate=cost_validate, keep_first=True)
