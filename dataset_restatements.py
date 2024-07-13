import numpy as np
import pandas as pd
from ALIGNED_learning.testing import performance_check
from pathlib import Path
import random
import warnings
import math

warnings.filterwarnings("ignore")

random.seed(2290)
np.random.seed(2290)

base_path = Path(__file__).parent

experiments = 'experiment_1'

path = (base_path / "data/csv/All_data_1.csv").resolve()
df_restatement_1 = pd.read_csv(path, index_col=0)

path = (base_path / "data/csv/All_data_2.csv").resolve()
df_restatement_2 = pd.read_csv(path, index_col=0)

df_restatement = pd.concat([df_restatement_1, df_restatement_2], ignore_index=True)

def setting_creater(df, feature_names, train_period_list,
                    test_period_list, stakeholder, validation_list):
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

    df = df[['Res_per', 'CIK', 'Year', 'Restatement Key',
             'Market_cap_all_loss_2016', 'Market_cap_5_per_loss_2016', 'F_score',
             'M_score'] + feature_names]

    # replace infs with nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    """
    undersampling
    """
    #
    # if data_majority_undersample is not None:
    #     df = df.drop(
    #         df.query('Res_per == 0').sample(frac=data_majority_undersample, random_state=2290).index)
    #     df = df.reset_index(drop=True)

    df = df.reset_index(drop=True)

    """
    train/val/test split
    """

    X = df[feature_names]

    df_y = df.copy()
    df_y_c = df.copy()
    df_m_score = df.copy()
    df_f_score = df.copy()

    id_fraud_indenticator = df_y['Restatement Key'].copy()
    labeled_fraud_indicator = df_y['Res_per'].copy()

    if stakeholder == 'Regulator':
        y_c_copy_cost = df_y_c['Res_per'].copy()
        y_c_copy_cost = np.multiply(df_y_c['Market_cap_all_loss_2016'], y_c_copy_cost) + \
                        np.multiply(- df_y_c['Market_cap_5_per_loss_2016'], 1 - y_c_copy_cost)
        df_y_c['Res_per'] = y_c_copy_cost

    y = df_y['Res_per']
    y_c = df_y_c['Res_per']
    m_score = df_m_score['M_score']
    f_score = df_f_score['F_score']

    for train_periods, test_periods, validations in zip(train_period_list, test_period_list, validation_list):

        train_indexs = []
        validation_indexs = []
        test_indexs = []
        names = []

        for train_period in train_periods:

            train_bool = ((df['Year'] >= train_period[0]) & (df['Year'] <= train_period[1]))

            if validations == True:
                val_bool = ((df['Year'] >= train_period[1]+1) & (df['Year'] <= train_period[1] + 1))
            else:
                val_bool = pd.Series(np.zeros(df.shape[0], dtype=bool))


            # if data_majority_undersample_train is not None:
            #     train_index = df.index[train_bool].tolist()
            #     df_train_index = df.loc[train_index]
            #
            #     df_train_index = df_train_index.drop(
            #         df_train_index.query('Res_per == 0').sample(frac=data_majority_undersample_train, random_state=2290).index)
            #
            #     train_index = df_train_index.index.tolist()
            #
            # else:

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
                if test_period[0] is None:
                    test_bool = pd.Series(np.zeros(df.shape[0], dtype=bool))
                else:
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

def get_par_dict(optimisation_metric):
    par_dict = {'General_val_test': {'n_ratio': 1,
                            'n_p_prec': 100,
                            'p_rbp': 0.9,
                            'n_p_ep': 100,
                             'n_p_ep_val': 100,
                             'n_n_found':100},
                'Logit': {'lambd': [0, 0.1, 1],
                          'sigma': [1],
                          'subsample_undersample': [[None, None]],
                          'indic_approx': ['lambdaloss'],  # 'lambdaloss', 'logit'
                          'metric': optimisation_metric  # basic, arp, roc_auc, ap, dcg, ep, rbp, ep, precision
                          },
                'Lgbm': {"num_leaves": [5],
                         "n_estimators": [50],  # [50, 100],
                         "lambd": [10], # [0, 10],
                         "alpha": [0],
                         "learning_rate": [0.01],  # [0.1, 0.01],
                         "colsample_bytree": [0.75],
                         "sample_subsample_undersample": [[0.1, None], [0.2, 1]],
                         "subsample_freq": [1],
                         "min_child_samples": [0],
                         "min_child_weight": [1e-3], # 1e-3 do not change to zero. this causes issues regarding validation 'binary' and 'lambdarank'
                         "sigma": [1],    # 1 for validation 'binary' and 'lambdarank'
                         "indic_approx": ['lambdaloss'],  # 'lambdaloss', 'logit'   #lambdaloss for validation 'binary' and 'lambdarank'
                         "metric": optimisation_metric},
                # basic, lambdarank, arp, roc_auc, ap, dcg, ep, rbp, ep, precision
                'ENSImb': {"max_depth": [1, 5],
                           "n_estimators": [50, 100],
                           "learning_rate": [0.01, 0.001],
                           "undersample": [0.5, 1],
                           "method": ['RUSBoost']}}

    return par_dict


# Dechow
feature_names = ['Wc_acc', 'Rsst_acc', 'Ch_rec', 'Ch_inv', 'Soft_assets', 'Ch_cs', 'Ch_cm', 'Ch_roa',
                 'Ch_fcf', 'Tax', 'Ch_emp', 'Ch_backlog', 'Leasedum', 'Oplease', 'Pension', 'Ch_pension',
                 'Exfin', 'Issue', 'Cff', 'Leverage', 'Bm', 'Ep']

# Bao
# feature_names = ['csho', 'act', 'sstk', 'ppegt',
#                  'ap', 'che', 'prcc_c', 're',
#                  'invt', 'ceq', 'dlc', 'dp',
#                  'rect', 'cogs', 'at', 'dltis',
#                  'ib', 'dltt', 'xint', 'txt',
#                  'lct', 'sale', 'txp', 'ivao',
#                  'lt', 'ivst', 'ni', 'pstk']



train_period_list = [[[2004, 2008]], [[2005, 2009]], [[2006, 2010]], [[2007, 2011]], [[2008, 2012]], [[2009, 2013]],
                                     [[2010, 2014]], [[2011, 2015]]]
test_period_list = [[[None, None]], [[2010, 2010]], [[2011, 2011]], [[2012, 2012]], [[2013, 2013]], [[2014, 2014]],
                                    [[2015, 2015]], [[2016, 2016]]]
validation_list = [True, False, False, False, False, False, False, False]


feature_importance = False
stakeholder = 'Regulator'


if 'experiment_1' in experiments:

    cost_train = False
    cost_validate = False
    data_majority_undersample_train = None

    X, y, y_c, m_score, f_score, train_index_list, validation_index_list, test_index_list, \
    name_list, df_experiment_info = \
        setting_creater(df_restatement,
                        feature_names=feature_names,
                        train_period_list=train_period_list,
                        test_period_list=test_period_list,
                        stakeholder=stakeholder, validation_list=validation_list)

    methods = ['Lgbm']

    cross_val_perf_ind = 'ep'
    optimisation_metric = 'ep'

    for i_1 in range(len(name_list)):
        for i_2 in range(len(name_list[i_1])):
            name_list[i_1][i_2] = 'Restatement_experiment_1_' + name_list[i_1][i_2]

    par_dict = get_par_dict(optimisation_metric=[optimisation_metric])


    par_dict["Lgbm"]["n_ratio"] = [1]
    par_dict["Lgbm"]["p_ep"] = [1/3, 0.5, 2/3]


    name = 'Restatement_experiment_1_info' + '.csv'
    df_experiment_info.to_csv((base_path / "tables/tables experiment info" / name).resolve())

    performance_check(methods=methods,
                      par_dict_init=par_dict,
                      X=X, y=y, y_c=y_c, m_score=m_score, f_score=f_score,
                      name_list=name_list, train_list=train_index_list,
                      validate_list=validation_index_list, test_list=test_index_list,
                      feature_importance=feature_importance, cross_val_perf_ind=cross_val_perf_ind,
                      cost_train=cost_train,
                      cost_validate=cost_validate, keep_first=True)


if 'experiment_2' in experiments:

    cost_train = False
    cost_validate = False

    X, y, y_c, m_score, f_score, train_index_list, validation_index_list, test_index_list, \
    name_list, df_experiment_info = \
        setting_creater(df_restatement,
                        feature_names=feature_names,
                        train_period_list=train_period_list,
                        test_period_list=test_period_list,
                        stakeholder=stakeholder, validation_list=validation_list)

    methods = ['Logit']

    cross_val_perf_ind = 'ep'
    optimisation_metric = 'ep'


    for i_1 in range(len(name_list)):
        for i_2 in range(len(name_list[i_1])):
            name_list[i_1][i_2] = 'Restatement_experiment_2_' + name_list[i_1][i_2]

    par_dict = get_par_dict(optimisation_metric=[optimisation_metric])


    par_dict["Logit"]["n_ratio"] = [1]
    par_dict["Lgbm"]["p_ep"] = [1/3, 0.5, 2/3]


    name = 'Restatement_experiment_2_info' + '.csv'
    df_experiment_info.to_csv((base_path / "tables/tables experiment info" / name).resolve())

    performance_check(methods=methods,
                      par_dict_init=par_dict,
                      X=X, y=y, y_c=y_c, m_score=m_score, f_score=f_score,
                      name_list=name_list, train_list=train_index_list,
                      validate_list=validation_index_list, test_list=test_index_list,
                      feature_importance=feature_importance, cross_val_perf_ind=cross_val_perf_ind,
                      cost_train=cost_train,
                      cost_validate=cost_validate, keep_first=True)


if 'experiment_3' in experiments:

    cost_train = False
    cost_validate = False
    data_majority_undersample_train = None

    X, y, y_c, m_score, f_score, train_index_list, validation_index_list, test_index_list, \
    name_list, df_experiment_info = \
        setting_creater(df_restatement,
                        feature_names=feature_names,
                        train_period_list=train_period_list,
                        test_period_list=test_period_list,
                        stakeholder=stakeholder, validation_list=validation_list)

    methods = ['Lgbm', 'ENSImb', 'M_score', 'F_score']

    cross_val_perf_ind = 'ap'
    optimisation_metric = 'basic'


    for i_1 in range(len(name_list)):
        for i_2 in range(len(name_list[i_1])):
            name_list[i_1][i_2] = 'Restatement_experiment_3_' + name_list[i_1][i_2]

    par_dict = get_par_dict(optimisation_metric=[optimisation_metric])

    par_dict["Lgbm"]["n_ratio"] = [1]

    name = 'Restatement_experiment_3_info' + '.csv'
    df_experiment_info.to_csv((base_path / "tables/tables experiment info" / name).resolve())

    performance_check(methods=methods,
                      par_dict_init=par_dict,
                      X=X, y=y, y_c=y_c, m_score=m_score, f_score=f_score,
                      name_list=name_list, train_list=train_index_list,
                      validate_list=validation_index_list, test_list=test_index_list,
                      feature_importance=feature_importance, cross_val_perf_ind=cross_val_perf_ind,
                      cost_train=cost_train,
                      cost_validate=cost_validate, keep_first=True)


if 'experiment_4' in experiments:

    cost_train = False
    cost_validate = False

    X, y, y_c, m_score, f_score, train_index_list, validation_index_list, test_index_list, \
    name_list, df_experiment_info = \
        setting_creater(df_restatement,
                        feature_names=feature_names,
                        train_period_list=train_period_list,
                        test_period_list=test_period_list,
                        stakeholder=stakeholder, validation_list=validation_list)

    methods = ['Logit']

    cross_val_perf_ind = 'ap'
    optimisation_metric = 'basic'


    for i_1 in range(len(name_list)):
        for i_2 in range(len(name_list[i_1])):
            name_list[i_1][i_2] = 'Restatement_experiment_4_' + name_list[i_1][i_2]

    par_dict = get_par_dict(optimisation_metric=[optimisation_metric])

    par_dict["Logit"]["n_ratio"] = [1]

    name = 'Restatement_experiment_4_info' + '.csv'
    df_experiment_info.to_csv((base_path / "tables/tables experiment info" / name).resolve())

    performance_check(methods=methods,
                      par_dict_init=par_dict,
                      X=X, y=y, y_c=y_c, m_score=m_score, f_score=f_score,
                      name_list=name_list, train_list=train_index_list,
                      validate_list=validation_index_list, test_list=test_index_list,
                      feature_importance=feature_importance, cross_val_perf_ind=cross_val_perf_ind,
                      cost_train=cost_train,
                      cost_validate=cost_validate, keep_first=True)

if 'experiment_5' in experiments:

    cost_train = True
    cost_validate = True
    data_majority_undersample_train = None

    X, y, y_c, m_score, f_score, train_index_list, validation_index_list, test_index_list, \
    name_list, df_experiment_info = \
        setting_creater(df_restatement,
                        feature_names=feature_names,
                        train_period_list=train_period_list,
                        test_period_list=test_period_list,
                        stakeholder=stakeholder, validation_list=validation_list)

    methods = ['Lgbm']

    cross_val_perf_ind = 'ep'
    optimisation_metric = 'ep'

    for i_1 in range(len(name_list)):
        for i_2 in range(len(name_list[i_1])):
            name_list[i_1][i_2] = 'Restatement_experiment_5_' + name_list[i_1][i_2]

    par_dict = get_par_dict(optimisation_metric=[optimisation_metric])

    par_dict["Lgbm"]["n_ratio"] = [1]
    par_dict["Lgbm"]["p_ep"] = [1/3, 0.5, 2/3]

    name = 'Restatement_experiment_5_info' + '.csv'
    df_experiment_info.to_csv((base_path / "tables/tables experiment info" / name).resolve())

    performance_check(methods=methods,
                      par_dict_init=par_dict,
                      X=X, y=y, y_c=y_c, m_score=m_score, f_score=f_score,
                      name_list=name_list, train_list=train_index_list,
                      validate_list=validation_index_list, test_list=test_index_list,
                      feature_importance=feature_importance, cross_val_perf_ind=cross_val_perf_ind,
                      cost_train=cost_train,
                      cost_validate=cost_validate, keep_first=True)

if 'experiment_6' in experiments:

    cost_train = True
    cost_validate = True

    X, y, y_c, m_score, f_score, train_index_list, validation_index_list, test_index_list, \
    name_list, df_experiment_info = \
        setting_creater(df_restatement,
                        feature_names=feature_names,
                        train_period_list=train_period_list,
                        test_period_list=test_period_list,
                        stakeholder=stakeholder, validation_list=validation_list)

    methods = ['Logit']

    cross_val_perf_ind = 'ep'
    optimisation_metric = 'ep'

    for i_1 in range(len(name_list)):
        for i_2 in range(len(name_list[i_1])):
            name_list[i_1][i_2] = 'Restatement_experiment_6_' + name_list[i_1][i_2]

    par_dict = get_par_dict(optimisation_metric=[optimisation_metric])

    par_dict["Logit"]["n_ratio"] = [1]
    par_dict["Lgbm"]["p_ep"] = [1/3, 0.5, 2/3]

    name = 'Restatement_experiment_6_info' + '.csv'
    df_experiment_info.to_csv((base_path / "tables/tables experiment info" / name).resolve())

    performance_check(methods=methods,
                      par_dict_init=par_dict,
                      X=X, y=y, y_c=y_c, m_score=m_score, f_score=f_score,
                      name_list=name_list, train_list=train_index_list,
                      validate_list=validation_index_list, test_list=test_index_list,
                      feature_importance=feature_importance, cross_val_perf_ind=cross_val_perf_ind,
                      cost_train=cost_train,
                      cost_validate=cost_validate, keep_first=True)

