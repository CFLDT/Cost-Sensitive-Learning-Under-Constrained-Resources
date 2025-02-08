import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import RepeatedStratifiedKFold, TimeSeriesSplit
import copy

from ..design import MethodLearner
from ..design import PerformanceMetrics
from ..plots_tables import performance_tables
from ..design import divide_clean

from pathlib import Path

base_path = Path(__file__).parent


def performance_check(methods, par_dict_init, X, y, y_c, name, fold, repeats, perf_ind, cost_train, cost_validate,
                      time_series_split, train_index_list, validation_index_list, test_index_list, cross_val=False):
    n_ratio = par_dict_init.get('General_val_test').get("n_ratio")
    n_p_prec = par_dict_init.get('General_val_test').get("n_p_prec")
    p_rbp = par_dict_init.get('General_val_test').get("p_rbp")
    n_p_ep = par_dict_init.get('General_val_test').get("n_p_ep")
    p_ep_val = par_dict_init.get('General_val_test').get("p_ep_val")
    n_n_found = par_dict_init.get('General_val_test').get("n_n_found")

    X_train_list = []
    X_test_list = []

    y_train_list = []
    y_test_list = []

    y_c_train_list = []
    y_c_test_list = []

    par_dict_list = []

    counter = 0

    if time_series_split == False:
        cross_validator = RepeatedStratifiedKFold(n_splits=fold, n_repeats=repeats, random_state=2290)

        for train_index, test_index in cross_validator.split(X, y):
            print('performance number ' + str(counter + 1))

            if cross_val == True:

                train_indexing = train_index

                y_train = y.iloc[train_index]

                cross_validator_2 = RepeatedStratifiedKFold(n_splits=int(fold - 1), n_repeats=1, random_state=2290)
                for train_index, validation_index in cross_validator_2.split(train_indexing, y_train):
                    train_index = train_indexing[train_index]
                    validation_index = train_indexing[validation_index]
                    break

                par_dict, X_train, y_train, y_c_train, X_val, y_val, y_c_val, datapipeline, st_scaler = cross_validation_t_v(
                    methods=methods, par_dic=par_dict_init, X=X, y=y, y_c=y_c, train_index=train_index,
                    validation_index=validation_index, name=name,
                    n_ratio=n_ratio, n_p_prec=n_p_prec, p_rbp=p_rbp,
                    p_ep_val=p_ep_val, n_n_found=n_n_found, cost_train=cost_train, cost_validate=cost_validate,
                    perf_ind=perf_ind)

                X_test = st_scaler.transform(np.array(datapipeline.pipeline_trans(X.iloc[test_index])))
                y_test = np.array(y.iloc[test_index])
                y_c_test = np.array(y_c.iloc[test_index])

            if cross_val == False:
                par_dict = par_dict_init
                X_train, y_train, y_c_train, X_test, y_test, y_c_test, datapipeline, st_scaler = divide_clean(X, y, y_c,
                                                                                                              train_index,
                                                                                                              test_index)

            X_train_list.append(X_train)
            y_train_list.append(y_train)
            y_c_train_list.append(y_c_train)
            X_test_list.append(X_test)
            y_test_list.append(y_test)
            y_c_test_list.append(y_c_test)
            par_dict_list.append(par_dict)

            counter = counter + 1

    if time_series_split == True:

        for train_index, validation_index, test_index in zip(train_index_list, validation_index_list, test_index_list):

            print('performance number ' + str(counter + 1))

            if cross_val == True:
                par_dict, X_train, y_train, y_c_train, X_val, y_val, y_c_val, datapipeline, st_scaler = cross_validation_t_v(
                    methods=methods, par_dic=par_dict_init, X=X, y=y, y_c=y_c, train_index=train_index,
                    validation_index=validation_index, name=name,
                    n_ratio=n_ratio, n_p_prec=n_p_prec, p_rbp=p_rbp,
                    p_ep_val=p_ep_val, n_n_found=n_n_found, cost_train=cost_train, cost_validate=cost_validate,
                    perf_ind=perf_ind)

                X_test = st_scaler.transform(np.array(datapipeline.pipeline_trans(X.iloc[test_index])))
                y_test = np.array(y.iloc[test_index])
                y_c_test = np.array(y_c.iloc[test_index])

            if cross_val == False:
                par_dict = par_dict_init
                X_train, y_train, y_c_train, X_test, y_test, y_c_test, datapipeline, st_scaler = divide_clean(X, y,
                                                                                                              y_c,
                                                                                                              train_index,
                                                                                                              test_index)
            X_train_list.append(X_train)
            y_train_list.append(y_train)
            y_c_train_list.append(y_c_train)
            X_test_list.append(X_test)
            y_test_list.append(y_test)
            y_c_test_list.append(y_c_test)
            par_dict_list.append(par_dict)

            counter = counter + 1

    roc_auc_df = pd.DataFrame()
    ap_df = pd.DataFrame()
    disc_cum_gain_df = pd.DataFrame()
    arp_df = pd.DataFrame()
    precision_df = pd.DataFrame()
    rbp_df = pd.DataFrame()
    uplift_df = pd.DataFrame()
    ep_df = pd.DataFrame()
    n_found_df = pd.DataFrame()
    n_found_0_1_df = pd.DataFrame()
    n_found_0_2_df = pd.DataFrame()
    n_found_0_3_df = pd.DataFrame()
    n_found_0_4_df = pd.DataFrame()
    n_found_0_5_df = pd.DataFrame()
    ep_1_3_df = pd.DataFrame()
    ep_1_2_df = pd.DataFrame()
    ep_2_3_df = pd.DataFrame()

    roc_auc_c_df = pd.DataFrame()
    ap_c_df = pd.DataFrame()
    disc_cum_gain_c_df = pd.DataFrame()
    arp_c_df = pd.DataFrame()
    precision_c_df = pd.DataFrame()
    rbp_c_df = pd.DataFrame()
    uplift_c_df = pd.DataFrame()
    ep_c_df = pd.DataFrame()
    n_found_c_df = pd.DataFrame()
    n_found_0_1_c_df = pd.DataFrame()
    n_found_0_2_c_df = pd.DataFrame()
    n_found_0_3_c_df = pd.DataFrame()
    n_found_0_4_c_df = pd.DataFrame()
    n_found_0_5_c_df = pd.DataFrame()
    ep_1_3_c_df = pd.DataFrame()
    ep_1_2_c_df = pd.DataFrame()
    ep_2_3_c_df = pd.DataFrame()

    if 'Logit' in methods:
        roc_auc_df['Logit'] = ""
        ap_df['Logit'] = ""
        disc_cum_gain_df['Logit'] = ""
        arp_df['Logit'] = ""
        precision_df['Logit'] = ""
        rbp_df['Logit'] = ""
        uplift_df['Logit'] = ""
        ep_df['Logit'] = ""
        n_found_0_1_df["Logit"] = ""
        n_found_0_2_df["Logit"] = ""
        n_found_0_3_df["Logit"] = ""
        n_found_0_4_df["Logit"] = ""
        n_found_0_5_df["Logit"] = ""
        ep_1_3_df["Logit"] = ""
        ep_1_2_df["Logit"] = ""
        ep_2_3_df["Logit"] = ""

        roc_auc_c_df['Logit'] = ""
        ap_c_df['Logit'] = ""
        disc_cum_gain_c_df['Logit'] = ""
        arp_c_df['Logit'] = ""
        precision_c_df['Logit'] = ""
        rbp_c_df['Logit'] = ""
        uplift_c_df['Logit'] = ""
        ep_c_df['Logit'] = ""
        n_found_c_df["Logit"] = ""
        n_found_0_1_c_df["Logit"] = ""
        n_found_0_2_c_df["Logit"] = ""
        n_found_0_3_c_df["Logit"] = ""
        n_found_0_4_c_df["Logit"] = ""
        n_found_0_5_c_df["Logit"] = ""
        ep_1_3_c_df["Logit"] = ""
        ep_1_2_c_df["Logit"] = ""
        ep_2_3_c_df["Logit"] = ""

    if 'Lgbm' in methods:
        roc_auc_df['Lgbm'] = ""
        ap_df['Lgbm'] = ""
        disc_cum_gain_df['Lgbm'] = ""
        arp_df['Lgbm'] = ""
        precision_df['Lgbm'] = ""
        rbp_df['Lgbm'] = ""
        uplift_df['Lgbm'] = ""
        ep_df['Lgbm'] = ""
        n_found_df["Lgbm"] = ""
        n_found_0_1_df["Lgbm"] = ""
        n_found_0_2_df["Lgbm"] = ""
        n_found_0_3_df["Lgbm"] = ""
        n_found_0_4_df["Lgbm"] = ""
        n_found_0_5_df["Lgbm"] = ""
        ep_1_3_df["Lgbm"] = ""
        ep_1_2_df["Lgbm"] = ""
        ep_2_3_df["Lgbm"] = ""

        roc_auc_c_df['Lgbm'] = ""
        ap_c_df['Lgbm'] = ""
        disc_cum_gain_c_df['Lgbm'] = ""
        arp_c_df['Lgbm'] = ""
        precision_c_df['Lgbm'] = ""
        rbp_c_df['Lgbm'] = ""
        uplift_c_df['Lgbm'] = ""
        ep_c_df['Lgbm'] = ""
        n_found_c_df["Lgbm"] = ""
        n_found_0_1_c_df["Lgbm"] = ""
        n_found_0_2_c_df["Lgbm"] = ""
        n_found_0_3_c_df["Lgbm"] = ""
        n_found_0_4_c_df["Lgbm"] = ""
        n_found_0_5_c_df["Lgbm"] = ""
        ep_1_3_c_df["Lgbm"] = ""
        ep_1_2_c_df["Lgbm"] = ""
        ep_2_3_c_df["Lgbm"] = ""

    if 'ENSImb' in methods:
        roc_auc_df['ENSImb'] = ""
        ap_df['ENSImb'] = ""
        disc_cum_gain_df['ENSImb'] = ""
        arp_df['ENSImb'] = ""
        precision_df['ENSImb'] = ""
        rbp_df['ENSImb'] = ""
        uplift_df['ENSImb'] = ""
        ep_df['ENSImb'] = ""
        n_found_df["ENSImb"] = ""
        n_found_0_1_df["ENSImb"] = ""
        n_found_0_2_df["ENSImb"] = ""
        n_found_0_3_df["ENSImb"] = ""
        n_found_0_4_df["ENSImb"] = ""
        n_found_0_5_df["ENSImb"] = ""
        ep_1_3_df["ENSImb"] = ""
        ep_1_2_df["ENSImb"] = ""
        ep_2_3_df["ENSImb"] = ""

        roc_auc_c_df['ENSImb'] = ""
        ap_c_df['ENSImb'] = ""
        disc_cum_gain_c_df['ENSImb'] = ""
        arp_c_df['ENSImb'] = ""
        precision_c_df['ENSImb'] = ""
        rbp_c_df['ENSImb'] = ""
        uplift_c_df['ENSImb'] = ""
        ep_c_df['ENSImb'] = ""
        n_found_c_df["ENSImb"] = ""
        n_found_0_1_c_df["ENSImb"] = ""
        n_found_0_2_c_df["ENSImb"] = ""
        n_found_0_3_c_df["ENSImb"] = ""
        n_found_0_4_c_df["ENSImb"] = ""
        n_found_0_5_c_df["ENSImb"] = ""
        ep_1_3_c_df["ENSImb"] = ""
        ep_1_2_c_df["ENSImb"] = ""
        ep_2_3_c_df["ENSImb"] = ""

    for i in range(len(X_train_list)):

        X_train = X_train_list[i]
        y_train = y_train_list[i]
        y_c_train = y_c_train_list[i]
        par_dict_opt = par_dict_list[i]

        X_test = X_test_list[i]
        y_test = y_test_list[i]
        y_c_test = y_c_test_list[i]
        name_test = name + str(i)

        if 'Logit' in methods:

            if cost_train == False:
                model_logit = MethodLearner.logit(par_dict_opt.get('Logit'),
                                                  X_train, y_train, y_train)

            if cost_train == True:
                model_logit = MethodLearner.logit(par_dict_opt.get('Logit'),
                                                  X_train, y_c_train, y_train)

        if 'Lgbm' in methods:

            if cost_train == False:
                lgbmboost, model_lgbm = MethodLearner.lgbmboost(par_dict_opt.get('Lgbm'),
                                                                X_train, y_train, y_train)
            if cost_train == True:
                lgbmboost, model_lgbm = MethodLearner.lgbmboost(par_dict_opt.get('Lgbm'),
                                                                X_train, y_c_train, y_train)

        if 'ENSImb' in methods:
            ens, model_ensimb = MethodLearner.ensimb(par_dict_opt.get('ENSImb'), X_train,
                                                     y_train)

        if 'Logit' in methods:

            predict = model_logit.predict_proba(X_test)
            roc, ap, precision, dcg, arp, rbp, uplift, ep, \
            n_found, _, n_found_0_1, n_found_0_2, n_found_0_3, n_found_0_4, n_found_0_5, ep_1_3, ep_1_2, ep_2_3 = performances(
                predict, y_test,
                n_ratio=n_ratio,
                n_p_prec=n_p_prec,
                p_rbp=p_rbp,
                n_p_ep=n_p_ep,
                n_n_found=n_n_found)

            roc_auc_df.loc[name_test, 'Logit'] = roc
            ap_df.loc[name_test, 'Logit'] = ap
            disc_cum_gain_df.loc[name_test, 'Logit'] = dcg
            arp_df.loc[name_test, 'Logit'] = arp
            precision_df.loc[name_test, 'Logit'] = precision
            rbp_df.loc[name_test, 'Logit'] = rbp
            uplift_df.loc[name_test, 'Logit'] = uplift
            ep_df.loc[name_test, 'Logit'] = ep
            n_found_df.loc[name_test, 'Logit'] = n_found
            n_found_0_1_df.loc[name_test, 'Logit'] = n_found_0_1
            n_found_0_2_df.loc[name_test, 'Logit'] = n_found_0_2
            n_found_0_3_df.loc[name_test, 'Logit'] = n_found_0_3
            n_found_0_4_df.loc[name_test, 'Logit'] = n_found_0_4
            n_found_0_5_df.loc[name_test, 'Logit'] = n_found_0_5
            ep_1_3_df.loc[name_test, 'Logit'] = ep_1_3
            ep_1_2_df.loc[name_test, 'Logit'] = ep_1_2
            ep_2_3_df.loc[name_test, 'Logit'] = ep_2_3

            roc, ap, precision, dcg, arp, rbp, uplift, ep, \
            n_found, _, n_found_0_1, n_found_0_2, n_found_0_3, n_found_0_4, n_found_0_5, ep_1_3, ep_1_2, ep_2_3 = performances(
                predict, y_c_test,
                n_ratio=n_ratio,
                n_p_prec=n_p_prec,
                p_rbp=p_rbp,
                n_p_ep=n_p_ep,
                n_n_found=n_n_found,
                cost=True)

            roc_auc_c_df.loc[name_test, 'Logit'] = roc
            ap_c_df.loc[name_test, 'Logit'] = ap
            disc_cum_gain_c_df.loc[name_test, 'Logit'] = dcg
            arp_c_df.loc[name_test, 'Logit'] = arp
            precision_c_df.loc[name_test, 'Logit'] = precision
            rbp_c_df.loc[name_test, 'Logit'] = rbp
            uplift_c_df.loc[name_test, 'Logit'] = uplift
            ep_c_df.loc[name_test, 'Logit'] = ep
            n_found_c_df.loc[name_test, 'Logit'] = n_found
            n_found_0_1_c_df.loc[name_test, 'Logit'] = n_found_0_1
            n_found_0_2_c_df.loc[name_test, 'Logit'] = n_found_0_2
            n_found_0_3_c_df.loc[name_test, 'Logit'] = n_found_0_3
            n_found_0_4_c_df.loc[name_test, 'Logit'] = n_found_0_4
            n_found_0_5_c_df.loc[name_test, 'Logit'] = n_found_0_5
            ep_1_3_c_df.loc[name_test, 'Logit'] = ep_1_3
            ep_1_2_c_df.loc[name_test, 'Logit'] = ep_1_2
            ep_2_3_c_df.loc[name_test, 'Logit'] = ep_2_3

        if 'Lgbm' in methods:
            predict = lgbmboost.predict_proba(model_lgbm, X_test)
            roc, ap, precision, dcg, arp, rbp, uplift, ep, \
            n_found, _, n_found_0_1, n_found_0_2, n_found_0_3, n_found_0_4, n_found_0_5, ep_1_3, ep_1_2, ep_2_3 = performances(
                predict, y_test,
                n_ratio=n_ratio,
                n_p_prec=n_p_prec,
                p_rbp=p_rbp,
                n_p_ep=n_p_ep,
                n_n_found=n_n_found,
                cost=False)

            roc_auc_df.loc[name_test, 'Lgbm'] = roc
            ap_df.loc[name_test, 'Lgbm'] = ap
            disc_cum_gain_df.loc[name_test, 'Lgbm'] = dcg
            arp_df.loc[name_test, 'Lgbm'] = arp
            precision_df.loc[name_test, 'Lgbm'] = precision
            rbp_df.loc[name_test, 'Lgbm'] = rbp
            uplift_df.loc[name_test, 'Lgbm'] = uplift
            ep_df.loc[name_test, 'Lgbm'] = ep
            n_found_df.loc[name_test, 'Lgbm'] = n_found
            n_found_0_1_df.loc[name_test, 'Lgbm'] = n_found_0_1
            n_found_0_2_df.loc[name_test, 'Lgbm'] = n_found_0_2
            n_found_0_3_df.loc[name_test, 'Lgbm'] = n_found_0_3
            n_found_0_4_df.loc[name_test, 'Lgbm'] = n_found_0_4
            n_found_0_5_df.loc[name_test, 'Lgbm'] = n_found_0_5
            ep_1_3_df.loc[name_test, 'Lgbm'] = ep_1_3
            ep_1_2_df.loc[name_test, 'Lgbm'] = ep_1_2
            ep_2_3_df.loc[name_test, 'Lgbm'] = ep_2_3

            roc, ap, precision, dcg, arp, rbp, uplift, ep, \
            n_found, _, n_found_0_1, n_found_0_2, n_found_0_3, n_found_0_4, n_found_0_5, ep_1_3, ep_1_2, ep_2_3 = performances(
                predict, y_c_test,
                n_ratio=n_ratio,
                n_p_prec=n_p_prec,
                p_rbp=p_rbp,
                n_p_ep=n_p_ep,
                n_n_found=n_n_found,
                cost=True)

            roc_auc_c_df.loc[name_test, 'Lgbm'] = roc
            ap_c_df.loc[name_test, 'Lgbm'] = ap
            disc_cum_gain_c_df.loc[name_test, 'Lgbm'] = dcg
            arp_c_df.loc[name_test, 'Lgbm'] = arp
            precision_c_df.loc[name_test, 'Lgbm'] = precision
            rbp_c_df.loc[name_test, 'Lgbm'] = rbp
            uplift_c_df.loc[name_test, 'Lgbm'] = uplift
            ep_c_df.loc[name_test, 'Lgbm'] = ep
            n_found_c_df.loc[name_test, 'Lgbm'] = n_found
            n_found_0_1_c_df.loc[name_test, 'Lgbm'] = n_found_0_1
            n_found_0_2_c_df.loc[name_test, 'Lgbm'] = n_found_0_2
            n_found_0_3_c_df.loc[name_test, 'Lgbm'] = n_found_0_3
            n_found_0_4_c_df.loc[name_test, 'Lgbm'] = n_found_0_4
            n_found_0_5_c_df.loc[name_test, 'Lgbm'] = n_found_0_5
            ep_1_3_c_df.loc[name_test, 'Lgbm'] = ep_1_3
            ep_1_2_c_df.loc[name_test, 'Lgbm'] = ep_1_2
            ep_2_3_c_df.loc[name_test, 'Lgbm'] = ep_2_3

        if 'ENSImb' in methods:
            predict = ens.predict_proba(model_ensimb, X_test)
            roc, ap, precision, dcg, arp, rbp, uplift, ep, \
            n_found, _, n_found_0_1, n_found_0_2, n_found_0_3, n_found_0_4, n_found_0_5, ep_1_3, ep_1_2, ep_2_3 = performances(
                predict, y_test,
                n_ratio=n_ratio,
                n_p_prec=n_p_prec,
                p_rbp=p_rbp,
                n_p_ep=n_p_ep,
                n_n_found=n_n_found,
                cost=False)

            roc_auc_df.loc[name_test, 'ENSImb'] = roc
            ap_df.loc[name_test, 'ENSImb'] = ap
            disc_cum_gain_df.loc[name_test, 'ENSImb'] = dcg
            arp_df.loc[name_test, 'ENSImb'] = arp
            precision_df.loc[name_test, 'ENSImb'] = precision
            rbp_df.loc[name_test, 'ENSImb'] = rbp
            uplift_df.loc[name_test, 'ENSImb'] = uplift
            ep_df.loc[name_test, 'ENSImb'] = ep
            n_found_df.loc[name_test, 'ENSImb'] = n_found
            n_found_0_1_df.loc[name_test, 'ENSImb'] = n_found_0_1
            n_found_0_2_df.loc[name_test, 'ENSImb'] = n_found_0_2
            n_found_0_3_df.loc[name_test, 'ENSImb'] = n_found_0_3
            n_found_0_4_df.loc[name_test, 'ENSImb'] = n_found_0_4
            n_found_0_5_df.loc[name_test, 'ENSImb'] = n_found_0_5
            ep_1_3_df.loc[name_test, 'ENSImb'] = ep_1_3
            ep_1_2_df.loc[name_test, 'ENSImb'] = ep_1_2
            ep_2_3_df.loc[name_test, 'ENSImb'] = ep_2_3

            roc, ap, precision, dcg, arp, rbp, uplift, ep, \
            n_found, _, n_found_0_1, n_found_0_2, n_found_0_3, n_found_0_4, n_found_0_5, ep_1_3, ep_1_2, ep_2_3 = performances(
                predict, y_c_test,
                n_ratio=n_ratio,
                n_p_prec=n_p_prec,
                p_rbp=p_rbp,
                n_p_ep=n_p_ep,
                n_n_found=n_n_found,
                cost=True)

            roc_auc_c_df.loc[name_test, 'ENSImb'] = roc
            ap_c_df.loc[name_test, 'ENSImb'] = ap
            disc_cum_gain_c_df.loc[name_test, 'ENSImb'] = dcg
            arp_c_df.loc[name_test, 'ENSImb'] = arp
            precision_c_df.loc[name_test, 'ENSImb'] = precision
            rbp_c_df.loc[name_test, 'ENSImb'] = rbp
            uplift_c_df.loc[name_test, 'ENSImb'] = uplift
            ep_c_df.loc[name_test, 'ENSImb'] = ep
            n_found_c_df.loc[name_test, 'ENSImb'] = n_found
            n_found_0_1_c_df.loc[name_test, 'ENSImb'] = n_found_0_1
            n_found_0_2_c_df.loc[name_test, 'ENSImb'] = n_found_0_2
            n_found_0_3_c_df.loc[name_test, 'ENSImb'] = n_found_0_3
            n_found_0_4_c_df.loc[name_test, 'ENSImb'] = n_found_0_4
            n_found_0_5_c_df.loc[name_test, 'ENSImb'] = n_found_0_5
            ep_1_3_c_df.loc[name_test, 'ENSImb'] = ep_1_3
            ep_1_2_c_df.loc[name_test, 'ENSImb'] = ep_1_2
            ep_2_3_c_df.loc[name_test, 'ENSImb'] = ep_2_3

    performance_tables(name,
                       roc_auc_df, ap_df, disc_cum_gain_df, arp_df, precision_df, rbp_df, uplift_df, ep_df,
                       n_found_df, n_found_0_1_df, n_found_0_2_df, n_found_0_3_df, n_found_0_4_df, n_found_0_5_df,
                       ep_1_3_df, ep_1_2_df, ep_2_3_df,
                       roc_auc_c_df, ap_c_df, disc_cum_gain_c_df, arp_c_df, precision_c_df, rbp_c_df, uplift_c_df,
                       ep_c_df, n_found_c_df, n_found_0_1_c_df, n_found_0_2_c_df, n_found_0_3_c_df, n_found_0_4_c_df,
                       n_found_0_5_c_df,
                       ep_1_3_c_df, ep_1_2_c_df, ep_2_3_c_df)


def performances(y_pred, y_test, n_ratio, n_p_prec, p_rbp, n_p_ep, n_n_found, cost=False):
    n = n_ratio * len(y_test)
    n_prec = n_p_prec
    p_ep = n_p_ep / len(y_test)
    n_ep = max(1 / (p_ep), 1 / (1 - p_ep))

    try:
        roc = PerformanceMetrics.performance_metrics_roc_auc(y_pred, y_test, n=n)
        ap = PerformanceMetrics.performance_metrics_ap(y_pred, y_test, n=n)
        precision = PerformanceMetrics.performance_metrics_precision(y_pred, y_test, n_prec, n=n)
        dcg = PerformanceMetrics.performance_metrics_dcg(y_pred, y_test, n=n)
        arp = PerformanceMetrics.performance_metrics_arp(y_pred, y_test, n=n)
        rbp = PerformanceMetrics.performance_metrics_rbp(y_pred, y_test, p_rbp, n=n)
        uplift = PerformanceMetrics.performance_metrics_uplift(y_pred, y_test, n=n)
        ep = PerformanceMetrics.performance_metrics_ep(y_pred, y_test, p_ep, n_ep, n=n)

        n_found = PerformanceMetrics.performance_metrics_n_found(y_pred, y_test, n_n_found, n=n)
        n_found_0_1 = PerformanceMetrics.performance_metrics_n_found(y_pred, y_test, 0.1 * len(y_test), n=n)
        n_found_0_2 = PerformanceMetrics.performance_metrics_n_found(y_pred, y_test, 0.2 * len(y_test), n=n)
        n_found_0_3 = PerformanceMetrics.performance_metrics_n_found(y_pred, y_test, 0.3 * len(y_test), n=n)
        n_found_0_4 = PerformanceMetrics.performance_metrics_n_found(y_pred, y_test, 0.4 * len(y_test), n=n)
        n_found_0_5 = PerformanceMetrics.performance_metrics_n_found(y_pred, y_test, 0.5 * len(y_test), n=n)
        qloss = PerformanceMetrics.qloss(y_pred, y_test)
        ep_1_3 = PerformanceMetrics.performance_metrics_ep(y_pred, y_test, 1 / 3, 3, n=n)
        ep_1_2 = PerformanceMetrics.performance_metrics_ep(y_pred, y_test, 1 / 2, 2, n=n)
        ep_2_3 = PerformanceMetrics.performance_metrics_ep(y_pred, y_test, 2 / 3, 3, n=n)

        if cost == False:
            roc = roc / PerformanceMetrics.performance_metrics_roc_auc(y_pred, y_test, maximum=True, n=n)
            ap = ap / PerformanceMetrics.performance_metrics_ap(y_pred, y_test, maximum=True, n=n)
            precision = precision / PerformanceMetrics.performance_metrics_precision(y_pred, y_test, n_prec,
                                                                                     maximum=True, n=n)
            dcg = dcg / PerformanceMetrics.performance_metrics_dcg(y_pred, y_test, maximum=True, n=n)
            arp = arp / PerformanceMetrics.performance_metrics_arp(y_pred, y_test, maximum=True, n=n)
            rbp = rbp / PerformanceMetrics.performance_metrics_rbp(y_pred, y_test, p_rbp, maximum=True, n=n)
            uplift = uplift / PerformanceMetrics.performance_metrics_uplift(y_pred, y_test, maximum=True, n=n)
            ep = ep / PerformanceMetrics.performance_metrics_ep(y_pred, y_test, p_ep, n_ep, maximum=True, n=n)

            n_found = PerformanceMetrics.performance_metrics_n_found(y_pred, y_test, n_n_found, n=n)
            n_found_0_1 = PerformanceMetrics.performance_metrics_n_found(y_pred, y_test, 0.1 * len(y_test), n=n)
            n_found_0_2 = PerformanceMetrics.performance_metrics_n_found(y_pred, y_test, 0.2 * len(y_test), n=n)
            n_found_0_3 = PerformanceMetrics.performance_metrics_n_found(y_pred, y_test, 0.3 * len(y_test), n=n)
            n_found_0_4 = PerformanceMetrics.performance_metrics_n_found(y_pred, y_test, 0.4 * len(y_test), n=n)
            n_found_0_5 = PerformanceMetrics.performance_metrics_n_found(y_pred, y_test, 0.5 * len(y_test), n=n)
            qloss = PerformanceMetrics.qloss(y_pred, y_test)
            ep_1_3 = PerformanceMetrics.performance_metrics_ep(y_pred, y_test, 1 / 3, 3, n=n)
            ep_1_2 = PerformanceMetrics.performance_metrics_ep(y_pred, y_test, 1 / 2, 2, n=n)
            ep_2_3 = PerformanceMetrics.performance_metrics_ep(y_pred, y_test, 2 / 3, 3, n=n)

    except:

        roc, ap, precision, dcg, arp, rbp, uplift, ep, \
        n_found, qloss, n_found_0_1, n_found_0_2, n_found_0_3, n_found_0_4, n_found_0_5, ep_1_3, ep_1_2, ep_2_3 = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    return roc, ap, precision, dcg, arp, rbp, uplift, ep, n_found, qloss, n_found_0_1, \
           n_found_0_2, n_found_0_3, n_found_0_4, n_found_0_5, ep_1_3, ep_1_2, ep_2_3


def cross_validation_t_v(methods, par_dic, X, y, y_c, train_index, validation_index, name, n_ratio, n_p_prec, p_rbp,
                         p_ep_val, n_n_found, cost_train, cost_validate, perf_ind):

    X_train, y_train, y_c_train, X_val, y_val, y_c_val, datapipeline, st_scaler = divide_clean(X, y, y_c, train_index,
                                                                                               validation_index)

    print('cross validation for ' + name)

    par_dict = copy.deepcopy(par_dic)
    dict = copy.deepcopy(par_dic)

    for k in dict:
        dict[k].clear()

    if 'Logit' in methods:
        cart_prod_log = list(product(*par_dict.get('Logit').values()))
        keys_log = par_dict.get('Logit').keys()
        par_dict_log = par_dict.fromkeys(keys_log)
        per_matrix_log = np.zeros(len(cart_prod_log))

    if 'Lgbm' in methods:
        cart_prod_lgbm = list(product(*par_dict.get('Lgbm').values()))
        keys_lgbm = par_dict.get('Lgbm').keys()
        par_dict_lgbm = par_dict.fromkeys(keys_lgbm)
        per_matrix_lgbm = np.zeros(len(cart_prod_lgbm))

    if 'ENSImb' in methods:
        cart_prod_ens = list(product(*par_dict.get('ENSImb').values()))
        keys_ens = par_dict.get('ENSImb').keys()
        par_dict_ens = par_dict.fromkeys(keys_ens)
        per_matrix_ens = np.zeros(len(cart_prod_ens))

    if 'Logit' in methods:

        for j, value in enumerate(cart_prod_log):
            for i, key in enumerate(keys_log):
                par_dict_log.update({key: value[i]})

            if cost_train == False:
                model = MethodLearner.logit(par_dict_log, X_train, y_train, y_train)
                predict = model.predict_proba(X_val)

            if cost_train == True:
                model = MethodLearner.logit(par_dict_log, X_train, y_c_train, y_train)
                predict = model.predict_proba(X_val)

            if cost_validate == False:
                roc, ap, precision, dcg, arp, rbp, uplift, ep, n_found, ql, _, _, _, _, _, _, _, _ = performances(
                    predict, y_val,
                    n_ratio=n_ratio,
                    n_p_prec=n_p_prec,
                    p_rbp=p_rbp,
                    n_p_ep=p_ep_val * len(
                        y_val),
                    n_n_found=n_n_found,
                    cost=False)

            if cost_validate == True:
                roc, ap, precision, dcg, arp, rbp, uplift, ep, n_found, ql, _, _, _, _, _, _, _, _ = performances(
                    predict, y_c_val,
                    n_ratio=n_ratio,
                    n_p_prec=n_p_prec,
                    p_rbp=p_rbp,
                    n_p_ep=p_ep_val * len(
                        y_val),
                    n_n_found=n_n_found,
                    cost=True)

            if perf_ind == 'ap':
                per_matrix_log[j] += ap

            if perf_ind == 'roc_auc':
                per_matrix_log[j] += roc

            if perf_ind == 'dcg':
                per_matrix_log[j] += dcg

            if perf_ind == 'arp':
                per_matrix_log[j] += arp

            if perf_ind == 'precision':
                per_matrix_log[j] += precision

            if perf_ind == 'rbp':
                per_matrix_log[j] += rbp

            if perf_ind == 'uplift':
                per_matrix_log[j] += uplift

            if perf_ind == 'ep':
                per_matrix_log[j] += ep

            if perf_ind == 'ql':
                per_matrix_log[j] += ql

    if 'Lgbm' in methods:

        for j, value in enumerate(cart_prod_lgbm):
            for i, key in enumerate(keys_lgbm):
                par_dict_lgbm.update({key: value[i]})

            if cost_train == False:
                lgbmboost, model = MethodLearner.lgbmboost(par_dict_lgbm, X_train, y_train, y_train)
                predict = lgbmboost.predict_proba(model, X_val)

            if cost_train == True:
                lgbmboost, model = MethodLearner.lgbmboost(par_dict_lgbm, X_train,
                                                           y_c_train, y_train)
                predict = lgbmboost.predict_proba(model, X_val)

            if cost_validate == False:
                roc, ap, precision, dcg, arp, rbp, uplift, ep, n_found, ql, _, _, _, _, _, _, _, _ = performances(
                    predict, y_val,
                    n_ratio=n_ratio,
                    n_p_prec=n_p_prec,
                    p_rbp=p_rbp,
                    n_p_ep=p_ep_val * len(
                        y_val),
                    n_n_found=n_n_found,
                    cost=False)

            if cost_validate == True:
                roc, ap, precision, dcg, arp, rbp, uplift, ep, n_found, ql, _, _, _, _, _, _, _, _ = performances(
                    predict, y_c_val,
                    n_ratio=n_ratio,
                    n_p_prec=n_p_prec,
                    p_rbp=p_rbp,
                    n_p_ep=p_ep_val * len(
                        y_val),
                    n_n_found=n_n_found,
                    cost=True)

            if perf_ind == 'ap':
                per_matrix_lgbm[j] += ap

            if perf_ind == 'roc_auc':
                per_matrix_lgbm[j] += roc

            if perf_ind == 'dcg':
                per_matrix_lgbm[j] += dcg

            if perf_ind == 'arp':
                per_matrix_lgbm[j] += arp

            if perf_ind == 'precision':
                per_matrix_lgbm[j] += precision

            if perf_ind == 'rbp':
                per_matrix_lgbm[j] += rbp

            if perf_ind == 'uplift':
                per_matrix_lgbm[j] += uplift

            if perf_ind == 'ep':
                per_matrix_lgbm[j] += ep

            if perf_ind == 'ql':
                per_matrix_lgbm[j] += ql

    if 'ENSImb' in methods:

        for j, value in enumerate(cart_prod_ens):
            for i, key in enumerate(keys_ens):
                par_dict_ens.update({key: value[i]})

            ensimb, model = MethodLearner.ensimb(par_dict_ens, X_train, y_train)
            predict = ensimb.predict_proba(model, X_val)

            if cost_validate == False:
                roc, ap, precision, dcg, arp, rbp, uplift, ep, n_found, ql, _, _, _, _, _, _, _, _ = performances(
                    predict, y_val,
                    n_ratio=n_ratio,
                    n_p_prec=n_p_prec,
                    p_rbp=p_rbp,
                    n_p_ep=p_ep_val * len(
                        y_val),
                    n_n_found=n_n_found,
                    cost=False)

            if cost_validate == True:
                roc, ap, precision, dcg, arp, rbp, uplift, ep, n_found, ql, _, _, _, _, _, _, _, _ = performances(
                    predict, y_c_val,
                    n_ratio=n_ratio,
                    n_p_prec=n_p_prec,
                    p_rbp=p_rbp,
                    n_p_ep=p_ep_val * len(
                        y_val),
                    n_n_found=n_n_found,
                    cost=False)

            if perf_ind == 'ap':
                per_matrix_ens[j] += ap

            if perf_ind == 'roc_auc':
                per_matrix_ens[j] += roc

            if perf_ind == 'dcg':
                per_matrix_ens[j] += dcg

            if perf_ind == 'arp':
                per_matrix_ens[j] += arp

            if perf_ind == 'precision':
                per_matrix_ens[j] += precision

            if perf_ind == 'rbp':
                per_matrix_ens[j] += rbp

            if perf_ind == 'uplift':
                per_matrix_ens[j] += uplift

            if perf_ind == 'ep':
                per_matrix_ens[j] += ep

            if perf_ind == 'ql':
                per_matrix_ens[j] += ql

    if 'Logit' in methods:

        max_ind = np.nanargmax(per_matrix_log)
        max_values = cart_prod_log[max_ind]

        for i, key in enumerate(keys_log):
            dict['Logit'][key] = max_values[i]

    if 'Lgbm' in methods:

        max_ind = np.nanargmax(per_matrix_lgbm)
        max_values = cart_prod_lgbm[max_ind]

        for i, key in enumerate(keys_lgbm):
            dict['Lgbm'][key] = max_values[i]

    if 'ENSImb' in methods:

        max_ind = np.nanargmax(per_matrix_ens)
        max_values = cart_prod_ens[max_ind]

        for i, key in enumerate(keys_ens):
            dict['ENSImb'][key] = max_values[i]

    print(name + ' The optimal hyperparameters are' + str(dict))

    return dict, X_train, y_train, y_c_train, X_val, y_val, y_c_val, datapipeline, st_scaler
