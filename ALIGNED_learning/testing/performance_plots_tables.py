import numpy as np
import pandas as pd
from itertools import product
import json
import copy

from ALIGNED_learning.design import MethodLearner
from ALIGNED_learning.design import PerformanceMetrics
from ALIGNED_learning.plots_tables import performance_tables, plot_feature_imp_shap
from ALIGNED_learning.design import DivClean

from sklearn.preprocessing import StandardScaler
from pathlib import Path
import math

base_path = Path(__file__).parent


def performance_check(methods, par_dict_init, X, y, y_c, m_score, f_score, name_list,
                      train_list, validate_list, test_list, feature_importance, cross_val_perf_ind,
                      cost_train, cost_validate, keep_first):

    n_ratio = par_dict_init.get('General_val_test').get("n_ratio")
    n_p_prec = par_dict_init.get('General_val_test').get("n_p_prec")
    p_rbp = par_dict_init.get('General_val_test').get("p_rbp")
    n_p_ep = par_dict_init.get('General_val_test').get("n_p_ep")
    n_p_ep_val = par_dict_init.get('General_val_test').get("n_p_ep_val")

    #n_c_ep = par_dict_init.get('General_val_test').get("n_c_ep")
    n_n_found = par_dict_init.get('General_val_test').get("n_n_found")


    X_train_list = []
    X_validation_list = []
    X_test_list = []

    y_train_list = []
    y_validation_list = []
    y_test_list = []

    y_cost_train_list = []
    y_cost_validation_list = []
    y_cost_test_list = []

    m_score_test_list = []
    f_score_test_list = []

    par_dict_list = []
    datapipeline_list = []

    skip_cross_validate = False
    par_dict_init_cv = par_dict_init.copy()

    for names, train_indexs, validation_indexs, test_indexs \
            in zip(name_list, train_list, validate_list, test_list):

        X_trains = []
        X_validations = []
        X_tests = []

        y_trains = []
        y_validations = []
        y_tests = []

        y_cost_trains = []
        y_cost_validations = []
        y_cost_tests = []

        m_score_tests = []
        f_score_tests = []

        par_dicts = []
        datapipelines = []

        for train_index, validation_index in zip(train_indexs, validation_indexs):
            X_trains.append(X.iloc[train_index])
            y_trains.append(np.array(y.iloc[train_index]))
            y_cost_trains.append(np.array(y_c.iloc[train_index]))

            X_validations.append(X.iloc[validation_index])
            y_validations.append(np.array(y.iloc[validation_index]))
            y_cost_validations.append(np.array(y_c.iloc[validation_index]))

            nam_spl = names[0].split('_')
            name = nam_spl[0] + '_' + nam_spl[1] + '_' + nam_spl[2] + '_' + nam_spl[3] + '_' + nam_spl[4]
            print('Cross validation ' + name)

            par_dict, datapipeline = cross_validation_train_val(
                methods=methods, par_dict_init_cv=par_dict_init_cv, X=X,
                y=y, y_c=y_c, train_index=train_index, validation_index=validation_index, name=name,
                perf_ind=cross_val_perf_ind,
                n_ratio=n_ratio, p_prec=n_p_prec, p_rbp=p_rbp, n_p_ep_val=n_p_ep_val, n_n_found = n_n_found, cost_train=cost_train,
                cost_validate=cost_validate, skip_cross_validate = skip_cross_validate)

            par_dicts.append(par_dict)
            datapipelines.append(datapipeline)

            if keep_first == True:
                par_dict_init_cv = par_dict
                skip_cross_validate = True

        X_train_list.append(X_trains)
        X_validation_list.append(X_validations)

        y_train_list.append(y_trains)
        y_validation_list.append(y_validations)

        y_cost_train_list.append(y_cost_trains)
        y_cost_validation_list.append(y_cost_validations)

        par_dict_list.append(par_dicts)
        datapipeline_list.append(datapipelines)

        for test_index in test_indexs:
            X_tests.append(X.iloc[test_index])
            y_tests.append(np.array(y.iloc[test_index]))
            y_cost_tests.append(np.array(y_c.iloc[test_index]))

            m_score_tests.append(np.array(m_score.iloc[test_index]))
            f_score_tests.append(np.array(f_score.iloc[test_index]))

        X_test_list.append(X_tests)
        y_test_list.append(y_tests)
        y_cost_test_list.append(y_cost_tests)

        m_score_test_list.append(m_score_tests)
        f_score_test_list.append(f_score_tests)

    roc_auc_df = pd.DataFrame()
    ap_df = pd.DataFrame()
    disc_cum_gain_df = pd.DataFrame()
    arp_df = pd.DataFrame()
    precision_df = pd.DataFrame()
    rbp_df = pd.DataFrame()
    uplift_df = pd.DataFrame()
    ep_df = pd.DataFrame()
    n_found_df = pd.DataFrame()

    roc_auc_c_df = pd.DataFrame()
    ap_c_df = pd.DataFrame()
    disc_cum_gain_c_df = pd.DataFrame()
    arp_c_df = pd.DataFrame()
    precision_c_df = pd.DataFrame()
    rbp_c_df = pd.DataFrame()
    uplift_c_df = pd.DataFrame()
    ep_c_df = pd.DataFrame()
    n_found_c_df = pd.DataFrame()

    if 'Logit' in methods:
        roc_auc_df['Logit'] = ""
        ap_df['Logit'] = ""
        disc_cum_gain_df['Logit'] = ""
        arp_df['Logit'] = ""
        precision_df['Logit'] = ""
        rbp_df['Logit'] = ""
        uplift_df['Logit'] = ""
        ep_df['Logit'] = ""
        n_found_df["Logit"] = ""

        roc_auc_c_df['Logit'] = ""
        ap_c_df['Logit'] = ""
        disc_cum_gain_c_df['Logit'] = ""
        arp_c_df['Logit'] = ""
        precision_c_df['Logit'] = ""
        rbp_c_df['Logit'] = ""
        uplift_c_df['Logit'] = ""
        ep_c_df['Logit'] = ""
        n_found_c_df["Logit"] = ""

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

        roc_auc_c_df['Lgbm'] = ""
        ap_c_df['Lgbm'] = ""
        disc_cum_gain_c_df['Lgbm'] = ""
        arp_c_df['Lgbm'] = ""
        precision_c_df['Lgbm'] = ""
        rbp_c_df['Lgbm'] = ""
        uplift_c_df['Lgbm'] = ""
        ep_c_df['Lgbm'] = ""
        n_found_c_df["Lgbm"] = ""

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

        roc_auc_c_df['ENSImb'] = ""
        ap_c_df['ENSImb'] = ""
        disc_cum_gain_c_df['ENSImb'] = ""
        arp_c_df['ENSImb'] = ""
        precision_c_df['ENSImb'] = ""
        rbp_c_df['ENSImb'] = ""
        uplift_c_df['ENSImb'] = ""
        ep_c_df['ENSImb'] = ""
        n_found_c_df["ENSImb"] = ""

    if 'M_score' in methods:
        roc_auc_df['M_score'] = ""
        ap_df['M_score'] = ""
        disc_cum_gain_df['M_score'] = ""
        arp_df['M_score'] = ""
        precision_df['M_score'] = ""
        rbp_df['M_score'] = ""
        uplift_df['M_score'] = ""
        ep_df['M_score'] = ""
        n_found_df["M_score"] = ""

        roc_auc_c_df['M_score'] = ""
        ap_c_df['M_score'] = ""
        disc_cum_gain_c_df['M_score'] = ""
        arp_c_df['M_score'] = ""
        precision_c_df['M_score'] = ""
        rbp_c_df['M_score'] = ""
        uplift_c_df['M_score'] = ""
        ep_c_df['M_score'] = ""
        n_found_c_df["M_score"] = ""

    if 'F_score' in methods:
        roc_auc_df['F_score'] = ""
        ap_df['F_score'] = ""
        disc_cum_gain_df['F_score'] = ""
        arp_df['F_score'] = ""
        precision_df['F_score'] = ""
        rbp_df['F_score'] = ""
        uplift_df['F_score'] = ""
        ep_df['F_score'] = ""
        n_found_df["F_score"] = ""

        roc_auc_c_df['F_score'] = ""
        ap_c_df['F_score'] = ""
        disc_cum_gain_c_df['F_score'] = ""
        arp_c_df['F_score'] = ""
        precision_c_df['F_score'] = ""
        rbp_c_df['F_score'] = ""
        uplift_c_df['F_score'] = ""
        ep_c_df['F_score'] = ""
        n_found_c_df["F_score"] = ""

    for i in range(np.shape(train_list)[0]):
        for j in range(len(test_list[i])):

            X_train = X_train_list[i][j]
            y_train = y_train_list[i][j]
            y_cost_train = y_cost_train_list[i][j]

            X_train_imp, X_test_imp, _ = scaler(
                np.array(datapipeline.pipeline_trans(X_train)))
            X_train_na, X_test_na, _ = scaler(
                np.array(datapipeline.pipeline_trans(X_train, keep_nan=True)))

            par_dict_opt = par_dict_list[i][j]
            datapipeline = datapipeline_list[i][j]

            if 'Logit' in methods:

                if cost_train == False:
                    model_logit = MethodLearner.logit(par_dict_opt.get('Logit'),
                                                      X_train_imp, y_train, y_train)

                if cost_train == True:
                    model_logit = MethodLearner.logit(par_dict_opt.get('Logit'),
                                                      X_train_imp, y_cost_train,y_train)

            if 'Lgbm' in methods:

                if cost_train == False:
                    lgbmboost, model_lgbm = MethodLearner.lgbmboost(par_dict_opt.get('Lgbm'),
                                                                    X_train_na, y_train, y_train)
                if cost_train == True:
                    lgbmboost, model_lgbm = MethodLearner.lgbmboost(par_dict_opt.get('Lgbm'),
                                                                    X_train_na, y_cost_train, y_train)

            if 'ENSImb' in methods:
                ens, model_ensimb = MethodLearner.ensimb(par_dict_opt.get('ENSImb'), X_train_imp,
                                                         y_train)  # RusboostClassifier can't handle non-binairy data

            for k in range(len(test_list[i])):

                X_test = X_test_list[i][k]
                y_test = y_test_list[i][k]
                y_cost_test = y_cost_test_list[i][k]

                if X_test.empty:
                    continue

                m_score_test = m_score_test_list[i][k]
                f_score_test = f_score_test_list[i][k]

                name_test = name_list[i][k]
                nam_spl = name_test.split('_')
                name = nam_spl[0] + '_' + nam_spl[1] + '_' + nam_spl[2] + '_' + nam_spl[3] + '_' + nam_spl[4]

                X_train_imp, X_test_imp, _ = scaler(
                    np.array(datapipeline.pipeline_trans(X_train)),
                    np.array(datapipeline.pipeline_trans(X_test)))
                X_train_na, X_test_na, _ = scaler(
                    np.array(datapipeline.pipeline_trans(X_train, keep_nan=True)),
                    np.array(datapipeline.pipeline_trans(X_test, keep_nan=True)))

                if 'Logit' in methods:

                    predict = model_logit.predict_proba(X_test_imp)
                    roc, ap, precision, dcg, arp, rbp, uplift, ep, n_found = performances(predict, y_test, n_ratio=n_ratio,
                                                                                          n_p_prec=n_p_prec, p_rbp=p_rbp,
                                                                                          n_p_ep=n_p_ep, n_n_found = n_n_found)

                    roc_auc_df.loc[name_test, 'Logit'] = roc
                    ap_df.loc[name_test, 'Logit'] = ap
                    disc_cum_gain_df.loc[name_test, 'Logit'] = dcg
                    arp_df.loc[name_test, 'Logit'] = arp
                    precision_df.loc[name_test, 'Logit'] = precision
                    rbp_df.loc[name_test, 'Logit'] = rbp
                    uplift_df.loc[name_test, 'Logit'] = uplift
                    ep_df.loc[name_test, 'Logit'] = ep
                    n_found_df.loc[name_test, 'Logit'] = n_found

                    roc, ap, precision, dcg, arp, rbp, uplift, ep, n_found = performances(predict, y_cost_test, n_ratio=n_ratio,
                                                                                          n_p_prec=n_p_prec,
                                                                                          p_rbp=p_rbp,
                                                                                          n_p_ep=n_p_ep,
                                                                                          n_n_found = n_n_found, cost=True)

                    roc_auc_c_df.loc[name_test, 'Logit'] = roc
                    ap_c_df.loc[name_test, 'Logit'] = ap
                    disc_cum_gain_c_df.loc[name_test, 'Logit'] = dcg
                    arp_c_df.loc[name_test, 'Logit'] = arp
                    precision_c_df.loc[name_test, 'Logit'] = precision
                    rbp_c_df.loc[name_test, 'Logit'] = rbp
                    uplift_c_df.loc[name_test, 'Logit'] = uplift
                    ep_c_df.loc[name_test, 'Logit'] = ep
                    n_found_c_df.loc[name_test, 'Logit'] = n_found

                    if ((feature_importance == True) & (k == 0)):
                        plot_feature_imp_shap('Logit', name, model_logit, X_train_imp, y_train, datapipeline.colnames)

                if 'Lgbm' in methods:

                    predict = lgbmboost.predict_proba(model_lgbm, X_test_na)
                    roc, ap, precision, dcg, arp, rbp, uplift, ep, n_found = performances(predict, y_test, n_ratio=n_ratio,
                                                                                          n_p_prec=n_p_prec, p_rbp=p_rbp,
                                                                                          n_p_ep=n_p_ep,
                                                                                          n_n_found = n_n_found, cost=False)

                    roc_auc_df.loc[name_test, 'Lgbm'] = roc
                    ap_df.loc[name_test, 'Lgbm'] = ap
                    disc_cum_gain_df.loc[name_test, 'Lgbm'] = dcg
                    arp_df.loc[name_test, 'Lgbm'] = arp
                    precision_df.loc[name_test, 'Lgbm'] = precision
                    rbp_df.loc[name_test, 'Lgbm'] = rbp
                    uplift_df.loc[name_test, 'Lgbm'] = uplift
                    ep_df.loc[name_test, 'Lgbm'] = ep
                    n_found_df.loc[name_test, 'Lgbm'] = n_found

                    roc, ap, precision, dcg, arp, rbp, uplift, ep, n_found = performances(predict, y_cost_test, n_ratio=n_ratio,
                                                                                          n_p_prec=n_p_prec,
                                                                                          p_rbp=p_rbp,
                                                                                          n_p_ep=n_p_ep,
                                                                                          n_n_found = n_n_found, cost=True)

                    roc_auc_c_df.loc[name_test, 'Lgbm'] = roc
                    ap_c_df.loc[name_test, 'Lgbm'] = ap
                    disc_cum_gain_c_df.loc[name_test, 'Lgbm'] = dcg
                    arp_c_df.loc[name_test, 'Lgbm'] = arp
                    precision_c_df.loc[name_test, 'Lgbm'] = precision
                    rbp_c_df.loc[name_test, 'Lgbm'] = rbp
                    uplift_c_df.loc[name_test, 'Lgbm'] = uplift
                    ep_c_df.loc[name_test, 'Lgbm'] = ep
                    n_found_c_df.loc[name_test, 'Lgbm'] = n_found

                    if ((feature_importance == True) & (k == 0)):
                        plot_feature_imp_shap('Lgbm', name, model_lgbm, X_train_na, y_train, datapipeline.colnames)

                if 'ENSImb' in methods:

                    predict = ens.predict_proba(model_ensimb, X_test_imp)
                    roc, ap, precision, dcg, arp, rbp, uplift, ep, n_found = performances(predict, y_test, n_ratio=n_ratio,
                                                                                          n_p_prec=n_p_prec, p_rbp=p_rbp,
                                                                                          n_p_ep=n_p_ep,
                                                                                          n_n_found = n_n_found, cost=False)

                    roc_auc_df.loc[name_test, 'ENSImb'] = roc
                    ap_df.loc[name_test, 'ENSImb'] = ap
                    disc_cum_gain_df.loc[name_test, 'ENSImb'] = dcg
                    arp_df.loc[name_test, 'ENSImb'] = arp
                    precision_df.loc[name_test, 'ENSImb'] = precision
                    rbp_df.loc[name_test, 'ENSImb'] = rbp
                    uplift_df.loc[name_test, 'ENSImb'] = uplift
                    ep_df.loc[name_test, 'ENSImb'] = ep
                    n_found_df.loc[name_test, 'ENSImb'] = n_found

                    roc, ap, precision, dcg, arp, rbp, uplift, ep, n_found = performances(predict, y_cost_test, n_ratio=n_ratio,
                                                                                          n_p_prec=n_p_prec,
                                                                                          p_rbp=p_rbp,
                                                                                          n_p_ep=n_p_ep,
                                                                                          n_n_found = n_n_found, cost=True)

                    roc_auc_c_df.loc[name_test, 'ENSImb'] = roc
                    ap_c_df.loc[name_test, 'ENSImb'] = ap
                    disc_cum_gain_c_df.loc[name_test, 'ENSImb'] = dcg
                    arp_c_df.loc[name_test, 'ENSImb'] = arp
                    precision_c_df.loc[name_test, 'ENSImb'] = precision
                    rbp_c_df.loc[name_test, 'ENSImb'] = rbp
                    uplift_c_df.loc[name_test, 'ENSImb'] = uplift
                    ep_c_df.loc[name_test, 'ENSImb'] = ep
                    n_found_c_df.loc[name_test, 'ENSImb'] = n_found

                    if ((feature_importance == True) & (k == 0)):
                        plot_feature_imp_shap('ENSImb', name, model_ensimb, X_train_imp, y_train, datapipeline.colnames)

                if 'M_score' in methods:
                    predict = m_score_test

                    roc, ap, precision, dcg, arp, rbp, uplift, ep, n_found = performances(predict, y_test, n_ratio=n_ratio,
                                                                                          n_p_prec=n_p_prec,
                                                                                          p_rbp=p_rbp,
                                                                                          n_p_ep=n_p_ep,
                                                                                          n_n_found = n_n_found, cost=False)

                    roc_auc_df.loc[name_test, 'M_score'] = roc
                    ap_df.loc[name_test, 'M_score'] = ap
                    disc_cum_gain_df.loc[name_test, 'M_score'] = dcg
                    arp_df.loc[name_test, 'M_score'] = arp
                    precision_df.loc[name_test, 'M_score'] = precision
                    rbp_df.loc[name_test, 'M_score'] = rbp
                    uplift_df.loc[name_test, 'M_score'] = uplift
                    ep_df.loc[name_test, 'M_score'] = ep
                    n_found_df.loc[name_test, 'M_score'] = n_found

                    roc, ap, precision, dcg, arp, rbp, uplift, ep, n_found = performances(predict, y_cost_test, n_ratio=n_ratio,
                                                                                          n_p_prec=n_p_prec,
                                                                                          p_rbp=p_rbp,
                                                                                          n_p_ep=n_p_ep,
                                                                                          n_n_found = n_n_found, cost=True)

                    roc_auc_c_df.loc[name_test, 'M_score'] = roc
                    ap_c_df.loc[name_test, 'M_score'] = ap
                    disc_cum_gain_c_df.loc[name_test, 'M_score'] = dcg
                    arp_c_df.loc[name_test, 'M_score'] = arp
                    precision_c_df.loc[name_test, 'M_score'] = precision
                    rbp_c_df.loc[name_test, 'M_score'] = rbp
                    uplift_c_df.loc[name_test, 'M_score'] = uplift
                    ep_c_df.loc[name_test, 'M_score'] = ep
                    n_found_c_df.loc[name_test, 'M_score'] = n_found

                if 'F_score' in methods:
                    predict = f_score_test

                    roc, ap, precision, dcg, arp, rbp, uplift, ep, n_found = performances(predict, y_test, n_ratio=n_ratio,
                                                                                          n_p_prec=n_p_prec,
                                                                                          p_rbp=p_rbp,
                                                                                          n_p_ep=n_p_ep,
                                                                                          n_n_found = n_n_found, cost=False)
                    roc_auc_df.loc[name_test, 'F_score'] = roc
                    ap_df.loc[name_test, 'F_score'] = ap
                    disc_cum_gain_df.loc[name_test, 'F_score'] = dcg
                    arp_df.loc[name_test, 'F_score'] = arp
                    precision_df.loc[name_test, 'F_score'] = precision
                    rbp_df.loc[name_test, 'F_score'] = rbp
                    uplift_df.loc[name_test, 'F_score'] = uplift
                    ep_df.loc[name_test, 'F_score'] = ep
                    n_found_df.loc[name_test, 'F_score'] = n_found

                    roc, ap, precision, dcg, arp, rbp, uplift, ep, n_found = performances(predict, y_cost_test, n_ratio=n_ratio,
                                                                                          n_p_prec=n_p_prec,
                                                                                          p_rbp=p_rbp,
                                                                                          n_p_ep=n_p_ep,
                                                                                          n_n_found = n_n_found, cost=True)

                    roc_auc_c_df.loc[name_test, 'F_score'] = roc
                    ap_c_df.loc[name_test, 'F_score'] = ap
                    disc_cum_gain_c_df.loc[name_test, 'F_score'] = dcg
                    arp_c_df.loc[name_test, 'F_score'] = arp
                    precision_c_df.loc[name_test, 'F_score'] = precision
                    rbp_c_df.loc[name_test, 'F_score'] = rbp
                    uplift_c_df.loc[name_test, 'F_score'] = uplift
                    ep_c_df.loc[name_test, 'F_score'] = ep
                    n_found_c_df.loc[name_test, 'F_score'] = n_found

    if roc_auc_df.empty == False:
        performance_tables(nam_spl[0] + '_' + nam_spl[1] + '_' + nam_spl[2],
                           roc_auc_df, ap_df, disc_cum_gain_df, arp_df, precision_df, rbp_df, uplift_df, ep_df,
                           n_found_df,
                           roc_auc_c_df, ap_c_df, disc_cum_gain_c_df, arp_c_df, precision_c_df, rbp_c_df, uplift_c_df,
                           ep_c_df, n_found_c_df)


def performances(predict_prob, y_test, n_ratio, n_p_prec, p_rbp, n_p_ep, n_n_found, cost=False):

    n = n_ratio * len(y_test)

    n_prec = n_p_prec
    p_ep = n_p_ep / len(y_test)
    n_ep = math.ceil(1/(p_ep * (1-p_ep)))
    n_ep = max(1 / (p_ep), 1 / (1 - p_ep))

    try:
        roc = PerformanceMetrics.performance_metrics_roc_auc(predict_prob, y_test, n=n)
        ap = PerformanceMetrics.performance_metrics_ap(predict_prob, y_test, n=n)
        precision = PerformanceMetrics.performance_metrics_precision(predict_prob, y_test, n_prec, n=n)
        dcg = PerformanceMetrics.performance_metrics_dcg(predict_prob, y_test, n=n)
        arp = PerformanceMetrics.performance_metrics_arp(predict_prob, y_test, n=n)
        rbp = PerformanceMetrics.performance_metrics_rbp(predict_prob, y_test, p_rbp, n=n)
        uplift = PerformanceMetrics.performance_metrics_uplift(predict_prob, y_test, n=n)
        ep = PerformanceMetrics.performance_metrics_ep(predict_prob, y_test, p_ep, n_ep, n=n)
        n_found = PerformanceMetrics.performance_metrics_n_found(predict_prob, y_test, n_n_found, n=n)

        if cost == False:
            roc = roc / PerformanceMetrics.performance_metrics_roc_auc(predict_prob, y_test, maximum=True, n=n)
            ap = ap / PerformanceMetrics.performance_metrics_ap(predict_prob, y_test, maximum=True, n=n)
            precision = precision / PerformanceMetrics.performance_metrics_precision(predict_prob, y_test, n_prec,
                                                                                     maximum=True, n=n)
            dcg = dcg / PerformanceMetrics.performance_metrics_dcg(predict_prob, y_test, maximum=True, n=n)
            arp = arp / PerformanceMetrics.performance_metrics_arp(predict_prob, y_test, maximum=True, n=n)
            rbp = rbp / PerformanceMetrics.performance_metrics_rbp(predict_prob, y_test, p_rbp, maximum=True, n=n)
            uplift = uplift / PerformanceMetrics.performance_metrics_uplift(predict_prob, y_test, maximum=True, n=n)
            ep = ep / PerformanceMetrics.performance_metrics_ep(predict_prob, y_test, p_ep, n_ep, maximum=True, n=n)
            n_found = PerformanceMetrics.performance_metrics_n_found(predict_prob, y_test, n_n_found, n=n)

    except:

        roc, ap, precision, dcg, arp, rbp, uplift, ep, n_found = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    return roc, ap, precision, dcg, arp, rbp, uplift, ep, n_found


def cross_validation_train_val(methods, par_dict_init_cv, X, y, y_c, train_index, validation_index,
                               name, perf_ind, n_ratio, p_prec, p_rbp, n_p_ep_val, n_n_found, cost_train, cost_validate, skip_cross_validate):

    datapipeline = DivClean.divide_clean(X, train_index, validation_index)

    par_dict = copy.deepcopy(par_dict_init_cv)
    dict = copy.deepcopy(par_dict)

    if skip_cross_validate == False:

        for k in dict:
            dict[k].clear()

        X_train = X.iloc[train_index]
        X_val = X.iloc[validation_index]

        y_train = np.array(y.iloc[train_index])
        y_val = np.array(y.iloc[validation_index])

        y_cost_train = np.array(y_c.iloc[train_index])
        y_cost_val = np.array(y_c.iloc[validation_index])

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

        X_train_imp, X_val_imp, _ = scaler(
            np.array(datapipeline.pipeline_trans(X_train)),
            np.array(datapipeline.pipeline_trans(X_val)))
        X_train_na, X_val_na, _ = scaler(
            np.array(datapipeline.pipeline_trans(X_train, keep_nan=True)),
            np.array(datapipeline.pipeline_trans(X_val, keep_nan=True)))

        if 'Logit' in methods:

            for j, value in enumerate(cart_prod_log):
                for i, key in enumerate(keys_log):
                    par_dict_log.update({key: value[i]})

                if cost_train == False:
                    model = MethodLearner.logit(par_dict_log, X_train_imp, y_train, y_train)
                    predict = model.predict_proba(X_val_imp)

                if cost_train == True:
                    model = MethodLearner.logit(par_dict_log, X_train_imp, y_cost_train, y_train)
                    predict = model.predict_proba(X_val_imp)

                if cost_validate == False:
                    roc, ap, precision, dcg, arp, rbp, uplift, ep, n_found = performances(predict, y_val, n_ratio=n_ratio,
                                                                                          n_p_prec=p_prec, p_rbp=p_rbp,
                                                                                          n_p_ep=n_p_ep_val,
                                                                                          n_n_found=n_n_found, cost=False)

                if cost_validate == True:
                    roc, ap, precision, dcg, arp, rbp, uplift, ep, n_found = performances(predict, y_cost_val, n_ratio=n_ratio,
                                                                                          n_p_prec=p_prec, p_rbp=p_rbp,
                                                                                          n_p_ep=n_p_ep_val,
                                                                                          n_n_found=n_n_found, cost=True)

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

        if 'Lgbm' in methods:

            for j, value in enumerate(cart_prod_lgbm):
                for i, key in enumerate(keys_lgbm):
                    par_dict_lgbm.update({key: value[i]})

                if cost_train == False:
                    lgbmboost, model = MethodLearner.lgbmboost(par_dict_lgbm, X_train_na, y_train, y_train)
                    predict = lgbmboost.predict_proba(model, X_val_na)

                if cost_train == True:
                    lgbmboost, model = MethodLearner.lgbmboost(par_dict_lgbm, X_train_na,
                                                               y_cost_train, y_train)
                    predict = lgbmboost.predict_proba(model, X_val_na)

                if cost_validate == False:
                    roc, ap, precision, dcg, arp, rbp, uplift, ep, n_found = performances(predict, y_val, n_ratio=n_ratio,
                                                                                          n_p_prec=p_prec, p_rbp=p_rbp,
                                                                                          n_p_ep=n_p_ep_val,
                                                                                          n_n_found=n_n_found, cost=False)

                if cost_validate == True:
                    roc, ap, precision, dcg, arp, rbp, uplift, ep, n_found = performances(predict, y_cost_val, n_ratio=n_ratio,
                                                                                          n_p_prec=p_prec, p_rbp=p_rbp,
                                                                                          n_p_ep=n_p_ep_val,
                                                                                          n_n_found=n_n_found, cost=True)

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

        if 'ENSImb' in methods:

            for j, value in enumerate(cart_prod_ens):
                for i, key in enumerate(keys_ens):
                    par_dict_ens.update({key: value[i]})

                ensimb, model = MethodLearner.ensimb(par_dict_ens, X_train_imp, y_train)
                predict = ensimb.predict_proba(model, X_val_imp)

                if cost_validate == False:
                    roc, ap, precision, dcg, arp, rbp, uplift, ep, n_found = performances(predict, y_val, n_ratio=n_ratio,
                                                                                          n_p_prec=p_prec, p_rbp=p_rbp,
                                                                                          n_p_ep=n_p_ep_val,
                                                                                          n_n_found=n_n_found, cost=False)

                if cost_validate == True:
                    roc, ap, precision, dcg, arp, rbp, uplift, ep, n_found = performances(predict, y_cost_val, n_ratio=n_ratio,
                                                                                          n_p_prec=p_prec, p_rbp=p_rbp,
                                                                                          n_p_ep=n_p_ep_val,
                                                                                          n_n_found=n_n_found, cost=False)

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

    print(name + ' The optimal hyperparameters are ' + str(dict))

    dic_name = name + '_dict.txt'
    path = (base_path / "../../tables/tables experiment info" / dic_name).resolve()
    with open(path, 'w') as f:
        json.dump(dict, f)

    return dict, datapipeline


def scaler(X_train_or, X_test_or=None):
    st_scaler = StandardScaler()
    st_scaler.fit(X_train_or)
    X_train = st_scaler.transform(X_train_or)
    if X_test_or is not None:
        X_test = st_scaler.transform(X_test_or)
    else:
        X_test = None

    return X_train, X_test, st_scaler
