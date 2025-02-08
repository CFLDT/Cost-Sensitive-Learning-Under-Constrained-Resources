import numpy as np

from ..plots_tables import dec_boundary_plotter
from ..design import MethodLearner
from ALIGNED_learning.design import PerformanceMetrics

def decision_boundary(methods, par_dict, X_train, y_train, y_train_clas, task_dict):

    name = task_dict['name']

    xx, yy = np.mgrid[X_train[:, 0].min() - 2:X_train[:, 0].max() + 2:0.02,
             X_train[:, 1].min() - 2:X_train[:, 1].max() + 2:0.02]

    grid = np.c_[xx.ravel(), yy.ravel()]

    for method in methods:

        if method == 'Logit':

            model = MethodLearner.logit(par_dict.get('Logit'), X_train, y_train, y_train_clas)
            grid_probs = model.predict_proba(grid)
            name_2 = par_dict.get('Logit').get('metric')
            method_name = method + '_' + name_2

        elif method == 'Lgbm':

            lgboost, model = MethodLearner.lgbmboost(par_dict.get('Lgbm'), X_train, y_train, y_train_clas)
            grid_probs = lgboost.predict_proba(model, grid)
            name_2 = par_dict.get('Lgbm').get('metric')
            method_name = method + '_' + name_2


        else:

            continue

        # n_ratio = par_dict.get('General_val_test').get("n_ratio")
        # n_p_prec = par_dict.get('General_val_test').get("n_p_prec")
        # p_rbp = par_dict.get('General_val_test').get("p_rbp")
        # n_p_ep = par_dict.get('General_val_test').get("n_p_ep")
        # n_n_found = par_dict.get('General_val_test').get("n_n_found")
        #
        # n = n_ratio * len(y_train)
        # n_prec = n_p_prec
        # p_ep = n_p_ep / len(y_train)
        # n_ep = max(1 / (p_ep), 1 / (1 - p_ep))
        #
        # if method == 'Logit':
        #     predict_prob = model.predict_proba(X_train)
        #
        # elif method == 'Lgbm':
        #     predict_prob = lgboost.predict_proba(model, X_train)
        #
        # roc = PerformanceMetrics.performance_metrics_roc_auc(predict_prob, y_train, n=n)
        # ap = PerformanceMetrics.performance_metrics_ap(predict_prob, y_train, n=n)
        # precision = PerformanceMetrics.performance_metrics_precision(predict_prob, y_train, n_prec, n=n)
        # dcg = PerformanceMetrics.performance_metrics_dcg(predict_prob, y_train, n=n)
        # arp = PerformanceMetrics.performance_metrics_arp(predict_prob, y_train, n=n)
        # rbp = PerformanceMetrics.performance_metrics_rbp(predict_prob, y_train, p_rbp, n=n)
        # uplift = PerformanceMetrics.performance_metrics_uplift(predict_prob, y_train, n=n)
        # ep = PerformanceMetrics.performance_metrics_ep(predict_prob, y_train, p_ep, n_ep, n=n)
        # n_found = PerformanceMetrics.performance_metrics_n_found(predict_prob, y_train, n_n_found, n=n)


        dec_boundary_plotter(name, method_name, X_train,  y_train, xx, yy, grid_probs)
