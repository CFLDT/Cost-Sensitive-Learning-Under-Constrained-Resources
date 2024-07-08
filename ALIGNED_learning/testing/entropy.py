
from scipy.optimize import fmin_slsqp
import numpy as np
from ALIGNED_learning.design import PerformanceMetrics
from ALIGNED_learning.plots_tables import informativeness_plotter
import numpy as np
from scipy.stats import binom, beta, betabinom
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import random

def inferred_p_calculator(y_pred, y_true, metric, n_relevant_value, p_ep = None, n_c_ep= None, p_rbp = None, n_prec =None, n = None):


    #original

    if metric == 'arp':
        max_value = PerformanceMetrics.performance_metrics_arp(y_pred, y_true, maximum=True, n=n)
        metric_value = PerformanceMetrics.performance_metrics_arp(y_pred, y_true, maximum=False, n=n)

    if metric == 'dcg':
        max_value = PerformanceMetrics.performance_metrics_dcg(y_pred, y_true, maximum=True, n=n)
        metric_value = PerformanceMetrics.performance_metrics_dcg(y_pred, y_true, maximum=False, n=n)

    if metric == 'ap':
        max_value = PerformanceMetrics.performance_metrics_ap(y_pred, y_true, maximum=True, n=n)
        metric_value = PerformanceMetrics.performance_metrics_ap(y_pred, y_true, maximum=False, n=n)

    if metric == 'ep':
        max_value = PerformanceMetrics.performance_metrics_ep(y_pred, y_true, p_ep, n_c_ep, maximum=True, n=n)
        metric_value = PerformanceMetrics.performance_metrics_ep(y_pred, y_true, p_ep, n_c_ep, maximum=False, n=n)

    if metric == 'rbp':
        max_value = PerformanceMetrics.performance_metrics_rbp(y_pred, y_true, p_rbp, maximum=True, n=n)
        metric_value = PerformanceMetrics.performance_metrics_rbp(y_pred, y_true, p_rbp, maximum=False, n=n)

    if metric == 'precision':
        max_value = PerformanceMetrics.performance_metrics_precision(y_pred, y_true, n_prec, maximum=True, n=n)
        metric_value = PerformanceMetrics.performance_metrics_precision(y_pred, y_true, n_prec, maximum=False, n=n)

    if metric == 'uplift':
        max_value = PerformanceMetrics.performance_metrics_uplift(y_pred, y_true, maximum=True, n=n)
        metric_value = PerformanceMetrics.performance_metrics_uplift(y_pred, y_true, maximum=False, n=n)

    if metric == 'roc_auc':
        max_value = PerformanceMetrics.performance_metrics_roc_auc(y_pred, y_true, maximum=True, n=n)
        metric_value = PerformanceMetrics.performance_metrics_roc_auc(y_pred, y_true, maximum=False, n=n)

    # inferred

    if metric == 'arp':
        def metric_constraint(true_values):
            discounter = np.zeros(len(true_values))
            for i in range(len(true_values)):
                discounter[i] = len(true_values) - i + 1  # index starts at zero, thus +1

            gains = true_values
            arp = np.sum(np.multiply(gains, discounter))

            return arp - metric_value

    if metric == 'dcg':
        def metric_constraint(true_values):
            discounter = np.zeros(len(true_values))
            for i in range(len(true_values)):
                discounter[i] = 1 / (np.log2(i + 2))  # index starts at zero, thus +2

            gains = (2 ** true_values) - 1
            dcg = np.sum(np.multiply(gains, discounter))

            return (dcg - metric_value) / max_value

    if metric == 'ap':
        def metric_constraint(true_values):
            discounter = np.zeros(len(true_values))
            gains = np.zeros(len(true_values))

            for i in range(len(true_values)):
                discounter[i] = 1 / (i + 1)
                gains[i] = true_values[i] * np.sum(true_values[0:i + 1])

            ap = np.sum(np.multiply(discounter, gains))

            return (ap - metric_value) / max_value

    if metric == 'ep':

        discounter = np.zeros(len(y_true))

        disc = 0

        for i in range(len(y_true), 0, -1):

            if i < len(y_true):
                top = beta.cdf((i + 0.5) / len(y_true), p_ep * n_c_ep, n_c_ep * (1 - p_ep))
            else:
                top = 1

            bot = beta.cdf((i - 0.5) / len(y_true), p_ep * n_c_ep, n_c_ep * (1 - p_ep))

            prob = top - bot

            disc = disc + (prob / (i))
            discounter[i - 1] = disc


        def metric_constraint(true_values):

            gains = true_values
            ep = np.sum(np.multiply(gains, discounter*len(true_values)))

            return (ep - metric_value) / max_value

    if metric == 'rbp':
        def metric_constraint(true_values):

            discounter = np.zeros(len(true_values))

            for i in range(len(true_values)):
                discounter[i] = p_rbp ** (i)

            gains = true_values
            rbp = np.sum(np.multiply(discounter, gains))  # can do times (1 - p_rbp)  to normalize

            return (rbp - metric_value) / max_value


    if metric == 'precision':
        def metric_constraint(true_values):

            discounter = np.zeros(len(true_values))

            for i in range(len(true_values)):
                if i < n_prec:
                    discounter[i] = (1 / n_prec)
            gains = true_values

            precision = np.sum(np.multiply(discounter, gains))

            return (precision - metric_value) / max_value

    if metric == 'uplift':
        def metric_constraint(true_values):

            discounter = np.zeros(len(true_values))

            disc = 0
            for i in range(len(true_values), 0, -1):
                disc = disc + 1 / (i)
                discounter[i - 1] = disc

            gains = true_values
            uplift = np.sum(np.multiply(gains, discounter))

            return (uplift - metric_value) / max_value

    if metric == 'roc_auc':
        def metric_constraint(true_values):

            discounter = np.zeros(len(true_values))
            gains = np.zeros(len(true_values))

            for i in range(len(true_values)):
                discounter[i] = 1
                gains[i] = (1 - true_values[i]) * np.sum(true_values[0:i + 1])

            roc_auc = np.sum(np.multiply(discounter, gains))

            return (roc_auc - metric_value) / max_value



    def sum_constraint(x):
        return np.sum(x) - n_relevant_value

    def objective(x):
        return np.sum(-x*np.log(x)-(1-x)*np.log(1-x))

    lower = np.zeros(len(y_true))
    upper = np.ones(len(y_true))
    bounds = list(zip(lower, upper))

    x0 = np.ones(len(y_true)) / np.sum(np.ones(len(y_true)) )

    inferred_p = fmin_slsqp(objective, x0=x0 , bounds=bounds, iter = 300, eqcons=[sum_constraint,metric_constraint])

    return inferred_p


def target_calculator(y_pred, y_true, y_infer, target_metric, p_inferred, p_ep = None, n_c_ep= None, p_rbp = None, n_prec = None, n = None):

    if target_metric == 'arp':
        actual_target_metric_value = PerformanceMetrics.performance_metrics_arp(y_pred, y_true, maximum=False, n=n) / \
                                     PerformanceMetrics.performance_metrics_arp(y_pred, y_true, maximum=True, n=n)
        inferred_target_metric_value = PerformanceMetrics.performance_metrics_arp(y_pred, p_inferred, maximum=False, n=n) / \
                                       PerformanceMetrics.performance_metrics_arp(y_pred, y_infer, maximum=True, n=n)
    if target_metric == 'dcg':
        actual_target_metric_value = PerformanceMetrics.performance_metrics_dcg(y_pred, y_true, maximum=False, n=n) / \
                                     PerformanceMetrics.performance_metrics_dcg(y_pred, y_true, maximum=True, n=n)
        inferred_target_metric_value = PerformanceMetrics.performance_metrics_dcg(y_pred, p_inferred, maximum=False, n=n) / \
                                       PerformanceMetrics.performance_metrics_dcg(y_pred, y_infer, maximum=True, n=n)
    if target_metric == 'ap':
        actual_target_metric_value = PerformanceMetrics.performance_metrics_ap(y_pred, y_true, maximum=False, n=n) / \
                                     PerformanceMetrics.performance_metrics_ap(y_pred, y_true, maximum=True, n=n)
        inferred_target_metric_value = PerformanceMetrics.performance_metrics_ap(y_pred, p_inferred, maximum=False, n=n) / \
                                       PerformanceMetrics.performance_metrics_ap(y_pred, y_infer, maximum=True, n=n)
    if target_metric == 'ep':
        actual_target_metric_value = PerformanceMetrics.performance_metrics_ep(y_pred, y_true, p_ep, n_c_ep, maximum=False, n=n) / \
                                     PerformanceMetrics.performance_metrics_ep(y_pred, y_true, p_ep, n_c_ep, maximum=True, n=n)
        inferred_target_metric_value = PerformanceMetrics.performance_metrics_ep(y_pred, p_inferred, p_ep, n_c_ep, maximum=False, n=n) / \
                                       PerformanceMetrics.performance_metrics_ep(y_pred, y_infer, p_ep, n_c_ep, maximum=True, n=n)
    if target_metric == 'rbp':
        actual_target_metric_value = PerformanceMetrics.performance_metrics_rbp(y_pred, y_true, p_rbp, maximum=False, n=n) / \
                                     PerformanceMetrics.performance_metrics_rbp(y_pred, y_true, p_rbp, maximum=True, n=n)
        inferred_target_metric_value = PerformanceMetrics.performance_metrics_rbp(y_pred, p_inferred, p_rbp, maximum=False, n=n) / \
                                       PerformanceMetrics.performance_metrics_rbp(y_pred, y_infer, p_rbp, maximum=True, n=n)
    if target_metric == 'precision':
        actual_target_metric_value = PerformanceMetrics.performance_metrics_precision(y_pred, y_true,n_prec, maximum=False, n=n) / \
                                     PerformanceMetrics.performance_metrics_precision(y_pred, y_true,n_prec, maximum=True, n=n)
        inferred_target_metric_value = PerformanceMetrics.performance_metrics_precision(y_pred, p_inferred,n_prec, maximum=False, n=n) / \
                                       PerformanceMetrics.performance_metrics_precision(y_pred, y_infer,n_prec, maximum=True, n=n)
    if target_metric == 'uplift':
        actual_target_metric_value = PerformanceMetrics.performance_metrics_uplift(y_pred, y_true, maximum=False, n=n) / \
                                     PerformanceMetrics.performance_metrics_uplift(y_pred, y_true, maximum=True, n=n)
        inferred_target_metric_value = PerformanceMetrics.performance_metrics_uplift(y_pred, p_inferred, maximum=False, n=n) / \
                                       PerformanceMetrics.performance_metrics_uplift(y_pred, y_infer, maximum=True, n=n)
    if target_metric == 'roc_auc':
        actual_target_metric_value = PerformanceMetrics.performance_metrics_roc_auc(y_pred, y_true, maximum=False, n=n) / \
                                     PerformanceMetrics.performance_metrics_roc_auc(y_pred, y_true, maximum=True, n=n)
        inferred_target_metric_value = PerformanceMetrics.performance_metrics_roc_auc(y_pred, p_inferred, maximum=False, n=n) / \
                                       PerformanceMetrics.performance_metrics_roc_auc(y_pred, y_infer, maximum=True, n=n)

    return actual_target_metric_value,inferred_target_metric_value




# dice example

# y = np.zeros(6)
#
# def sum_constraint(x):
#     return np.sum(x) - 1
#
# weight = np.arange(len(y))+1
#
# def metric_constraint(x):
#     return np.sum(np.multiply(x,weight)) - 5.5
#
# def objective(x):
#     return np.sum(-x*np.log(x))  #np.sum(-x*np.log(x)-(1-x)*np.log(1-x))
#
#
# lower = np.zeros(len(y))
# upper = np.ones(len(y))
#
# bounds = list(zip(lower, upper))
#
# x0 = np.ones(len(y)) * 0.5 / np.sum(np.ones(len(y)) * 0.5)
# result = fmin_slsqp(objective, x0=x0 , bounds=bounds, eqcons=[sum_constraint,metric_constraint])
# print(result)
