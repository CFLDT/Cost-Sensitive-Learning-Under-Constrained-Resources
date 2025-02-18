from ALIGNED_learning.plots_tables import informativeness_plotter, informativeness_plotter_2
from ALIGNED_learning.testing import inferred_p_calculator, target_calculator
import numpy as np
from scipy import stats
import random
import math


#random permutations

random.seed(2290)
np.random.seed(2290)


iterations = 100
prior = 0.1
n_t = 100


p_prec = 0.5
p_ep = 1/2
n_c_ep = max(1 / (p_ep), 1 / (1 - p_ep))
p_rbp = 0.9

original_metric = 'roc_auc'
target_metric = 'roc_auc'

cost_sensitive = True

actual_target_metric_value_array = np.zeros(iterations)
inferred_target_metric_value_array = np.zeros(iterations)



for i in range(iterations):

    y_true = np.random.choice([0, 1], size=(n_t,), p=[1 - prior, prior])

    y_infer = y_true.copy()

    n_relevant_value = np.count_nonzero(y_infer)

    # Predictions

    #prediction_noise = ((i + 30)/2) / iterations      #worse
    prediction_noise = ((i + 1)/2) / iterations

    #y_predic = np.where((np.random.random(n_t) < prediction_noise), 1-y_true, y_true)
    y_predic = np.where((np.random.random(n_t) < prediction_noise), 1-y_infer, y_infer)

    y_pred = np.zeros(n_t)
    y_pred[y_predic==1] = y_predic[y_predic==1] - np.random.uniform(0,0.5,len(y_predic[y_predic==1]))
    y_pred[y_predic==0] = y_predic[y_predic==0] + np.random.uniform(0,0.5,len(y_predic[y_predic==0]))

    if cost_sensitive == True:

        #cost_1
        y_true = y_true
        #cost_2
        # y_true = y_true + (1-y_true)*np.random.uniform(-0.05, 0, len(y_true))
        #cost_3
        # y_true = y_true*np.random.uniform(0, 1, len(y_true)) + (1-y_true)*np.random.uniform(-0.05, 0, len(y_true))
        #cost_4
        # y_true = y_true*np.random.uniform(0, 1, len(y_true)) + (1-y_true)*np.random.uniform(-0.15, 0, len(y_true))
        #cost_5
        # y_true = y_true + (1-y_true)*-np.log(0.05*np.random.exponential(scale = 10, size=len(y_true))+1)
        #cost_6
        # y_true = y_true*np.log(np.random.exponential(scale = 10, size=len(y_true))+1) + \
        #                                 (1-y_true)*-np.log(0.05*np.random.exponential(scale = 10, size=len(y_true))+1)
        #cost_7
        # y_true = y_true*np.log(np.random.exponential(scale = 10, size=len(y_true))+1) + \
        #                                 (1-y_true)*-np.log(0.15*np.random.exponential(scale = 10, size=len(y_true))+1)
        #cost_8
        # y_true =  y_true*np.log(np.random.exponential(scale = 10, size=len(y_true))+1) +\
        #           (1-y_true)*-0.05*np.log(np.random.exponential(scale = 10, size=len(y_true))+1)


    outp = y_pred.argsort()[::-1][:n_t]

    y_true = y_true[outp]
    y_infer = y_infer[outp]
    y_pred = y_pred[outp]

    if cost_sensitive == False:

        # Aslam method
        # inferred_p = inferred_p_calculator(y_pred, y_infer, metric=original_metric, n_relevant_value = n_relevant_value,
        #                                    p_ep=p_ep, n_c_ep=n_c_ep, p_rbp=p_rbp, n_prec = int(p_prec*len(y_true)), n=None)

        inferred_p = y_infer.copy()

        actual_target_metric_value,inferred_target_metric_value = target_calculator(y_pred, y_true, y_infer, target_metric=target_metric,
                                                                                    p_inferred= inferred_p,
                                                                                    p_ep=p_ep, n_c_ep=n_c_ep, p_rbp=p_rbp,
                                                                                    n_prec= int(p_prec*len(y_true)),
                                                                                    n=None)

    if cost_sensitive == True:

        inferred_p = y_infer.copy()

        actual_target_metric_value, inferred_target_metric_value = target_calculator(y_pred, y_true, y_infer,
                                                                                     target_metric=target_metric,
                                                                                     p_inferred=inferred_p,
                                                                                     p_ep=p_ep, n_c_ep=n_c_ep,
                                                                                     p_rbp=p_rbp,
                                                                                     n_prec=int(p_prec * len(y_true)),
                                                                                     n=None)

    actual_target_metric_value_array[i] = actual_target_metric_value.round(decimals=3)
    inferred_target_metric_value_array[i] = inferred_target_metric_value.round(decimals=3)


tau, p_value = stats.kendalltau(actual_target_metric_value_array, inferred_target_metric_value_array)
tau = round(tau, 3)

if cost_sensitive == False:
    informativeness_plotter(target_metric, original_metric, actual_target_metric_value_array, inferred_target_metric_value_array,
                            tau)

if cost_sensitive == True:
    informativeness_plotter_2(target_metric, original_metric, actual_target_metric_value_array, inferred_target_metric_value_array,
                            tau)





# if label_noise == True:
#
#     if label_noise_0 != 0:
#         y_infer = np.where( ((np.random.random(n_t) < label_noise_0) & (y_true == 0)), 1, y_infer)
#     if label_noise_1 != 0:
#         y_infer = np.where( ((np.random.random(n_t) < label_noise_1) & (y_true == 1)), 0, y_infer)