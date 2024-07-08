from timeit import default_timer as timer
import numpy as np
import random
from .logit import Lgt
from scipy.stats import betabinom, beta
from ALIGNED_learning.design.performance_metrics import PerformanceMetrics
import math


class Logit(Lgt):

    def __init__(self, lambd, sigma, indic_approx, theta=None):

        super().__init__(lambd, sigma, indic_approx, theta)

    def fitting(self, X, y, y_clas, init_theta, metric, n_ratio = None, p_prec=None, p_rbp=None, p_ep=None):

        random.seed(2290)
        np.random.seed(2290)

        starttimer = timer()

        self.n = int(n_ratio * len(y))

        div = 1 / X.shape[0]

        if metric == 'basic':

            if np.array_equal(y, y.astype(bool)) == True:

                weight_1 = y
                weight_0 = 1 - y

                def obj_func(theta):
                    return self.objective_function_basic(theta, X, div, weight_1, weight_0)

                self.theta, func_min = self.optimization_basic(obj_func, init_theta)

            else:

                raise ValueError("Cost sensitive basic logistic regression is not implemented")

        if metric == 'arp':

            self.arp_max = PerformanceMetrics.performance_metrics_arp(y, y, maximum=True)

            def obj_func(theta):
                return self.arp(theta, X, y)

            self.theta, func_min = self.optimization_rank(obj_func, init_theta)

        if metric == 'roc_auc':

            self.roc_auc_max = PerformanceMetrics.performance_metrics_roc_auc(y, y, maximum=True)

            def obj_func(theta):
                return self.roc_auc(theta, X, y)

            self.theta, func_min = self.optimization_rank(obj_func, init_theta)

        if metric == 'ap':

            self.ap_max = PerformanceMetrics.performance_metrics_ap(y, y, maximum=True)

            def obj_func(theta):
                return self.ap(theta, X, y)

            self.theta, func_min = self.optimization_rank(obj_func, init_theta)

        if metric == 'dcg':

            self.ndcg_max = PerformanceMetrics.performance_metrics_dcg(y, y, maximum=True)

            def obj_func(theta):
                return self.dcg(theta, X, y)

            self.theta, func_min = self.optimization_rank(obj_func, init_theta)

        if metric == 'ep':

            self.p_ep = p_ep
            self.n_ep = math.ceil(1 / (p_ep * (1 - p_ep)))
            #self.n_ep = max(1/(p_ep), 1/(1 - p_ep))

            discounter = np.zeros(len(y))
            disc = 0
            for i in range(len(y), 0, -1):

                if i < len(y):
                    top = beta.cdf((i + 0.5) / len(y), self.p_ep * self.n_ep, self.n_ep * (1 - self.p_ep))
                else:
                    top = 1

                bot = beta.cdf((i - 0.5) / len(y), self.p_ep * self.n_ep, self.n_ep * (1 - self.p_ep))

                prob = top - bot

                disc = disc + (prob / (i))
                discounter[i - 1] = disc

            self.discounter = discounter*len(y)
            self.ep_max = PerformanceMetrics.performance_metrics_ep(y, y, self.p_ep, self.n_ep, maximum=True)

            def obj_func(theta):
                return self.ep(theta, X, y)

            self.theta, func_min = self.optimization_rank(obj_func, init_theta)

        if metric == 'rbp':
            self.p_rbp = p_rbp
            self.rbp_max = PerformanceMetrics.performance_metrics_rbp(y, y, self.p_rbp, maximum=True)

            def obj_func(theta):
                return self.rbp(theta, X, y)

            self.theta, func_min = self.optimization_rank(obj_func, init_theta)

        if metric == 'precision':
            self.n_prec = int(p_prec*len(y))
            self.precision_max = PerformanceMetrics.performance_metrics_precision(y, y, self.n_prec, maximum=True)

            def obj_func(theta):
                return self.precision(theta, X, y)

            self.theta, func_min = self.optimization_rank(obj_func, init_theta)

        endtimer = timer()

        return func_min, endtimer - starttimer
