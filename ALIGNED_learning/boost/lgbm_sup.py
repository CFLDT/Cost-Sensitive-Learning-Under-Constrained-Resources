from timeit import default_timer as timer
import lightgbm as lgb
from .lgbm import Lightgbm
import numpy as np
import random
import sys, os
from scipy.stats import betabinom, beta
from ALIGNED_learning.design.performance_metrics import PerformanceMetrics


class Lgbm(Lightgbm):

    def __init__(self, n_estimators, num_leaves, reg_lambda, learning_rate, subsample, min_child_samples,
                 min_child_weight, sigma, indic_approx):

        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.min_child_samples = min_child_samples
        self.min_child_weight = min_child_weight
        self.sigma = sigma
        self.indic_approx = indic_approx

        super().__init__(sigma,indic_approx)

    def fitting(self, X, y, metric, n = None, p_prec = None, p_rbp = None, p_ep = None, n_c_ep = None):
        random.seed(2290)
        np.random.seed(2290)

        starttimer = timer()

        if metric == 'basic':

            if np.array_equal(y, y.astype(bool)) == True:

                model = lgb.LGBMClassifier(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                       learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                       subsample=self.subsample,
                                       min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                       verbose=-1)

            else:

                raise ValueError("Cost sensitive basic LGBM is not implemented")

        if metric == 'lambdarank':
            self.n = int(n * len(y))

            model = lgb.LGBMRanker(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                   learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                   subsample=self.subsample, lambdarank_truncation_level=self.n,
                                   lambdarank_norm=False,
                                   min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                   verbose=-1, objective='lambdarank')

        if metric == 'arp':
            model = lgb.LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                   learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                   subsample=self.subsample,
                                   min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                   verbose=-1, objective=self.arp)

            self.arp_max = PerformanceMetrics.performance_metrics_arp(y, y, maximum=True)


        if metric == 'lambdarank_1':
            model = lgb.LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                   learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                   subsample=self.subsample,
                                   min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                   verbose=-1, objective=self.lambdarank)

            n_ones = np.count_nonzero(y)
            n_zeros = len(y) - n_ones

            n_ones = int(np.sum(y))
            array_ones = np.ones(n_ones)
            array_linspace = np.linspace(0, n_ones - 1, n_ones)
            self.lambdmax_max = np.sum((2 ** (array_ones) - 1) / (np.log2(array_linspace + 2)))

        if metric == 'dcg':
            model = lgb.LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                   learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                   subsample=self.subsample,
                                   min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                   verbose=-1, objective=self.dcg)
            self.ndcg_max = PerformanceMetrics.performance_metrics_dcg(y, y, maximum=True)

        if metric == 'roc_auc':
            model = lgb.LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                   learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                   subsample=self.subsample,
                                   min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                   verbose=-1, objective=self.roc_auc)
            self.roc_auc_max = PerformanceMetrics.performance_metrics_roc_auc(y, y, maximum=True)

        if metric == 'ap':
            model = lgb.LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                   learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                   subsample=self.subsample,
                                   min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                   verbose=-1, objective=self.ap)

            self.ap_max = PerformanceMetrics.performance_metrics_ap(y, y, maximum=True)

        if metric == 'ep':

            self.p_ep = p_ep
            self.n_c_ep = n_c_ep

            disc = 0
            discounter = np.zeros(len(y))

            for i in range(len(y), 0, -1):

                if i < len(y):
                    top = beta.cdf((i + 0.5) / len(y), p_ep * n_c_ep, n_c_ep * (1 - p_ep))
                else:
                    top = 1

                bot = beta.cdf((i - 0.5) / len(y), p_ep * n_c_ep, n_c_ep * (1 - p_ep))

                prob = top - bot

                disc = disc + (prob / (i))
                discounter[i - 1] = disc

            self.discounter = discounter*len(y)
            self.ep_max = PerformanceMetrics.performance_metrics_ep(y, y, self.p_ep, self.n_c_ep, maximum=True)


            model = lgb.LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                   learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                   subsample=self.subsample,
                                   min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                   verbose=-1, objective=self.ep)

        if metric == 'rbp':

            self.p_rbp = p_rbp

            model = lgb.LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                   learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                   subsample=self.subsample,
                                   min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                   verbose=-1, objective=self.rbp)

            self.rbp_max = PerformanceMetrics.performance_metrics_rbp(y, y, self.p_rbp, maximum=True)

        if metric == 'precision':

            self.n_prec = int(p_prec*len(y))

            model = lgb.LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                   learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                   subsample=self.subsample,
                                   min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                   verbose=-1, objective=self.precision)

            self.precision_max = PerformanceMetrics.performance_metrics_precision(y, y, self.n_prec, maximum=True)

        # Disable
        def blockPrint():
            sys.stdout = open(os.devnull, 'w')

        # Restore
        def enablePrint():
            sys.stdout = sys.__stdout__

        # Output of Lightgbm is kinda annoying, and verbose is bugged
        blockPrint()
        if metric != 'basic':
            #model.fit(X, y, group=[X.shape[0]])
            model.fit(X, y)
        if metric == 'basic':
            model.fit(X, y)
        enablePrint()

        endtimer = timer()

        return model, endtimer - starttimer
