from timeit import default_timer as timer
import lightgbm as lgb
from .lgbm import Lightgbm
import numpy as np
import random
import sys, os
from scipy.stats import betabinom, beta
from ALIGNED_learning.design.performance_metrics import PerformanceMetrics
import math


class Lgbm(Lightgbm):

    def __init__(self, n_estimators, num_leaves, reg_lambda, reg_alpha, learning_rate, colsample_bytree, subsample, subsample_freq,
                 undersample, min_child_samples, min_child_weight, sigma, indic_approx):

        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.learning_rate = learning_rate
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.undersample = undersample
        self.min_child_samples = min_child_samples
        self.min_child_weight = min_child_weight
        self.sigma = sigma
        self.indic_approx = indic_approx

        super().__init__(sigma,indic_approx)

    def fitting(self, X, y, y_clas, metric, n_ratio = None, p_prec = None, p_rbp = None, p_ep = None):

        random.seed(2290)
        np.random.seed(2290)

        starttimer = timer()

        indices_0 = np.where(y_clas == 0)[0]
        indices_1 = np.where(y_clas == 1)[0]
        all_ind = np.arange(len(y_clas))

        if ((self.undersample is not None) or (self.subsample is not None)):
            self.indices_list = []
            self.it_index = 0
            for i in range(self.n_estimators):
                if ((self.undersample is not None) and (self.subsample is None)):
                    ind_0 = np.random.choice(indices_0, size=int(np.rint((1 / self.undersample) * indices_1.shape[0])), replace=False)
                    ind_tr = list(np.append(indices_1, ind_0))
                if ((self.undersample is None) and (self.subsample is not None)):
                    ind = np.random.choice(all_ind, size=int(np.rint(self.subsample * all_ind.shape[0])), replace=False)
                    ind_tr = list(ind)
                if ((self.undersample is not None) and (self.subsample is not None)):
                    ind_0 = np.random.choice(indices_0, size=int(np.rint((1 / self.undersample) * indices_1.shape[0])), replace=False)
                    ind_a = list(np.append(indices_1, ind_0))
                    ind_tr = np.random.choice(ind_a, size=int(np.rint(self.subsample * len(ind_a))), replace=False)

                self.indices_list.append(ind_tr)


        if metric == 'basic':

            if self.undersample is None:
                neg_bagging_fraction = 1

            if self.undersample is not None:
                n_majority = np.count_nonzero(y) / self.undersample
                neg_bagging_fraction = n_majority / (len(y) - np.count_nonzero(y))
                neg_bagging_fraction = min(neg_bagging_fraction, 1)

            if np.array_equal(y, y.astype(bool)) == True:

                model = lgb.LGBMClassifier(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                       learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                       reg_alpha=self.reg_alpha, colsample_bytree = self.colsample_bytree,
                                       subsample=self.subsample,  subsample_freq = self.subsample_freq,
                                       neg_bagging_fraction = neg_bagging_fraction,
                                       min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                       verbose=-1)

            else:

                model = lgb.LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                       learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                       reg_alpha=self.reg_alpha, colsample_bytree = self.colsample_bytree,
                                       subsample=self.subsample,  subsample_freq = self.subsample_freq,
                                       neg_bagging_fraction = neg_bagging_fraction,
                                       min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                       verbose=-1)
        if metric == 'lambdarank':
            self.n = int(n_ratio * len(y))

            model = lgb.LGBMRanker(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                   learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                   reg_alpha=self.reg_alpha, colsample_bytree = self.colsample_bytree,
                                   lambdarank_truncation_level=self.n, lambdarank_norm=False,
                                   min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                   verbose=-1, objective='lambdarank')

        if metric == 'arp':
            model = lgb.LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                   learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                      reg_alpha=self.reg_alpha, colsample_bytree = self.colsample_bytree,
                                   min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                   verbose=-1, objective=self.arp)


        if metric == 'lambdarank_1':
            model = lgb.LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                   learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                      reg_alpha=self.reg_alpha, colsample_bytree = self.colsample_bytree,
                                   min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                   verbose=-1, objective=self.lambdarank)


        if metric == 'dcg':
            model = lgb.LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                   learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                      reg_alpha=self.reg_alpha, colsample_bytree = self.colsample_bytree,
                                   min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                   verbose=-1, objective=self.dcg)

        if metric == 'roc_auc':
            model = lgb.LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                   learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                      reg_alpha=self.reg_alpha, colsample_bytree = self.colsample_bytree,
                                   min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                   verbose=-1, objective=self.roc_auc)

        if metric == 'ap':
            model = lgb.LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                   learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                      reg_alpha=self.reg_alpha, colsample_bytree = self.colsample_bytree,
                                   min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                   verbose=-1, objective=self.ap)


        if metric == 'ep':

            if ((self.undersample is None) and (self.subsample is None)):
                lengh = len(y_clas)
            else:
                lengh = len(ind_tr)

            self.p_ep = p_ep
            #self.n_ep = math.ceil(1 / (p_ep * (1 - p_ep)))
            self.n_ep = max(1/(p_ep), 1/(1 - p_ep))

            disc = 0
            discounter = np.zeros(lengh)

            for i in range(lengh, 0, -1):

                if i < lengh:
                    top = beta.cdf((i + 0.5) / lengh, self.p_ep * self.n_ep, self.n_ep * (1 - self.p_ep))
                else:
                    top = 1

                bot = beta.cdf((i - 0.5) / lengh, self.p_ep * self.n_ep, self.n_ep * (1 - self.p_ep))

                prob = top - bot

                disc = disc + (prob / (i))
                discounter[i - 1] = disc

            self.discounter = discounter*lengh

            model = lgb.LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                   learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                   reg_alpha=self.reg_alpha, colsample_bytree = self.colsample_bytree,
                                      min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                   verbose=-1, objective=self.ep)

        if metric == 'rbp':

            self.p_rbp = p_rbp

            model = lgb.LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                   learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                    reg_alpha=self.reg_alpha, colsample_bytree = self.colsample_bytree,
                                   min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                   verbose=-1, objective=self.rbp)

        if metric == 'precision':

            self.n_prec = int(p_prec*len(y_clas))

            model = lgb.LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                   learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                      reg_alpha=self.reg_alpha, colsample_bytree = self.colsample_bytree,
                                   min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                   verbose=-1, objective=self.precision)

        # Disable
        def blockPrint():
            sys.stdout = open(os.devnull, 'w')

        # Restore
        def enablePrint():
            sys.stdout = sys.__stdout__

        # Output of Lightgbm is kinda annoying, and verbose is bugged
        #blockPrint()
        if metric != 'basic':
            #model.fit(X, y, group=[X.shape[0]])
            model.fit(X, y)
        if metric == 'basic':
            model.fit(X, y)
        #enablePrint()

        endtimer = timer()

        return model, endtimer - starttimer
