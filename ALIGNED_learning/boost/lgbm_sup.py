from timeit import default_timer as timer
import lightgbm as lgb
from .lgbm import Lightgbm
import numpy as np
import random
import sys, os
from scipy.stats import beta


class Lgbm(Lightgbm):

    def __init__(self, n_estimators, num_leaves, reg_lambda, reg_alpha, learning_rate, colsample_bytree, subsample,
                 subsample_freq,
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

        super().__init__(sigma, indic_approx)

    def fitting(self, X, y, y_clas, metric, n_ratio=None, p_prec=None, p_rbp=None, p_ep=None):

        random.seed(2290)
        np.random.seed(2290)

        starttimer = timer()

        lengh = len(y_clas)

        # The code in the comments enables us to use majority undersampling during training on custom (ranking) cost functions.
        #
        # ### Steps to Apply:
        # - Remove the `subsample` keywords in the `LGBMRegressor`.
        # - Uncomment the corresponding modification in the loss function of interest.

        # indices_0 = np.where(y_clas == 0)[0]
        # indices_1 = np.where(y_clas == 1)[0]
        # all_ind = np.arange(len(y_clas))
        #
        # if ((self.undersample is not None) or (self.subsample is not None)):
        #     self.indices_list = []
        #     self.it_index = 0
        #     for i in range(self.n_estimators):
        #         if ((self.undersample is not None) and (self.subsample is None)):
        #             ind_0 = np.random.choice(indices_0, size=int(np.rint((1 / self.undersample) * indices_1.shape[0])),
        #                                      replace=False)
        #             ind_tr = list(np.append(indices_1, ind_0))
        #         if ((self.undersample is None) and (self.subsample is not None)):
        #             ind = np.random.choice(all_ind, size=int(np.rint(self.subsample * all_ind.shape[0])), replace=False)
        #             ind_tr = list(ind)
        #         if ((self.undersample is not None) and (self.subsample is not None)):
        #             ind_0 = np.random.choice(indices_0, size=int(np.rint((1 / self.undersample) * indices_1.shape[0])),
        #                                      replace=False)
        #             ind_a = list(np.append(indices_1, ind_0))
        #             ind_tr = np.random.choice(ind_a, size=int(np.rint(self.subsample * len(ind_a))), replace=False)
        #
        #         self.indices_list.append(ind_tr)
        #
        # if ((self.undersample is None) and (self.subsample is None)):
        #     lengh = len(y_clas)
        # else:
        #     lengh = len(ind_tr)

        if metric == 'basic':

            if np.array_equal(y, y.astype(bool)) == True:

                model = lgb.LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                           learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                           reg_alpha=self.reg_alpha, colsample_bytree=self.colsample_bytree,
                                           subsample=self.subsample, subsample_freq=self.subsample_freq,
                                           min_child_samples=self.min_child_samples,
                                           min_child_weight=self.min_child_weight,
                                           verbose=-1,objective=self.basic)

            else:

                model = lgb.LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                          learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                          reg_alpha=self.reg_alpha, colsample_bytree=self.colsample_bytree,
                                          subsample=self.subsample, subsample_freq=self.subsample_freq,
                                          min_child_samples=self.min_child_samples,
                                          min_child_weight=self.min_child_weight,
                                          verbose=-1,objective=self.reg)

        if metric == 'lambdarank':

            self.n = int(n_ratio * len(y))
            model = lgb.LGBMRanker(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                   learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                   reg_alpha=self.reg_alpha, colsample_bytree=self.colsample_bytree,
                                   subsample=self.subsample, subsample_freq=self.subsample_freq,
                                   lambdarank_truncation_level=self.n, lambdarank_norm=False,
                                   min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                   verbose=-1, objective='lambdarank')

        if metric == 'arp':
            model = lgb.LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                      learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                      reg_alpha=self.reg_alpha, colsample_bytree=self.colsample_bytree,
                                      subsample=self.subsample, subsample_freq=self.subsample_freq,
                                      min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                      verbose=-1, objective=self.arp)

        if metric == 'dcg':
            model = lgb.LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                      learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                      reg_alpha=self.reg_alpha, colsample_bytree=self.colsample_bytree,
                                      subsample=self.subsample, subsample_freq=self.subsample_freq,
                                      min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                      verbose=-1, objective=self.dcg)

        if metric == 'roc_auc':
            model = lgb.LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                      learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                      reg_alpha=self.reg_alpha, colsample_bytree=self.colsample_bytree,
                                      subsample=self.subsample, subsample_freq=self.subsample_freq,
                                      min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                      verbose=-1, objective=self.roc_auc)

        if metric == 'ap':
            model = lgb.LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                      learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                      reg_alpha=self.reg_alpha, colsample_bytree=self.colsample_bytree,
                                      subsample=self.subsample, subsample_freq=self.subsample_freq,
                                      min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                      verbose=-1, objective=self.ap)

        if metric == 'ep':

            self.p_ep = p_ep
            self.n_ep = max(1 / (p_ep), 1 / (1 - p_ep))

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

            self.discounter = discounter * lengh

            model = lgb.LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                      learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                      reg_alpha=self.reg_alpha, colsample_bytree=self.colsample_bytree,
                                      subsample=self.subsample, subsample_freq=self.subsample_freq,
                                      min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                      verbose=-1, objective=self.ep)

        if metric == 'precision':
            self.n_prec = int(p_prec * lengh)

            model = lgb.LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                                      learning_rate=self.learning_rate, reg_lambda=self.reg_lambda,
                                      reg_alpha=self.reg_alpha, colsample_bytree=self.colsample_bytree,
                                      subsample=self.subsample, subsample_freq=self.subsample_freq,
                                      min_child_samples=self.min_child_samples, min_child_weight=self.min_child_weight,
                                      verbose=-1, objective=self.precision)

        # Disable
        def blockPrint():
            sys.stdout = open(os.devnull, 'w')

        # Restore
        def enablePrint():
            sys.stdout = sys.__stdout__

        # blockPrint()
        model.fit(X, y)
        # enablePrint()

        endtimer = timer()

        return model, endtimer - starttimer
