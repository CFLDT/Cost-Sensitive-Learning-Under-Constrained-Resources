from timeit import default_timer as timer
import numpy as np
import random
from .logit import Lgt
from scipy.stats import beta


class Logit(Lgt):

    def __init__(self, lambd, sigma, indic_approx, theta=None):

        super().__init__(lambd, sigma, indic_approx, theta)

    def fitting(self, X, y, y_clas, init_theta, metric, n_ratio=None, p_prec=None, p_rbp=None, p_ep=None):

        random.seed(2290)
        np.random.seed(2290)

        starttimer = timer()

        self.n = int(n_ratio * len(y_clas))
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

            def obj_func(theta):
                return self.arp(theta, X, y)

            self.theta, func_min = self.optimization_rank(obj_func, init_theta)

        if metric == 'roc_auc':

            def obj_func(theta):
                return self.roc_auc(theta, X, y)

            self.theta, func_min = self.optimization_rank(obj_func, init_theta)

        if metric == 'ap':

            def obj_func(theta):
                return self.ap(theta, X, y)

            self.theta, func_min = self.optimization_rank(obj_func, init_theta)

        if metric == 'dcg':

            def obj_func(theta):
                return self.dcg(theta, X, y)

            self.theta, func_min = self.optimization_rank(obj_func, init_theta)

        if metric == 'ep':

            self.p_ep = p_ep
            self.n_ep = max(1 / (p_ep), 1 / (1 - p_ep))

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

            self.discounter = discounter * len(y)

            def obj_func(theta):
                return self.ep(theta, X, y)

            self.theta, func_min = self.optimization_rank(obj_func, init_theta)

        if metric == 'precision':
            self.n_prec = int(p_prec * len(y))

            def obj_func(theta):
                return self.precision(theta, X, y)

            self.theta, func_min = self.optimization_rank(obj_func, init_theta)

        endtimer = timer()

        return func_min, endtimer - starttimer
