import numpy as np
import scipy.optimize
from scipy.stats import rankdata
from ALIGNED_learning.design.performance_metrics import PerformanceMetrics
from scipy.stats import betabinom


class Lgt:

    def __init__(self, lambd, sigma, indic_approx, theta=None, n_prec=None, p_rbp=None, p_ep=None, n_c_ep=None):

        self.lambd = lambd
        self.theta = theta
        self.sigma = sigma
        self.indic_approx = indic_approx
        self.n_prec = n_prec
        self.p_rbp = p_rbp
        self.p_ep = p_ep
        self.n_c_ep = n_c_ep

    def predict(self, X_predict):
        if X_predict.shape[1] == len(self.theta):
            scores = X_predict.dot(self.theta)
        else:
            scores = self.theta[0] + X_predict.dot(self.theta[1:])
        # predictions = (scores > treshhold).astype(int)

        return scores

    def predict_proba(self, X_predict):
        if X_predict.shape[1] == len(self.theta):
            scores = X_predict.dot(self.theta)
        else:
            scores = self.theta[0] + X_predict.dot(self.theta[1:])
        scores = 1 / (1 + np.exp(-scores))

    def optimization_basic(self, obj_func, initial_theta):
        opt_res = scipy.optimize.minimize(obj_func, initial_theta, method="L-BFGS-B",  # L-BFGS-B uses approx of hessian
                                          options={'disp': False, 'maxiter': 100000})
        theta_opt, func_min = opt_res.x, opt_res.fun
        return theta_opt, func_min

    def objective_function_basic(self, theta, X, div, weight_1_0, weight_0_0):
        scores = theta[0] + X.dot(theta[1:])

        loss_1_0 = np.log(1 + np.exp(-scores))
        loss_0_0 = np.log(1 + np.exp(scores))

        objective = -div * (weight_1_0.dot(loss_1_0) + weight_0_0.dot(loss_0_0)) + self.lambd * np.sum(theta[1:] ** 2)

        return objective

    def optimization_rank(self, obj_func, initial_theta):
        # atm we do not use hessian information
        opt_res = scipy.optimize.minimize(obj_func, initial_theta, method="L-BFGS-B",
                                          jac=True, options={'disp': False, 'maxiter': 100000})
        theta_opt, func_min = opt_res.x, opt_res.fun
        return theta_opt, func_min

    def variables_init(self, y_true, y_pred):

        diff_ij = np.zeros(shape=(len(y_true), len(y_true)))
        G_ij = np.zeros(shape=np.shape(diff_ij))
        H_ij = np.zeros(shape=np.shape(diff_ij))
        P_ij = np.zeros(shape=np.shape(diff_ij))
        P_ji = np.zeros(shape=np.shape(diff_ij))

        constant_ij = np.zeros(shape=np.shape(diff_ij))
        delta_ij = np.zeros(shape=np.shape(diff_ij))
        lamb_ij = np.zeros(shape=np.shape(diff_ij))
        lamb_ij_2 = np.zeros(shape=np.shape(diff_ij))
        # lambd_der_ij = np.zeros(shape=np.shape(diff_ij))
        # lambd_der_ij_2 = np.zeros(shape=np.shape(diff_ij))
        sigma = self.sigma

        if np.all(y_pred == 0):
            ranks = np.arange(len(y_pred)) + 1
            np.random.shuffle(ranks)
        else:
            ranks = rankdata(-y_pred, method='ordinal')  # ordinal. If average dcg issues due to log(0)

        ranks_v_stack = np.vstack([ranks] * np.shape(ranks)[0])
        ranks_v_stack_trans = ranks_v_stack.T

        y_true_v_stack = np.vstack([y_true] * np.shape(y_true)[0])
        y_true_v_stack_trans = y_true_v_stack.T

        y_pred_v_stack = np.vstack([y_pred] * np.shape(y_pred)[0])
        y_pred_v_stack_trans = y_pred_v_stack.T

        bool_true = y_true_v_stack_trans > y_true_v_stack

        return diff_ij, G_ij, H_ij, P_ij, P_ji, constant_ij, delta_ij, lamb_ij, lamb_ij_2, sigma, ranks_v_stack_trans, \
               ranks_v_stack, y_true_v_stack_trans, y_true_v_stack, y_pred_v_stack_trans, y_pred_v_stack, bool_true

    def lambda_calc(self, bool_true, sigma, constant_ij, H_ij, diff_ij, P_ji, P_ij, lamb_ij, lamb_ij_2):

        if self.indic_approx == 'logit':
            constant_ij[bool_true] = np.multiply(sigma, H_ij[bool_true])
            P_ji[bool_true] = 1 / (1 + np.exp(sigma * (diff_ij[bool_true])))
            P_ij[bool_true] = 1 / (1 + np.exp(-sigma * (diff_ij[bool_true])))

            lamb_ij[bool_true] = -np.multiply(constant_ij[bool_true], np.multiply(P_ji[bool_true], 1 - P_ji[bool_true]))
            lamb_ij_2[bool_true] = np.multiply(constant_ij[bool_true],
                                               np.multiply(P_ij[bool_true], 1 - P_ij[bool_true]))


        if self.indic_approx == 'lambdaloss':
            constant_ij[bool_true] = np.multiply(sigma, H_ij[bool_true]) / np.log(2)
            P_ij[bool_true] = np.log2(1 + np.exp(sigma * (diff_ij[bool_true])))
            P_ji[bool_true] = np.log2(1 + np.exp(-sigma * (diff_ij[bool_true])))

            lamb_ij[bool_true] = -np.multiply(constant_ij[bool_true], 1 / (1 + np.exp(sigma * (diff_ij[bool_true]))))
            lamb_ij_2[bool_true] = -lamb_ij[bool_true]

        return P_ji, P_ij, lamb_ij, lamb_ij_2

    def grad(self, X, theta, lamb_ij, lamb_ij_2):

        # Gradient
        grad_i = np.sum(lamb_ij, axis=1) + np.sum(lamb_ij_2, axis=0)
        grad = grad_i.dot(X) + 2 * self.lambd * theta

        # Hessian

        # hess_i = np.abs(np.sum(lambd_der_ij, axis=1) + np.sum(lambd_der_ij_2, axis=0))
        # hess = hess_i.dot(X)

        return grad  # hess

    def arp(self, theta, X, y_true):

        y_pred = X.dot(theta)

        # Variables

        diff_ij, G_ij, H_ij, P_ij, P_ji, constant_ij, delta_ij, lamb_ij, lamb_ij_2, sigma, ranks_v_stack_trans, \
        ranks_v_stack, y_true_v_stack_trans, y_true_v_stack, y_pred_v_stack_trans, \
        y_pred_v_stack, bool_true = self.variables_init(y_true, y_pred)

        # Base Value

        #arp_max = PerformanceMetrics.performance_metrics_arp(y_true, y_true, maximum=True)

        # Lambda_ij calculation

        diff_ij[bool_true] = y_pred_v_stack_trans[bool_true] - y_pred_v_stack[bool_true]
        G_ij[bool_true] = ((y_true_v_stack_trans[bool_true] - y_true_v_stack[bool_true])) / (self.arp_max)

        delta_ij[bool_true] = 1

        H_ij[bool_true] = np.multiply(delta_ij[bool_true], G_ij[bool_true])

        P_ji, P_ij, lamb_ij, lamb_ij_2 = self.lambda_calc(bool_true, sigma, constant_ij, H_ij, diff_ij, P_ji, P_ij,
                                                                                        lamb_ij, lamb_ij_2)
        # Grad

        grad = self.grad(X, theta, lamb_ij, lamb_ij_2)

        # Performance metrics

        obj = np.sum(np.multiply(H_ij[bool_true], P_ji[bool_true])) + self.lambd * np.sum(theta ** 2)
        # obj = -PerformanceMetricsTrain.performance_metrics_arp(y_pred, y_true)

        #print('Optimisation Objective: ' + str(obj))

        return obj, grad

    def roc_auc(self, theta, X, y_true):

        y_pred = X.dot(theta)

        # Variables

        diff_ij, G_ij, H_ij, P_ij, P_ji, constant_ij, delta_ij, lamb_ij, lamb_ij_2, sigma, ranks_v_stack_trans, \
        ranks_v_stack, y_true_v_stack_trans, y_true_v_stack, y_pred_v_stack_trans, \
        y_pred_v_stack, bool_true = self.variables_init(y_true, y_pred)

        # Base Value

        # roc_auc_max = PerformanceMetrics.performance_metrics_roc_auc(y_true, y_true, maximum=True)

        # Lambda_ij calculation

        diff_ij[bool_true] = y_pred_v_stack_trans[bool_true] - y_pred_v_stack[bool_true]
        G_ij[bool_true] = (y_true_v_stack_trans[bool_true] - y_true_v_stack[bool_true]) / (self.roc_auc_max)

        input_1 = np.zeros(shape=np.shape(G_ij), dtype=np.int8)
        input_2 = np.zeros(shape=np.shape(G_ij), dtype=np.int8)
        diff_output_1 = np.zeros(shape=np.shape(G_ij))
        diff_output_2 = np.zeros(shape=np.shape(G_ij))

        input_1[(bool_true)] = np.abs(ranks_v_stack_trans[(bool_true)] - ranks_v_stack[(bool_true)])
        input_2[(bool_true)] = np.abs(ranks_v_stack_trans[(bool_true)] - ranks_v_stack[(bool_true)]) + 1

        ind = ranks_v_stack[0, :]
        y_true_ranked = np.array(y_true)
        sorted_ar = [x for _, x in sorted(zip(ind, y_true_ranked))]
        value = np.cumsum(np.array(sorted_ar))

        diff_output_1[(bool_true)] = value[input_1[(bool_true)]-1]
        diff_output_2[(bool_true)] = value[input_2[(bool_true)]-1]


        # diff_output_1[(bool_true)] = np.array([value[int(i - 1)] for i in input_1[(bool_true)]])
        # diff_output_2[(bool_true)] = np.array([value[int(i - 1)] for i in input_2[(bool_true)]])

        delta_ij[bool_true] = np.abs(diff_output_1[bool_true] - diff_output_2[bool_true])

        H_ij[bool_true] = np.multiply(delta_ij[bool_true], G_ij[bool_true])

        P_ji, P_ij, lamb_ij, lamb_ij_2 = self.lambda_calc(bool_true, sigma, constant_ij, H_ij, diff_ij, P_ji, P_ij,
                                                          lamb_ij, lamb_ij_2)

        # Grad

        grad = self.grad(X, theta, lamb_ij, lamb_ij_2)

        # Performance metrics

        obj = np.sum(np.multiply(H_ij[bool_true], P_ji[bool_true])) + self.lambd * np.sum(theta ** 2)
        # obj = -PerformanceMetricsTrain.performance_metrics_ap(y_pred, y_true)
        if np.all(y_pred == 0):
            obj = obj + 100

        #print('Optimisation Objective: ' + str(obj))

        return obj, grad

    def ap(self, theta, X, y_true):

        y_pred = X.dot(theta)

        # Variables

        diff_ij, G_ij, H_ij, P_ij, P_ji, constant_ij, delta_ij, lamb_ij, lamb_ij_2, sigma, ranks_v_stack_trans, \
        ranks_v_stack, y_true_v_stack_trans, y_true_v_stack, y_pred_v_stack_trans, \
        y_pred_v_stack, bool_true = self.variables_init(y_true, y_pred)

        # Base Value

        # ap_max = PerformanceMetrics.performance_metrics_ap(y_true, y_true, maximum=True)

        # Lambda_ij calculation

        diff_ij[bool_true] = y_pred_v_stack_trans[bool_true] - y_pred_v_stack[bool_true]
        G_ij[bool_true] = (y_true_v_stack_trans[bool_true] - y_true_v_stack[bool_true]) / (self.ap_max)

        input_1 = np.zeros(shape=np.shape(G_ij), dtype=np.int8)
        input_2 = np.zeros(shape=np.shape(G_ij), dtype=np.int8)
        diff_output_1 = np.zeros(shape=np.shape(G_ij))
        diff_output_2 = np.zeros(shape=np.shape(G_ij))

        input_1[(bool_true)] = np.abs(ranks_v_stack_trans[(bool_true)] - ranks_v_stack[(bool_true)])
        input_2[(bool_true)] = np.abs(ranks_v_stack_trans[(bool_true)] - ranks_v_stack[(bool_true)]) + 1

        ind = ranks_v_stack[0, :]
        y_true_ranked = np.array(y_true)
        sorted_ar = [x for _, x in sorted(zip(ind, y_true_ranked))]
        denom = np.arange(len(y_pred)) + 1
        value = np.cumsum(sorted_ar) / denom

        diff_output_1[(bool_true)] = value[input_1[(bool_true)]-1]
        diff_output_2[(bool_true)] = value[input_2[(bool_true)]-1]


        # diff_output_1[(bool_true)] = np.array([value[int(i-1)] for i in input_1[(bool_true)]])
        # diff_output_2[(bool_true)] = np.array([value[int(i-1)] for i in input_2[(bool_true)]])

        delta_ij[bool_true] = np.abs(diff_output_1[bool_true] - diff_output_2[bool_true])



        H_ij[bool_true] = np.multiply(delta_ij[bool_true], G_ij[bool_true])

        P_ji, P_ij, lamb_ij, lamb_ij_2 = self.lambda_calc(bool_true, sigma, constant_ij, H_ij, diff_ij, P_ji, P_ij,
                                                                                        lamb_ij, lamb_ij_2)

        # Grad

        grad = self.grad(X, theta, lamb_ij, lamb_ij_2)

        # Performance metrics

        obj = np.sum(np.multiply(H_ij[bool_true], P_ji[bool_true])) + self.lambd * np.sum(theta ** 2)
        # obj = -PerformanceMetricsTrain.performance_metrics_ap(y_pred, y_true)
        if np.all(y_pred == 0):
            obj = obj + 100

        #print('Optimisation Objective: ' + str(obj))

        return obj, grad

    def dcg(self, theta, X, y_true):

        # Variables

        y_pred = X.dot(theta)

        # Variables

        diff_ij, G_ij, H_ij, P_ij, P_ji, constant_ij, delta_ij, lamb_ij, lamb_ij_2, sigma, ranks_v_stack_trans, \
        ranks_v_stack, y_true_v_stack_trans, y_true_v_stack, y_pred_v_stack_trans, \
        y_pred_v_stack, bool_true = self.variables_init(y_true, y_pred)

        # Base Value

        # ndcg_max = PerformanceMetrics.performance_metrics_dcg(y_true, y_true, maximum=True)

        # Lambda_ij calculation

        diff_ij[bool_true] = y_pred_v_stack_trans[bool_true] - y_pred_v_stack[bool_true]
        G_ij[bool_true] = (2 ** (y_true_v_stack_trans[bool_true]) - 2 ** (y_true_v_stack[bool_true])) / (self.ndcg_max)

        delta_ij[bool_true] = np.abs((1 / (np.log2(np.abs(ranks_v_stack_trans[((bool_true))] -
                                                          ranks_v_stack[((bool_true))]) + 1))) - \
                                     (1 / (np.log2(np.abs(ranks_v_stack_trans[((bool_true))] -
                                                          ranks_v_stack[((bool_true))]) + 1 + 1))))

        H_ij[bool_true] = np.multiply(delta_ij[bool_true], G_ij[bool_true])

        P_ji, P_ij, lamb_ij, lamb_ij_2 = self.lambda_calc(bool_true, sigma, constant_ij, H_ij, diff_ij, P_ji, P_ij,
                                                                                        lamb_ij, lamb_ij_2)

        # Grad

        grad = self.grad(X, theta, lamb_ij, lamb_ij_2)

        # Performance metrics

        obj = np.sum(np.multiply(H_ij[bool_true], P_ji[bool_true])) + self.lambd * np.sum(theta ** 2)
        # obj = -PerformanceMetricsTrain.performance_metrics_dcg(y_pred, y_true)

        #print('Optimisation Objective: ' + str(obj))

        return obj, grad

    def ep(self, theta, X, y_true):

        # Variables

        y_pred = X.dot(theta)

        # Variables

        diff_ij, G_ij, H_ij, P_ij, P_ji, constant_ij, delta_ij, lamb_ij, lamb_ij_2, sigma, ranks_v_stack_trans, \
        ranks_v_stack, y_true_v_stack_trans, y_true_v_stack, y_pred_v_stack_trans, \
        y_pred_v_stack, bool_true = self.variables_init(y_true, y_pred)

        # Base Value

        # ep_max = PerformanceMetrics.performance_metrics_ep(y_true, y_true, self.p_ep, self.n_c_ep, maximum=True)

        # Lambda_ij calculation

        diff_ij[bool_true] = y_pred_v_stack_trans[bool_true] - y_pred_v_stack[bool_true]

        G_ij[bool_true] = (y_true_v_stack_trans[bool_true] - y_true_v_stack[bool_true]) / (self.ep_max)


        input_1 = np.zeros(shape=np.shape(G_ij), dtype=np.int8)
        input_2 = np.zeros(shape=np.shape(G_ij), dtype=np.int8)
        beta_bin_output_1 = np.zeros(shape=np.shape(G_ij))
        beta_bin_output_2 = np.zeros(shape=np.shape(G_ij))

        input_1[(bool_true)] = np.abs(ranks_v_stack_trans[(bool_true)] - ranks_v_stack[(bool_true)])
        input_2[(bool_true)] = np.abs(ranks_v_stack_trans[(bool_true)] - ranks_v_stack[(bool_true)]) + 1


        beta_bin_output_1[(bool_true)] = self.discounter[input_1[(bool_true)]-1]
        beta_bin_output_2[(bool_true)] = self.discounter[input_2[(bool_true)]-1]

        delta_ij[bool_true] = np.abs(beta_bin_output_1[bool_true] - beta_bin_output_2[bool_true])

        H_ij[bool_true] = np.multiply(delta_ij[bool_true], G_ij[bool_true])

        P_ji, P_ij, lamb_ij, lamb_ij_2 = self.lambda_calc(bool_true, sigma, constant_ij, H_ij, diff_ij, P_ji, P_ij,
                                                                                        lamb_ij, lamb_ij_2)

        # Grad

        grad = self.grad(X, theta, lamb_ij, lamb_ij_2)

        # Performance metrics

        obj = np.sum(np.multiply(H_ij[bool_true], P_ji[bool_true])) + self.lambd * np.sum(theta ** 2)
        # obj = -PerformanceMetricsTrain.performance_metrics_ep(y_pred, y_true)

        #print('Optimisation Objective: ' + str(obj))

        return obj, grad

    def rbp(self, theta, X, y_true):

        # Variables

        y_pred = X.dot(theta)

        # Variables

        diff_ij, G_ij, H_ij, P_ij, P_ji, constant_ij, delta_ij, lamb_ij, lamb_ij_2, sigma, ranks_v_stack_trans, \
        ranks_v_stack, y_true_v_stack_trans, y_true_v_stack, y_pred_v_stack_trans, \
        y_pred_v_stack, bool_true = self.variables_init(y_true, y_pred)

        # Base Value

        # rbp_max = PerformanceMetrics.performance_metrics_rbp(y_true, y_true, self.p_rbp, maximum=True)

        # Lambda_ij calculation

        diff_ij[bool_true] = y_pred_v_stack_trans[bool_true] - y_pred_v_stack[bool_true]

        G_ij[bool_true] = (y_true_v_stack_trans[bool_true] - y_true_v_stack[bool_true]) / (self.rbp_max)

        delta_ij[bool_true] = np.abs(self.p_rbp ** (np.abs(ranks_v_stack_trans[bool_true] -
                                                           ranks_v_stack[bool_true]) - 1) -
                                     self.p_rbp ** (np.abs(ranks_v_stack_trans[bool_true] -
                                                           ranks_v_stack[bool_true])))

        H_ij[bool_true] = np.multiply(delta_ij[bool_true], G_ij[bool_true])

        P_ji, P_ij, lamb_ij, lamb_ij_2 = self.lambda_calc(bool_true, sigma, constant_ij, H_ij, diff_ij, P_ji, P_ij,
                                                                                        lamb_ij, lamb_ij_2)

        # Grad

        grad = self.grad(X, theta, lamb_ij, lamb_ij_2)

        # Performance metrics

        obj = np.sum(np.multiply(H_ij[bool_true], P_ji[bool_true])) + self.lambd * np.sum(theta ** 2)
        # obj = -PerformanceMetricsTrain.performance_metrics_rbp(y_pred, y_true)

        #print('Optimisation Objective: ' + str(obj))

        return obj, grad

    def precision(self, theta, X, y_true):

        # Variables

        y_pred = X.dot(theta)

        # Variables

        diff_ij, G_ij, H_ij, P_ij, P_ji, constant_ij, delta_ij, lamb_ij, lamb_ij_2, sigma, ranks_v_stack_trans, \
        ranks_v_stack, y_true_v_stack_trans, y_true_v_stack, y_pred_v_stack_trans, \
        y_pred_v_stack, bool_true = self.variables_init(y_true, y_pred)

        # Base Value

        # precision_max = PerformanceMetrics.performance_metrics_precision(y_true, y_true, self.n_prec, maximum=True)

        # Lambda_ij calculation

        diff_ij[bool_true] = y_pred_v_stack_trans[bool_true] - y_pred_v_stack[bool_true]

        G_ij[bool_true] = (y_true_v_stack_trans[bool_true] - y_true_v_stack[bool_true]) / (self.precision_max)

        input_1 = np.zeros(shape=np.shape(G_ij))
        input_2 = np.zeros(shape=np.shape(G_ij))

        diff_output_1 = np.zeros(shape=np.shape(G_ij))
        diff_output_2 = np.zeros(shape=np.shape(G_ij))

        input_1[bool_true] = np.abs(ranks_v_stack_trans[bool_true] - ranks_v_stack[bool_true])
        diff_output_1[bool_true] = np.where(input_1[bool_true] <= self.n_prec, 1 / self.n_prec, 0)

        input_2[bool_true] = np.abs(ranks_v_stack_trans[bool_true] - ranks_v_stack[bool_true]) + 1
        diff_output_2[bool_true] = np.where(input_2[bool_true] <= self.n_prec, 1 / self.n_prec, 0)

        delta_ij[bool_true] = np.abs(diff_output_1[bool_true] - diff_output_2[bool_true])

        H_ij[bool_true] = np.multiply(delta_ij[bool_true], G_ij[bool_true])

        P_ji, P_ij, lamb_ij, lamb_ij_2 = self.lambda_calc(bool_true, sigma, constant_ij, H_ij, diff_ij, P_ji, P_ij,
                                                                                        lamb_ij, lamb_ij_2)

        # Grad

        grad = self.grad(X, theta, lamb_ij, lamb_ij_2)

        # Performance metrics

        obj = np.sum(np.multiply(H_ij[bool_true], P_ji[bool_true])) + self.lambd * np.sum(theta ** 2)
        # obj = -PerformanceMetricsTrain.performance_metrics_precision(y_pred, y_true)

        #print('Optimisation Objective: ' + str(obj))

        return obj, grad
