import numpy as np
from scipy.stats import rankdata
from ALIGNED_learning.design.performance_metrics import PerformanceMetrics

class Lightgbm:

    def __init__(self, sigma=1, indic_approx='lambdaloss', n_prec=None, p_rbp=None, p_ep=None, n_c_ep=None):

        self.sigma = sigma
        self.indic_approx = indic_approx
        self.n_prec = n_prec
        self.p_rbp = p_rbp
        self.p_ep = p_ep
        self.n_ep = n_c_ep

    def predict_proba(self, model, X_predict):

        scores = model.predict(X_predict, raw_score=True)

        return scores

    # For LambdaLoss with relative rank difference, the scale of loss becomes
    # much smaller when applying LambdaWeight. This affects the training can
    # make the optimal learning rate become much larger. We use a heuristic to
    # scale it up to the same magnitude as standard pairwise loss. --> we solve it by always dividing by the max

    # https://github.com/tensorflow/ranking/blob/master/tensorflow_ranking/python/losses_impl.py

    def variables_init(self, y_true, y_pred):

        diff_ij = np.zeros(shape=(len(y_true), len(y_true)), dtype=np.float32)
        G_ij = np.zeros(shape=np.shape(diff_ij), dtype=np.float32)
        H_ij = np.zeros(shape=np.shape(diff_ij), dtype=np.float32)
        P_ij = np.zeros(shape=np.shape(diff_ij), dtype=np.float32)
        P_ji = np.zeros(shape=np.shape(diff_ij), dtype=np.float32)

        constant_ij = np.zeros(shape=np.shape(diff_ij), dtype=np.float32)
        delta_ij = np.zeros(shape=np.shape(diff_ij), dtype=np.float32)
        lamb_ij = np.zeros(shape=np.shape(diff_ij), dtype=np.float32)
        lamb_ij_2 = np.zeros(shape=np.shape(diff_ij), dtype=np.float32)
        lambd_der_ij = np.zeros(shape=np.shape(diff_ij), dtype=np.float32)
        lambd_der_ij_2 = np.zeros(shape=np.shape(diff_ij), dtype=np.float32)

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

        return diff_ij, G_ij, H_ij, P_ij, P_ji, constant_ij, delta_ij, lamb_ij, lamb_ij_2, lambd_der_ij, lambd_der_ij_2, ranks_v_stack_trans, \
               ranks_v_stack, y_true_v_stack_trans, y_true_v_stack, y_pred_v_stack_trans, y_pred_v_stack, bool_true

    def lambda_calc(self, bool_true, constant_ij, H_ij, diff_ij, P_ji, P_ij, lamb_ij, lamb_ij_2, lambd_der_ij,
                    lambd_der_ij_2):

        if self.indic_approx == 'logit':
            constant_ij[bool_true] = np.multiply(self.sigma, H_ij[bool_true])
            P_ji[bool_true] = 1 / (1 + np.exp(self.sigma * (diff_ij[bool_true])))
            P_ij[bool_true] = 1 / (1 + np.exp(-self.sigma * (diff_ij[bool_true])))

            lamb_ij[bool_true] = -np.multiply(constant_ij[bool_true], np.multiply(P_ji[bool_true], 1 - P_ji[bool_true]))
            lamb_ij_2[bool_true] = np.multiply(constant_ij[bool_true],
                                               np.multiply(P_ij[bool_true], 1 - P_ij[bool_true]))

            lambd_der_ij[bool_true] = np.multiply(-lamb_ij[bool_true], self.sigma * (1 - 2 * P_ji[bool_true]))
            lambd_der_ij_2[bool_true] = np.multiply(lamb_ij_2[bool_true], self.sigma * (1 - 2 * P_ij[bool_true]))

        if self.indic_approx == 'lambdaloss':

            constant_ij[bool_true] = np.multiply(self.sigma, H_ij[bool_true]) / np.log(2)
            P_ji[bool_true] = np.log2(1 + np.exp(-self.sigma * (diff_ij[bool_true])))
            P_ij[bool_true] = np.log2(1 + np.exp(self.sigma * (diff_ij[bool_true])))

            lamb_ij[bool_true] = -np.multiply(constant_ij[bool_true], 1 / (1 + np.exp(self.sigma * (diff_ij[bool_true]))))
            lamb_ij_2[bool_true] = -lamb_ij[bool_true]

            lambd_der_ij[bool_true] = -np.multiply(
                np.multiply(constant_ij[bool_true], np.multiply(1 / (1 + np.exp(self.sigma * (diff_ij[bool_true]))),
                                                                1 / (1 + np.exp(self.sigma * (diff_ij[bool_true]))))),
                np.exp(self.sigma * (diff_ij[bool_true])))

            lambd_der_ij_2[bool_true] = lambd_der_ij[bool_true]

        return P_ji, P_ij, lamb_ij, lamb_ij_2, lambd_der_ij, lambd_der_ij_2

    def grad_hess(self, lamb_ij, lamb_ij_2, lambd_der_ij, lambd_der_ij_2):

        # Gradient

        grad = np.sum(lamb_ij, axis=1) + np.sum(lamb_ij_2, axis=0)

        # Hessian

        hess = np.abs(np.sum(lambd_der_ij, axis=1) + np.sum(lambd_der_ij_2, axis=0))
        if np.all(hess == 0):
            hess = hess + 1

        return grad, hess

    def arp(self, y_true, y_pred):

        # Variables

        diff_ij, G_ij, H_ij, P_ij, P_ji, constant_ij, delta_ij, lamb_ij, lamb_ij_2, lambd_der_ij, lambd_der_ij_2, ranks_v_stack_trans, \
        ranks_v_stack, y_true_v_stack_trans, y_true_v_stack, y_pred_v_stack_trans, \
        y_pred_v_stack, bool_true = self.variables_init(y_true, y_pred)

        # Base Value

        arp_max = PerformanceMetrics.performance_metrics_arp(y_true, y_true, maximum=True)
        arp_min = PerformanceMetrics.performance_metrics_arp(y_true, y_true, minimum=True)

        # Lambda_ij calculation

        diff_ij[bool_true] = y_pred_v_stack_trans[bool_true] - y_pred_v_stack[bool_true]
        G_ij[bool_true] = ((y_true_v_stack_trans[bool_true] - y_true_v_stack[bool_true])) / (arp_max-arp_min)

        delta_ij[bool_true] = 1

        H_ij[bool_true] = np.multiply(delta_ij[bool_true], G_ij[bool_true])

        P_ji, P_ij, lamb_ij, lamb_ij_2, lambd_der_ij, lambd_der_ij_2 = self.lambda_calc(bool_true,                                                                                         constant_ij, H_ij, diff_ij,
                                                                                        P_ji, P_ij, lamb_ij, lamb_ij_2,
                                                                                        lambd_der_ij, lambd_der_ij_2)

        # Grad_Hess

        grad, hess = self.grad_hess(lamb_ij, lamb_ij_2, lambd_der_ij, lambd_der_ij_2)

        # Performance metrics

        # PerformanceMetricsTrain.performance_metrics_arp(y_pred, y_true, relative=True)
        # print('Optimisation Objective: ' + str(np.sum(np.multiply(H_ij[bool_true], P_ji[bool_true]))))

        return grad, hess

    def dcg(self, y_true, y_pred):

        # Variables

        diff_ij, G_ij, H_ij, P_ij, P_ji, constant_ij, delta_ij, lamb_ij, lamb_ij_2, lambd_der_ij, lambd_der_ij_2, ranks_v_stack_trans, \
        ranks_v_stack, y_true_v_stack_trans, y_true_v_stack, y_pred_v_stack_trans, \
        y_pred_v_stack, bool_true = self.variables_init(y_true, y_pred)

        # Base Value

        ndcg_max = PerformanceMetrics.performance_metrics_dcg(y_true, y_true, maximum=True)
        ndcg_min = PerformanceMetrics.performance_metrics_dcg(y_true, y_true, minimum=True)

        # Lambda_ij calculation

        diff_ij[bool_true] = y_pred_v_stack_trans[bool_true] - y_pred_v_stack[bool_true]
        G_ij[bool_true] = (2 ** (y_true_v_stack_trans[bool_true]) - 2 ** (y_true_v_stack[bool_true])) / (ndcg_max-ndcg_min)

        delta_ij[bool_true] = np.abs((1 / (np.log2(np.abs(ranks_v_stack_trans[((bool_true))] -
                                                          ranks_v_stack[((bool_true))]) + 1))) - \
                                     (1 / (np.log2(np.abs(ranks_v_stack_trans[((bool_true))] -
                                                          ranks_v_stack[((bool_true))]) + 1 + 1))))

        H_ij[bool_true] = np.multiply(delta_ij[bool_true], G_ij[bool_true])

        P_ji, P_ij, lamb_ij, lamb_ij_2, lambd_der_ij, lambd_der_ij_2 = self.lambda_calc(bool_true,
                                                                                        constant_ij, H_ij, diff_ij,
                                                                                        P_ji, P_ij, lamb_ij, lamb_ij_2,
                                                                                        lambd_der_ij, lambd_der_ij_2)

        # Grad_Hess

        grad, hess = self.grad_hess(lamb_ij, lamb_ij_2, lambd_der_ij, lambd_der_ij_2)

        # Performance metrics

        # PerformanceMetrics.performance_metrics_dcg(y_pred, y_true)
        # print('Optimisation Objective: ' + str(np.sum(np.multiply(H_ij[bool_true], P_ji[bool_true]))))

        return grad, hess

    def roc_auc(self, y_true, y_pred):

        # Variables

        diff_ij, G_ij, H_ij, P_ij, P_ji, constant_ij, delta_ij, lamb_ij, lamb_ij_2, lambd_der_ij, lambd_der_ij_2, ranks_v_stack_trans, \
        ranks_v_stack, y_true_v_stack_trans, y_true_v_stack, y_pred_v_stack_trans, \
        y_pred_v_stack, bool_true = self.variables_init(y_true, y_pred)

        # Base Value

        roc_auc_max = PerformanceMetrics.performance_metrics_roc_auc(y_true, y_true, maximum=True)
        roc_auc_min = PerformanceMetrics.performance_metrics_roc_auc(y_true, y_true, minimum=True)

        # Lambda_ij calculation

        diff_ij[bool_true] = y_pred_v_stack_trans[bool_true] - y_pred_v_stack[bool_true]
        G_ij[bool_true] = (y_true_v_stack_trans[bool_true] - y_true_v_stack[bool_true]) / (roc_auc_max-roc_auc_min)

        input_1 = np.zeros(shape=np.shape(G_ij), dtype=np.int32)
        input_2 = np.zeros(shape=np.shape(G_ij), dtype=np.int32)
        diff_output_1 = np.zeros(shape=np.shape(G_ij), dtype=np.float32)
        diff_output_2 = np.zeros(shape=np.shape(G_ij), dtype=np.float32)

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

        P_ji, P_ij, lamb_ij, lamb_ij_2, lambd_der_ij, lambd_der_ij_2 = self.lambda_calc(bool_true,
                                                                                        constant_ij, H_ij, diff_ij,
                                                                                        P_ji, P_ij, lamb_ij, lamb_ij_2,
                                                                                        lambd_der_ij, lambd_der_ij_2)

        # Grad_Hess

        grad, hess = self.grad_hess(lamb_ij, lamb_ij_2, lambd_der_ij, lambd_der_ij_2)

        # Performance metrics

        # PerformanceMetricsTrain.performance_metrics_ap(y_pred, y_true)
        # print('Optimisation Objective: ' + str(np.sum(np.multiply(H_ij[bool_true], P_ji[bool_true]))))

        return grad, hess

    def ap(self, y_true, y_pred):

        # Variables

        diff_ij, G_ij, H_ij, P_ij, P_ji, constant_ij, delta_ij, lamb_ij, lamb_ij_2, lambd_der_ij, lambd_der_ij_2, ranks_v_stack_trans, \
        ranks_v_stack, y_true_v_stack_trans, y_true_v_stack, y_pred_v_stack_trans, \
        y_pred_v_stack, bool_true = self.variables_init(y_true, y_pred)

        # Base Value

        ap_max = PerformanceMetrics.performance_metrics_ap(y_true, y_true, maximum=True)
        ap_min = PerformanceMetrics.performance_metrics_ap(y_true, y_true, minimum=True)

        # Lambda_ij calculation

        diff_ij[bool_true] = y_pred_v_stack_trans[bool_true] - y_pred_v_stack[bool_true]
        G_ij[bool_true] = (y_true_v_stack_trans[bool_true] - y_true_v_stack[bool_true]) / (ap_max-ap_min)

        input_1 = np.zeros(shape=np.shape(G_ij), dtype=np.int32)
        input_2 = np.zeros(shape=np.shape(G_ij), dtype=np.int32)
        diff_output_1 = np.zeros(shape=np.shape(G_ij), dtype=np.float32)
        diff_output_2 = np.zeros(shape=np.shape(G_ij), dtype=np.float32)

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

        P_ji, P_ij, lamb_ij, lamb_ij_2, lambd_der_ij, lambd_der_ij_2 = self.lambda_calc(bool_true,
                                                                                        constant_ij, H_ij, diff_ij,
                                                                                        P_ji, P_ij, lamb_ij, lamb_ij_2,
                                                                                        lambd_der_ij, lambd_der_ij_2)

        # Grad_Hess

        grad, hess = self.grad_hess(lamb_ij, lamb_ij_2, lambd_der_ij, lambd_der_ij_2)

        # Performance metrics

        # PerformanceMetricsTrain.performance_metrics_ap(y_pred, y_true)
        #print('Optimisation Objective: ' + str(np.sum(np.multiply(H_ij[bool_true], P_ji[bool_true]))))

        return grad, hess

    def ep(self, y_true, y_pred):

        # Variables

        diff_ij, G_ij, H_ij, P_ij, P_ji, constant_ij, delta_ij, lamb_ij, lamb_ij_2, lambd_der_ij, lambd_der_ij_2, ranks_v_stack_trans, \
        ranks_v_stack, y_true_v_stack_trans, y_true_v_stack, y_pred_v_stack_trans, \
        y_pred_v_stack, bool_true = self.variables_init(y_true, y_pred)

        # Base Value

        ep_max = PerformanceMetrics.performance_metrics_ep(y_true, y_true, self.p_ep, self.n_ep, maximum=True)
        ep_min = PerformanceMetrics.performance_metrics_ep(y_true, y_true, self.p_ep, self.n_ep, minimum=True)

        # Lambda_ij calculation

        diff_ij[bool_true] = y_pred_v_stack_trans[bool_true] - y_pred_v_stack[bool_true]

        #G_ij[bool_true] = (y_true_v_stack_trans[bool_true] - y_true_v_stack[bool_true]) / ep_max
        G_ij[bool_true] = (y_true_v_stack_trans[bool_true] - y_true_v_stack[bool_true]) / (ep_max-ep_min)

        input_1 = np.zeros(shape=np.shape(G_ij), dtype=np.int32)
        input_2 = np.zeros(shape=np.shape(G_ij), dtype=np.int32)
        diff_output_1 = np.zeros(shape=np.shape(G_ij), dtype=np.float32)
        diff_output_2 = np.zeros(shape=np.shape(G_ij), dtype=np.float32)

        input_1[(bool_true)] = np.abs(ranks_v_stack_trans[(bool_true)] - ranks_v_stack[(bool_true)])
        input_2[(bool_true)] = np.abs(ranks_v_stack_trans[(bool_true)] - ranks_v_stack[(bool_true)]) + 1

        diff_output_1[(bool_true)] = self.discounter[input_1[(bool_true)]-1]
        diff_output_2[(bool_true)] = self.discounter[input_2[(bool_true)]-1]

        delta_ij[bool_true] = np.abs(diff_output_1[bool_true] - diff_output_2[bool_true])

        H_ij[bool_true] = np.multiply(delta_ij[bool_true], G_ij[bool_true])

        P_ji, P_ij, lamb_ij, lamb_ij_2, lambd_der_ij, lambd_der_ij_2 = self.lambda_calc(bool_true,
                                                                                        constant_ij, H_ij, diff_ij,
                                                                                        P_ji, P_ij,
                                                                                        lamb_ij, lamb_ij_2,
                                                                                        lambd_der_ij,
                                                                                        lambd_der_ij_2)

        # Grad_Hess

        grad, hess = self.grad_hess(lamb_ij, lamb_ij_2, lambd_der_ij, lambd_der_ij_2)

        # Performance metrics

        # PerformanceMetricsTrain.performance_metrics_ep(y_pred, y_true)
        #print('Optimisation Objective: ' + str(np.sum(np.multiply(H_ij[bool_true], P_ji[bool_true]))))

        return grad, hess


    def precision(self, y_true, y_pred):

        # Variables

        diff_ij, G_ij, H_ij, P_ij, P_ji, constant_ij, delta_ij, lamb_ij, lamb_ij_2, lambd_der_ij, lambd_der_ij_2, ranks_v_stack_trans, \
        ranks_v_stack, y_true_v_stack_trans, y_true_v_stack, y_pred_v_stack_trans, \
        y_pred_v_stack, bool_true = self.variables_init(y_true, y_pred)

        # Base Value

        precision_max = PerformanceMetrics.performance_metrics_precision(y_true, y_true, self.n_prec, maximum=True)
        precision_min = PerformanceMetrics.performance_metrics_precision(y_true, y_true, self.n_prec, minimum=True)

        # Lambda_ij calculation

        diff_ij[bool_true] = y_pred_v_stack_trans[bool_true] - y_pred_v_stack[bool_true]

        G_ij[bool_true] = (y_true_v_stack_trans[bool_true] - y_true_v_stack[bool_true]) / (precision_max-precision_min)

        input_1 = np.zeros(shape=np.shape(G_ij), dtype=np.int32)
        input_2 = np.zeros(shape=np.shape(G_ij), dtype=np.int32)

        diff_output_1 = np.zeros(shape=np.shape(G_ij), dtype=np.float32)
        diff_output_2 = np.zeros(shape=np.shape(G_ij), dtype=np.float32)

        input_1[bool_true] = np.abs(ranks_v_stack_trans[bool_true] - ranks_v_stack[bool_true])
        diff_output_1[bool_true] = np.where(input_1[bool_true] <= self.n_prec, 1 / self.n_prec, 0)

        input_2[bool_true] = np.abs(ranks_v_stack_trans[bool_true] - ranks_v_stack[bool_true]) + 1
        diff_output_2[bool_true] = np.where(input_2[bool_true] <= self.n_prec, 1 / self.n_prec, 0)

        delta_ij[bool_true] = np.abs(diff_output_1[bool_true] - diff_output_2[bool_true])

        H_ij[bool_true] = np.multiply(delta_ij[bool_true], G_ij[bool_true])

        P_ji, P_ij, lamb_ij, lamb_ij_2, lambd_der_ij, lambd_der_ij_2 = self.lambda_calc(bool_true,
                                                                                        constant_ij, H_ij, diff_ij,
                                                                                        P_ji, P_ij,
                                                                                        lamb_ij, lamb_ij_2,
                                                                                        lambd_der_ij,
                                                                                        lambd_der_ij_2)

        # Grad_Hess

        grad, hess = self.grad_hess(lamb_ij, lamb_ij_2, lambd_der_ij, lambd_der_ij_2)

        # Performance metrics

        # PerformanceMetricsTrain.performance_metrics_precision(y_pred, y_true)
        # print('Optimisation Objective: ' + str(np.sum(np.multiply(H_ij[bool_true], P_ji[bool_true]))))

        return grad, hess


    def basic(self, y_true, y_pred):

        # if ((self.undersample is not None) or (self.subsample is not None)):
        #     grad_1 = np.zeros(shape=np.shape(y_true), dtype=np.float32)
        #     hess_1 = np.zeros(shape=np.shape(y_true), dtype=np.float32)
        #
        #     y_true = y_true.flat[self.indices_list[self.it_index]]
        #     y_pred = y_pred.flat[self.indices_list[self.it_index]]

        scores_1 = -1 / (1 + np.exp(y_pred))
        scores_0 = 1 / (1 + np.exp(-y_pred))

        sec_der_1 = np.multiply(-scores_1, 1 + scores_1)
        sec_der_0 = np.multiply(scores_0, 1 - scores_0)

        grad = np.multiply(y_true, scores_1) \
               + np.multiply(1-y_true, scores_0)
        hess = np.abs(np.multiply(y_true, sec_der_1)
                      + np.multiply(1-y_true, sec_der_0))

        # if ((self.undersample is not None) or (self.subsample is not None)):
        #     grad_1.flat[self.indices_list[self.it_index]] = grad
        #     hess_1.flat[self.indices_list[self.it_index]] = hess
        #
        #     grad = grad_1
        #     hess = hess_1
        #     self.it_index = self.it_index + 1

        return grad, hess


    def reg(self, y_true, y_pred):

        grad = 2*(y_pred - y_true)
        hess = 2*np.ones(len(y_true))

        return grad, hess