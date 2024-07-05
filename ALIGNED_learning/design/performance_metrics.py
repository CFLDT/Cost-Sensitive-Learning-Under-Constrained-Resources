import numpy as np
from scipy.stats import binom, beta, betabinom


class PerformanceMetrics:

    @staticmethod
    def performance_metrics_arp(y_pred, y_true, maximum=False, n=None):

        if n == None:
            n = len(y_true)

        def arp_calculator(true_values):
            discounter = np.zeros(len(true_values))
            for i in range(len(true_values)):
                discounter[i] = len(true_values) - i + 1  # index starts at zero, thus +1

            gains = true_values
            arp = np.sum(np.multiply(gains, discounter))

            return arp

        if maximum == False:
            if np.all(y_pred == 0):
                value = 0
            else:
                outp = y_pred.argsort()[::-1][:n]
                true_values = y_true[outp]
                value = arp_calculator(np.array(true_values))

        else:

            outp_max = y_true.argsort()[::-1][:n]
            true_values_max = y_true[outp_max]
            value = arp_calculator(np.array(true_values_max))

        return value

    @staticmethod
    def performance_metrics_dcg(y_pred, y_true, maximum=False, n=None):

        if n == None:
            n = len(y_true)

        def dcg_calculator(true_values):
            discounter = np.zeros(len(true_values))
            for i in range(len(true_values)):
                discounter[i] = 1 / (np.log2(i + 2))  # index starts at zero, thus +2

            gains = (2 ** true_values) - 1
            dcg = np.sum(np.multiply(gains, discounter))

            return dcg

        if maximum == False:

            if np.all(y_pred == 0):
                value = 0
            else:
                outp = y_pred.argsort()[::-1][:n]
                true_values = y_true[outp]
                value = dcg_calculator(np.array(true_values))

        else:

            outp_max = y_true.argsort()[::-1][:n]
            true_values_max = y_true[outp_max]
            value = dcg_calculator(np.array(true_values_max))

        return value

    @staticmethod
    def performance_metrics_ap(y_pred, y_true, maximum=False, n=None):

        if n == None:
            n = len(y_true)

        def ap_calculator(true_values):

            discounter = np.zeros(len(true_values))
            gains = np.zeros(len(true_values))

            for i in range(len(true_values)):
                discounter[i] = 1 / (i + 1)
                gains[i] = true_values[i] * np.sum(true_values[0:i + 1])

            ap = np.sum(np.multiply(discounter, gains))

            return ap

        if maximum == False:
            if np.all(y_pred == 0):
                value = 0
            else:
                outp = y_pred.argsort()[::-1][:n]
                true_values = y_true[outp]
                value = ap_calculator(np.array(true_values))

        else:

            outp_max = y_true.argsort()[::-1][:n]
            true_values_max = y_true[outp_max]
            value = ap_calculator(np.array(true_values_max))

        return value

    @staticmethod
    def performance_metrics_ep(y_pred, y_true, p_ep, n_ep, maximum=False, n=None):

        if n == None:
            n = len(y_true)

        def ep_calculator(true_values):

            discounter = np.zeros(len(true_values))

            disc = 0
            for i in range(len(true_values), 0, -1):

                if i < len(true_values):
                    top = beta.cdf((i + 0.5) / len(true_values), p_ep * n_ep, n_ep * (1 - p_ep))
                else:
                    top = 1

                bot = beta.cdf((i - 0.5) / len(true_values), p_ep * n_ep, n_ep * (1 - p_ep))

                prob = top - bot

                disc = disc + (prob / (i))
                discounter[i - 1] = disc

            gains = true_values
            ep = np.sum(np.multiply(gains, discounter*len(true_values)))

            return ep

        if maximum == False:
            if np.all(y_pred == 0):
                value = 0
            else:
                outp = y_pred.argsort()[::-1][:n]
                true_values = y_true[outp]
                value = ep_calculator(np.array(true_values))

        else:

            outp_max = y_true.argsort()[::-1][:n]
            true_values_max = y_true[outp_max]
            value = ep_calculator(np.array(true_values_max))

        return value

    @staticmethod
    def performance_metrics_rbp(y_pred, y_true, p_rbp, maximum=False, n=None):

        if n == None:
            n = len(y_true)

        def rbp_calculator(true_values):

            discounter = np.zeros(len(true_values))

            for i in range(len(true_values)):
                discounter[i] = p_rbp ** (i)

            gains = true_values
            rbp = np.sum(np.multiply(discounter, gains))  # can do times (1 - p_rbp)  to normalize

            return rbp

        if maximum == False:
            if np.all(y_pred == 0):
                value = 0
            else:
                outp = y_pred.argsort()[::-1][:n]
                true_values = y_true[outp]
                value = rbp_calculator(np.array(true_values))

        else:

            outp_max = y_true.argsort()[::-1][:n]
            true_values_max = y_true[outp_max]
            value = rbp_calculator(np.array(true_values_max))

        return value

    @staticmethod
    def performance_metrics_precision(y_pred, y_true, n_prec, maximum=False, n=None):

        if n == None:
            n = len(y_true)

        def precision_calculator(true_values):

            discounter = np.zeros(len(true_values))

            for i in range(len(true_values)):
                if i < n_prec:
                    discounter[i] = (1 / n_prec)
            gains = true_values

            precision = np.sum(np.multiply(discounter, gains))

            return precision

        if maximum == False:
            if np.all(y_pred == 0):
                value = 0
            else:
                outp = y_pred.argsort()[::-1][:n]
                true_values = y_true[outp]
                value = precision_calculator(np.array(true_values))

        else:

            outp_max = y_true.argsort()[::-1][:n]
            true_values_max = y_true[outp_max]
            value = precision_calculator(np.array(true_values_max))

        return value

    @staticmethod
    def performance_metrics_n_found(y_pred, y_true, n_n_found, maximum=False, n=None):

        if n == None:
            n = len(y_true)

        def n_found_calculator(true_values):

            discounter = np.zeros(len(true_values))

            for i in range(len(true_values)):
                if i < n_n_found:
                    discounter[i] = 1
            gains = true_values

            n_found = np.sum(np.multiply(discounter, gains))

            return n_found

        if maximum == False:
            if np.all(y_pred == 0):
                value = 0
            else:
                outp = y_pred.argsort()[::-1][:n]
                true_values = y_true[outp]
                value = n_found_calculator(np.array(true_values))

        else:

            outp_max = y_true.argsort()[::-1][:n]
            true_values_max = y_true[outp_max]
            value = n_found_calculator(np.array(true_values_max))

        return value

    @staticmethod
    def performance_metrics_uplift(y_pred, y_true, maximum=False, n=None):

        if n == None:
            n = len(y_true)

        def uplift_calculator(true_values):

            discounter = np.zeros(len(true_values))

            disc = 0
            for i in range(len(true_values), 0, -1):
                disc = disc + 1 / (i)
                discounter[i - 1] = disc

            gains = true_values
            uplift = np.sum(np.multiply(gains, discounter))

            return uplift

        if maximum == False:
            if np.all(y_pred == 0):
                value = 0
            else:
                outp = y_pred.argsort()[::-1][:n]
                true_values = y_true[outp]
                value = uplift_calculator(np.array(true_values))

        else:

            outp_max = y_true.argsort()[::-1][:n]
            true_values_max = y_true[outp_max]
            value = uplift_calculator(np.array(true_values_max))

        return value

    @staticmethod
    def performance_metrics_roc_auc(y_pred, y_true, maximum=False, n=None):

        # fpr, tpr, threshold = metrics.roc_curve(y, preds)
        # roc_auc = metrics.auc(fpr, tpr)

        def roc_auc_calculator(true_values):

            discounter = np.zeros(len(true_values))
            gains = np.zeros(len(true_values))

            for i in range(len(true_values)):
                discounter[i] = 1
                gains[i] = (1 - true_values[i]) * np.sum(true_values[0:i + 1])

            roc_auc = np.sum(np.multiply(discounter, gains))

            return roc_auc

        if maximum == False:
            if np.all(y_pred == 0):
                value = 0
            else:
                outp = y_pred.argsort()[::-1][:n]
                true_values = y_true[outp]
                value = roc_auc_calculator(np.array(true_values))

        else:

            outp_max = y_true.argsort()[::-1][:n]
            true_values_max = y_true[outp_max]
            value = roc_auc_calculator(np.array(true_values_max))

        return value
