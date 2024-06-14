import numpy as np
import ALIGNED_learning.boost as ab

from ..logit import Logit
from sklearn.linear_model import LogisticRegression


class MethodLearner:

    @staticmethod
    def logit(par_dict_logit, par_dict_general, X_train, y_train):

        lambd = par_dict_logit.get("lambd")
        metric = par_dict_logit.get("metric")
        sigma = par_dict_logit.get("sigma")
        indic_approx = par_dict_logit.get("indic_approx")

        n = par_dict_general.get("n")
        p_rbp = par_dict_general.get("p_rbp")
        p_ep = par_dict_general.get("p_ep")
        n_c_ep = par_dict_general.get("n_c_ep")
        p_prec = par_dict_general.get("p_prec")

        if metric == 'basic':
            init_theta = np.zeros(np.shape(X_train)[1] + 1)
            #init_theta = np.random.uniform(low=-1, high=1, size=np.shape(X_train)[1] + 1)
        else:
            init_theta = np.zeros(np.shape(X_train)[1])
            #init_theta = np.random.uniform(low=-1, high=1, size=np.shape(X_train)[1])


        logist = Logit(lambd=lambd, sigma=sigma, indic_approx=indic_approx)
        logist.fitting(X_train, y_train, init_theta, metric=metric, n = n, p_prec = p_prec, p_rbp = p_rbp, p_ep = p_ep, n_c_ep = n_c_ep)

        return logist


    @staticmethod
    def lgbmboost(par_dict_lgbm, par_dict_general, X_train, y_train):

        n_estimators = par_dict_lgbm.get("n_estimators")
        num_leaves = par_dict_lgbm.get("num_leaves")
        reg_lambda = par_dict_lgbm.get("lambd")
        learning_rate = par_dict_lgbm.get("learning_rate")
        subsample = par_dict_lgbm.get("subsample")
        min_child_samples = par_dict_lgbm.get("min_child_samples")
        min_child_weight = par_dict_lgbm.get("min_child_weight")
        sigma = par_dict_lgbm.get("sigma")
        indic_approx = par_dict_lgbm.get("indic_approx")
        metric = par_dict_lgbm.get("metric")

        n = par_dict_general.get("n")
        p_rbp = par_dict_general.get("p_rbp")
        p_ep = par_dict_general.get("p_ep")
        n_c_ep = par_dict_general.get("n_c_ep")
        p_prec = par_dict_general.get("p_prec")

        lgboost = ab.Lgbm(n_estimators=n_estimators, num_leaves=num_leaves, reg_lambda=reg_lambda,
                       learning_rate=learning_rate, subsample=subsample, min_child_samples=min_child_samples,
                       min_child_weight=min_child_weight, sigma=sigma, indic_approx=indic_approx)

        lgbst_train, time = lgboost.fitting(X_train, y_train, metric=metric,n = n, p_prec = p_prec, p_rbp = p_rbp, p_ep = p_ep, n_c_ep = n_c_ep)

        return lgboost, lgbst_train

    @staticmethod
    def ensimb(par_dict, X_clas_train, s_train):

        max_depth = par_dict.get("max_depth")
        n_estimators = par_dict.get("n_estimators")
        learning_rate = par_dict.get("learning_rate")
        ensemble_method = par_dict.get("method")
        sampling_strategy = par_dict.get("sampling_strategy")

        if sampling_strategy == None:
            sampling_strategy = np.sum(s_train) / (np.sum(1 - s_train))

        ensimb = ab.ensimb_sup.ENSMImb(n_estimators=n_estimators, max_depth=max_depth,
                         learning_rate=learning_rate, sampling_strategy=sampling_strategy)

        ensbimb_train, time = ensimb.fitting(X_clas_train, s_train, ensemble_method=ensemble_method)

        return ensimb, ensbimb_train
