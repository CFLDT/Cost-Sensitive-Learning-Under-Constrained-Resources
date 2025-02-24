import numpy as np
import ALIGNED_learning.boost as ab
from ..logit import Logit


class MethodLearner:

    @staticmethod
    def logit(par_dict_logit, X_train, y_train, y_train_clas):

        lambd = par_dict_logit.get("lambd")
        metric = par_dict_logit.get("metric")
        sigma = par_dict_logit.get("sigma")
        indic_approx = par_dict_logit.get("indic_approx")

        n_ratio = par_dict_logit.get("n_ratio")
        p_rbp = par_dict_logit.get("p_rbp")
        p_ep = par_dict_logit.get("p_ep")
        p_prec = par_dict_logit.get("p_prec")

        if metric == 'basic':
            init_theta = np.zeros(np.shape(X_train)[1] + 1)
        else:
            init_theta = np.zeros(np.shape(X_train)[1])


        logist = Logit(lambd=lambd, sigma=sigma, indic_approx=indic_approx)
        logist.fitting(X_train, y_train, y_train_clas, init_theta, metric=metric, n_ratio= n_ratio, p_prec = p_prec, p_rbp = p_rbp, p_ep = p_ep)

        return logist


    @staticmethod
    def lgbmboost(par_dict_lgbm, X_train, y_train, y_train_clas):

        n_estimators = par_dict_lgbm.get("n_estimators")
        num_leaves = par_dict_lgbm.get("num_leaves")
        reg_lambda = par_dict_lgbm.get("lambd")
        reg_alpha = par_dict_lgbm.get("alpha")
        learning_rate = par_dict_lgbm.get("learning_rate")
        colsample_bytree = par_dict_lgbm.get("colsample_bytree")
        sample = par_dict_lgbm.get("sample_subsample_undersample")
        subsample = sample[0]
        undersample = sample[1]
        subsample_freq = par_dict_lgbm.get("subsample_freq")
        min_child_samples = par_dict_lgbm.get("min_child_samples")
        min_child_weight = par_dict_lgbm.get("min_child_weight")
        sigma = par_dict_lgbm.get("sigma")
        indic_approx = par_dict_lgbm.get("indic_approx")
        metric = par_dict_lgbm.get("metric")

        n_ratio = par_dict_lgbm.get("n_ratio")
        p_rbp = par_dict_lgbm.get("p_rbp")
        p_ep = par_dict_lgbm.get("p_ep")
        p_prec = par_dict_lgbm.get("p_prec")


        lgboost = ab.Lgbm(n_estimators=n_estimators, num_leaves=num_leaves, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                          learning_rate=learning_rate, colsample_bytree = colsample_bytree, subsample=subsample, subsample_freq=subsample_freq,
                          undersample=undersample, min_child_samples=min_child_samples,
                          min_child_weight=min_child_weight, sigma=sigma, indic_approx=indic_approx)

        lgbst_train, time = lgboost.fitting(X_train, y_train, y_train_clas, metric=metric, n_ratio= n_ratio, p_prec = p_prec,
                                            p_rbp = p_rbp, p_ep = p_ep)

        return lgboost, lgbst_train

    @staticmethod
    def ensimb(par_dict, X_train, y_train):

        max_depth = par_dict.get("max_depth")
        n_estimators = par_dict.get("n_estimators")
        learning_rate = par_dict.get("learning_rate")
        ensemble_method = par_dict.get("method")
        undersample = par_dict.get("undersample")

        if undersample == None:
            undersample = np.sum(y_train) / (np.sum(1 - y_train))

        ensimb = ab.ENSMImb(n_estimators=n_estimators, max_depth=max_depth,
                                       learning_rate=learning_rate, undersample=undersample)

        ensbimb_train, time = ensimb.fitting(X_train, y_train, ensemble_method=ensemble_method)

        return ensimb, ensbimb_train
