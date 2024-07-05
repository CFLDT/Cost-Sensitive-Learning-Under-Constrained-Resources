import numpy as np
import ALIGNED_learning.boost as ab

from ..logit import Logit
from sklearn.linear_model import LogisticRegression


class MethodLearner:

    @staticmethod
    def logit(par_dict_logit, X_train, y_train,y_train_clas):

        lambd = par_dict_logit.get("lambd")
        metric = par_dict_logit.get("metric")
        sigma = par_dict_logit.get("sigma")
        indic_approx = par_dict_logit.get("indic_approx")

        n_ratio = par_dict_logit.get("n_ratio")
        p_rbp = par_dict_logit.get("p_rbp")
        p_ep = par_dict_logit.get("p_ep")
        #n_c_ep = par_dict_logit.get("n_c_ep")
        p_prec = par_dict_logit.get("p_prec")

        if metric == 'basic':
            init_theta = np.zeros(np.shape(X_train)[1] + 1)
            #init_theta = np.random.uniform(low=-1, high=1, size=np.shape(X_train)[1] + 1)
        else:
            init_theta = np.zeros(np.shape(X_train)[1])
            #init_theta = np.random.uniform(low=-1, high=1, size=np.shape(X_train)[1])


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
        subsample = par_dict_lgbm.get("subsample")
        subsample_freq = par_dict_lgbm.get("subsample_freq")
        sampling_strategy = par_dict_lgbm.get("sampling_strategy")
        min_child_samples = par_dict_lgbm.get("min_child_samples")
        min_child_weight = par_dict_lgbm.get("min_child_weight")
        sigma = par_dict_lgbm.get("sigma")
        indic_approx = par_dict_lgbm.get("indic_approx")
        metric = par_dict_lgbm.get("metric")

        n_ratio = par_dict_lgbm.get("n_ratio")
        p_rbp = par_dict_lgbm.get("p_rbp")
        p_ep = par_dict_lgbm.get("p_ep")
        #n_c_ep = par_dict_lgbm.get("n_c_ep")
        p_prec = par_dict_lgbm.get("p_prec")


        # if sampling_strategy is None:
        #     neg_bagging_fraction = 1
        #
        # if sampling_strategy is not None:
        #     n_majority = np.count_nonzero(y_train) / sampling_strategy
        #     neg_bagging_fraction = n_majority / (len(y_train) - np.count_nonzero(y_train))
        #     neg_bagging_fraction = min(neg_bagging_fraction,1)


        lgboost = ab.Lgbm(n_estimators=n_estimators, num_leaves=num_leaves, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                       learning_rate=learning_rate, colsample_bytree = colsample_bytree, subsample=subsample, subsample_freq=subsample_freq,
                          sampling_strategy=sampling_strategy, min_child_samples=min_child_samples,
                       min_child_weight=min_child_weight, sigma=sigma, indic_approx=indic_approx)

        lgbst_train, time = lgboost.fitting(X_train, y_train, y_train_clas, metric=metric, n_ratio= n_ratio, p_prec = p_prec, p_rbp = p_rbp, p_ep = p_ep)

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
