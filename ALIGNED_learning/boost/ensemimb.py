import numpy as np

class ENSEMImb:

    def __init__(self):

        pass

    def predict_proba(self, model, X_predict):

        scores = model.predict_proba(X_predict)[:,1]

        return scores


    def predict(self, model, X_predict):

        scores = model.predict(X_predict)

        return scores

    def likelihood_basic(self, p_clas, weight_div, div, s, weight_1=1, weight_0 =1):

        a = np.clip(p_clas, 1e-16, None)
        b = np.clip(1- p_clas, 1e-16, None)

        objective = div * weight_div * (s.dot(np.log(a))*weight_1 + (1-s).dot(np.log(b))*weight_0)        #exponential loss

        return objective
