import numpy as np
from imblearn.ensemble import RUSBoostClassifier, BalancedRandomForestClassifier, BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from timeit import default_timer as timer
from .ensemimb import ENSEMImb
import random


class ENSMImb(ENSEMImb):

    def __init__(self, n_estimators, max_depth, learning_rate, sampling_strategy):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.sampling_strategy = sampling_strategy

        super().__init__()

    def fitting(self, X_clas, s, ensemble_method):

        random.seed(2290)
        np.random.seed(2290)

        starttimer = timer()

        if ensemble_method == 'RUSBoost':
            base_estimator = DecisionTreeClassifier(max_depth=self.max_depth)

            model = RUSBoostClassifier(random_state=2290, n_estimators=self.n_estimators,
                                       learning_rate=self.learning_rate,
                                       base_estimator=base_estimator, sampling_strategy=self.sampling_strategy)
            model.fit(X_clas, s)

        endtimer = timer()

        return model, endtimer - starttimer
