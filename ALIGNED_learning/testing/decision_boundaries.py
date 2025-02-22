import numpy as np

from ..plots_tables import dec_boundary_plotter
from ..design import MethodLearner

def decision_boundary(methods, par_dict, X_train, y_train, y_train_clas, task_dict):

    name = task_dict['name']

    xx, yy = np.mgrid[X_train[:, 0].min() - 2:X_train[:, 0].max() + 2:0.02,
             X_train[:, 1].min() - 2:X_train[:, 1].max() + 2:0.02]

    grid = np.c_[xx.ravel(), yy.ravel()]

    for method in methods:

        if method == 'Logit':

            model = MethodLearner.logit(par_dict.get('Logit'), X_train, y_train, y_train_clas)
            grid_probs = model.predict_proba(grid)
            name_2 = par_dict.get('Logit').get('metric')
            method_name = method + '_' + name_2

        elif method == 'Lgbm':

            lgboost, model = MethodLearner.lgbmboost(par_dict.get('Lgbm'), X_train, y_train, y_train_clas)
            grid_probs = lgboost.predict_proba(model, grid)
            name_2 = par_dict.get('Lgbm').get('metric')
            method_name = method + '_' + name_2


        else:

            continue

        dec_boundary_plotter(name, method_name, X_train,  y_train, xx, yy, grid_probs)
