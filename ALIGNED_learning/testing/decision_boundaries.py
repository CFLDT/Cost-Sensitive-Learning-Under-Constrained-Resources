import numpy as np

from ..plots_tables import dec_boundary_plotter
from ..design import MethodLearner


def decision_boundary(methods, par_dict, X_train, y_train, task_dict):

    name = task_dict['name']

    xx, yy = np.mgrid[X_train[:, 0].min() - 2:X_train[:, 0].max() + 2:0.02,
             X_train[:, 1].min() - 2:X_train[:, 1].max() + 2:0.02]

    grid = np.c_[xx.ravel(), yy.ravel()]

    # def normalizer(data):
    #     return (data - np.min(data)) / (np.max(data) - np.min(data))

    for method in methods:

        if method == 'Logit':

            model = MethodLearner.logit(par_dict.get('Logit'),par_dict.get('General'), X_train, y_train)
            grid_probs = model.predict(grid)
            name_2 = par_dict.get('Logit').get('metric')
            method = method + '_' + name_2

        elif method == 'Lgbm':

            lgboost, model = MethodLearner.lgbmboost(par_dict.get('Lgbm'), par_dict.get('General'), X_train, y_train)
            grid_probs = lgboost.predict(model, grid)
            name_2 = par_dict.get('Lgbm').get('metric')
            method = method + '_' + name_2
            #grid_probs = normalizer(grid_probs)


        else:

            continue


        dec_boundary_plotter(name, method, X_train,  y_train, xx, yy, grid_probs)
