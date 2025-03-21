import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import random
from ALIGNED_learning.testing import decision_boundary
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
from sklearn.preprocessing import StandardScaler

random.seed(2290)
np.random.seed(2290)
base_path = Path(__file__).parent

n_n = 200
n_fe = 30
n_fc = 30

x_n_1, x_n_2 = np.random.multivariate_normal([6, 5], [[1, 0], [0, 2]], size=n_n).T
x_fe_1, x_fe_2 = np.random.multivariate_normal([11.5, 1.5], [[0.3, 0], [0, 0.3]], size=n_fe).T
x_fc_1, x_fc_2 = np.random.multivariate_normal([8.5, 10.5], [[0.8, 0], [0, 0.5]], size=n_fc).T


x_n_1 = x_n_1.reshape(-1, 1)
x_n_2 = x_n_2.reshape(-1, 1)
x_fe_1 = x_fe_1.reshape(-1, 1)
x_fe_2 = x_fe_2.reshape(-1, 1)
x_fc_1 = x_fc_1.reshape(-1, 1)
x_fc_2 = x_fc_2.reshape(-1, 1)

y_n = np.zeros(n_n)
y_fc = np.ones(n_fc)
y_fe = np.ones(n_fe)

#y_n = np.random.choice([0, 1], size=(n_n,), p=[0.7, 0.3])
y_fc = np.random.choice([0, 1], size=(n_fc,), p=[0.5, 0.5])
y_fe = np.random.choice([0, 1], size=(n_fe,), p=[0.5, 0.5])


x_n = np.concatenate((x_n_1, x_n_2), axis=1)
x_fe = np.concatenate((x_fe_1, x_fe_2), axis=1)
x_fc = np.concatenate((x_fc_1, x_fc_2), axis=1)

X = np.concatenate((x_n, x_fe), axis=0)
X = np.concatenate((X, x_fc), axis=0)

y = np.concatenate((y_n, y_fe), axis=0)
y = np.concatenate((y, y_fc), axis=0)

min_max_scaler = MinMaxScaler()
min_max_scaler.fit((np.array(X)))
X = min_max_scaler.transform(np.array(X)) * 0.7 + 0.15

dataset = pd.DataFrame({'X1': X[:, 1], 'X2': X[:, 0], 'y': y})

X = dataset[['X1', 'X2']]
y = dataset['y']

train_index = list(np.linspace(0, len(y) - 1, num=len(y), dtype=int))
test_index = list(np.linspace(0, len(y) - 1, num=len(y), dtype=int))

def scaler(X_train_or, X_test_or):
    st_scaler = StandardScaler()
    st_scaler.fit(X_train_or)
    X_train = st_scaler.transform(X_train_or)
    X_test = st_scaler.transform(X_test_or)

    return X_train, X_test, st_scaler

X_train, X_val, _ = scaler(
    np.array(X),
    np.array(X))


task_dict = {'name': 'Toy_Data_1'}

opt_par_dict = {'General_val_test': {'n_ratio': 1,
                            'n_p_prec': 26,
                            'p_rbp': 0.9,
                            'n_p_ep': 26,
                            'n_p_ep_val': 26,
                            'n_n_found': 26},
                'Logit': {'lambd': 0,
                          'sigma': 1,
                          'indic_approx': 'lambdaloss', #'lambdaloss', 'logit'
                          'metric': 'basic'
                          },
                'Lgbm': {"num_leaves": 5,
                         "n_estimators": 100,
                         "lambd": 0,
                         "alpha": 0,
                         "learning_rate": 0.1,
                         "sample_subsample_undersample": [0.5, None],     #undersample not yet implemented
                         "subsample_freq": 1,
                         "min_child_samples": 0,
                         "min_child_weight": 1e-3 ,
                         "sigma": 1,
                         "indic_approx": 'lambdaloss',  #'lambdaloss', 'logit'
                         "metric": 'ep'}}

opt_par_dict["Lgbm"]["p_ep"] = 2/3
opt_par_dict["Lgbm"]["n_c_ep"] = max(1/(2/3), 1/(1 - 2/3))

# opt_par_dict["Lgbm"]["p_ep"] = 1/3
# opt_par_dict["Lgbm"]["n_c_ep"] = max(1/(1/3), 1/(1 - 1/3))


dec_bound = True
if dec_bound:
    decision_boundary(['Lgbm'], opt_par_dict, X_train, y, y, task_dict=task_dict)

