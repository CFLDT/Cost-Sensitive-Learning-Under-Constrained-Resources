import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import random
from ALIGNED_learning.testing import decision_boundary
from ALIGNED_learning.design import DivClean
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
from sklearn.preprocessing import StandardScaler



random.seed(2290)
np.random.seed(2290)
base_path = Path(__file__).parent

n_n = 200
n_fe = 10
n_fc = 60

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

y_n = np.random.choice([0, 1], size=(n_n,), p=[0.5, 0.5])
# y_fc = np.random.choice([0, 1], size=(n_fc,), p=[0.5, 0.5])
# y_fe = np.random.choice([0, 1], size=(n_fe,), p=[0.5, 0.5])


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

datapipeline = \
    DivClean.divide_clean(X, train_index, test_index)

def scaler(X_train_or, X_test_or):
    st_scaler = StandardScaler()
    st_scaler.fit(X_train_or)
    X_train = st_scaler.transform(X_train_or)
    X_test = st_scaler.transform(X_test_or)

    return X_train, X_test, st_scaler

X_train, X_val, _ = scaler(
    np.array(datapipeline.pipeline_trans(X)),
    np.array(datapipeline.pipeline_trans(X)))


task_dict = {'name': 'Toy_Data_1'}

opt_par_dict = {'General': {'n':1,
                            'p_prec':  0.1,
                            'p_rbp':  0.8,
                            'p_ep': 0.1,
                            'n_c_ep': 100},
                'Logit': {'lambd': 1,
                          'sigma': 100,
                          'indic_approx': 'logit', #'lambdaloss', 'logit'
                          'metric': 'roc_auc'  # basic, roc_auc, arp, ap, dcg, ep, rbp, ep, precision
                          },
                'Lgbm': {"num_leaves": 5,
                         "n_estimators": 100,
                         "lambd": 0,
                         "learning_rate": 0.1,
                         "subsample": 1,
                         "min_child_samples": 0,
                         "min_child_weight": 1e-3 ,   #1e-3 do not change. this causes issues regarding validation 'binary' and 'lambdarank'
                         "sigma": 1,                   # 1 for validation 'binary' and 'lambdarank'
                         "indic_approx": 'logit',   #'lambdaloss', 'logit'   #lambdaloss for validation 'binary' and 'lambdarank'
                         "metric": 'ep'}}    # basic, lambdarank, arp, roc_auc, ap, dcg, ep, rbp, precision


# TO COMPARE GO TO LGBM FILE AND GO TO THE LINE(S) WITH THE WORDS 'COMPARE'

# TEST CORRECTNESS ROC AUC ETC PERFORMANCE MEASURES

#https://lightgbm.readthedocs.io/en/latest/Parameters.html#lambdarank_truncation_level
dec_bound = True
if dec_bound:
    decision_boundary(['Logit'], opt_par_dict, X_train, y, task_dict=task_dict)

