import numpy as np
from sklearn.preprocessing import StandardScaler
from ..design import data_pipeline



def divide_clean(X, y, y_c, train_index, test_index):
    X_train, X_test, y_train, y_test, y_c_train, y_c_test = \
        divider(X, y, y_c, train_index, test_index)

    X_train, X_test, y_train, \
    y_test, y_c_train, y_c_test, datapipeline = cleaner(X_train, X_test, y_train, y_test, y_c_train, y_c_test)

    X_train, X_test, st_scaler = \
        scaler(X_train, X_test)

    return X_train, y_train, y_c_train, X_test, y_test, y_c_test, datapipeline, st_scaler


def scaler(X_train_or, X_test_or):
    st_scaler = StandardScaler()
    st_scaler.fit(X_train_or)
    X_train = st_scaler.transform(X_train_or)
    X_test = st_scaler.transform(X_test_or)

    return X_train, X_test, st_scaler


def divider(X, y, y_c, train_index, test_index):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    y_c_train, y_c_test = y_c.iloc[train_index], y_c.iloc[test_index]

    return X_train, X_test, y_train, y_test, y_c_train, y_c_test


def cleaner(X_train, X_test, y_train, y_test, y_c_train, y_c_test):
    datapipeline = data_pipeline.DataPipeline()
    X_train, X_test = datapipeline.pipeline_fit_trans(X_train, X_test, y_train)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_c_train = np.array(y_c_train)
    y_c_test = np.array(y_c_test)

    return X_train, X_test, y_train, y_test, y_c_train, y_c_test, datapipeline

