import numpy as np
import pandas as pd
from ALIGNED_learning.design import DataHandler
from sklearn.preprocessing import OneHotEncoder

class DataPipeline:

    def __init__(self):
        pass

    def pipeline_fit_trans(self, X_train, X_test):

        self.num_col_names = X_train.select_dtypes(include=np.number).columns.tolist()
        all_col = X_train.columns.tolist()
        self.cat_col_names = list(set(all_col) ^ set(self.num_col_names))

        """
        Inpute the median for numerical features
        """

        X_train_median = X_train.copy()
        X_test_median = X_test.copy()
        X_train_median, X_test_median, self.median = DataHandler. \
            impute_median(X_train_median, X_test_median,
                          self.num_col_names)

        """
        One hot encoding 
        """

        X_train_ohe = X_train_median.copy()
        X_test_ohe = X_test_median.copy()

        self.ohe = OneHotEncoder(categories='auto', sparse=False, handle_unknown="ignore")

        OH_cols_train = pd.DataFrame(self.ohe.fit_transform(X_train_ohe[self.cat_col_names]))
        OH_cols_valid = pd.DataFrame(self.ohe.transform(X_test_ohe[self.cat_col_names]))

        OH_cols_train.columns = self.ohe.get_feature_names_out(self.cat_col_names)
        OH_cols_valid.columns = self.ohe.get_feature_names_out(self.cat_col_names)

        OH_cols_train.index = X_train_ohe.index
        OH_cols_valid.index = X_test_ohe.index

        num_X_train = X_train_ohe.drop(self.cat_col_names, axis=1)
        num_X_valid = X_test_ohe.drop(self.cat_col_names, axis=1)

        X_train_ohe = pd.concat([num_X_train, OH_cols_train], axis=1)
        X_test_ohe = pd.concat([num_X_valid, OH_cols_valid], axis=1)

        """
        Delete constant and all NaN columns (also constant columns with NaN values)
        """

        bool =  (X_train_ohe != X_train_ohe.iloc[0]).any()
        bool_nan = ~X_train_ohe.isnull().all()
        X_train_ohe = X_train_ohe.loc[:, (bool & bool_nan)]
        X_test_ohe = X_test_ohe.loc[:,(bool & bool_nan)]

        self.colnames = X_train_ohe.columns.tolist()

        return X_train_ohe, X_test_ohe

    def pipeline_trans(self, X, keep_nan = False):

        """
        Inpute the median for numerical features
        """

        X_median = X.copy()
        if keep_nan == False:
            X_median,X_test_median = DataHandler.\
                impute_values(X_median, None,
                              self.num_col_names, self.median)


        """
        One hot encoding 
        """

        X_ohe = X_median.copy()

        OH_cols = pd.DataFrame(self.ohe.transform(X_ohe[self.cat_col_names]))

        OH_cols.columns = self.ohe.get_feature_names_out(self.cat_col_names)

        OH_cols.index = X_ohe.index

        num_X = X_ohe.drop(self.cat_col_names, axis=1)

        X_ohe = pd.concat([num_X, OH_cols], axis=1)

        X_ohe = X_ohe[self.colnames]

        return X_ohe

