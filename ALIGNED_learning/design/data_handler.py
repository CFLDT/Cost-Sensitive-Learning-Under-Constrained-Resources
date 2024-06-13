import numpy as np

class DataHandler:

    @staticmethod
    def impute_median(data_train, data_test, columns):

        array_median = np.empty(len(columns))
        for i, col in enumerate(columns):
            median = data_train[col].median()
            array_median[i] = median
            data_train[col].fillna(median, inplace=True)
            data_test[col].fillna(median, inplace=True)
        return data_train, data_test, array_median

    @staticmethod
    def impute_values(data_train, data_test, columns, values):

        for i, col in enumerate(columns):
            data_train[col].fillna(values[i], inplace=True)
            if data_test is not None:
                data_test[col].fillna(values[i], inplace=True)
        return data_train, data_test

