from ..design import data_pipeline

class DivClean:

    @staticmethod
    def divide_clean(X, train_index, test_index):

        X_train, X_test = \
            DivClean.divider(X, train_index, test_index)

        datapipeline = DivClean.cleaner(X_train, X_test)

        return datapipeline

    @staticmethod
    def divider(X, train_index, test_index):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]

        return X_train, X_test


    @staticmethod
    def cleaner(X_train, X_test):

        datapipeline = data_pipeline.DataPipeline()
        _, _ = datapipeline.pipeline_fit_trans(X_train, X_test)

        return datapipeline
