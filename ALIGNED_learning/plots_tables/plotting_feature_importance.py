import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import shap

import numpy as np
import pandas as pd

from ALIGNED_learning.boost.ensemimb import ENSEMImb
from ALIGNED_learning.boost.lgbm import Lightgbm

from sklearn.model_selection import train_test_split



from pathlib import Path


def plot_feature_imp_shap(method, name, model, X, y, features):

        # https://github.com/slundberg/shap

        base_path = Path(__file__).parent

        if method == 'Logit' :

            pred_fun = model.predict_proba

        elif method == 'Lgbm':

            def pred(X):
                return Lightgbm().predict_proba(model, X)

            pred_fun = pred

        elif method == 'ENSImb':

            def pred(X):
                return ENSEMImb().predict(model, X)

            pred_fun = pred

        if (method == 'Lgbm'):

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            f, ax = plt.subplots(figsize=(8, 6))
            shap.summary_plot(shap_values, X, max_display=15, feature_names=features, show=False)

            # shap.plots.beeswarm(shap_values)
            plot_name_shap = "shap_" + name + '_' + method + ".png"

            plt.tight_layout()
            plt.savefig((base_path / "../../plots/plots feature importance" / method / plot_name_shap).resolve())
            plt.close()

            shap_sum = np.abs(shap_values).mean(axis=0)
            importance_df = pd.DataFrame([features, shap_sum.tolist()]).T
            importance_df.columns = ['column_name', 'shap_importance']
            importance_df = importance_df.sort_values('shap_importance', ascending=False)

            feature_importance_shap = "shap_feature_importance" + name + '_' + method + ".xlsx"
            importance_df.to_excel(
                (base_path / "../../plots/plots feature importance" / method / feature_importance_shap).resolve())


       # Kernelshap is quite slow.

        elif ((method == 'Logit') or (method == 'ENSImb')):

            if len(y) > 100:
                _, X = train_test_split(X, test_size=100 / len(y),
                                        stratify=y)

            explainer = shap.KernelExplainer(pred_fun, X)
            shap_values = explainer.shap_values(X)

            f, ax = plt.subplots(figsize=(8, 6))
            shap.summary_plot(shap_values, X, max_display=15, feature_names=features, show=False)

            # shap.plots.beeswarm(shap_values)
            plot_name_shap = "shap_" + name + '_' + method + ".png"

            plt.tight_layout()
            plt.savefig((base_path / "../../plots/plots feature importance" / method / plot_name_shap).resolve())
            plt.close()

            shap_sum = np.abs(shap_values).mean(axis=0)
            importance_df = pd.DataFrame([features, shap_sum.tolist()]).T
            importance_df.columns = ['column_name', 'shap_importance']
            importance_df = importance_df.sort_values('shap_importance', ascending=False)

            feature_importance_shap = "shap_feature_importance" + name + '_' + method + ".xlsx"
            importance_df.to_excel(
                (base_path / "../../plots/plots feature importance" / method / feature_importance_shap).resolve())



