import pandas as pd
import pathlib
from pathlib import Path
import os
import glob


import warnings
warnings.filterwarnings("ignore")

base_path = Path(__file__).parent

path = (base_path / "../tables/database results").resolve()
all_files = glob.glob(os.path.join(path , "*.csv"))

experiment_1 = pd.DataFrame()
experiment_2 = pd.DataFrame()
experiment_3 = pd.DataFrame()
experiment_4 = pd.DataFrame()
experiment_5 = pd.DataFrame()
experiment_6 = pd.DataFrame()
experiment_7 = pd.DataFrame()
experiment_8 = pd.DataFrame()

for filename in all_files:
    df = pd.read_csv(filename,index_col=0)
    df = df.rename(columns={"0": pathlib.PurePath(filename).name})
    df= df.T

    if ('all' not in filename):
        df = df.rename(columns={'ENSImb': 'RUSBoost'})
        df = df.reindex(columns=['Logit', 'M_score', 'F_score', 'RUSBoost', 'Lgbm'])
        df.index = df.index.str.replace('ROC_AUC', 'ROC-AUC')
        if ('experiment_1' in filename):
            df.index = df.index.str.replace('experiment\_1\_', '')
            df.index = df.index.str.replace('.csv', '')
            experiment_1 = experiment_1.append(df)
        if ('experiment_2' in filename):
            df.index = df.index.str.replace('experiment\_2\_', '')
            df.index = df.index.str.replace('.csv', '')
            experiment_2 = experiment_2.append(df)
        if ('experiment_3' in filename):
            df.index = df.index.str.replace('experiment\_3\_', '')
            df.index = df.index.str.replace('.csv', '')
            experiment_3 = experiment_3.append(df)
        if ('experiment_4' in filename):
            df.index = df.index.str.replace('experiment\_4\_', '')
            df.index = df.index.str.replace('.csv', '')
            experiment_4 = experiment_4.append(df)
        if ('experiment_5' in filename):
            df.index = df.index.str.replace('experiment\_5\_', '')
            df.index = df.index.str.replace('.csv', '')
            experiment_5 = experiment_5.append(df)
        if ('experiment_6' in filename):
            df.index = df.index.str.replace('experiment\_6\_', '')
            df.index = df.index.str.replace('.csv', '')
            experiment_6 = experiment_6.append(df)
        if ('experiment_7' in filename):
            df.index = df.index.str.replace('experiment\_7\_', '')
            df.index = df.index.str.replace('.csv', '')
            experiment_7 = experiment_7.append(df)
        if ('experiment_8' in filename):
            df.index = df.index.str.replace('experiment\_8\_', '')
            df.index = df.index.str.replace('.csv', '')
            experiment_8 = experiment_8.append(df)

def add_hline(latex: str, index: int) -> str:

    lines = latex.splitlines()
    lines.insert(len(lines) - index - 2, r'\midrule')
    return '\n'.join(lines).replace('NaN', '')

experiment_1.index = ['AP', 'ARP', 'DCG', 'EP', 'Precision', 'RBP', 'ROC-AUC', 'Uplift']
experiment_1 = experiment_1.reindex(columns=['Logit', 'M_score', 'F_score', 'RUSBoost', 'Lgbm'])

experiment_2.index = ['AP', 'ARP', 'DCG', 'EP', 'Precision', 'RBP', 'ROC-AUC', 'Uplift']
experiment_2 = experiment_2.reindex(columns=['Logit', 'M_score', 'F_score', 'RUSBoost', 'Lgbm'])

experiment_3.index = ['AP', 'ARP', 'DCG', 'EP', 'Precision', 'RBP', 'ROC-AUC', 'Uplift']
experiment_3 = experiment_3.reindex(columns=['Logit', 'M_score', 'F_score', 'RUSBoost', 'Lgbm'])

experiment_4.index = ['AP', 'ARP', 'DCG', 'EP', 'Precision', 'RBP', 'ROC-AUC', 'Uplift']
experiment_4 = experiment_4.reindex(columns=['Logit', 'M_score', 'F_score', 'RUSBoost', 'Lgbm'])

experiment_5.index = ['AP', 'ARP', 'DCG', 'EP', 'Precision', 'RBP', 'ROC-AUC', 'Uplift']
experiment_5 = experiment_5.reindex(columns=['Logit', 'M_score', 'F_score', 'RUSBoost', 'Lgbm'])

experiment_6.index = ['AP', 'ARP', 'DCG', 'EP', 'Precision', 'RBP', 'ROC-AUC', 'Uplift']
experiment_6 = experiment_6.reindex(columns=['Logit', 'M_score', 'F_score', 'RUSBoost', 'Lgbm'])

experiment_7.index = ['AP', 'ARP', 'DCG', 'EP', 'Precision', 'RBP', 'ROC-AUC', 'Uplift']
experiment_7 = experiment_7.reindex(columns=['Logit', 'M_score', 'F_score', 'RUSBoost', 'Lgbm'])

experiment_8.index = ['AP', 'ARP', 'DCG', 'EP', 'Precision', 'RBP', 'ROC-AUC', 'Uplift']
experiment_8 = experiment_8.reindex(columns=['Logit', 'M_score', 'F_score', 'RUSBoost', 'Lgbm'])

experiment_1 = experiment_1.T
experiment_2 = experiment_2.T
experiment_3 = experiment_3.T
experiment_4 = experiment_4.T
experiment_5 = experiment_5.T
experiment_6 = experiment_6.T
experiment_7 = experiment_7.T
experiment_8 = experiment_8.T

print('experiment_1')
experiment_1 = experiment_1.applymap('{:,.3f}'.format)
latex = experiment_1.to_latex()
for i in range(len(experiment_1)-1):
    latex = add_hline(latex=latex, index=2 * i + 1)
print(latex)

print('experiment_2')
experiment_2 = experiment_2.applymap('{:,.3f}'.format)
latex = experiment_2.to_latex()
for i in range(len(experiment_2) - 1):
    latex = add_hline(latex=latex, index=2 * i + 1)
print(latex)

print('experiment_3')
experiment_3 = experiment_3.applymap('{:,.3f}'.format)
latex = experiment_3.to_latex()
for i in range(len(experiment_3) - 1):
    latex = add_hline(latex=latex, index=2 * i + 1)
print(latex)

print('experiment_4')
experiment_4 = experiment_4.applymap('{:,.3f}'.format)
latex = experiment_4.to_latex()
for i in range(len(experiment_4) - 1):
    latex = add_hline(latex=latex, index=2 * i + 1)
print(latex)

print('experiment_5')
experiment_5 = experiment_5.applymap('{:,.3f}'.format)
latex = experiment_5.to_latex()
for i in range(len(experiment_5) - 1):
    latex = add_hline(latex=latex, index=2 * i + 1)
print(latex)

print('experiment_6')
experiment_6 = experiment_6.applymap('{:,.3f}'.format)
latex = experiment_6.to_latex()
for i in range(len(experiment_6) - 1):
    latex = add_hline(latex=latex, index=2 * i + 1)
print(latex)

print('experiment_7')
experiment_7 = experiment_7.applymap('{:,.3f}'.format)
latex = experiment_7.to_latex()
for i in range(len(experiment_7) - 1):
    latex = add_hline(latex=latex, index=2 * i + 1)
print(latex)


print('experiment_8')
experiment_8 = experiment_8.applymap('{:,.3f}'.format)
latex = experiment_8.to_latex()
for i in range(len(experiment_8) - 1):
    latex = add_hline(latex=latex, index=2 * i + 1)
print(latex)

