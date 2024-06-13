import pandas as pd
import pathlib
from pathlib import Path
import os
import glob

import warnings
warnings.filterwarnings("ignore")

base_path = Path(__file__).parent

path = (base_path / "../tables/database results/AAER/basic").resolve()
all_files = glob.glob(os.path.join(path , "*.csv"))

def add_hline(latex: str, index: int) -> str:

    lines = latex.splitlines()
    lines.insert(len(lines) - index - 2, r'\midrule')
    return '\n'.join(lines).replace('NaN', '')

for filename in all_files:
    df = pd.read_csv(filename,index_col=0)
    df = df.rename(columns={"0": pathlib.PurePath(filename).name})

    if ('all' in filename):

        df = df.reindex(columns=['Logit', 'M_score', 'F_score', 'Rusboost', 'Lgbm'])
        if ('ROC_AUC' in filename):
            print('ROC_AUC')
        if ('AP' in filename):
            print('AP')
        if ('DCG' in filename):
            print('DCG')
        if ('ARP' in filename):
            print('ARP')
        if ('RBP' in filename):
            print('RBP')
        if ('Uplift' in filename):
            print('Uplift')
        if ('EP' in filename):
            print('EP')
        if ('Precision' in filename):
            print('Precision')
        if ('experiment_1' in filename):
                print('experiment_1')
                df.index = df.index.str.replace('experiment\_1\_', '')
        if ('experiment_2' in filename):
                print('experiment_2')
                df.index = df.index.str.replace('experiment\_2\_', '')
        if ('experiment_3' in filename):
                print('experiment_3')
                df.index = df.index.str.replace('experiment\_3\_', '')
        if ('experiment_4' in filename):
                print('experiment_4')
                df.index = df.index.str.replace('experiment\_4\_', '')
        if ('experiment_5' in filename):
                print('experiment_5')
                df.index = df.index.str.replace('experiment\_5\_', '')
        if ('experiment_6' in filename):
                print('experiment_6')
                df.index = df.index.str.replace('experiment\_6\_', '')
        if ('experiment_7' in filename):
                print('experiment_7')
                df.index = df.index.str.replace('experiment\_7\_', '')
        if ('experiment_8' in filename):
                print('experiment_8')
                df.index = df.index.str.replace('experiment\_8\_', '')
        df = df.T
        df = df.applymap('{:,.3f}'.format)
        latex = df.to_latex()
        for i in range(len(df) - 1):
            latex = add_hline(latex=latex, index=2 * i + 1)
        print(latex)

