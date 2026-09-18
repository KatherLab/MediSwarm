#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys
from itertools import product
from tqdm import tqdm

from sklearn.metrics import roc_auc_score

MODELS = ('1DivideAndConquer', '2BCN_AIM', '3agaldran', '4LME_ABMIL', '5Pimed', 'MST')
RUNS = ('A', 'B', 'C', 'D', 'E')
SITES = ('CAM', 'MHA', 'RSH', 'RUMC', 'UKA', 'UMCU', 'Z_Centralized')

def load_prediction(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, skiprows=1, names=['UID', 'label', 'prediction', 'score_0', 'score_1', 'score_2'])
    df = df.drop('UID', axis=1)
    df = df.drop('prediction', axis=1)
    return df

def compute_ovo_auroc(df: pd.DataFrame) -> float:
    if set(df.label.unique()) == {0, 1, 2}:
        return roc_auc_score(df.label,
                             df[['score_0', 'score_1', 'score_2']],
                             multi_class='ovo')
    else:
        return np.nan

def print_table(folder: str, mode: str, csv_filename: str) -> None:
    auroc_dfs = []

    for model, site, run in product(MODELS, SITES, RUNS):
        filename = f'{folder}/{model}_{run}/{mode}/{site}/{csv_filename}'
        df = load_prediction(filename)
        auroc = compute_ovo_auroc(df)
        auroc_dfs.append(pd.DataFrame(data=[{'model': model,
                                             'site': site,
                                             'run': run,
                                             'AUROC': auroc}]))
    aurocs = pd.concat(auroc_dfs)

    table_data = {}

    table_output = ['&' + ' & '.join(SITES) + r' \\']
    for model in MODELS:
        table_output_ = [f'{model}']
        for site in SITES:
            aurocs_model_site = aurocs[aurocs.model==model]
            aurocs_model_site = aurocs_model_site[aurocs_model_site.site==site]
            mean = np.mean(aurocs_model_site.AUROC)
            std = np.std(aurocs_model_site.AUROC)
            table_output_.append(f'{mean:0.3f}±{std:0.3f}')
            print (f'{model} {site}: {mean:0.3f}±{std:0.3f}')
        table_output.append(' & '.join(table_output_) + r' \\')
    print('\n'.join(table_output))

if __name__ == '__main__':
    print_table('local_training_reproduction', 'local', 'site_model_gt_and_classprob_test.csv')
    print_table('local_swarm_reproduction', 'swarm', 'final_aggregated_model_gt_and_classprob_test.csv')
