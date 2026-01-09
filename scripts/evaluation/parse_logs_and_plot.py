#!/usr/bin/env python3

import re, sys
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

def load_log_lines(filename: str) -> List[str]:
    with open(filename) as infile:
        contents = infile.read()

    # replace carriage return and escape [A by newline
    contents.replace('\r', '\n')
    ansi_escape = re.compile(r'\x1b\[A')
    contents = ansi_escape.sub('\n', contents)

    lines = contents.splitlines()
    return lines

def get_num_train_val_images(contents: List[str]) -> None:
    # TODO extract numbers rather than printing the entire line
    lines = [c for c in contents if 'Train set' in c]
    assert len(lines) == 1
    line = lines[0]
    # 2025-12-16 12:07:07,678 - SubprocessLauncher - INFO - Train set: 194, Val set: 50
    line_matcher = re.compile(r'.*Train set: (?P<num_train>\d*), Val set: (?P<num_val>\d*)$')
    match = line_matcher.match(line)
    return match.group('num_train'), match.group('num_val')

def _extract_validation_AUC_ROC_lines(contents: List[str]) -> List[str]:
    lines = [c for c in contents if 'val ACC' in c]                  # the lines contain ACC and AUC_ROC
    lines = lines[:1] + lines[2:]                                    # lines[1] seems to be from the sanity check that just tries two images (unclear why it is not lines[0] …)
    return lines

def _parse_AUC_ROC_values(lines: List[str], regex: str) -> Dict[int, float]:
    values:Dict[int, float] = {}
    for l in lines:
        line_matcher = re.compile(regex)
        match = line_matcher.match(l)
        values[int(match.group('epoch'))] = float(match.group('auroc'))
    return values


def parse_training_AUC_ROCs(contents: List[str]) -> Dict[int, float]:
    lines = [c for c in contents if 'train ACC' in c]  # the lines contain ACC and AUC_ROC
    values = _parse_AUC_ROC_values(lines, r'.*Epoch (?P<epoch>\d*).* AUC_ROC: (?P<auroc>[0-9.]*)$')
    return values

def parse_validation_AUC_ROCs(contents: List[str]) -> List[float]:
    lines = [l for l in _extract_validation_AUC_ROC_lines(contents) if l.startswith('Epoch ')]  # validation after training epochs
    values = _parse_AUC_ROC_values(lines, r'^Epoch (?P<epoch>\d*).* AUC_ROC: (?P<auroc>[0-9.]*)$')
    return values

def parse_validation_AUC_ROCs_aggregated_models(contents: List[str]) -> Dict[int, float]:
    lines = [l for l in _extract_validation_AUC_ROC_lines(contents) if not l.startswith('Epoch ')]  # validation using distributed aggregated model before epochs of the round
    values = _parse_AUC_ROC_values(lines, r'.+Epoch (?P<epoch>\d*).* AUC_ROC: (?P<auroc>[0-9.]*)$')
    return values


# from https://colorbrewer2.org/?type=qualitative&scheme=Set1&n=8
color_for_site = {'CAM' : '#e41a1c',
                  'MHA' : '#377eb8',
                  'RSH' : '#4daf4a',
                  'RUMC': '#984ea3',
                  'UKA' : '#ff7f00',
                  'UMCU': '#ffff33',
                  'USZ' : '#a65628',
                  'VHIO': '#f781bf' }

def plot_per_site(data: Dict[str, Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]]) -> None:
    fig, ax = plt.subplots(4, 2, figsize=(12,16))
    for pos, site_name in [((0, 0), 'CAM' ),
                           ((0, 1), 'MHA' ),
                           ((1, 0), 'RSH' ),
                           ((1, 1), 'RUMC'),
                           ((2, 0), 'UKA' ),
                           ((2, 1), 'UMCU'),
                           ((3, 0), 'USZ' ),
                           ((3, 1), 'VHIO') ]:
        training_auc_roc, validation_auc_roc, validation_auc_roc_agm = data[site_name]
        ax[pos].plot(*zip(*sorted(training_auc_roc.items())),       '-',   c=color_for_site[site_name], linewidth=0.5, label='training AUC_ROC')
        ax[pos].plot(*zip(*sorted(validation_auc_roc.items())),     '-',   c=color_for_site[site_name], linewidth=2,   label='validation AUC_ROC')
        ax[pos].plot(*zip(*sorted(validation_auc_roc_agm.items())), '--x', c=color_for_site[site_name], markersize=6,  label='validation AUC_ROC aggregated model')
        ax[pos].set_xlim([0.0, 100.0])
        ax[pos].set_ylim([0.0, 1.0])
        ax[pos].legend()
        ax[pos].set_title(f'{site_name}')
    plt.savefig(f'convergence_per_site.png')


def plot_overviews(data: Dict[str, Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]]) -> None:
    fig, ax = plt.subplots(3, 1, figsize=(6,12))
    for site_name, site_data in data.items():
        training_auc_roc, validation_auc_roc, validation_auc_roc_agm = site_data
        ax[0].plot(*zip(*sorted(training_auc_roc.items())),       '-',   c=color_for_site[site_name], linewidth=1,  label=site_name)
        ax[1].plot(*zip(*sorted(validation_auc_roc.items())),     '-',   c=color_for_site[site_name], linewidth=2,  label=site_name)
        ax[2].plot(*zip(*sorted(validation_auc_roc_agm.items())), '--x', c=color_for_site[site_name], markersize=6, label=site_name)

        ax[0].set_title('training AUC_ROC')
        ax[0].legend()  # only one legend where it is least distracting
        ax[1].set_title('validation AUC_ROC')
        ax[2].set_title('validation AUC_ROC (aggregated model)')

        for i in range(3):
            ax[i].set_xlim([0.0, 100.0])
            ax[i].set_ylim([0.0, 1.0])

    plt.savefig(f'convergence_overview.png')


if __name__ == '__main__':
    # this script expects a folder structure SITE_NAME/log.txt with optional SITE_NAME/local_training_console_output.txt
    data: Dict[str, Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]] = {}
    for site_name in color_for_site.keys():
        logfilename = Path(site_name)/'log.txt'
        contents = load_log_lines(logfilename)
        training_auc_roc = parse_training_AUC_ROCs(contents)
        validation_auc_roc = parse_validation_AUC_ROCs(contents)
        validation_auc_roc_agm = parse_validation_AUC_ROCs_aggregated_models(contents)
        num_train, num_val = get_num_train_val_images(contents)
        print(f'{site_name: <4}: {num_train: >5} training images, {num_val: >5} validation images, ' +
              f'validation AUROC (last global model): {validation_auc_roc_agm[95]:.4f}, ' +
              f'training AUROC (last local model): {training_auc_roc[99]:.4f}, ' +
              f'validation AUROC (last local model): {validation_auc_roc[99]:.4f}'
              )
        data[site_name] = (training_auc_roc, validation_auc_roc, validation_auc_roc_agm)

    plot_per_site(data)
    plot_overviews(data)
