#!/usr/bin/env python3

import re, sys
from typing import List, Dict
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

def print_num_train_val_images(contents: List[str]) -> None:
    # TODO extract numbers rather than printing the entire line
    lines = [c for c in contents if 'Train set' in c]
    assert len(lines) == 1
    line = lines[0]
    print(line)

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
color_for_site = {'CAM':  '#e41a1c',
                  'MHA':  '#377eb8',
                  'RSH':  '#4daf4a',
                  'RUMC': '#984ea3',
                  'UKA':  '#ff7f00',
                  'UMCU': '#ffff33',
                  'USZ':  '#a65628',
                  'VHIO': '#f781bf' }

def plot_for_site(training_roc_auc: Dict[int, float],
                  validation_roc_auc: Dict[int, float],
                  validation_roc_auc_agm: Dict[int, float],
                  site_name: str) -> None:
    print(training_auc_roc, '\n', validation_auc_roc, '\n', validation_auc_roc_agm, '\n', site_name)
    fig, ax = plt.subplots()
    ax.plot(*zip(*sorted(training_roc_auc.items())),       '-',  c=color_for_site[site_name], linewidth=0.5, label='training AUC_ROC')
    ax.plot(*zip(*sorted(validation_roc_auc.items())),     '-',  c=color_for_site[site_name], linewidth=2,   label='validation AUC_ROC')
    ax.plot(*zip(*sorted(validation_roc_auc_agm.items())), '-x', c=color_for_site[site_name], markersize=6,  label='validation AUC_ROC aggregated model')
    plt.ylim([0.0, 1.0])
    plt.legend()
    plt.title(f'{site_name}')
    plt.savefig(f'convergence_{site_name}.png')


contents = load_log_lines(sys.argv[1])
training_auc_roc = parse_training_AUC_ROCs(contents)
validation_auc_roc = parse_validation_AUC_ROCs(contents)
validation_auc_roc_agm = parse_validation_AUC_ROCs_aggregated_models(contents)

print_num_train_val_images(contents)
print(validation_auc_roc_agm[95], training_auc_roc[99], validation_auc_roc[99])

site_name = sys.argv[1].split('/')[0]
plot_for_site(training_auc_roc, validation_auc_roc, validation_auc_roc_agm, site_name)

# TODO plot site comparsion (requires loading multiple files)
