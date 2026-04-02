#!/usr/bin/env python3

import os, re
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, field
from itertools import product
import matplotlib.pyplot as plt

@dataclass
class _LearningResults:
    training_auc_roc: Dict[int, float] = field(default_factory = lambda: ({}))
    validation_auc_roc: Dict[int, float] = field(default_factory = lambda: ({}))
    num_train_images: int = 0
    num_val_images: int = 0

    def has_data(self) -> bool:
        return self.training_auc_roc and self.validation_auc_roc


@dataclass
class LocalTrainingResults(_LearningResults):
    pass

@dataclass
class SwarmLearningResults(_LearningResults):
    validation_auc_roc_global_model: Dict[int, float] = field(default_factory = lambda: ({}))

    def has_data(self) -> bool:
        return super().has_data() and self.validation_auc_roc_global_model


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
    lines = [c for c in contents if 'Train set' in c]
    assert len(lines) == 1
    line = lines[0]
    # 2025-12-16 12:07:07,678 - SubprocessLauncher - INFO - Train set: 194, Val set: 50
    line_matcher = re.compile(r'.*Train set: (?P<num_train>\d*), Val set: (?P<num_val>\d*)$')
    match = line_matcher.match(line)
    return int(match.group('num_train')), int(match.group('num_val'))

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
    values = _parse_AUC_ROC_values(lines, r'.*Epoch (?P<epoch>\d*) \- .* AUC_ROC: (?P<auroc>[0-9.]*).*')
    return values

def parse_validation_AUC_ROCs(contents: List[str]) -> List[float]:
    lines = [l for l in _extract_validation_AUC_ROC_lines(contents) if l.startswith('Epoch ')]  # validation after training epochs
    values = _parse_AUC_ROC_values(lines, r'.*Epoch (?P<epoch>\d*) \- .* AUC_ROC: (?P<auroc>[0-9.]*).*$')
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

def plot_per_site(swarm_data: Dict[str, SwarmLearningResults], local_data: Dict[str, SwarmLearningResults]) -> None:
    pos_site_name = (((0, 0), 'CAM'), ((0, 1), 'MHA'), ((1, 0), 'RSH'), ((1, 1), 'RUMC'), ((2, 0), 'UKA'), ((2, 1), 'UMCU'), ((3, 0), 'USZ'), ((3, 1), 'VHIO'))

    def _plot_swarm_data(swarm_data: Dict[str, SwarmLearningResults], ax) -> None:
        for pos, site_name in pos_site_name:
            swarm_data_for_site = swarm_data[site_name]
            if swarm_data_for_site.has_data():
                ax[pos].plot(*zip(*sorted(swarm_data_for_site.training_auc_roc.items())), '-', c=color_for_site[site_name], linewidth=0.5, label='swarm training AUC_ROC')
                ax[pos].plot(*zip(*sorted(swarm_data_for_site.validation_auc_roc.items())), '-', c=color_for_site[site_name], linewidth=2, label='swarm validation AUC_ROC')
                ax[pos].plot(*zip(*sorted(swarm_data_for_site.validation_auc_roc_global_model.items())), '--x', c=color_for_site[site_name], markersize=6, label='swarm validation AUC_ROC aggregated model')

    def _plot_local_data(local_data: Dict[str, SwarmLearningResults], ax) -> None:
        for pos, site_name in pos_site_name:
            local_data_for_site = local_data[site_name]
            if local_data_for_site.has_data():
                ax[pos].plot(*zip(*sorted(local_data_for_site.training_auc_roc.items())), '-', c='#a0a0a0', linewidth=0.5, label='local training AUC_ROC')
                ax[pos].plot(*zip(*sorted(local_data_for_site.validation_auc_roc.items())), '-', c='#a0a0a0', linewidth=2, label='local validation AUC_ROC')
            else:
                ax[pos].text(10, 0.1, "no data for local training", color='red')

    def _add_num_images(swarm_data: Dict[str, SwarmLearningResults], local_data: Dict[str, SwarmLearningResults], ax) -> None:
        def _check_for_mismatch(num_swarm, num_local) -> None:
            if num_swarm > 0 and num_local > 0 and num_swarm != num_local:
                raise ("Mismatch in number of images used in swarm and local training")

        for pos, site_name in pos_site_name:
            num_train_images = 0
            num_val_images = 0
            swarm_data_for_site = swarm_data[site_name]
            if swarm_data_for_site.has_data():
                num_train_images = max(num_train_images, swarm_data_for_site.num_train_images)
                num_val_images = max(num_val_images, swarm_data_for_site.num_val_images)
            local_data_for_site = local_data[site_name]
            if local_data_for_site.has_data():
                num_train_images_local = local_data_for_site.num_train_images
                _check_for_mismatch(num_train_images, num_train_images_local)
                num_train_images = max(num_train_images, num_train_images_local)

                num_val_images_local = local_data_for_site.num_val_images
                _check_for_mismatch(num_val_images, num_val_images_local)
                num_val_images = max(num_val_images, num_val_images_local)

            ax[pos].set_title(f'{site_name}: {num_train_images} train, {num_val_images} val images')

    fig, ax = plt.subplots(4, 2, figsize=(12,16))
    _plot_swarm_data(swarm_data, ax)
    _plot_local_data(local_data, ax)
    _add_num_images(swarm_data, local_data, ax)

    for pos, _ in pos_site_name:
        ax[pos].set_xlim([0.0, 100.0])
        ax[pos].set_ylim([0.0, 1.0])

    ax[(0,0)].legend()

    plt.savefig(f'convergence_per_site.png')


def plot_overviews(swarm_data: Dict[str, SwarmLearningResults], local_data: Dict[str, SwarmLearningResults]) -> None:
    def _plot_swarm_results(swarm_data: Dict[str, SwarmLearningResults], ax) -> None:
        for site_name in color_for_site.keys():
            swarm_data_for_site = swarm_data[site_name]
            if swarm_data_for_site.has_data():
                ax[0][0].plot(*zip(*sorted(swarm_data_for_site.training_auc_roc.items())), '-', c=color_for_site[site_name], linewidth=1, label=site_name)
                ax[1][0].plot(*zip(*sorted(swarm_data_for_site.validation_auc_roc.items())), '-', c=color_for_site[site_name], linewidth=2, label=site_name)
                ax[2][0].plot(*zip(*sorted(swarm_data_for_site.validation_auc_roc_global_model.items())), '--x', c=color_for_site[site_name], markersize=6, label=site_name)

            ax[0][0].set_title('swarm training AUC_ROC')
            ax[1][0].set_title('swarm validation AUC_ROC')
            ax[2][0].set_title('swarm validation AUC_ROC (aggregated model)')

    def _plot_local_results(local_data: Dict[str, SwarmLearningResults], ax) -> None:
        for site_name in color_for_site.keys():
            if local_data[site_name].has_data():
                ax[0][1].plot(*zip(*sorted(local_data[site_name].training_auc_roc.items())), '-', c=color_for_site[site_name], linewidth=0.5, label=site_name)
                ax[1][1].plot(*zip(*sorted(local_data[site_name].validation_auc_roc.items())), '-', c=color_for_site[site_name], linewidth=1, label=site_name)

            ax[0][1].set_title('local training AUC_ROC')
            ax[1][1].set_title('local validation AUC_ROC')

    fig, ax = plt.subplots(3, 2, figsize=(12,12), sharex=True)
    _plot_swarm_results(swarm_data, ax)
    _plot_local_results(local_data, ax)

    for i, j in product(range(3), range(2)):
        ax[i][j].set_xlim([0.0, 100.0])
        ax[i][j].set_ylim([0.0, 1.0])

    ax[2][1].text(10, 0.1, "No aggregated models in local training", color='green')
    ax[0][0].legend()  # only one legend
    plt.savefig(f'convergence_overview.png')


def parse_swarm_training_log(filename: Path) -> SwarmLearningResults:
    contents = load_log_lines(filename)
    results = SwarmLearningResults()
    results.training_auc_roc = parse_training_AUC_ROCs(contents)
    results.validation_auc_roc = parse_validation_AUC_ROCs(contents)
    results.validation_auc_roc_global_model = parse_validation_AUC_ROCs_aggregated_models(contents)
    num_train, num_val = get_num_train_val_images(contents)
    results.num_train_images = num_train
    results.num_val_images = num_val
    print(f'{site_name: <4}: {num_train: >5} training images, {num_val: >5} validation images, ' +
          f'validation AUROC (last global model): {results.validation_auc_roc_global_model[95]:.4f}, ' +
          f'training AUROC (last local model): {results.training_auc_roc[99]:.4f}, ' +
          f'validation AUROC (last local model): {results.validation_auc_roc[99]:.4f}'
          )
    return results


def parse_local_training_log(filename: Path) -> LocalTrainingResults:
    contents = load_log_lines(filename)
    results = LocalTrainingResults()
    results.training_auc_roc = parse_training_AUC_ROCs(contents)
    results.validation_auc_roc = parse_validation_AUC_ROCs(contents)
    num_train, num_val = get_num_train_val_images(contents)
    results.num_train_images = num_train
    results.num_val_images = num_val
    print(f'{site_name: <4}: {num_train: >5} training images, {num_val: >5} validation images, ' +
          '                                              ' +
          f'final local training AUROC:        {results.training_auc_roc[99]:.4f}, ' +
          f'final local validation AUROC:        {results.validation_auc_roc[99]:.4f}'
          )
    return results


if __name__ == '__main__':
    # this script expects a folder structure SITE_NAME/nohup.out with optional SITE_NAME/local_training_console_output.txt
    swarm_data: Dict[str, SwarmLearningResults] = {}
    local_data: Dict[str, LocalTrainingResults] = {}

    for site_name in color_for_site.keys():
        log_filename = Path(site_name) / 'nohup.out'
        swarm_data[site_name] = SwarmLearningResults()
        if os.path.exists(log_filename):
            swarm_data[site_name] = parse_swarm_training_log(log_filename)
        else:
            print(f'No swarm training log file {log_filename} found for site {site_name}')

        local_training_log_filename = Path(site_name) / 'local_training_console_output.txt'
        local_data[site_name] = LocalTrainingResults()
        if os.path.exists(local_training_log_filename):
            local_data[site_name] = parse_local_training_log(local_training_log_filename)
        else:
            print(f'No local training log file {local_training_log_filename} found for site {site_name}')

    plot_per_site(swarm_data, local_data)
    plot_overviews(swarm_data, local_data)
