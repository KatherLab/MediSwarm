#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import roc_auc_score
from matplotlib.patches import Patch
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from warnings import warn
from typing import List, Dict


def add_file_or_warn(file_path, file_list):
    if file_path.exists():
        file_list.append(file_path)
    else:
        warn(f'{file_path} not found')


def get_setting_files(root_dir) -> Dict[str, List[Path]]:
    print("Gathering relevant files...")

    local_dir = root_dir / "local"
    swarm_dir = root_dir / "swarm"

    # Check that directories exist
    if not local_dir.exists():
        raise FileNotFoundError(f"Local directory not found: {local_dir}")
    if not swarm_dir.exists():
        raise FileNotFoundError(f"Swarm directory not found: {swarm_dir}")

    swarm_agg_train_files = []
    swarm_agg_val_files = []
    swarm_site_train_files = []
    swarm_site_val_files = []
    local_train_files = []
    local_val_files = []

    # Gather local files
    for site_dir in [d for d in local_dir.iterdir() if d.is_dir()]:
        add_file_or_warn(site_dir / "site_model_gt_and_classprob_train.csv", local_train_files)
        add_file_or_warn(site_dir / "site_model_gt_and_classprob_validation.csv", local_val_files)

    # Gather swarm files
    for site_dir in [d for d in swarm_dir.iterdir() if d.is_dir()]:
        add_file_or_warn(site_dir / "site_model_gt_and_classprob_train.csv", swarm_site_train_files)
        add_file_or_warn(site_dir / "site_model_gt_and_classprob_validation.csv", swarm_site_val_files)
        add_file_or_warn(site_dir / "aggregated_model_gt_and_classprob_train.csv", swarm_agg_train_files)
        add_file_or_warn(site_dir / "aggregated_model_gt_and_classprob_validation.csv", swarm_agg_val_files)

    setting_files = { "Swarm (agg, train)": swarm_agg_train_files,
                      "Swarm (agg, val)": swarm_agg_val_files,
                      "Swarm (site, train)": swarm_site_train_files,
                      "Swarm (site, val)": swarm_site_val_files,
                      "Local (train)": local_train_files,
                      "Local (val)": local_val_files
                     }
    return setting_files


# Helper function to verify labels don't change across epochs
def _verify_constant_labels_across_epochs(df: pd.DataFrame, name: str) -> None:
    for site in df.site.unique():
        site_df = df[df.site == site]
        epochs = site_df.epoch.unique()

        # Get label distribution for first epoch
        first_epoch = epochs[0]
        ref_labels = site_df[site_df.epoch == first_epoch].label.value_counts().sort_index()

        # Check all other epochs have same distribution
        for epoch in epochs[1:]:
            epoch_labels = site_df[site_df.epoch == epoch].label.value_counts().sort_index()
            if not ref_labels.equals(epoch_labels):
                warn(f"{name} site {site}: Label distribution changed between epoch {first_epoch} and {epoch}")


def load_data(setting_files: Dict[str, List[Path]]) -> Dict[str, pd.DataFrame]:
    # Store merged dataframes for label distribution
    merged_dfs = {}

    for setting, files in setting_files.items():
        print("Analyzing setting: " + setting)

        dfs = []
        for file in files:
            df = pd.read_csv(file, names=["epoch", "label", "score_0", "score_1", "score_2"])
            df.loc[:, "site"] = file.parts[1]
            dfs.append(df)

        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_dfs[setting] = merged_df

    return merged_dfs


def compute_aurocs(merged_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    print("Computing AUROCs...")

    auroc_dfs = []
    for setting in merged_dfs.keys():
        print("Analyzing setting: " + setting)
        merged_df = merged_dfs[setting]

        for site in merged_df.site.unique():
            print("Site: " + site)
            for epoch in tqdm(merged_df.epoch.unique()):
                filter = (merged_df.epoch == epoch) & (merged_df.site == site)

                if set(merged_df[filter].label.unique()) == {0, 1, 2}:
                    macro_auroc = roc_auc_score(merged_df[filter].label,
                                                merged_df[filter][["score_0", "score_1", "score_2"]],
                                                multi_class='ovo')
                else:
                    macro_auroc = np.nan

                tumor_scores_1 = merged_df[filter][["score_1", "score_2"]].max(axis=1)
                tumor_scores_2 = merged_df[filter].score_0
                tumor_scores = tumor_scores_1 - tumor_scores_2  # score for tumor yes/no
                tumor_labels = (merged_df[filter].label > 0).astype(int)
                if len(tumor_labels.unique()) == 2:
                    tumor_auroc = roc_auc_score(tumor_labels, tumor_scores)
                else:
                    tumor_auroc = np.nan

                tumor_filter = filter & (merged_df.label > 0)
                tumor_malignancy_scores = merged_df[tumor_filter].score_2 / merged_df[tumor_filter][["score_1", "score_2"]].sum(axis=1)
                tumor_malignancy_labels = merged_df[tumor_filter].label > 1
                if len(tumor_malignancy_labels.unique()) == 2:
                    tumor_malignancy_auroc = roc_auc_score(tumor_malignancy_labels, tumor_malignancy_scores)
                else:
                    tumor_malignancy_auroc = np.nan

                auroc_dfs.append(pd.DataFrame({"epoch": epoch,
                                               "site": site,
                                               "setting": setting,
                                               "AUROC": [macro_auroc, tumor_auroc, tumor_malignancy_auroc],
                                               "auroc_type": ["macro", "tumor (0v1/2)", "malignancy (1v2)"]}))

    auroc_df = pd.concat(auroc_dfs, ignore_index=True)
    return auroc_df


def verify_constant_labels_across_epochs(merged_dfs: Dict[str, pd.DataFrame]) -> None:
    # Verify and prepare label distributions
    print("Verifying label distributions...")

    # Verify each dataframe has constant labels across epochs
    for setting_name, df in merged_dfs.items():
        _verify_constant_labels_across_epochs(df, setting_name)

    print("Verified: Label distributions are constant across epochs for all settings")


def verify_same_label_distributions_at_epoch_zero(merged_dfs: Dict[str, pd.DataFrame]) -> None:
    # For train: verify swarm agg and swarm site have same label distribution at epoch 0
    success = True
    if "Swarm (agg, train)" in merged_dfs.keys():
        for site in merged_dfs["Swarm (agg, train)"].site.unique():
            agg_labels = merged_dfs["Swarm (agg, train)"][(merged_dfs["Swarm (agg, train)"].site == site) &
                                                          (merged_dfs["Swarm (agg, train)"].epoch == 0)].label.value_counts().sort_index()
            site_labels = merged_dfs["Swarm (site, train)"][(merged_dfs["Swarm (site, train)"].site == site) &
                                                            (merged_dfs["Swarm (site, train)"].epoch == 0)].label.value_counts().sort_index()
            if not agg_labels.equals(site_labels):
                success = False
                warn(f"Train label mismatch at epoch 0 for site {site}: agg={agg_labels.to_dict()}, site={site_labels.to_dict()}")

    # For val: verify swarm agg and swarm site have same label distribution at epoch 0
    if "Swarm (agg, val)" in merged_dfs.keys():
        for site in merged_dfs["Swarm (agg, val)"].site.unique():
            agg_labels = merged_dfs["Swarm (agg, val)"][(merged_dfs["Swarm (agg, val)"].site == site) &
                                                        (merged_dfs["Swarm (agg, val)"].epoch == 0)].label.value_counts().sort_index()
            site_labels = merged_dfs["Swarm (site, val)"][(merged_dfs["Swarm (site, val)"].site == site) &
                                                          (merged_dfs["Swarm (site, val)"].epoch == 0)].label.value_counts().sort_index()
            if not agg_labels.equals(site_labels):
                success = False
                warn(f"Val label mismatch at epoch 0 for site {site}: agg={agg_labels.to_dict()}, site={site_labels.to_dict()}")

    if success:
        print("Verified: Swarm agg and site have same label distributions at epoch 0")


def compute_label_distributions(merged_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Train distributions - use only epoch 0 since we verified labels are constant across epochs
    if "Swarm (agg, train)" in merged_dfs:
        swarm_train_dist = merged_dfs["Swarm (agg, train)"][merged_dfs["Swarm (agg, train)"].epoch == 0][['site', 'label']].copy()
    else:
        swarm_train_dist = pd.DataFrame()
    swarm_train_dist['source'] = 'Swarm'
    swarm_train_dist['split'] = 'Train'

    if "Local (train)" in merged_dfs:
        local_train_dist = merged_dfs["Local (train)"][merged_dfs["Local (train)"].epoch == 0][['site', 'label']].copy()
    else:
        local_train_dist = pd.DataFrame()

    local_train_dist['source'] = 'Local'
    local_train_dist['split'] = 'Train'

    # Val distributions - use only epoch 0
    if "Swarm (agg, val)" in merged_dfs:
        swarm_val_dist = merged_dfs["Swarm (agg, val)"][merged_dfs["Swarm (agg, val)"].epoch == 0][['site', 'label']].copy()
    else:
        swarm_val_dist = pd.DataFrame()
    swarm_val_dist['source'] = 'Swarm'
    swarm_val_dist['split'] = 'Val'

    if "Local (val)" in merged_dfs:
        local_val_dist = merged_dfs["Local (val)"][merged_dfs["Local (val)"].epoch == 0][['site', 'label']].copy()
    else:
        local_val_dist = pd.DataFrame()

    local_val_dist['source'] = 'Local'
    local_val_dist['split'] = 'Val'

    label_dist_df = pd.concat([swarm_train_dist, local_train_dist, swarm_val_dist, local_val_dist], ignore_index=True)
    return label_dist_df


def plot_aurocs(auroc_df: pd.DataFrame, axes):
    n_sites = len(auroc_df.site.unique())
    sites = sorted(auroc_df.site.unique())
    auroc_types = sorted(auroc_df.auroc_type.unique())

    # Plot AUROC metrics (rows 0-2)
    for row_idx, auroc_type in enumerate(auroc_types):
        for col_idx, site in enumerate(sites):
            ax = axes[row_idx, col_idx]

            # Filter and plot
            plot_data = auroc_df[(auroc_df.site == site) & (auroc_df.auroc_type == auroc_type)]
            sns.lineplot(data=plot_data, x='epoch', y='AUROC', hue='setting',
                         style='setting', ax=ax, legend=(row_idx == 0 and col_idx == n_sites - 1))

            ax.set_ylim([0, 1.01])
            ax.set_xlim([auroc_df.epoch.min(), auroc_df.epoch.max() + 1])
            ax.set_ylabel('AUROC' if col_idx == 0 else '')
            ax.set_xlabel('Epoch')

            # Row labels
            if col_idx == 0:
                ax.text(-0.15, 0.5, auroc_type, transform=ax.transAxes,
                        fontsize=14, fontweight='bold', rotation=90, va='center', ha='right')

            # Column labels
            if row_idx == 0:
                ax.set_title(site, fontsize=14, fontweight='bold', pad=10)

            # Legend only on top-right
            if row_idx == 0 and col_idx == n_sites - 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)


def verify_same_label_distribution_swarm_local(label_dist_df: pd.DataFrame) -> None:
    success = True
    print("Verifying Swarm and Local have identical label distributions...")
    for site in sorted(label_dist_df.site.unique()):
        for split in ['Train', 'Val']:
            swarm_counts = label_dist_df[(label_dist_df.site == site) &
                                         (label_dist_df.split == split) &
                                         (label_dist_df.source == 'Swarm')].label.value_counts().sort_index()
            local_counts = label_dist_df[(label_dist_df.site == site) &
                                         (label_dist_df.split == split) &
                                         (label_dist_df.source == 'Local')].label.value_counts().sort_index()
            if not swarm_counts.equals(local_counts):
                success = False
                warn(f"Label distribution mismatch for {site} {split}: Swarm={swarm_counts.to_dict()}, Local={local_counts.to_dict()}")
    if success:
        print("Verified: Swarm and Local have identical label distributions")


def plot_label_distributions(label_dist_df: pd.DataFrame, axes, logscale_hist: bool) -> None:
    # Plot combined label distributions (row 3) - just use Swarm data since they're identical
    # Compute max count for shared y-axis

    label_counts_df = label_dist_df[label_dist_df.source == 'Swarm'].groupby(['site', 'split', 'label']).size()
    ymax = label_counts_df.max()

    for col_idx, site in enumerate(sorted(label_dist_df.site.unique())):
        ax = axes[3, col_idx]

        # Filter data - use Swarm only since we verified they're identical
        plot_data = label_dist_df[(label_dist_df.site == site) & (label_dist_df.source == 'Swarm')]
        plot_data_train = plot_data[plot_data.split == 'Train']
        plot_data_val = plot_data[plot_data.split == 'Val']

        if plot_data.empty:
            return

        # Plot with split as hue (Train vs Val)
        histogram = sns.histplot(data=plot_data, x='label', hue='split', multiple='dodge',
                                 discrete=True, stat='count', shrink=0.8, ax=ax,
                                 hue_order=['Train', 'Val'],
                                 palette=['#1f77b4', '#ff7f0e'],
                                 legend=False, alpha=1)

        ax.set_ylabel('Count' if col_idx == 0 else '')
        ax.set_xlabel('Label')
        ax.set_xticks([0, 1, 2])

        # Add numbers in histogram
        for spot in histogram.patches:
            histogram.text(spot.get_x(), spot.get_height()+3, f'{spot.get_height()}')

        if logscale_hist:
            ax.set_yscale('log')
            ax.set_ylim([0.5, ymax * 2])  # Start at 0.5 for log scale (can't start at 0)
            ax.grid(True, which='major', alpha=1.0, linewidth=1.0, axis='y')
            ax.grid(True, which='minor', alpha=1.0, linewidth=0.8, axis='y')
            ax.minorticks_on()

        else:
            ax.set_ylim([0, ymax * 1.1])

        # Add total sample count + split ratio in upper right
        ax.text(1.0, 1.0, f'\n  n = {len(plot_data)}  \n  split: {len(plot_data_train)/len(plot_data):.0g}/{len(plot_data_val)/len(plot_data):.0g}  \n',
               transform=ax.transAxes, fontsize=11,
               va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Row label
        if col_idx == 0:
            ax.text(-0.15, 0.5, 'Label Distribution',
                   transform=ax.transAxes, fontsize=14, fontweight='bold',
                   rotation=90, va='center', ha='right')


def plot(auroc_df: pd.DataFrame, label_dist_df: pd.DataFrame, logscale_hist: bool) -> None:
    print("Plotting...")

    sns.set_style("whitegrid", rc={"axes.spines.left": False, "axes.spines.right": False, "axes.spines.top": False})

    # Create figure with proper subplots: 4 rows x n_sites columns (3 AUROC + 1 label dist)
    n_sites = len(auroc_df.site.unique())
    fig, axes = plt.subplots(4, n_sites, figsize=(6 * n_sites, 15), dpi=150)
    if n_sites == 1:
        axes = axes.reshape(-1, 1)

    plot_aurocs(auroc_df, axes)

    verify_same_label_distribution_swarm_local(label_dist_df)

    plot_label_distributions(label_dist_df, axes, logscale_hist)

    # Add legend for bar plots on bottom-right
    legend_handles = [
        Patch(facecolor='#1f77b4', label='Train'),
        Patch(facecolor='#ff7f0e', label='Val')
    ]
    axes[3, -1].legend(handles=legend_handles, bbox_to_anchor=(1.05, 1),
                      loc='upper left', fontsize=12, frameon=True)

    plt.tight_layout()
    plt.savefig("evaluation.png", bbox_inches='tight', dpi=150)
    plt.close()


def analyze(root_dir, logscale_hist=False):
    setting_files = get_setting_files(root_dir)

    for setting, files in setting_files.items():
        print(f'Identified {len(files)} {setting} files.')

    merged_dfs = load_data(setting_files)
    auroc_df = compute_aurocs(merged_dfs)

    verify_constant_labels_across_epochs(merged_dfs)
    verify_same_label_distributions_at_epoch_zero(merged_dfs)

    label_dist_df = compute_label_distributions(merged_dfs)

    plot(auroc_df, label_dist_df, logscale_hist)

    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=Path, nargs='?',
                        default=Path("."),
                        help='Top-level directory containing run results (csv files)')
    parser.add_argument('--logscale_hist', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Plot sample count histograms in log scale')

    args = parser.parse_args()

    analyze(args.data_dir, args.logscale_hist)
