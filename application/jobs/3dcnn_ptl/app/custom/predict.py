#!/usr/bin/env python3

import torch
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data.datasets import DUKE_Dataset3D, DUKE_Dataset3D_external, DUKE_Dataset3D_collab
from data.datamodules import DataModule
from utils.roc_curve import plot_roc_curve, cm2acc, cm2x
from models import ResNet, VisionTransformer, EfficientNet, DenseNet121, UNet3D

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict(model_dir, test_data_dir, model_name, last_flag, prediction_flag, cohort_flag='aachen'):
    """
    Predicts and evaluates the model on the test dataset.

    Args:
        model_dir (str): Directory containing the model.
        test_data_dir (str): Directory containing the test data.
        model_name (str): Name of the model to use.
        last_flag (bool): Whether to use the last checkpoint or the best checkpoint.
        prediction_flag (str): Flag to indicate which dataset to use ('ext', 'internal', 'collab').
        cohort_flag (str, optional): Cohort flag for the output directory name. Defaults to 'aachen'.
    """
    try:
        path_run = Path(model_dir)
        path_out = Path(path_run, f"{prediction_flag}_{cohort_flag}")
        logger.info(f"Output path: {path_out.absolute()}")
        path_out.mkdir(parents=True, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fontdict = {'fontsize': 10, 'fontweight': 'bold'}

        # Load Data
        if prediction_flag == 'ext':
            ds = DUKE_Dataset3D_external(flip=False, path_root=test_data_dir)
        elif prediction_flag == 'internal':
            ds = DUKE_Dataset3D(flip=False, path_root=test_data_dir)
        elif prediction_flag == 'collab':
            ds = DUKE_Dataset3D_collab(flip=False, path_root=test_data_dir)
        else:
            raise ValueError("Invalid prediction_flag specified")

        logger.info(f"Number of test samples: {len(ds)}")
        dm = DataModule(ds_test=ds, batch_size=1)

        # Initialize Model
        model = initialize_model(model_name, path_run, last_flag)
        model.to(device)
        model.eval()

        results = {'uid': [], 'GT': [], 'NN': [], 'NN_pred': []}
        threshold = 0.5

        for batch in tqdm(dm.test_dataloader()):
            source, target = batch['source'], batch['target']

            # Run Model
            pred = model(source.to(device)).cpu()
            pred_proba = torch.sigmoid(pred).squeeze()
            pred_binary = (pred_proba > threshold).long()

            results['GT'].extend(target.tolist())
            results['NN'].extend(pred_binary.tolist() if isinstance(pred_binary.tolist(), list) else [pred_binary.tolist()])
            results['NN_pred'].extend(pred_proba.tolist() if isinstance(pred_proba.tolist(), list) else [pred_proba.tolist()])
            results['uid'].extend(batch['uid'])

        df = pd.DataFrame(results)
        save_results(df, path_out, last_flag)
        evaluate_results(df, path_out, last_flag, fontdict)

        del model
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Error in predict function: {e}")
        raise

def initialize_model(model_name, path_run, last_flag):
    """
    Initializes the model based on the provided model name.

    Args:
        model_name (str): Name of the model to initialize.
        path_run (Path): Path to the model directory.
        last_flag (bool): Whether to use the last checkpoint or the best checkpoint.

    Returns:
        nn.Module: The initialized model.
    """
    try:
        layers = None
        if model_name in ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']:
            layers = {'ResNet18': [2, 2, 2, 2], 'ResNet34': [3, 4, 6, 3], 'ResNet50': [3, 4, 6, 3], 'ResNet101': [3, 4, 23, 3], 'ResNet152': [3, 8, 36, 3]}[model_name]
            if last_flag:
                return ResNet.load_last_checkpoint(path_run, version=0, layers=layers)
            return ResNet.load_best_checkpoint(path_run, version=0, layers=layers)

        if model_name in ['efficientnet_l1', 'efficientnet_l2', 'efficientnet_b4', 'efficientnet_b7']:
            if last_flag:
                return EfficientNet.load_last_checkpoint(path_run, version=0, model_name=model_name)
            return EfficientNet.load_best_checkpoint(path_run, version=0, model_name=model_name)

        if model_name.startswith('EfficientNet3D'):
            blocks_args_str = {
                'EfficientNet3Db0': ["r1_k3_s11_e1_i32_o16_se0.25", "r2_k3_s22_e6_i16_o24_se0.25", "r2_k5_s22_e6_i24_o40_se0.25", "r3_k3_s22_e6_i40_o80_se0.25", "r3_k5_s11_e6_i80_o112_se0.25", "r4_k5_s22_e6_i112_o192_se0.25", "r1_k3_s11_e6_i192_o320_se0.25"],
                'EfficientNet3Db4': ["r1_k3_s11_e1_i48_o24_se0.25", "r3_k3_s22_e6_i24_o32_se0.25", "r3_k5_s22_e6_i32_o56_se0.25", "r4_k3_s22_e6_i56_o112_se0.25", "r4_k5_s11_e6_i112_o160_se0.25", "r5_k5_s22_e6_i160_o272_se0.25", "r2_k3_s11_e6_i272_o448_se0.25"],
                'EfficientNet3Db7': ["r1_k3_s11_e1_i32_o32_se0.25", "r4_k3_s22_e6_i32_o48_se0.25", "r4_k5_s22_e6_i48_o80_se0.25", "r4_k3_s22_e6_i80_o160_se0.25", "r6_k5_s11_e6_i160_o256_se0.25", "r6_k5_s22_e6_i256_o384_se0.25", "r3_k3_s11_e6_i384_o640_se0.25"]
            }[model_name]
            if last_flag:
                return EfficientNet3D.load_last_checkpoint(path_run, version=0, blocks_args_str=blocks_args_str)
            return EfficientNet3D.load_best_checkpoint(path_run, version=0, blocks_args_str=blocks_args_str)

        if model_name == 'DenseNet121':
            if last_flag:
                return DenseNet121.load_last_checkpoint(path_run, version=0)
            return DenseNet121.load_best_checkpoint(path_run, version=0)

        if model_name == 'UNet3D':
            if last_flag:
                return UNet3D.load_last_checkpoint(path_run, version=0)
            return UNet3D.load_best_checkpoint(path_run, version=0)

        raise ValueError("Invalid network model specified")
    except Exception as e:
        logger.error(f"Error in initialize_model function: {e}")
        raise

def save_results(df, path_out, last_flag):
    """
    Saves the prediction results to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame containing the results.
        path_out (Path): Path to the output directory.
        last_flag (bool): Whether to save results for the last checkpoint or the best checkpoint.
    """
    try:
        file_name = 'results_last.csv' if last_flag else 'results.csv'
        df.to_csv(path_out / file_name, index=False)
    except Exception as e:
        logger.error(f"Error in save_results function: {e}")
        raise

def evaluate_results(df, path_out, last_flag, fontdict):
    """
    Evaluates the prediction results and saves metrics and plots.

    Args:
        df (pd.DataFrame): DataFrame containing the results.
        path_out (Path): Path to the output directory.
        last_flag (bool): Whether to save results for the last checkpoint or the best checkpoint.
        fontdict (dict): Font dictionary for plot titles and labels.
    """
    try:
        f1 = f1_score(df['GT'], df['NN'])
        logger.info(f"F1 Score: {f1:.2f}")

        cm = confusion_matrix(df['GT'], df['NN'])
        tn, fp, fn, tp = cm.ravel()
        n = len(df)
        logger.info(f"Confusion Matrix: TN {tn} ({tn / n * 100:.2f}%), FP {fp} ({fp / n * 100:.2f}%), FN {fn} ({fn / n * 100:.2f}%), TP {tp} ({tp / n * 100:.2f}%)")

        fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        y_pred_lab = np.asarray(df['NN_pred'])
        y_true_lab = np.asarray(df['GT'])
        tprs, fprs, auc_val, thrs, opt_idx, cm = plot_roc_curve(y_true_lab, y_pred_lab, axis, fontdict=fontdict)
        fig.tight_layout()
        file_name = 'roc_last.png' if last_flag else 'roc.png'
        fig.savefig(path_out / file_name, dpi=300)

        precision, recall, _ = precision_recall_curve(y_true_lab, y_pred_lab)
        ap = average_precision_score(y_true_lab, y_pred_lab)

        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)

        acc = cm2acc(cm)
        _, _, sens, spec = cm2x(cm)
        df_cm = pd.DataFrame(data=cm, columns=['False', 'True'], index=['False', 'True'])
        fig, axis = plt.subplots(1, 1, figsize=(4, 4))
        sns.heatmap(df_cm, ax=axis, cbar=False, fmt='d', annot=True)
        axis.set_title(f'Confusion Matrix ACC={acc:.2f}', fontdict=fontdict)
        axis.set_xlabel('Prediction', fontdict=fontdict)
        axis.set_ylabel('True', fontdict=fontdict)
        fig.tight_layout()
        file_name = 'confusion_matrix_last.png' if last_flag else 'confusion_matrix.png'
        fig.savefig(path_out / file_name, dpi=300)

        logger.info(f"Malign Objects: {np.sum(y_true_lab)}")
        logger.info(f"Confusion Matrix {cm}")
        logger.info(f"Sensitivity {sens:.2f}")
        logger.info(f"Specificity {spec:.2f}")

        with open(path_out / 'metrics.txt', 'w') as f:
            f.write(f"AUC: {auc_val:.2f}\n")
            f.write(f"F1 Score: {f1:.2f}\n")
            f.write(f"Sensitivity: {sens:.2f}\n")
            f.write(f"Specificity: {spec:.2f}\n")
            f.write(f"PPV: {ppv:.2f}\n")
            f.write(f"NPV: {npv:.2f}\n")
            f.write(f"ACC: {acc:.2f}\n")
            f.write(f"AP: {ap:.2f}\n")

        print(f"AUC: {auc_val:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"Sensitivity: {sens:.2f}")
        print(f"Specificity: {spec:.2f}")
        print(f"PPV: {ppv:.2f}")
        print(f"NPV: {npv:.2f}")
        print(f"ACC: {acc:.2f}")
        print(f"AP: {ap:.2f}")
    except Exception as e:
        logger.error(f"Error in evaluate_results function: {e}")
        raise

if __name__ == "__main__":
    wouter_data_path = "/mnt/sda1/swarm-learning/wouter_data/preprocessed_re/"
    athens_data_path = "/mnt/sda1/swarm-learning/athens_data/preprocessed_athens/"
    predict(
        model_dir=Path('/mnt/sda1/odelia_paper_trained_results/2023_07_04_180000_DUKE_ext_ResNet101_swarm_learning'),
        test_data_dir=athens_data_path,
        model_name='ResNet101',
        last_flag=False,
        prediction_flag='collab',
        cohort_flag='athens'
    )
