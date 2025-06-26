import numpy as np
import matplotlib
from sklearn.metrics import roc_curve, auc, confusion_matrix


def auc_bootstrapping(y_true, y_score, bootstrapping=1000, drop_intermediate=False):
    """Perform bootstrapping to compute variability of ROC curve and AUC.

    Args:
        y_true (np.ndarray): True binary labels.
        y_score (np.ndarray): Predicted scores or probabilities.
        bootstrapping (int): Number of bootstrap samples.
        drop_intermediate (bool): Whether to drop some thresholds for faster computation.

    Returns:
        Tuple[list, list, list, np.ndarray]:
            - List of interpolated TPRs,
            - List of AUCs,
            - List of optimal thresholds,
            - Mean FPR values used for interpolation.
    """
    tprs, aucs, thrs = [], [], []
    mean_fpr = np.linspace(0, 1, 100)
    rng = np.random.default_rng(seed)

    # Generate bootstrap samples with replacement
    rand_idxs = rng.integers(0, len(y_true), size=(bootstrapping, len(y_true)))

    for rand_idx in rand_idxs:
        y_true_set = y_true[rand_idx]
        y_score_set = y_score[rand_idx]

        # Compute ROC for the sample
        fpr, tpr, thresholds = roc_curve(y_true_set, y_score_set, drop_intermediate=drop_intermediate)

        # Interpolate TPRs to a common FPR scale
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tprs.append(tpr_interp)
        aucs.append(auc(fpr, tpr))

        # Identify optimal threshold (Youden's J statistic)
        optimal_idx = np.argmax(tpr - fpr)
        thrs.append(thresholds[optimal_idx])

    return tprs, aucs, thrs, mean_fpr


def plot_roc_curve(y_true, y_score, axis, bootstrapping=1000, drop_intermediate=False, fontdict={},
                   name='ROC', color='b', show_wp=True):
    """Plot ROC curve with bootstrapped AUC and shaded confidence interval.

    Args:
        y_true (np.ndarray): True binary labels.
        y_score (np.ndarray): Predicted probabilities or scores.
        axis (matplotlib.axes.Axes): Axis to plot on.
        bootstrapping (int): Number of bootstrap samples.
        drop_intermediate (bool): Drop thresholds for faster computation.
        fontdict (dict): Font styling dictionary.
        name (str): Curve label.
        color (str): Line color.
        show_wp (bool): Show working point (optimal threshold marker).

    Returns:
        Tuple[np.ndarray, np.ndarray, float, np.ndarray, int]:
            FPR, TPR, AUC, thresholds, and index of optimal threshold.
    """
    # Bootstrapping
    tprs, aucs, thrs, mean_fpr = auc_bootstrapping(y_true, y_score, bootstrapping, drop_intermediate)

    mean_tpr = np.nanmean(tprs, axis=0)
    mean_tpr[-1] = 1.0  # Ensure proper endpoint
    std_tpr = np.nanstd(tprs, axis=0, ddof=1)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    mean_auc = np.nanmean(aucs)
    std_auc = np.nanstd(aucs, ddof=1)

    # Compute actual ROC
    fprs, tprs_, thrs_ = roc_curve(y_true, y_score, drop_intermediate=drop_intermediate)
    auc_val = auc(fprs, tprs_)
    opt_idx = np.argmax(tprs_ - fprs)
    opt_tpr = tprs_[opt_idx]
    opt_fpr = fprs[opt_idx]

    # Plot ROC
    axis.plot(fprs, tprs_, color=color, label=rf"{name} (AUC={auc_val:.2f}$\pm${std_auc:.2f})", lw=2, alpha=.8)
    axis.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    if show_wp:
        axis.hlines(y=opt_tpr, xmin=0.0, xmax=opt_fpr, color=color, linestyle='--')
        axis.vlines(x=opt_fpr, ymin=0.0, ymax=opt_tpr, color=color, linestyle='--')
        axis.plot(opt_fpr, opt_tpr, color=color, marker='o')

    axis.plot([0, 1], [0, 1], linestyle='--', color='k')
    axis.set_xlim([0.0, 1.0])
    axis.set_ylim([0.0, 1.0])

    axis.legend(loc='lower right')
    axis.set_xlabel('1 - Specificity', fontdict=fontdict)
    axis.set_ylabel('Sensitivity', fontdict=fontdict)

    # Aesthetic tweaks
    axis.grid(color='#dddddd')
    axis.set_axisbelow(True)
    axis.tick_params(colors='#dddddd', which='both')
    for xtick in axis.get_xticklabels():
        xtick.set_color('k')
    for ytick in axis.get_yticklabels():
        ytick.set_color('k')
    for child in axis.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color('#dddddd')

    return fprs, tprs_, auc_val, thrs_, opt_idx


def cm2acc(cm):
    """Calculate accuracy from a 2x2 confusion matrix."""
    tn, fp, fn, tp = cm.ravel()
    return (tn + tp) / (tn + tp + fn + fp)


def safe_div(x, y):
    """Safely divide x by y, return NaN if y is zero."""
    return float('nan') if y == 0 else x / y


def specificity_at_fixed_sensitivity(y_true, y_scores, tpr, thresholds, sensitivity_target=0.90):
    """Calculate specificity at a given sensitivity level.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_scores (np.ndarray): Predicted scores.
        tpr (np.ndarray): True positive rates from ROC.
        thresholds (np.ndarray): Thresholds from ROC.
        sensitivity_target (float): Desired sensitivity level.

    Returns:
        float: Specificity at the closest sensitivity.
    """
    idx = np.argmin(np.abs(tpr - sensitivity_target))
    chosen_threshold = thresholds[idx]
    y_pred = (y_scores >= chosen_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def sensitivity_at_fixed_specificity(y_true, y_scores, fpr, thresholds, specificity_target=0.90):
    """Calculate sensitivity at a given specificity level.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_scores (np.ndarray): Predicted scores.
        fpr (np.ndarray): False positive rates from ROC.
        thresholds (np.ndarray): Thresholds from ROC.
        specificity_target (float): Desired specificity level.

    Returns:
        float: Sensitivity at the closest specificity.
    """
    specificity = 1 - fpr
    idx = np.argmin(np.abs(specificity - specificity_target))
    chosen_threshold = thresholds[idx]
    y_pred = (y_scores >= chosen_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn)


def cm2x(cm, average='macro', pos_label=1):
    """Compute PPV, NPV, Sensitivity (TPR), and Specificity (TNR) from confusion matrix.

    Args:
        cm (np.ndarray): Confusion matrix.
        average (str): 'binary', 'micro', 'macro', or 'weighted'.
        pos_label (int): Class considered positive in binary mode.

    Returns:
        dict: Dictionary with PPV, NPV, TPR, and TNR.
    """
    num_classes = cm.shape[0]
    metrics_per_class = {}

    if average == 'micro':
        TP = np.sum([cm[i, i] for i in range(num_classes)])
        FP = np.sum([cm[:, i].sum() - cm[i, i] for i in range(num_classes)])
        FN = np.sum([cm[i, :].sum() - cm[i, i] for i in range(num_classes)])
        TN = np.sum([cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i]) for i in range(num_classes)])

        return {
            "PPV": safe_div(TP, TP + FP),
            "NPV": safe_div(TN, TN + FN),
            "TPR": safe_div(TP, TP + FN),
            "TNR": safe_div(TN, TN + FP),
        }

    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        metrics_per_class[i] = {
            "PPV": safe_div(TP, TP + FP),
            "NPV": safe_div(TN, TN + FN),
            "TPR": safe_div(TP, TP + FN),
            "TNR": safe_div(TN, TN + FP),
        }

    if average == 'binary':
        if pos_label not in metrics_per_class:
            raise ValueError(f"pos_label={pos_label} not in class labels: {list(metrics_per_class.keys())}")
        return metrics_per_class[pos_label]

    ppv_vals = [metrics_per_class[i]["PPV"] for i in range(num_classes)]
    npv_vals = [metrics_per_class[i]["NPV"] for i in range(num_classes)]
    tpr_vals = [metrics_per_class[i]["TPR"] for i in range(num_classes)]
    tnr_vals = [metrics_per_class[i]["TNR"] for i in range(num_classes)]

    if average == 'macro':
        return {
            "PPV": np.mean(ppv_vals),
            "NPV": np.mean(npv_vals),
            "TPR": np.mean(tpr_vals),
            "TNR": np.mean(tnr_vals),
        }

    if average == 'weighted':
        support = cm.sum(axis=1)
        weights = support / support.sum()
        return {
            "PPV": np.sum(weights * np.array(ppv_vals)),
            "NPV": np.sum(weights * np.array(npv_vals)),
            "TPR": np.sum(weights * np.array(tpr_vals)),
            "TNR": np.sum(weights * np.array(tnr_vals)),
        }

    raise ValueError("Invalid average method. Choose from {'binary', 'micro', 'macro', 'weighted'}.")
