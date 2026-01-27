#!/usr/bin/env python3

import sys
import csv
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score

@dataclass
class GT_Classprob:
    gt: int
    classprob: List[float]

    def __init__(self, data: List[str]):
        self.gt = int(data[0])
        self.classprob = [float(d) for d in data[1:]]


@dataclass
class ConvergenceResults:
    gt_classprob_for_epoch: Dict[int, List[GT_Classprob]]

    def __init__(self, filename: str):
        self.gt_classprob_for_epoch = {}

        with open(filename) as csvfile:
            datareader = csv.reader(csvfile)
            for row in datareader:
                epoch = int(row[0])
                if epoch not in self.gt_classprob_for_epoch:
                    self.gt_classprob_for_epoch[epoch] = []
                self.gt_classprob_for_epoch[epoch].append(GT_Classprob(row[1:]))

    def print_class_statistics(self, epoch: int = 0) -> None:
        gt_values = [gtcb.gt for gtcb in self.gt_classprob_for_epoch[epoch]]
        for value in sorted(list(set(gt_values))):
            print(f'{value}: {gt_values.count(value)}')


def compute_auc_roc(data: List[GT_Classprob]) -> float:
    y_true = [gtcb.gt for gtcb in data]
    y_score = [gtcb.classprob for gtcb in data]
    roc_auc_ovr_micro = roc_auc_score(y_true, y_score, average='micro', multi_class='ovr')
    roc_auc_ovr_macro = roc_auc_score(y_true, y_score, average='macro', multi_class='ovr')
    # ovo micro not implemented
    roc_auc_ovo_macro = roc_auc_score(y_true, y_score, average='macro', multi_class='ovo')
    print(f"ovr_micro: {roc_auc_ovr_micro:0.4f}, ovr_macro: {roc_auc_ovr_macro:0.4f}, ovo_macro: {roc_auc_ovo_macro:0.4f}")

    return roc_auc_ovo_macro

if __name__ == '__main__':
    cr_test = ConvergenceResults(sys.argv[1])
    cr_test.print_class_statistics()
    for epoch, data in cr_test.gt_classprob_for_epoch.items():
        compute_auc_roc(data)
