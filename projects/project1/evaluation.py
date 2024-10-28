#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from cross_validation import *

def compute_accuracy(y, y_pred):
    return np.sum(y == y_pred) / len(y)

def compute_f1_score(y, y_pred):
    tp = np.sum(np.logical_and(y == 1, y_pred == 1))
    fp = np.sum(np.logical_and(y == -1, y_pred == 1))
    fn = np.sum(np.logical_and(y == 1, y_pred == -1))
    return tp / (tp + (fp + fn) / 2)

def print_scores(x_tr, y_tr, x_tr_full, y_tr_full, w):
    print("\nTraining set:")
    f1_train = compute_f1_score(y_tr, compute_predictions_logistic(x_tr, w))
    acc_train = compute_accuracy(y_tr, compute_predictions_logistic(x_tr, w))
    print(f"F1 score on training set: {f1_train}")
    print(f"Accuracy on training set: {acc_train}")

    

    print("\nFull set:")
    f1_full = compute_f1_score(y_tr_full, compute_predictions_logistic(x_tr_full, w))
    acc_full = compute_accuracy(y_tr_full, compute_predictions_logistic(x_tr_full, w))
    print(f"F1 score on full set: {f1_full}")
    print(f"Accuracy on full set: {acc_full}")

    return None
