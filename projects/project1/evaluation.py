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

def print_scores(x, y, w):
    f1_train = compute_f1_score(y, compute_predictions_logistic(x, w))
    acc_train = compute_accuracy(y, compute_predictions_logistic(x, w))
    print(f"F1 score: {f1_train}")
    print(f"Accuracy: {acc_train}")
    return None
