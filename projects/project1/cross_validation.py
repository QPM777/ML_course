#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import evaluation as eval
import implementations as imp

def compute_predictions_logistic(x_te, w):
    y_pred = np.dot(x_te, w)
    y_pred = imp.sigmoid(y_pred)
    return np.where(y_pred >= 0.5, 1, -1)

def cross_validation(x, y, k_indices, kth, init_w, max_iters, gamma):
    te_idx = k_indices[kth]
    tr_idx = np.concatenate([k_indices[i] for i in range(k_indices.shape[0]) if i != kth])
    x_tr, y_tr = x[tr_idx], y[tr_idx]
    x_te, y_te = x[te_idx], y[te_idx]

    w, _ = imp.logistic_regression(y=y_tr, tx=x_tr, initial_w=init_w, max_iters=max_iters, gamma=gamma)
    preds = compute_predictions_logistic(x_te, w)

    acc = eval.compute_accuracy(y_te, preds)
    f1 = eval.compute_f1_score(y_te, preds)
    
    return acc, f1, w

def run_cross_validation(x, y, k, init_w, max_iters, gamma):
    n = x.shape[0]
    interval = n // k
    indices = np.random.permutation(n)
    k_idx = np.array([indices[i * interval:(i + 1) * interval] for i in range(k)])
    
    acc_list, f1_list, weights = [], [], []
    
    for kth in range(k):
        acc, f1, w = cross_validation(x, y, k_idx, kth, init_w, max_iters, gamma)
        acc_list.append(acc)
        f1_list.append(f1)
        weights.append(w)

    mean_acc = np.mean(acc_list)
    mean_f1 = np.mean(f1_list)
    mean_weights = np.mean(weights, axis=0)
    
    return mean_acc, mean_f1, mean_weights

