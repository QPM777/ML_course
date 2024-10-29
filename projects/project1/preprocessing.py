import os as os
import numpy as np


def build_poly(x, degree=1):
    poly = None
    for deg in range(1, degree + 1):
        poly = np.power(x, deg) if poly is None else np.c_[poly, np.power(x, deg)]
    return poly

def standardize(data):
    mean = np.mean(data, axis=0)
    std = np.where(np.std(data, axis=0) == 0, 1, np.std(data, axis=0))
    return (data - mean) / std


def replace_nan_mean(x):
    for col in range(x.shape[1]):
        col_values = x[:, col]
        mean = np.sum(col_values[~(col_values != col_values)]) / np.sum(col_values == col_values)
        x[col_values != col_values, col] = mean
    return x

def remove_nan_columns(x, nan_percentage=80.0):
    rows = x.shape[0]
    idx_to_drop = [col for col in range(x.shape[1]) if np.sum(x[:, col] != x[:, col]) / rows * 100 > nan_percentage]
    return x[:, [i for i in range(x.shape[1]) if i not in idx_to_drop]], idx_to_drop

def remove_nan_rows(x, nan_percentage=50.0):
    cols = x.shape[1]
    idx_to_drop = [row for row in range(x.shape[0]) if np.sum(x[row, :] != x[row, :]) / cols * 100 > nan_percentage]
    return x[[i for i in range(x.shape[0]) if i not in idx_to_drop], :], idx_to_drop


def balance_data(x, y, scale=2):
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == -1)[0]
    target_neg_size = min(len(neg_indices),int(len(pos_indices) * scale))
    balanced_neg_indices = np.random.choice(neg_indices, target_neg_size, replace=False)
    balanced_indices = np.sort(np.concatenate([pos_indices, balanced_neg_indices]))

    return x[balanced_indices], y[balanced_indices]


def prepare_train_data(x, y, degree=1):
    new_x, rm_nan_columns_idx = remove_nan_columns(x, nan_percentage=80)
    new_x, rm_nan_rows_idx = remove_nan_rows(new_x, nan_percentage=50)
    y = np.delete(y, rm_nan_rows_idx, 0)
    new_x = replace_nan_mean(new_x)
    new_x, y = balance_data(new_x, y)
    new_x = build_poly(new_x, degree=degree)
    new_x = standardize(new_x)
    new_x = np.c_[np.ones((len(new_x), 1)), new_x]
    return new_x, y, rm_nan_columns_idx

def prepare_test_data(x, rm_rows_index=[], degree=1):
    x = np.delete(x, rm_rows_index, 1)
    x = replace_nan_mean(x)
    x = build_poly(x, degree=degree)
    x = standardize(x)
    x = np.c_[np.ones((len(x), 1)), x]
    return x

def initialize_weights(x):
    return np.random.random((x.shape[1], 1))

def build_all(x_train, y_train, degree_pol=1):
    print(f"Size before preprocessing : {x_train.shape}")
    print("Let's preprocess")
    new_x_train, new_y_train, rm_nan_columns_idx = prepare_train_data(x_train, y_train, degree=degree_pol)
    print(f"Size after preprocessing {new_x_train.shape}")
    x_train_full = prepare_test_data(x_train, rm_nan_columns_idx, degree=degree_pol)
    initial_w = initialize_weights(new_x_train)
    return new_x_train, new_y_train, rm_nan_columns_idx, x_train_full, initial_w

