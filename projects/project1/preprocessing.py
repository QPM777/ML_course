import os as os
import numpy as np

def remove_intermediate_variables(x):
    with open(os.path.join("./data/dataset", "x_train.csv")) as f:
        first_line = f.readline().strip("\n")
    cols_name = np.asarray(first_line.split(",")[1:])
    startwith_ = np.array([c.startswith("_") for c in cols_name])
    endwith_ = np.array([c.endswith("_") for c in cols_name])
    to_drop = np.logical_or(startwith_, endwith_)
    exclude = [c not in ["_DENSTR2", "_GEOSTR", "_STATE"] for c in cols_name]
    to_drop = np.logical_and(to_drop, exclude)
    calculated_features_idx = np.where(to_drop)[0]
    new_data = np.delete(x, calculated_features_idx, axis=1)
    return new_data, calculated_features_idx

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


def balance_data(x, y, scale=1):
    y1_size = np.sum(y[y == 1])
    y0_balanced_size = int(y1_size * scale)
    y0_idx = np.random.choice(np.arange(0, len(y))[(y == -1).squeeze()], y0_balanced_size)
    balanced_idx = np.concatenate([y0_idx, np.arange(0, len(y))[(y == 1).squeeze()]])
    balanced_idx = np.sort(balanced_idx)
    return x[balanced_idx], y[balanced_idx]

def prepare_train_data(x, y, degree=1):
    new_x, iv_cols_idx = remove_intermediate_variables(x)
    new_x, rm_nan_columns_idx = remove_nan_columns(new_x, nan_percentage=80)
    new_x, rm_nan_rows_idx = remove_nan_rows(new_x, nan_percentage=50)
    y = np.delete(y, rm_nan_rows_idx, 0)
    new_x = replace_nan_mean(new_x)
    new_x, y = balance_data(new_x, y)
    new_x = build_poly(new_x, degree=degree)
    new_x = standardize(new_x)
    new_x = np.c_[np.ones((len(new_x), 1)), new_x]
    return new_x, y, iv_cols_idx, rm_nan_columns_idx

def prepare_test_data(x, rm_columns_index=[], rm_rows_index=[], degree=1):
    x = np.delete(np.delete(x, rm_columns_index, 1), rm_rows_index, 1)
    x = replace_nan_mean(x)
    x = build_poly(x, degree=degree)
    x = standardize(x)
    x = np.c_[np.ones((len(x), 1)), x]
    return x

def initialize_weights(x, method = "zeros"):
    if method == "zeros":
        return np.zeros((x.shape[1], 1))
    elif method == "ones":
        return np.ones((x.shape[1], 1))
    elif method == "random":
        return np.random.random((x.shape[1], 1))

def build_all(x_train, y_train, degree_pol=1):
    print(f"Size before preprocessing : {x_train.shape}")
    print("Let's preprocess")
    new_x_train, new_y_train, iv_cols_idx, rm_nan_columns_idx = prepare_train_data(x_train, y_train, degree=degree_pol)
    print(f"Size after preprocessing {new_x_train.shape}")
    x_train_full = prepare_test_data(x_train, iv_cols_idx, rm_nan_columns_idx, degree=degree_pol)
    initial_w = initialize_weights(new_x_train, "random")
    return new_x_train, new_y_train, iv_cols_idx, rm_nan_columns_idx, x_train_full, initial_w
