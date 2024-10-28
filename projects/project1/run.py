import os
import numpy as np
from implementations import *
from preprocessing import *
from evaluation import *
from cross_validation import *
from helpers import *

degree_pol = 5

print("Loading data...")
x_train, x_test, y_train, train_ids, test_ids = load_csv_data("./data/dataset")
y_train = np.expand_dims(y_train, 1)
y_train = y_train.reshape((y_train.shape[0], 1))

(x_train_balanced, y_train_balanced, rm_column_index, idx_nan_percent, x_train_full, initial_w) = build_all(x_train=x_train, y_train=y_train, degree_pol=degree_pol)

acc, f1, w = run_cross_validation(x_train_balanced, y_train_balanced, 5, init_w=initial_w, max_iters=500, gamma=0.15)

print_scores(x_train_balanced, y_train_balanced, x_train_full, y_train, w)

x_test_standardized = prepare_test_data(x_test, rm_column_index, idx_nan_percent, degree=degree_pol)
pred = compute_predictions_logistic(x_test_standardized, w)

create_csv_submission(test_ids, pred, name="submission_test.csv")
