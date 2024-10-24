from implementations import *
from helpers import *


import pandas as pd 

def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids

x_train, x_test, y_train, train_ids, test_ids =load_csv_data("./data/dataset")

print(np.shape(x_test))
print(np.shape(x_train))
print(np.shape(y_train))

def clean_columns_with_nan(x1, x2):
    # Compter les NaN dans chaque colonne pour x1 et x2
    nan_counts_x1 = np.isnan(x1).sum(axis=0)
    nan_counts_x2 = np.isnan(x2).sum(axis=0)
    
    num_rows_x1 = x1.shape[0]
    num_rows_x2 = x2.shape[0]

    valid_columns = []

    for i in range(x1.shape[1]):
        
        if nan_counts_x1[i] < num_rows_x1 / 2 and nan_counts_x2[i] < num_rows_x2 / 2:
            # Remplacer les NaN par la moyenne correspondante pour chaque tableau
            mean_value_x1 = np.nanmean(x1[:, i])
            x1[:, i] = np.where(np.isnan(x1[:, i]), mean_value_x1, x1[:, i])
            
            mean_value_x2 = np.nanmean(x2[:, i])
            x2[:, i] = np.where(np.isnan(x2[:, i]), mean_value_x2, x2[:, i])
            
            valid_columns.append(i)  # Conserver l'index de la colonne valide

    # Retourner les tableaux nettoyés et la liste des colonnes valides
    return x1[:, valid_columns], x2[:, valid_columns], valid_columns
def normalize_data(x):
    
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)


def remove_constant_columns(x1, x2):
    variances_x1 = np.var(x1, axis=0)
    variances_x2 = np.var(x2, axis=0)
    threshold = 1e-10
    constant_columns_x1 = variances_x1 <= threshold
    constant_columns_x2 = variances_x2 <= threshold
    columns_to_remove = constant_columns_x1 | constant_columns_x2
    x1_cleaned = x1[:, ~columns_to_remove]
    x2_cleaned = x2[:, ~columns_to_remove]
    return x1_cleaned, x2_cleaned

def initialize_weights(x_train, initial_value=1/len(x_train[0])):
    """Initialise un vecteur de poids avec une valeur initiale donnée."""
    return np.full(x_train.shape[1], initial_value)


x_train, x_test, _ = clean_columns_with_nan(x_train,x_test)
x_train, x_test = remove_constant_columns(x_train,x_test)
x_train = normalize_data(x_train)
x_test = normalize_data(x_test)



w,loss = least_squares(y_train,x_train)
print(f"weight: {w}\nLoss value: {loss:.4f}")

y_predict = np.dot(x_test,w)
y_pred_class = [1 if pred >= 0 else -1 for pred in y_predict]

create_csv_submission(test_ids, y_pred_class, "name")