import numpy as np


def compute_subgradient_mae(y, tx, w):
    """Compute a subgradient of the MAE at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the subgradient of the MAE at w.
    """
    # Compute the residuals
    residuals = y - np.dot(tx, w)  # shape (N, )
    
    # Compute the subgradient
    subgradient = np.zeros_like(w, dtype=np.float64)  # Assure-toi que c'est en float

    
    for i in range(len(residuals)):
        if residuals[i] > 0:
            subgradient += -tx[i]  # corresponding to -∇q(w) where q(w) = y - tx * w
        elif residuals[i] < 0:
            subgradient += tx[i]   # corresponding to ∇q(w)
    
    return subgradient / len(y)  # average over N
