import numpy as np


## Computation of loss functions

def compute_loss(y,tx,w,loss_method="MSE"):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, 1)
        tx: shape=(N, D)
        w: shape=(D, 1). The vector of model parameters.
        loss_method: for the moment string ("MSE" or "MAE"), otherwise exception
        

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    if loss_method == "MSE":
        loss = np.mean((y- np.dot(tx,w))**2)
    elif loss_method == "MAE":
        loss = np.mean(np.abs(y- np.dot(tx,w)))
    else:
        raise Exception("For the moment we only use MAE or MSE")
    return loss 

def compute_logistic_loss(y,tx,w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, )
        tx: shape=(N, D)
        w:  shape=(D, )

    Returns:
        a non-negative loss
    """
    sigmoid_term = 1/(1+np.exp(np.dot(-tx,w)))
    loss = -1/len(y) * (np.sum(y*np.log(sigmoid_term)+ (1-y)*np.log(1-sigmoid_term)))
    return loss 


### Computation of gradients 

def compute_gradient(y, tx, w, lambda_=0):
    """Computes the gradient of the MSE loss at w.

    Args:
        y: numpy array of shape=(N,)
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        A numpy array of shape (2,), containing the gradient of the loss at w.
    """
    gradient = -(1 / len(y)) * np.dot(tx.T, y - np.dot(tx, w)) + 2*lambda_*w

    return gradient

def compute_logistic_gradient(y, tx, w, lambda_=0):
    """compute the gradient of logistic loss.

    Args:
        y:  shape=(N, )
        tx: shape=(N, D)
        w:  shape=(D, )

    Returns:
        a vector of shape (D, 1)
    """
    sigmoid_term = 1/(1+np.exp(np.dot(-tx,w)))
    gradient = -(1/len(y))*np.dot(tx.T, y - sigmoid_term)  + 2*lambda_*w
    return gradient


## Gradient descent algorithms

def gradient_descent(y, tx, initial_w, max_iters, gamma, isLogistic=False, lambda_= 0):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    dict_case = {
        True: (compute_logistic_gradient, compute_logistic_loss),
        False: (compute_gradient, compute_loss)
    }

    gradient_func, loss_func = dict_case[isLogistic]

    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        gradient = gradient_func(y, tx, w, lambda_)  
        loss = loss_func(y, tx, w)          
        w = w - gamma * gradient  
        ws.append(w)
        losses.append(loss)

    return ws[-1], losses[-1]


def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, isLogistic=False):
    """The Stochastic Gradient Descent (SGD) algorithm with a mini-batch size of 1.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """
    dict_case = {
        True: (compute_logistic_gradient, compute_logistic_loss),
        False: (compute_gradient, compute_loss)
    }

    gradient_func, loss_func = dict_case[isLogistic]

    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        rand_idx = np.random.randint(0, len(y))
        minibatch_y = y[rand_idx:rand_idx+1]
        minibatch_tx = tx[rand_idx:rand_idx+1]

        gradient = gradient_func(minibatch_y, minibatch_tx, w)
        loss = loss_func(minibatch_y, minibatch_tx, w)

        w = w - gamma * gradient
        ws.append(w)
        losses.append(loss)
    
    return ws[-1], losses[-1]

# List of function 

def mean_squared_error_gd(y, tx, initial_w,max_iters, gamma):
    return gradient_descent(y, tx, initial_w, max_iters, gamma)

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma)

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss
    
def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
    """
    reg_term = 2 * tx.shape[0] * lambda_ * np.eye(tx.shape[1])
    
    a = tx.T.dot(tx) + reg_term
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss

def logistic_regression(y, tx, initial_w,max_iters, gamma):
    return gradient_descent(y, tx, initial_w, max_iters, gamma, isLogistic=True)

def reg_logistic_regression(y, tx, lambda_, initial_w,max_iters, gamma):
    return gradient_descent(y, tx, initial_w, max_iters, gamma, isLogistic=True, lambda_=lambda_)
