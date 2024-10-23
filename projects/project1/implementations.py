import numpy as np

## Computations of loss functions 

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
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss
    """
    sigmoid_term = 1/(1+np.exp(np.dot(-tx,w)))
    loss = -1/len(y) * (np.sum(y*np.log(sigmoid_term)+ (1-y)*np.log(1-sigmoid_term)))
    return loss 


### Computation of gradients 

def compute_gradient(y, tx, w):
    """Computes the gradient of the MSE loss at w.

    Args:
        y: numpy array of shape=(N,)
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        A numpy array of shape (2,), containing the gradient of the loss at w.
    """
    gradient = -(1 / len(y)) * np.dot(tx.T, y - np.dot(tx, w))
    return gradient

def compute_logistic_gradient(y, tx, w):
    """compute the gradient of logistic loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """
    sigmoid_term = 1/(1+np.exp(np.dot(-tx,w)))
    gradient = -(1/len(y))*np.dot(tx.T, y - sigmoid_term) 
    return gradient


def gradient_descent(y, tx, initial_w, max_iters, gamma, isLogistic=False):
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
        gradient = gradient_func(y, tx, w)  
        loss = loss_func(y, tx, w)          
        
        w = w - gamma * gradient  
        ws.append(w)
        losses.append(loss)
        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    return losses, ws

def mean_squared_error_gd(y, tx, initial_w,max_iters, gamma):
    return gradient_descent(y, tx, initial_w, max_iters, gamma)


