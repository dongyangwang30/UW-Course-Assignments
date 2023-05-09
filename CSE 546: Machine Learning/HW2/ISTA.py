from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float, eta: float
) -> Tuple[np.ndarray, float]:
    """Single step in ISTA algorithm.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        bias (float): Bias returned from the step before.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.
        eta (float): Step-size. Determines how far the ISTA iteration moves for each step.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
    
    """
    bias_prime = bias - 2 * eta * np.sum(X @ weight + bias -y)
    weight_prime_all =  weight - 2 * eta * (X.T @ (X @ weight + bias -y))
    check = 2 * eta * _lambda
    for k in range(X.shape[1]):
        weight_prime = weight_prime_all[k]
        #print("item", weight[k] - 2 * eta * (X[:, k].T @ (X @ weight + bias -y)))
        #print("check", check)
        #print("weight_prime",wegiht_prime)
        if weight_prime < -1 * check:  
            weight_prime += check
        elif weight_prime > check: 
            weight_prime -= check
        else: weight_prime = 0.0
        #print("wegiht_prime", weight_prime)
        weight_prime_all[k] = weight_prime
    #print(weight_prime_all)
    return weight_prime_all, bias_prime

@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized MSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    return (np.sum((X @ weight + bias - y)**2) + _lambda * (np.linalg.norm(weight, ord=1))) 

@problem.tag("hw2-A", start_line=5)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    eta: float = 0.001,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
    start_bias: float = None
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight and bias.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        eta (float): Step size.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.
        start_bias (np.ndarray, optional): Bias for hot-starting model.
            If None, defaults to zero. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float representing the bias.

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
        start_bias = 0
    old_w: Optional[np.ndarray] = None
    old_b: Optional[np.ndarray] = None
    
    res = False
    i = 0

    while not res:
        old_w = np.copy(start_weight)
        start_weight, start_bias = step(X, y, weight = start_weight, bias = start_bias, _lambda = _lambda, eta = eta)
        res = convergence_criterion(weight = start_weight, old_w = old_w, bias = None, old_b = None, convergence_delta = convergence_delta) 
        
        print("iteration", i)
        #print("old_w", old_w, "start_weight", start_weight)
        print(max(np.abs(start_weight - old_w))) 
        #print("res", res)
        i += 1
    return start_weight, start_bias

@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, bias: float, old_b: float, convergence_delta: float
) -> bool:
    """Function determining whether weight has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compate it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of coordinate gradient descent.
        old_w (np.ndarray): Weight from previous iteration of coordinate gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight has not converged yet. True otherwise.
    """
    if old_w is None:
        return False
    max_diff = max(np.abs(weight - old_w))
    return max_diff < convergence_delta

@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    # randomize with seed
    np.random.seed(42)
    
    # initialize
    n, d, k, sigma = 500, 1000, 100, 1

    # weight generation
    weight = np.zeros((d, ))
    for j in range(k):
        weight[j] = (j+1)/k

    # data generation
    X = np.random.normal(size=(n, d))
    y = X @ weight + np.random.normal(size=(n,))
    
    # normalize
    X_mean = np.mean(X, axis = 0)
    X_sigma = np.std(X, axis = 0)
    X = (X - X_mean) / X_sigma

    # lambda
    _lambda = max(2*np.sum(X * (y-np.mean(y)) [:, None], axis=0))

    lambdas, nonzeros, fdrs, tprs = [], [], [], [] 

    for _ in range(30):
        # train
        updated_weight, updated_bias = train(X, y, _lambda = _lambda, eta = 0.0001, convergence_delta = 1e-3)
        
        # nonzero
        nonzero = np.sum(updated_weight != 0)
        nonzeros.append(nonzero)

        # fdr and tpr
        if nonzero == 0:
            fdr = 0
        else:
            fdr = np.sum(updated_weight[k:] != 0) / nonzero
        tpr = np.sum(updated_weight[:k] != 0)  / k
        
        fdrs.append(fdr)
        tprs.append(tpr)

        # save and update lambda
        lambdas.append(_lambda)
        _lambda /= 2

    # plot nonzero
    plt.plot(lambdas, nonzeros)
    plt.xscale('log')
    plt.title("Non-zeros vs $\lambda$")
    plt.xlabel('$\lambda$')
    plt.ylabel('Number of Non-zeros')
    plt.show()

    # plot fdr and tpr
    plt.plot(fdrs, tprs)
    plt.title("False Discovery Rate vs True Positive Rate")
    plt.xlabel('FDR')
    plt.ylabel('TPR')
    plt.show()
    
if __name__ == "__main__":
    main()
