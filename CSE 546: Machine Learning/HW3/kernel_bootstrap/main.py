from ast import AsyncFunctionDef
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem

RNG = np.random.RandomState(seed=446)

def f_true(x: np.ndarray) -> np.ndarray:
    """True function, which was used to generate data.
    Should be used for plotting.

    Args:
        x (np.ndarray): A (n,) array. Input.

    Returns:
        np.ndarray: A (n,) array.
    """
    return 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)


@problem.tag("hw3-A")
def poly_kernel(x_i: np.ndarray, x_j: np.ndarray, d: int) -> np.ndarray:
    """Polynomial kernel.

    Given two indices a and b it should calculate:
    K[a, b] = (x_i[a] * x_j[b] + 1)^d

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        d (int): Degree of polynomial.

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    return (np.outer(x_i, x_j) + 1) ** d


@problem.tag("hw3-A")
def rbf_kernel(x_i: np.ndarray, x_j: np.ndarray, gamma: float) -> np.ndarray:
    """Radial Basis Function (RBF) kernel.

    Given two indices a and b it should calculate:
    K[a, b] = exp(-gamma*(x_i[a] - x_j[b])^2)

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        gamma (float): Gamma parameter for RBF kernel. (Inverse of standard deviation)

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    return np.exp(-1 * gamma * np.subtract.outer(x_i, x_j) ** 2)

@problem.tag("hw3-A")
def train(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
) -> np.ndarray:
    """Trains and returns an alpha vector, that can be used to make predictions.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.

    Returns:
        np.ndarray: Array of shape (n,) containing alpha hat as described in the pdf.
    """
    #gamma = 1 / np.median(np.subtract.outer(x, x) ** 2)

    if kernel_function == poly_kernel:
        reg_matrix = _lambda * np.eye(len(x))
        kernel_part = poly_kernel(x,x,kernel_param)

    else:
        reg_matrix = _lambda * np.eye(len(x))
        kernel_part = rbf_kernel(x,x,kernel_param)
    alpha_hat = np.linalg.inv(kernel_part + reg_matrix).dot(y)
    #alpha_hat1 = np.linalg.solve(kernel_part + reg_matrix, y)
    return alpha_hat

@problem.tag("hw3-A", start_line=1)
def cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    num_folds: int,
) -> float:
    """Performs cross validation.

    In a for loop over folds:
        1. Set current fold to be validation, and set all other folds as training set.
        2, Train a function on training set, and then get mean squared error on current fold (validation set).
    Return validation loss averaged over all folds.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        float: Average loss of trained function on validation sets across folds.
    """
    fold_size = len(x) // num_folds
    idx = RNG.permutation(len(x))
    x_cur = np.array(x[idx])
    y_cur = np.array(y[idx])
    loss = []

    
    for i in range(num_folds):
        x_val = x_cur[i:i+fold_size]
        y_val = y_cur[i:i+fold_size]
        x_train_part1 = np.array(x_cur[:i])
        x_train_part2 = np.array(x_cur[i+fold_size:])
        y_train_part1 = np.array(y_cur[:i])
        y_train_part2 = np.array(y_cur[i+fold_size:])
        
        x_train = np.concatenate((x_train_part1, x_train_part2))
        y_train = np.concatenate((y_train_part1, y_train_part2))

        alpha = train(x_train, y_train, kernel_function, kernel_param = kernel_param, _lambda = _lambda)
        if kernel_function == poly_kernel:
            k = poly_kernel(x_i = x_train, x_j = x_val, d = kernel_param)
        else:
            k = rbf_kernel(x_i = x_train, x_j = x_val, gamma = kernel_param)
        
        pred_val = alpha @ k
        loss_single = np.mean((y_val - pred_val)**2)
        loss.append(loss_single)

    return np.mean(loss)


@problem.tag("hw3-A")
def rbf_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, float]:
    """
    Parameter search for RBF kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambda, loop over them and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda from some distribution and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be len(x) for LOO.

    Returns:
        Tuple[float, float]: Tuple containing best performing lambda and gamma pair.

    Note:
        - You do not really need to search over gamma. 1 / (median(dist(x_i, x_j)^2) for all unique pairs x_i, x_j in x
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
    """
    gamma = 1 / np.median(np.subtract.outer(x, x) ** 2)

    lambda_best = 0
    gamma_best = gamma
    
    lambda_power = np.linspace(-5,-1, num = 50)
    lambdas = 10 ** lambda_power
    res = []

    for i in range(len(lambdas)):
        res_cur = cross_validation(x, y, rbf_kernel, gamma, lambdas[i], num_folds)
        res.append(res_cur)

    ind = np.argmin(res)
    lambda_best = lambdas[ind]
    return lambda_best, gamma_best

@problem.tag("hw3-A")
def poly_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, int]:
    """
    Parameter search for Poly kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambdas and ds.
            Have nested loop over all possibilities and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda, d from some distributions and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, int]: Tuple containing best performing lambda and d pair.

    Note:
        - You can use gamma = 1 / median((x_i - x_j)^2) for all unique pairs x_i, x_j in x) for this problem. 
          However, if you would like to search over other possible values of gamma, you are welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
            and d from distribution [5, 6, ..., 24, 25]
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
            and possible ds to [5, 6, ..., 24, 25]
    """

    lambda_best = 0
    d_best = 0
    
    d_list = np.arange(5, 26, step = 1)

    lambda_power = np.linspace(-5,-1, num = 50)
    lambdas = 10 ** lambda_power
    #res_all = []
    min_res = float('inf')
    ind_1 = 0
    ind_2 = 0

    for i in range(len(lambdas)):
        res = []
        for j in range(len(d_list)):        
            res_cur = cross_validation(x, y, poly_kernel, d_list[j], lambdas[i], num_folds)
            res.append(res_cur)
        if min(res) < min_res:
           min_res = min(res)
           ind_1 = i
           ind_2 = np.argmin(res)
        #res_all.append(res)

    lambda_best = lambdas[ind_1]
    d_best = d_list[ind_2]
    return lambda_best, d_best

@problem.tag("hw3-A", start_line=1)
def main():
    """
    Main function of the problem

    It should:
        A. Using x_30, y_30, rbf_param_search and poly_param_search report optimal values for lambda (for rbf), gamma, lambda (for poly) and d.
            Note that x_30, y_30 has been loaded in for you. You do not need to use (x_300, y_300) or (x_1000, y_1000).
        B. For both rbf and poly kernels, train a function using x_30, y_30 and plot predictions on a fine grid

    Note:
        - In part b fine grid can be defined as np.linspace(0, 1, num=100)
        - When plotting you might find that your predictions go into hundreds, causing majority of the plot to look like a flat line.
            To avoid this call plt.ylim(-6, 6).
    """
    (x_30, y_30), (x_300, y_300), (x_1000, y_1000) = load_dataset("kernel_bootstrap")
    rbf_lambda, rbf_gamma = rbf_param_search(x_30, y_30, num_folds=10)
    print("the best rbf lambda value is", rbf_lambda, " and the best rbf gamma value is", rbf_gamma)

    # rbf
    rbf_k = rbf_kernel(x_i = x_30, x_j = x_30, gamma = rbf_gamma)
    rbf_alpha = train(x_30, y_30, rbf_kernel, rbf_gamma, rbf_lambda)
    rbf_pred = rbf_alpha @ rbf_k

    x_sim = np.linspace(0,1,1000)
    true_values = f_true(x_sim)

    plt.scatter(x_30, y_30, label='Original Data')
    plt.plot(x_sim, true_values, label='True $f(x)$')
    plt.plot(x_30[np.argsort(x_30)], rbf_pred[np.argsort(x_30)], label='$\widehat f(x)$')
    x = np.arange(0, 1, 0.01)
    plt.ylim(-6, 6)
    plt.title("RBF plot")
    plt.legend()
    
    plt.savefig('rbf.png')
    plt.show()

    # polynomial
    poly_lambda, poly_d = poly_param_search(x_30, y_30, num_folds=10)
    print("the best polynomial lambda value is", poly_lambda, " and the best polynomial gamma value is", poly_d)

    poly_k = poly_kernel(x_i = x_30, x_j = x_30, d = poly_d)
    poly_alpha = train(x_30, y_30, poly_kernel, poly_d, poly_lambda)
    poly_pred = poly_alpha @ poly_k

    plt.scatter(x_30, y_30, label='Original Data')
    plt.plot(x_sim, true_values, label='True $f(x)$')
    plt.plot(x_30[np.argsort(x_30)], poly_pred[np.argsort(x_30)], label='$\widehat f(x)$')
    x = np.arange(0, 1, 0.01)
    plt.ylim(-6, 6)
    plt.title("Polynomial plot")
    plt.legend()
    
    plt.savefig('Polynomial.png')
    plt.show()

if __name__ == "__main__":
    main()
