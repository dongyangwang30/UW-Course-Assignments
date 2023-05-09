if __name__ == "__main__":
    from ISTA import train  # type: ignore
else:
    from .ISTA import train

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    #df_train, df_test = load_dataset("crime")
    df_train = pd.read_table("crime-train.txt")
    df_test = pd.read_table("crime-test.txt")
    
    # data
    X_train, y_train = df_train.drop("ViolentCrimesPerPop", axis = 1), df_train["ViolentCrimesPerPop"]
    X_test, y_test = df_test.drop("ViolentCrimesPerPop", axis = 1), df_test["ViolentCrimesPerPop"]

    # normalize
    X_train_mean = np.mean(X_train, axis = 0)
    X_train_sigma = np.std(X_train, axis = 0)
    X_train = (X_train - X_train_mean) / X_train_sigma

    X_test_mean = np.mean(X_test, axis = 0)
    X_test_sigma = np.std(X_test, axis = 0)
    X_test = (X_test - X_test_mean) / X_test_sigma
    
    # Get target columns for part d
    query_cols = ['agePct12t29', 'pctWSocSec', 'pctUrban', 'agePct65up', 'householdsize']
    cols_index = [X_train.columns.get_loc(col) for col in query_cols]

    # lambda
    X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
    X_test, y_test = X_test.to_numpy(), y_test.to_numpy()
    _lambda = max(2*np.sum(X_train * (y_train-np.mean(y_train))[:, None], axis=0))

    # initialize
    nonzeros, lambdas = [], []
    MSE_train, MSE_test = [], []

    # weight_path = []
    df = pd.DataFrame(columns=query_cols)

    # train model
    while _lambda > 0.01:
        # train
        updated_weight, updated_bias = train(X_train, y_train, _lambda = _lambda, eta = 0.00001, convergence_delta = 1e-4)
        
        # nonzero
        nonzero = np.sum(np.abs(updated_weight) > 0)
        nonzeros.append(nonzero)

        # regularization path
        weight_current = [updated_weight[i, ] for i in cols_index]
        #weight_path.append(weight_current)
        df.loc[len(df)] = weight_current

        # MSE
        mse_train_one = np.mean((y_train - X_train @ updated_weight)**2 - updated_bias)
        mse_test_one = np.mean((y_test - X_test @ updated_weight)**2 - updated_bias)
        MSE_train.append(mse_train_one)
        MSE_test.append(mse_test_one)

        # save and update lambda
        lambdas.append(_lambda)
        _lambda /= 2
    
    # plot nonzero (part c)
    plt.plot(lambdas, nonzeros)
    plt.xscale('log')
    plt.title("Non-zeros vs $\lambda$")
    plt.xlabel('$\lambda$')
    plt.ylabel('Number of Non-zeros')
    plt.show()

    # plot regularization path (part d)
    plt.plot(lambdas, df[['agePct12t29', 'pctWSocSec', 'pctUrban', 'agePct65up', 'householdsize']])
    plt.xscale('log')
    plt.title("Variables vs $\lambda$")
    plt.legend(['agePct12t29', 'pctWSocSec', 'pctUrban', 'agePct65up', 'householdsize'])
    plt.xlabel('$\lambda$')
    plt.ylabel('Weight')
    plt.show()

    # plot MSE (part e)
    plt.plot(lambdas, MSE_train)
    plt.plot(lambdas, MSE_test)
    plt.xscale('log')
    plt.title("MSE vs $\lambda$")
    plt.legend(['Training MSE', 'Test MSE'])
    plt.xlabel('$\lambda$')
    plt.ylabel('Value')
    plt.show()

    # Display weights for lambda = 30
    weight_30, bias_30 = train(X_train, y_train, _lambda = 30, eta = 0.00001, convergence_delta = 1e-4)
    print('The most positive feacture is', df_train.columns[np.argmax(weight_30)], 'with value', max(weight_30))
    print('The most negative feacture is', df_train.columns[np.argmin(weight_30)], 'with value', min(weight_30))
        
if __name__ == "__main__":
    main()
