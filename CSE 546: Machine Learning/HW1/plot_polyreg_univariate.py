import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset

if __name__ == "__main__":
    from polyreg import PolynomialRegression  # type: ignore
else:
    from .polyreg import PolynomialRegression

if __name__ == "__main__":
    """
        Main function to test polynomial regression
    """

    # load the data
    allData = load_dataset("polyreg")

    X = allData[:, [0]]
    y = allData[:, [1]]

    # regression with degree = d
    d = 8

    lambdas = [0, 0.1, 1, 10, 100, 1000]
    figure, ax = plt.subplots(3, 2)
    
    for i in range(3):
        for j in range(2):
            model = PolynomialRegression(degree=d, reg_lambda=lambdas[j+2*i])
            model.fit(X, y)

            # output predictions
            xpoints = np.linspace(np.max(X), np.min(X), 100).reshape(-1, 1)
            ypoints = model.predict(xpoints)

            # plot curve

            ax[i,j].plot(X, y, "rx")
            ax[i,j].set_title(f"PolyRegression with d = {d} and lambda = {lambdas[j+2*i]}")
            ax[i,j].plot(xpoints, ypoints, "b-")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
