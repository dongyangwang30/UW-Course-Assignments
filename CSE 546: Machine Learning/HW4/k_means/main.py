if __name__ == "__main__":
    from k_means import calculate_error, lloyd_algorithm  # type: ignore
else:
    from .k_means import lloyd_algorithm, calculate_error

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw4-A", start_line=1)
def main():
    """Main function of k-means problem

    Run Lloyd's Algorithm for k=10, and report 10 centers returned.

    NOTE: This code takes a while to run. For debugging purposes you might want to change:
        x_train to x_train[:10000]. CHANGE IT BACK before submission.
    """
    (x_train, _), (x_test, _) = load_dataset("mnist")
    #x_train = x_train[:10000]
    k = 10 
    centers, errors = lloyd_algorithm(x_train, k)

    fig, axes = plt.subplots(1, 10, figsize=(10, 1))
    fig.suptitle('Clusters for k = 10')
    
    for i, ax in enumerate(axes):
        ax.imshow(np.reshape(centers[i], (28, 28)), cmap='gray')
        ax.set_title(f'k = {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('a3.png')
    plt.show()

if __name__ == "__main__":
    main()
