if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from losses import CrossEntropyLossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from .optimizers import SGDOptimizer
    from .losses import CrossEntropyLossLayer
    from .train import plot_model_guesses, train

from audioop import cross
from typing import Any, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

RNG = torch.Generator()
RNG.manual_seed(446)


@problem.tag("hw3-A")
def crossentropy_parameter_search(
    dataset_train: TensorDataset, dataset_val: TensorDataset
) -> Dict[str, Any]:
    """
    Main subroutine of the CrossEntropy problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers
    NOTE: Each model should end with a Softmax layer due to CrossEntropyLossLayer requirement.

    Notes:
        - Try using learning rate between 1e-5 and 1e-3.
        - When choosing the number of epochs, consider effect of other hyperparameters on it.
            For example as learning rate gets smaller you will need more epochs to converge.
        - When searching over batch_size using powers of 2 (starting at around 32) is typically a good heuristic.
            Make sure it is not too big as you can end up with standard (or almost) gradient descent!

    Args:
        dataset_train (TensorDataset): Dataset for training.
        dataset_val (TensorDataset): Dataset for validation.

    Returns:
        Dict[str, Any]: Dictionary/Map containing history of training of all models.
            You are free to employ any structure of this dictionary, but we suggest the following:
            {
                name_of_model: {
                    "train": Per epoch losses of model on train set,
                    "val": Per epoch losses of model on validation set,
                    "model": Actual PyTorch model (type: nn.Module),
                }
            }
    """
    lr = 1e-4
    epochs = 100
    dim_in = dim_out = 2
    model_1 = torch.nn.Sequential(LinearLayer(dim_in = dim_in, dim_out= dim_out, generator=RNG),\
            SoftmaxLayer())
    model_2 = torch.nn.Sequential(LinearLayer(dim_in = dim_in, dim_out= dim_out, generator=RNG),\
            LinearLayer(dim_in = dim_in, dim_out= dim_out, generator=RNG),\
            SigmoidLayer(),\
            SoftmaxLayer())
    model_3 = torch.nn.Sequential(LinearLayer(dim_in = dim_in, dim_out= dim_out, generator=RNG),\
            LinearLayer(dim_in = dim_in, dim_out= dim_out, generator=RNG),\
            ReLULayer(),\
            SoftmaxLayer())
    model_4 = torch.nn.Sequential(LinearLayer(dim_in = dim_in, dim_out= dim_out, generator=RNG),\
            SigmoidLayer(),\
            LinearLayer(dim_in = dim_in, dim_out= dim_out, generator=RNG),\
            ReLULayer(),\
            SoftmaxLayer())
    model_5 = torch.nn.Sequential(LinearLayer(dim_in = dim_in, dim_out= dim_out, generator=RNG),\
            ReLULayer(),\
            LinearLayer(dim_in = dim_in, dim_out= dim_out, generator=RNG),\
            SigmoidLayer(),\
            SoftmaxLayer())

    model1 = train(dataset_train, model_1, CrossEntropyLossLayer(), SGDOptimizer, val_loader= dataset_val, epochs = epochs)
    model2 = train(dataset_train, model_2, CrossEntropyLossLayer(), SGDOptimizer, val_loader= dataset_val, epochs = epochs)
    model3 = train(dataset_train, model_3, CrossEntropyLossLayer(), SGDOptimizer, val_loader= dataset_val, epochs = epochs)
    model4 = train(dataset_train, model_4, CrossEntropyLossLayer(), SGDOptimizer, val_loader= dataset_val, epochs = epochs)
    model5 = train(dataset_train, model_5, CrossEntropyLossLayer(), SGDOptimizer, val_loader= dataset_val, epochs = epochs)

    model_names = ["linear", "NN_1_sigmoid", "NN_1_ReLU", "NN_2_sigmoid_ReLU", "NN_2_ReLU_sigmoid"]
    models = [model1, model2, model3, model4, model5]
    result = {}

    for i in range(len(models)):
        result[model_names[i]] = {"train": models[i]["train"], "val": models[i]["val"], "model": models[i]}
    
    return result  
    
        

@problem.tag("hw3-A")
def accuracy_score(model, dataloader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for CrossEntropy.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is an integer representing a correct class to a corresponding observation.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to MSE accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """
    
    total_correct = 0
    for x,y in dataloader:

        y_pred = torch.argmax(model(x),1)
        total_correct += torch.sum(y_pred==y).item()
    
    return total_correct/len(dataloader)
    
@problem.tag("hw3-A", start_line=7)
def main():
    """
    Main function of the Crossentropy problem.
    It should:
        1. Call crossentropy_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            x-axis should be epochs, y-axis should me Crossentropy loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    dataset_train = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(to_one_hot(y)))
    dataset_val = TensorDataset(
        torch.from_numpy(x_val).float(), torch.from_numpy(to_one_hot(y_val))
    )
    dataset_test = TensorDataset(
        torch.from_numpy(x_test).float(), torch.from_numpy(to_one_hot(y_test))
    )

    mse_configs = crossentropy_parameter_search(dataset_train, dataset_val)
    
    model_names = ["linear", "NN_1_sigmoid", "NN_1_ReLU", "NN_2_sigmoid_ReLU", "NN_2_ReLU_sigmoid"]
    mse_val = []
    for i in range(len(mse_configs)):
        mse_val.append(mse_configs[model_names[i]]["val"][-1])

    ind = np.argmin(mse_val)
    print("The lowest validation MSE is", mse_val[ind], "with the model", model_names[ind])
    
    best_model = mse_configs[model_names[i]]["model"]
    print("Train with MSE accuracy:", accuracy_score(best_model, dataset_test))
    plot_model_guesses(dataset_test, best_model, "Plot Guessing MSE")


if __name__ == "__main__":
    main()
