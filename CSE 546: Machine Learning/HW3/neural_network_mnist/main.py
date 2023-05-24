# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
from gettext import translation
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.alpha0 = 1/math.sqrt(d)
        self.alpha1 = 1/math.sqrt(h)

        self.weight0 = Uniform(-self.alpha0, self.alpha0).sample([h,d])
        self.bias0 = Uniform(-self.alpha0, self.alpha0).sample([h,1])
        self.weight1 = Uniform(-self.alpha1, self.alpha1).sample([k,h])
        self.bias1 = Uniform(-self.alpha1, self.alpha1).sample([k,1])

        self.parameters = [self.weight0, self.bias0, self.weight1, self.bias1]

        for item in self.parameters: 
             item.requires_grad =True
             item = Parameter(item)
        
        
    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        
        w0x_b0 = torch.add(self.weight0 @ x.T, self.bias0)
        w1x_b1 = self.weight1 @ relu(w0x_b0) + self.bias1
        return w1x_b1.transpose(0,1)

class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.alpha0 = 1/math.sqrt(d)
        self.alpha1 = 1/math.sqrt(h0)
        self.alpha2 = 1/math.sqrt(h1)

        self.weight0 = Uniform(-self.alpha0, self.alpha0).sample([h0,d])
        self.bias0 = Uniform(-self.alpha0, self.alpha0).sample([h0,1])
        self.weight1 = Uniform(-self.alpha1, self.alpha1).sample([h1,h0])
        self.bias1 = Uniform(-self.alpha1, self.alpha1).sample([h1,1])
        self.weight2 = Uniform(-self.alpha2, self.alpha2).sample([k,h1])
        self.bias2 = Uniform(-self.alpha2, self.alpha2).sample([k,1])

        self.parameters = [self.weight0, self.bias0, self.weight1, self.bias1, self.weight2, self.bias2]

        for item in self.parameters: 
             item.requires_grad =True
             item = Parameter(item)
        

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        w0x_b0 = self.weight0 @ x.transpose(0,1) + self.bias0
        w1x_b1 = self.weight1 @ relu(w0x_b0) + self.bias1
        w2x_b2 = self.weight2 @ relu(w1x_b1) + self.bias2
        return w2x_b2.transpose(0,1)


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).
            Note that y contains the classes as integers.

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    optimum = optimizer(model.parameters, 2e-3)
    average_loss = []
    
    for epoch in range(100):
        #print(epoch)
        loss_epoch = 0
        total_correct = 0

        for x, y in train_loader: 
            
            optimum.zero_grad()
            total_correct += torch.sum(torch.argmax(model(x), 1) == y).item()

            loss = cross_entropy(model(x), y)     
            loss_epoch += loss.item()

            loss.backward()
            optimum.step()

        average_loss.append(loss_epoch/len(train_loader))
        accuracy = total_correct/len(train_loader.dataset)
        #print("epoch", epoch, "accuracy", accuracy)
        if accuracy>0.99:
            print("epoch", epoch, "accuracy", accuracy)
            break

    return average_loss


@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()
    
    train_set = DataLoader(TensorDataset(x,y), batch_size = 32)
    test_set = DataLoader(TensorDataset(x_test, y_test), batch_size =32)
    
    d = 784
    h = 64
    h0 = h1 = 32
    k = 10
    F1_train = F1(h, d, k)
    F2_train  = F2(h0, h1, d, k)
    
    param_num1 =  sum(parameter.numel() for parameter in F1_train.parameters)
    print("F1 Number of parameters in model:", param_num1)
    param_num2 =  sum(parameter.numel() for parameter in F2_train.parameters)
    print("F2 Number of parameters in model:", param_num2)

    err1 = train(F1_train, Adam, train_set)
    err2 = train(F2_train, Adam, train_set)

    loss_1 = 0
    loss_2 = 0
    F1_correct = 0
    F2_correct = 0

    for x,y in test_set:

        F1_correct += torch.sum(torch.argmax(F1_train(x), 1) == y).item()
        F2_correct += torch.sum(torch.argmax(F2_train(x), 1) == y).item()

        loss_1 += cross_entropy(F1_train(x), y).item()
        loss_2 += cross_entropy(F2_train(x), y).item()

    loss1 = loss_1/len(test_set)
    loss2 = loss_2/len(test_set)
    accuracy_1 = F1_correct/len(test_set.dataset)
    accuracy_2 = F2_correct/len(test_set.dataset)

    print("The F1 model has test loss", loss1, "accuracy", accuracy_1)
    print("The F2 model has test loss", loss2, "accuracy", accuracy_2)
    
    plt.figure()
    plt.plot(range(len(err1)), err1)
    plt.title('F1 Model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('F1')
    plt.show()

    plt.figure()
    plt.plot(range(len(err2)), err2)
    plt.title('F2 Model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('F2')
    plt.show()

if __name__ == "__main__":
    main()
