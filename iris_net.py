import torch
import torch.nn as nn
import torch.optim as optim

class IrisNet(nn.Module):

    def __init__(self):
        """
        Initialization of IrisNet class. IrisNet class inherits Module class.
        Input layer have 4 neurons, hidden 10 and output 3.
        Activation function is ReLU.
        """
        super(IrisNet, self).__init__()

        self.fc1 = nn.Linear(4, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        """
        Defines network architecture.
        Variable x is going through layers defined in __init method.

        :param x: input data that will be processed by neural network.
        :return: output data after processing by neural network.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x