""" Model classes defined here! """

import torch
from torch import nn
import torch.nn.functional as F

class FeedForward(torch.nn.Module):
    def __init__(self, hidden_dim):
        """
        In the constructor we instantiate two nn.Linear modules and 
        assign them as member variables.
        """
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(784, hidden_dim);
        self.linear2 = nn.Linear(hidden_dim, 10);

    def forward(self, x):
        """
        Compute the forward pass of our model, which outputs logits.
        """
        m = nn.ReLU()
        x = self.linear1(x.cuda())
        x = m(x)
        x = self.linear2(x)
        return x

class SimpleConvNN(torch.nn.Module):
    def __init__(self, n1_chan, n1_kern, n2_kern):
        super(SimpleConvNN, self).__init__()
        self.cov1 = nn.Conv2d(1, n1_chan, n1_kern)
        self.cov2 = nn.Conv2d(n1_chan, 10, n2_kern, stride=2)
        
  
    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], 1, 28, 28))
        m = nn.ReLU()
        x = self.cov1(x)
        x = m(x)
        x = self.cov2(x)
        x = m(x)
        out = nn.MaxPool2d((x.shape[2], x.shape[3]))
        x = out(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1]))
        return x

class BestNN(torch.nn.Module):
    # take hyperparameters from the command line args!
    def __init__(self, n1_channels, n1_kernel, n2_channels, n2_kernel, pool1,
                 n3_channels, n3_kernel, n4_channels, n4_kernel, pool2, linear_features):
        super(BestNN, self).__init__()

    def forward(self, x):
        return x
