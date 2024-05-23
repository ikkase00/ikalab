import torch
import torch.nn as nn

a = nn.Sequential( 
    nn.Conv2d(5, 10, 3, 2, 1, bias = False),
    nn.BatchNorm2d(10), 
    nn.ReLU(),
    nn.Conv2d(10, 20, 3, 2, 1, bias = False),
    nn.BatchNorm2d(20),
    nn.LeakyReLU(),
)


