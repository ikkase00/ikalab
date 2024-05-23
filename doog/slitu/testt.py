from make_parals import chanpanzee
from test import a
import torch

dum = torch.randn((4, 5, 1, 1), device = 'cuda')
doms, data = chanpanzee(a, dum, (2, 3), True, ['cpu', 'cuda:0'])
print(doms[0].device, data[0].device, data[0].shape)
print(doms[1].device, data[1].device, data[1].shape)

import sys
print(sys.path)