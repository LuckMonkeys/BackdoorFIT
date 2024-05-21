
import torch
import numpy as np

from utils import logger


def vectorize_net(net):
    return torch.cat([p.view(-1) for p in net.parameters()])

def vectorize_dict(d, key_order):
    # raise NotImplementedError("The order of dict is not guaranteed.")
    return torch.cat([d[key].view(-1) for key in key_order])

class Defense():
    def __init__(self, name, *args, **kwargs):
       self.name = name 

    def exec(self, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.exec(*args, **kwargs)

