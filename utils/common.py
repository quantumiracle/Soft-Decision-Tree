import torch
import torch.nn as nn
import numpy as np

def onehot_coding(target, output_dim):
    target_onehot = torch.FloatTensor(target.size()[0], output_dim)
    target_onehot.data.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1.)
    return target_onehot