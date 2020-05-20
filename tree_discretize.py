# -*- coding: utf-8 -*-
""" Discretize the (soft) differentiable tree into normal decision tree according to DDT paper"""
import torch
import torch.nn as nn
from utils.dataset import Dataset
import numpy as np

def discretize_tree(tree):
    for name, parameter in tree.named_parameters():
        print(name)
        if name == 'beta':
            setattr(tree, name, nn.Parameter(100*torch.ones(parameter.shape)))

        elif name == 'linear.weight':
            parameters=[]
            print(parameter)
            for weights in parameter:
                bias = weights[0]
                max_id = np.argmax(np.abs(weights[1:].detach()))+1
                max_v = weights[max_id].detach()
                new_weights = torch.zeros(weights.shape)
                if max_v>0:
                    new_weights[max_id] = torch.tensor(1)
                else:
                    new_weights[max_id] = torch.tensor(-1)
                new_weights[0] = bias/np.abs(max_v)
                parameters.append(new_weights)
            tree.linear.weight = nn.Parameter(torch.stack(parameters))
    print(tree.linear.weight.data)

if __name__ == '__main__':    
    from sdt_train import learner_args
    from SDT import SDT

    learner_args['cuda'] = False  # cpu

    tree = SDT(learner_args)
    tree.load_model(learner_args['model_path'])

    discretize_tree(tree)

    tree.save_model(model_path = learner_args['model_path']+'_discretized')