# -*- coding: utf-8 -*-
""" Discretize the (soft) differentiable tree into normal decision tree according to DDT paper"""
import torch
import torch.nn as nn
from utils.dataset import Dataset
import numpy as np

def discretize_tree(tree):
    for name, parameter in tree.named_parameters():
        # newParam = myFunction(parameter)
        # setattr(tree, name, newParam)
        print(name)
        if name == 'beta':
            setattr(tree, name, nn.Parameter(100*torch.ones(parameter.shape)))
            # print(name, parameter)

        elif name == 'linear.weight':
            parameters=[]
            for weights in parameter:
                bias = weights[0]
                max_id = np.argmax(np.abs(weights[1:].detach()))+1
                max_v = np.abs(weights[max_id].detach())
                new_weights = torch.zeros(weights.shape)
                new_weights[max_id] = torch.tensor(1)
                new_weights[0] = bias/max_v
                parameters.append(new_weights)
            print(torch.stack(parameters))
            setattr(tree, name, nn.Parameter(torch.stack(parameters)))
    # print(tree.beta.data)
    print(tree.linear.weight.data)
if __name__ == '__main__':    
    from sdt_train import learner_args
    from SDT import SDT

    learner_args['cuda'] = False  # cpu
    learner_args['lamda'] = 0.001
    learner_args['model_path']='./model/trees/sdt_'+str(learner_args['lamda'])+'_id'+str(1)+'beta'


    tree = SDT(learner_args)
    tree.load_model(learner_args['model_path'])

    discretize_tree(tree)