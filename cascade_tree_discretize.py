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

        elif name == 'fl_inner_nodes.weight' or 'dc_inner_nodes.weight':
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
            if name == 'fl_inner_nodes.weight':
                tree.fl_inner_nodes.weight = nn.Parameter(torch.stack(parameters))
                print(tree.fl_inner_nodes.weight.data)
            elif name == 'dc_inner_nodes.weight':
                tree.dc_inner_nodes.weight = nn.Parameter(torch.stack(parameters))
                print(tree.dc_inner_nodes.weight.data)

if __name__ == '__main__':    
    from cascade_tree_train import learner_args
    from cascade_tree import Cascade_DDT
    learner_args['num_intermediate_variables']=3
    learner_args['feature_learning_depth']=3
    learner_args['decision_depth']=3

    learner_args['model_path'] = './model/trees/cascade_'+str(learner_args['feature_learning_depth'])+'_'\
        +str(learner_args['decision_depth'])+'_var'+str(learner_args['num_intermediate_variables'])+'_id'+str(2)

    learner_args['cuda'] = False  # cpu

    tree = Cascade_DDT(learner_args)
    tree.load_model(learner_args['model_path'])
    print(tree.named_parameters())
    discretize_tree(tree)

    tree.save_model(model_path = learner_args['model_path']+'_discretized')