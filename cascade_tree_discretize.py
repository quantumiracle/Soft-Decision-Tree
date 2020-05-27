# -*- coding: utf-8 -*-
""" Discretize the (soft) differentiable tree into normal decision tree according to DDT paper"""
import torch
import torch.nn as nn
from utils.dataset import Dataset
import numpy as np
import copy

def discretize_tree(original_tree, FL=True, DC=True):
    """ 
    Discretize the cascading tree
    if FL: discretize the feature learning tree;  
    if DC: discretize the decision making tree.
    """
    tree = copy.deepcopy(original_tree)
    for name, parameter in tree.named_parameters():
        # print(name)

        # discretize feature learning tree and decision making tree separately
        if FL:
            if name == 'beta_fl':
                setattr(tree, name, nn.Parameter(100*torch.ones(parameter.shape)))

            elif name == 'fl_inner_nodes.weight':
                parameters=[]
                # print(parameter)
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

                tree.fl_inner_nodes.weight = nn.Parameter(torch.stack(parameters))
                # print(tree.fl_inner_nodes.weight.data)

        if DC:
            if name == 'beta_dc':
                setattr(tree, name, nn.Parameter(100*torch.ones(parameter.shape)))

            elif name == 'dc_inner_nodes.weight':
                parameters=[]
                # print(parameter)
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

                tree.dc_inner_nodes.weight = nn.Parameter(torch.stack(parameters))
                # print(tree.dc_inner_nodes.weight.data)

    return tree

def onehot_coding(target, output_dim):
    target_onehot = torch.FloatTensor(target.size()[0], output_dim)
    target_onehot.data.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1.)
    return target_onehot

def discretization_evaluation(tree, discretized_tree):
    # Load data
    data_dir = './data/discrete_'
    data_path = data_dir+'state.npy'
    label_path = data_dir+'action.npy'

    # a data loader with all data in dataset
    test_loader = torch.utils.data.DataLoader(Dataset(data_path, label_path, partition='test'),
                                    batch_size=int(1e4),
                                    shuffle=True)
    accuracy_list=[]
    accuracy_list_=[]
    correct=0.
    correct_=0.
    for batch_idx, (data, target) in enumerate(test_loader):
        # data, target = data.to(device), target.to(device)
        target_onehot = onehot_coding(target, tree.args['output_dim'])
        prediction, _, _ = tree.forward(data)
        prediction_, _, _ = discretized_tree.forward(data)
        with torch.no_grad():
            pred = prediction.data.max(1)[1]
            correct += pred.eq(target.view(-1).data).sum()
            pred_ = prediction_.data.max(1)[1]
            correct_ += pred_.eq(target.view(-1).data).sum()
    accuracy = 100. * float(correct) / len(test_loader.dataset)
    accuracy_ = 100. * float(correct_) / len(test_loader.dataset)
    print('Original Tree Accuracy: {:.4f} | Discretized Tree Accuracy: {:.4f}'.format(accuracy, accuracy_))
    

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

    discretized_tree = discretize_tree(tree, FL=True, DC=False)
    discretization_evaluation(tree, discretized_tree)

    tree.save_model(model_path = learner_args['model_path']+'_discretized')