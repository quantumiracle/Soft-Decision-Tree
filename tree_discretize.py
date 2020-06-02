# -*- coding: utf-8 -*-
""" Discretize the (soft) differentiable tree into normal decision tree according to DDT paper"""
import torch
import torch.nn as nn
from utils.dataset import Dataset
from utils.common import onehot_coding
import numpy as np
import copy

def discretize_tree(original_tree):
    tree=copy.deepcopy(original_tree)
    for name, parameter in tree.named_parameters():
        # print(name)
        if name == 'beta':
            setattr(tree, name, nn.Parameter(100*torch.ones(parameter.shape)))

        elif name == 'linear.weight':
            parameters=[]
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
    tree.save_model(tree.args['model_path']+'_discretized')
    return tree

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
        prediction, _, _, _ = tree.forward(data)
        prediction_, _, _, _ = discretized_tree.forward(data)
        with torch.no_grad():
            pred = prediction.data.max(1)[1]
            correct += pred.eq(target.view(-1).data).sum()
            pred_ = prediction_.data.max(1)[1]
            correct_ += pred_.eq(target.view(-1).data).sum()
    accuracy = 100. * float(correct) / len(test_loader.dataset)
    accuracy_ = 100. * float(correct_) / len(test_loader.dataset)
    print('Original Tree Accuracy: {:.4f} | Discretized Tree Accuracy: {:.4f}'.format(accuracy, accuracy_))

if __name__ == '__main__':    
    from sdt_train import learner_args
    from SDT import SDT
    Increasing_Beta = False  # choose whether it's using increasing beta during training
    learner_args['cuda'] = False  # cpu
    learner_args['lamda'] = -0.01
    # id=1
    for id in range(1, 4):
        if Increasing_Beta:
            learner_args['model_path']='./model/trees/sdt_'+str(learner_args['lamda'])+'_id'+str(id)+'beta'
            # since beta is not stored with the tree, we need manually calculate it here
            beta=1.
            for epoch in range(1, learner_args['epochs']+1):
                if epoch % 5 ==0:
                    beta = beta*2.
            learner_args['beta']=beta
        else:
            learner_args['beta'] = True
            learner_args['model_path']='./model/trees/sdt_'+str(learner_args['lamda'])+'_id'+str(id)


        tree = SDT(learner_args)
        tree.load_model(learner_args['model_path'])
        tree.eval()

        discretized_tree = discretize_tree(tree)
        discretization_evaluation(tree, discretized_tree)