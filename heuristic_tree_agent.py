# -*- coding: utf-8 -*-
'''' 
Soft Decision Forests: 
ensemble of Soft Decision Trees
'''
import torch
import torch.nn as nn
import numpy as np

class Node():
    def __init__(self, id, weights, left_child, right_child):
        super(Node, self).__init__()
        self.id = id
        self.weights = weights
        self.left_child = left_child
        self.right_child = right_child

    def decide(self, aug_x):
        prod = self.weights*aug_x  # weights include bias
        if prod>0:
            return self.left_child
        else:
            return self.right_child


class Leaf():
    super(Leaf, self).__init__()
    self.id = id
    self.weights = weights



{6:1, 7:1},
{0:0.25, 2:0.5, 4:-0.5, 5:-1},
{3:-0.5},
{3:-0.5, 8:-0.05},
{0:0.25, 2:0.5, 4:-0.5, 5:-1},
{}



def dict_to_vector(dict, dim=8):
    v = np.zeros(dim+1)  # the last dim is bias
    for key, value in dict.items():
        v[key]=value
    return v

    Node()


class HeuristicTree(object):
    """ Soft Desicion Tree """
    def __init__(self, args, device):
        super(SDF, self).__init__()
        self.tree_list=[]
        self.args = args
        for _ in range(self.args['num_trees']):
            self.tree_list.append(SDT(self.args).to(device))

    def forward(self, data, LogProb=True, Train=False):
        if Train:
            # randomly select a tree if training
            self.tree_id = np.random.randint(0, self.args['num_trees'])
            forest_prediction, forest_output, forest_penalty, _ = self.tree_list[self.tree_id].forward(data, LogProb=False)
        else: 
            # take average over all tree if inference    
            forest_prediction = []
            forest_output = []
            forest_penalty = []

            for tree in self.tree_list:
                prediction, output, penalty, _ = tree.forward(data, LogProb=False)
                forest_prediction.append(prediction)
                forest_output.append(output)
                forest_penalty.append(penalty)
            
            forest_prediction = torch.mean(torch.stack(forest_prediction), dim=0)
            forest_output = torch.mean(torch.stack(forest_output), dim=0)
            forest_penalty = torch.mean(torch.stack(forest_penalty), dim=0)

        if LogProb:
            forest_prediction = torch.log(forest_prediction)
            forest_output =  torch.log(forest_output)

        return forest_prediction, forest_output, forest_penalty

    def train(self):
        for tree in self.tree_list:
            tree.train()        

    def eval(self):
        for tree in self.tree_list:
            tree.eval()

    def optimizers_clear(self):
        # for tree in self.tree_list:
        #     tree.optimizer.zero_grad()
        self.tree_list[self.tree_id].optimizer.zero_grad()

    def optimizers_step(self):
        # for tree in self.tree_list:
        #     tree.optimizer.step()
        self.tree_list[self.tree_id].optimizer.step()

    def save_model(self):
        for id in range(len(self.tree_list)):
            self.tree_list[id].save_model(model_path=self.args['model_path'], id='_'+str(id))

    def load_model(self):
        for id in range(len(self.tree_list)):
            self.tree_list[id].load_model(model_path=self.args['model_path'], id='_'+str(id))


