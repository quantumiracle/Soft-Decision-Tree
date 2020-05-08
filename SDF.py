# -*- coding: utf-8 -*-
'''' 
Soft Decision Forests: 
ensemble of Soft Decision Trees
'''
import torch
import torch.nn as nn
from SDT import SDT

class SDF(object):
    """ Soft Desicion Tree """
    def __init__(self, args, device):
        super(SDF, self).__init__()
        self.tree_list=[]
        self.args = args
        for _ in range(self.args['num_trees']):
            self.tree_list.append(SDT(self.args).to(device))

    def forward(self, data, LogProb=True):
        forest_prediction = []
        forest_output = []
        forest_penalty = []
        for tree in self.tree_list:
            prediction, output, penalty = tree.forward(data, LogProb=False)
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
        for tree in self.tree_list:
            tree.optimizer.zero_grad()

    def optimizers_step(self):
        for tree in self.tree_list:
            tree.optimizer.step()

    def save_model(self):
        for id in range(len(self.tree_list)):
            self.tree_list[id].save_model(model_path=self.args['model_path'], id='_'+str(id))

    def load_model(self):
        for id in range(len(self.tree_list)):
            self.tree_list[id].load_model(model_path=self.args['model_path'], id='_'+str(id))


