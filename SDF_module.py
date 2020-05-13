# -*- coding: utf-8 -*-
'''' 
Soft Decision Forests: 
ensemble of Soft Decision Trees
'''
import torch
import torch.nn as nn
from SDT import SDT
import numpy as np

class SDF(nn.Module):
    """ Soft Desicion Tree """
    def __init__(self, args, device):
        super(SDF, self).__init__()
        self.tree_list=[]
        self.args = args

        fusion_params = torch.ones(self.args['num_trees']).to(device)
        self.fusion_params = nn.Parameter(fusion_params)
        self.softmax = nn.Softmax(dim=0)

        self.parameters_=[]
        for _ in range(self.args['num_trees']):
            tree = SDT(self.args).to(device)
            self.tree_list.append(tree)
            self.parameters_ += list(tree.parameters())

        self.forest_optimizer = torch.optim.Adam(self.parameters_, lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        self.fusion_optimizer  = torch.optim.Adam(self.parameters(), lr=self.args['lr'])


    def forward(self, data, LogProb=True):
        forest_prediction = []
        forest_output = []
        forest_penalty = []

        normalized_fusion_params = self.softmax(self.fusion_params)

        for i, tree in enumerate(self.tree_list):
            prediction, output, penalty, _ = tree.forward(data, LogProb=False)
            forest_prediction.append(normalized_fusion_params[i]*prediction)
            forest_output.append(normalized_fusion_params[i]*output)
            forest_penalty.append(normalized_fusion_params[i]*penalty)

        forest_prediction = torch.sum(torch.stack(forest_prediction), dim=0)
        forest_output = torch.sum(torch.stack(forest_output), dim=0)
        forest_penalty = torch.sum(torch.stack(forest_penalty), dim=0)

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
        self.forest_optimizer.zero_grad()
        self.fusion_optimizer.zero_grad()

    def optimizers_step(self):
        self.forest_optimizer.step()
        self.fusion_optimizer.step()

    def save_model(self):
        for id in range(len(self.tree_list)):
            self.tree_list[id].save_model(model_path=self.args['model_path'], id='_'+str(id))

    def load_model(self):
        for id in range(len(self.tree_list)):
            self.tree_list[id].load_model(model_path=self.args['model_path'], id='_'+str(id))


if __name__ == '__main__':    
    from sdf_train import learner_args, device
    forest = SDF(learner_args, device)
    from utils.dataset import Dataset


    # Load data
    data_dir = './data/discrete_'
    data_path = data_dir+'state.npy'
    label_path = data_dir+'action.npy'
    train_loader = torch.utils.data.DataLoader(Dataset(data_path, label_path, partition='train'),
                                    batch_size=learner_args['batch_size'],
                                    shuffle=True)

    forest.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        prediction, output, penalty = forest.forward(data, Train=False)
        break