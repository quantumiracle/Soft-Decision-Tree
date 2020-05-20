# -*- coding: utf-8 -*-
""" Cascade a feature learning tree and a soft decision tree (sparse in features) """
import torch
import torch.nn as nn
from utils.dataset import Dataset
import numpy as np
from SDT import SDT

class Cascade_SDT(nn.Module):
    def __init__(self):
        super(Cascade_SDT, self).__init__()

        features = self.feature_learning_module()

        self.sparse_SDT(features)

    def feature_learning_module(self):

        pass

    def sparse_SDT(self, features):
        pass

    def forward(self,):
        pass

    
    def load_model(self, path):
        pass

    def save_model(self, path):
        pass

    

if __name__ == '__main__':    
