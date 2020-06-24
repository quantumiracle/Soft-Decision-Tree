# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
# from newSDT import SDT
from landmarkSDT import SDT
from torchvision import datasets, transforms
import numpy as np
from dataset import Dataset

def onehot_coding(target, device, output_dim):
    target_onehot = torch.FloatTensor(target.size()[0], output_dim).to(device)
    target_onehot.data.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1.)
    return target_onehot

use_cuda = False

learner_args = {'input_shape': (1,60,50),
                'output_dim': 3,
                'depth': 5,
                'lamda': 0.,
                'lr': 1e-3,
                'weight_decay': 0.,  # 5e-4
                'batch_size': 128,
                'epochs': 40,
                'cuda': use_cuda,
                'log_interval': 10,
                'model_path': './model/freeway',
                'beta' : True,  # temperature 
                'exp_scheduler_gamma': 1.,
                'l1_regularization': False,  # for feature sparsity on nodes
                'greatest_path_probability': True  # when forwarding the SDT, \
                # choose the leaf with greatest path probability or average over distributions of all leaves; \
                # the former one has better explainability while the latter one achieves higher accuracy
                }
device = torch.device('cuda' if use_cuda else 'cpu')

def train_tree(tree):
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()  # since we already have log probability, simply using Negative Log-likelihood loss can provide cross-entropy loss

    
    # Load data
    data_dir = './freeway_data/'
    data_path = data_dir+'state.npy'
    label_path = data_dir+'action.npy'
    train_loader = torch.utils.data.DataLoader(Dataset(data_path, label_path, partition='train'),
                                    batch_size=learner_args['batch_size'],
                                    shuffle=True)

    test_loader = torch.utils.data.DataLoader(Dataset(data_path, label_path, partition='test'),
                                    batch_size=learner_args['batch_size'],
                                    shuffle=True)
    # Utility variables
    best_testing_acc = 0.
    testing_acc_list = []
    training_loss_list = []
    
    for epoch in range(1, learner_args['epochs']+1):
        
        # Training stage
        tree.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device).float(), target.to(device)
            target_onehot = onehot_coding(target, device, learner_args['output_dim'])
            prediction, output, penalty, _ = tree.forward(data)
            # print(np.sum(output.detach().cpu().numpy(), axis=1))
            
            tree.optimizer.zero_grad()
            loss = criterion(output, target.view(-1))
            loss += penalty
            loss.backward()
            tree.optimizer.step()
            
            # Print intermediate training status
            if batch_idx % learner_args['log_interval'] == 0:
                with torch.no_grad():
                    pred = prediction.data.max(1)[1]
                    correct = pred.eq(target.view(-1).data).sum()
                    loss = criterion(output, target.view(-1))
                    training_loss_list.append(loss.cpu().data.numpy())
                    print('Epoch: {:02d} | Batch: {:03d} | CrossEntropy-loss: {:.5f} | Correct: {}/{}'.format(
                            epoch, batch_idx, loss.data, correct, output.size()[0]))

                    tree.save_model(model_path = learner_args['model_path'])
        
        # Testing stage
        tree.eval()
        correct = 0.
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device).float(), target.to(device)
            batch_size = data.size()[0]
            prediction, _, _, _ = tree.forward(data)
            pred = prediction.data.max(1)[1]
            correct += pred.eq(target.view(-1).data).sum()
        accuracy = 100. * float(correct) / len(test_loader.dataset)
        if accuracy > best_testing_acc:
            best_testing_acc = accuracy
        testing_acc_list.append(accuracy)
        print('\nEpoch: {:02d} | Testing Accuracy: {}/{} ({:.3f}%) | Historical Best: {:.3f}%\n'.format(epoch, correct, len(test_loader.dataset), accuracy, best_testing_acc))



def test_tree(tree, epochs=10):
    # Utility variables
    best_testing_acc = 0.
    testing_acc_list = []
    
    # Load data
    data_dir = './freeway_data/'
    data_path = data_dir+'state.npy'
    label_path = data_dir+'action.npy'
    test_loader = torch.utils.data.DataLoader(Dataset(data_path, label_path, partition='test'),
                                    batch_size=learner_args['batch_size'],
                                    shuffle=True)

    tree.load_model(learner_args['model_path'])
    for epoch in range(epochs):
        # Testing stage
        tree.eval()
        correct = 0.
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device).float(), target.to(device)
            batch_size = data.size()[0]
            prediction, _, _, _ = tree.forward(data)
            pred = prediction.data.max(1)[1]
            correct += pred.eq(target.view(-1).data).sum()
        accuracy = 100. * float(correct) / len(test_loader.dataset)
        if accuracy > best_testing_acc:
            best_testing_acc = accuracy
        testing_acc_list.append(accuracy)
        print('\nEpoch: {:02d} | Testing Accuracy: {}/{} ({:.3f}%) | Historical Best: {:.3f}%\n'.format(epoch, correct, len(test_loader.dataset), accuracy, best_testing_acc))


if __name__ == '__main__':
    tree = SDT(learner_args).to(device)
    train_tree(tree)
    # test_tree(tree)