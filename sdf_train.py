# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from SDF import SDF
from utils.dataset import Dataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def onehot_coding(target, device, output_dim):
    target_onehot = torch.FloatTensor(target.size()[0], output_dim).to(device)
    target_onehot.data.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1.)
    return target_onehot
use_cuda = False
learner_args = {'num_trees': 3,
                'input_dim': 8,
                'output_dim': 4,
                'depth': 4,
                'lamda': 1e-3,
                'lr': 1e-3,
                'weight_decay': 0,
                'batch_size': 1280,
                'epochs': 40,
                'cuda': use_cuda,
                'log_interval': 100,
                'exp_scheduler_gamma': 1.,
                'beta' : True,  # temperature 
                'greatest_path_probability': True  # when forwarding the SDT, \
                # choose the leaf with greatest path probability or average over distributions of all leaves; \
                # the former one has better explainability while the latter one achieves higher accuracy
                }
learner_args['model_path'] = './model/forests/sdt_'+str(learner_args['depth'])

device = torch.device('cuda' if use_cuda else 'cpu')

def train_forest(forest):
    writer = SummaryWriter()
    criterion = nn.CrossEntropyLoss()  # torch CrossEntropyLoss = LogSoftmax + NLLLoss
    # criterion = nn.NLLLoss()  # since we already have log probability, simply using Negative Log-likelihood loss can provide cross-entropy loss
    
    
    # Load data
    data_dir = './data/discrete_'
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
    
    for epoch in range(1, learner_args['epochs']+1):
        epoch_training_loss_list = []

        # Training stage
        forest.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            target_onehot = onehot_coding(target, device, learner_args['output_dim'])
            prediction, output, penalty = forest.forward(data, Train=True)
            
            forest.optimizers_clear()
            loss = criterion(output, target.view(-1))
            loss += penalty
            loss.backward()
            forest.optimizers_step()
            
            # Print intermediate training status
            if batch_idx % learner_args['log_interval'] == 0:
                with torch.no_grad():
                    pred = prediction.data.max(1)[1]
                    correct = pred.eq(target.view(-1).data).sum()
                    loss = criterion(output, target.view(-1))
                    epoch_training_loss_list.append(loss.detach().cpu().data.numpy())
                    print('Epoch: {:02d} | Batch: {:03d} | CrossEntropy-loss: {:.5f} | Correct: {}/{}'.format(
                            epoch, batch_idx, loss.data, correct, output.size()[0]))

                    forest.save_model()
        writer.add_scalar('Training Loss', np.mean(epoch_training_loss_list), epoch)

        # Testing stage
        forest.eval()
        correct = 0.
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            batch_size = data.size()[0]
            prediction, _, _ = forest.forward(data)
            pred = prediction.data.max(1)[1]
            correct += pred.eq(target.view(-1).data).sum()
        accuracy = 100. * float(correct) / len(test_loader.dataset)
        if accuracy > best_testing_acc:
            best_testing_acc = accuracy
        testing_acc_list.append(accuracy)
        writer.add_scalar('Testing Accuracy', accuracy, epoch)
        print('\nEpoch: {:02d} | Testing Accuracy: {}/{} ({:.3f}%) | Historical Best: {:.3f}%\n'.format(epoch, correct, len(test_loader.dataset), accuracy, best_testing_acc))


def test_forest(forest, epochs=10):
    criterion = nn.CrossEntropyLoss()

    # Utility variables
    best_testing_acc = 0.
    testing_acc_list = []
    
    # Load data
    data_dir = './data/discrete_'
    data_path = data_dir+'state.npy'
    label_path = data_dir+'action.npy'
    test_loader = torch.utils.data.DataLoader(Dataset(data_path, label_path, partition='test'),
                                    batch_size=learner_args['batch_size'],
                                    shuffle=True)

    for epoch in range(epochs):
        # Testing stage
        forest.eval()
        correct = 0.
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            batch_size = data.size()[0]
            prediction, _, _ = forest.forward(data)
            pred = prediction.data.max(1)[1]
            correct += pred.eq(target.view(-1).data).sum()
        accuracy = 100. * float(correct) / len(test_loader.dataset)
        if accuracy > best_testing_acc:
            best_testing_acc = accuracy
        testing_acc_list.append(accuracy)
        print('\nEpoch: {:02d} | Testing Accuracy: {}/{} ({:.3f}%) | Historical Best: {:.3f}%\n'.format(epoch, correct, len(test_loader.dataset), accuracy, best_testing_acc))


if __name__ == '__main__':    
    forest = SDF(learner_args, device)
    train_forest(forest)