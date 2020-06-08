# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import gym
from torch.distributions import Categorical
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tree_plot import draw_tree, get_path
from heuristic_evaluation import normalize
import os
import copy
from cascade_tree_evaluation import *

EnvName = 'CartPole-v1'  # LunarLander-v2

if __name__ == '__main__':
    from cdt_ppo_gae_discrete import PPO
    learner_args = {
    'num_intermediate_variables': 2,
    'feature_learning_depth': 1,
    'decision_depth': 2,
    'input_dim': 4,
    'output_dim': 2,
    'lr': 1e-3,
    'weight_decay': 0.,  # 5e-4
    'batch_size': 1280,
    'exp_scheduler_gamma': 1.,
    'cuda': False,
    'epochs': 40,
    'log_interval': 100,
    'greatest_path_probability': True,
    'beta_fl' : False,  # temperature for feature learning
    'beta_dc' : False,  # temperature for decision making
    }
    path='cdt_ppo_discrete_'+EnvName+'depth_'+str(learner_args['feature_learning_depth'])+str(learner_args['decision_depth'])+'_id'+str(4)
    learner_args['model_path'] = './model_cdt_ppo/'+path
    learner_args['device'] = torch.device('cuda' if learner_args['cuda'] else 'cpu')

    # for reproduciblility
    seed=3
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)

    env = gym.make(EnvName)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n 
    model = PPO(state_dim, action_dim, learner_args)
    model.load_model()
    tree = model.cdt
    
    # Discretized=True  # whether load the discretized tree
    # if Discretized:
    #     tree.load_model(learner_args['model_path']+'_discretized')
    # else:
    #     tree.load_model(learner_args['model_path'])

    num_params = 0
    for key, v in tree.state_dict().items():
        print(key, v.reshape(-1).shape[0])
        num_params+=v.reshape(-1).shape[0]
    print('Total number of parameters in model: ', num_params)

    model = lambda x: tree.forward(x)[0].data.max(1)[1].squeeze().detach().numpy()
    img_path = 'img/eval_tree_{}_{}'.format(tree.args['feature_learning_depth'], tree.args['decision_depth'])
    # if Discretized:
    #     img_path += '_discretized'
    evaluate(model, tree, episodes=10, frameskip=1, seed=seed, DrawTree='FL', DrawImportance=False, \
        img_path=img_path)

    # plot_importance_single_episode(epi_id=0)