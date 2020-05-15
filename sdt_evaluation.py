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

def evaluate(model, tree, episodes=1, frameskip=1, seed=None, DrawTree=True, DrawImportance=True):
    env = gym.make('LunarLander-v2')
    if seed:
        env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n  # discrete
    img_path='img/eval_abs2_{}'.format(tree.args['depth'])
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    tree_weights = tree.get_tree_weights()
    average_weight_list = []

    for n_epi in range(episodes):
        print('Episode: ', n_epi)
        average_weight_list_epi = []
        s = env.reset()
        done = False
        reward = 0.0
        step=0
        while not done:
            a = model(torch.Tensor([s]))
            if step%frameskip==0:
                if DrawTree:
                    draw_tree(tree, (tree.args['input_dim'],), input_img=s, savepath=img_path+'/{:04}.png'.format(step))
                if DrawImportance:
                    path_idx = get_path(tree, s)
                    weights_on_path = tree_weights[path_idx[:-1]]  # remove leaf node, i.e. the last index 
                    average_weight = np.mean(np.abs(normalize(weights_on_path)), axis=0)  # take absolute to prevent that positive and negative will counteract
                    average_weight_list_epi.append(average_weight)

            s_prime, r, done, info = env.step(a)
            # env.render()
            s = s_prime

            reward += r
            step+=1
            if done:
                break

        average_weight_list.append(average_weight_list_epi)
        print("# of episode :{}, reward : {:.1f}, episode length: {}".format(n_epi, reward, step))
    np.save('data/sdt_importance.npy', average_weight_list)

    env.close()


def plot_importance_single_episode(data_path='data/sdt_importance.npy', save_path='./img/sdt_importance.png', epi_id=0):
    data = np.load(data_path, allow_pickle=True)[epi_id]
    for i, weights_per_feature in enumerate(np.array(data).T):
        plt.plot(weights_per_feature, label='Dim: {}'.format(i))
    plt.legend(loc=4)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':
    from sdt_train import learner_args
    from SDT import SDT

    # for reproduciblility
    seed=3
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
    learner_args['cuda'] = False  # cpu

    tree = SDT(learner_args)
    tree.load_model(learner_args['model_path'])
    model = lambda x: tree.forward(x)[0].data.max(1)[1].squeeze().detach().numpy()
    evaluate(model, tree, episodes=1, frameskip=1, seed=seed, DrawTree=True, DrawImportance=True)
    plot_importance_single_episode(epi_id=0)